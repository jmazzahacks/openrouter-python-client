"""
Streaming request classes for OpenRouter Client.

This module provides classes for handling streaming requests to the OpenRouter API,
with support for resumption and state management.

Exported:
- OpenRouterStreamingState: State manager for streaming requests
- StreamingCompletionsRequest: Streaming request handler for completions endpoint
- StreamingChatCompletionsRequest: Streaming request handler for chat completions endpoint
"""

import re
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union, List, TypeVar, Iterator
from base64 import b64decode

from pydantic import Field, field_validator, field_serializer
from smartsurge.streaming import AbstractStreamingRequest, StreamingState as BaseStreamingState
from smartsurge.exceptions import ResumeError, StreamingError

from .http import HTTPManager
from .auth import AuthManager


logger = logging.getLogger(__name__)
T = TypeVar('T')


class OpenRouterStreamingState(BaseStreamingState):
    """
    State of a streaming request for resumption with OpenRouter-specific handling.
    
    This class extends the base StreamingState with custom handling for OpenRouter's
    Server-Sent Events format, including comments and delta handling.
    
    Attributes:
        endpoint: The endpoint being requested
        method: The HTTP method being used
        headers: HTTP headers for the request
        params: Optional query parameters
        data: Optional request body data
        accumulated_data: Data accumulated so far
        last_position: Last position in the stream
        total_size: Total size of the stream if known
        etag: ETag of the resource if available
        last_updated: Timestamp of when this state was last updated
        request_id: Unique identifier for the request (for tracking)
    """
    endpoint: str = Field(..., min_length=1)
    method: str
    headers: Dict[str, str]
    params: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    accumulated_data: bytes
    last_position: int = Field(..., ge=0)
    total_size: Optional[int] = None
    etag: Optional[str] = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    @field_validator("accumulated_data", mode="before")
    def decode_accumulated_data(cls, v: Union[str, bytes]):
        """
        Process accumulated data from OpenRouter API.
        
        This method extends the base implementation to handle OpenRouter-specific aspects:
        1. Skips lines that don't begin with 'data: ' (Server-Sent Events comments)
        2. Handles the 'message' key being replaced with 'delta' in streaming responses
        3. Performs base64 decoding when needed
        
        Args:
            v: The input that might be base64 encoded. Can be a string or bytes.
            
        Returns:
            The processed data as bytes
        """
        # Convert to string if bytes
        if isinstance(v, bytes):
            return v
        elif isinstance(v, str):
            # Check if the string matches base64 pattern
            # Base64 strings consist of letters, digits, '+', '/', and may end with '='
            if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', v):
                logger.warning(f"String does not match base64 pattern: {v}")
                try:
                    v = v.encode("utf-8")
                except UnicodeEncodeError:
                    logger.warning(f"Error encoding string as UTF-8: {v}")
                finally:
                    return v
            
            # The length of base64 encoded string should be a multiple of 4
            # (or it should be padded to be)
            if len(v) % 4 != 0:
                logger.warning(f"Base64 string length is not a multiple of 4: {v}")
                try:
                    v = v.encode("utf-8")
                except UnicodeEncodeError:
                    logger.error(f"Error encoding string as UTF-8: {v}")
                    v = b""
                finally:
                    return v
                
            try:
                # Try to decode
                decoded_data = b64decode(v)
                logger.debug(f"Decoded base64 string: {decoded_data}")
                
                # Try to convert to UTF-8 string if possible
                try:
                    decoded_str = decoded_data.decode("utf-8")
                    
                    # Process OpenRouter SSE format - filter out comment lines
                    # and process only data lines
                    processed_lines = []
                    for line in decoded_str.splitlines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            # Extract the JSON data
                            data_content = line[6:]  # Skip 'data: '
                            try:
                                # Parse JSON to handle delta/message key replacement
                                json_data = json.loads(data_content)
                                # For chat completions, if there's a 'delta' key in choices,
                                # rename it to 'message' for compatibility
                                if 'choices' in json_data:
                                    for choice in json_data['choices']:
                                        if 'delta' in choice:
                                            # Keep delta but ensure it's processed correctly
                                            pass
                                processed_lines.append(f"data: {json.dumps(json_data, separators=(',', ':'))}")
                            except json.JSONDecodeError:
                                # If not valid JSON, just include the line as is
                                processed_lines.append(line)
                    
                    # If we have processed lines, rejoin them and return
                    if processed_lines:
                        processed_data = "\n".join(processed_lines)
                        return processed_data.encode("utf-8")
                    
                    # Otherwise return the original decoded data
                    return decoded_data
                except UnicodeDecodeError:
                    # If it can't be decoded as UTF-8, return the bytes
                    logger.warning(f"Failed to decode as UTF-8, returning bytes")
                    return decoded_data
            except Exception as e:
                # If any other error occurs, return the original input
                logger.error(f"Unexpected error decoding data: {e}", exc_info=True)
                return b""
        else:
            return v
        
    @field_serializer("accumulated_data")
    def serialize_accumulated_data(self, v: bytes):
        """
        Serialize accumulated_data to a base64 string for JSON.
        
        Args:
            v: The bytes to serialize
            
        Returns:
            The base64 encoded string
        """
        import base64
        if isinstance(v, bytes):
            return base64.b64encode(v).decode('ascii')
        return v


class StreamingCompletionsRequest(AbstractStreamingRequest):
    """
    Streaming request handler for completions endpoint.
    
    This class handles streaming text completions from the OpenRouter API,
    with support for resumption, state management, and cancellation.
    """
    def __init__(self, 
                 http_manager: HTTPManager,
                 auth_manager: AuthManager,
                 endpoint: str, 
                 headers: Dict[str, str],
                 prompt: str,
                 params: Optional[Dict[str, Any]] = None,
                 chunk_size: int = 8192,
                 state_file: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 request_id: Optional[str] = None):
        """
        Initialize a streaming completions request.
        
        Args:
            http_manager: HTTPManager instance for making requests
            auth_manager: AuthManager instance for authentication
            endpoint: The completions endpoint URL
            headers: HTTP headers for the request
            prompt: The text prompt to complete
            params: Additional parameters for the completions request
            chunk_size: Size of chunks to process
            state_file: File to save state for resumption
            logger: Optional custom logger to use
            request_id: Optional request ID for tracking and correlation
        """
        # Initialize params first
        params = params if params is not None else {}
        params['stream'] = True
        
        # Build the data that will be used for the request
        data = {
            'prompt': prompt,
            **params
        }
        
        # Call parent init first
        super().__init__(endpoint, headers, params=params, data=data, chunk_size=chunk_size, state_file=state_file, logger=logger, request_id=request_id)
        
        # Then set instance attributes
        self.http_manager = http_manager
        self.auth_manager = auth_manager
        self.prompt = prompt
        self.params = params
        self.completions = []
        self._cancelled = False
        self._response = None
        
    def stream(self) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterator that yields completion results as they arrive from the API.
        
        This method initiates the streaming request to the completions endpoint
        and yields each result as it arrives, allowing for real-time processing
        of partial results.
        
        Yields:
            List[Dict[str, Any]]: JSON data from each chunk
            
        Raises:
            StreamingError: If the request fails
        """
        # Build the request data
        data = {
            'prompt': self.prompt,
            **self.params
        }
        buffer = b""
        self._response = None
        self._cancelled = False
        
        try:
            # Make the request with stream=True using HTTPManager's client
            self._response = self.http_manager.client.post(
                self.endpoint,
                headers=self.headers,
                json=data,
                stream=True
            )
            
            if self._response.status_code != 200:
                raise StreamingError(
                    f"Request failed with status {self._response.status_code}: {self._response.text}",
                    endpoint=self.endpoint,
                    position=self.position,
                    response=self._response
                )
            
            # Process the streaming response as Server-Sent Events (SSE)
            for chunk in self._response.iter_content(chunk_size=self.chunk_size):
                # Check if the request has been cancelled
                if self._cancelled:
                    self.logger.info(f"[{self.request_id}] Stream cancelled during iteration")
                    break
                    
                if chunk:
                    self.position += len(chunk)
                    self.accumulated_data.extend(chunk)
                    
                    # Add the chunk to our buffer and process complete lines
                    buffer += chunk
                    
                    # Process any complete SSE messages
                    while b'\n\n' in buffer or b'\r\n\r\n' in buffer:
                        # Find the end of the SSE message
                        if b'\r\n\r\n' in buffer:
                            message, buffer = buffer.split(b'\r\n\r\n', 1)
                        else:
                            message, buffer = buffer.split(b'\n\n', 1)
                        
                        # Process the SSE message and yield any valid completions
                        completions = self.process_chunk(message + b'\n\n')
                        for completion in completions:
                            yield completion
                    
                    self.save_state()
            
            # Process any remaining buffer content if not cancelled
            if buffer and not self._cancelled:
                completions = self.process_chunk(buffer)
                for completion in completions:
                    yield completion
            
            # Mark as completed only if not cancelled
            if not self._cancelled:
                self.completed = True
            
        except Exception as e:
            raise StreamingError(
                f"Streaming request failed: {e}",
                endpoint=self.endpoint,
                position=self.position,
                response=self._response
            )

    def start(self) -> None:
        """
        Start the streaming completions request.
        
        This method initiates the streaming request to the completions endpoint
        and processes all chunks of data. It accumulates all results in self.completions.
        For an iterator-based approach that yields results as they arrive, use the stream() method.
        
        Raises:
            StreamingError: If the request fails
        """
        # Consume the entire stream and store results
        try:
            for _ in self.stream():
                pass  # Results are accumulated in self.completions by the stream() method
        except Exception as e:
            # Re-raise but keep the original error type
            raise e
    
    def resume_stream(self) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterator that resumes a streaming completions request from saved state.
        
        This method resumes a streaming request that was previously paused, by submitting
        a new completions request with the original prompt concatenated with the accumulated
        response so far. It yields each new completion result as it arrives.
        
        Yields:
            List[Dict[str, Any]]: JSON data from each completion chunk
            
        Raises:
            ResumeError: If resuming the request fails
        """
        # Load the saved state
        state = self.load_state()
        if not state:
            raise ResumeError("No state file found or state loading failed", state_file=self.state_file)
        
        # Extract state information
        self.endpoint = state.endpoint
        # Use fresh authentication headers from auth_manager
        self.headers = self.auth_manager.get_auth_headers()
        data = state.data
        
        # Ensure we get data to rebuild our request
        if not data:
            raise ResumeError("State does not contain request data", state_file=self.state_file)
        
        # Extract accumulated response from the raw accumulated_data
        accumulated_text = ""
        if hasattr(state, 'accumulated_data') and state.accumulated_data:
            try:
                # Convert accumulated_data to string - it might be bytes or already decoded
                if isinstance(state.accumulated_data, bytes):
                    raw_data = state.accumulated_data.decode('utf-8')
                else:
                    raw_data = str(state.accumulated_data)
                
                # Process the SSE format data
                for line in raw_data.splitlines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            # Extract JSON from the data line
                            json_str = line[6:]  # Skip 'data: '
                            chunk_data = json.loads(json_str)
                            
                            # Extract content from the chunk
                            if 'choices' in chunk_data and chunk_data['choices']:
                                choice = chunk_data['choices'][0]
                                if 'text' in choice:
                                    accumulated_text += choice['text'] or ''
                                elif 'delta' in choice and 'content' in choice['delta']:
                                    accumulated_text += choice['delta']['content'] or ''
                                elif 'message' in choice and 'content' in choice['message']:
                                    accumulated_text += choice['message']['content'] or ''
                        except json.JSONDecodeError:
                            self.logger.warning(f"[{self.request_id}] Failed to parse JSON from line: {line}")
                            continue
            except Exception as e:
                self.logger.error(f"[{self.request_id}] Error extracting accumulated text: {e}")
                # Fall back to empty accumulated text
                accumulated_text = ""
        
        self.logger.debug(f"[{self.request_id}] Accumulated text: {accumulated_text}")
        
        # Extract the original prompt from the saved data
        original_prompt = data.get('prompt', '')
        
        # Create a new prompt by concatenating the original prompt with the accumulated text
        new_prompt = original_prompt + accumulated_text
        
        # Build the new request data with the concatenated prompt
        # Use the saved data but update the prompt
        new_data = data.copy()
        new_data['prompt'] = new_prompt
        
        buffer = b""
        self._response = None
        self._cancelled = False
        
        try:
            # Make a new request with stream=True using HTTPManager's client
            # Extract the endpoint path from the full URL
            endpoint_path = self.endpoint
            if endpoint_path.startswith('http'):
                # Parse out just the path portion
                from urllib.parse import urlparse
                parsed = urlparse(endpoint_path)
                endpoint_path = parsed.path
                # Remove the base path (e.g., "/api/v1/") if it's included
                if '/api/v1/' in endpoint_path:
                    endpoint_path = endpoint_path.split('/api/v1/')[-1]
            
            self._response = self.http_manager.client.post(
                self.endpoint,
                headers=self.headers,
                json=new_data,
                stream=True
            )
            
            if self._response.status_code != 200:
                raise ResumeError(
                    f"Resume request failed with status {self._response.status_code}: {self._response.text}",
                    state_file=self.state_file,
                )
            
            # Process the streaming response
            for chunk in self._response.iter_content(chunk_size=self.chunk_size):
                # Check if the request has been cancelled
                if self._cancelled:
                    self.logger.info(f"[{self.request_id}] Resume stream cancelled during iteration")
                    break
                    
                if chunk:
                    self.position += len(chunk)
                    self.accumulated_data.extend(chunk)
                    
                    # Add the chunk to our buffer and process complete lines
                    buffer += chunk
                    
                    # Process any complete SSE messages
                    while b'\n\n' in buffer or b'\r\n\r\n' in buffer:
                        # Find the end of the SSE message
                        if b'\r\n\r\n' in buffer:
                            message, buffer = buffer.split(b'\r\n\r\n', 1)
                        else:
                            message, buffer = buffer.split(b'\n\n', 1)
                        
                        # Process the SSE message and yield any valid completions
                        completions = self.process_chunk(message + b'\n\n')
                        for completion in completions:
                            yield completion
                    
                    self.save_state()
            
            # Process any remaining buffer content if not cancelled
            if buffer and not self._cancelled:
                completions = self.process_chunk(buffer)
                for completion in completions:
                    yield completion
            
            # Mark as completed only if not cancelled
            if not self._cancelled:
                self.completed = True
            
        except Exception as e:
            raise ResumeError(
                f"Resuming streaming request failed: {e}",
                state_file=self.state_file,
                original_error=e
            )
    
    def resume(self) -> None:
        """
        Resume the streaming completions request from saved state.
        
        This method resumes a streaming request that was previously
        paused, by submitting a new completions request with the original prompt
        concatenated with the accumulated response so far.
        
        Raises:
            ResumeError: If resuming the request fails
        """
        # Consume the entire resumed stream
        try:
            for _ in self.resume_stream():
                pass  # Results are accumulated in self.completions
        except Exception as e:
            # Re-raise but keep the original error type
            raise e
    
    def process_chunk(self, chunk: bytes) -> List[Dict[str, Any]]:
        """
        Process a chunk of data from the completions stream in SSE format.
        
        OpenRouter's streaming API uses Server-Sent Events (SSE) format where each event
        starts with 'data: ' and multiple events can be included in a single chunk.
        
        Args:
            chunk: A chunk of data to process in SSE format
            
        Returns:
            List of parsed JSON data from the chunk, or empty list if no valid data
        """
        # Decode chunk to string
        results = []
        try:
            chunk_str = chunk.decode('utf-8')
            
            # Process SSE format - each line starts with 'data: '
            for line in chunk_str.splitlines():
                if not line or line.isspace():
                    continue
                    
                if line.startswith("data: "):
                    if line == "data: [DONE]":
                        self.logger.debug(f"[{self.request_id}] Received completion message: {line}")
                        continue
                    
                    # Extract the JSON data
                    data_content = line[6:]  # Skip 'data: '
                    try:
                        json_data = json.loads(data_content)
                        self.completions.append(json_data)
                        results.append(json_data)
                        
                        # Log information about the chunk for debugging
                        if self.logger.isEnabledFor(logging.DEBUG):
                            if 'choices' in json_data:
                                for choice in json_data['choices']:
                                    if 'text' in choice:
                                        self.logger.debug(f"[{self.request_id}] Received text: {choice['text']}")
                                    if "message" in choice:
                                        if "content" in choice["message"]:
                                            self.logger.debug(f"[{self.request_id}] Received content: {choice['message']['content']}")
                                        if "reasoning" in choice["message"]:
                                            self.logger.debug(f"[{self.request_id}] Received reasoning: {choice['message']['reasoning']}")
                                    if "delta" in choice:
                                        if "content" in choice["delta"]:
                                            self.logger.debug(f"[{self.request_id}] Received content: {choice['delta']['content']}")
                                        if "reasoning" in choice["delta"]:
                                            self.logger.debug(f"[{self.request_id}] Received reasoning: {choice['delta']['reasoning']}")
                    except json.JSONDecodeError:
                        self.logger.warning(f"[{self.request_id}] Invalid JSON in chunk: {data_content}")
                elif line.startswith(":") or line.startswith("id:") or line.startswith("event:"):
                    # These are valid SSE fields we can ignore
                    self.logger.debug(f"[{self.request_id}] Received SSE metadata: {line}")
                else:
                    # Unexpected line format
                    self.logger.warning(f"[{self.request_id}] Unexpected line format in SSE: {line}")
        except UnicodeDecodeError:
            self.logger.warning(f"[{self.request_id}] Failed to decode chunk as UTF-8")
            
        return results
    
    def cancel(self) -> None:
        """
        Cancel an ongoing streaming request.
        
        This method will terminate the streaming request immediately without
        waiting for more chunks to arrive from the server. After cancellation,
        the connection to the server is closed and no more data will be processed.
        
        Note that partial results received before cancellation will still be available
        through get_result() method.
        """
        self.logger.info(f"[{self.request_id}] Cancelling streaming request")
        self._cancelled = True
        
        # Close the underlying connection if it exists
        if self._response and hasattr(self._response, 'close'):
            try:
                self._response.close()
                self.logger.debug(f"[{self.request_id}] Response connection closed")
            except Exception as e:
                self.logger.warning(f"[{self.request_id}] Error closing response: {str(e)}")
    
    def get_result(self) -> List[Dict[str, Any]]:
        """
        Get the final result after all chunks have been processed.
        
        Returns:
            The list of completion response chunks
            
        Raises:
            StreamingError: If the streaming request is not complete
        """
        if not self.completed and not self._cancelled:
            self.logger.warning(f"[{self.request_id}] Getting incomplete result")
        
        return self.completions


class StreamingChatCompletionsRequest(AbstractStreamingRequest):
    """
    Streaming request handler for chat completions endpoint.
    
    This class handles streaming chat completions from the OpenRouter API,
    with support for resumption, state management, and cancellation.
    """
    def __init__(self, 
                 http_manager: HTTPManager,
                 auth_manager: AuthManager,
                 endpoint: str, 
                 headers: Dict[str, str],
                 messages: List[Dict[str, Any]],
                 data: Optional[Dict[str, Any]] = None,
                 params: Optional[Dict[str, Any]] = None,
                 chunk_size: int = 8192,
                 state_file: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 request_id: Optional[str] = None):
        """
        Initialize a streaming chat completions request.
        
        Args:
            http_manager: HTTPManager instance for making requests
            auth_manager: AuthManager instance for authentication
            endpoint: The chat completions endpoint URL
            headers: HTTP headers for the request
            messages: The conversation messages
            params: Additional parameters for the chat completions request
            data: Optional data for the chat completions request
            chunk_size: Size of chunks to process
            state_file: File to save state for resumption
            logger: Optional custom logger to use
            request_id: Optional request ID for tracking and correlation
        """
        super().__init__(endpoint, headers, params=params, data=data, chunk_size=chunk_size, state_file=state_file, logger=logger, request_id=request_id)
        self.http_manager = http_manager
        self.auth_manager = auth_manager
        self.messages = messages
        # MUST explicitly set params and data to empty dicts if None
        self.params = params if params is not None else {}
        self.data = data if data is not None else {}
        self.completions = []
        self._cancelled = False
        self._response = None
        
        # Ensure stream is set to True
        self.params['stream'] = True
        
    def stream(self) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterator that yields chat completion results as they arrive from the API.
        
        This method initiates the streaming request to the chat completions endpoint
        and yields each result as it arrives, allowing for real-time processing
        of partial results.
        
        Yields:
            List[Dict[str, Any]]: JSON data from each chunk
            
        Raises:
            StreamingError: If the request fails
        """
        # Build the request data
        data = {
            'messages': self.messages,
            **self.params
        }
        buffer = b""
        self._response = None
        self._cancelled = False
        
        try:
            # Make the request with stream=True using HTTPManager's client
            self._response = self.http_manager.client.post(
                self.endpoint,
                headers=self.headers,
                json=data,
                stream=True
            )
            
            if self._response.status_code != 200:
                raise StreamingError(
                    f"Request failed with status {self._response.status_code}: {self._response.text}",
                    endpoint=self.endpoint,
                    position=self.position,
                    response=self._response
                )
            
            # Process the streaming response as Server-Sent Events (SSE)
            for chunk in self._response.iter_content(chunk_size=self.chunk_size):
                # Check if the request has been cancelled
                if self._cancelled:
                    self.logger.info(f"[{self.request_id}] Stream cancelled during iteration")
                    break
                    
                if chunk:
                    self.position += len(chunk)
                    self.accumulated_data.extend(chunk)
                    
                    # Add the chunk to our buffer and process complete lines
                    buffer += chunk
                    
                    # Process any complete SSE messages
                    while b'\n\n' in buffer or b'\r\n\r\n' in buffer:
                        # Find the end of the SSE message
                        if b'\r\n\r\n' in buffer:
                            message, buffer = buffer.split(b'\r\n\r\n', 1)
                        else:
                            message, buffer = buffer.split(b'\n\n', 1)
                        
                        # Process the SSE message and yield any valid completions
                        completions = self.process_chunk(message + b'\n\n')
                        self.completions.append(completions)
                        for completion in completions:
                            yield completion
                    
                    self.save_state()
            
            # Process any remaining buffer content if not cancelled
            if buffer and not self._cancelled:
                completions = self.process_chunk(buffer)
                self.completions.append(completions)
                for completion in completions:
                    yield completion
            
            # Mark as completed only if not cancelled
            if not self._cancelled:
                self.completed = True
            
        except Exception as e:
            raise StreamingError(
                f"Streaming request failed: {e}",
                endpoint=self.endpoint,
                position=self.position,
                response=self._response
            )
    
    def start(self) -> None:
        """
        Start the streaming chat completions request.
        
        This method initiates the streaming request to the chat completions endpoint
        and processes all chunks of data. It accumulates all results in self.completions.
        For an iterator-based approach that yields results as they arrive, use the stream() method.
        
        Raises:
            StreamingError: If the request fails
        """
        # Consume the entire stream and store results
        try:
            for _ in self.stream():
                pass  # Results are accumulated in self.completions by the stream() method
        except Exception as e:
            # Re-raise but keep the original error type
            raise e
    
    def resume_stream(self) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterator that resumes a streaming chat completions request from saved state.
        
        This method resumes a streaming request that was previously paused, by submitting
        a new chat completions request with the original messages concatenated with the accumulated
        response so far. It yields each new chat completion result as it arrives.
        
        Yields:
            List[Dict[str, Any]]: JSON data from each chat completion chunk
            
        Raises:
            ResumeError: If resuming the request fails
        """
        # Load the saved state
        state = self.load_state()
        if not state:
            raise ResumeError("No state file found or state loading failed", state_file=self.state_file)
        
        # Extract state information
        self.endpoint = state.endpoint
        self.params = state.params
        
        # Use fresh authentication headers from auth_manager
        self.headers = self.auth_manager.get_auth_headers()
        
        # Extract accumulated response from the raw accumulated_data
        accumulated_text = ""
        if hasattr(state, 'accumulated_data') and state.accumulated_data:
            try:
                # Convert accumulated_data to string - it might be bytes or already decoded
                if isinstance(state.accumulated_data, bytes):
                    raw_data = state.accumulated_data.decode('utf-8')
                else:
                    raw_data = str(state.accumulated_data)
                
                # Process the SSE format data
                for line in raw_data.split("\n\n"):
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            # Extract JSON from the data line
                            json_str = line[6:]  # Skip 'data: '
                            try:
                                chunk_data = json.loads(json_str)
                            except json.JSONDecodeError:
                                self.logger.warning(f"Incomplete or invalid JSON: {json_str}")
                                continue
                            
                            # Extract content from the chunk
                            if 'choices' in chunk_data and chunk_data['choices']:
                                choice = chunk_data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    accumulated_text += choice['delta']['content'] or ''
                                elif 'message' in choice and 'content' in choice['message']:
                                    accumulated_text += choice['message']['content'] or ''
                        except json.JSONDecodeError:
                            self.logger.warning(f"[{self.request_id}] Failed to parse JSON from line: {line}")
                            continue
            except Exception as e:
                self.logger.error(f"[{self.request_id}] Error extracting accumulated text: {e}")
                # Fall back to empty accumulated text
                accumulated_text = ""
        
        self.logger.debug(f"[{self.request_id}] Accumulated text: {accumulated_text}")
        
        # Create a new message array by adding the AI's response so far
        new_messages = list(self.messages)  # Create a copy of the original messages
        if accumulated_text.strip():
            new_messages.append({"role": "assistant", "content": accumulated_text})
        
        # Build the new request data with the updated messages
        new_data = {
            'messages': new_messages,
            **self.params
        }
        
        buffer = b""
        self._response = None
        self._cancelled = False
        
        try:
            # Make a new request with stream=True using HTTPManager's client
            self._response = self.http_manager.client.post(
                endpoint=self.endpoint,
                headers=self.headers,
                json=new_data,
                stream=True
            )
            
            if self._response.status_code != 200:
                raise ResumeError(
                    f"Resume request failed with status {self._response.status_code}: {self._response.text}",
                    state_file=self.state_file,
                )
            
            # Process the streaming response
            for chunk in self._response.iter_content(chunk_size=self.chunk_size):
                # Check if the request has been cancelled
                if self._cancelled:
                    self.logger.info(f"[{self.request_id}] Resume stream cancelled during iteration")
                    break
                    
                if chunk:
                    self.position += len(chunk)
                    self.accumulated_data.extend(chunk)
                    
                    # Add the chunk to our buffer and process complete lines
                    buffer += chunk
                    
                    # Process any complete SSE messages
                    while b'\n\n' in buffer or b'\r\n\r\n' in buffer:
                        # Find the end of the SSE message
                        if b'\r\n\r\n' in buffer:
                            message, buffer = buffer.split(b'\r\n\r\n', 1)
                        else:
                            message, buffer = buffer.split(b'\n\n', 1)
                        
                        # Process the SSE message and yield any valid completions
                        completions = self.process_chunk(message + b'\n\n')
                        self.completions.append(completions)
                        for completion in completions:
                            yield completion
                    
                    self.save_state()
            
            # Process any remaining buffer content if not cancelled
            if buffer and not self._cancelled:
                completions = self.process_chunk(buffer)
                self.completions.append(completions)
                for completion in completions:
                    yield completion
            
            # Mark as completed only if not cancelled
            if not self._cancelled:
                self.completed = True
            
        except Exception as e:
            raise ResumeError(
                f"Resuming streaming request failed: {e}",
                state_file=self.state_file,
                original_error=e
            )
    
    def resume(self) -> None:
        """
        Resume the streaming chat completions request from saved state.
        
        This method resumes a streaming request that was previously
        paused, by submitting a new chat completions request with the original messages
        concatenated with the accumulated response so far.
        
        Raises:
            ResumeError: If resuming the request fails
        """
        # Consume the entire resumed stream
        try:
            for _ in self.resume_stream():
                pass  # Results are accumulated in self.completions
        except Exception as e:
            # Re-raise but keep the original error type
            raise e
    
    def process_chunk(self, chunk: bytes) -> List[Dict[str, Any]]:
        """
        Process a chunk of data from the chat completions stream in SSE format.
        
        OpenRouter's streaming API uses Server-Sent Events (SSE) format where each event
        starts with 'data: ' and multiple events can be included in a single chunk.
        
        Args:
            chunk: A chunk of data to process in SSE format
            
        Returns:
            List of parsed JSON data from the chunk, or empty list if no valid data
        """
        # Decode chunk to string
        results = []
        try:
            chunk_str = chunk.decode('utf-8')
            
            # Process SSE format - each line starts with 'data: '
            for line in chunk_str.splitlines():
                if not line or line.isspace():
                    continue
                    
                if line.startswith("data: "):
                    if line == "data: [DONE]":
                        self.logger.debug(f"[{self.request_id}] Received completion message: {line}")
                        continue
                    
                    # Extract the JSON data
                    data_content = line[6:]  # Skip 'data: '
                    try:
                        json_data = json.loads(data_content)
                        self.completions.append(json_data)
                        results.append(json_data)
                        
                        # Log information about the chunk for debugging
                        if self.logger.isEnabledFor(logging.DEBUG):
                            if 'choices' in json_data:
                                for choice in json_data['choices']:
                                    if "delta" in choice and "content" in choice["delta"]:
                                        self.logger.debug(f"[{self.request_id}] Received content: {choice['delta']['content']}")
                                    elif "message" in choice and "content" in choice["message"]:
                                        self.logger.debug(f"[{self.request_id}] Received content: {choice['message']['content']}")
                    except json.JSONDecodeError:
                        self.logger.warning(f"[{self.request_id}] Invalid JSON in chunk: {data_content}")
                elif line.startswith(":") or line.startswith("id:") or line.startswith("event:"):
                    # These are valid SSE fields we can ignore
                    self.logger.debug(f"[{self.request_id}] Received SSE metadata: {line}")
                else:
                    # Unexpected line format
                    self.logger.warning(f"[{self.request_id}] Unexpected line format in SSE: {line}")
        except UnicodeDecodeError:
            self.logger.warning(f"[{self.request_id}] Failed to decode chunk as UTF-8")
            
        return results
    
    def cancel(self) -> None:
        """
        Cancel an ongoing streaming request.
        
        This method will terminate the streaming request immediately without
        waiting for more chunks to arrive from the server. After cancellation,
        the connection to the server is closed and no more data will be processed.
        
        Note that partial results received before cancellation will still be available
        through get_result() method.
        """
        self.logger.info(f"[{self.request_id}] Cancelling streaming request")
        self._cancelled = True
        
        # Close the underlying connection if it exists
        if self._response and hasattr(self._response, 'close'):
            try:
                self._response.close()
                self.logger.debug(f"[{self.request_id}] Response connection closed")
            except Exception as e:
                self.logger.warning(f"[{self.request_id}] Error closing response: {str(e)}")
    
    def get_result(self) -> List[Dict[str, Any]]:
        """
        Get the final result after all chunks have been processed.
        
        Returns:
            The list of chat completion response chunks
            
        Raises:
            StreamingError: If the streaming request is not complete
        """
        if not self.completed and not self._cancelled:
            self.logger.warning(f"[{self.request_id}] Getting incomplete result")
        
        return self.completions