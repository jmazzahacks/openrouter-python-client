"""
Chat completions endpoint implementation.

This module provides the endpoint handler for chat completions API,
supporting both synchronous and streaming requests.

Exported:
- ChatEndpoint: Handler for chat completions endpoint
"""

import logging
from typing import Dict, List, Optional, Union, Any, Iterator

from ..auth import AuthManager
from ..http import HTTPManager
from .base import BaseEndpoint
from ..models import Message, ToolChoice, FunctionParameters, FunctionToolChoice
from ..models.chat import ReasoningConfig
from ..streaming import StreamingChatCompletionsRequest
from ..exceptions import StreamingError, ResumeError


class ChatEndpoint(BaseEndpoint):
    """
    Handler for the chat completions API endpoint.
    
    Provides methods for generating completions from chat conversations.
    Supports prompt caching for compatible providers (OpenAI, Anthropic Claude, DeepSeek).
    
    For caching with Anthropic Claude, use cache_control in TextContent parts to mark
    content that should be cached:
    
    ```python
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Regular content"},
            {"type": "text", "text": "Content to cache", "cache_control": {"type": "ephemeral"}}
        ]
    }
    ```
    
    OpenAI caching is automatic for prompts over 1024 tokens.
    """
    
    def __init__(self, auth_manager: AuthManager, http_manager: HTTPManager):
        """
        Initialize the chat endpoint handler.
        
        Args:
            auth_manager (AuthManager): Authentication manager.
            http_manager (HTTPManager): HTTP communication manager.
        """
        super().__init__(auth_manager, http_manager, 'chat/completions')
        self.logger.info(f"Initialized chat completions endpoint handler")
    
    def create(self, messages: List[Union[Dict[str, Any], Message]], 
               model: Optional[str] = None, temperature: Optional[float] = None, 
               top_p: Optional[float] = None, max_tokens: Optional[int] = None,
               stop: Optional[Union[str, List[str]]] = None, n: Optional[int] = None,
               stream: Optional[bool] = None, presence_penalty: Optional[float] = None,
               frequency_penalty: Optional[float] = None, user: Optional[str] = None,
               functions: Optional[List[FunctionParameters]] = None,
               function_call: Optional[Union[str, FunctionToolChoice]] = None,
               tools: Optional[List[Dict[str, Any]]] = None,
               tool_choice: Optional[ToolChoice] = None,
               response_format: Optional[Dict[str, Any]] = None,
               reasoning: Optional[Union[Dict[str, Any], ReasoningConfig]] = None,
               include_reasoning: Optional[bool] = None,
               state_file: Optional[str] = None, chunk_size: int = 8192,
               include: Optional[Dict[str, bool]] = None,
               **kwargs) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Create a chat completion for a conversation.
        
        Args:
            messages (List[Union[Dict[str, Any], Message]]): The conversation messages.
            model (Optional[str]): Model identifier to use.
            temperature (Optional[float]): Sampling temperature (0.0 to 2.0).
            top_p (Optional[float]): Nucleus sampling parameter (0.0 to 1.0).
            max_tokens (Optional[int]): Maximum tokens to generate.
            stop (Optional[Union[str, List[str]]]): Stop sequences to end generation.
            n (Optional[int]): Number of completions to generate.
            stream (Optional[bool]): Whether to stream responses.
            presence_penalty (Optional[float]): Penalty for token presence (-2.0 to 2.0).
            frequency_penalty (Optional[float]): Penalty for token frequency (-2.0 to 2.0).
            user (Optional[str]): User identifier for tracking.
            functions (Optional[List[FunctionParameters]]): Function definitions with JSON Schema parameters.
            function_call (Optional[Union[str, FunctionToolChoice]]): Function calling control ("auto", "none", or a specific function).
            tools (Optional[List[Dict[str, Any]]]): Tool definitions.
            tool_choice (Optional[ToolChoice]): Tool choice control ("auto", "none", or a specific function).
            response_format (Optional[Dict[str, Any]]): Format specification for response.
            reasoning (Optional[Union[Dict[str, Any], ReasoningConfig]]): Control reasoning tokens settings.
                Can include 'effort' ("high", "medium", or "low") or 'max_tokens' (int) and 'exclude' (bool).
                Accepts either a dict or a ReasoningConfig object.
            include_reasoning (Optional[bool]): Legacy parameter to include reasoning tokens in the response.
                When True, equivalent to reasoning={}, when False, equivalent to reasoning={'exclude': True}.
            state_file (Optional[str]): File path to save streaming state for resumption.
            chunk_size (int): Size of chunks for streaming responses.
            include (Optional[Dict[str, bool]]): Fields to include in the response. 
                Set {"usage": true} to get token usage statistics including cache metrics.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            Union[Dict[str, Any], Iterator[Dict[str, Any]]]: Either a JSON response (non-streaming) 
            or an iterator of response chunks (streaming).
            
        Raises:
            APIError: If the API request fails.
            StreamingError: If the streaming request fails.
        """
        # Convert any Message objects in messages to dictionaries
        processed_messages = []
        for msg in messages:
            if isinstance(msg, Message):
                processed_messages.append(msg.model_dump())
            else:
                processed_messages.append(msg)
        
        # Build data dictionary with non-None values that will be used for both streaming and non-streaming
        data = {"messages": processed_messages}
        
        # Add optional parameters if provided
        if model is not None:
            data["model"] = model
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if stop is not None:
            data["stop"] = stop
        if n is not None:
            data["n"] = n
        if presence_penalty is not None:
            data["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            data["frequency_penalty"] = frequency_penalty
        if user is not None:
            data["user"] = user
        if functions is not None:
            data["functions"] = functions
        if function_call is not None:
            data["function_call"] = function_call
        if tools is not None:
            data["tools"] = tools
        if tool_choice is not None:
            data["tool_choice"] = tool_choice
        if response_format is not None:
            data["response_format"] = response_format
        if reasoning is not None:
            # Convert ReasoningConfig object to dict if necessary
            if isinstance(reasoning, ReasoningConfig):
                data["reasoning"] = reasoning.model_dump(exclude_none=True)
            else:
                data["reasoning"] = reasoning
        if include_reasoning is not None:
            # Legacy parameter, only use if reasoning isn't set
            if reasoning is None:
                if include_reasoning:
                    data["reasoning"] = {}
                else:
                    data["reasoning"] = {"exclude": True}
            # When both are specified, reasoning takes precedence (already added above)
            
        # Handle stream parameter explicitly for streaming path
        if stream:
            # Ensure stream=True is set in data dictionary
            data["stream"] = True
        elif stream is not None:
            # For non-streaming, only add if explicitly provided
            data["stream"] = stream
            
        # Add include parameter for usage statistics (including cache information)
        if include is not None:
            data["include"] = include
            
        # Add any additional kwargs (for backward compatibility)
        for key, value in kwargs.items():
            # Don't override explicit parameters with kwargs
            # Skip stream-specific parameters for data dictionary
            if key not in data and key not in ['state_file', 'chunk_size']:
                data[key] = value
        
        # Get authentication headers
        headers = self._get_headers()
        
        # For streaming requests, use the streaming implementation
        if stream:
            # Build the full endpoint URL
            endpoint_url = f"{self.http_manager.base_url}/{self._get_endpoint_url()}"
            
            # Create streaming request handler
            streamer = StreamingChatCompletionsRequest(
                endpoint=endpoint_url,
                headers=headers,
                messages=processed_messages,
                params=data,  # Use the single data dictionary
                chunk_size=chunk_size,
                state_file=state_file,
                logger=self.logger,
                client=self.http_manager.client
            )
            
            # Start streaming
            try:
                streamer.start()
                return streamer.get_result()
            except Exception as e:
                self.logger.error(f"Streaming chat completions failed: {e}")
                raise StreamingError(
                    f"Streaming chat completions failed: {e}",
                    endpoint=endpoint_url,
                    position=streamer.position if hasattr(streamer, 'position') else 0
                )
        
        # For non-streaming requests, use the regular implementation
        else:
            # Make POST request to chat completions endpoint
            response = self.http_manager.post(
                endpoint=self._get_endpoint_url(),
                headers=headers,
                json=data  # Use the single data dictionary
            )
            
            # Return parsed JSON response
            return response.json()
    
    def resume_stream(self, state_file: str) -> Iterator[Dict[str, Any]]:
        """
        Resume a streaming chat completions request from saved state.
        
        Args:
            state_file (str): File containing the saved state.
            
        Returns:
            Iterator[Dict[str, Any]]: Resumed stream of chat completion chunks.
            
        Raises:
            ResumeError: If resuming the request fails.
        """
        # Create streaming request handler with just the state file
        streamer = StreamingChatCompletionsRequest(
            endpoint="",   # Will be loaded from state
            headers={},    # Will be loaded from state
            messages=[],   # Will be loaded from state
            state_file=state_file,
            logger=self.logger,
            client=self.http_manager.client
        )
        
        # Resume streaming
        try:
            streamer.resume()
            return streamer.get_result()
        except Exception as e:
            self.logger.error(f"Resuming chat completions stream failed: {e}")
            raise ResumeError(
                f"Resuming chat completions stream failed: {e}",
                state_file=state_file,
                original_error=e
            )
