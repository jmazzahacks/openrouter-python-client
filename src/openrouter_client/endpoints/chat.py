"""
Chat completions endpoint implementation.

This module provides the endpoint handler for chat completions API,
supporting both synchronous and streaming requests.

Exported:
- ChatEndpoint: Handler for chat completions endpoint
"""

from typing import Any, Dict, Iterator, List, Optional, Union

from ..auth import AuthManager
from ..exceptions import ResumeError, StreamingError
from ..http import HTTPManager
from ..models import FunctionParameters, Message, ToolChoice
from ..models.chat import (
    ChatCompletionFunction,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatCompletionStreamResponseDelta,
    ChatCompletionTool,
    ChatCompletionToolCall,
    FunctionToolChoice,
    ReasoningConfig,
    ToolCallFunction,
    Usage,
)
from ..models.core import FunctionDefinition, ResponseFormat
from ..streaming import StreamingChatCompletionsRequest
from ..types import FinishReason
from .base import BaseEndpoint


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
    
    def _parse_streaming_response(self, response_iterator: Iterator[Dict[str, Any]]) -> Iterator[ChatCompletionStreamResponse]:
        """
        Parse streaming response chunks into ChatCompletionStreamResponse models.
        
        Args:
            response_iterator: Iterator of raw response dictionaries.
            
        Yields:
            ChatCompletionStreamResponse: Parsed streaming response chunks.
        """
        for chunk in response_iterator:
            try:
                # Create a copy of the chunk to modify
                modified_chunk = chunk.copy()
                
                # Parse choices into ChatCompletionStreamResponseChoice objects for validation
                if 'choices' in chunk:
                    parsed_choices = []
                    for choice_data in chunk['choices']:
                        try:
                            modified_choice_data = choice_data.copy()
                            
                            # Parse finish reason
                            if 'finish_reason' in choice_data and choice_data['finish_reason'] is not None:
                                try:
                                    parsed_finish_reason = FinishReason(choice_data['finish_reason'])
                                    modified_choice_data['finish_reason'] = parsed_finish_reason.value
                                except Exception as e:
                                    self.logger.warning(f"Failed to parse streaming finish reason: {e}")
                                    # Keep the raw finish reason if parsing fails
                                    modified_choice_data['finish_reason'] = choice_data['finish_reason']
                            
                            # Parse finish reason
                            if 'delta' in choice_data and choice_data['delta'] is not None:
                                try:
                                    parsed_delta = ChatCompletionStreamResponseDelta.model_validate(choice_data['finish_reason'])
                                    modified_choice_data['delta'] = parsed_delta
                                except Exception as e:
                                    self.logger.warning(f"Failed to parse streaming delta: {e}")
                                    # Keep the raw delta if parsing fails
                                    modified_choice_data['delta'] = choice_data['delta']
                            
                            parsed_choice = ChatCompletionStreamResponseChoice.model_validate(modified_choice_data)
                            unparsed_choice = parsed_choice.model_dump()
                            
                            if 'finish_reason' in unparsed_choice and isinstance(unparsed_choice["finish_reason"], FinishReason):
                                parsed_choice["finish_reason"] = unparsed_choice["finish_reason"].value
                                
                            if 'delta' in unparsed_choice and isinstance(unparsed_choice["delta"], ChatCompletionStreamResponseDelta):
                                parsed_choice["delta"] = unparsed_choice["delta"].model_dump()
                            
                            parsed_choices.append(unparsed_choice)
                        except Exception as e:
                            self.logger.warning(f"Failed to parse streaming choice: {e}")
                            # Keep the raw choice data if parsing fails
                            parsed_choices.append(choice_data)
                    modified_chunk['choices'] = parsed_choices
                
                # Parse usage into Usage object if present
                if 'usage' in chunk and chunk['usage'] is not None:
                    try:
                        parsed_usage = Usage.model_validate(chunk['usage'])
                        modified_chunk['usage'] = parsed_usage.model_dump()
                    except Exception as e:
                        self.logger.warning(f"Failed to parse streaming usage: {e}")
                        # Keep the raw usage data if parsing fails
                        modified_chunk['usage'] = chunk['usage']
                
                # Validate the complete modified chunk
                yield ChatCompletionStreamResponse.model_validate(modified_chunk)
            except Exception as e:
                self.logger.warning(f"Failed to parse streaming response chunk: {e}")
                # Still yield the raw chunk to avoid breaking the stream
                yield chunk
    
    def _create_request_model(self, messages: List[Union[Dict[str, Any], Message]], 
                            model: Optional[str] = None, **kwargs) -> ChatCompletionRequest:
        """
        Create and validate a ChatCompletionRequest model.
        
        Args:
            messages: The conversation messages.
            model: Model identifier to use.
            **kwargs: Additional parameters.
            
        Returns:
            ChatCompletionRequest: Validated request model.
            
        Raises:
            ValueError: If the request parameters are invalid.
        """
        # Build request data with validated nested components
        request_data = {}
        
        # Parse messages into Message objects
        processed_messages = []
        for msg in messages:
            if isinstance(msg, Message):
                processed_messages.append(msg.model_dump())
            else:
                try:
                    parsed_message = Message.model_validate(msg)
                    processed_messages.append(parsed_message.model_dump())
                except Exception as e:
                    self.logger.warning(f"Failed to parse message: {e}")
                    # Keep raw message if parsing fails
                    processed_messages.append(msg)
        request_data["messages"] = processed_messages
        
        # Add model if provided
        if model is not None:
            request_data["model"] = model
        
        # Parse nested components from kwargs
        for key, value in kwargs.items():
            if value is not None:
                if key == "tools" and isinstance(value, list):
                    # Parse tools into ChatCompletionTool objects
                    parsed_tools = []
                    for tool_data in value:
                        try:
                            if isinstance(tool_data, ChatCompletionTool):
                                parsed_tools.append(tool_data.model_dump())
                            else:
                                parsed_tool = ChatCompletionTool.model_validate(tool_data)
                                parsed_tools.append(parsed_tool.model_dump())
                        except Exception as e:
                            self.logger.warning(f"Failed to parse tool: {e}")
                            # Keep raw tool if parsing fails
                            parsed_tools.append(tool_data)
                    request_data[key] = parsed_tools
                elif key == "functions" and isinstance(value, list):
                    # Parse functions into FunctionDefinition objects
                    parsed_functions = []
                    for func_data in value:
                        try:
                            if isinstance(func_data, FunctionDefinition):
                                parsed_functions.append(func_data.model_dump())
                            else:
                                parsed_function = FunctionDefinition.model_validate(func_data)
                                parsed_functions.append(parsed_function.model_dump())
                        except Exception as e:
                            self.logger.warning(f"Failed to parse function: {e}")
                            # Keep raw function if parsing fails
                            parsed_functions.append(func_data)
                    request_data[key] = parsed_functions
                elif key == "reasoning":
                    # Parse reasoning into ReasoningConfig object
                    if isinstance(value, ReasoningConfig):
                        request_data[key] = value.model_dump(exclude_none=True)
                    elif isinstance(value, dict):
                        try:
                            parsed_reasoning = ReasoningConfig.model_validate(value)
                            request_data[key] = parsed_reasoning.model_dump(exclude_none=True)
                        except Exception as e:
                            self.logger.warning(f"Failed to parse reasoning config: {e}")
                            # Keep raw reasoning if parsing fails
                            request_data[key] = value
                    else:
                        request_data[key] = value
                elif key == "response_format" and isinstance(value, dict):
                    # Parse response_format into ResponseFormat object
                    try:
                        parsed_response_format = ResponseFormat.model_validate(value)
                        request_data[key] = parsed_response_format.model_dump()
                    except Exception as e:
                        self.logger.warning(f"Failed to parse response format: {e}")
                        # Keep raw response_format if parsing fails
                        request_data[key] = value
                else:
                    # For other parameters, use them as-is
                    request_data[key] = value
            
        # Create and validate the request model
        return ChatCompletionRequest.model_validate(request_data)
    
    def _parse_tool_calls(self, tool_calls_data: List[Dict[str, Any]]) -> List[ChatCompletionToolCall]:
        """
        Parse raw tool calls data into ChatCompletionToolCall models.
        
        Args:
            tool_calls_data: Raw tool calls data from API response.
            
        Returns:
            List[ChatCompletionToolCall]: Parsed tool calls.
        """
        parsed_calls = []
        for call_data in tool_calls_data:
            try:
                # Create a copy of the tool call data to modify
                modified_call_data = call_data.copy()
                
                # Parse the function component into ToolCallFunction object
                if 'function' in call_data:
                    try:
                        parsed_function = ToolCallFunction.model_validate(call_data['function'])
                        modified_call_data['function'] = parsed_function.model_dump()
                    except Exception as e:
                        self.logger.warning(f"Failed to parse tool call function: {e}")
                        # Keep the raw function data if parsing fails
                        modified_call_data['function'] = call_data['function']
                
                # Validate the complete modified tool call
                parsed_tool_call = ChatCompletionToolCall.model_validate(modified_call_data)
                parsed_calls.append(parsed_tool_call)
            except Exception as e:
                self.logger.warning(f"Failed to parse tool call: {e}")
                # Keep the raw data if parsing fails
                parsed_calls.append(call_data)
        return parsed_calls
    
    def _parse_function_call(self, function_call_data: Dict[str, Any]) -> ChatCompletionFunction:
        """
        Parse raw function call data into ChatCompletionFunction model.
        
        Args:
            function_call_data: Raw function call data from API response.
            
        Returns:
            ChatCompletionFunction: Parsed function call.
        """
        try:
            # Create a copy of the function call data to modify
            modified_function_data = function_call_data.copy()
            
            # Validate the arguments field as JSON if present
            if 'arguments' in function_call_data:
                try:
                    # The ChatCompletionFunction model already validates JSON in arguments field
                    # But we can add additional validation here if needed
                    import json
                    json.loads(function_call_data['arguments'])
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON in function arguments: {e}")
                    # Keep the raw arguments if JSON parsing fails
                    pass
            
            # Validate the complete function call
            return ChatCompletionFunction.model_validate(modified_function_data)
        except Exception as e:
            self.logger.warning(f"Failed to parse function call: {e}")
            # Return raw data if parsing fails
            return function_call_data
    
    def create(self, messages: List[Union[Dict[str, Any], Message]], 
               model: Optional[str] = None, temperature: Optional[float] = None, 
               top_p: Optional[float] = None, max_tokens: Optional[int] = None,
               stop: Optional[Union[str, List[str]]] = None, n: Optional[int] = None,
               stream: Optional[bool] = None, presence_penalty: Optional[float] = None,
               frequency_penalty: Optional[float] = None, user: Optional[str] = None,
               functions: Optional[List[FunctionParameters]] = None,
               function_call: Optional[Union[str, FunctionToolChoice]] = None,
               tools: Optional[List[ChatCompletionTool]] = None,
               tool_choice: Optional[ToolChoice] = None,
               response_format: Optional[Dict[str, Any]] = None,
               reasoning: Optional[Union[Dict[str, Any], ReasoningConfig]] = None,
               include_reasoning: Optional[bool] = None,
               state_file: Optional[str] = None, chunk_size: int = 8192,
               include: Optional[Dict[str, bool]] = None,
               validate_request: bool = False,
               **kwargs) -> Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]:
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
            tools (Optional[List[ChatCompletionTool]]): Tool definitions.
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
            validate_request (bool): Whether to validate the request using ChatCompletionRequest model.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]: Either a parsed 
            response model (non-streaming) or an iterator of parsed response chunks (streaming).
            
        Raises:
            APIError: If the API request fails.
            StreamingError: If the streaming request fails.
        """
        # Optionally validate the request using ChatCompletionRequest model
        if validate_request:
            try:
                request_params = {
                    'temperature': temperature, 'top_p': top_p, 'max_tokens': max_tokens,
                    'stop': stop, 'n': n, 'stream': stream, 'presence_penalty': presence_penalty,
                    'frequency_penalty': frequency_penalty, 'user': user, 'functions': functions,
                    'function_call': function_call, 'tools': tools, 'tool_choice': tool_choice,
                    'response_format': response_format, 'reasoning': reasoning,
                    'include_reasoning': include_reasoning, 'include': include, **kwargs
                }
                validated_request = self._create_request_model(messages, model, **request_params)
                self.logger.debug("Request validation successful")
            except Exception as e:
                self.logger.error(f"Request validation failed: {e}")
                raise ValueError(f"Invalid request parameters: {e}")
        
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
                return self._parse_streaming_response(streamer.get_result())
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
            
            # Parse and return ChatCompletionResponse model
            try:
                return ChatCompletionResponse.model_validate(response.json())
            except Exception as e:
                self.logger.warning(f"Failed to parse chat completion response: {e}")
                # Fall back to raw response if parsing fails
                return response.json()
    
    def resume_stream(self, state_file: str) -> Iterator[ChatCompletionStreamResponse]:
        """
        Resume a streaming chat completions request from saved state.
        
        Args:
            state_file (str): File containing the saved state.
            
        Returns:
            Iterator[ChatCompletionStreamResponse]: Resumed stream of parsed chat completion chunks.
            
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
            return self._parse_streaming_response(streamer.get_result())
        except Exception as e:
            self.logger.error(f"Resuming chat completions stream failed: {e}")
            raise ResumeError(
                f"Resuming chat completions stream failed: {e}",
                state_file=state_file,
                original_error=e
            )
