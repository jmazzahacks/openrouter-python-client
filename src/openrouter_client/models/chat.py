"""Chat completion and function-calling models for OpenRouter Client.

This module defines Pydantic models for Chat Completions API requests and responses,
as well as function-calling models for parameter types, function calls, and tool calling capabilities.

Exported Chat Completion Models:
- ChatCompletionRequest: Request model for chat completions
- ChatCompletionResponse: Response model for chat completions
- ChatCompletionResponseChoice: Choice model for chat completion responses
- ChatCompletionStreamResponse: Stream response model for chat completions
- ChatCompletionStreamResponseChoice: Choice model for streaming chat completion responses
- ChatCompletionStreamResponseDelta: Delta model for streaming chat completion responses
- ReasoningConfig: Configuration for model reasoning/thinking tokens
- Usage: Token usage statistics model
- ChatCompletionFunction: Function call representation in chat completions
- ChatCompletionFunctionCall: Function call preference for chat completions
- ToolCallFunction: Function details for a tool call
- ChatCompletionToolCall: Tool call in chat completion responses
- ToolCallChunk: Tool call chunk for streaming responses
- ChatCompletionTool: Tool definition for chat completions
- ChatCompletionToolChoiceOption: Tool choice option for chat completions

Exported Function-Calling Models:
- ParameterType: Enum for parameter types in function definitions
- ParameterDefinition: Base model for parameter definitions in function schemas
- StringParameter: String parameter definition
- NumberParameter: Number parameter definition for both floating point and integer values
- BooleanParameter: Boolean parameter definition
- ArrayParameter: Array parameter definition
- ObjectParameter: Object parameter definition
- FunctionParameters: Container for function parameters following JSON Schema
- FunctionCall: Function call model for responses
- FunctionCallResult: Result of a function call from tool messages
- StructuredToolResult: Structured result of a tool call
- FunctionToolChoice: Tool choice specifying a particular function to call
- ToolChoice: Union type for tool choice options
"""

from enum import Enum
import json
from typing import Annotated, Dict, List, Optional, Any, Union, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

from ..types import ModelRole, FinishReason
from .core import Message, FunctionDefinition, ResponseFormat, ToolDefinition, Prediction
from .providers import ProviderPreferences


class Usage(BaseModel):
    """
    Token usage statistics for a completion.
    
    Attributes:
        prompt_tokens (int): Number of tokens in the prompt.
        completion_tokens (int): Number of tokens in the completion.
        total_tokens (int): Total number of tokens used.
        cached_tokens (Optional[int]): Number of tokens served from cache.
        cache_discount (Optional[float]): Discount amount applied due to cached tokens.
    """
    prompt_tokens: int = Field(..., ge=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., ge=0, description="Number of tokens in the completion")
    total_tokens: int = Field(..., ge=0, description="Total number of tokens used")
    cached_tokens: Optional[int] = Field(None, ge=0, description="Number of tokens served from cache")
    cache_discount: Optional[float] = Field(None, ge=0, description="Discount amount applied due to cached tokens")


class ChatCompletionFunction(BaseModel):
    """
    Function call representation in chat completions.
    
    Attributes:
        name (str): Name of the function to call.
        arguments (str): String containing the function arguments as a JSON object.
    """
    name: str = Field(..., min_length=1, description="Name of the function to call")
    arguments: str = Field(..., min_length=1, description="String containing the function arguments as a JSON object")

    @field_validator("arguments")
    def validate_arguments(cls, value):
        try:
            json.loads(value)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON")
        
        return value

class ChatCompletionFunctionCall(BaseModel):
    """
    Function call preference for chat completions.
    
    Attributes:
        name (Optional[str]): Name of the function to call.
    """
    model_config = ConfigDict(extra='forbid')
    
    name: Optional[str] = Field(None, min_length=1, description="Name of the function to call")


class ToolCallFunction(BaseModel):
    """
    Function details for a tool call.
    
    Attributes:
        name (str): Name of the function to call.
        arguments (str): String containing the function arguments as a JSON object.
    """
    name: str = Field(..., min_length=1, description="Name of the function to call")
    arguments: str = Field(..., min_length=1, description="String containing the function arguments as a JSON object")
    
    @field_validator("arguments")
    def validate_arguments(cls, value):
        try:
            json.loads(value)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON")

        return value


class ChatCompletionToolCall(BaseModel):
    """
    Tool call in chat completion responses.
    
    Attributes:
        id (str): Unique ID for the tool call.
        type (str): Type of tool call, typically "function".
        function (ToolCallFunction): Function details for the tool call.
    """
    id: str = Field(..., description="Unique ID for the tool call")
    type: str = Field("function", description="Type of tool call, typically 'function'")
    function: ToolCallFunction = Field(..., description="Function details for the tool call")


class ToolCallChunk(BaseModel):
    """
    Tool call chunk for streaming responses.
    
    Attributes:
        id (str): Unique ID for the tool call.
        type (str): Type of tool call, typically "function".
        function (ToolCallFunction): Function details for the tool call.
        index (int): Index of the tool call in the list of tool calls.
    """
    id: str = Field(..., description="Unique ID for the tool call")
    type: str = Field("function", description="Type of tool call, typically 'function'")
    function: Dict[str, str] = Field(..., description="Function details for the tool call")
    index: int = Field(..., description="Index of the tool call in the list of tool calls")


class ChatCompletionTool(BaseModel):
    """
    Tool definition for chat completions.
    
    Attributes:
        type (str): Type of tool, typically "function".
        function (FunctionDefinition): Function definition for the tool.
    """
    type: str = Field("function", description="Type of tool, currently only 'function' is supported")
    function: FunctionDefinition = Field(..., description="Function definition for the tool")


class ChatCompletionToolChoiceOption(BaseModel):
    """
    Tool choice option for chat completions.
    
    Attributes:
        type (str): Type of tool choice, typically "function".
        function (ChatCompletionFunctionCall): Function call preference.
    """
    type: str = Field("function", description="Type of tool choice, typically 'function'")
    function: ChatCompletionFunctionCall = Field(..., description="Function call preference")


class ChatCompletionRequest(BaseModel):
    """
    Request model for chat completions.
    
    Attributes:
        messages (List[Message]): List of messages in the conversation.
        model (str): Model identifier to use.
        temperature (Optional[float]): Sampling temperature (0.0 to 2.0).
        top_p (Optional[float]): Nucleus sampling parameter (0.0 to 1.0).
        max_tokens (Optional[int]): Maximum tokens to generate.
        stop (Optional[Union[str, List[str]]]): Stop sequences to end generation.
        n (Optional[int]): Number of completions to generate.
        stream (Optional[bool]): Whether to stream responses.
        logprobs (Optional[int]): Number of log probabilities to include.
        echo (Optional[bool]): Whether to echo the prompt.
        presence_penalty (Optional[float]): Penalty for token presence (-2.0 to 2.0).
        frequency_penalty (Optional[float]): Penalty for token frequency (-2.0 to 2.0).
        user (Optional[str]): User identifier for tracking.
        functions (Optional[List[FunctionDefinition]]): Function definitions.
        function_call (Optional[Union[str, Dict[str, Any]]]): Function calling control.
        tools (Optional[List[ToolDefinition]]): Tool definitions.
        tool_choice (Optional[Union[str, Dict[str, Any]]]): Tool choice control.
        response_format (Optional[ResponseFormat]): Format specification for response.
        models (Optional[List[str]]): List of model IDs to use as fallbacks.
        provider (Optional[ProviderPreferences]): Provider routing preferences.
        extra_headers (Optional[Dict[str, str]]): Additional headers for the request.
        http_referer (Optional[str]): HTTP referer for the request.
        x_title (Optional[str]): Title for the request for rankings on openrouter.ai.
        include (Optional[Dict[str, bool]]): Fields to include in the response.
        seed (Optional[int]): Random seed for deterministic results.
        top_k (Optional[int]): Number of most likely tokens to consider (range: [1, infinity)).
        top_logprobs (Optional[int]): Number of most likely tokens to return logprobs for.
        min_p (Optional[float]): Minimum probability threshold for token choices (range: [0, 1]).
        top_a (Optional[float]): Dynamic top-k value calculated based on token probability (range: [0, 1]).
        repetition_penalty (Optional[float]): Penalty for repetitive tokens (range: (0, 2]).
        logit_bias (Optional[Dict[int, float]]): Bias to add to logits of specified tokens.
        prediction (Optional[Prediction]): Predicted output to reduce latency.
        transforms (Optional[List[str]]): OpenRouter-specific prompt transforms.
        route (Optional[str]): Model routing strategy, e.g., 'fallback'.
        reasoning (Optional[ReasoningConfig]): Configuration for reasoning tokens.
    """
    messages: List[Message]
    model: str = Field(..., min_length=1, description="Model identifier to use")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences to end generation")
    n: Optional[int] = Field(None, gt=0, description="Number of completions to generate")
    stream: Optional[bool] = Field(None, description="Whether to stream responses")
    logprobs: Optional[int] = Field(None, ge=0, description="Number of log probabilities to include")
    echo: Optional[bool] = Field(None, description="Whether to echo the prompt")
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Penalty for token presence")
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Penalty for token frequency")
    user: Optional[str] = Field(None, description="User identifier for tracking")
    functions: Optional[List[FunctionDefinition]] = Field(None, description="Function definitions")
    function_call: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Function calling control")
    tools: Optional[List[ToolDefinition]] = Field(None, description="Tool definitions")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Tool choice control")
    response_format: Optional[ResponseFormat] = Field(None, description="Format specification for response")
    models: Optional[List[str]] = Field(None, description="List of model IDs to use as fallbacks")
    provider: Optional[ProviderPreferences] = Field(None, description="Provider routing preferences")
    extra_headers: Optional[Dict[str, str]] = Field(None, description="Additional headers for the request")
    http_referer: Optional[str] = Field(None, description="HTTP referer for the request")
    x_title: Optional[str] = Field(None, description="Title for the request for rankings on openrouter.ai")
    include: Optional[Dict[str, bool]] = Field(None, description="Fields to include in the response")
    seed: Optional[int] = Field(None, description="Random seed for deterministic results")
    top_k: Optional[int] = Field(None, gt=0, description="Number of most likely tokens to consider (range: [1, infinity))")
    top_logprobs: Optional[int] = Field(None, ge=0, description="Number of most likely tokens to return logprobs for")
    min_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum probability threshold for token choices (range: [0, 1])")
    top_a: Optional[float] = Field(None, ge=0.0, le=1.0, description="Dynamic top-k value calculated based on token probability (range: [0, 1])")
    repetition_penalty: Optional[float] = Field(None, gt=0.0, le=2.0, description="Penalty for repetitive tokens (range: (0, 2])")
    logit_bias: Optional[Dict[int, float]] = Field(None, description="Bias to add to logits of specified tokens")
    prediction: Optional[Prediction] = Field(None, description="Predicted output to reduce latency")
    transforms: Optional[List[str]] = Field(None, description="OpenRouter-specific prompt transforms")
    route: Optional[str] = Field(None, description="Model routing strategy, e.g., 'fallback'")
    reasoning: Optional["ReasoningConfig"] = Field(None, description="Configuration for reasoning tokens")
    
    @model_validator(mode="after")
    def validate_function_and_tools(self) -> "ChatCompletionRequest":
        """
        Validate that functions and tools are not used together.
        
        Returns:
            ChatCompletionRequest: Self after validation.
            
        Raises:
            ValueError: If both functions and tools are specified.
        """
        if self.functions and self.tools:
            raise ValueError("Cannot specify both 'functions' and 'tools' parameters")
        return self


class ChatCompletionResponseChoice(BaseModel):
    """
    Choice model for chat completion responses.
    
    Attributes:
        index (int): Index of this choice in the list of choices.
        message (Message): The message generated by the model.
        finish_reason (Optional[FinishReason]): The normalized reason why the model stopped generating tokens.
        native_finish_reason (Optional[str]): The raw finish_reason from the provider.
        logprobs (Optional[Dict[str, Any]]): Log probabilities for tokens.
    """
    index: int = Field(..., description="Index of this choice in the list of choices")
    message: Message = Field(..., description="The message generated by the model")
    finish_reason: Optional[FinishReason] = Field(None, description="The normalized reason why the model stopped generating tokens")
    native_finish_reason: Optional[str] = Field(None, description="The raw finish_reason from the provider")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Log probabilities for tokens")


class ReasoningConfig(BaseModel):
    """
    Configuration for model reasoning/thinking tokens.
    
    Attributes:
        effort (Optional[str]): Reasoning effort level ("high", "medium", or "low").
        max_tokens (Optional[int]): Maximum tokens to use for reasoning.
        exclude (Optional[bool]): Whether to exclude reasoning tokens from response.
    """
    effort: Optional[str] = Field(None, description="Reasoning effort level (high, medium, or low)")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to use for reasoning")
    exclude: Optional[bool] = Field(None, description="Whether to exclude reasoning tokens from response")
    
    @field_validator('effort')
    def validate_effort(cls, v: Optional[str]) -> Optional[str]:
        """Validate that effort is one of the allowed values."""
        if v is not None and v not in ["high", "medium", "low"]:
            raise ValueError("effort must be one of: high, medium, low")
        return v


class ChatCompletionStreamResponseDelta(BaseModel):
    """
    Delta model for streaming chat completion responses.
    
    Attributes:
        role (Optional[ModelRole]): Role of the message sender.
        content (Optional[Union[str, List[Dict[str, Any]]]]): Content of the message.
            Can be a string or a list of content parts (text/images/files).
        function_call (Optional[Dict[str, Any]]): Function call details if applicable.
        tool_calls (Optional[List[Dict[str, Any]]]): Tool calls details if applicable.
    """
    role: Optional[ModelRole] = Field(None, description="Role of the message sender")
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(None, description="Content of the message")
    function_call: Optional[Dict[str, Any]] = Field(None, description="Function call details if applicable")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls details if applicable")


class ChatCompletionStreamResponseChoice(BaseModel):
    """
    Choice model for streaming chat completion responses.
    
    Attributes:
        index (int): Index of this choice in the list of choices.
        delta (ChatCompletionStreamResponseDelta): The message delta generated by the model.
        finish_reason (Optional[FinishReason]): The normalized reason why the model stopped generating tokens.
        native_finish_reason (Optional[str]): The raw finish_reason from the provider.
        logprobs (Optional[Dict[str, Any]]): Log probabilities for tokens.
    """
    index: int = Field(..., description="Index of this choice in the list of choices")
    delta: ChatCompletionStreamResponseDelta = Field(..., description="The message delta generated by the model")
    finish_reason: Optional[FinishReason] = Field(None, description="The normalized reason why the model stopped generating tokens")
    native_finish_reason: Optional[str] = Field(None, description="The raw finish_reason from the provider")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Log probabilities for tokens")


class ChatCompletionResponse(BaseModel):
    """
    Response model for chat completions.
    
    Attributes:
        id (str): Unique ID for the completion.
        object (str): Object type, typically "chat.completion".
        created (int): Unix timestamp (in seconds) of when the completion was created.
        model (str): Model used for the completion.
        choices (List[ChatCompletionResponseChoice]): List of completion choices.
        usage (Optional[Usage]): Token usage statistics if requested.
        system_fingerprint (Optional[str]): System fingerprint for the completion.
    """
    id: str = Field(..., description="Unique ID for the completion")
    object: str = Field("chat.completion", description="Object type, typically 'chat.completion'")
    created: int = Field(..., description="Unix timestamp (in seconds) of when the completion was created")
    model: str = Field(..., description="Model used for the completion")
    choices: List[ChatCompletionResponseChoice] = Field(..., description="List of completion choices")
    usage: Optional[Usage] = Field(None, description="Token usage statistics if requested")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint for the completion")


class ChatCompletionStreamResponse(BaseModel):
    """
    Stream response model for chat completions.
    
    Attributes:
        id (str): Unique ID for the completion.
        object (str): Object type, typically "chat.completion.chunk".
        created (int): Unix timestamp (in seconds) of when the completion was created.
        model (str): Model used for the completion.
        choices (List[ChatCompletionStreamResponseChoice]): List of completion chunks.
        usage (Optional[Usage]): Token usage statistics if requested.
        system_fingerprint (Optional[str]): System fingerprint for the completion.
    """
    id: str = Field(..., description="Unique ID for the completion")
    object: str = Field("chat.completion.chunk", description="Object type, typically 'chat.completion.chunk'")
    created: int = Field(..., description="Unix timestamp (in seconds) of when the completion was created")
    model: str = Field(..., description="Model used for the completion")
    choices: List[ChatCompletionStreamResponseChoice] = Field(..., description="List of completion chunks")
    usage: Optional[Usage] = Field(None, description="Token usage statistics if requested")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint for the completion")


# Function-calling models moved from models.py

class ParameterType(str, Enum):
    """Enum for parameter types in function definitions."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


class ParameterDefinition(BaseModel):
    """
    Base model for parameter definitions in function schemas.
    
    Attributes:
        type (ParameterType): Type of the parameter.
        description (Optional[str]): Description of the parameter.
        enum (Optional[List[Any]]): List of allowed values for the parameter.
        default (Optional[Any]): Default value for the parameter.
    """
    type: Union[ParameterType, Annotated[List[ParameterType], Field(min_length=1)]] = Field(..., description="Type of the parameter")
    description: Optional[str] = Field(None, description="Description of the parameter")
    enum: Optional[List[Any]] = Field(None, description="List of allowed values for the parameter")
    default: Optional[Any] = Field(None, description="Default value for the parameter")


class StringParameter(ParameterDefinition):
    """
    String parameter definition.
    
    Attributes:
        type (ParameterType): Type of the parameter, must be "string".
        min_length (Optional[int]): Minimum length of the string.
        max_length (Optional[int]): Maximum length of the string.
        pattern (Optional[str]): Regex pattern the string must match.
        format (Optional[str]): Format of the string (e.g., "date-time", "email").
    """
    type: Literal[ParameterType.STRING] = Field(ParameterType.STRING, description="Type of the parameter, must be 'string'")
    min_length: Optional[int] = Field(None, ge=0, description="Minimum length of the string")
    max_length: Optional[int] = Field(None, ge=0, description="Maximum length of the string")
    pattern: Optional[str] = Field(None, description="Regex pattern the string must match")
    format: Optional[str] = Field(None, description="Format of the string (e.g., 'date-time', 'email')")
    
    @model_validator(mode="after")
    def validate_length_constraints(self) -> "StringParameter":
        """Validate that maxLength is greater than or equal to minLength."""
        if self.min_length is not None and self.max_length is not None:
            if self.max_length < self.min_length:
                raise ValueError("maxLength must be greater than or equal to minLength")
        return self


class NumberParameter(ParameterDefinition):
    """
    Number parameter definition, for both floating point and integer values.
    
    Attributes:
        type (ParameterType): Type of the parameter, either "number" or "integer".
        minimum (Optional[float]): Minimum value of the number.
        maximum (Optional[float]): Maximum value of the number.
        exclusive_minimum (Optional[bool]): Whether the minimum value is exclusive.
        exclusive_maximum (Optional[bool]): Whether the maximum value is exclusive.
        multiple_of (Optional[float]): The number must be a multiple of this value.
    """
    type: Union[Literal[ParameterType.NUMBER], Literal[ParameterType.INTEGER]] = Field(
        ..., description="Type of the parameter, either 'number' or 'integer'"
    )
    minimum: Optional[float] = Field(None, description="Minimum value of the number")
    maximum: Optional[float] = Field(None, description="Maximum value of the number")
    exclusive_minimum: Optional[bool] = Field(None, description="Whether the minimum value is exclusive")
    exclusive_maximum: Optional[bool] = Field(None, description="Whether the maximum value is exclusive")
    multiple_of: Optional[float] = Field(None, gt=0, description="The number must be a multiple of this value")
    
    @model_validator(mode="after")
    def validate_range_constraints(self) -> "NumberParameter":
        """Validate that maximum is greater than or equal to minimum."""
        if self.minimum is not None and self.maximum is not None:
            if self.maximum < self.minimum:
                raise ValueError("maximum must be greater than or equal to minimum")
        return self


class BooleanParameter(ParameterDefinition):
    """
    Boolean parameter definition.
    
    Attributes:
        type (ParameterType): Type of the parameter, must be "boolean".
    """
    type: Literal[ParameterType.BOOLEAN] = Field(ParameterType.BOOLEAN, description="Type of the parameter, must be 'boolean'")


class ArrayParameter(ParameterDefinition):
    """
    Array parameter definition.
    
    Attributes:
        type (ParameterType): Type of the parameter, must be "array".
        items (Union[ParameterDefinition, List[ParameterDefinition]]): Schema for the array items.
        min_items (Optional[int]): Minimum number of items in the array.
        max_items (Optional[int]): Maximum number of items in the array.
        unique_items (Optional[bool]): Whether all items in the array must be unique.
    """
    type: Literal[ParameterType.ARRAY] = Field(ParameterType.ARRAY, description="Type of the parameter, must be 'array'")
    items: Dict[str, Any] = Field(..., description="Schema for the array items")
    min_items: Optional[int] = Field(None, ge=0, description="Minimum number of items in the array")
    max_items: Optional[int] = Field(None, ge=0, description="Maximum number of items in the array")
    unique_items: Optional[bool] = Field(None, description="Whether all items in the array must be unique")
    
    @model_validator(mode="after")
    def validate_items_constraints(self) -> "ArrayParameter":
        """Validate that maxItems is greater than or equal to minItems."""
        if self.min_items is not None and self.max_items is not None:
            if self.max_items < self.min_items:
                raise ValueError("maxItems must be greater than or equal to minItems")
        return self


class ObjectParameter(ParameterDefinition):
    """
    Object parameter definition.
    
    Attributes:
        type (ParameterType): Type of the parameter, must be "object".
        properties (Dict[str, ParameterDefinition]): Properties of the object.
        required (Optional[List[str]]): List of required property names.
        additional_properties (Optional[Union[bool, ParameterDefinition]]): 
            Whether additional properties are allowed, or a schema for additional properties.
    """
    type: Literal[ParameterType.OBJECT] = Field(ParameterType.OBJECT, description="Type of the parameter, must be 'object'")
    properties: Dict[str, Dict[str, Any]] = Field({}, description="Properties of the object")
    required: Optional[List[str]] = Field(None, description="List of required property names")
    additional_properties: Optional[Union[bool, Dict[str, Any]]] = Field(
        None, description="Whether additional properties are allowed, or a schema for additional properties"
    )
    
    @model_validator(mode="after")
    def validate_required_properties(self) -> "ObjectParameter":
        """Validate that all required properties exist in properties."""
        if self.required is not None and self.properties is not None:
            for prop in self.required:
                if prop not in self.properties:
                    raise ValueError(f"Required property '{prop}' not found in properties")
        return self


class FunctionParameters(BaseModel):
    """
    Container for function parameters following JSON Schema.
    
    Attributes:
        type (Literal["object"]): Type of the parameters, must be "object".
        properties (Dict[str, Any]): Properties of the parameters object.
        required (Optional[List[str]]): List of required parameter names.
    """
    type: Literal["object"] = Field("object", description="Type of the parameters, must be 'object'")
    properties: Dict[str, Any] = Field(..., description="Properties of the parameters object")
    required: Optional[List[str]] = Field(None, description="List of required parameter names")


class FunctionCall(BaseModel):
    """
    Function call model for responses.
    
    Attributes:
        name (str): Name of the function to call.
        arguments (str): String containing the function arguments as a JSON object.
        id (Optional[str]): Unique ID for the function call.
    """
    name: str = Field(..., min_length=1, description="Name of the function to call")
    arguments: str = Field(..., min_length=1, description="String containing the function arguments as a JSON object")
    id: Optional[str] = Field(None, description="Unique ID for the function call")


class FunctionCallResult(BaseModel):
    """
    Result of a function call from tool messages.
    
    Attributes:
        name (str): Name of the function that was called.
        arguments (Dict[str, Any]): Arguments that were passed to the function.
        result (Any): Result of the function call.
    """
    name: str = Field(..., min_length=1, description="Name of the function that was called")
    arguments: Dict[str, Any] = Field(..., description="Arguments that were passed to the function")
    result: Any = Field(..., description="Result of the function call")


class StructuredToolResult(BaseModel):
    """
    Structured result of a tool call.
    
    Attributes:
        tool_call_id (str): ID of the tool call this result is for.
        function (FunctionCallResult): Result of the function call.
    """
    tool_call_id: str = Field(..., min_length=1, description="ID of the tool call this result is for")
    function: FunctionCallResult = Field(..., description="Result of the function call")


class FunctionToolChoice(BaseModel):
    """
    Tool choice specifying a particular function to call.
    
    Attributes:
        type (Literal["function"]): Type of the tool choice, must be "function".
        function (ChatCompletionFunctionCall): Function to call.
    """
    type: Literal["function"] = Field("function", description="Type of the tool choice, must be 'function'")
    function: ChatCompletionFunctionCall = Field(..., description="Function to call")


# Type aliases for improved code readability
ToolChoice = Union[Literal["none"], Literal["auto"], FunctionToolChoice, Dict[str, Any]]