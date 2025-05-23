"""Completions models for OpenRouter Client.

This module defines Pydantic models for the Completions API requests and responses.

Exported:
- CompletionRequest: Request model for completions
- CompletionResponse: Response model for completions
- CompletionResponseChoice: Choice model for completion responses
- CompletionStreamResponse: Stream response model for completions
- LogProbs: Log probabilities model
"""

from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field, model_validator

from ..types import FinishReason
from .core import ResponseFormat, FunctionDefinition, ToolDefinition, Prediction
from .providers import ProviderPreferences
from .chat import Usage


class LogProbs(BaseModel):
    """
    Log probabilities for tokens in completions.
    
    Attributes:
        tokens (List[str]): The tokens that the model used.
        token_logprobs (List[float]): Log probabilities of the tokens.
        top_logprobs (Optional[List[Dict[str, float]]]): The most likely tokens for each position.
        text_offset (List[int]): Character offsets for the tokens in the generated text.
    """
    tokens: List[str] = Field(..., description="The tokens that the model used")
    token_logprobs: List[float] = Field(..., description="Log probabilities of the tokens")
    top_logprobs: Optional[List[Dict[str, float]]] = Field(None, description="The most likely tokens for each position")
    text_offset: List[int] = Field(..., description="Character offsets for the tokens in the generated text")
    
    @model_validator(mode="after")
    def validate_list_lengths(self) -> "LogProbs":
        """
        Validates that the lengths of the input lists are equal.

        Args:
            values (Dict[str, Any]): The input values.
        
        Returns:
            self: The validated instance.
        """
        if len(self.tokens) != len(self.token_logprobs):
            raise ValueError("All lists must have the same length")
        if len(self.tokens) != len(self.text_offset):
            raise ValueError("All lists must have the same length")
        if self.top_logprobs is not None and len(self.tokens) != len(self.top_logprobs):
            raise ValueError("All lists must have the same length")
        
        return self


class CompletionsRequest(BaseModel):
    """
    Request model for completions.
    
    Attributes:
        prompt (str): The prompt to generate completions for.
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
    """
    prompt: str = Field(..., min_length=1, description="The prompt to generate completions for")
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
    
    @model_validator(mode="after")
    def validate_function_and_tools(self) -> "CompletionsRequest":
        """
        Validate that functions and tools are not used together.
        
        Returns:
            CompletionRequest: Self after validation.
            
        Raises:
            ValueError: If both functions and tools are specified.
        """
        if self.functions is not None and self.tools is not None:
            raise ValueError("Cannot specify both 'functions' and 'tools' parameters")
        return self


class CompletionsResponseChoice(BaseModel):
    """
    Choice model for completions responses.
    
    Attributes:
        text (str): The generated text.
        index (int): Index of this choice in the list of choices.
        logprobs (Optional[LogProbs]): Log probabilities for tokens if requested.
        finish_reason (Optional[FinishReason]): The normalized reason why the model stopped generating tokens.
        native_finish_reason (Optional[str]): The raw finish_reason from the provider.
    """
    text: str = Field(..., description="The generated text")
    index: int = Field(..., ge=0, description="Index of this choice in the list of choices")
    logprobs: Optional[LogProbs] = Field(None, description="Log probabilities for tokens if requested")
    finish_reason: Optional[FinishReason] = Field(None, description="The normalized reason why the model stopped generating tokens")
    native_finish_reason: Optional[str] = Field(None, description="The raw finish_reason from the provider")


class CompletionsResponse(BaseModel):
    """
    Response model for completions.
    
    Attributes:
        id (str): Unique ID for the completion.
        object (str): Object type, one of "chat.completion" or "chat.completion.chunk".
        created (int): Unix timestamp (in seconds) of when the completion was created.
        model (str): Model used for the completion.
        choices (List[CompletionsResponseChoice]): List of completion choices.
        usage (Optional[Usage]): Token usage statistics if requested.
        system_fingerprint (Optional[str]): System fingerprint for the completion.
    """
    id: str = Field(..., description="Unique ID for the completion")
    object: str = Field("chat.completion", description="Object type, one of 'chat.completion' or 'chat.completion.chunk'")
    created: int = Field(..., description="Unix timestamp (in seconds) of when the completion was created")
    model: str = Field(..., description="Model used for the completion")
    choices: List[CompletionsResponseChoice] = Field(..., description="List of completion choices")
    usage: Optional[Usage] = Field(None, description="Token usage statistics if requested")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint for the completion")


class CompletionsStreamResponse(BaseModel):
    """
    Stream response model for completions.
    
    Attributes:
        id (str): Unique ID for the completion.
        object (str): Object type, always "chat.completion.chunk" for streaming.
        created (int): Unix timestamp (in seconds) of when the completion was created.
        model (str): Model used for the completion.
        choices (List[CompletionsResponseChoice]): List of completion chunks.
        usage (Optional[Usage]): Token usage statistics in the final chunk.
        system_fingerprint (Optional[str]): System fingerprint for the completion.
    """
    id: str = Field(..., description="Unique ID for the completion")
    object: str = Field("chat.completion.chunk", description="Object type, always 'chat.completion.chunk' for streaming")
    created: int = Field(..., description="Unix timestamp (in seconds) of when the completion was created")
    model: str = Field(..., description="Model used for the completion")
    choices: List[CompletionsResponseChoice] = Field(..., description="List of completion chunks")
    usage: Optional[Usage] = Field(None, description="Token usage statistics in the final chunk")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint for the completion")