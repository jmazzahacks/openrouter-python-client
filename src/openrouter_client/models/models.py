"""Models-related API models for OpenRouter Client.

This module defines Pydantic models for the Models API requests and responses.

Exported:
- Model: Model information
- ModelList: List of available models
- ModelPermission: Model permission settings
- ModelProvider: Model provider information
- ModelPricing: Model pricing information
- ModelQuantization: Model quantization information
- ModelContextWindow: Model context window information
- ModelDataPolicy: Model data policy information
- ModelEndpoint: Model endpoint information
- ModelEndpointsRequest: Request parameters for listing model endpoints
- ModelEndpointsResponse: Response for model endpoints list
"""

from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class ModelPermission(BaseModel):
    """
    Permission settings for a model.
    
    Attributes:
        id (str): Unique identifier for the permission.
        object (str): Object type, typically "model_permission".
        created (int): Unix timestamp (in seconds) of when the permission was created.
        allow_create_engine (bool): Whether the model allows creating engines.
        allow_sampling (bool): Whether the model allows sampling.
        allow_logprobs (bool): Whether the model allows log probabilities.
        allow_search_indices (bool): Whether the model allows search indices.
        allow_view (bool): Whether the model allows viewing.
        allow_fine_tuning (bool): Whether the model allows fine-tuning.
        organization (str): Organization associated with the permission.
        group (Optional[str]): Group associated with the permission.
        is_blocking (bool): Whether the permission is blocking.
    """
    id: str = Field(..., description="Unique identifier for the permission")
    object: str = Field("model_permission", description="Object type, typically 'model_permission'")
    created: int = Field(..., description="Unix timestamp (in seconds) of when the permission was created")
    allow_create_engine: bool = Field(..., description="Whether the model allows creating engines")
    allow_sampling: bool = Field(..., description="Whether the model allows sampling")
    allow_logprobs: bool = Field(..., description="Whether the model allows log probabilities")
    allow_search_indices: bool = Field(..., description="Whether the model allows search indices")
    allow_view: bool = Field(..., description="Whether the model allows viewing")
    allow_fine_tuning: bool = Field(..., description="Whether the model allows fine-tuning")
    organization: str = Field(..., description="Organization associated with the permission")
    group: Optional[str] = Field(None, description="Group associated with the permission")
    is_blocking: bool = Field(..., description="Whether the permission is blocking")


class ModelDataPolicy(BaseModel):
    """
    Data policy for a model provider.
    
    Attributes:
        retention (Optional[str]): Data retention policy (e.g., "7d", "30d", "indefinite").
        logging (Optional[bool]): Whether the provider logs data.
        training (Optional[bool]): Whether the provider uses data for training.
    """
    retention: Optional[str] = Field(None, description="Data retention policy (e.g., '7d', '30d', 'indefinite')")
    logging: Optional[bool] = Field(None, description="Whether the provider logs data")
    training: Optional[bool] = Field(None, description="Whether the provider uses data for training")


class ModelPricing(BaseModel):
    """
    Pricing information for a model.
    
    Attributes:
        prompt (Optional[str]): Price per prompt token in USD.
        completion (Optional[str]): Price per completion token in USD.
        request (Optional[str]): Price per request in USD.
        image (Optional[str]): Price per image in USD.
        web_search (Optional[str]): Price per web search in USD.
        internal_reasoning (Optional[str]): Price per internal reasoning in USD.
        input_cache_read (Optional[str]): Price per input cache read in USD.
        input_cache_write (Optional[str]): Price per input cache write in USD.
    """
    prompt: Optional[str] = Field(None, description="Price per prompt token in USD")
    completion: Optional[str] = Field(None, description="Price per completion token in USD")
    request: Optional[str] = Field(None, description="Price per request in USD")
    image: Optional[str] = Field(None, description="Price per image in USD")
    web_search: Optional[str] = Field(None, description="Price per web search in USD")
    internal_reasoning: Optional[str] = Field(None, description="Price per internal reasoning in USD")
    input_cache_read: Optional[str] = Field(None, description="Price per input cache read in USD")
    input_cache_write: Optional[str] = Field(None, description="Price per input cache write in USD")


class ModelQuantization(BaseModel):
    """
    Quantization information for a model.
    
    Attributes:
        bits (Optional[int]): Number of bits used for quantization.
        method (Optional[str]): Quantization method used.
        type (Optional[str]): Type of quantization (e.g., "int", "fp").
    """
    bits: Optional[int] = Field(None, description="Number of bits used for quantization")
    method: Optional[str] = Field(None, description="Quantization method used")
    type: Optional[str] = Field(None, description="Type of quantization (e.g., 'int', 'fp')")


class ModelContextWindow(BaseModel):
    """
    Context window information for a model.
    
    Attributes:
        default (int): Default context window size in tokens.
        maximum (Optional[int]): Maximum context window size in tokens.
    """
    default: int = Field(..., description="Default context window size in tokens")
    maximum: Optional[int] = Field(None, description="Maximum context window size in tokens")


class ModelProvider(BaseModel):
    """
    Provider information for a model.
    
    Attributes:
        id (str): Provider identifier.
        name (str): Provider name.
        status (Optional[str]): Provider status (e.g., "operational", "degraded").
        latency (Optional[float]): Average time to first token in seconds.
        throughput (Optional[float]): Average tokens per second.
        pricing (Optional[ModelPricing]): Pricing information.
        context_window (Optional[ModelContextWindow]): Context window information.
        quantization (Optional[ModelQuantization]): Quantization information.
        data_policy (Optional[ModelDataPolicy]): Data policy information.
    """
    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Provider name")
    status: Optional[str] = Field(None, description="Provider status (e.g., 'operational', 'degraded')")
    latency: Optional[float] = Field(None, description="Average time to first token in seconds")
    throughput: Optional[float] = Field(None, description="Average tokens per second")
    pricing: Optional[ModelPricing] = Field(None, description="Pricing information")
    context_window: Optional[ModelContextWindow] = Field(None, description="Context window information")
    quantization: Optional[ModelQuantization] = Field(None, description="Quantization information")
    data_policy: Optional[ModelDataPolicy] = Field(None, description="Data policy information")


class Architecture(BaseModel):
    """
    Architecture information for a model.
    
    Attributes:
        input_modalities (List[str]): List of supported input modalities (e.g., ["text", "image"]).
        output_modalities (List[str]): List of supported output modalities (e.g., ["text"]).
        tokenizer (Optional[str]): Tokenizer type used by the model (e.g., "GPT").
        instruct_type (Optional[str]): Type of instruction format supported.
    """
    input_modalities: List[str] = Field(..., description="List of supported input modalities (e.g., ['text', 'image'])")
    output_modalities: List[str] = Field(..., description="List of supported output modalities (e.g., ['text'])")
    tokenizer: Optional[str] = Field(None, description="Tokenizer type used by the model (e.g., 'GPT')")
    instruct_type: Optional[str] = Field(None, description="Type of instruction format supported")


class TopProvider(BaseModel):
    """
    Top provider information for a model.
    
    Attributes:
        is_moderated (bool): Whether the provider is moderated.
        context_length (Optional[int]): Context length in tokens.
        max_completion_tokens (Optional[int]): Maximum completion tokens.
    """
    is_moderated: bool = Field(..., description="Whether the provider is moderated")
    context_length: Optional[int] = Field(None, description="Context length in tokens")
    max_completion_tokens: Optional[int] = Field(None, description="Maximum completion tokens")


class Model(BaseModel):
    """
    Model information from the Models API.
    
    Attributes:
        id (str): Unique model identifier.
        object (str): Object type, typically "model".
        created (int): Unix timestamp (in seconds) of when the model was created.
        owned_by (str): Owner of the model.
        permissions (List[ModelPermission]): List of permissions for the model.
        root (Optional[str]): Root model identifier.
        parent (Optional[str]): Parent model identifier.
        context_window (Optional[int]): Context window size in tokens.
        pricing (Optional[ModelPricing]): Pricing information.
        providers (Optional[List[ModelProvider]]): List of providers offering this model.
        variants (Optional[List[str]]): List of model variants.
        description (Optional[str]): Description of the model.
        features (Optional[List[str]]): List of features supported by the model.
        formats (Optional[List[str]]): List of formats supported by the model.
        tags (Optional[List[str]]): List of tags associated with the model.
    """
    id: str = Field(..., description="Unique model identifier")
    object: str = Field("model", description="Object type, typically 'model'")
    created: int = Field(..., description="Unix timestamp (in seconds) of when the model was created")
    owned_by: str = Field(..., description="Owner of the model")
    permissions: List[ModelPermission] = Field(..., description="List of permissions for the model")
    root: Optional[str] = Field(None, description="Root model identifier")
    parent: Optional[str] = Field(None, description="Parent model identifier")
    context_window: Optional[int] = Field(None, description="Context window size in tokens")
    pricing: Optional[ModelPricing] = Field(None, description="Pricing information")
    providers: Optional[List[ModelProvider]] = Field(None, description="List of providers offering this model")
    variants: Optional[List[str]] = Field(None, description="List of model variants")
    description: Optional[str] = Field(None, description="Description of the model")
    features: Optional[List[str]] = Field(None, description="List of features supported by the model")
    formats: Optional[List[str]] = Field(None, description="List of formats supported by the model")
    tags: Optional[List[str]] = Field(None, description="List of tags associated with the model")


class ModelData(BaseModel):
    """
    Model information structure from the OpenRouter API.
    
    Attributes:
        id (str): Model identifier (provider/model-name).
        name (str): Human-readable model name.
        created (int): Unix timestamp of model creation date.
        description (Optional[str]): Optional description of the model.
        architecture (Optional[Architecture]): Architecture information including input/output modalities.
        top_provider (Optional[TopProvider]): Top provider information.
        pricing (ModelPricing): Pricing information for the model.
        canonical_slug (Optional[str]): Canonical slug for the model.
        context_length (int): Maximum context length in tokens.
        hugging_face_id (Optional[str]): Hugging Face model ID.
        per_request_limits (Dict[str, Any]): Per-request limits.
        supported_parameters (List[str]): List of supported parameters.
        max_completion_tokens (Optional[int]): Optional maximum tokens in completions.
        quantization (Optional[str]): Optional model quantization level (e.g., "fp16").
    """
    id: str = Field(..., description="Model identifier (provider/model-name)")
    name: str = Field(..., description="Human-readable model name")
    created: int = Field(..., description="Unix timestamp of model creation date")
    description: Optional[str] = Field(None, description="Optional description of the model")
    architecture: Optional[Architecture] = Field(None, description="Architecture information including input/output modalities")
    top_provider: Optional[TopProvider] = Field(None, description="Top provider information")
    pricing: ModelPricing = Field(..., description="Pricing information for the model")
    canonical_slug: Optional[str] = Field(None, description="Canonical slug for the model")
    context_length: int = Field(..., description="Maximum context length in tokens")
    hugging_face_id: Optional[str] = Field(None, description="Hugging Face model ID")
    per_request_limits: Optional[Dict[str, Any]] = Field(None, description="Per-request limits")
    supported_parameters: List[str] = Field(default_factory=list, description="List of supported parameters")
    max_completion_tokens: Optional[int] = Field(None, description="Optional maximum tokens in completions")
    quantization: Optional[str] = Field(None, description="Optional model quantization level (e.g., \"fp16\")")


class ModelsResponse(BaseModel):
    """
    Response structure for models list endpoint.
    
    Attributes:
        data (List[ModelData]): List of available models.
    """
    data: List[ModelData] = Field(..., description="List of available models")


class ModelList(BaseModel):
    """
    List of available models from the Models API.
    
    Attributes:
        object (str): Object type, typically "list".
        data (List[Model]): List of model objects.
    """
    object: str = Field("list", description="Object type, typically 'list'")
    data: List[Model] = Field(..., description="List of model objects")


class ModelEndpoint(BaseModel):
    """
    Endpoint information for a model.
    
    Attributes:
        id (str): Unique endpoint identifier.
        name (str): Human-readable endpoint name.
        description (Optional[str]): Description of the endpoint.
        url (str): API URL for the endpoint.
        method (str): HTTP method for the endpoint (e.g., 'GET', 'POST').
        parameters (Optional[List[str]]): List of parameters accepted by the endpoint.
    """
    id: str = Field(..., description="Unique endpoint identifier")
    name: str = Field(..., description="Human-readable endpoint name")
    description: Optional[str] = Field(None, description="Description of the endpoint")
    url: str = Field(..., description="API URL for the endpoint")
    method: str = Field(..., description="HTTP method for the endpoint (e.g., 'GET', 'POST')")
    parameters: Optional[List[str]] = Field(None, description="List of parameters accepted by the endpoint")


class ModelEndpointsRequest(BaseModel):
    """
    Request parameters for listing model endpoints.
    
    Attributes:
        author (str): Model author/owner identifier.
        slug (str): Model slug identifier.
    """
    author: str = Field(..., description="Model author/owner identifier")
    slug: str = Field(..., description="Model slug identifier")


class ModelEndpointsResponse(BaseModel):
    """
    Response structure for the model endpoints list.
    
    Attributes:
        object (str): Object type, typically "list".
        data (dict): Model information dictionary containing model details.
    """
    object: str = Field("list", description="Object type, typically 'list'")
    data: dict = Field(..., description="Model information dictionary containing model details")

