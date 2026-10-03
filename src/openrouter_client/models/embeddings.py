"""Typed text embedding requests, vectors, and optional usage accounting."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .chat import CostDetails
from .providers import ProviderPreferences


class EmbeddingsRequest(BaseModel):
    """Embed one text or a batch; additional API parameters are passed through."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., min_length=1)
    input: Union[str, List[str]]
    dimensions: Optional[int] = Field(None, ge=1)
    encoding_format: Optional[Literal["float", "base64"]] = None
    input_type: Optional[str] = None
    provider: Optional[Union[Dict[str, Any], ProviderPreferences]] = None
    user: Optional[str] = None
    session_id: Optional[str] = Field(None, max_length=256)
    trace: Optional[Dict[str, Any]] = None

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: Union[str, List[str]]) -> Union[str, List[str]]:
        """Reject empty text or batches before making a billable request."""
        if not value or (isinstance(value, list) and any(not text for text in value)):
            raise ValueError("input must contain at least one non-empty text")
        return value


class Embedding(BaseModel):
    """One vector (or base64 string); index identifies its original input."""

    model_config = ConfigDict(extra="allow")

    embedding: Union[List[float], str]
    index: int = Field(..., ge=0)
    object: Literal["embedding"] = "embedding"


class EmbeddingCostDetails(CostDetails):
    """BYOK provider charges, compatible with chat's cost breakdown."""

    model_config = ConfigDict(extra="allow")

    upstream_inference_prompt_cost: Optional[float] = None
    upstream_inference_completions_cost: Optional[float] = None


class EmbeddingUsage(BaseModel):
    """Token counts and cost in credits; absent accounting stays unknown."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: Optional[int] = Field(None, ge=0)
    total_tokens: Optional[int] = Field(None, ge=0)
    cost: Optional[float] = Field(None, ge=0)
    is_byok: Optional[bool] = None
    cost_details: Optional[EmbeddingCostDetails] = None


class EmbeddingsResponse(BaseModel):
    """Embedding results in server order; map them to inputs using index."""

    model_config = ConfigDict(extra="allow")

    data: List[Embedding]
    model: str
    object: Literal["list"] = "list"
    id: Optional[str] = None
    usage: Optional[EmbeddingUsage] = None
