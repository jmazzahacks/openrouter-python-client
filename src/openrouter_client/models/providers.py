"""Provider models for OpenRouter Client.

This module defines Pydantic models for provider preferences and settings.

Exported:
- ProviderPreferences: Provider routing preferences
- ProviderMaxPrice: Provider maximum price settings
"""

from typing import List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict


class ProviderMaxPrice(BaseModel):
    """
    Maximum price settings for provider routing.
    
    Attributes:
        prompt (Optional[float]): Maximum price per million prompt tokens.
        completion (Optional[float]): Maximum price per million completion tokens.
        request (Optional[float]): Maximum price per request.
        image (Optional[float]): Maximum price per image.
    """
    model_config = ConfigDict(strict=True)
    
    prompt: Optional[float] = Field(None, ge=0.0, description="Maximum price per million prompt tokens")
    completion: Optional[float] = Field(None, ge=0.0, description="Maximum price per million completion tokens")
    request: Optional[float] = Field(None, ge=0.0, description="Maximum price per request")
    image: Optional[float] = Field(None, ge=0.0, description="Maximum price per image")


class ProviderPreferences(BaseModel):
    """
    Provider routing preferences for requests.
    
    Attributes:
        order (Optional[List[str]]): List of provider names to try in order.
        allow_fallbacks (Optional[bool]): Whether to allow backup providers when the primary is unavailable.
        require_parameters (Optional[bool]): Only use providers that support all parameters in the request.
        data_collection (Optional[Literal["allow", "deny"]]): Control whether to use providers that may store data.
        only (Optional[List[str]]): List of provider names to allow for this request.
        ignore (Optional[List[str]]): List of provider names to skip for this request.
        quantizations (Optional[List[str]]): List of quantization levels to filter by.
        sort (Optional[Literal["price", "throughput", "latency"]]): Sort providers by attribute.
        max_price (Optional[ProviderMaxPrice]): The maximum pricing to pay for this request.
    """
    model_config = ConfigDict(strict=True)
    
    order: Optional[List[str]] = Field(None, description="List of provider names to try in order")
    allow_fallbacks: Optional[bool] = Field(True, description="Whether to allow backup providers when the primary is unavailable")
    require_parameters: Optional[bool] = Field(False, description="Only use providers that support all parameters in the request")
    data_collection: Optional[Literal["allow", "deny"]] = Field("allow", description="Control whether to use providers that may store data")
    only: Optional[List[str]] = Field(None, description="List of provider names to allow for this request")
    ignore: Optional[List[str]] = Field(None, description="List of provider names to skip for this request")
    quantizations: Optional[List[str]] = Field(None, description="List of quantization levels to filter by")
    sort: Optional[Literal["price", "throughput", "latency"]] = Field(None, description="Sort providers by attribute")
    max_price: Optional[ProviderMaxPrice] = Field(None, description="The maximum pricing to pay for this request")
