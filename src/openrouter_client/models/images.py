"""Images-related API models for OpenRouter Client.

This module defines Pydantic models for the Images API requests and responses.

Exported:
- ImagesRequest: Request for image generation
- ImagesResponse: Response from image generation
- ImageObject: Generated image object
- ImagesGenerationRequest: Request for image generation
"""

import base64
import re
from typing import Dict, List, Optional
import urllib3

from pydantic import BaseModel, Field, field_validator

from .providers import ProviderPreferences


class ImageObject(BaseModel):
    """
    Generated image object.
    
    Attributes:
        url (Optional[str]): URL of the generated image.
        b64_json (Optional[str]): Base64-encoded JSON representation of the generated image.
        revised_prompt (Optional[str]): Revised prompt used for image generation.
    """
    url: Optional[str] = Field(None, description="URL of the generated image")
    b64_json: Optional[str] = Field(None, description="Base64-encoded JSON representation of the generated image")
    revised_prompt: Optional[str] = Field(None, min_length=1, description="Revised prompt used for image generation")
    
    @field_validator('url')
    def validate_url(cls, value):
        if value is not None:
            try:
                urllib3.parse_url(value)
            except urllib3.exceptions.LocationParseError:
                raise ValueError("Invalid URL format")
        
        return value
    
    @field_validator('b64_json')
    def validate_b64_json(cls, value):
        if value is not None:
            try:
                base64.b64decode(value, validate=True)
            except ValueError:
                raise ValueError("Invalid base64-encoded JSON format")
        
        return value


class ImagesGenerationRequest(BaseModel):
    """
    Request for image generation.
    
    Attributes:
        prompt (str): Text prompt to generate images from.
        model (str): Model identifier to use for image generation.
        n (Optional[int]): Number of images to generate.
        size (Optional[str]): Size of the generated images.
        quality (Optional[str]): Quality of the generated images.
        style (Optional[str]): Style for the generated images.
        response_format (Optional[str]): Format of the response.
        user (Optional[str]): User identifier for tracking.
    """
    prompt: str = Field(..., min_length=1, description="Text prompt to generate images from")
    model: str = Field(..., min_length=1, description="Model identifier to use for image generation")
    n: Optional[int] = Field(1, ge=1, le=10, description="Number of images to generate")
    size: Optional[str] = Field("1024x1024", description="Size of the generated images")
    quality: Optional[str] = Field("standard", description="Quality of the generated images")
    style: Optional[str] = Field("vivid", description="Style for the generated images")
    response_format: Optional[str] = Field("url", description="Format of the response")
    user: Optional[str] = Field(None, description="User identifier for tracking")

    @field_validator('size')
    def validate_size(cls, value):
        if value is not None:
            if not re.match(r'^[0-9]+x[0-9]+$', value):
                raise ValueError("Invalid size format")
        return value

class ImagesRequest(ImagesGenerationRequest):
    """
    Extended request for image generation with OpenRouter-specific fields.
    
    Attributes:
        models (Optional[List[str]]): List of model IDs to use as fallbacks.
        provider (Optional[ProviderPreferences]): Provider routing preferences.
        extra_headers (Optional[Dict[str, str]]): Additional headers for the request.
        http_referer (Optional[str]): HTTP referer for the request.
        x_title (Optional[str]): Title for the request for rankings on openrouter.ai.
    """
    models: Optional[List[str]] = Field(None, min_length=1, description="List of model IDs to use as fallbacks")
    provider: Optional[ProviderPreferences] = Field(None, description="Provider routing preferences")
    extra_headers: Optional[Dict[str, str]] = Field(None, description="Additional headers for the request")
    http_referer: Optional[str] = Field(None, description="HTTP referer for the request")
    x_title: Optional[str] = Field(None, description="Title for the request for rankings on openrouter.ai")


class ImagesResponse(BaseModel):
    """
    Response from image generation.
    
    Attributes:
        created (int): Unix timestamp (in seconds) of when the images were created.
        data (List[ImageObject]): List of generated image objects.
        model (str): Model used for image generation.
    """
    created: int = Field(..., description="Unix timestamp (in seconds) of when the images were created")
    data: List[ImageObject] = Field(..., description="List of generated image objects")
    model: str = Field(..., min_length=1, description="Model used for image generation")