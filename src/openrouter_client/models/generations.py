"""
Generations API models for OpenRouter Client.

This module defines Pydantic models for the Generations API requests and responses,
which provide details about previous completions, usage statistics, and billing information.

Exported:
- Generation: Detailed information about a single generation
- GenerationList: Paginated list of generations
- GenerationListParams: Parameters for listing generations
- GenerationUsage: Token usage information for a generation
- GenerationStats: Aggregated statistics about generations
- GenerationStatsParams: Parameters for generation statistics
- ModelStats: Statistics about generations by model
- ModelStatsParams: Parameters for model statistics
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dateutil.parser import parse as parse_datetime

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class GenerationUsage(BaseModel):
    """
    Token usage information for a generation.
    
    Attributes:
        prompt_tokens (int): Number of tokens in the prompt.
        completion_tokens (int): Number of tokens in the completion.
        total_tokens (int): Total number of tokens used.
    """
    prompt_tokens: int = Field(..., ge=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., ge=0, description="Number of tokens in the completion")
    total_tokens: int = Field(..., ge=0, description="Total number of tokens used")


class GenerationCost(BaseModel):
    """
    Cost information for a generation.
    
    Attributes:
        prompt_tokens (float): Cost of prompt tokens in credits.
        completion_tokens (float): Cost of completion tokens in credits.
        total_tokens (float): Total cost in credits.
    """
    prompt_tokens: float = Field(..., ge=0.0, description="Cost of prompt tokens in credits")
    completion_tokens: float = Field(..., ge=0.0, description="Cost of completion tokens in credits")
    total_tokens: float = Field(..., ge=0.0, description="Total cost in credits")


class Generation(BaseModel):
    """
    Detailed information about a single generation.
    
    Attributes:
        id (str): Unique identifier for the generation.
        object (str): Object type, typically "generation".
        created (int): Unix timestamp (in seconds) of when the generation was created.
        model (str): Model used for the generation.
        usage (GenerationUsage): Token usage information.
        cost (GenerationCost): Cost information in credits.
        prompt (Optional[Dict[str, Any]]): Original prompt sent to the model.
        response (Optional[Dict[str, Any]]): Response received from the model.
        organization_id (Optional[str]): Organization identifier.
        user_id (Optional[str]): User identifier.
        http_referer (Optional[str]): HTTP referer for the request.
        title (Optional[str]): Title of the application that made the request.
        route (Optional[str]): Routing strategy used.
        routing_duration (Optional[int]): Time taken for routing in milliseconds.
        status (Optional[str]): Status of the generation (e.g., "completed", "failed").
        error (Optional[Dict[str, Any]]): Error information if generation failed.
    """
    id: str = Field(..., description="Unique identifier for the generation")
    object: str = Field("generation", description="Object type, typically 'generation'")
    created: int = Field(..., description="Unix timestamp (in seconds) of when the generation was created")
    model: str = Field(..., description="Model used for the generation")
    usage: GenerationUsage = Field(..., description="Token usage information")
    cost: GenerationCost = Field(..., description="Cost information in credits")
    prompt: Optional[Dict[str, Any]] = Field(None, description="Original prompt sent to the model")
    response: Optional[Dict[str, Any]] = Field(None, description="Response received from the model")
    organization_id: Optional[str] = Field(None, description="Organization identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    http_referer: Optional[str] = Field(None, description="HTTP referer for the request")
    title: Optional[str] = Field(None, description="Title of the application that made the request")
    route: Optional[str] = Field(None, description="Routing strategy used")
    routing_duration: Optional[int] = Field(None, description="Time taken for routing in milliseconds")
    status: Optional[str] = Field(None, description="Status of the generation (e.g., 'completed', 'failed')")
    error: Optional[Dict[str, Any]] = Field(None, description="Error information if generation failed")


class GenerationListMeta(BaseModel):
    """
    Metadata for generation listing results.
    
    Attributes:
        limit (int): Maximum number of items returned.
        offset (int): Offset used for pagination.
        total (int): Total number of items available.
    """
    limit: int = Field(..., ge=0, description="Maximum number of items returned")
    offset: int = Field(..., ge=0, description="Offset used for pagination")
    total: int = Field(..., ge=0, description="Total number of items available")

    @model_validator(mode='after')
    def validate_pagination_values(self):
        if self.offset > self.total:
            raise ValueError(f"offset ({self.offset}) must be <= total ({self.total})")
        return self

class GenerationList(BaseModel):
    """
    Paginated list of generations.
    
    Attributes:
        object (str): Object type, typically "list".
        data (List[Generation]): List of generation objects.
        meta (GenerationListMeta): Pagination metadata.
    """
    object: str = Field("list", description="Object type, typically 'list'")
    data: List[Generation] = Field(..., description="List of generation objects")
    meta: GenerationListMeta = Field(..., description="Pagination metadata")

    @field_validator('meta')
    def validate_meta_total(cls, meta_value, info: ValidationInfo):
        data_value = info.data.get('data', [])
        if meta_value.total != len(data_value):
            raise ValueError(f"meta.total ({meta_value.total}) must match the length of data ({len(data_value)})")
        return meta_value

class GenerationListParams(BaseModel):
    """
    Parameters for listing generations.
    
    Attributes:
        limit (Optional[int]): Maximum number of items to return.
        offset (Optional[int]): Number of items to skip for pagination.
        start_date (Optional[Union[str, datetime]]): Start date for filtering.
        end_date (Optional[Union[str, datetime]]): End date for filtering.
        model (Optional[str]): Filter by specific model.
    """
    limit: Optional[int] = Field(None, ge=0, description="Maximum number of items to return")
    offset: Optional[int] = Field(None, ge=0, description="Number of items to skip for pagination")
    start_date: Optional[Union[str, datetime]] = Field(None, description="Start date for filtering")
    end_date: Optional[Union[str, datetime]] = Field(None, description="End date for filtering")
    model: Optional[str] = Field(None, description="Filter by specific model")

    @model_validator(mode='after')
    def validate_date_range(self):
        if self.start_date and self.end_date:
            if self.end_date < self.start_date:
                raise ValueError(f"end_date ({self.end_date}) must be >= start_date ({self.start_date})")
        return self
    
    @model_validator(mode='after')
    def validate_limit_offset(self):
        if self.limit and self.offset:
            if self.limit < self.offset:
                raise ValueError(f"limit ({self.limit}) must be >= offset ({self.offset})")
        return self
    
    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, value):
        if isinstance(value, str):
            try:
                # Attempt to parse the string into a datetime
                datetime.fromisoformat(value)
            except ValueError:
                raise ValueError(f"Invalid date format: {value}")
        return value

class StatsPoint(BaseModel):
    """
    Single data point in generation statistics.
    
    Attributes:
        date (str): Date string for this data point.
        count (int): Number of generations in this period.
        tokens (int): Total tokens used in this period.
        cost (float): Total cost in credits for this period.
    """
    date: str = Field(..., description="Date string for this data point")
    count: int = Field(..., ge=0, description="Number of generations in this period")
    tokens: int = Field(..., ge=0, description="Total tokens used in this period")
    cost: float = Field(..., ge=0.0, description="Total cost in credits for this period")

    @field_validator('date', mode="before")
    def validate_date_format(cls, value):
        try:
            # Attempt to parse the string into a datetime
            value = parse_datetime(value).strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {value}")
        return value

class GenerationStats(BaseModel):
    """
    Aggregated statistics about generations.
    
    Attributes:
        period (str): Time period granularity ("day", "week", "month", "year").
        data (List[StatsPoint]): List of statistics points.
        totals (Optional[Dict[str, Any]]): Aggregated totals across all periods.
    """
    period: str = Field(..., description="Time period granularity (\"day\", \"week\", \"month\", \"year\")")
    data: List[StatsPoint] = Field(..., description="List of statistics points")
    totals: Optional[Dict[str, Any]] = Field(None, description="Aggregated totals across all periods")

    @field_validator('period')
    def validate_period(cls, value):
        if not isinstance(value, str):
            raise ValueError("period must be a string")
        if value not in ["day", "week", "month", "year"]:
            raise ValueError("period must be one of 'day', 'week', 'month', 'year'")
        return value

class GenerationStatsParams(BaseModel):
    """
    Parameters for generation statistics.
    
    Attributes:
        period (Optional[str]): Time period granularity ("day", "week", "month", "year").
        start_date (Optional[Union[str, datetime]]): Start date for filtering.
        end_date (Optional[Union[str, datetime]]): End date for filtering.
    """
    period: Optional[str] = Field("month", description="Time period granularity (\"day\", \"week\", \"month\", \"year\")")
    start_date: Optional[Union[str, datetime]] = Field(None, description="Start date for filtering")
    end_date: Optional[Union[str, datetime]] = Field(None, description="End date for filtering")

    @model_validator(mode='after')
    def validate_dates(self):
        if self.start_date and self.end_date:
            if not isinstance(self.start_date, datetime):
                start_date = datetime.fromisoformat(self.start_date)
            else:
                start_date = self.start_date
                
            if not isinstance(self.end_date, datetime):
                end_date = datetime.fromisoformat(self.end_date)
            else:
                end_date = self.end_date
            if end_date < start_date:
                raise ValueError(f"end_date ({self.end_date}) must be >= start_date ({self.start_date})")
        return self
    
    @field_validator('start_date', 'end_date')
    def validate_date_format(cls, value):
        if isinstance(value, str):
            try:
                # Attempt to parse the string into a datetime
                datetime.fromisoformat(value)
            except ValueError:
                raise ValueError(f"Invalid date format: {value}")
        return value
    
    @field_validator("period")
    def validate_period(cls, value):
        if not isinstance(value, str):
            raise ValueError("period must be a string")
        if value not in ["day", "week", "month", "year"]:
            raise ValueError("period must be one of 'day', 'week', 'month', 'year'")
        return value

class ModelStatsPoint(BaseModel):
    """
    Statistics for a specific model.
    
    Attributes:
        model (str): Model identifier.
        count (int): Number of generations with this model.
        tokens (int): Total tokens used with this model.
        cost (float): Total cost in credits for this model.
    """
    model: str = Field(..., min_length=1, description="Model identifier")
    count: int = Field(..., ge=0, description="Number of generations with this model")
    tokens: int = Field(..., ge=0, description="Total tokens used with this model")
    cost: float = Field(..., ge=0.0, description="Total cost in credits for this model")


class ModelStats(BaseModel):
    """
    Statistics about generations by model.
    
    Attributes:
        data (List[ModelStatsPoint]): List of model statistics points.
        totals (Optional[Dict[str, Any]]): Aggregated totals across all models.
    """
    data: List[ModelStatsPoint] = Field(..., description="List of model statistics points")
    totals: Optional[Dict[str, Any]] = Field(None, description="Aggregated totals across all models")


class ModelStatsParams(BaseModel):
    """
    Parameters for model statistics.
    
    Attributes:
        start_date (Optional[Union[str, datetime]]): Start date for filtering.
        end_date (Optional[Union[str, datetime]]): End date for filtering.
    """
    start_date: Optional[Union[str, datetime]] = Field(None, description="Start date for filtering")
    end_date: Optional[Union[str, datetime]] = Field(None, description="End date for filtering")
    
    @field_validator('start_date', 'end_date')
    def validate_date_format(cls, value):
        if isinstance(value, str):
            try:
                # Attempt to parse the string into a datetime
                datetime.fromisoformat(value)
            except ValueError:
                raise ValueError(f"Invalid date format: {value}")
        return value
    
    @model_validator(mode='after')
    def validate_dates(self):
        if self.start_date and self.end_date:
            if not isinstance(self.start_date, datetime):
                start_date = datetime.fromisoformat(self.start_date)
            else:
                start_date = self.start_date
            
            if not isinstance(self.end_date, datetime):
                end_date = datetime.fromisoformat(self.end_date)
            else:
                end_date = self.end_date
                
            if end_date < start_date:
                raise ValueError(f"end_date ({self.end_date}) must be >= start_date ({self.start_date})")
        
        return self
