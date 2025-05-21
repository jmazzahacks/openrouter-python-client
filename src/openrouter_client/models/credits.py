"""Credits-related API models for OpenRouter Client.

This module defines Pydantic models for the Credits API responses.

Exported:
- CreditsResponse: Response from the Credits API
"""

from typing import Optional

from pydantic import BaseModel, Field, StrictFloat


class CreditsResponse(BaseModel):
    """
    Response from the Credits API.
    
    Attributes:
        credits (float): Total credits available to the user.
        used (float): Total credits used by the user.
        purchase_credits (Optional[float]): Credits purchased by the user.
        gifted_credits (Optional[float]): Credits gifted to the user.
        remaining_free_credits (Optional[float]): Remaining free credits.
    """
    credits: StrictFloat = Field(..., description="Total credits available to the user")
    used: StrictFloat = Field(..., description="Total credits used by the user")
    purchase_credits: Optional[StrictFloat] = Field(None, description="Credits purchased by the user")
    gifted_credits: Optional[StrictFloat] = Field(None, description="Credits gifted to the user")
    remaining_free_credits: Optional[StrictFloat] = Field(None, description="Remaining free credits")