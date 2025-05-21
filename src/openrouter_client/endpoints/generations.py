"""
Generations endpoint implementation.

This module provides the endpoint handler for generation statistics API,
giving insight into usage patterns and history.

Exported:
- GenerationsEndpoint: Handler for generations endpoint
"""

import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

from ..auth import AuthManager
from ..http import HTTPManager
from .base import BaseEndpoint
from ..models.generations import (
    Generation,
    GenerationList,
    GenerationStats,
    ModelStats
)


class GenerationsEndpoint(BaseEndpoint):
    """
    Handler for the generations API endpoint.
    
    Provides methods for retrieving information about generation history and statistics.
    """
    
    def __init__(self, auth_manager: AuthManager, http_manager: HTTPManager):
        """
        Initialize the generations endpoint handler.
        
        Args:
            auth_manager (AuthManager): Authentication manager.
            http_manager (HTTPManager): HTTP communication manager.
        """
        # Call parent initializer with 'generations' as endpoint_path
        super().__init__(auth_manager, http_manager, "generations")
        
        # Log initialization of generations endpoint
        self.logger.debug("Initialized generations endpoint handler")
    
    def list(self,
             limit: Optional[int] = None,
             offset: Optional[int] = None,
             start_date: Optional[Union[str, datetime]] = None,
             end_date: Optional[Union[str, datetime]] = None,
             model: Optional[str] = None,
             **kwargs) -> GenerationList:
        """
        List generation history.
        
        Args:
            limit (Optional[int]): Maximum number of items to return.
            offset (Optional[int]): Number of items to skip for pagination.
            start_date (Optional[Union[str, datetime]]): Start date for filtering.
            end_date (Optional[Union[str, datetime]]): End date for filtering.
            model (Optional[str]): Filter by specific model.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            GenerationList: Paginated list of generations with metadata.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare query parameters from function arguments
        params = {}
        if limit is not None:
            params['limit'] = limit
        if offset is not None:
            params['offset'] = offset
        if model is not None:
            params['model'] = model
            
        # If start_date or end_date is datetime object, convert to ISO format string
        if start_date is not None:
            if isinstance(start_date, datetime):
                params['start_date'] = start_date.isoformat()
            else:
                params['start_date'] = start_date
                
        if end_date is not None:
            if isinstance(end_date, datetime):
                params['end_date'] = end_date.isoformat()
            else:
                params['end_date'] = end_date
                
        # Add any additional kwargs to params
        params.update(kwargs)
        
        # Get authentication headers
        headers = self._get_headers()
        
        # Make GET request to generations endpoint
        response = self.http_manager.get(
            self._get_endpoint_url(),
            headers=headers,
            params=params
        )
        
        # Parse and return the response
        response_data = response.json()
        return GenerationList.model_validate(response_data)
    
    def get(self, generation_id: str) -> Generation:
        """
        Get details about a specific generation.
        
        Args:
            generation_id (str): Generation ID.
            
        Returns:
            Generation: Detailed information about the generation.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Make GET request to specific generation endpoint using generation_id
        response = self.http_manager.get(
            self._get_endpoint_url(generation_id),
            headers=headers
        )
        
        # Parse and return the response
        response_data = response.json()
        return Generation.model_validate(response_data)
    
    def stats(self,
               period: str = "month",
               start_date: Optional[Union[str, datetime]] = None,
               end_date: Optional[Union[str, datetime]] = None,
               **kwargs) -> GenerationStats:
        """
        Get generation statistics for a time period.
        
        Args:
            period (str): Time period granularity ("day", "week", "month", "year").
            start_date (Optional[Union[str, datetime]]): Start date for filtering.
            end_date (Optional[Union[str, datetime]]): End date for filtering.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            GenerationStats: Aggregated statistics about generations.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare query parameters from function arguments
        params = {'period': period}
        
        # If start_date or end_date is datetime object, convert to ISO format string
        if start_date is not None:
            if isinstance(start_date, datetime):
                params['start_date'] = start_date.isoformat()
            else:
                params['start_date'] = start_date
                
        if end_date is not None:
            if isinstance(end_date, datetime):
                params['end_date'] = end_date.isoformat()
            else:
                params['end_date'] = end_date
                
        # Add any additional kwargs to params
        params.update(kwargs)
        
        # Get authentication headers
        headers = self._get_headers()
        
        # Make GET request to generations/stats endpoint
        response = self.http_manager.get(
            self._get_endpoint_url('stats'),
            headers=headers,
            params=params
        )
        
        # Parse and return the response
        response_data = response.json()
        return GenerationStats.model_validate(response_data)
    
    def models(self, 
               start_date: Optional[Union[str, datetime]] = None,
               end_date: Optional[Union[str, datetime]] = None,
               **kwargs) -> ModelStats:
        """
        Get model usage statistics.
        
        Args:
            start_date (Optional[Union[str, datetime]]): Start date for filtering.
            end_date (Optional[Union[str, datetime]]): End date for filtering.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            ModelStats: Statistics about generations by model.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare query parameters from function arguments
        params = {}
        
        # If start_date or end_date is datetime object, convert to ISO format string
        if start_date is not None:
            if isinstance(start_date, datetime):
                params['start_date'] = start_date.isoformat()
            else:
                params['start_date'] = start_date
                
        if end_date is not None:
            if isinstance(end_date, datetime):
                params['end_date'] = end_date.isoformat()
            else:
                params['end_date'] = end_date
                
        # Add any additional kwargs to params
        params.update(kwargs)
        
        # Get authentication headers
        headers = self._get_headers()
        
        # Make GET request to generations/models endpoint
        response = self.http_manager.get(
            self._get_endpoint_url('models'),
            headers=headers,
            params=params
        )
        
        # Parse and return the response
        response_data = response.json()
        return ModelStats.model_validate(response_data)
