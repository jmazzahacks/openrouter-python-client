"""
Web search endpoint implementation.

This module provides the endpoint handler for web search API,
enabling contextual information retrieval from the internet.

Exported:
- WebEndpoint: Handler for web search endpoint
"""

from typing import Dict, Optional, Any, cast

from ..auth import AuthManager
from ..http import HTTPManager
from .base import BaseEndpoint


class WebEndpoint(BaseEndpoint):
    """
    Handler for the web search API endpoint.
    
    Provides methods for searching and retrieving content from the web.
    """
    
    def __init__(self, auth_manager: AuthManager, http_manager: HTTPManager):
        """
        Initialize the web endpoint handler.
        
        Args:
            auth_manager (AuthManager): Authentication manager.
            http_manager (HTTPManager): HTTP communication manager.
        """
        super().__init__(auth_manager, http_manager, 'web')
        self.logger.info("Initialized web endpoint handler")
    
    def search(self, 
               query: str,
               max_results: Optional[int] = None,
               recent: Optional[bool] = None,
               **kwargs: Any) -> Dict[str, Any]:
        """
        Search the web for information.
        
        Args:
            query (str): Search query.
            max_results (Optional[int]): Maximum number of results to return.
            recent (Optional[bool]): Whether to prioritize recent results.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            Dict[str, Any]: Search results.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare request data with query and parameters
        data: Dict[str, Any] = {"query": query}
        
        # Add optional parameters
        if max_results is not None:
            data["max_results"] = max_results
        if recent is not None:
            data["recent"] = recent
            
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in data:
                data[key] = value
        
        # Get authentication headers
        headers = self._get_headers()
        
        # Make POST request to web/search endpoint
        response = self.http_manager.post(
            endpoint=self._get_endpoint_url('search'),
            headers=headers,
            json=data
        )
        
        # Return parsed JSON response
        return cast(Dict[str, Any], response.json())
    
    def content(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Retrieve and extract content from a webpage.
        
        Args:
            url (str): URL of the webpage to retrieve.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            Dict[str, Any]: Extracted content from the webpage.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare request data with URL
        data: Dict[str, Any] = {"url": url}
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in data:
                data[key] = value
        
        # Get authentication headers
        headers = self._get_headers()
        
        # Make POST request to web/content endpoint
        response = self.http_manager.post(
            endpoint=self._get_endpoint_url('content'),
            headers=headers,
            json=data
        )
        
        # Return parsed JSON response
        return cast(Dict[str, Any], response.json())
    
    def summarize(self, 
                  url: str,
                  length: Optional[str] = None,
                  format: Optional[str] = None,
                  **kwargs: Any) -> Dict[str, Any]:
        """
        Generate a summary of a webpage.
        
        Args:
            url (str): URL of the webpage to summarize.
            length (Optional[str]): Desired length of summary ("short", "medium", "long").
            format (Optional[str]): Format of the summary ("plain", "bullets", "structured").
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            Dict[str, Any]: Webpage summary.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare request data with URL and parameters
        data: Dict[str, Any] = {"url": url}
        
        # Add optional parameters
        if length is not None:
            data["length"] = length
        if format is not None:
            data["format"] = format
            
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in data:
                data[key] = value
        
        # Get authentication headers
        headers = self._get_headers()
        
        # Make POST request to web/summarize endpoint
        response = self.http_manager.post(
            endpoint=self._get_endpoint_url('summarize'),
            headers=headers,
            json=data
        )
        
        # Return parsed JSON response
        return cast(Dict[str, Any], response.json())
