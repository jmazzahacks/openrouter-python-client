"""
Base implementation for API endpoint handlers.

This module provides the base class for all endpoint handlers, defining
common functionality for API interactions.

Exported:
- BaseEndpoint: Abstract base class for endpoint handlers
"""

import logging
from typing import Dict, TypeVar

from ..auth import AuthManager
from ..http import HTTPManager

T = TypeVar('T')


class BaseEndpoint:
    """
    Base class for all API endpoint handlers.
    
    Attributes:
        auth_manager (AuthManager): Authentication manager.
        http_manager (HTTPManager): HTTP communication manager.
        logger (logging.Logger): Endpoint-specific logger.
    """
    
    def __init__(self, 
                 auth_manager: AuthManager,
                 http_manager: HTTPManager,
                 endpoint_path: str):
        """
        Initialize a base endpoint handler.
        
        Args:
            auth_manager (AuthManager): Authentication manager.
            http_manager (HTTPManager): HTTP communication manager.
            endpoint_path (str): Relative API endpoint path.
        """
        # Store the auth_manager for authentication
        self.auth_manager = auth_manager
        
        # Store the http_manager for making HTTP requests
        self.http_manager = http_manager
        
        # Store the endpoint_path for forming full endpoint URLs
        self.endpoint_path = endpoint_path.lstrip('/')
        
        # Create a logger specific to this endpoint type
        self.logger = logging.getLogger(f"openrouter_client.endpoints.{self.__class__.__name__.lower()}")
    
    def _get_headers(self, require_provisioning: bool = False) -> Dict[str, str]:
        """
        Get headers for API requests.
        
        Args:
            require_provisioning (bool): Whether to use provisioning API key.
            
        Returns:
            Dict[str, str]: Headers for API requests.
        """
        # Get authentication headers from auth_manager
        auth_headers = self.auth_manager.get_auth_headers(require_provisioning=require_provisioning)
        
        # Add standard headers for API requests
        standard_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Return the combined headers
        headers = {**standard_headers, **auth_headers}
        self.logger.debug(f"Generated headers for API request")
        return headers
    
    def _get_endpoint_url(self, path: str = "") -> str:
        """
        Get the full endpoint URL.
        
        Args:
            path (str): Additional path to append to the endpoint path.
            
        Returns:
            str: Full endpoint URL.
        """
        # Combine endpoint_path with additional path
        # Handle path separators carefully to avoid double slashes
        path = path.lstrip('/') if path else ""
        
        if self.endpoint_path and path:
            # Both endpoint_path and additional path exist
            url = f"{self.endpoint_path}/{path}"
        elif self.endpoint_path:
            # Only endpoint_path exists
            url = self.endpoint_path
        else:
            # Only additional path exists
            url = path
            
        self.logger.debug(f"Generated endpoint URL: {url}")
        return url
