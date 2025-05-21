"""
API keys endpoint implementation.

This module provides the endpoint handler for API key management,
supporting creation, listing, and revocation of API keys.

Exported:
- KeysEndpoint: Handler for API keys endpoint
"""

import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from ..auth import AuthManager
from ..http import HTTPManager
from .base import BaseEndpoint


class KeysEndpoint(BaseEndpoint):
    """
    Handler for the API keys endpoint.
    
    Provides methods for managing API keys.
    """
    
    def __init__(self, auth_manager: AuthManager, http_manager: HTTPManager):
        """
        Initialize the keys endpoint handler.
        
        Args:
            auth_manager (AuthManager): Authentication manager.
            http_manager (HTTPManager): HTTP communication manager.
        """
        # Call parent initializer with 'keys' as endpoint_path
        super().__init__(auth_manager, http_manager, "keys")
        
        # Log initialization of keys endpoint
        self.logger.debug("Initialized keys endpoint handler")
    
    def list(self) -> List[Dict[str, Any]]:
        """
        List all API keys.
        
        Returns:
            List[Dict[str, Any]]: List of API keys with metadata.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Make GET request to keys endpoint
        response = self.http_manager.get(
            self._get_endpoint_url(),
            headers=headers
        )
        
        # Return parsed JSON response
        return response.json()
    
    def create(self, 
               name: Optional[str] = None,
               expiry: Optional[Union[str, datetime, int]] = None,
               permissions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a new API key.
        
        Args:
            name (Optional[str]): Friendly name for the API key.
            expiry (Optional[Union[str, datetime, int]]): Expiration date, timestamp, or days.
            permissions (Optional[List[str]]): Specific permissions for the key.
            
        Returns:
            Dict[str, Any]: Created API key information.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare request data from function arguments
        data = {}
        
        if name is not None:
            data["name"] = name
            
        # Handle expiry parameter based on its type
        if expiry is not None:
            if isinstance(expiry, datetime):
                # If expiry is datetime object, convert to ISO format string
                data["expiry"] = expiry.isoformat()
            elif isinstance(expiry, int):
                # If expiry is integer, assume it's days from now
                # The API expects an ISO date or days as integer
                data["expiry"] = expiry  # Send the number of days directly
            else:
                # Otherwise, pass the expiry as-is (assumed to be properly formatted string)
                data["expiry"] = expiry
                
        if permissions is not None:
            data["permissions"] = permissions
                
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Make POST request to keys endpoint
        response = self.http_manager.post(
            self._get_endpoint_url(),
            headers=headers,
            json=data
        )
        
        # Get the response with the new API key
        result = response.json()
        
        # Log warning that key will only be shown once and should be saved
        if "key" in result:
            self.logger.warning("API key will only be shown once. Make sure to save it securely.")
            
        return result
    
    def revoke(self, key_id: str) -> Dict[str, Any]:
        """
        Revoke an API key.
        
        Args:
            key_id (str): ID of the API key to revoke.
            
        Returns:
            Dict[str, Any]: Revocation confirmation.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Make DELETE request to specific key endpoint using key_id
        response = self.http_manager.delete(
            self._get_endpoint_url(key_id),
            headers=headers
        )
        
        # Return parsed JSON response
        return response.json()
    
    def rotate(self, key_id: str) -> Dict[str, Any]:
        """
        Rotate an API key (revoke old and create new with same permissions).
        
        Args:
            key_id (str): ID of the API key to rotate.
            
        Returns:
            Dict[str, Any]: New API key information.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Make POST request to specific key rotation endpoint using key_id
        response = self.http_manager.post(
            self._get_endpoint_url(f"{key_id}/rotate"),
            headers=headers
        )
        
        # Get the response with the new API key
        result = response.json()
        
        # Log warning that key will only be shown once and should be saved
        if "key" in result:
            self.logger.warning("New API key will only be shown once. Make sure to save it securely.")
            
        return result
