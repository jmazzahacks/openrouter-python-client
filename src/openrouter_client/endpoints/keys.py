"""
API keys endpoint implementation.

This module provides the endpoint handler for API key management,
supporting creation, listing, and revocation of API keys.

Exported:
- KeysEndpoint: Handler for API keys endpoint
"""

from typing import Dict, Optional, Any

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
    
    def list(self, offset: Optional[int] = None, include_disabled: Optional[bool] = None) -> Dict[str, Any]:
        """
        List all API keys.
        
        Args:
            offset (Optional[int]): Offset for pagination.
            include_disabled (Optional[bool]): Whether to include disabled keys.
        
        Returns:
            Dict[str, Any]: Response containing list of API keys in 'data' array.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Build query parameters
        params = {}
        if offset is not None:
            params["offset"] = offset
        if include_disabled is not None:
            params["includeDisabled"] = str(include_disabled).lower()
        
        # Make GET request to keys endpoint
        response = self.http_manager.get(
            self._get_endpoint_url(),
            headers=headers,
            params=params if params else None
        )
        
        # Return parsed JSON response
        return response.json()
    
    def get(self, key_hash: str) -> Dict[str, Any]:
        """
        Get details about a specific API key.
        
        Args:
            key_hash (str): The hash of the API key.
            
        Returns:
            Dict[str, Any]: API key details.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Make GET request to specific key endpoint
        response = self.http_manager.get(
            self._get_endpoint_url(key_hash),
            headers=headers
        )
        
        # Return parsed JSON response
        return response.json()
    
    def create(self, 
               name: str,
               label: Optional[str] = None,
               limit: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a new API key.
        
        Args:
            name (str): Name for the API key (required).
            label (Optional[str]): Label for the API key.
            limit (Optional[float]): Credit limit for the key.
            
        Returns:
            Dict[str, Any]: Created API key information including the key string.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare request data from function arguments
        data = {"name": name}
        
        if label is not None:
            data["label"] = label
            
        if limit is not None:
            data["limit"] = limit
                
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
    
    def update(self, key_hash: str, name: Optional[str] = None, disabled: Optional[bool] = None) -> Dict[str, Any]:
        """
        Update an existing API key.
        
        Args:
            key_hash (str): The hash of the API key to update.
            name (Optional[str]): New name for the key.
            disabled (Optional[bool]): Enable/disable status.
            
        Returns:
            Dict[str, Any]: Updated API key details.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare request data
        data = {}
        if name is not None:
            data["name"] = name
        if disabled is not None:
            data["disabled"] = disabled
            
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Make PATCH request to specific key endpoint
        response = self.http_manager.patch(
            self._get_endpoint_url(key_hash),
            headers=headers,
            json=data
        )
        
        # Return parsed JSON response
        return response.json()
    
    def delete(self, key_hash: str) -> Dict[str, Any]:
        """
        Delete an API key.
        
        Args:
            key_hash (str): The hash of the API key to delete.
            
        Returns:
            Dict[str, Any]: Deletion confirmation.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Make DELETE request to specific key endpoint
        response = self.http_manager.delete(
            self._get_endpoint_url(key_hash),
            headers=headers
        )
        
        # Return parsed JSON response
        return response.json()
    
    def get_current(self) -> Dict[str, Any]:
        """
        Get information about the currently authenticated API key.
        
        Returns:
            Dict[str, Any]: Current API key information including usage and limits.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers (uses current API key, not provisioning key)
        headers = self._get_headers(require_provisioning=False)
        
        # Make GET request to auth/key endpoint
        response = self.http_manager.get(
            "auth/key",
            headers=headers
        )
        
        # Return parsed JSON response
        return response.json()