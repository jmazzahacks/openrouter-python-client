"""
Plugins endpoint implementation.

This module provides the endpoint handler for plugin management API,
supporting registration, listing, and usage of plugins.

Exported:
- PluginsEndpoint: Handler for plugins endpoint
"""

from typing import Dict, List, Optional, Any, cast

from ..auth import AuthManager
from ..http import HTTPManager
from .base import BaseEndpoint


class PluginsEndpoint(BaseEndpoint):
    """
    Handler for the plugins API endpoint.
    
    Provides methods for managing and using plugins.
    """
    
    def __init__(self, auth_manager: AuthManager, http_manager: HTTPManager):
        """
        Initialize the plugins endpoint handler.
        
        Args:
            auth_manager (AuthManager): Authentication manager.
            http_manager (HTTPManager): HTTP communication manager.
        """
        super().__init__(auth_manager, http_manager, 'plugins')
        self.logger.info("Initialized plugins endpoint handler")
    
    def list(self) -> List[Dict[str, Any]]:
        """
        List available plugins.
        
        Returns:
            List[Dict[str, Any]]: List of available plugins.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers
        headers = self._get_headers()
        
        # Make GET request to plugins endpoint
        response = self.http_manager.get(
            endpoint=self._get_endpoint_url(),
            headers=headers
        )
        
        # Return parsed JSON response
        return cast(List[Dict[str, Any]], response.json())
    
    def get(self, plugin_id: str) -> Dict[str, Any]:
        """
        Get details about a specific plugin.
        
        Args:
            plugin_id (str): Plugin ID.
            
        Returns:
            Dict[str, Any]: Plugin details.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers
        headers = self._get_headers()
        
        # Make GET request to specific plugin endpoint using plugin_id
        response = self.http_manager.get(
            endpoint=self._get_endpoint_url(plugin_id),
            headers=headers
        )
        
        # Return parsed JSON response
        return cast(Dict[str, Any], response.json())
    
    def register(self, 
                 manifest_url: str,
                 auth: Optional[Dict[str, Any]] = None,
                 **kwargs: Any) -> Dict[str, Any]:
        """
        Register a new plugin.
        
        Args:
            manifest_url (str): URL to the plugin manifest.
            auth (Optional[Dict[str, Any]]): Authentication credentials for the plugin.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            Dict[str, Any]: Registered plugin information.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare request data from function arguments
        data: Dict[str, Any] = {"manifest_url": manifest_url}
        
        # Add auth if provided
        if auth is not None:
            data["auth"] = auth
            
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in data:
                data[key] = value
        
        # Get authentication headers
        headers = self._get_headers()
        
        # Make POST request to plugins endpoint
        response = self.http_manager.post(
            endpoint=self._get_endpoint_url(),
            headers=headers,
            json=data
        )
        
        # Return parsed JSON response
        return cast(Dict[str, Any], response.json())
    
    def unregister(self, plugin_id: str) -> Dict[str, Any]:
        """
        Unregister a plugin.
        
        Args:
            plugin_id (str): Plugin ID to unregister.
            
        Returns:
            Dict[str, Any]: Unregistration confirmation.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers
        headers = self._get_headers()
        
        # Make DELETE request to specific plugin endpoint using plugin_id
        response = self.http_manager.delete(
            endpoint=self._get_endpoint_url(plugin_id),
            headers=headers
        )
        
        # Return parsed JSON response
        return cast(Dict[str, Any], response.json())
    
    def invoke(self, 
               plugin_id: str,
               action: str,
               parameters: Optional[Dict[str, Any]] = None,
               **kwargs: Any) -> Dict[str, Any]:
        """
        Invoke a plugin action.
        
        Args:
            plugin_id (str): Plugin ID.
            action (str): Action to invoke.
            parameters (Optional[Dict[str, Any]]): Parameters for the action.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            Dict[str, Any]: Result of the plugin action.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare request data with action and parameters
        data: Dict[str, Any] = {"action": action}
        
        # Add parameters if provided
        if parameters is not None:
            data["parameters"] = parameters
            
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in data:
                data[key] = value
        
        # Get authentication headers
        headers = self._get_headers()
        
        # Make POST request to plugin invoke endpoint using plugin_id
        response = self.http_manager.post(
            endpoint=self._get_endpoint_url(f"{plugin_id}/invoke"),
            headers=headers,
            json=data
        )
        
        # Return parsed JSON response
        return cast(Dict[str, Any], response.json())
