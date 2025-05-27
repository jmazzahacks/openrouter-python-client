"""
Credits endpoint implementation.

This module provides the endpoint handler for credits management API,
allowing checking balance.

Exported:
- CreditsEndpoint: Handler for credits endpoint
"""

from typing import Dict, Any

from ..auth import AuthManager
from ..http import HTTPManager
from .base import BaseEndpoint


class CreditsEndpoint(BaseEndpoint):
    """
    Handler for the credits API endpoint.
    
    Provides method for retrieving credit balance.
    """
    
    def __init__(self, auth_manager: AuthManager, http_manager: HTTPManager):
        """
        Initialize the credits endpoint handler.
        
        Args:
            auth_manager (AuthManager): Authentication manager.
            http_manager (HTTPManager): HTTP communication manager.
        """
        # Call parent initializer with 'credits' as endpoint_path
        super().__init__(auth_manager, http_manager, "credits")
        
        # Log initialization of credits endpoint
        self.logger.debug("Initialized credits endpoint handler")
    
    def get(self) -> Dict[str, Any]:
        """
        Get current credit balance and information.
        
        Returns:
            Dict[str, Any]: Credit information including balance.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Make GET request to credits endpoint
        response = self.http_manager.get(
            self._get_endpoint_url(),
            headers=headers
        )
        
        # Return parsed JSON response
        return response.json()