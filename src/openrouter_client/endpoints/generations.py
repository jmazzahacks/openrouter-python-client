"""
Generations endpoint implementation.

This module provides the endpoint handler for generation metadata API,
allowing retrieval of information about specific generation requests.

Exported:
- GenerationsEndpoint: Handler for generation endpoint
"""

from typing import Dict, Any

from ..auth import AuthManager
from ..http import HTTPManager
from .base import BaseEndpoint


class GenerationsEndpoint(BaseEndpoint):
    """
    Handler for the generation API endpoint.
    
    Provides method for retrieving metadata about a specific generation request.
    """
    
    def __init__(self, auth_manager: AuthManager, http_manager: HTTPManager):
        """
        Initialize the generation endpoint handler.
        
        Args:
            auth_manager (AuthManager): Authentication manager.
            http_manager (HTTPManager): HTTP communication manager.
        """
        # Call parent initializer with 'generation' as endpoint_path
        # Note: The actual API endpoint is singular 'generation', not plural
        super().__init__(auth_manager, http_manager, "generation")
        
        # Log initialization of generation endpoint
        self.logger.debug("Initialized generation endpoint handler")
    
    def get(self, generation_id: str) -> Dict[str, Any]:
        """
        Get metadata about a specific generation request.
        
        Args:
            generation_id (str): The ID of the generation to retrieve.
            
        Returns:
            Dict[str, Any]: Metadata about the generation request.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare query parameters with the generation ID
        params = {"id": generation_id}
        
        # Get authentication headers
        headers = self._get_headers()
        
        # Make GET request to generation endpoint with ID as query parameter
        response = self.http_manager.get(
            self._get_endpoint_url(),
            headers=headers,
            params=params
        )
        
        # Return parsed JSON response
        return response.json()