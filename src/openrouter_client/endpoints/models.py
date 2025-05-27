"""
Models endpoint implementation.

This module provides the endpoint handler for models API,
allowing listing and retrieval of model information and endpoints.

Exported:
- ModelsEndpoint: Handler for models endpoint operations including listing models,
  retrieving model details, and listing model endpoints
"""

from typing import List, Union

from ..auth import AuthManager
from ..http import HTTPManager
from .base import BaseEndpoint
from ..models.models import ModelData, ModelEndpointsResponse, ModelPricing, ModelsResponse


class ModelsEndpoint(BaseEndpoint):
    """
    Handler for the models API endpoint.
    
    Provides methods for listing and retrieving model information.
    """
    
    def __init__(self, auth_manager: AuthManager, http_manager: HTTPManager):
        """
        Initialize the models endpoint handler.
        
        Args:
            auth_manager (AuthManager): Authentication manager.
            http_manager (HTTPManager): HTTP communication manager.
        """
        # Call parent initializer with 'models' as endpoint_path
        super().__init__(auth_manager, http_manager, "models")
        
        # Log initialization of models endpoint
        self.logger.debug("Initialized models endpoint handler")
    
    def list(self, details: bool = False, **kwargs) -> Union[List[str], ModelsResponse]:
        """
        List available models.
        
        Args:
            details (bool): Whether to include full model details. Defaults to False.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            Union[List[str], ModelsResponse]: 
                If details=False: List of model IDs.
                If details=True: ModelsResponse with full model details including context lengths, pricing, etc.
            
        Raises:
            APIError: If the API request fails.
            
        Example response structure with details=True:
        ```json
        {
          "data": [
            {
              "id": "anthropic/claude-3-opus",
              "name": "Anthropic: Claude 3 Opus",
              "created": 1709596800,
              "description": "Anthropic's most capable model...", 
              "context_length": 200000,
              "max_completion_tokens": 4096,
              "quantization": "fp16",
              "pricing": {
                "prompt": "0.000015",
                "completion": "0.000075"
              }
            },
            // More models...
          ]
        }
        ```
        """
        # Prepare query parameters with details flag and kwargs
        params = {"details": str(details).lower(), **kwargs}
        
        # Get authentication headers
        headers = self._get_headers()
        
        # Make GET request to models endpoint
        response = self.http_manager.get(
            self._get_endpoint_url(),
            headers=headers,
            params=params
        )
        
        # Parse JSON response
        response_data = response.json()
        
        # If details is False and response contains 'data' list, extract and return just the model IDs
        if not details and isinstance(response_data, dict) and "data" in response_data:
            # Extract model IDs from the data list
            if isinstance(response_data["data"], list):
                return [model["id"] if isinstance(model, dict) and "id" in model else model 
                         for model in response_data["data"]]
        
        # Otherwise convert and return the full response data
        return ModelsResponse.model_validate(response_data)
    
    def get(self, model_id: str) -> ModelData:
        """
        Get information about a specific model.
        
        Args:
            model_id (str): The model identifier (e.g., "anthropic/claude-3-opus").
            
        Returns:
            ModelData: Model information including context length, pricing, etc.
            
        Raises:
            APIError: If the API request fails.
            
        Example response structure:
        ```json
        {
          "id": "anthropic/claude-3-opus",
          "name": "Anthropic: Claude 3 Opus",
          "created": 1709596800,
          "description": "Anthropic's most capable model...", 
          "context_length": 200000,
          "max_completion_tokens": 4096,
          "quantization": "fp16",
          "pricing": {
            "prompt": "0.000015",
            "completion": "0.000075"
          }
        }
        ```
        """
        # Get authentication headers
        headers = self._get_headers()
        
        # Make GET request to specific model endpoint using model_id
        response = self.http_manager.get(
            self._get_endpoint_url(model_id),
            headers=headers
        )
        
        # Parse and validate the response as a ModelData object
        return ModelData.model_validate(response.json())
        
    def get_context_length(self, model_id: str) -> int:
        """
        Get the context length for a specific model.
        
        Args:
            model_id (str): The model identifier (e.g., "anthropic/claude-3-opus").
            
        Returns:
            int: The context length in tokens for the model.
            
        Raises:
            APIError: If the API request fails.
            ValueError: If the model doesn't have a context length specified.
        """
        model_data = self.get(model_id)
        return model_data.context_length
        
    def get_model_pricing(self, model_id: str) -> ModelPricing:
        """
        Get the pricing information for a specific model.
        
        Args:
            model_id (str): The model identifier (e.g., "anthropic/claude-3-opus").
            
        Returns:
            ModelPricing: The pricing information for the model.
            
        Raises:
            APIError: If the API request fails.
        """
        model_data = self.get(model_id)
        return model_data.pricing
        
    def list_endpoints(self, author: str, slug: str) -> ModelEndpointsResponse:
        """
        List available endpoints for a specific model.
        
        Args:
            author (str): The model author/owner identifier.
            slug (str): The model slug identifier.
            
        Returns:
            ModelEndpointsResponse: Response containing a list of available endpoints for the model.
            
        Raises:
            APIError: If the API request fails.
            
        Example response structure:
        ```json
        {
          "object": "list",
          "data": [
            {
              "id": "chat",
              "name": "Chat Completion",
              "description": "Generate text responses based on conversation history",
              "url": "/api/v1/chat/completions",
              "method": "POST",
              "parameters": ["messages", "model", "temperature"]
            },
            // More endpoints...
          ]
        }
        ```
        """
        # Get authentication headers
        headers = self._get_headers()
        
        # Construct the URL for the model endpoints
        endpoint_url = f"{self._get_endpoint_url()}/{author}/{slug}/endpoints"
        
        # Make GET request to the endpoints endpoint
        response = self.http_manager.get(
            endpoint_url,
            headers=headers
        )
        
        # Parse and validate the response
        return ModelEndpointsResponse.model_validate(response.json())
