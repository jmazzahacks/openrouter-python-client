"""
Main OpenRouter client implementation.

This module defines the public OpenRouterClient class, which serves as the main
entry point for interacting with all OpenRouter API endpoints and services.

Exported:
- OpenRouterClient: Main client class for OpenRouter API interaction
"""

import logging
from typing import Dict, Optional, Any

from .auth import AuthManager, SecretsManager
from .http import HTTPManager
from .logging import configure_logging
from .endpoints.completions import CompletionsEndpoint
from .endpoints.chat import ChatEndpoint
from .endpoints.models import ModelsEndpoint
from .endpoints.images import ImagesEndpoint
from .endpoints.generations import GenerationsEndpoint
from .endpoints.credits import CreditsEndpoint
from .endpoints.keys import KeysEndpoint
from .endpoints.plugins import PluginsEndpoint
from .endpoints.web import WebEndpoint


class OpenRouterClient:
    """
    Main client for interacting with the OpenRouter API.
    
    Attributes:
        auth_manager (AuthManager): Authentication and API key manager.
        http_manager (HTTPManager): HTTP communication manager with rate limiting.
        secrets_manager (SecretsManager): Secrets manager for API keys.
        completions (CompletionsEndpoint): Text completions endpoint handler.
        chat (ChatEndpoint): Chat completions endpoint handler.
        models (ModelsEndpoint): Model information endpoint handler.
        images (ImagesEndpoint): Image generation endpoint handler.
        generations (GenerationsEndpoint): Generation statistics endpoint handler.
        credits (CreditsEndpoint): Credits management endpoint handler.
        keys (KeysEndpoint): API key management endpoint handler.
        plugins (PluginsEndpoint): Plugin operations endpoint handler.
        web (WebEndpoint): Web search endpoint handler.
        logger (logging.Logger): Client logger.
    """

    def __init__(self, 
                 api_key: Optional[str] = None, 
                 provisioning_api_key: Optional[str] = None, 
                 secrets_manager: Optional[SecretsManager] = None,
                 base_url: str = "https://openrouter.ai/api/v1", 
                 organization_id: Optional[str] = None,
                 reference_id: Optional[str] = None,
                 **kwargs):
        # Initialize context lengths registry
        self._context_lengths: Dict[str, int] = {}
        """
        Initialize the OpenRouterClient with authentication and configuration.
        
        Args:
            api_key (Optional[str]): API key for authentication. If None, environment variable OPENROUTER_API_KEY is used.
            provisioning_api_key (Optional[str]): API key for provisioning operations. If None, environment variable OPENROUTER_PROVISIONING_API_KEY is used.
            secrets_manager (Optional[SecretsManager]): Secrets manager for API keys. If None, environment variables are used.
            base_url (str): Base URL for the API. Defaults to "https://openrouter.ai/api/v1".
            organization_id (Optional[str]): Organization ID for request tracking.
            reference_id (Optional[str]): Reference ID for request tracking.
            **kwargs: Additional configuration options.
        
        Raises:
            AuthenticationError: If no valid API key is available.
        """
        # Create and configure logger for this client instance
        log_level = kwargs.get('log_level', logging.INFO)
        self.logger = configure_logging(level=log_level)
        
        try:
            # Initialize auth_manager with provided credentials
            self.auth_manager = AuthManager(
                api_key=api_key,
                provisioning_api_key=provisioning_api_key,
                organization_id=organization_id,
                reference_id=reference_id,
                secrets_manager=secrets_manager
            )
            
            # Store base_url for API requests
            self.base_url = base_url
            
            # Configure SmartSurge client parameters
            timeout = kwargs.get('timeout', 60.0)
            retries = kwargs.get('retries', 3)
            backoff_factor = kwargs.get('backoff_factor', 0.5)
            rate_limit = kwargs.get('rate_limit', None)
            
            # Create http_manager with SmartSurge client
            surge_kwargs = {
                'timeout': timeout,
                'retries': retries,
                'backoff_factor': backoff_factor,
                'rate_limit': rate_limit
            }
            
            self.http_manager = HTTPManager(base_url=base_url, **surge_kwargs)
            self.secrets_manager = secrets_manager
            
            # Initialize all endpoint handlers
            self._initialize_endpoints()
            
            self.logger.info(
                f"OpenRouterClient initialized successfully with base_url={base_url}, "
                f"provisioning_key={'available' if provisioning_api_key else 'not available'}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenRouterClient: {str(e)}")
            raise

    def _initialize_endpoints(self) -> None:
        """
        Initialize all endpoint handlers.
        
        Creates instances of all endpoint handler classes and assigns them
        as attributes of the client instance.
        
        Raises:
            Exception: If any of the endpoint handlers fail to initialize.
        """
        # Create CompletionsEndpoint with auth_manager and http_manager
        self.completions = CompletionsEndpoint(
            auth_manager=self.auth_manager,
            http_manager=self.http_manager
        )
        
        # Create ChatEndpoint with auth_manager and http_manager
        self.chat = ChatEndpoint(
            auth_manager=self.auth_manager,
            http_manager=self.http_manager
        )
        
        # Create ModelsEndpoint with auth_manager and http_manager
        self.models = ModelsEndpoint(
            auth_manager=self.auth_manager,
            http_manager=self.http_manager
        )
        
        # Create ImagesEndpoint with auth_manager and http_manager
        self.images = ImagesEndpoint(
            auth_manager=self.auth_manager,
            http_manager=self.http_manager
        )
        
        # Create GenerationsEndpoint with auth_manager and http_manager
        self.generations = GenerationsEndpoint(
            auth_manager=self.auth_manager,
            http_manager=self.http_manager
        )
        
        # Create CreditsEndpoint with auth_manager and http_manager
        self.credits = CreditsEndpoint(
            auth_manager=self.auth_manager,
            http_manager=self.http_manager
        )
        
        # Create KeysEndpoint with auth_manager and http_manager
        self.keys = KeysEndpoint(
            auth_manager=self.auth_manager,
            http_manager=self.http_manager
        )
        
        # Create PluginsEndpoint with auth_manager and http_manager
        self.plugins = PluginsEndpoint(
            auth_manager=self.auth_manager,
            http_manager=self.http_manager
        )
        
        # Create WebEndpoint with auth_manager and http_manager
        self.web = WebEndpoint(
            auth_manager=self.auth_manager,
            http_manager=self.http_manager
        )
        
        self.logger.debug("All endpoint handlers initialized successfully")

    def refresh_context_lengths(self) -> Dict[str, int]:
        """
        Refresh model context lengths from the API.
        
        Returns:
            Dict[str, int]: Mapping of model IDs to their maximum context lengths.
        
        Raises:
            APIError: If the API request fails.
        """
        # Log beginning of context lengths refresh
        self.logger.info("Refreshing model context lengths from API")
        
        try:
            # Retrieve models data from the models endpoint with details=True to get full model information
            models_data = self.models.list(details=True)
            
            # Process the retrieved models data
            context_lengths = {}
            # When details=True, models_data is the full response which might have a different structure
            if isinstance(models_data, dict) and 'data' in models_data:
                # Extract from data array if present
                models_list = models_data['data']
                # Ensure models_list is actually a list
                if not isinstance(models_list, list):
                    models_list = []
            elif isinstance(models_data, list):
                # Or use directly if it's already a list
                models_list = models_data
            else:
                models_list = []
                
            for model in models_list:
                if isinstance(model, dict):
                    model_id = model.get('id')
                    context_length = model.get('context_length', 0)
                    if model_id and context_length:
                        context_lengths[model_id] = context_length
            
            # Update instance context length registry
            self._context_lengths.update(context_lengths)
            
            # Log successful completion
            self.logger.info(f"Successfully refreshed context lengths for {len(context_lengths)} models")
            
            # Return a copy of the mapping
            return self._context_lengths.copy()
            
        except Exception as e:
            # Log error details
            self.logger.error(f"Failed to refresh context lengths: {str(e)}")
            
            # Re-raise as APIError
            from openrouter_client.exceptions import APIError
            raise APIError(
                message=f"Failed to retrieve model context lengths: {str(e)}",
                code="model_fetch_error"
            ) from e

    def get_context_length(self, model_id: str) -> int:
        """
        Get the context length for a model.
        
        Args:
            model_id (str): The model ID to look up.
            
        Returns:
            int: The context length for the model, or 4096 if not found.
        """
        try:
            return self._context_lengths.get(model_id, 4096)
        except TypeError:
            # Handle unhashable types (e.g., lists, dicts) gracefully
            return 4096

    def calculate_rate_limits(self) -> Dict[str, Any]:
        """
        Calculate rate limits based on remaining credits.
        
        Returns:
            Dict[str, Any]: Rate limit configuration based on available credits.
        
        Raises:
            APIError: If the credits API request fails.
        """
        # Log beginning of rate limit calculation
        self.logger.info("Calculating rate limits based on remaining credits")
        
        try:
            # Retrieve credit information from the credits endpoint
            credits_info = self.credits.get()
            
            # Handle None or invalid responses
            if not isinstance(credits_info, dict):
                credits_info = {}
            
            # Extract remaining credits and credit refresh rate
            remaining_credits = credits_info.get('remaining', 0)
            if not isinstance(remaining_credits, (int, float)):
                remaining_credits = 0
            
            refresh_rate = credits_info.get('refresh_rate', {}) or {}
            if not isinstance(refresh_rate, dict):
                refresh_rate = {}
            
            seconds_until_refresh = refresh_rate.get('seconds', 3600)  # Default to 1 hour
            if not isinstance(seconds_until_refresh, (int, float)):
                seconds_until_refresh = 3600
            
            # Calculate appropriate rate limits based on remaining credits
            requests_per_period = max(1, int(remaining_credits / 10))  # Use 10% of credits per period
            
            # Create rate limit configuration dictionary
            rate_limits = {
                "requests_per_period": requests_per_period,
                "seconds_per_period": 60,  # Default to 1 minute periods
                "cooldown": seconds_until_refresh if remaining_credits < 10 else 0
            }
            
            # Log the calculated rate limits
            self.logger.info(
                f"Calculated rate limits: {requests_per_period} requests per minute, "
                f"cooldown: {rate_limits['cooldown']} seconds"
            )
            
            # Return the rate limit configuration
            return rate_limits
            
        except Exception as e:
            # Log the error details
            self.logger.error(f"Failed to calculate rate limits: {str(e)}")
            
            # Re-raise as APIError
            from openrouter_client.exceptions import APIError
            raise APIError(
                message=f"Failed to retrieve credit information for rate limiting: {str(e)}",
                code="credits_fetch_error"
            ) from e

    def close(self) -> None:
        """
        Close the client and release all resources.
        """
        # Log beginning of client shutdown process
        self.logger.info("Shutting down OpenRouterClient")
        
        # Close the HTTP manager to release network resources
        if hasattr(self, 'http_manager') and self.http_manager is not None:
            try:
                self.http_manager.close()
            except Exception as e:
                # Log the error but continue with cleanup
                self.logger.error(f"Error closing HTTP manager: {str(e)}")
        
        # Clear all endpoint instances to release their resources
        for endpoint_name in ['completions', 'chat', 'models', 'images', 'generations', 
                             'credits', 'keys', 'plugins', 'web']:
            if hasattr(self, endpoint_name):
                try:
                    setattr(self, endpoint_name, None)
                except Exception as e:
                    # Log the error but continue with cleanup
                    self.logger.error(f"Error clearing {endpoint_name}: {str(e)}")
        
        # Log successful client shutdown
        self.logger.info("OpenRouterClient shut down successfully")

    def __enter__(self) -> 'OpenRouterClient':
        """
        Enter context manager for use in 'with' statements.
        
        Returns:
            OpenRouterClient: Self for use in with statement.
        """
        # Return self for use in the context manager
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit context manager, closing the client.
        
        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        # If an exception occurred during the context, log it
        if exc_type is not None:
            self.logger.error(
                f"Exception occurred in OpenRouterClient context: {exc_type.__name__}: {exc_val}"
            )
            
        # Close the client to release all resources
        self.close()
