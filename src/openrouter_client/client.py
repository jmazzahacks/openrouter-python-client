"""
Main OpenRouter client implementation.

This module defines the OpenRouterClient class, which serves as the primary interface
for interacting with the OpenRouter API. The client provides a unified, type-safe
interface to all available endpoints with automatic rate limiting, retry logic,
and comprehensive error handling.

Key Features:
- Unified access to all OpenRouter API endpoints (chat, completions, models, etc.)
- Automatic rate limiting based on API key limits
- Smart retry logic with exponential backoff
- Context length management and tracking
- Secure API key handling with optional encryption
- Context manager support for resource cleanup
- Credit-based rate limit calculation
- Comprehensive logging and error handling

Available Endpoints:
- client.chat: Chat completions API with streaming and function calling support
- client.completions: Text completions API with streaming support
- client.models: Model information, context lengths, and pricing
- client.generations: Generation metadata and details
- client.credits: Credit balance and usage tracking
- client.keys: API key management and provisioning

Example:
    >>> client = OpenRouterClient(api_key="your-api-key")
    >>> 
    >>> # Chat completion
    >>> response = client.chat.create(
    ...     model="anthropic/claude-3-opus",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
    >>> 
    >>> # Context manager usage
    >>> with OpenRouterClient(api_key="your-api-key") as client:
    ...     response = client.chat.create(...)
    
    >>> # Rate limit management
    >>> rate_limits = client.calculate_rate_limits()
    >>> context_length = client.get_context_length("anthropic/claude-3-opus")

Exported:
- OpenRouterClient: Main client class for OpenRouter API interaction
"""

import logging
import re
from typing import Dict, Optional, Any, Union

from .auth import AuthManager, SecretsManager
from .http import HTTPManager
from .logging import configure_logging
from .endpoints.completions import CompletionsEndpoint
from .endpoints.chat import ChatEndpoint
from .endpoints.models import ModelsEndpoint
from .endpoints.generations import GenerationsEndpoint
from .endpoints.credits import CreditsEndpoint
from .endpoints.keys import KeysEndpoint


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
        generations (GenerationsEndpoint): Generation statistics endpoint handler.
        credits (CreditsEndpoint): Credits management endpoint handler.
        keys (KeysEndpoint): API key management endpoint handler.
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
            
            # Set rate limit from API key information
            self._initialize_rate_limit()
            
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
        
        
        self.logger.debug("All endpoint handlers initialized successfully")
    
    def _parse_interval_to_seconds(self, interval: str) -> float:
        """
        Parse an interval string to seconds.
        
        Args:
            interval (str): Interval string like "10s", "5m", "1h", etc.
            
        Returns:
            float: Number of seconds.
            
        Raises:
            ValueError: If the interval format is invalid.
        """
        # Match patterns like "10s", "5m", "1h", "30", "1.5h"
        match = re.match(r'^(\d+(?:\.\d+)?)\s*([smhd]?)$', interval.strip())
        if not match:
            raise ValueError(f"Invalid interval format: {interval}")
        
        value = float(match.group(1))
        unit = match.group(2) or 's'  # Default to seconds if no unit
        
        multipliers = {
            's': 1,        # seconds
            'm': 60,       # minutes
            'h': 3600,     # hours
            'd': 86400     # days
        }
        
        return value * multipliers[unit]
    
    def _initialize_rate_limit(self) -> None:
        """
        Initialize rate limit based on the current API key's limits.
        
        Fetches the current key information and sets the global rate limit
        accordingly. If fetching fails, logs a warning and continues.
        """
        try:
            # Get current API key information
            key_info = self.keys.get_current()
            
            if not isinstance(key_info, dict) or 'data' not in key_info:
                self.logger.warning("Unexpected response format from keys.get_current")
                return
            
            data = key_info['data']
            if not isinstance(data, dict) or 'rate_limit' not in data:
                self.logger.debug("No rate limit information in API key response")
                return
            
            rate_limit = data['rate_limit']
            if not isinstance(rate_limit, dict):
                self.logger.warning("Invalid rate_limit format in API key response")
                return
            
            # Extract rate limit parameters
            requests = rate_limit.get('requests')
            interval = rate_limit.get('interval')
            
            if requests is None or interval is None:
                self.logger.warning("Missing requests or interval in rate_limit")
                return
            
            # Convert interval to seconds
            try:
                time_period = self._parse_interval_to_seconds(interval)
            except ValueError as e:
                self.logger.warning(f"Failed to parse interval '{interval}': {e}")
                return
            
            # Set the global rate limit
            self._set_global_rate_limit(
                max_requests=int(requests),
                time_period=time_period
            )
            
            self.logger.info(
                f"Rate limit initialized from API key: {requests} requests per {interval} "
                f"({requests} requests per {time_period}s)"
            )
            
        except Exception as e:
            # Log warning but don't fail initialization
            self.logger.warning(f"Failed to initialize rate limit from API key: {e}")
            # Continue without setting rate limit

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
            # When details=True, models_data is a ModelsResponse object
            if hasattr(models_data, 'data'):
                # Extract from data attribute if present (ModelsResponse)
                models_list = models_data.data
            elif isinstance(models_data, dict) and 'data' in models_data:
                # Extract from data array if dict
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
                if hasattr(model, 'id') and hasattr(model, 'context_length'):
                    # Handle Pydantic model objects
                    if model.id and model.context_length:
                        context_lengths[model.id] = model.context_length
                elif isinstance(model, dict):
                    # Handle dict objects
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
            requests = max(1, int(remaining_credits / 10))  # Use 10% of credits per period
            
            # Create rate limit configuration dictionary
            rate_limits = {
                "requests": requests,
                "period": 60,  # Default to 1 minute periods
                "cooldown": seconds_until_refresh if remaining_credits < 10 else 0
            }
            
            # Log the calculated rate limits
            self.logger.info(
                f"Calculated rate limits: {requests} requests per minute, "
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
    
    def _set_rate_limit(self,
                       endpoint: str,
                       method: Union[str, Any],
                       max_requests: int,
                       time_period: float,
                       cooldown: Optional[float] = None) -> None:
        """
        Set the rate limit for a specific API endpoint and method.
        
        This is a protected method that delegates to HTTPManager.set_rate_limit().
        Use this for fine-grained control over individual endpoint rate limits.
        
        Args:
            endpoint (str): The API endpoint to set rate limit for.
            method (Union[str, Any]): HTTP method (GET, POST, etc.) or RequestMethod enum.
            max_requests (int): Maximum number of requests allowed per time period.
            time_period (float): Time period in seconds for the rate limit.
            cooldown (Optional[float]): Cooldown period in seconds after hitting the limit.
            
        Raises:
            AttributeError: If the HTTP manager doesn't support rate limiting.
            ValueError: If invalid rate limit parameters are provided.
        """
        self.logger.debug(
            f"Setting rate limit for {method} {endpoint}: "
            f"max_requests={max_requests}, time_period={time_period}s"
        )
        
        # Delegate to HTTP manager
        self.http_manager.set_rate_limit(
            endpoint=endpoint,
            method=method,
            max_requests=max_requests,
            time_period=time_period,
            cooldown=cooldown
        )
    
    def _set_global_rate_limit(self,
                              max_requests: int,
                              time_period: float,
                              cooldown: Optional[float] = None) -> None:
        """
        Set a global rate limit for all common OpenRouter API endpoints.
        
        This is a protected convenience method that applies the same rate limit to all
        standard endpoints. For fine-grained control, use _set_rate_limit().
        
        Args:
            max_requests (int): Maximum number of requests allowed per time period.
            time_period (float): Time period in seconds for the rate limit.
            cooldown (Optional[float]): Cooldown period in seconds after hitting the limit.
            
        Example:
            # Limit to 100 requests per minute with 5 second cooldown
            client._set_global_rate_limit(100, 60, 5)
        """
        self.logger.info(
            f"Setting global rate limit: max_requests={max_requests}, "
            f"time_period={time_period}s, cooldown={cooldown}s"
        )
        
        # Delegate to HTTP manager
        self.http_manager.set_global_rate_limit(
            max_requests=max_requests,
            time_period=time_period,
            cooldown=cooldown
        )

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
        for endpoint_name in ['completions', 'chat', 'models', 'generations', 
                             'credits', 'keys']:
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
