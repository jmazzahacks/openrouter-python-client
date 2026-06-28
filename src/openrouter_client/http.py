"""
HTTP communication management for OpenRouter Client.

This module handles HTTP requests, response handling, rate limiting, and
error handling for all API interactions.

Exported:
- HTTPManager: HTTP request and response manager with rate limiting
"""

import logging
import random
import time
from typing import Dict, Optional, Any, Union, Tuple

import requests
from pydantic import BaseModel, Field
from smartsurge.client import SmartSurgeClient
from smartsurge.exceptions import RateLimitExceeded as SmartSurgeRateLimitExceeded

from .exceptions import APIError, RateLimitExceeded, OpenRouterError
from .types import RequestMethod


class RetryConfig(BaseModel):
    """
    Opt-in configuration for retrying HTTP 429 (rate limit) responses with backoff.

    Disabled by default (``enabled=False``) so existing behavior is unchanged: a
    429 propagates immediately. When enabled, ``HTTPManager`` retries a rate-limited
    request, honoring the upstream ``Retry-After`` header when present and otherwise
    applying exponential backoff with jitter.

    Attributes:
        enabled (bool): Master switch. When False, no 429 retrying is performed.
        max_retries (int): Maximum retry attempts after the initial request
            (so total attempts == max_retries + 1).
        base_delay (float): Base delay in seconds for the exponential schedule
            (delay for the first retry when no Retry-After is available).
        factor (float): Exponential growth factor applied per attempt.
        max_delay (float): Upper bound (seconds) on any single sleep, including a
            server-provided Retry-After.
        jitter (float): Maximum random jitter (seconds) added to each computed
            backoff delay (0 disables jitter).
        respect_retry_after (bool): When True, an upstream Retry-After header takes
            precedence over the computed exponential delay.
    """

    enabled: bool = False
    max_retries: int = Field(default=5, ge=0)
    base_delay: float = Field(default=1.0, gt=0)
    factor: float = Field(default=2.0, ge=1.0)
    max_delay: float = Field(default=30.0, gt=0)
    jitter: float = Field(default=0.25, ge=0)
    respect_retry_after: bool = True


class HTTPManager:
    """
    Manages HTTP communications with the OpenRouter API.
    
    Attributes:
        client (SmartSurgeClient or requests.Session): HTTP client with rate limiting.
        base_url (str): Base URL for API requests.
        logger (logging.Logger): HTTP communication logger.
    """
    
    def __init__(self,
                 base_url: Optional[str] = None,
                 client: Optional[SmartSurgeClient] = None,
                 retry_config: Optional[RetryConfig] = None,
                 **kwargs):
        """
        Initialize the HTTP manager.

        Args:
            base_url (str): Base URL for API requests. If None, uses the URL from pre-configured client.
            client (Optional[SmartSurgeClient]): Pre-configured HTTP client. If None, one is created.
            retry_config (Optional[RetryConfig]): Opt-in 429 retry-with-backoff policy. If None,
                a disabled default is used (429s propagate immediately, preserving prior behavior).
            kwargs: Additional arguments for SmartSurgeClient.

        Raises:
            OpenRouterError: If neither base_url nor client is provided.
        """
        if base_url is None and client is None:
            raise OpenRouterError("Either base_url or client must be provided")

        # Set up logger for HTTP operations
        self.logger = logging.getLogger("openrouter_client.http")

        # 429 retry/backoff policy (disabled by default for backward compatibility)
        self.retry_config = retry_config if retry_config is not None else RetryConfig()
        
        # Store base_url for forming full request URLs
        if base_url is None and client is not None:
            # Extract base_url from client or set a default
            self.base_url = getattr(client, "base_url", "")
        else:
            self.base_url = base_url.rstrip("/") if base_url else ""
        
        if client:
            # Use the provided client
            self.client = client
            self.logger.debug("Using provided HTTP client")
            if base_url is not None:
                self.logger.warning("base_url is ignored when using a pre-configured client")
        else:
            # Create a SmartSurgeClient with appropriate configuration
            self.client = SmartSurgeClient(
                base_url=self.base_url,
                **kwargs
            )
            self.logger.debug("Created SmartSurgeClient with rate limiting")
        
        self.logger.debug(f"HTTP manager initialized with base_url={base_url}")
        
        # Workaround: Configure smartsurge logging to respect root logger level
        # SmartSurge doesn't follow proper logging hierarchy and explicitly sets
        # smartsurge.client to INFO level after initialization, overriding user's settings
        root_level = logging.getLogger().getEffectiveLevel()
        
        # Set parent smartsurge logger
        smartsurge_logger = logging.getLogger('smartsurge')
        if smartsurge_logger.level == logging.NOTSET:
            smartsurge_logger.setLevel(root_level)
            
        # Workaround: SmartSurge also explicitly sets smartsurge.client to INFO, so fix that too
        smartsurge_client_logger = logging.getLogger('smartsurge.client')
        if smartsurge_client_logger.level <= logging.INFO:  # Only if not set to WARNING+ by user
            smartsurge_client_logger.setLevel(root_level)

    @staticmethod
    def _parse_retry_after(value: Optional[str]) -> Optional[int]:
        """Parse a Retry-After header value into integer seconds, or None if absent/unparseable."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            # HTTP-date form of Retry-After is not handled here; fall back to backoff.
            return None

    def _backoff_delay(self, retry_after: Optional[int], attempt: int) -> float:
        """
        Compute the sleep (seconds) before the next retry.

        Honors ``Retry-After`` when present and enabled; otherwise uses exponential
        backoff (base * factor**attempt) with random jitter, capped at max_delay.
        """
        cfg = self.retry_config
        if cfg.respect_retry_after and retry_after is not None:
            return float(min(retry_after, cfg.max_delay))
        delay = cfg.base_delay * (cfg.factor ** attempt)
        delay = min(delay, cfg.max_delay)
        if cfg.jitter > 0:
            delay += random.uniform(0, cfg.jitter)
        return delay

    def _invoke(self, **request_kwargs) -> requests.Response:
        """
        Call the underlying client, applying the 429 retry/backoff policy.

        When ``retry_config.enabled`` is False this is a thin pass-through (any
        rate-limit exception propagates exactly as before). When enabled, 429s are
        retried up to ``max_retries`` times with backoff; on exhaustion an
        ``openrouter_client`` ``RateLimitExceeded`` is raised carrying the attempt
        count and total elapsed seconds.

        Args:
            **request_kwargs: Forwarded verbatim to ``self.client.request``.

        Returns:
            requests.Response: The (non-429, or un-retried) response.

        Raises:
            RateLimitExceeded: When retries are exhausted on a 429.
        """
        if not self.retry_config.enabled:
            return self.client.request(**request_kwargs)

        attempt = 0
        start = time.time()
        while True:
            try:
                response = self.client.request(**request_kwargs)
            except SmartSurgeRateLimitExceeded as e:
                retry_after = getattr(e, 'retry_after', None)
                if attempt >= self.retry_config.max_retries:
                    raise self._rate_limit_exhausted(attempt, start, retry_after) from e
                self._sleep_before_retry(retry_after, attempt)
                attempt += 1
                continue

            if response.status_code == 429:
                retry_after = self._parse_retry_after(response.headers.get('Retry-After'))
                if attempt >= self.retry_config.max_retries:
                    raise self._rate_limit_exhausted(attempt, start, retry_after, response)
                self._sleep_before_retry(retry_after, attempt)
                attempt += 1
                continue

            return response

    def _sleep_before_retry(self, retry_after: Optional[int], attempt: int) -> None:
        """Log and sleep for the computed backoff before the next attempt."""
        delay = self._backoff_delay(retry_after, attempt)
        self.logger.warning(
            f"Rate limited (429); retry {attempt + 1}/{self.retry_config.max_retries} "
            f"after {delay:.2f}s backoff"
        )
        time.sleep(delay)

    def _rate_limit_exhausted(self,
                              attempt: int,
                              start: float,
                              retry_after: Optional[int],
                              response: Optional[requests.Response] = None) -> RateLimitExceeded:
        """Build the RateLimitExceeded raised once retries are exhausted."""
        attempts = attempt + 1
        elapsed = round(time.time() - start, 3)
        return RateLimitExceeded(
            message=(
                f"Rate limit exceeded; gave up after {attempts} attempts "
                f"over {elapsed}s"
            ),
            retry_after=retry_after,
            response=response,
            attempts=attempts,
            elapsed_seconds=elapsed,
        )

    def request(self,
                method: RequestMethod,
                endpoint: str,
                headers: Optional[Dict[str, str]] = None,
                params: Optional[Dict[str, Any]] = None,
                json: Optional[Dict[str, Any]] = None,
                data: Optional[Union[Dict[str, Any], str, bytes]] = None,
                files: Optional[Dict[str, Any]] = None,
                stream: bool = False,
                timeout: Optional[Union[float, Tuple[float, float]]] = None) -> requests.Response:
        """
        Make an HTTP request to the OpenRouter API.
        
        Args:
            method (RequestMethod): HTTP method to use.
            endpoint (str): API endpoint to call (will be combined with base_url).
            headers (Optional[Dict[str, str]]): HTTP headers to include.
            params (Optional[Dict[str, Any]]): URL query parameters.
            json (Optional[Dict[str, Any]]): JSON body to send.
            data (Optional[Union[Dict[str, Any], str, bytes]]): Form data or raw data to send.
            files (Optional[Dict[str, Any]]): Files to upload.
            stream (bool): Whether to stream the response. Defaults to False.
            timeout (Optional[Union[float, Tuple[float, float]]]): Request timeout override.
            
        Returns:
            Response: API response.
            
        Raises:
            APIError: For API-related errors.
            RateLimitExceeded: When rate limits are exceeded.
            requests.RequestException: For network-related errors.
        """
        # Check if the method is a valid RequestMethod value
        if not isinstance(method, RequestMethod):
            raise TypeError(f"'method' must be a RequestMethod enum, not {type(method).__name__}")
        
        # Check if endpoint is a valid string
        if not isinstance(endpoint, str):
            raise TypeError(f"'endpoint' must be a string, not {type(endpoint).__name__}")
        
        # Check if headers is a dictionary or None
        if headers is not None and not isinstance(headers, dict):
            raise TypeError(f"'headers' must be a dictionary, not {type(headers).__name__}")

        # Check if params is a dictionary or None
        if params is not None and not isinstance(params, dict):
            raise TypeError(f"'params' must be a dictionary, not {type(params).__name__}")

        # Check if json is a dictionary or None
        if json is not None and not isinstance(json, dict):
            raise TypeError(f"'json' must be a dictionary, not {type(json).__name__}")
        
        # Form the full URL by combining base_url and endpoint
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # If headers are None, initialize as empty dictionary
        if headers is None:
            headers = {}
        
        # Generate a request ID for logging and correlation
        request_id = f"req_{int(time.time() * 1000)}_{id(self)}"
        
        # Log the outgoing request details (sanitize sensitive information)
        sanitized_json = None
        if json:
            # Create a shallow copy to avoid modifying the original
            sanitized_json = {**json}
            # Sanitize sensitive fields like API keys if present
            if 'api_key' in sanitized_json:
                sanitized_json['api_key'] = '***'
            if 'Authorization' in sanitized_json:
                sanitized_json['Authorization'] = '***'
        
        self.logger.debug(
            f"Request {request_id}: {method.value} {url} | "
            f"Headers: {headers} | Params: {params} | "
            f"JSON: {sanitized_json} | Stream: {stream}"
        )
        
        # Determine the appropriate timeout to use (given or a reasonable default)
        actual_timeout = timeout if timeout is not None else 60.0
        
        # Record the start time for performance tracking
        start_time = time.time()
        
        try:
            response = self._invoke(
                method=method.value,
                endpoint=url,
                headers=headers,
                params=params,
                json=json,
                data=data,
                files=files,
                stream=stream,
                timeout=actual_timeout
            )

            # Calculate request duration for logging
            duration = time.time() - start_time
            
            # Log response details including status code and duration
            self.logger.debug(
                f"Response {request_id}: Status {response.status_code} | "
                f"Duration: {duration:.2f}s"
            )
            
            # Handle error responses (non-2xx status codes)
            if not 200 <= response.status_code < 300:
                if 300 <= response.status_code < 400:
                    # Handle redirects by retrying with allow_redirects=True
                    self.logger.info(f"Received {response.status_code} redirect, retrying with allow_redirects=True")
                    try:
                        response = self.client.request(
                            method=method.value,
                            endpoint=url,
                            headers=headers,
                            params=params,
                            json=json,
                            data=data,
                            files=files,
                            stream=stream,
                            timeout=actual_timeout,
                            allow_redirects=True
                        )
                    except Exception as e:
                        self.logger.error(f"Error when following redirect: {str(e)}")
                        raise
                elif response.status_code == 429:
                    # Rate limit exceeded
                    retry_after = response.headers.get('Retry-After')
                    raise RateLimitExceeded(
                        message="Rate limit exceeded",
                        retry_after=retry_after,
                        response=response
                    )

                elif 400 <= response.status_code < 500:
                    # Client error
                    error_detail = {}
                    error_message = f"API Error: {response.status_code}"
                    
                    try:
                        response_data = response.json()
                        
                        # Check if response has OpenRouter's error structure
                        if 'error' in response_data:
                            error_info = response_data['error']
                            error_message = error_info.get('message', error_message)
                            error_detail = error_info
                        # Check for other common error formats
                        elif 'message' in response_data:
                            error_message = response_data['message']
                            error_detail = response_data
                        elif 'detail' in response_data:
                            error_message = response_data['detail']
                            error_detail = response_data
                        elif 'reason' in response_data:
                            error_message = response_data['reason']
                            error_detail = response_data
                        else:
                            # Fallback to using the whole response as error detail
                            error_detail = response_data
                            error_message = response_data.get('message', error_message)
                            
                    except Exception:
                        # If JSON parsing fails, use the raw response text
                        if response.text.strip():
                            error_message = f"API Error {response.status_code}: {response.text}"
                        error_detail = {'message': error_message}
                    
                    # Enhanced logging with full error details for debugging
                    # Log the main error message
                    self.logger.error(f"API Error {response.status_code}: {error_message}")
                    
                    if error_detail and len(error_detail) > 1:  # More than just the message
                        # Pretty print the error details for better readability
                        try:
                            import json
                            # Handle nested JSON strings in metadata (like the 'raw' field)
                            formatted_error_detail = {}
                            for key, value in error_detail.items():
                                if isinstance(value, dict):
                                    formatted_error_detail[key] = {}
                                    for sub_key, sub_value in value.items():
                                        if isinstance(sub_value, str) and sub_value.startswith('{') and sub_value.endswith('}'):
                                            try:
                                                # Try to parse and pretty-print nested JSON strings
                                                parsed_sub = json.loads(sub_value)
                                                formatted_error_detail[key][sub_key] = parsed_sub
                                            except:
                                                formatted_error_detail[key][sub_key] = sub_value
                                        else:
                                            formatted_error_detail[key][sub_key] = sub_value
                                else:
                                    formatted_error_detail[key] = value
                            
                            formatted_details = json.dumps(formatted_error_detail, indent=2, ensure_ascii=False)
                            self.logger.error("Error Details:")
                            for line in formatted_details.split('\n'):
                                if line.strip():
                                    self.logger.error(f"  {line}")
                        except Exception:
                            # Fallback to simple logging if JSON formatting fails
                            self.logger.error(f"Error Details: {error_detail}")
                    
                    # Add a separator to make it clear this is one complete error
                    self.logger.error("--- End of API Error ---")
                    
                    raise APIError(
                        message=error_message,
                        code=error_detail.get('code', response.status_code),
                        param=error_detail.get('param'),
                        type=error_detail.get('type'),
                        status_code=response.status_code,
                        response=response
                    )
                elif 500 <= response.status_code < 600:
                    # Server error
                    raise APIError(
                        message=f"Server error: {response.status_code}",
                        status_code=response.status_code,
                        response=response
                    )
            
            return response
            
        except (RateLimitExceeded, APIError):
            # Re-raise our custom exceptions
            raise
        except requests.RequestException as e:
            # Convert requests exceptions to APIError
            self.logger.error(f"Request error: {str(e)}")
            raise APIError(
                message=f"Request failed: {str(e)}"
            ) from e

    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """
        Make a GET request to the OpenRouter API.
        
        Args:
            endpoint (str): API endpoint to call.
            **kwargs: Additional parameters to pass to request().
            
        Returns:
            Response: API response.
        """
        return self.request(method=RequestMethod.GET, endpoint=endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs) -> requests.Response:
        """
        Make a POST request to the OpenRouter API.
        
        Args:
            endpoint (str): API endpoint to call.
            **kwargs: Additional parameters to pass to request().
            
        Returns:
            Response: API response.
        """
        return self.request(method=RequestMethod.POST, endpoint=endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs) -> requests.Response:
        """
        Make a PUT request to the OpenRouter API.
        
        Args:
            endpoint (str): API endpoint to call.
            **kwargs: Additional parameters to pass to request().
            
        Returns:
            Response: API response.
        """
        return self.request(method=RequestMethod.PUT, endpoint=endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> requests.Response:
        """
        Make a DELETE request to the OpenRouter API.
        
        Args:
            endpoint (str): API endpoint to call.
            **kwargs: Additional parameters to pass to request().
            
        Returns:
            Response: API response.
        """
        return self.request(method=RequestMethod.DELETE, endpoint=endpoint, **kwargs)

    def patch(self, endpoint: str, **kwargs) -> requests.Response:
        """
        Make a PATCH request to the OpenRouter API.
        
        Args:
            endpoint (str): API endpoint to call.
            **kwargs: Additional parameters to pass to request().
            
        Returns:
            Response: API response.
        """
        return self.request(method=RequestMethod.PATCH, endpoint=endpoint, **kwargs)

    def stream_request(self, method: RequestMethod, endpoint: str, **kwargs) -> requests.Response:
        """
        Make a streaming request to the OpenRouter API.
        
        Args:
            method (RequestMethod): HTTP method to use.
            endpoint (str): API endpoint to call.
            **kwargs: Additional parameters to pass to request().
            
        Returns:
            Response: Streaming API response.
        """
        # Force stream=True in kwargs to ensure streaming behavior
        kwargs['stream'] = True
        return self.request(method=method, endpoint=endpoint, **kwargs)

    def set_rate_limit(self, 
                       endpoint: str,
                       method: Union[str, RequestMethod],
                       max_requests: int, 
                       time_period: float,
                       cooldown: Optional[float] = None) -> None:
        """
        Set the rate limit for a specific API endpoint and method.
        
        This method passes through to SmartSurgeClient.set_rate_limit to dynamically
        adjust rate limiting parameters for a specific endpoint/method combination.
        
        Args:
            endpoint (str): The API endpoint to set rate limit for.
            method (Union[str, RequestMethod]): HTTP method (GET, POST, etc.).
            max_requests (int): Maximum number of requests allowed per time period.
            time_period (float): Time period in seconds for the rate limit.
            cooldown (Optional[float]): Cooldown period in seconds after hitting the limit.
            
        Raises:
            AttributeError: If the client doesn't support set_rate_limit.
            ValueError: If invalid rate limit parameters are provided.
        """
        if not hasattr(self.client, 'set_rate_limit'):
            raise AttributeError(
                "The HTTP client does not support dynamic rate limit configuration. "
                "Ensure you're using SmartSurgeClient."
            )
        
        # Convert RequestMethod enum to string if needed
        if isinstance(method, RequestMethod):
            method = method.value
        
        # Log the rate limit change
        self.logger.info(
            f"Setting rate limit for {method} {endpoint}: "
            f"max_requests={max_requests}, time_period={time_period}s, cooldown={cooldown}s"
        )
        
        try:
            # Pass through to SmartSurgeClient
            self.client.set_rate_limit(
                endpoint=endpoint,
                method=method,
                max_requests=max_requests,
                time_period=time_period,
                cooldown=cooldown
            )
            self.logger.debug(f"Rate limit successfully updated for {method} {endpoint}")
        except Exception as e:
            self.logger.error(f"Failed to set rate limit: {str(e)}")
            raise
    
    def set_global_rate_limit(self,
                            max_requests: int,
                            time_period: float,
                            cooldown: Optional[float] = None) -> None:
        """
        Set a global rate limit for all common OpenRouter API endpoints.
        
        This is a convenience method that applies the same rate limit to all
        standard OpenRouter endpoints. For fine-grained control, use set_rate_limit().
        
        Args:
            max_requests (int): Maximum number of requests allowed per time period.
            time_period (float): Time period in seconds for the rate limit.
            cooldown (Optional[float]): Cooldown period in seconds after hitting the limit.
        """
        # Common OpenRouter endpoints
        common_endpoints = [
            ('/chat/completions', 'POST'),
            ('/completions', 'POST'),
            ('/models', 'GET'),
            ('/credits', 'GET'),
            ('/generation', 'GET'),
            ('/auth/key', 'GET'),
            ('/auth/keys', 'POST'),
            ('/keys', 'POST'),
        ]
        
        self.logger.info(
            f"Setting global rate limit: max_requests={max_requests}, "
            f"time_period={time_period}s, cooldown={cooldown}s"
        )
        
        for endpoint, method in common_endpoints:
            try:
                self.set_rate_limit(
                    endpoint=endpoint,
                    method=method,
                    max_requests=max_requests,
                    time_period=time_period,
                    cooldown=cooldown
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to set rate limit for {method} {endpoint}: {str(e)}"
                )
    
    def close(self) -> None:
        """
        Close the HTTP manager and release resources.
        """
        if hasattr(self.client, 'close'):
            self.client.close()
            self.logger.debug("HTTP manager closed and resources released")
