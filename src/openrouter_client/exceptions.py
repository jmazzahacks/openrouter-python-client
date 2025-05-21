"""
Exceptions for OpenRouter Client.

This module defines custom exceptions for better error handling and
clear error messages throughout the library.

Exported:
- OpenRouterError: Base exception for all OpenRouter Client errors
- AuthenticationError: Error during authentication
- APIError: Error from API responses
- RateLimitExceeded: Rate limit exceeded error
- ValidationError: Validation error for inputs
- ProviderError: Error from specific model providers
- ContextLengthExceededError: Error when context length is exceeded
- StreamingError: Error during streaming operations
- ResumeError: Error when attempting to resume a streaming operation
"""

from typing import Optional, Any


class OpenRouterError(Exception):
    """
    Base exception for all OpenRouter Client errors.
    
    Attributes:
        message (str): Error message.
        details (Dict[str, Any]): Additional error details.
    """

    def __init__(self, message: str, **kwargs):
        """
        Initialize the exception.
        
        Args:
            message (str): Error message.
            **kwargs: Additional error details.
        """
        self.message = message
        self.details = kwargs
        super().__init__(message)


class AuthenticationError(OpenRouterError):
    """
    Exception for authentication errors.
    
    Attributes:
        message (str): Error message.
        details (Dict[str, Any]): Additional error details.
    """
    pass


class APIError(OpenRouterError):
    """
    Exception for API errors.
    
    Attributes:
        message (str): Error message.
        status_code (Optional[int]): HTTP status code.
        response (Optional[Any]): Raw response object.
        details (Dict[str, Any]): Additional error details from the API.
            May include 'json_error' if JSON parsing failed or
            'parse_error' for other response parsing issues.
    """
    def __init__(self, message: str, status_code: Optional[int] = None,
                response: Optional[Any] = None, **kwargs):
        self.status_code = status_code
        self.response = response
        
        # Call parent constructor first
        super().__init__(message, **kwargs)
        
        # Then set details from response (which will override parent setting)
        if response is not None:
            try:
                if hasattr(response, 'json') and callable(response.json):
                    error_details = response.json()
                    if isinstance(error_details, dict):
                        self.details.update(error_details)
            except ValueError as e:
                self.details = {'json_error': str(e)}
            except Exception as e:
                self.details = {'parse_error': str(e)}



class RateLimitExceeded(APIError):
    """
    Exception for rate limit exceeded errors.
    
    Attributes:
        message (str): Error message.
        retry_after (Optional[int]): Seconds to wait before retrying.
        details (Dict[str, Any]): Additional error details.
    """
    def __init__(self, 
                 message: str, 
                 retry_after: Optional[int] = None, 
                 **kwargs):
        """
        Initialize the exception.
        
        Args:
            message (str): Error message.
            retry_after (Optional[int]): Seconds to wait before retrying.
            **kwargs: Additional error details.
        """
        self.retry_after = retry_after
        status_code = kwargs.pop('status_code', 429)
        super().__init__(message, status_code=status_code, **kwargs)
        
        if retry_after is not None:
            if not hasattr(self, 'details'):
                self.details = {}
            self.details['retry_after'] = retry_after


class ValidationError(OpenRouterError):
    """
    Exception for validation errors.
    
    Attributes:
        message (str): Error message.
        field (Optional[str]): Field that failed validation.
        details (Dict[str, Any]): Additional error details.
    """

    def __init__(self, 
                 message: str, 
                 field: Optional[str] = None, 
                 **kwargs):
        """
        Initialize the exception.
        
        Args:
            message (str): Error message.
            field (Optional[str]): Field that failed validation.
            **kwargs: Additional error details.
        """
        self.field = field
        super().__init__(message, field=field, **kwargs)


class ProviderError(APIError):
    """
    Exception for provider-specific errors.
    
    Attributes:
        message (str): Error message.
        provider (Optional[str]): Provider name that generated the error.
        status_code (Optional[int]): HTTP status code.
        response (Optional[Any]): Raw response object.
        details (Dict[str, Any]): Additional error details.
    """
    
    def __init__(self,
                 message: str,
                 provider: Optional[str] = None,
                 status_code: Optional[int] = None,
                 response: Optional[Any] = None,
                 **kwargs):
        """
        Initialize the exception.
        
        Args:
            message (str): Error message.
            provider (Optional[str]): Provider name that generated the error.
            status_code (Optional[int]): HTTP status code.
            response (Optional[Any]): Raw response object.
            **kwargs: Additional error details.
        """
        self.provider = provider
        super().__init__(message, status_code=status_code, response=response, provider=provider, **kwargs)


class ContextLengthExceededError(ValidationError):
    """
    Exception for context length exceeded errors.
    
    Attributes:
        message (str): Error message.
        max_tokens (Optional[int]): Maximum tokens allowed.
        token_count (Optional[int]): Actual token count in the request.
        details (Dict[str, Any]): Additional error details.
    """
    
    def __init__(self,
                 message: str,
                 max_tokens: Optional[int] = None,
                 token_count: Optional[int] = None,
                 **kwargs):
        """
        Initialize the exception.
        
        Args:
            message (str): Error message.
            max_tokens (Optional[int]): Maximum tokens allowed.
            token_count (Optional[int]): Actual token count in the request.
            **kwargs: Additional error details.
        """
        self.max_tokens = max_tokens
        self.token_count = token_count
        super().__init__(message, field="messages", max_tokens=max_tokens, token_count=token_count, **kwargs)


class StreamingError(OpenRouterError):
    """
    Exception for streaming operation errors.
    
    Attributes:
        message (str): Error message.
        response (Optional[Any]): Raw response object that caused the error.
        details (Dict[str, Any]): Additional error details.
    """
    
    def __init__(self,
                 message: str,
                 response: Optional[Any] = None,
                 **kwargs):
        """
        Initialize the exception.
        
        Args:
            message (str): Error message.
            response (Optional[Any]): Raw response object that caused the error.
            **kwargs: Additional error details.
        """
        self.response = response
        super().__init__(message, response=response, **kwargs)


class ResumeError(StreamingError):
    """
    Exception for errors when attempting to resume a streaming operation.
    
    Attributes:
        message (str): Error message.
        position (Optional[int]): Position where resumption was attempted.
        details (Dict[str, Any]): Additional error details.
    """
    
    def __init__(self,
                 message: str,
                 position: Optional[int] = None,
                 **kwargs):
        """
        Initialize the exception.
        
        Args:
            message (str): Error message.
            position (Optional[int]): Position where resumption was attempted.
            **kwargs: Additional error details.
        """
        self.position = position
        super().__init__(message, position=position, **kwargs)
