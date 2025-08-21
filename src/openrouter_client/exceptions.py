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
    
    def __str__(self):
        """Enhanced string representation with more error details."""
        base_msg = f"API Error {self.status_code}: {self.message}"
        
        # Add additional details if available
        if self.details:
            detail_parts = []
            for key, value in self.details.items():
                if key not in ['message', 'json_error', 'parse_error'] and value is not None:
                    # Handle nested structures more elegantly
                    if isinstance(value, dict) and len(str(value)) > 100:
                        # For long nested structures, just show the key and type
                        detail_parts.append(f"{key}: <{type(value).__name__} with {len(value)} items>")
                    elif isinstance(value, str) and len(value) > 100:
                        # For long strings, truncate and show length
                        detail_parts.append(f"{key}: {value[:100]}... (truncated, {len(value)} chars)")
                    else:
                        detail_parts.append(f"{key}: {value}")
            
            if detail_parts:
                base_msg += f" | Details: {'; '.join(detail_parts)}"
        
        return base_msg
    
    def __repr__(self):
        """Detailed representation for debugging."""
        return f"APIError(message='{self.message}', status_code={self.status_code}, details={self.details})"
    
    def get_detailed_error_info(self) -> str:
        """
        Get a comprehensive error summary for debugging.
        
        Returns:
            str: Detailed error information including status code, message, and all available details.
        """
        lines = [f"API Error {self.status_code}: {self.message}"]
        
        if self.details:
            lines.append("Error Details:")
            for key, value in self.details.items():
                if value is not None:
                    if isinstance(value, dict):
                        # Pretty print dictionaries
                        try:
                            import json
                            formatted_value = json.dumps(value, indent=2, ensure_ascii=False)
                            lines.append(f"  {key}:")
                            for line in formatted_value.split('\n'):
                                if line.strip():
                                    lines.append(f"    {line}")
                        except Exception:
                            lines.append(f"  {key}: {value}")
                    elif isinstance(value, str) and ('\n' in value or len(value) > 100):
                        # Handle multiline strings and long strings
                        lines.append(f"  {key}:")
                        if len(value) > 100:
                            lines.append(f"    {value[:100]}... (truncated, {len(value)} chars)")
                        else:
                            for line in value.split('\n'):
                                if line.strip():
                                    lines.append(f"    {line}")
                    else:
                        lines.append(f"  {key}: {value}")
        
        if self.response and hasattr(self.response, 'text') and self.response.text.strip():
            lines.append("Raw Response:")
            try:
                # Try to parse and pretty-print as JSON
                import json
                parsed_response = json.loads(self.response.text)
                formatted_response = json.dumps(parsed_response, indent=2, ensure_ascii=False)
                for line in formatted_response.split('\n'):
                    if line.strip():
                        lines.append(f"  {line}")
            except Exception:
                # If not valid JSON, format as plain text
                raw_text = self.response.text.strip()
                if '\n' in raw_text:
                    for line in raw_text.split('\n'):
                        if line.strip():
                            lines.append(f"  {line}")
                else:
                    lines.append(f"  {raw_text}")
        
        return "\n".join(lines)



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
                 state_file: Optional[str] = None,
                 position: Optional[int] = None,
                 original_error: Optional[Exception] = None,
                 **kwargs):
        """
        Initialize the exception.
        
        Args:
            message (str): Error message.
            position (Optional[int]): Position where resumption was attempted.
            **kwargs: Additional error details.
        """
        self.position = position
        self.state_file = state_file
        self.original_error = original_error
        super().__init__(message, position=position, **kwargs)
