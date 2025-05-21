import pytest
from unittest.mock import MagicMock

# Placeholder for imports from openrouter_client
from openrouter_client.exceptions import (
    OpenRouterError,
    AuthenticationError,
    APIError,
    RateLimitExceeded,
    ValidationError,
    ProviderError,
    ContextLengthExceededError,
    StreamingError,
    ResumeError
)


# OpenRouterError Tests

class Test_OpenRouterError_Init_01_NominalBehaviors:
    """Test class for nominal behaviors of OpenRouterError initialization."""
    
    @pytest.mark.parametrize("message, details", [
        ("Simple error message", {}),
        ("Error with details", {"key": "value"}),
        ("Error with multiple details", {"key1": "value1", "key2": "value2"}),
        ("Error with nested details", {"nested": {"key": "value"}}),
        ("Error with list details", {"list": [1, 2, 3]})
    ])
    def test_initialization_with_message_and_details(self, message, details):
        """
        Test that OpenRouterError correctly initializes with a message and details.
        
        This is vital because it's the minimum requirement for creating any exception
        in the hierarchy. Without proper message handling, error reporting would be broken.
        """
        error = OpenRouterError(message, **details)
        
        # Verify message is set correctly
        assert error.message == message
        
        # Verify details are stored
        for key, value in details.items():
            assert error.details.get(key) == value
        
        # Verify string representation includes the message
        assert message in str(error)


class Test_OpenRouterError_Init_02_NegativeBehaviors:
    """Test class for negative behaviors of OpenRouterError initialization."""
    
    def test_initialization_without_message(self):
        """
        Test that OpenRouterError raises TypeError when initialized without a message.
        
        This is vital because it enforces the contract that all exceptions must have
        a message, preventing silent failures or misleading errors.
        """
        with pytest.raises(TypeError):
            OpenRouterError()


class Test_OpenRouterError_Init_04_ErrorHandlingBehaviors:
    """Test class for error handling behaviors of OpenRouterError."""
    
    def test_exception_hierarchy_preservation(self):
        """
        Test that OpenRouterError maintains the exception hierarchy when caught.
        
        This is vital because exception handling code often relies on catching parent
        exception types; the exception must function correctly in try/except blocks.
        """
        try:
            raise OpenRouterError("Test error")
        except Exception as e:
            assert isinstance(e, OpenRouterError)
            assert "Test error" in str(e)
    
    def test_exception_chaining(self):
        """
        Test that OpenRouterError properly chains with other exceptions.
        
        This is vital because exceptions may be raised from other exceptions,
        and maintaining the full chain preserves the error context.
        """
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise OpenRouterError("Chained error") from e
        except OpenRouterError as e:
            assert "Chained error" in str(e)
            assert isinstance(e.__cause__, ValueError)
            assert "Original error" in str(e.__cause__)


class Test_OpenRouterError_Init_05_StateTransitionBehaviors:
    """Test class for state transition behaviors of OpenRouterError."""
    
    def test_immutability_after_instantiation(self):
        """
        Test that exception state doesn't change after instantiation.
        
        This is vital because exceptions should be reliable snapshots of error state;
        changing state after creation could lead to inconsistent error handling.
        """
        error = OpenRouterError("Test error", key="value")
        original_message = error.message
        original_details = error.details.copy()
        
        # Attempt state modification through error handling
        try:
            raise error
        except OpenRouterError as e:
            # Verify the state hasn't changed
            assert e.message == original_message
            assert e.details == original_details


# AuthenticationError Tests

class Test_AuthenticationError_Init_01_NominalBehaviors:
    """Test class for nominal behaviors of AuthenticationError initialization."""
    
    @pytest.mark.parametrize("message, details", [
        ("Authentication failed", {}),
        ("Invalid API key", {"hint": "Check your API key"}),
        ("Token expired", {"expiry": "2025-05-18T12:00:00Z"})
    ])
    def test_type_specificity_with_base_functionality(self, message, details):
        """
        Test that AuthenticationError maintains type specificity while inheriting base functionality.
        
        This is vital because code needs to distinguish authentication errors from other types,
        allowing for targeted handling of auth failures.
        """
        error = AuthenticationError(message, **details)
        
        # Verify correct type
        assert isinstance(error, AuthenticationError)
        assert isinstance(error, OpenRouterError)
        
        # Verify base functionality
        assert error.message == message
        for key, value in details.items():
            assert error.details.get(key) == value
        
        # Test specific catching
        try:
            raise error
        except AuthenticationError as e:
            assert e.message == message
        
        # Test parent catching
        try:
            raise error
        except OpenRouterError as e:
            assert isinstance(e, AuthenticationError)
            assert e.message == message


# APIError Tests

class Test_APIError_Init_01_NominalBehaviors:
    """Test class for nominal behaviors of APIError initialization."""
    
    @pytest.mark.parametrize("message, status_code, response_data", [
        ("API error", 400, {"error": "Bad request"}),
        ("Not found", 404, {"message": "Resource not found"}),
        ("Server error", 500, {"server_error": "Internal error"}),
        ("Simple API error", None, None),
    ])
    def test_status_code_and_response_storage(self, message, status_code, response_data):
        """
        Test that APIError correctly stores status_code and response.
        
        This is vital because these provide critical information about what went wrong
        with an API call; without them, debugging API issues would be significantly harder.
        """
        mock_response = None
        if response_data is not None:
            mock_response = MagicMock()
            mock_response.json.return_value = response_data
        
        error = APIError(message, status_code=status_code, response=mock_response)
        
        # Verify attributes are set correctly
        assert error.message == message
        assert error.status_code == status_code
        assert error.response == mock_response
    
    @pytest.mark.parametrize("response_data", [
        {"error": "Bad request", "code": "INVALID_PARAM"},
        {"message": "Validation failed", "field": "email"},
        {"error": {"type": "invalid_request", "message": "Missing required parameter"}},
    ])
    def test_json_error_detail_extraction(self, response_data):
        """
        Test that APIError extracts error details from JSON responses.
        
        This is vital because API responses typically include detailed error information
        in JSON format; automatic extraction saves developers from manual parsing.
        """
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        
        error = APIError("API error", response=mock_response)
        
        # Verify details were extracted
        for key, value in response_data.items():
            assert error.details.get(key) == value


class Test_APIError_Init_02_NegativeBehaviors:
    """Test class for negative behaviors of APIError initialization."""
    
    @pytest.mark.parametrize("exception_type, exception_msg", [
        (ValueError, "Invalid JSON"),
        (TypeError, "Response is not JSON serializable"),
        (AttributeError, "No json method"),
    ])
    def test_graceful_handling_of_non_parseable_responses(self, exception_type, exception_msg):
        """
        Test that APIError handles non-parseable responses gracefully.
        
        This is vital because not all API responses will contain valid JSON, and handling
        these cases prevents secondary exceptions that could mask the original error.
        """
        mock_response = MagicMock()
        mock_response.json.side_effect = exception_type(exception_msg)
        
        error = APIError("API error", response=mock_response)
        
        # Verify appropriate error detail was recorded
        if exception_type == ValueError:
            assert "json_error" in error.details
            assert exception_msg in error.details["json_error"]
        else:
            assert "parse_error" in error.details
            assert exception_msg in error.details["parse_error"]


class Test_APIError_Init_04_ErrorHandlingBehaviors:
    """Test class for error handling behaviors of APIError."""
    
    @pytest.mark.parametrize("exception_type, error_key", [
        (ValueError, "json_error"),
        (Exception, "parse_error"),
    ])
    def test_preservation_of_parsing_errors(self, exception_type, error_key):
        """
        Test that APIError preserves parsing errors in its details.
        
        This is vital because when JSON parsing fails, the original error must be preserved
        to help diagnose the issue; storing this information provides essential context.
        """
        mock_response = MagicMock()
        mock_response.json.side_effect = exception_type("Error parsing response")
        
        error = APIError("API error", response=mock_response)
        
        # Verify error was preserved with correct key
        assert error_key in error.details
        assert "Error parsing response" in error.details[error_key]


# RateLimitExceeded Tests

class Test_RateLimitExceeded_Init_01_NominalBehaviors:
    """Test class for nominal behaviors of RateLimitExceeded initialization."""
    
    @pytest.mark.parametrize("retry_after", [
        30,
        60,
        120,
        None
    ])
    def test_retry_after_information_storage(self, retry_after):
        """
        Test that RateLimitExceeded correctly stores retry-after information.
        
        This is vital because this information is essential for implementing proper
        backoff strategies; without it, clients wouldn't know how long to wait.
        """
        error = RateLimitExceeded("Rate limit exceeded", retry_after=retry_after)
        
        assert error.retry_after == retry_after
        if retry_after is not None:
            assert error.details.get("retry_after") == retry_after
    
    @pytest.mark.parametrize("explicit_status", [
        None,  # Default
        429,   # Explicit but same as default
        403,   # Different status code
    ])
    def test_default_status_code_429(self, explicit_status):
        """
        Test that RateLimitExceeded sets status_code to 429 by default.
        
        This is vital because 429 is the standard HTTP status code for rate limiting;
        providing this default ensures consistency even when API responses don't include it.
        """
        kwargs = {}
        if explicit_status is not None:
            kwargs["status_code"] = explicit_status
        
        error = RateLimitExceeded("Rate limit exceeded", **kwargs)
        
        # Should be 429 if not explicitly provided, or the explicit value if provided
        expected_status = 429 if explicit_status is None else explicit_status
        assert error.status_code == expected_status


# ValidationError Tests

class Test_ValidationError_Init_01_NominalBehaviors:
    """Test class for nominal behaviors of ValidationError initialization."""
    
    @pytest.mark.parametrize("field, details", [
        ("username", {}),
        ("email", {"valid_format": "user@example.com"}),
        (None, {"general": "validation error"}),
    ])
    def test_field_information_storage(self, field, details):
        """
        Test that ValidationError correctly stores field information.
        
        This is vital because knowing which specific field failed validation allows
        developers to provide targeted error messages and fix the specific issue.
        """
        message = f"Validation failed for field: {field}" if field else "Validation failed"
        error = ValidationError(message, field=field, **details)
        
        # Verify field is stored correctly
        assert error.field == field
        
        # Verify field is in details if provided
        if field is not None:
            assert error.details.get("field") == field
        
        # Verify other details are stored
        for key, value in details.items():
            assert error.details.get(key) == value


# ProviderError Tests

class Test_ProviderError_Init_01_NominalBehaviors:
    """Test class for nominal behaviors of ProviderError initialization."""
    
    @pytest.mark.parametrize("provider, status_code, response_data", [
        ("openai", 400, {"error": "Model not found"}),
        ("anthropic", 503, {"status": "unavailable"}),
        (None, None, None),
    ])
    def test_provider_name_storage(self, provider, status_code, response_data):
        """
        Test that ProviderError correctly stores the provider name.
        
        This is vital because in a multi-provider system like OpenRouter, identifying
        which provider generated an error is essential for debugging and error handling.
        """
        mock_response = None
        if response_data is not None:
            mock_response = MagicMock()
            mock_response.json.return_value = response_data
        
        error = ProviderError(
            "Provider error", 
            provider=provider, 
            status_code=status_code,
            response=mock_response
        )
        
        # Verify provider is stored correctly
        assert error.provider == provider
        
        # Verify provider is in details if provided
        if provider is not None:
            assert error.details.get("provider") == provider
        
        # Verify inheritance from APIError
        assert error.status_code == status_code
        assert error.response == mock_response
        
        # Verify response data extraction if provided
        if response_data is not None:
            for key, value in response_data.items():
                assert error.details.get(key) == value


# ContextLengthExceededError Tests

class Test_ContextLengthExceededError_Init_01_NominalBehaviors:
    """Test class for nominal behaviors of ContextLengthExceededError initialization."""
    
    @pytest.mark.parametrize("max_tokens, token_count", [
        (4000, 4500),
        (8000, 10000),
        (None, None),
    ])
    def test_token_count_information_storage(self, max_tokens, token_count):
        """
        Test that ContextLengthExceededError correctly stores token count information.
        
        This is vital because developers need to know both the maximum allowed tokens
        and the actual count to properly adjust their inputs.
        """
        message = "Context length exceeded"
        error = ContextLengthExceededError(
            message,
            max_tokens=max_tokens,
            token_count=token_count
        )
        
        # Verify token counts are stored correctly
        assert error.max_tokens == max_tokens
        assert error.token_count == token_count
        
        # Verify token counts are in details if provided
        if max_tokens is not None:
            assert error.details.get("max_tokens") == max_tokens
        if token_count is not None:
            assert error.details.get("token_count") == token_count
    
    def test_default_field_setting_to_messages(self):
        """
        Test that ContextLengthExceededError sets field to "messages" by default.
        
        This is vital because this is the specific field that typically contains
        the tokens that exceeded the limit; this default saves developers time.
        """
        error = ContextLengthExceededError("Context length exceeded")
        
        # Verify field is set to "messages"
        assert error.field == "messages"


# StreamingError Tests

class Test_StreamingError_Init_01_NominalBehaviors:
    """Test class for nominal behaviors of StreamingError initialization."""
    
    @pytest.mark.parametrize("response_data", [
        None,
        {"type": "connection_error", "detail": "Lost connection"},
        {"type": "stream_closed", "detail": "Stream was closed"},
    ])
    def test_response_object_storage(self, response_data):
        """
        Test that StreamingError correctly stores the response object.
        
        This is vital because streaming errors often require inspection of the
        specific response that failed; storing this provides essential context.
        """
        mock_response = None
        if response_data is not None:
            mock_response = MagicMock()
            mock_response.json.return_value = response_data
        
        error = StreamingError("Stream error", response=mock_response)
        
        # Verify response is stored correctly
        assert error.response == mock_response
        
        # Verify response is in details if provided
        assert error.details.get("response") == mock_response


# ResumeError Tests

class Test_ResumeError_Init_01_NominalBehaviors:
    """Test class for nominal behaviors of ResumeError initialization."""
    
    @pytest.mark.parametrize("position", [
        0,
        100,
        1000,
        None,
    ])
    def test_position_information_storage(self, position):
        """
        Test that ResumeError correctly stores position information.
        
        This is vital because knowing where in the stream the resumption failed
        is essential for debugging or implementing retry logic.
        """
        error = ResumeError("Resume error", position=position)
        
        # Verify position is stored correctly
        assert error.position == position
        
        # Verify position is in details if provided
        if position is not None:
            assert error.details.get("position") == position
        
        # Verify inheritance
        assert isinstance(error, StreamingError)
        assert isinstance(error, OpenRouterError)


# Common Tests for All Exception Types

class Test_CommonExceptionBehaviors_01_NominalBehaviors:
    """Test class for common nominal behaviors across all exception types."""
    
    @pytest.mark.parametrize("exception_class, args, parent_classes", [
        (OpenRouterError, ["Test error"], [Exception]),
        (AuthenticationError, ["Auth error"], [OpenRouterError, Exception]),
        (APIError, ["API error", 400], [OpenRouterError, Exception]),
        (RateLimitExceeded, ["Rate limited", 30], [APIError, OpenRouterError, Exception]),
        (ValidationError, ["Invalid data", "field"], [OpenRouterError, Exception]),
        (ProviderError, ["Provider error", "openai"], [APIError, OpenRouterError, Exception]),
        (ContextLengthExceededError, ["Too long", 4000, 4500], 
         [ValidationError, OpenRouterError, Exception]),
        (StreamingError, ["Stream error"], [OpenRouterError, Exception]),
        (ResumeError, ["Resume error", 100], [StreamingError, OpenRouterError, Exception]),
    ])
    def test_inheritance_integrity(self, exception_class, args, parent_classes):
        """
        Test that each exception maintains proper inheritance relationships.
        
        This is vital because each exception must be catchable both as its specific
        type and as any parent type; this polymorphic behavior is essential.
        """
        # Create the exception with provided args
        error = exception_class(*args)
        
        # Verify it's an instance of its own class
        assert isinstance(error, exception_class)
        
        # Verify it's an instance of each parent class
        for parent in parent_classes:
            assert isinstance(error, parent), f"Not an instance of {parent.__name__}"


class Test_CommonExceptionBehaviors_04_ErrorHandlingBehaviors:
    """Test class for common error handling behaviors across all exception types."""
    
    @pytest.mark.parametrize("exception_class, args", [
        (OpenRouterError, ["Test error"]),
        (AuthenticationError, ["Auth error"]),
        (APIError, ["API error", 400]),
        (RateLimitExceeded, ["Rate limited", 30]),
        (ValidationError, ["Invalid data", "field"]),
        (ProviderError, ["Provider error", "openai"]),
        (ContextLengthExceededError, ["Too long", 4000, 4500]),
        (StreamingError, ["Stream error"]),
        (ResumeError, ["Resume error", 100]),
    ])
    def test_proper_exception_chaining(self, exception_class, args):
        """
        Test that all exceptions can be properly chained with other exceptions.
        
        This is vital because these exceptions may be raised from other exceptions;
        maintaining the chain preserves the full context of what went wrong.
        """
        try:
            try:
                raise ValueError("Root cause")
            except ValueError as e:
                # Create and raise the exception with chaining
                raise exception_class(*args) from e
        except exception_class as caught_error:
            # Verify the exception was caught correctly
            assert isinstance(caught_error, exception_class)
            # Verify exception chaining worked
            assert isinstance(caught_error.__cause__, ValueError)
            assert "Root cause" in str(caught_error.__cause__)


class Test_CommonExceptionBehaviors_05_StateTransitionBehaviors:
    """Test class for common state transition behaviors across all exception types."""
    
    @pytest.mark.parametrize("exception_class, args, additional_kwargs", [
        (OpenRouterError, ["Test error"], {"extra": "info"}),
        (AuthenticationError, ["Auth error"], {"token": "expired"}),
        (APIError, ["API error"], {"status_code": 400}),
        (RateLimitExceeded, ["Rate limited"], {"retry_after": 30}),
        (ValidationError, ["Invalid data"], {"field": "username"}),
        (ProviderError, ["Provider error"], {"provider": "openai"}),
        (ContextLengthExceededError, ["Too long"], {"max_tokens": 4000, "token_count": 4500}),
        (StreamingError, ["Stream error"], {"response": "mock_response"}),
        (ResumeError, ["Resume error"], {"position": 100}),
    ])
    def test_state_immutability_after_instantiation(self, exception_class, args, additional_kwargs):
        """
        Test that all exceptions maintain immutable state after instantiation.
        
        This is vital because exceptions should be reliable snapshots of error state;
        changing state after creation could lead to inconsistent error handling.
        """
        # Create exception with args and kwargs
        error = exception_class(*args, **additional_kwargs)
        
        # Store original state
        original_str = str(error)
        original_message = error.message
        
        # Attempt to use the exception in different contexts
        try:
            # First raise and catch
            try:
                raise error
            except exception_class as e:
                # Verify state is preserved
                assert str(e) == original_str
                assert e.message == original_message
                
                # Try raising again from this point
                raise
        except exception_class as e2:
            # Verify state is still preserved after re-raising
            assert str(e2) == original_str
            assert e2.message == original_message