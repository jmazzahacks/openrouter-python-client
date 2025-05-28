import pytest
from unittest.mock import Mock, patch
import requests
import time
import logging

from openrouter_client.exceptions import APIError, RateLimitExceeded, OpenRouterError
from openrouter_client.types import RequestMethod
from openrouter_client.http import HTTPManager
from smartsurge.client import SmartSurgeClient


class Test_HTTPManager_Init_01_NominalBehaviors:
    """Test nominal behaviors for the HTTPManager initialization."""
    
    def test_initialize_with_valid_base_url(self):
        """
        Test initialization with a valid base URL.
        
        Verifies that HTTPManager properly initializes when given only a base_url,
        creating a SmartSurgeClient with the correct parameters.
        """
        base_url = "https://api.example.com"
        with patch("openrouter_client.http.SmartSurgeClient") as mock_client_cls:
            http_manager = HTTPManager(base_url=base_url)
            
            # Verify that SmartSurgeClient was created with the correct base_url
            mock_client_cls.assert_called_once_with(base_url=base_url)
            
            # Verify that the base_url is stored correctly
            assert http_manager.base_url == base_url
            
            # Verify that the client is set correctly
            assert http_manager.client == mock_client_cls.return_value
    
    def test_initialize_with_preconfigured_client(self):
        """
        Test initialization with a pre-configured client.
        
        Verifies that HTTPManager correctly uses a pre-configured client
        when provided, without attempting to create a new one.
        """
        mock_client = Mock(spec=SmartSurgeClient)
        
        with patch("openrouter_client.http.SmartSurgeClient") as mock_client_cls:
            http_manager = HTTPManager(client=mock_client)
            
            # Verify that SmartSurgeClient was not created
            mock_client_cls.assert_not_called()
            
            # Verify that the client is set to the provided client
            assert http_manager.client == mock_client


class Test_HTTPManager_Init_02_NegativeBehaviors:
    """Test negative behaviors for the HTTPManager initialization."""
    
    def test_initialize_with_neither_base_url_nor_client(self):
        """
        Test that initialization with neither base_url nor client raises an error.
        
        Verifies that HTTPManager raises an OpenRouterError when neither
        a base_url nor a client is provided.
        """
        with pytest.raises(OpenRouterError) as exc_info:
            HTTPManager()
            
        # Verify the error message
        assert "Either base_url or client must be provided" in str(exc_info.value)


class Test_HTTPManager_Init_03_BoundaryBehaviors:
    """Test boundary behaviors for the HTTPManager initialization."""
    
    @pytest.mark.parametrize("base_url,expected_base_url", [
        ("https://api.example.com/", "https://api.example.com"),  # Single trailing slash
        ("https://api.example.com//", "https://api.example.com"),  # Multiple trailing slashes
        ("https://api.example.com///", "https://api.example.com"),  # Multiple trailing slashes
    ])
    def test_handle_base_url_with_trailing_slashes(self, base_url, expected_base_url):
        """
        Test that base_url with trailing slashes is handled correctly.
        
        Verifies that HTTPManager correctly strips trailing slashes from the base_url
        to ensure consistent URL formatting.
        """
        with patch("openrouter_client.http.SmartSurgeClient") as mock_client_cls:
            http_manager = HTTPManager(base_url=base_url)
            
            # Verify that the base_url is stored without trailing slashes
            assert http_manager.base_url == expected_base_url
            
            # Verify that SmartSurgeClient was created with the correct base_url
            mock_client_cls.assert_called_once_with(base_url=expected_base_url)


class Test_HTTPManager_Init_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for the HTTPManager initialization."""
    
    def test_handle_smartsurgeclient_initialization_failures(self):
        """
        Test handling of SmartSurgeClient initialization failures.
        
        Verifies that exceptions raised during SmartSurgeClient initialization
        are properly propagated.
        """
        base_url = "https://api.example.com"
        
        with patch("openrouter_client.http.SmartSurgeClient") as mock_client_cls:
            # Make SmartSurgeClient initialization fail
            mock_client_cls.side_effect = ValueError("Invalid configuration")
            
            # Initialization should propagate the exception
            with pytest.raises(ValueError) as exc_info:
                HTTPManager(base_url=base_url)
                
            # Verify the error message
            assert "Invalid configuration" in str(exc_info.value)


class Test_HTTPManager_Init_05_StateTransitionBehaviors:
    """Test state transition behaviors for the HTTPManager initialization."""
    
    def test_store_base_url_and_client_correctly(self):
        """
        Test that base_url and client are stored correctly.
        
        Verifies that HTTPManager correctly stores the base_url and client 
        as instance attributes.
        """
        base_url = "https://api.example.com"
        
        with patch("openrouter_client.http.SmartSurgeClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            
            http_manager = HTTPManager(base_url=base_url)
            
            # Verify that base_url is stored
            assert http_manager.base_url == base_url
            
            # Verify that client is stored
            assert http_manager.client == mock_client
            
            # Verify that logger is set up
            assert isinstance(http_manager.logger, logging.Logger)
            assert http_manager.logger.name == "openrouter_client.http"


class Test_HTTPManager_Request_01_NominalBehaviors:
    """Test nominal behaviors for the HTTPManager request method."""
    
    @pytest.fixture
    def http_manager(self):
        """Create an HTTPManager instance with a mocked client for testing."""
        mock_client = Mock(spec=SmartSurgeClient)
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_client.request.return_value = mock_response
        
        return HTTPManager(base_url="https://api.example.com", client=mock_client)
    
    def test_execute_http_requests_with_proper_routing(self, http_manager):
        """
        Test that HTTP requests are executed with proper routing.
        
        Verifies that request correctly combines the base_url and endpoint
        to form the full URL for the request.
        """
        # Set up
        method = RequestMethod.GET
        endpoint = "users"
        expected_url = f"{http_manager.base_url}/{endpoint}"
        
        # Execute
        response = http_manager.request(method=method, endpoint=endpoint)
        
        # Verify
        http_manager.client.request.assert_called_once()
        call_args = http_manager.client.request.call_args[1]
        assert call_args["method"] == method.value
        assert call_args["endpoint"] == expected_url
    
    @pytest.mark.parametrize(
        "payload_type,payload,expected_param",
        [
            ("json", {"name": "test"}, "json"),
            ("data", "raw data", "data"),
            ("params", {"query": "value"}, "params"),
            ("files", {"file": "content"}, "files"),
            ("headers", {"X-API-Key": "123"}, "headers"),
            ("stream", True, "stream"),
            ("timeout", 30.0, "timeout"),
        ],
    )
    def test_handle_various_payload_types(self, http_manager, payload_type, payload, expected_param):
        """
        Test handling of various payload types.
        
        Verifies that request correctly passes different types of payloads 
        (JSON, form data, files, etc.) to the underlying client.
        """
        # Set up
        method = RequestMethod.POST
        endpoint = "upload"
        
        # Create kwargs with the specific payload type
        kwargs = {payload_type: payload}
        
        # Execute
        response = http_manager.request(method=method, endpoint=endpoint, **kwargs)
        
        # Verify
        http_manager.client.request.assert_called_once()
        call_args = http_manager.client.request.call_args[1]
        assert call_args[expected_param] == payload


class Test_HTTPManager_Request_02_NegativeBehaviors:
    """Test negative behaviors for the HTTPManager request method."""
    
    @pytest.fixture
    def http_manager(self):
        """Create an HTTPManager instance with a mocked client for testing."""
        mock_client = Mock(spec=SmartSurgeClient)
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_client.request.return_value = mock_response
        
        return HTTPManager(base_url="https://api.example.com", client=mock_client)
    
    @pytest.mark.parametrize(
        "invalid_input,expected_exception",
        [
            # Invalid method type
            ({"method": "INVALID", "endpoint": "test"}, TypeError),
            # None endpoint
            ({"method": RequestMethod.GET, "endpoint": None}, TypeError),
            # Invalid headers type
            ({"method": RequestMethod.GET, "endpoint": "test", "headers": "not-a-dict"}, TypeError),
            # Invalid params type
            ({"method": RequestMethod.GET, "endpoint": "test", "params": "not-a-dict"}, TypeError),
            # Invalid json type
            ({"method": RequestMethod.POST, "endpoint": "test", "json": set()}, TypeError),
        ],
    )
    def test_handle_invalid_input_structures(self, http_manager, invalid_input, expected_exception):
        """
        Test handling of invalid input structures.
        
        Verifies that request raises appropriate exceptions when given invalid inputs.
        """
        # The request should raise the expected exception
        with pytest.raises(expected_exception):
            http_manager.request(**invalid_input)


class Test_HTTPManager_Request_03_BoundaryBehaviors:
    """Test boundary behaviors for the HTTPManager request method."""
    
    @pytest.fixture
    def http_manager(self):
        """Create an HTTPManager instance with a mocked client for testing."""
        mock_client = Mock(spec=SmartSurgeClient)
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_client.request.return_value = mock_response
        
        return HTTPManager(base_url="https://api.example.com", client=mock_client)
    
    @pytest.mark.parametrize(
        "endpoint,expected_url",
        [
            ("users", "https://api.example.com/users"),  # No leading slash
            ("/users", "https://api.example.com/users"),  # With leading slash
            ("users/", "https://api.example.com/users/"),  # With trailing slash
            ("/users/", "https://api.example.com/users/"),  # With both leading and trailing slash
        ],
    )
    def test_process_endpoints_with_or_without_leading_slashes(self, http_manager, endpoint, expected_url):
        """
        Test processing of endpoints with or without leading slashes.
        
        Verifies that request correctly handles endpoints with different slash patterns.
        """
        # Execute
        response = http_manager.request(method=RequestMethod.GET, endpoint=endpoint)
        
        # Verify
        http_manager.client.request.assert_called_once()
        call_args = http_manager.client.request.call_args[1]
        assert call_args["endpoint"] == expected_url


class Test_HTTPManager_Request_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for the HTTPManager request method."""
    
    @pytest.fixture
    def setup_redirect_scenario(self):
        """Set up a scenario with a 307 redirect response followed by a 200 success response."""
        mock_client = Mock(spec=SmartSurgeClient)
        
        # First response - 307 redirect
        mock_redirect_response = Mock(spec=requests.Response)
        mock_redirect_response.status_code = 307
        
        # Second response - 200 success after redirect
        mock_success_response = Mock(spec=requests.Response)
        mock_success_response.status_code = 200
        
        # Setup the mock to return first the redirect, then the success
        mock_client.request.side_effect = [
            mock_redirect_response,
            mock_success_response
        ]
        
        http_manager = HTTPManager(base_url="https://api.example.com", client=mock_client)
        
        return http_manager, mock_redirect_response, mock_success_response
    
    def test_process_3xx_redirect_responses(self, setup_redirect_scenario):
        """
        Test processing of 3xx redirect responses.
        
        Verifies that request correctly handles 3xx redirect responses
        by retrying with allow_redirects=True.
        """
        http_manager, mock_redirect_response, mock_success_response = setup_redirect_scenario
        
        # Execute
        response = http_manager.request(method=RequestMethod.GET, endpoint="redirected")
        
        # Verify that two requests were made
        assert http_manager.client.request.call_count == 2
        
        # Verify the second call included allow_redirects=True
        second_call_kwargs = http_manager.client.request.call_args_list[1][1]
        assert second_call_kwargs["allow_redirects"] is True
        
        # Verify the final response is the success response
        assert response == mock_success_response
    
    @pytest.mark.parametrize(
        "status_code,retry_after",
        [
            (429, "60"),  # With Retry-After header
            (429, None),  # Without Retry-After header
        ],
    )
    def test_handle_429_rate_limit_responses(self, status_code, retry_after):
        """
        Test handling of 429 rate limit responses.
        
        Verifies that request correctly raises RateLimitExceeded
        when receiving a 429 response, with or without Retry-After header.
        """
        mock_client = Mock(spec=SmartSurgeClient)
        
        # Create rate limit response
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = status_code
        mock_response.headers = {}
        if retry_after is not None:
            mock_response.headers["Retry-After"] = retry_after
            
        mock_client.request.return_value = mock_response
        
        http_manager = HTTPManager(base_url="https://api.example.com", client=mock_client)
        
        # Execute - should raise RateLimitExceeded
        with pytest.raises(RateLimitExceeded) as exc_info:
            http_manager.request(method=RequestMethod.GET, endpoint="ratelimited")
        
        # Verify the exception contains the retry-after information if provided
        assert exc_info.value.retry_after == retry_after
        assert exc_info.value.response == mock_response
    
    @pytest.mark.parametrize(
        "status_code,error_json,expected_message,expected_param,expected_type",
        [
            (
                400, 
                {"message": "Bad request", "code": "invalid_parameter", "param": "user_id", "type": "validation_error"},
                "Bad request", "user_id", "validation_error"
            ),
            (
                404,
                {"message": "Not found", "code": "resource_not_found"},
                "Not found", None, None
            ),
            (
                422,
                {"message": "Validation failed"},
                "Validation failed", None, None
            ),
            (
                400,
                None,  # No JSON, only text
                "API Error: 400", None, None
            ),
        ],
    )
    def test_process_4xx_client_errors(
        self, status_code, error_json, expected_message, expected_param, expected_type
    ):
        """
        Test processing of 4xx client errors.
        
        Verifies that request correctly transforms 4xx responses
        into APIError exceptions with the appropriate details.
        """
        mock_client = Mock(spec=SmartSurgeClient)
        
        # Create client error response
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = status_code
        
        if error_json is not None:
            mock_response.json.return_value = error_json
        else:
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_response.text = "Error text"
            
        mock_client.request.return_value = mock_response
        
        http_manager = HTTPManager(base_url="https://api.example.com", client=mock_client)
        
        # Execute - should raise APIError
        with pytest.raises(APIError) as exc_info:
            http_manager.request(method=RequestMethod.GET, endpoint="error")
        
        # Verify the exception contains the error details
        assert exc_info.value.message == expected_message
        assert exc_info.value.status_code == status_code
        assert exc_info.value.details.get("param") == expected_param
        assert exc_info.value.details.get("type") == expected_type
        assert exc_info.value.response == mock_response
    
    @pytest.mark.parametrize(
        "status_code,expected_message",
        [
            (500, "Server error: 500"),
            (502, "Server error: 502"),
            (503, "Server error: 503"),
        ],
    )
    def test_handle_5xx_server_errors(self, status_code, expected_message):
        """
        Test handling of 5xx server errors.
        
        Verifies that request correctly transforms 5xx responses
        into APIError exceptions.
        """
        mock_client = Mock(spec=SmartSurgeClient)
        
        # Create server error response
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = status_code
        mock_client.request.return_value = mock_response
        
        http_manager = HTTPManager(base_url="https://api.example.com", client=mock_client)
        
        # Execute - should raise APIError
        with pytest.raises(APIError) as exc_info:
            http_manager.request(method=RequestMethod.GET, endpoint="server-error")
        
        # Verify the exception contains the error details
        assert exc_info.value.message == expected_message
        assert exc_info.value.status_code == status_code
        assert exc_info.value.response == mock_response
    
    @pytest.mark.parametrize(
        "exception_cls,exception_msg",
        [
            (requests.ConnectionError, "Connection error"),
            (requests.Timeout, "Request timed out"),
            (requests.RequestException, "Generic request error"),
        ],
    )
    def test_manage_network_failures(self, exception_cls, exception_msg):
        """
        Test management of network failures.
        
        Verifies that request correctly transforms network-related exceptions
        into APIError exceptions.
        """
        mock_client = Mock(spec=SmartSurgeClient)
        
        # Setup network failure
        mock_client.request.side_effect = exception_cls(exception_msg)
        
        http_manager = HTTPManager(base_url="https://api.example.com", client=mock_client)
        
        # Execute - should raise APIError
        with pytest.raises(APIError) as exc_info:
            http_manager.request(method=RequestMethod.GET, endpoint="network-error")
        
        # Verify the exception contains the error details
        assert f"Request failed: {exception_msg}" in exc_info.value.message


class Test_HTTPManager_Request_05_StateTransitionBehaviors:
    """Test state transition behaviors for the HTTPManager request method."""
    
    @pytest.fixture
    def http_manager(self):
        """Create an HTTPManager instance with a mocked client for testing."""
        mock_client = Mock(spec=SmartSurgeClient)
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_client.request.return_value = mock_response
        
        return HTTPManager(base_url="https://api.example.com", client=mock_client)
    
    def test_generate_unique_request_ids(self, http_manager, monkeypatch):
        """
        Test generation of unique request IDs.
        
        Verifies that request generates unique IDs for each request
        to facilitate traceability in logs.
        """
        # Mock time.time to return a fixed value for deterministic testing
        mock_time = Mock(return_value=1000.0)
        monkeypatch.setattr(time, "time", mock_time)
        
        # Patch the logger to capture log messages
        mock_logger = Mock(spec=logging.Logger)
        http_manager.logger = mock_logger
        
        # Execute the request
        response = http_manager.request(method=RequestMethod.GET, endpoint="test")
        
        # Verify that a unique request ID was generated and used in logging
        mock_logger.debug.assert_called()
        log_message = mock_logger.debug.call_args_list[0][0][0]
        assert "Request req_1000000" in log_message
    
    @pytest.mark.parametrize(
        "sensitive_data,sensitive_fields",
        [
            (
                {"api_key": "secret-api-key", "other_field": "non-sensitive"},
                ["api_key"]
            ),
            (
                {"Authorization": "Bearer secret-token", "data": "test"},
                ["Authorization"]
            ),
            (
                {"api_key": "secret1", "Authorization": "secret2", "normal": "visible"},
                ["api_key", "Authorization"]
            ),
        ],
    )
    def test_sanitize_sensitive_information(self, http_manager, sensitive_data, sensitive_fields):
        """
        Test sanitization of sensitive information.
        
        Verifies that request sanitizes sensitive fields (like API keys)
        before logging to prevent security issues.
        """
        # Patch the logger to capture log messages
        mock_logger = Mock(spec=logging.Logger)
        http_manager.logger = mock_logger
        
        # Execute a request with sensitive information
        response = http_manager.request(
            method=RequestMethod.POST,
            endpoint="authenticate",
            json=sensitive_data
        )
        
        # Verify that sensitive information was sanitized in the log
        mock_logger.debug.assert_called()
        log_message = mock_logger.debug.call_args_list[0][0][0]
        
        # Verify each sensitive field was masked
        for field in sensitive_fields:
            assert field in log_message
            assert sensitive_data[field] not in log_message
            assert "***" in log_message
        
        # Verify non-sensitive fields were preserved
        for key, value in sensitive_data.items():
            if key not in sensitive_fields:
                assert key in log_message
                assert value in log_message


class Test_HTTPManager_HttpMethods_01_NominalBehaviors:
    """Test nominal behaviors for the HTTPManager HTTP method wrapper functions."""
    
    @pytest.fixture
    def http_manager(self):
        """Create an HTTPManager instance with a mocked request method."""
        manager = HTTPManager(base_url="https://api.example.com")
        # Mock the request method
        manager.request = Mock(return_value=Mock(spec=requests.Response))
        return manager
    
    @pytest.mark.parametrize(
        "method_name,expected_request_method",
        [
            ("get", RequestMethod.GET),
            ("post", RequestMethod.POST),
            ("put", RequestMethod.PUT),
            ("delete", RequestMethod.DELETE),
            ("patch", RequestMethod.PATCH),
        ],
    )
    def test_delegate_to_request_method_with_correct_http_method(
        self, http_manager, method_name, expected_request_method
    ):
        """
        Test that HTTP method wrappers delegate to request method with correct HTTP method.
        
        Verifies that each HTTP method wrapper (get, post, etc.) correctly calls
        the request method with the appropriate RequestMethod enum value.
        """
        # Get the method from the instance
        method = getattr(http_manager, method_name)
        
        # Execute the method with various parameters
        endpoint = "test-endpoint"
        kwargs = {"headers": {"X-Test": "value"}, "params": {"q": "search"}}
        
        response = method(endpoint, **kwargs)
        
        # Verify the request method was called with the correct parameters
        http_manager.request.assert_called_once_with(
            method=expected_request_method,
            endpoint=endpoint,
            **kwargs
        )


class Test_HTTPManager_StreamRequest_01_NominalBehaviors:
    """Test nominal behaviors for the HTTPManager stream_request method."""
    
    @pytest.fixture
    def http_manager(self):
        """Create an HTTPManager instance with a mocked request method."""
        manager = HTTPManager(base_url="https://api.example.com")
        # Mock the request method
        manager.request = Mock(return_value=Mock(spec=requests.Response))
        return manager
    
    @pytest.mark.parametrize(
        "method,endpoint,kwargs,expected_kwargs",
        [
            (
                RequestMethod.GET,
                "streaming",
                {"headers": {"X-Test": "value"}},
                {"headers": {"X-Test": "value"}, "stream": True}
            ),
            (
                RequestMethod.POST,
                "streaming",
                {"stream": False},  # Attempting to set stream=False
                {"stream": True}    # Should be overridden to True
            ),
            (
                RequestMethod.GET,
                "streaming",
                {},  # No kwargs
                {"stream": True}  # stream=True added
            ),
        ],
    )
    def test_force_stream_true_parameter(self, http_manager, method, endpoint, kwargs, expected_kwargs):
        """
        Test that stream_request forces stream=True parameter.
        
        Verifies that stream_request always sets stream=True in the kwargs
        passed to the request method, regardless of what was provided.
        """
        # Execute
        response = http_manager.stream_request(method=method, endpoint=endpoint, **kwargs)
        
        # Verify request was called with stream=True
        http_manager.request.assert_called_once_with(
            method=method,
            endpoint=endpoint,
            **expected_kwargs
        )


class Test_HTTPManager_StreamRequest_05_StateTransitionBehaviors:
    """Test state transition behaviors for the HTTPManager stream_request method."""
    
    @pytest.fixture
    def http_manager_with_real_request(self):
        """Create an HTTPManager instance where request returns a streaming response."""
        manager = HTTPManager(base_url="https://api.example.com")
        
        # Create a mock response with streaming capabilities
        mock_response = Mock(spec=requests.Response)
        mock_response.iter_content.return_value = iter([b"chunk1", b"chunk2"])
        
        # Mock the request method to return the streaming response
        manager.request = Mock(return_value=mock_response)
        
        return manager, mock_response
    
    def test_return_streaming_response_without_consumption(self, http_manager_with_real_request):
        """
        Test that stream_request returns a streaming response without consuming it.
        
        Verifies that stream_request returns the response object directly
        without consuming the response content, preserving the streaming nature.
        """
        http_manager, mock_response = http_manager_with_real_request
        
        # Execute
        response = http_manager.stream_request(method=RequestMethod.GET, endpoint="stream")
        
        # Verify the response is returned as-is
        assert response == mock_response
        
        # Verify that iter_content wasn't called (response not consumed)
        mock_response.iter_content.assert_not_called()


class Test_HTTPManager_Close_01_NominalBehaviors:
    """Test nominal behaviors for the HTTPManager close method."""
    
    def test_close_client_and_release_resources(self):
        """
        Test that close method closes the client and releases resources.
        
        Verifies that close correctly calls the client's close method
        to release resources.
        """
        # Create a client that has a close method
        mock_client = Mock(spec=SmartSurgeClient)
        mock_client.close = Mock()
        
        # Create the HTTP manager
        http_manager = HTTPManager(client=mock_client)
        
        # Execute
        http_manager.close()
        
        # Verify client.close was called
        mock_client.close.assert_called_once()


class Test_HTTPManager_Close_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for the HTTPManager close method."""
    
    def test_handle_client_without_close_method(self):
        """
        Test handling of a client without a close method.
        
        Verifies that close gracefully handles clients that don't have
        a close method, avoiding AttributeError.
        """
        # Create a client without a close method
        mock_client = Mock()
        # Ensure there's no close method
        if hasattr(mock_client, "close"):
            delattr(mock_client, "close")
        
        # Create the HTTP manager
        http_manager = HTTPManager(client=mock_client)
        
        # Execute - should not raise an exception
        http_manager.close()  # This should complete without error


class Test_HTTPManager_SetRateLimit_01_NominalBehaviors:
    """Test nominal behaviors for the HTTPManager set_rate_limit method."""
    
    def test_set_rate_limit_with_valid_parameters(self):
        """
        Test setting rate limit with valid parameters.
        
        Verifies that set_rate_limit correctly delegates to SmartSurgeClient
        with the provided parameters.
        """
        # Create a mock client with set_rate_limit method
        mock_client = Mock(spec=SmartSurgeClient)
        mock_client.set_rate_limit = Mock()
        
        # Create the HTTP manager
        http_manager = HTTPManager(client=mock_client)
        
        # Execute
        http_manager.set_rate_limit(
            endpoint="/chat/completions",
            method="POST",
            max_requests=10,
            time_period=60.0,
            cooldown=5.0
        )
        
        # Verify client.set_rate_limit was called with correct parameters
        mock_client.set_rate_limit.assert_called_once_with(
            endpoint="/chat/completions",
            method="POST",
            max_requests=10,
            time_period=60.0,
            cooldown=5.0
        )
    
    def test_set_rate_limit_with_request_method_enum(self):
        """
        Test setting rate limit with RequestMethod enum.
        
        Verifies that set_rate_limit correctly converts RequestMethod enum
        to string before passing to SmartSurgeClient.
        """
        # Create a mock client with set_rate_limit method
        mock_client = Mock(spec=SmartSurgeClient)
        mock_client.set_rate_limit = Mock()
        
        # Create the HTTP manager
        http_manager = HTTPManager(client=mock_client)
        
        # Execute with RequestMethod enum
        http_manager.set_rate_limit(
            endpoint="/models",
            method=RequestMethod.GET,
            max_requests=50,
            time_period=60.0
        )
        
        # Verify client.set_rate_limit was called with string method
        mock_client.set_rate_limit.assert_called_once_with(
            endpoint="/models",
            method="GET",
            max_requests=50,
            time_period=60.0,
            cooldown=None
        )
    
    def test_set_rate_limit_without_cooldown(self):
        """
        Test setting rate limit without cooldown parameter.
        
        Verifies that set_rate_limit works correctly when cooldown
        is not provided (uses default None).
        """
        # Create a mock client with set_rate_limit method
        mock_client = Mock(spec=SmartSurgeClient)
        mock_client.set_rate_limit = Mock()
        
        # Create the HTTP manager
        http_manager = HTTPManager(client=mock_client)
        
        # Execute without cooldown
        http_manager.set_rate_limit(
            endpoint="/credits",
            method="GET",
            max_requests=100,
            time_period=60.0
        )
        
        # Verify client.set_rate_limit was called with cooldown=None
        mock_client.set_rate_limit.assert_called_once_with(
            endpoint="/credits",
            method="GET",
            max_requests=100,
            time_period=60.0,
            cooldown=None
        )


class Test_HTTPManager_SetRateLimit_02_NegativeBehaviors:
    """Test negative behaviors for the HTTPManager set_rate_limit method."""
    
    def test_set_rate_limit_with_client_without_method(self):
        """
        Test setting rate limit when client doesn't support it.
        
        Verifies that set_rate_limit raises AttributeError when the
        client doesn't have set_rate_limit method.
        """
        # Create a client without set_rate_limit method
        mock_client = Mock()
        # Ensure there's no set_rate_limit method
        if hasattr(mock_client, "set_rate_limit"):
            delattr(mock_client, "set_rate_limit")
        
        # Create the HTTP manager
        http_manager = HTTPManager(client=mock_client)
        
        # Execute and expect AttributeError
        with pytest.raises(AttributeError) as exc_info:
            http_manager.set_rate_limit(
                endpoint="/chat/completions",
                method="POST",
                max_requests=10,
                time_period=60.0
            )
        
        # Verify error message
        assert "does not support dynamic rate limit configuration" in str(exc_info.value)


class Test_HTTPManager_SetRateLimit_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for the HTTPManager set_rate_limit method."""
    
    def test_set_rate_limit_propagates_client_errors(self):
        """
        Test that set_rate_limit propagates errors from the client.
        
        Verifies that exceptions from SmartSurgeClient.set_rate_limit
        are properly propagated.
        """
        # Create a mock client that raises an exception
        mock_client = Mock(spec=SmartSurgeClient)
        mock_client.set_rate_limit = Mock(side_effect=ValueError("Invalid rate limit"))
        
        # Create the HTTP manager
        http_manager = HTTPManager(client=mock_client)
        
        # Execute and expect ValueError
        with pytest.raises(ValueError) as exc_info:
            http_manager.set_rate_limit(
                endpoint="/chat/completions",
                method="POST",
                max_requests=-1,  # Invalid value
                time_period=60.0
            )
        
        # Verify error message
        assert "Invalid rate limit" in str(exc_info.value)


class Test_HTTPManager_SetGlobalRateLimit_01_NominalBehaviors:
    """Test nominal behaviors for the HTTPManager set_global_rate_limit method."""
    
    def test_set_global_rate_limit_applies_to_all_endpoints(self):
        """
        Test that set_global_rate_limit applies rate limit to all common endpoints.
        
        Verifies that the method calls set_rate_limit for each common
        OpenRouter endpoint.
        """
        # Create a mock client with set_rate_limit method
        mock_client = Mock(spec=SmartSurgeClient)
        mock_client.set_rate_limit = Mock()
        
        # Create the HTTP manager
        http_manager = HTTPManager(client=mock_client)
        
        # Execute
        http_manager.set_global_rate_limit(
            max_requests=50,
            time_period=60.0,
            cooldown=10.0
        )
        
        # Define expected endpoints
        expected_endpoints = [
            ('/chat/completions', 'POST'),
            ('/completions', 'POST'),
            ('/models', 'GET'),
            ('/credits', 'GET'),
            ('/generation', 'GET'),
            ('/auth/key', 'GET'),
            ('/auth/keys', 'POST'),
            ('/keys', 'POST'),
        ]
        
        # Verify set_rate_limit was called for each endpoint
        assert mock_client.set_rate_limit.call_count == len(expected_endpoints)
        
        # Verify each call
        for i, (endpoint, method) in enumerate(expected_endpoints):
            call_args = mock_client.set_rate_limit.call_args_list[i]
            assert call_args == ((endpoint, method, 50, 60.0, 10.0),) or \
                   call_args == ((), {'endpoint': endpoint, 'method': method, 
                                     'max_requests': 50, 'time_period': 60.0, 
                                     'cooldown': 10.0})


class Test_HTTPManager_SetGlobalRateLimit_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for the HTTPManager set_global_rate_limit method."""
    
    def test_set_global_rate_limit_continues_on_individual_failures(self):
        """
        Test that set_global_rate_limit continues even if some endpoints fail.
        
        Verifies that the method logs warnings but continues setting
        rate limits for other endpoints if one fails.
        """
        # Create a mock client that fails on specific endpoints
        mock_client = Mock(spec=SmartSurgeClient)
        call_count = 0
        
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Fail on the third call
            if call_count == 3:
                raise ValueError("Endpoint not supported")
            return None
        
        mock_client.set_rate_limit = Mock(side_effect=side_effect)
        
        # Create the HTTP manager
        http_manager = HTTPManager(client=mock_client)
        
        # Execute - should not raise exception
        http_manager.set_global_rate_limit(
            max_requests=50,
            time_period=60.0
        )
        
        # Verify set_rate_limit was called 8 times (all endpoints)
        assert mock_client.set_rate_limit.call_count == 8
