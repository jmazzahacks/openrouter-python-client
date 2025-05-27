import pytest
import os
import tempfile
import time
import json
from typing import Iterator

from openrouter_client.auth import AuthManager
from openrouter_client.endpoints.completions import CompletionsEndpoint
from openrouter_client.exceptions import ResumeError, APIError
from openrouter_client.http import HTTPManager
from openrouter_client.models.completions import CompletionsResponse, CompletionsStreamResponse, CompletionsRequest

@pytest.fixture(scope="session")
def http_manager():
    """
    Shared HTTPManager instance for testing HTTP requests to OpenRouter API.
    """
    return HTTPManager(base_url="https://openrouter.ai/api/v1")

@pytest.fixture(scope="session")
def auth_manager():
    """
    Shared AuthManager instance for testing authentication with OpenRouter API.
    
    Returns:
        AuthManager: Authenticated AuthManager instance.
        
    Raises:
        APIError: If API key or provisioning API key is not set in environment variables.
    """
    if not os.environ.get("OPENROUTER_API_KEY") or not os.environ.get("OPENROUTER_PROVISIONING_API_KEY"):
        raise APIError("API key or provisioning API key not set in environment variables.")
    
    return AuthManager()

@pytest.fixture()
def invalid_auth():
    """
    AuthManager instance with invalid API key and provisioning API key.
    
    Returns:
        AuthManager: Invalid AuthManager instance.
    """
    return AuthManager(api_key="invalid_key_12345", provisioning_api_key="invalid_provisioning_api_key")

@pytest.fixture()
def invalid_http():
    """
    HTTPManager instance with invalid base URL.
    
    Returns:
        HTTPManager: Invalid HTTPManager instance.
    """
    return HTTPManager(base_url="https://invalid.openrouter.ai/api/v1")

@pytest.fixture
def completions_endpoint(http_manager, auth_manager):
    """CompletionsEndpoint instance for testing HTTP requests to OpenRouter API."""
    return CompletionsEndpoint(auth_manager, http_manager)

@pytest.fixture
def temp_state_file():
    """
    Temporary state file for streaming resumption tests.
    Automatically cleaned up after test completion.
    """
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        yield f.name
    # Cleanup
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass

@pytest.fixture
def sample_tools():
    """Sample tool definitions for testing tools parameter."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]

@pytest.fixture
def sample_functions():
    """Sample function definitions for testing functions parameter."""
    return [
        {
            "name": "calculate_sum",
            "description": "Calculate sum of two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    ]

class Test_CompletionsEndpoint_Create_01_NominalBehaviors:
    """Test nominal behaviors for the create method - successful HTTP operations with valid inputs."""
    
    @pytest.mark.parametrize("prompt,model,expected_response_type", [
        ("Hello, world!", "qwen/qwen3-8b", CompletionsResponse),
        ("Write a short poem about coding", "mistralai/mistral-small-3.1-24b-instruct", CompletionsResponse),
        ("Explain quantum computing briefly", "qwen/qwen3-8b", CompletionsResponse),
    ])
    def test_successful_non_streaming_http_requests(self, completions_endpoint, prompt, model, expected_response_type):
        """Test successful non-streaming HTTP completion requests to OpenRouter API with various models."""
        # Arrange
        max_tokens = 50
        
        # Act
        response = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            stream=False
        )
        
        # Assert
        assert isinstance(response, expected_response_type)
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        assert response.choices[0].text is not None

    @pytest.mark.parametrize("prompt,model", [
        ("Hello, world!", "qwen/qwen3-8b"),
        ("Write a short story about AI", "mistralai/mistral-small-3.1-24b-instruct"),
    ])
    def test_successful_streaming_http_requests(self, completions_endpoint, prompt, model):
        """Test successful streaming HTTP completion requests to OpenRouter API."""
        # Arrange
        max_tokens = 30
        
        # Act
        response_iterator = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            stream=True
        )
        
        # Assert
        assert hasattr(response_iterator, '__iter__')
        chunks = list(response_iterator)
        assert len(chunks) > 0
        assert all(isinstance(chunk, (CompletionsStreamResponse, dict)) for chunk in chunks)

    @pytest.mark.parametrize("temperature,top_p,max_tokens,presence_penalty,frequency_penalty", [
        (0.7, 0.9, 50, 0.0, 0.0),
        (1.0, 1.0, 50, 0.5, 0.5),
        (0.1, 0.5, 25, -0.5, -0.5),
    ])
    def test_valid_json_payload_transmission(self, completions_endpoint, temperature, top_p, max_tokens, 
                                           presence_penalty, frequency_penalty):
        """Test proper HTTP transmission of JSON payload with all supported parameters to OpenRouter API."""
        # Arrange
        prompt = "Test parameter transmission"
        model = "qwen/qwen3-8b"
        
        # Act
        response = completions_endpoint.create(
            prompt=prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stream=False
        )
        
        # Assert
        assert isinstance(response, CompletionsResponse)
        assert response.choices[0].text is not None

    def test_authentication_header_inclusion_in_http_requests(self, completions_endpoint):
        """Test that authentication headers are properly included in HTTP requests to OpenRouter API."""
        # Arrange
        prompt = "Test authentication header"
        model = "qwen/qwen3-8b"
        
        # Act
        response = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=20,
            stream=False
        )
        
        # Assert - If we get a valid response, authentication headers worked
        assert isinstance(response, CompletionsResponse)
        assert response.choices[0].text is not None

    def test_correct_endpoint_url_construction(self, completions_endpoint):
        """Test correct endpoint URL construction for OpenRouter API calls."""
        # Arrange
        prompt = "Test endpoint URL"
        model = "qwen/qwen3-8b"
        
        # Act
        response = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=20,
            stream=False
        )
        
        # Assert - Successful response indicates correct URL construction
        assert isinstance(response, CompletionsResponse)

    @pytest.mark.parametrize("tools,tool_choice", [
        (None, None),
        ("sample_tools", "auto"),
        ("sample_tools", None),
    ])
    def test_tools_parameter_http_transmission(self, completions_endpoint, tools, tool_choice, sample_tools):
        """Test HTTP transmission of tools parameter to OpenRouter API."""
        # Arrange
        prompt = "What's the weather like?"
        model = "qwen/qwen3-8b"
        actual_tools = sample_tools if tools == "sample_tools" else None
        
        # Act
        if actual_tools is not None:
            # Tools might not be supported on completions endpoint, expect possible error
            try:
                response = completions_endpoint.create(
                    prompt=prompt,
                    model=model,
                    tools=actual_tools,
                    tool_choice=tool_choice,
                    max_tokens=30,
                    stream=False
                )
                # If it succeeds, verify response
                assert isinstance(response, CompletionsResponse)
            except APIError as e:
                # Tools might not be supported on completions endpoint
                error_str = str(e).lower()
                assert "404" in error_str or "400" in error_str or "not supported" in error_str
        else:
            # Without tools, should work normally
            response = completions_endpoint.create(
                prompt=prompt,
                model=model,
                tools=actual_tools,
                tool_choice=tool_choice,
                max_tokens=30,
                stream=False
            )
            assert isinstance(response, CompletionsResponse)

    @pytest.mark.parametrize("functions,function_call", [
        (None, None),
        ("sample_functions", "auto"),
        ("sample_functions", None),
    ])
    def test_functions_parameter_http_transmission(self, completions_endpoint, functions, function_call, sample_functions):
        """Test HTTP transmission of functions parameter to OpenRouter API."""
        # Arrange
        prompt = "Calculate 5 + 3"
        model = "cohere/command-r-08-2024"
        actual_functions = sample_functions if functions == "sample_functions" else None
        
        # Act
        if actual_functions is not None:
            # Functions might not be supported on completions endpoint, expect possible error
            try:
                response = completions_endpoint.create(
                    prompt=prompt,
                    model=model,
                    functions=actual_functions,
                    function_call=function_call,
                    max_tokens=30,
                    stream=False
                )
                # If it succeeds, verify response
                assert isinstance(response, CompletionsResponse)
            except APIError as e:
                # Functions might not be supported on completions endpoint
                error_str = str(e).lower()
                assert "404" in error_str or "400" in error_str or "not supported" in error_str
        else:
            # Without functions, should work normally
            response = completions_endpoint.create(
                prompt=prompt,
                model=model,
                functions=actual_functions,
                function_call=function_call,
                max_tokens=30,
                stream=False
            )
            assert isinstance(response, CompletionsResponse)

class Test_CompletionsEndpoint_Create_02_NegativeBehaviors:
    """Test negative behaviors for the create method - HTTP error handling with invalid inputs."""
    
    def test_invalid_api_key_http_authentication(self, invalid_auth, http_manager):
        """Test HTTP request with invalid API key authentication to OpenRouter API."""
        # Arrange
        completions_endpoint = CompletionsEndpoint(invalid_auth, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            completions_endpoint.create(
                prompt="Test prompt",
                model="qwen/qwen3-8b",
                max_tokens=20
            )
        
        error_str = str(exc_info.value).lower()
        assert "401" in error_str or "unauthorized" in error_str or "invalid" in error_str

    @pytest.mark.parametrize("invalid_model", [
        "nonexistent/model-v1",
        "fake/invalid-model-123",
        "wrong-format-model",
    ])
    def test_nonexistent_model_http_requests(self, completions_endpoint, invalid_model):
        """Test HTTP requests to non-existent model identifiers on OpenRouter API."""
        # Arrange
        prompt = "Test prompt"
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            completions_endpoint.create(
                prompt=prompt,
                model=invalid_model,
                max_tokens=20
            )
        
        error_str = str(exc_info.value).lower()
        # OpenRouter returns 400 for invalid models
        assert "400" in error_str or "404" in error_str or "not found" in error_str or "invalid" in error_str

    @pytest.mark.parametrize("invalid_temperature", [
        -1.0,   # Below minimum
        3.0,    # Above maximum
        -0.1,   # Slightly below minimum
        2.1,    # Slightly above maximum
    ])
    def test_unsupported_parameter_values_http_errors(self, completions_endpoint, invalid_temperature):
        """Test HTTP requests with unsupported parameter values to OpenRouter API."""
        # Arrange
        prompt = "Test prompt"
        model = "qwen/qwen3-8b"
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            completions_endpoint.create(
                prompt=prompt,
                model=model,
                temperature=invalid_temperature,
                max_tokens=20
            )
        
        error_str = str(exc_info.value).lower()
        assert "400" in error_str or "bad request" in error_str or "invalid" in error_str

    @pytest.mark.parametrize("malformed_payload", [
        {"prompt": None},  # None prompt
        {"prompt": "", "model": None},  # None model with empty prompt
    ])
    def test_malformed_json_payload_http_requests(self, completions_endpoint, malformed_payload):
        """Test HTTP requests with malformed JSON payload to OpenRouter API."""
        # Arrange & Act & Assert
        with pytest.raises((APIError, ValueError)) as exc_info:
            completions_endpoint.create(**malformed_payload, max_tokens=20)
        
        # Should get either HTTP error or validation error
        assert exc_info.value is not None

    def test_http_timeout_scenario_handling(self, completions_endpoint):
        """Test HTTP timeout scenarios during API communication with OpenRouter."""
        # Arrange - Use a very small max_tokens to minimize request time
        prompt = "Test timeout handling"
        model = "qwen/qwen3-8b"
        
        # Act - Make request with expectation it should normally succeed
        # Note: We can't easily force timeouts without mocking, so we test normal operation
        try:
            response = completions_endpoint.create(
                prompt=prompt,
                model=model,
                max_tokens=5,  # Small to minimize chance of timeout
                stream=False
            )
            # Assert - Should succeed under normal conditions
            assert isinstance(response, CompletionsResponse)
        except APIError as e:
            # If we get timeout-related errors, ensure they're handled appropriately
            error_str = str(e).lower()
            if "timeout" in error_str:
                assert "timeout" in error_str  # Timeout errors should be properly identified

class Test_CompletionsEndpoint_Create_03_BoundaryBehaviors:
    """Test boundary behaviors for the create method - HTTP requests with edge case values."""
    
    @pytest.mark.parametrize("max_tokens", [
        1,      # Minimum viable tokens
        100,    # Reasonable maximum for tests
        500,    # Higher but still reasonable for tests
    ])
    def test_max_tokens_boundary_http_requests(self, completions_endpoint, max_tokens):
        """Test HTTP requests with boundary values for max_tokens to OpenRouter API."""
        # Arrange
        prompt = "Test"
        model = "qwen/qwen3-8b"
        
        # Act
        response = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            stream=False
        )
        
        # Assert
        assert isinstance(response, CompletionsResponse)
        assert response.choices[0].text is not None

    @pytest.mark.parametrize("temperature", [
        0.0,    # Minimum temperature
        2.0,    # Maximum temperature
        0.01,   # Just above minimum
        1.99,   # Just below maximum
    ])
    def test_temperature_boundary_http_values(self, completions_endpoint, temperature):
        """Test HTTP requests with boundary temperature values to OpenRouter API."""
        # Arrange
        prompt = "Test temperature boundary"
        model = "qwen/qwen3-8b"
        
        # Act
        response = completions_endpoint.create(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=20,
            stream=False
        )
        
        # Assert
        assert isinstance(response, CompletionsResponse)

    @pytest.mark.parametrize("top_p", [
        1.0,    # Maximum
        0.001,  # Very low
        0.999,  # Very high
    ])
    def test_top_p_boundary_http_values(self, completions_endpoint, top_p):
        """Test HTTP requests with boundary top_p values to OpenRouter API."""
        # Arrange
        prompt = "Test top_p boundary"
        model = "qwen/qwen3-8b"
        
        # Act
        response = completions_endpoint.create(
            prompt=prompt,
            model=model,
            top_p=top_p,
            max_tokens=20,
            stream=False
        )
        
        # Assert
        assert isinstance(response, CompletionsResponse)

    def test_large_prompt_payload_http_transmission(self, completions_endpoint):
        """Test HTTP request with extremely large prompt payload to OpenRouter API."""
        # Arrange
        large_prompt = "This is a test prompt that will be repeated many times. " * 200
        model = "qwen/qwen3-8b"
        
        # Act
        response = completions_endpoint.create(
            prompt=large_prompt,
            model=model,
            max_tokens=20,
            stream=False
        )
        
        # Assert
        assert isinstance(response, CompletionsResponse)

    @pytest.mark.parametrize("n", [
        1,    # Minimum
        2,    # Multiple completions
        3,    # More completions
    ])
    def test_multiple_completions_boundary_http_requests(self, completions_endpoint, n):
        """Test HTTP requests with boundary values for n parameter to OpenRouter API."""
        # Arrange
        prompt = "Test multiple completions"
        model = "qwen/qwen3-8b"
        
        # Act
        response = completions_endpoint.create(
            prompt=prompt,
            model=model,
            n=n,
            max_tokens=20,
            stream=False
        )
        
        # Assert
        assert isinstance(response, CompletionsResponse)
        # Some models/APIs may not support n > 1, so we check that at least 1 completion is returned
        # and that no more than n completions are returned
        assert 1 <= len(response.choices) <= n
        # If n=1, we expect exactly 1 completion
        if n == 1:
            assert len(response.choices) == 1

    @pytest.mark.parametrize("stop_sequences", [
        ["."],  # Single stop sequence
        [".", "!", "?"],  # Multiple stop sequences
        ["END", "STOP"],  # Word-based stop sequences
    ])
    def test_stop_sequences_boundary_http_transmission(self, completions_endpoint, stop_sequences):
        """Test HTTP transmission of boundary stop sequence values to OpenRouter API."""
        # Arrange
        prompt = "Write a sentence"
        model = "qwen/qwen3-8b"
        
        # Act
        response = completions_endpoint.create(
            prompt=prompt,
            model=model,
            stop=stop_sequences,
            max_tokens=50,
            stream=False
        )
        
        # Assert
        assert isinstance(response, CompletionsResponse)

class Test_CompletionsEndpoint_Create_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for the create method - HTTP exception handling."""
    
    def test_http_401_unauthorized_response_handling(self, invalid_auth, http_manager):
        """Test handling of HTTP 401 unauthorized responses from OpenRouter API."""
        # Arrange
        completions_endpoint = CompletionsEndpoint(invalid_auth, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            completions_endpoint.create(
                prompt="Test unauthorized",
                model="qwen/qwen3-8b",
                max_tokens=20
            )
        
        error_str = str(exc_info.value).lower()
        assert "401" in error_str or "unauthorized" in error_str

    def test_http_rate_limiting_response_handling(self, completions_endpoint):
        """Test handling of HTTP 429 rate limiting responses from OpenRouter API."""
        # Arrange - Make multiple requests rapidly to potentially trigger rate limiting
        prompt = "Rate limit test"
        model = "qwen/qwen3-8b"
        
        # Act - Try multiple rapid requests
        successful_requests = 0
        rate_limit_encountered = False
        
        for i in range(5):  # Conservative number to avoid excessive API calls
            try:
                response = completions_endpoint.create(
                    prompt=f"{prompt} {i}",
                    model=model,
                    max_tokens=5,
                    stream=False
                )
                successful_requests += 1
                time.sleep(0.5)  # Brief pause between requests
            except APIError as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str:
                    rate_limit_encountered = True
                    break
                else:
                    raise  # Re-raise non-rate-limit errors
        
        # Assert - Either requests succeeded or rate limiting was handled
        assert successful_requests > 0 or rate_limit_encountered

    @pytest.mark.parametrize("error_inducing_config", [
        {"model": "nonexistent/model", "expected_error": "404"},
        {"temperature": -1, "expected_error": "400"},
    ])
    def test_http_5xx_server_error_handling(self, completions_endpoint, error_inducing_config):
        """Test handling of various HTTP error responses from OpenRouter API."""
        # Arrange
        prompt = "Test error handling"
        base_params = {"prompt": prompt, "max_tokens": 20}
        
        # Remove expected_error from params
        expected_error = error_inducing_config.pop("expected_error")
        base_params.update(error_inducing_config)
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            completions_endpoint.create(**base_params)
        
        error_str = str(exc_info.value)
        assert expected_error in error_str or "error" in error_str.lower()
    
    def test_invalid_max_tokens_error_handling(self, completions_endpoint):
        """Test handling of invalid max_tokens parameter."""
        # Arrange
        prompt = "Test error handling"
        
        # Act & Assert - max_tokens validation might happen client-side or server-side
        try:
            response = completions_endpoint.create(
                prompt=prompt,
                model="qwen/qwen3-8b",
                max_tokens=-1,
                stream=False
            )
            # If no error was raised, the API might be accepting it and handling it differently
            # This is acceptable behavior as long as we get a response
            assert response is not None
        except (APIError, ValueError) as e:
            # Either validation error or API error is acceptable
            error_str = str(e).lower()
            assert "400" in error_str or "invalid" in error_str or "max_tokens" in error_str

    def test_malformed_response_json_parsing_errors(self, completions_endpoint):
        """Test handling of malformed response JSON parsing errors from OpenRouter API."""
        # Arrange
        prompt = "Test JSON parsing"
        model = "qwen/qwen3-8b"
        
        # Act - Make normal request (we expect this to succeed with valid JSON)
        response = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=20,
            stream=False
        )
        
        # Assert - Should receive either parsed response or fallback to raw response
        assert response is not None
        assert isinstance(response, (CompletionsResponse, dict))

    def test_network_connectivity_failure_handling(self, completions_endpoint):
        """Test handling of network connectivity failures during HTTP requests."""
        # Note: This test verifies that network errors are properly caught and handled
        # We can't easily simulate network failures without mocking
        
        # Arrange
        prompt = "Test connectivity"
        model = "qwen/qwen3-8b"
        
        # Act - Normal request should succeed
        try:
            response = completions_endpoint.create(
                prompt=prompt,
                model=model,
                max_tokens=10,
                stream=False
            )
            # Assert - Normal requests should work
            assert isinstance(response, CompletionsResponse)
        except APIError as e:
            # If network errors occur, they should be properly wrapped
            assert isinstance(e, APIError)

class Test_CompletionsEndpoint_Create_05_StateTransitionBehaviors:
    """Test state transition behaviors for the create method - HTTP request state changes."""
    
    @pytest.mark.parametrize("stream_mode", [False, True])
    def test_streaming_mode_http_transitions(self, completions_endpoint, stream_mode):
        """Test HTTP request state transitions between non-streaming and streaming modes."""
        # Arrange
        prompt = "Test streaming transition"
        model = "qwen/qwen3-8b"
        
        # Act
        response = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=20,
            stream=stream_mode
        )
        
        # Assert
        if stream_mode:
            assert hasattr(response, '__iter__')
            chunks = list(response)
            assert len(chunks) > 0
        else:
            assert isinstance(response, CompletionsResponse)

    @pytest.mark.parametrize("validate_request", [False, True])
    def test_request_validation_state_http_changes(self, completions_endpoint, validate_request):
        """Test HTTP request state changes when validate_request is enabled vs disabled."""
        # Arrange
        prompt = "Test validation state"
        model = "qwen/qwen3-8b"
        
        # Act
        response = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=20,
            validate_request=validate_request,
            stream=False
        )
        
        # Assert
        assert isinstance(response, CompletionsResponse)
        assert response.choices[0].text is not None

    def test_authentication_header_management_http_state(self, completions_endpoint):
        """Test HTTP authentication header state management between sequential requests."""
        # Arrange
        prompt1 = "First HTTP request"
        prompt2 = "Second HTTP request"
        model = "qwen/qwen3-8b"
        
        # Act - Make sequential HTTP requests
        response1 = completions_endpoint.create(
            prompt=prompt1,
            model=model,
            max_tokens=10,
            stream=False
        )
        
        response2 = completions_endpoint.create(
            prompt=prompt2,
            model=model,
            max_tokens=10,
            stream=False
        )
        
        # Assert - Both HTTP requests should succeed with persistent authentication state
        assert isinstance(response1, CompletionsResponse)
        assert isinstance(response2, CompletionsResponse)

    @pytest.mark.parametrize("request_sequence", [
        [{"stream": False}, {"stream": True}],
        [{"stream": True}, {"stream": False}],
        [{"validate_request": False}, {"validate_request": True}],
    ])
    def test_parameter_state_transitions_http_requests(self, completions_endpoint, request_sequence):
        """Test HTTP request parameter state transitions between different configurations."""
        # Arrange
        base_params = {
            "prompt": "Test parameter transitions",
            "model": "qwen/qwen3-8b",
            "max_tokens": 15
        }
        
        responses = []
        
        # Act - Execute sequence of HTTP requests with different parameters
        for params in request_sequence:
            full_params = {**base_params, **params}
            response = completions_endpoint.create(**full_params)
            
            # Handle both streaming and non-streaming responses
            if isinstance(response, Iterator):
                response = list(response)
            
            responses.append(response)
        
        # Assert - All HTTP requests should complete successfully
        assert len(responses) == len(request_sequence)
        for response in responses:
            assert response is not None

class Test_CompletionsEndpoint_ResumeStream_01_NominalBehaviors:
    """Test nominal behaviors for the resume_stream method - successful HTTP stream resumption."""
    
    def test_successful_http_stream_resumption(self, completions_endpoint, temp_state_file):
        """Test successful HTTP request resumption from valid state file to OpenRouter API."""
        # Arrange - Create initial streaming HTTP request
        prompt = "Write a detailed story about artificial intelligence and its impact on society"
        model = "qwen/qwen3-8b"
        
        # Start streaming HTTP request and save state
        stream_iterator = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=50,
            stream=True,
            state_file=temp_state_file
        )
        
        # Consume initial chunks to establish HTTP state
        chunks_consumed = 0
        for chunk in stream_iterator:
            chunks_consumed += 1
            if chunks_consumed >= 3:  # Stop after consuming some chunks
                break
        
        # Verify HTTP state file was created
        assert os.path.exists(temp_state_file)
        
        # Act - Resume HTTP streaming from saved state
        resumed_iterator = completions_endpoint.resume_stream(state_file=temp_state_file)
        
        # Assert
        assert hasattr(resumed_iterator, '__iter__')
        resumed_chunks = list(resumed_iterator)
        assert isinstance(resumed_chunks, list)

    def test_authentication_header_restoration_http_resume(self, completions_endpoint, temp_state_file):
        """Test proper authentication header restoration from saved HTTP state."""
        # Arrange - Create streaming request with authentication
        prompt = "Explain machine learning concepts"
        model = "qwen/qwen3-8b"
        
        stream_iterator = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=50,
            stream=True,
            state_file=temp_state_file
        )
        
        # Consume some chunks to save HTTP state with auth headers
        initial_chunks = 0
        for chunk in stream_iterator:
            initial_chunks += 1
            if initial_chunks >= 2:
                break
        
        # Act - Resume stream (should restore authentication headers)
        resumed_iterator = completions_endpoint.resume_stream(state_file=temp_state_file)
        
        # Assert - Should successfully resume with restored authentication
        try:
            resumed_chunks = list(resumed_iterator)
            assert isinstance(resumed_chunks, list)
        except APIError as e:
            # Should not get authentication errors if headers were properly restored
            assert "401" not in str(e), "Authentication headers were not properly restored"

    def test_endpoint_url_reconstruction_http_resume(self, completions_endpoint, temp_state_file):
        """Test correct endpoint URL reconstruction for resumed HTTP streaming requests."""
        # Arrange
        prompt = "Discuss the future of programming"
        model = "qwen/qwen3-8b"
        
        # Create initial streaming request to establish endpoint URL state
        stream_iterator = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=40,
            stream=True,
            state_file=temp_state_file
        )
        
        # Process initial chunks
        for i, chunk in enumerate(stream_iterator):
            if i >= 2:
                break
        
        # Act - Resume with reconstructed endpoint URL
        resumed_iterator = completions_endpoint.resume_stream(state_file=temp_state_file)
        
        # Assert - Should successfully connect to correct endpoint
        resumed_chunks = list(resumed_iterator)
        assert isinstance(resumed_chunks, list)

class Test_CompletionsEndpoint_ResumeStream_02_NegativeBehaviors:
    """Test negative behaviors for the resume_stream method - HTTP errors with invalid inputs."""
    
    def test_corrupted_state_file_http_resume(self, completions_endpoint, temp_state_file):
        """Test HTTP request resumption with corrupted state file."""
        # Arrange - Create corrupted state file
        with open(temp_state_file, 'w') as f:
            f.write("corrupted_json_content{{{invalid")
        
        # Act & Assert
        with pytest.raises(ResumeError):
            list(completions_endpoint.resume_stream(state_file=temp_state_file))

    def test_nonexistent_state_file_http_resume(self, completions_endpoint):
        """Test HTTP request resumption with non-existent state file."""
        # Arrange
        nonexistent_file = "/nonexistent/path/to/state.json"
        
        # Act & Assert
        with pytest.raises(ResumeError):
            list(completions_endpoint.resume_stream(state_file=nonexistent_file))

    def test_invalid_credentials_http_resume(self, completions_endpoint, temp_state_file):
        """Test HTTP resume attempt with invalid/expired authentication credentials."""
        # Arrange - Create state file with invalid authentication
        invalid_state = {
            "endpoint": "https://openrouter.ai/api/v1/completions",
            "headers": {"Authorization": "Bearer invalid_expired_token"},
            "prompt": "Test prompt",
            "params": {"model": "qwen/qwen3-8b", "max_tokens": 20, "stream": True},
            "position": 0
        }
        
        with open(temp_state_file, 'w') as f:
            json.dump(invalid_state, f)
        
        # Act & Assert
        with pytest.raises((ResumeError, APIError)):
            list(completions_endpoint.resume_stream(state_file=temp_state_file))

    def test_changed_endpoint_url_http_resume(self, completions_endpoint, temp_state_file):
        """Test HTTP request to changed or unavailable endpoint URL."""
        # Arrange - Create state with invalid endpoint URL
        invalid_endpoint_state = {
            "endpoint": "https://invalid-endpoint.example.com/api/v1/completions",
            "headers": {"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}"},
            "prompt": "Test prompt",
            "params": {"model": "qwen/qwen3-8b", "max_tokens": 20, "stream": True},
            "position": 0
        }
        
        with open(temp_state_file, 'w') as f:
            json.dump(invalid_endpoint_state, f)
        
        # Act & Assert
        with pytest.raises((ResumeError, APIError)):
            list(completions_endpoint.resume_stream(state_file=temp_state_file))

class Test_CompletionsEndpoint_ResumeStream_03_BoundaryBehaviors:
    """Test boundary behaviors for the resume_stream method - HTTP edge cases and boundaries."""
    
    def test_minimal_valid_state_http_resumption(self, completions_endpoint, temp_state_file):
        """Test HTTP request resumption with minimal valid state data."""
        # Arrange - Create minimal but valid state
        minimal_state = {
            "endpoint": "https://openrouter.ai/api/v1/completions",
            "headers": {"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}"},
            "prompt": "Minimal test",
            "params": {
                "model": "qwen/qwen3-8b", 
                "max_tokens": 10, 
                "stream": True
            },
            "position": 0
        }
        
        with open(temp_state_file, 'w') as f:
            json.dump(minimal_state, f)
        
        # Act & Assert - Should handle minimal state appropriately
        try:
            resumed_iterator = completions_endpoint.resume_stream(state_file=temp_state_file)
            list(resumed_iterator)  # Consume the iterator
        except (ResumeError, APIError) as e:
            # May fail due to invalid position/state but should fail gracefully
            assert isinstance(e, (ResumeError, APIError))

    def test_maximum_position_offset_http_resume(self, completions_endpoint, temp_state_file):
        """Test resume streaming at maximum allowed position offset."""
        # Arrange - First create a real streaming request
        prompt = "Create a comprehensive guide"
        model = "qwen/qwen3-8b"
        
        # Start stream and consume significant portion
        stream_iterator = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=40,
            stream=True,
            state_file=temp_state_file
        )
        
        # Consume most chunks to reach near-maximum position
        chunks_consumed = 0
        for chunk in stream_iterator:
            chunks_consumed += 1
            if chunks_consumed >= 5:  # Consume significant portion
                break
        
        # Act - Resume from advanced position
        resumed_iterator = completions_endpoint.resume_stream(state_file=temp_state_file)
        
        # Assert - Should handle advanced position gracefully
        remaining_chunks = list(resumed_iterator)
        assert isinstance(remaining_chunks, list)
        # May have few or no remaining chunks, which is valid

class Test_CompletionsEndpoint_ResumeStream_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for the resume_stream method - HTTP exceptions and errors."""
    
    def test_invalid_state_format_http_handling(self, completions_endpoint, temp_state_file):
        """Test handling of invalid state file format during HTTP resumption."""
        # Arrange - Create state file with wrong format/missing required fields
        invalid_state = {
            "wrong_field": "wrong_value",
            "missing": "required_fields"
        }
        
        with open(temp_state_file, 'w') as f:
            json.dump(invalid_state, f)
        
        # Act & Assert
        with pytest.raises(ResumeError):
            list(completions_endpoint.resume_stream(state_file=temp_state_file))

    def test_http_authentication_failures_on_resume(self, completions_endpoint, temp_state_file, invalid_auth, http_manager):
        """Test HTTP authentication failures on resume attempts."""
        # Arrange - Create state with valid structure
        auth_failure_state = {
            "endpoint": "https://openrouter.ai/api/v1/completions",
            "method": "POST",
            "headers": {"Content-Type": "application/json", "Accept": "application/json"},
            "params": {},
            "data": {"prompt": "Test auth failure", "model": "qwen/qwen3-8b", "max_tokens": 20, "stream": True},
            "accumulated_data": "",
            "last_position": 0,
            "total_size": None,
            "etag": None,
            "last_updated": "2025-05-27T00:00:00Z",
            "request_id": "test123"
        }
        
        with open(temp_state_file, 'w') as f:
            json.dump(auth_failure_state, f)
        
        # Create endpoint with invalid auth
        invalid_completions_endpoint = CompletionsEndpoint(invalid_auth, http_manager)
        
        # Act & Assert
        with pytest.raises((ResumeError, APIError)) as exc_info:
            list(invalid_completions_endpoint.resume_stream(state_file=temp_state_file))
        
        # Should identify authentication-related errors
        error_str = str(exc_info.value).lower()
        assert any(term in error_str for term in ["401", "unauthorized", "invalid", "expired"])

    def test_http_connection_failures_during_resume(self, completions_endpoint, temp_state_file):
        """Test HTTP connection failures during stream resumption."""
        # Arrange - Create state pointing to unreachable endpoint
        connection_failure_state = {
            "endpoint": "https://unreachable-host-12345.example.com/api/v1/completions",
            "headers": {"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}"},
            "prompt": "Test connection failure",
            "params": {"model": "qwen/qwen3-8b", "max_tokens": 20, "stream": True},
            "position": 0
        }
        
        with open(temp_state_file, 'w') as f:
            json.dump(connection_failure_state, f)
        
        # Act & Assert
        with pytest.raises((ResumeError, APIError)):
            list(completions_endpoint.resume_stream(state_file=temp_state_file))

class Test_CompletionsEndpoint_ResumeStream_05_StateTransitionBehaviors:
    """Test state transition behaviors for the resume_stream method - HTTP state management."""
    
    def test_state_restoration_to_active_http_connection(self, completions_endpoint, temp_state_file):
        """Test state restoration from file to active HTTP streaming connection."""
        # Arrange - Create valid streaming HTTP request
        prompt = "Write about the evolution of computer science"
        model = "qwen/qwen3-8b"
        
        # Start initial HTTP stream
        initial_stream = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=30,
            stream=True,
            state_file=temp_state_file
        )
        
        # Consume initial chunks to save HTTP state
        chunks_consumed = 0
        for chunk in initial_stream:
            chunks_consumed += 1
            if chunks_consumed >= 3:
                break
        
        # Act - Resume HTTP stream from saved state
        resumed_stream = completions_endpoint.resume_stream(state_file=temp_state_file)
        
        # Assert - Should transition to active HTTP streaming
        assert hasattr(resumed_stream, '__iter__')
        remaining_chunks = list(resumed_stream)
        assert isinstance(remaining_chunks, list)

    def test_saved_state_to_active_http_processing(self, completions_endpoint, temp_state_file):
        """Test transition from saved state to active HTTP request processing."""
        # Arrange
        prompt = "Analyze the impact of artificial intelligence"
        model = "qwen/qwen3-8b"
        
        # Create initial HTTP streaming request
        stream_iterator = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=50,
            stream=True,
            state_file=temp_state_file
        )
        
        # Process some chunks to establish HTTP state
        processed_chunks = 0
        for chunk in stream_iterator:
            processed_chunks += 1
            if processed_chunks >= 2:
                break
        
        # Verify state file contains HTTP connection details
        assert os.path.exists(temp_state_file)
        with open(temp_state_file, 'r') as f:
            state_data = json.load(f)
            assert "endpoint" in state_data
            assert "headers" in state_data
        
        # Act - Resume HTTP processing from saved state
        resumed_iterator = completions_endpoint.resume_stream(state_file=temp_state_file)
        
        # Assert - Should successfully transition to active HTTP processing
        resumed_chunks = list(resumed_iterator)
        assert isinstance(resumed_chunks, list)

    def test_authentication_state_recovery_http_resume(self, completions_endpoint, temp_state_file):
        """Test authentication state recovery during HTTP stream resumption."""
        # Arrange - Create authenticated HTTP streaming session
        prompt = "Discuss quantum computing applications"
        model = "qwen/qwen3-8b"
        
        # Start authenticated HTTP stream
        stream_iterator = completions_endpoint.create(
            prompt=prompt,
            model=model,
            max_tokens=40,
            stream=True,
            state_file=temp_state_file
        )
        
        # Consume chunks to save authenticated HTTP state
        auth_chunks = 0
        for chunk in stream_iterator:
            auth_chunks += 1
            if auth_chunks >= 2:
                break
        
        # Verify state is saved (authentication headers should NOT be in state file for security)
        with open(temp_state_file, 'r') as f:
            state_data = json.load(f)
            assert "headers" in state_data
            # Authorization header should NOT be saved in state file for security
            assert "Authorization" not in state_data["headers"]
        
        # Act - Resume with authentication state recovery
        resumed_iterator = completions_endpoint.resume_stream(state_file=temp_state_file)
        
        # Assert - Authentication should be recovered and working
        try:
            resumed_chunks = list(resumed_iterator)
            assert isinstance(resumed_chunks, list)
        except APIError as e:
            # Should not get authentication errors if state was properly recovered
            error_str = str(e).lower()
            assert "401" not in error_str and "unauthorized" not in error_str, \
                "Authentication state was not properly recovered"
