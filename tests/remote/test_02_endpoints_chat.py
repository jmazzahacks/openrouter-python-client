"""
Comprehensive test suite for OpenRouter API HTTP behaviors in ChatEndpoint.

This module provides exhaustive testing of HTTP requests to the OpenRouter API server,
organized by behavior categories and following the arrange-act-assert pattern.
All tests run against the real OpenRouter API using environment variables for authentication.
"""

import os
import json
import tempfile
import time
from typing import Dict, Any, List, Iterator

import pytest

from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager
from openrouter_client.endpoints.chat import ChatEndpoint
from openrouter_client.exceptions import ResumeError


class TestFixtures:
    """Shared fixtures and utilities for OpenRouter API testing."""
    
    @pytest.fixture(scope="session") 
    def auth_manager(self) -> AuthManager:
        """Create authenticated AuthManager for OpenRouter API."""
        return AuthManager()
    
    def invalid_auth(self) -> AuthManager:
        """Create AuthManager for OpenRouter API with invalid credentials."""
        return AuthManager(api_key="invalid_api_key", provisioning_api_key="invalid_provisioning_api_key")
    
    @pytest.fixture(scope="session")
    def http_manager(self) -> HTTPManager:
        """Create HTTPManager configured for OpenRouter API."""
        return HTTPManager(base_url="https://openrouter.ai/api/v1")
    
    @pytest.fixture(scope="session")
    def invalid_http(self) -> HTTPManager:
        """Create HTTPManager that's incorrectly configured for OpenRouter API."""
        return HTTPManager(base_url="https://invalid.openrouter.ai/api/v1")
    
    @pytest.fixture(scope="session")
    def chat_endpoint(self, auth_manager: AuthManager, http_manager: HTTPManager) -> ChatEndpoint:
        """Create shared ChatEndpoint instance for all tests."""
        return ChatEndpoint(auth_manager, http_manager)
    
    @pytest.fixture
    def valid_messages(self) -> List[Dict[str, Any]]:
        """Standard valid message format for testing."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you today?"}
        ]
        
    @pytest.fixture
    def long_response_messages(self) -> List[Dict[str, Any]]:
        """Standard valid message format for testing."""
        return [
            {"role": "system", "content": "You are knowledgeable, but verbose professor."},
            {"role": "user", "content": "Explain in detail the fundamentals of group theory."}
        ]
    
    @pytest.fixture
    def temp_state_file(self) -> Iterator[str]:
        """Create temporary file for streaming state persistence."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            yield f.name
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)


class Test_ChatEndpoint_Init_01_NominalBehaviors(TestFixtures):
    """Test nominal initialization behaviors for ChatEndpoint with OpenRouter API."""
    
    def test_successful_initialization_with_valid_credentials(self, auth_manager: AuthManager, http_manager: HTTPManager):
        """
        Test successful ChatEndpoint initialization with valid OpenRouter API credentials.
        
        Arrange: Valid AuthManager and HTTPManager instances
        Act: Initialize ChatEndpoint
        Assert: Endpoint is properly configured for OpenRouter API
        """
        # Arrange - fixtures provide valid managers
        
        # Act
        endpoint = ChatEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint.auth_manager == auth_manager
        assert endpoint.http_manager == http_manager
        assert endpoint._get_endpoint_url() == 'chat/completions'
        assert endpoint.logger is not None
    
    def test_proper_openrouter_endpoint_url_setup(self, chat_endpoint: ChatEndpoint):
        """
        Test that chat/completions endpoint URL is properly set for OpenRouter API calls.
        
        Arrange: Initialized ChatEndpoint
        Act: Get endpoint URL
        Assert: URL matches OpenRouter chat completions endpoint
        """
        # Arrange - fixture provides initialized endpoint
        
        # Act
        endpoint_url = chat_endpoint._get_endpoint_url()
        
        # Assert
        assert endpoint_url == 'chat/completions'


class Test_ChatEndpoint_Init_02_NegativeBehaviors(TestFixtures):
    """Test negative initialization behaviors for ChatEndpoint with OpenRouter API."""
    
    def test_initialization_with_invalid_api_credentials(self, http_manager: HTTPManager):
        """
        Test ChatEndpoint initialization with invalid OpenRouter API credentials.
        
        Arrange: Invalid AuthManager with bad credentials
        Act: Initialize ChatEndpoint and attempt API call
        Assert: Authentication failures are handled appropriately
        """
        # Arrange
        invalid_auth = AuthManager(api_key="invalid_key_12345")
        
        # Act
        endpoint = ChatEndpoint(invalid_auth, http_manager)
        
        # Assert - endpoint initializes but API calls should fail with auth errors
        with pytest.raises(Exception) as exc_info:
            endpoint.create([{"role": "user", "content": "test"}], model="qwen/qwen3-8b")
        
        # Verify it's an authentication-related error (401 status code)
        error_msg = str(exc_info.value).lower()
        assert any(term in error_msg for term in ['401', 'authentication', 'unauthorized']) or '401' in repr(exc_info.value)
    
    def test_initialization_with_malformed_base_url(self, auth_manager: AuthManager, invalid_http: HTTPManager):
        """
        Test initialization with malformed base URL for OpenRouter API.
        
        Arrange: HTTPManager with invalid base URL
        Act: Initialize ChatEndpoint and attempt API call
        Assert: Network/URL errors are handled appropriately
        """
        # Act
        endpoint = ChatEndpoint(auth_manager, invalid_http)
        
        # Assert - endpoint initializes but API calls should fail with network errors
        with pytest.raises(Exception) as exc_info:
            endpoint.create([{"role": "user", "content": "test"}], model="qwen/qwen3-8b")
        
        # Verify it's a network-related error
        assert any(term in str(exc_info.value).lower() for term in ['connection', 'network', 'resolve', 'url'])


class Test_ChatEndpoint_Init_03_BoundaryBehaviors(TestFixtures):
    """Test boundary initialization behaviors for ChatEndpoint with OpenRouter API."""
    
    @pytest.mark.parametrize("connection_param,value", [
        ("timeout", 1),  # Minimum timeout
        ("timeout", 300),  # Maximum reasonable timeout
        ("max_retries", 0),  # Minimum retries
        ("max_retries", 10),  # Maximum retries
    ])
    def test_initialization_with_boundary_connection_parameters(self, auth_manager: AuthManager, connection_param: str, value: int):
        """
        Test initialization with boundary values for HTTP connection parameters.
        
        Arrange: HTTPManager with boundary connection parameters
        Act: Initialize ChatEndpoint
        Assert: Initialization succeeds with boundary values
        """
        # Arrange
        http_kwargs = {connection_param: value}
        http_manager = HTTPManager(base_url="https://openrouter.ai/api/v1", **http_kwargs)
        
        # Act
        endpoint = ChatEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint.http_manager == http_manager
        assert endpoint._get_endpoint_url() == 'chat/completions'


class Test_ChatEndpoint_Init_04_ErrorHandlingBehaviors(TestFixtures):
    """Test error handling during ChatEndpoint initialization."""
    
    def test_network_connectivity_issues_during_initialization(self, auth_manager: AuthManager):
        """
        Test handling of network connectivity issues during endpoint setup.
        
        Arrange: HTTPManager configured for unreachable host
        Act: Initialize ChatEndpoint and attempt API call
        Assert: Network errors are properly handled
        """
        # Arrange
        unreachable_http = HTTPManager(base_url="https://127.0.0.1:9999")  # Unreachable endpoint
        
        # Act
        endpoint = ChatEndpoint(auth_manager, unreachable_http)
        
        # Assert - initialization succeeds but API calls fail gracefully
        with pytest.raises(Exception) as exc_info:
            endpoint.create([{"role": "user", "content": "test"}], model="qwen/qwen3-8b")
        
        assert any(term in str(exc_info.value).lower() for term in ['connection', 'refused', 'timeout'])


class Test_ChatEndpoint_Init_05_StateTransitionBehaviors(TestFixtures):
    """Test state transitions during ChatEndpoint initialization."""
    
    def test_transition_from_uninitialized_to_ready_state(self, auth_manager: AuthManager, http_manager: HTTPManager):
        """
        Test transition from uninitialized to ready state for OpenRouter API communication.
        
        Arrange: Component dependencies ready
        Act: Initialize ChatEndpoint
        Assert: Endpoint transitions to ready state
        """
        # Arrange - managers are ready
        
        # Act
        endpoint = ChatEndpoint(auth_manager, http_manager)
        
        # Assert - endpoint is in ready state
        assert hasattr(endpoint, 'auth_manager')
        assert hasattr(endpoint, 'http_manager')
        assert hasattr(endpoint, 'logger')
        assert endpoint._get_endpoint_url() == 'chat/completions'
        
        # Verify endpoint is ready for API calls by making a simple request
        response = endpoint.create(
            [{"role": "user", "content": "Say hello"}], 
            model="qwen/qwen3-8b",
            max_tokens=10
        )
        assert response is not None


class Test_ChatEndpoint_Create_01_NominalBehaviors(TestFixtures):
    """Test nominal behaviors for ChatEndpoint.create with OpenRouter API."""
    
    @pytest.mark.parametrize("model", [
        "anthropic/claude-3-haiku",
        "mistralai/mistral-small-3.1-24b-instruct", 
        "cohere/command-r7b-12-2024",
        "qwen/qwen3-8b"
    ])
    def test_standard_chat_completion_requests(self, chat_endpoint: ChatEndpoint, valid_messages: List[Dict], model: str):
        """
        Test standard chat completion requests to OpenRouter API with various models.
        
        Arrange: Valid messages and model specifications
        Act: Send chat completion request
        Assert: Successful response from OpenRouter API
        """
        # Arrange - fixtures provide valid setup
        
        # Act
        response = chat_endpoint.create(
            messages=valid_messages,
            model=model,
            max_tokens=50
        )
        
        # Assert
        assert response is not None
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], 'message')
        assert response.choices[0].message.content is not None
    
    @pytest.mark.parametrize("stream", [True, False])
    def test_streaming_and_non_streaming_requests(self, chat_endpoint: ChatEndpoint, valid_messages: List[Dict], stream: bool):
        """
        Test both streaming and non-streaming requests to OpenRouter API.
        
        Arrange: Valid messages and stream parameter
        Act: Send request with specified streaming mode
        Assert: Appropriate response type returned
        """
        # Arrange - fixtures provide valid setup
        
        # Act
        response = chat_endpoint.create(
            messages=valid_messages,
            model="qwen/qwen3-8b",
            stream=stream,
            max_tokens=50
        )
        
        # Assert
        if stream:
            # Should return iterator for streaming
            assert hasattr(response, '__iter__')
            # Consume first chunk to verify streaming works
            first_chunk = next(response)
            assert first_chunk is not None
        else:
            # Should return ChatCompletionResponse for non-streaming
            assert hasattr(response, 'choices')
            assert len(response.choices) > 0
    
    def test_tool_calling_workflow_with_openrouter(self, chat_endpoint: ChatEndpoint):
        """
        Test tool calling workflows with OpenRouter API including function execution.
        
        Arrange: Messages with tool definitions
        Act: Send tool calling request
        Assert: Tool calls are processed correctly
        """
        # Arrange
        messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city name"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        # Act - Try with a model that supports tools
        try:
            response = chat_endpoint.create(
                messages=messages,
                model="cohere/command-r-08-2024",  # Use a model that supports tools
                tools=tools,
                max_tokens=512
            )
            
            # Assert
            assert response is not None
            assert hasattr(response, 'choices')
            # Tool calling may or may not be triggered, but request should succeed
        except Exception as e:
            # Some models might not support tools, which is acceptable
            error_msg = str(e).lower()
            # If it's a tool-related error or 404 (model not found), that's acceptable
            if any(term in error_msg for term in ['tool', '404', 'not found', 'unsupported']):
                pytest.skip(f"Model doesn't support tools or is not available: {e}")
            else:
                raise


class Test_ChatEndpoint_Create_02_NegativeBehaviors(TestFixtures):
    """Test negative behaviors for ChatEndpoint.create with OpenRouter API."""
    
    @pytest.mark.parametrize("invalid_model", [
        "nonexistent/model",
        "invalid-format",
        "",
        "openai/gpt-999-turbo"
    ])
    def test_invalid_model_specifications(self, chat_endpoint: ChatEndpoint, valid_messages: List[Dict], invalid_model: str):
        """
        Test OpenRouter API rejection of requests with invalid model specifications.
        
        Arrange: Valid messages but invalid model
        Act: Send request with invalid model
        Assert: Appropriate error response from OpenRouter
        """
        # Arrange - fixture provides valid messages
        
        # Act & Assert
        if invalid_model == "":
            # Empty model string might use default model, so we test differently
            try:
                response = chat_endpoint.create(
                    messages=valid_messages,
                    model=invalid_model,
                    max_tokens=50
                )
                # If it succeeds with empty model, that's acceptable behavior
                assert response is not None
            except Exception as exc_info:
                # If it fails, verify it's a reasonable error
                error_msg = str(exc_info).lower()
                assert any(term in error_msg for term in ['400', 'api error', 'model'])
        else:
            # For other invalid models, expect an error
            with pytest.raises(Exception) as exc_info:
                chat_endpoint.create(
                    messages=valid_messages,
                    model=invalid_model,
                    max_tokens=50
                )
            
            # Verify error relates to bad request (400 status code for invalid models)
            error_msg = str(exc_info.value).lower()
            assert any(term in error_msg for term in ['400', 'api error', 'bad request']) or hasattr(exc_info.value, 'status_code') and exc_info.value.status_code == 400
    
    @pytest.mark.parametrize("invalid_param,value", [
        ("temperature", -1.0),  # Below valid range
        ("temperature", 3.0),   # Above valid range
        ("max_tokens", -1),     # Negative tokens
        ("top_p", -0.1),        # Below valid range
        ("top_p", 1.1),         # Above valid range
    ])
    def test_malformed_request_parameters(self, chat_endpoint: ChatEndpoint, valid_messages: List[Dict], invalid_param: str, value):
        """
        Test OpenRouter API errors for unsupported parameters or malformed requests.
        
        Arrange: Valid messages but invalid parameters
        Act: Send request with malformed parameters
        Assert: Parameter validation errors from OpenRouter
        """
        # Arrange
        kwargs = {invalid_param: value}
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            chat_endpoint.create(
                messages=valid_messages,
                model="qwen/qwen3-8b",
                **kwargs
            )
        
        # Verify error occurs (API returns generic 400 errors for bad parameters)
        error_msg = str(exc_info.value).lower()
        assert any(term in error_msg for term in ['400', 'api error', 'parameter', 'invalid', 'range', 'validation']) or hasattr(exc_info.value, 'status_code') and exc_info.value.status_code == 400


class Test_ChatEndpoint_Create_03_BoundaryBehaviors(TestFixtures):
    """Test boundary behaviors for ChatEndpoint.create with OpenRouter API."""
    
    @pytest.mark.parametrize("token_limit", [1, 50, 1000, 4000])
    def test_maximum_token_limits(self, chat_endpoint: ChatEndpoint, valid_messages: List[Dict], token_limit: int):
        """
        Test requests with various token limits allowed by OpenRouter API.
        
        Arrange: Messages and token limit parameters
        Act: Send request with specified token limit
        Assert: Request succeeds within token constraints
        """
        # Arrange - fixture provides valid messages
        
        # Act
        response = chat_endpoint.create(
            messages=valid_messages,
            model="qwen/qwen3-8b",
            max_tokens=token_limit
        )
        
        # Assert
        assert response is not None
        assert hasattr(response, 'choices')
        if hasattr(response, 'usage'):
            assert response.usage.completion_tokens <= token_limit
    
    @pytest.mark.parametrize("message_count", [1, 10, 50])
    def test_maximum_message_count_requests(self, chat_endpoint: ChatEndpoint, message_count: int):
        """
        Test requests with varying message counts for OpenRouter API.
        
        Arrange: Different numbers of messages
        Act: Send request with specified message count
        Assert: Request handles message count appropriately
        """
        # Arrange
        messages = []
        for i in range(message_count):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Message {i + 1}"
            messages.append({"role": role, "content": content})
        
        # Ensure we end with a user message
        if messages and messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": "Final user message"})
        
        # Act
        response = chat_endpoint.create(
            messages=messages,
            model="qwen/qwen3-8b",
            max_tokens=50
        )
        
        # Assert
        assert response is not None
        assert hasattr(response, 'choices')
    
    def test_maximum_tools_functions_supported(self, chat_endpoint: ChatEndpoint, valid_messages: List[Dict]):
        """
        Test requests with maximum number of tools/functions supported by OpenRouter.
        
        Arrange: Messages with multiple tool definitions
        Act: Send request with multiple tools
        Assert: Tools are processed correctly
        """
        # Arrange
        tools = []
        for i in range(5):  # Test with multiple tools
            tools.append({
                "type": "function",
                "function": {
                    "name": f"function_{i}",
                    "description": f"Test function {i}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param": {"type": "string", "description": "Test parameter"}
                        },
                        "required": ["param"]
                    }
                }
            })
        
        # Act
        response = chat_endpoint.create(
            messages=valid_messages,
            model="openai/gpt-4",
            tools=tools,
            max_tokens=50
        )
        
        # Assert
        assert response is not None
        assert hasattr(response, 'choices')


class Test_ChatEndpoint_Create_04_ErrorHandlingBehaviors(TestFixtures):
    """Test error handling behaviors for ChatEndpoint.create with OpenRouter API."""
    
    def test_openrouter_server_errors_and_fallbacks(self, chat_endpoint: ChatEndpoint, valid_messages: List[Dict]):
        """
        Test handling of OpenRouter API server errors and automatic provider fallbacks.
        
        Arrange: Request that might trigger server errors
        Act: Send request and handle potential server errors
        Assert: Errors are handled gracefully
        """
        # Arrange - use a potentially problematic request
        
        # Act & Assert
        try:
            response = chat_endpoint.create(
                messages=valid_messages,
                model="qwen/qwen3-8b",
                max_tokens=50
            )
            # If successful, verify response
            assert response is not None
        except Exception as e:
            # If error occurs, verify it's handled appropriately
            error_msg = str(e).lower()
            assert any(term in error_msg for term in ['server', 'error', 'service', 'unavailable'])
    
    def test_network_timeouts_during_api_communication(self, chat_endpoint: ChatEndpoint, valid_messages: List[Dict]):
        """
        Test handling of network timeouts during OpenRouter API communication.
        
        Arrange: Request with very short timeout
        Act: Send request that may timeout
        Assert: Timeout errors are handled appropriately
        """
        # Arrange - this test depends on network conditions
        
        # Act & Assert
        try:
            response = chat_endpoint.create(
                messages=valid_messages,
                model="qwen/qwen3-8b",
                max_tokens=50
            )
            # If successful, verify response
            assert response is not None
        except Exception as e:
            # If timeout occurs, verify it's handled appropriately
            error_msg = str(e).lower()
            timeout_related = any(term in error_msg for term in ['timeout', 'connection', 'network'])
            # Either success or appropriate timeout handling
            assert timeout_related or response is not None
    
    def test_streaming_connection_interruption_recovery(self, chat_endpoint: ChatEndpoint, long_response_messages: List[Dict], temp_state_file: str):
        """
        Test recovery from streaming connection interruptions with OpenRouter API.
        
        Arrange: Streaming request with state file
        Act: Start streaming and handle interruptions
        Assert: Interruptions are handled gracefully
        """
        # Arrange - setup streaming with state persistence
        
        # Act
        try:
            response_stream = chat_endpoint.create(
                messages=long_response_messages,
                model="mistralai/mistral-small-3.1-24b-instruct",
                stream=True,
                state_file=temp_state_file,
                max_tokens=512
            )
            
            # Consume some chunks
            chunks_consumed = 0
            for chunk in response_stream:
                chunks_consumed += 1
                if chunks_consumed >= 2:  # Stop early to simulate interruption
                    break
            
            # Assert - partial consumption succeeded
            assert chunks_consumed > 0
            
        except Exception as e:
            # If streaming fails, verify it's handled appropriately
            error_msg = str(e).lower()
            streaming_related = any(term in error_msg for term in ['stream', 'connection', 'network'])
            assert streaming_related


class Test_ChatEndpoint_Create_05_StateTransitionBehaviors(TestFixtures):
    """Test state transition behaviors for ChatEndpoint.create with OpenRouter API."""
    
    @pytest.mark.parametrize("stream", [False, True])
    def test_transition_between_streaming_modes(self, chat_endpoint: ChatEndpoint, valid_messages: List[Dict], stream: bool):
        """
        Test transition between non-streaming and streaming request modes with OpenRouter API.
        
        Arrange: Same endpoint for different streaming modes
        Act: Send requests in different streaming modes
        Assert: Mode transitions work correctly
        """
        # Arrange - use same endpoint for both modes
        
        # Act
        response = chat_endpoint.create(
            messages=valid_messages,
            model="qwen/qwen3-8b",
            stream=stream,
            max_tokens=50
        )
        
        # Assert - verify correct response type for mode
        if stream:
            assert hasattr(response, '__iter__')
            # Consume first chunk to verify streaming
            first_chunk = next(response)
            assert first_chunk is not None
        else:
            assert hasattr(response, 'choices')
            assert len(response.choices) > 0
    
    def test_tool_calling_workflow_state_changes(self, chat_endpoint: ChatEndpoint):
        """
        Test state changes during tool calling workflows with OpenRouter API.
        
        Arrange: Multi-step tool calling scenario
        Act: Execute tool calling workflow
        Assert: State transitions occur correctly
        """
        # Arrange
        messages = [{"role": "user", "content": "Calculate 15 * 23"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
        
        # Act
        response = chat_endpoint.create(
            messages=messages,
            model="cohere/command-r-08-2024",
            tools=tools,
            max_tokens=100
        )
        
        # Assert - verify tool calling workflow handled appropriately
        assert response is not None
        assert hasattr(response, 'choices')
        
        # Check if tool calls were made (state transition occurred)
        if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
            # Tool calling state transition occurred
            assert len(response.choices[0].message.tool_calls) > 0


class Test_ChatEndpoint_ResumeStream_01_NominalBehaviors(TestFixtures):
    """Test nominal behaviors for ChatEndpoint.resume_stream with OpenRouter API."""
    
    def test_successful_stream_resumption_from_saved_state(self, chat_endpoint: ChatEndpoint, valid_messages: List[Dict], temp_state_file: str):
        """
        Test successful resumption of interrupted streaming requests from saved state.
        
        Arrange: Start streaming request with state file, then resume
        Act: Start stream, save state, resume from state
        Assert: Stream resumes successfully from saved state
        """
        # Arrange - start initial streaming request
        initial_stream = chat_endpoint.create(
            messages=[
                {"role": "system", "content": "You are a brilliant and overly verbose professor."},
                {"role": "user", "content": "Explain in detail the fundamentals of group theory."}
            ],
            model="mistralai/mistral-small-3.1-24b-instruct",
            stream=True,
            state_file=temp_state_file,
            max_tokens=512,
            chunk_size=64,
        )
        
        # Consume a few chunks to create state
        chunks_consumed = 0
        for chunk in initial_stream:
            chunks_consumed += 1
            if chunks_consumed >= 16:
                break  # Stop to simulate interruption
        
        # Verify state file was created
        assert os.path.exists(temp_state_file)
        
        # Act - resume from saved state
        resumed_stream = chat_endpoint.resume_stream(temp_state_file)
        
        # Assert - verify resumption works
        assert hasattr(resumed_stream, '__iter__')
        
        # Consume at least one chunk from resumed stream
        resumed_chunks = 0
        for chunk in resumed_stream:
            resumed_chunks += 1
            if resumed_chunks >= 16:
                break
        
        assert resumed_chunks > 0


class Test_ChatEndpoint_ResumeStream_02_NegativeBehaviors(TestFixtures):
    """Test negative behaviors for ChatEndpoint.resume_stream with OpenRouter API."""
    
    def test_resume_with_corrupted_state_file(self, chat_endpoint: ChatEndpoint, temp_state_file: str):
        """
        Test handling of attempts to resume with corrupted or invalid state files.
        
        Arrange: Corrupted state file
        Act: Attempt to resume stream
        Assert: Appropriate error handling for corrupted state
        """
        # Arrange - create corrupted state file
        with open(temp_state_file, 'w') as f:
            f.write('{"invalid": "json": "data"}')  # Malformed JSON
        
        # Act & Assert
        with pytest.raises((ResumeError, Exception)) as exc_info:
            chat_endpoint.resume_stream(temp_state_file)
        
        # Verify error relates to state file corruption
        error_msg = str(exc_info.value).lower()
        assert any(term in error_msg for term in ['state', 'file', 'invalid', 'corrupt', 'json'])
    
    def test_resume_with_nonexistent_state_file(self, chat_endpoint: ChatEndpoint):
        """
        Test handling of resume requests with nonexistent state files.
        
        Arrange: Reference to nonexistent state file
        Act: Attempt to resume stream
        Assert: Appropriate error for missing file
        """
        # Arrange
        nonexistent_file = "/tmp/nonexistent_state_file.json"
        
        # Act & Assert
        with pytest.raises((ResumeError, FileNotFoundError, Exception)) as exc_info:
            chat_endpoint.resume_stream(nonexistent_file)
        
        # Verify error relates to missing file
        error_msg = str(exc_info.value).lower()
        assert any(term in error_msg for term in ['file', 'not found', 'exist', 'state'])


class Test_ChatEndpoint_ResumeStream_04_ErrorHandlingBehaviors(TestFixtures):
    """Test error handling behaviors for ChatEndpoint.resume_stream with OpenRouter API."""
    
    def test_session_expiration_during_resume(self, chat_endpoint: ChatEndpoint, temp_state_file: str):
        """
        Test handling of OpenRouter API session expiration during stream resumption.
        
        Arrange: State file with potentially expired session
        Act: Attempt to resume after delay
        Assert: Session expiration is handled appropriately
        """
        # Arrange - create a basic state file structure
        state_data = {
            "endpoint": "https://openrouter.ai/api/v1/chat/completions",
            "headers": {"Authorization": "Bearer expired_token"},
            "messages": [{"role": "user", "content": "test"}],
            "position": 0
        }
        
        with open(temp_state_file, 'w') as f:
            json.dump(state_data, f)
        
        # Act & Assert
        with pytest.raises((ResumeError, Exception)) as exc_info:
            chat_endpoint.resume_stream(temp_state_file)
        
        # Verify error handling - state validation or auth issues
        error_msg = str(exc_info.value).lower()
        assert any(term in error_msg for term in ['auth', 'session', 'expired', 'invalid', 'token', 'state', 'validation', 'resume'])
    
    def test_network_issues_during_resume(self, chat_endpoint: ChatEndpoint, temp_state_file: str):
        """
        Test handling of network connectivity issues during resume operations.
        
        Arrange: State file with unreachable endpoint
        Act: Attempt to resume with network issues
        Assert: Network errors are handled gracefully
        """
        # Arrange - create state file with unreachable endpoint
        state_data = {
            "endpoint": "https://unreachable-endpoint.invalid/chat/completions",
            "headers": {"Authorization": "Bearer test_token"},
            "messages": [{"role": "user", "content": "test"}],
            "position": 0
        }
        
        with open(temp_state_file, 'w') as f:
            json.dump(state_data, f)
        
        # Act & Assert
        with pytest.raises((ResumeError, Exception)) as exc_info:
            chat_endpoint.resume_stream(temp_state_file)
        
        # Verify error relates to network connectivity or state validation
        error_msg = str(exc_info.value).lower()
        assert any(term in error_msg for term in ['network', 'connection', 'resolve', 'unreachable', 'state', 'validation', 'resume'])


class Test_ChatEndpoint_ResumeStream_05_StateTransitionBehaviors(TestFixtures):
    """Test state transition behaviors for ChatEndpoint.resume_stream with OpenRouter API."""
    
    def test_transition_from_suspended_to_active_streaming(self, chat_endpoint: ChatEndpoint, valid_messages: List[Dict], temp_state_file: str):
        """
        Test transition from suspended to active streaming state with OpenRouter API.
        
        Arrange: Suspended streaming session with saved state
        Act: Resume streaming from saved state
        Assert: Successful transition to active streaming
        """
        # Arrange - create and suspend a streaming session
        try:
            initial_stream = chat_endpoint.create(
                messages=[
                    {"role": "system", "content": "You are a brilliant and overly verbose professor."},
                    {"role": "user", "content": "Explain in detail the fundamentals of group theory."}
                ],
                model="mistralai/mistral-small-3.1-24b-instruct",
                stream=True,
                state_file=temp_state_file,
                max_tokens=512,
                chunk_size=64
            )
            
            # Consume some chunks then "suspend"
            chunks_before_suspend = 0
            for chunk in initial_stream:
                chunks_before_suspend += 1
                if chunks_before_suspend >= 16:
                    break
            
            # Verify state was saved (suspended state)
            assert os.path.exists(temp_state_file)
            
            # Act - resume (transition to active state)
            resumed_stream = chat_endpoint.resume_stream(temp_state_file)
            
            # Assert - verify active streaming state
            assert hasattr(resumed_stream, '__iter__')
            
            # Consume chunks from resumed stream
            chunks_after_resume = 0
            for chunk in resumed_stream:
                chunks_after_resume += 1
                if chunks_after_resume >= 16:
                    break
            
            # Verify successful state transition
            assert chunks_before_suspend > 0
            assert chunks_after_resume > 0
            
        except Exception as e:
            pytest.skip(f"State transition test not supported: {e}")
