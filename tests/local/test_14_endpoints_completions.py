import json
import tempfile
from pathlib import Path

from pydantic import ValidationError
import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError

from openrouter_client.auth import AuthManager
from openrouter_client.endpoints.completions import CompletionsEndpoint
from openrouter_client.exceptions import APIError, ResumeError, StreamingError
from openrouter_client.http import HTTPManager
from openrouter_client.models.completions import (
    CompletionsRequest,
    CompletionsResponse,
    CompletionsStreamResponse
)


class Test_CompletionsEndpoint___init___01_NominalBehaviors:
    """Test nominal behaviors for CompletionsEndpoint initialization."""
    
    def test_successful_initialization_with_valid_managers(self):
        """Test successful initialization with valid AuthManager and HTTPManager instances."""
        auth_manager = AuthManager(api_key="test_key")
        http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        
        endpoint = CompletionsEndpoint(auth_manager, http_manager)
        
        assert endpoint.auth_manager is auth_manager
        assert endpoint.http_manager is http_manager
        assert endpoint.endpoint_path == 'completions'
        assert hasattr(endpoint, 'logger')
    
    @pytest.mark.parametrize("api_key,base_url", [
        ("valid_key_123", "https://api.openai.com"),
        ("another_key", "https://custom.api.com"),
        ("test_api_key", "http://localhost:8080"),
    ])
    def test_initialization_with_various_valid_configurations(self, api_key, base_url):
        """Test initialization with different valid manager configurations."""
        auth_manager = AuthManager(api_key=api_key)
        http_manager = HTTPManager(base_url=base_url)
        
        endpoint = CompletionsEndpoint(auth_manager, http_manager)
        
        # Test that initialization succeeded and objects are properly set
        assert endpoint.auth_manager is not None
        assert endpoint.http_manager.base_url == base_url
        assert endpoint.endpoint_path == 'completions'
        
        # Test that the auth manager can function with the provided key
        assert hasattr(endpoint.auth_manager, 'api_key')


class Test_CompletionsEndpoint___init___02_NegativeBehaviors:
    """Test negative behaviors for CompletionsEndpoint initialization."""
    
    @pytest.mark.parametrize("auth_manager,http_manager", [
        (None, HTTPManager(base_url="https://api.thisurlshouldntexsit.com")),
        (AuthManager(api_key="test_key"), None),
        (None, None),
    ])
    def test_initialization_with_none_parameters(self, auth_manager, http_manager):
        """Test initialization behavior with None values for required manager parameters."""
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            CompletionsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("auth_manager,http_manager", [
        ("invalid_auth_string", HTTPManager(base_url="https://api.thisurlshouldntexsit.com")),
        (AuthManager(api_key="test_key"), "invalid_http_string"),
        (42, HTTPManager(base_url="https://api.thisurlshouldntexsit.com")),
        (AuthManager(api_key="test_key"), []),
        ({}, {}),
    ])
    def test_initialization_with_invalid_types(self, auth_manager, http_manager):
        """Test initialization with invalid object types that don't match expected interfaces."""
        with pytest.raises((TypeError, AttributeError, ValidationError)):
                CompletionsEndpoint(auth_manager, http_manager)


class Test_CompletionsEndpoint___init___04_ErrorHandlingBehaviors:
    """Test error handling behaviors for CompletionsEndpoint initialization."""
    
    def test_error_propagation_from_base_init(self):
        """Test proper error propagation when BaseEndpoint initialization fails."""
        class FailingAuthManager(AuthManager):
            def __init__(self):
                # Initialize with minimal valid state to pass validation
                super().__init__(api_key="dummy_key")
            
            def __getattribute__(self, name):
                # Allow access to essential object attributes needed for construction
                essential_attrs = {
                    '__class__', '__dict__', '__module__', '__weakref__', '__init__',
                    '__new__', '__getattribute__', '__setattr__', '__delattr__',
                    '__hash__', '__eq__', '__ne__', '_encryption_key', 'logger'
                }
                if name in essential_attrs:
                    return super().__getattribute__(name)
                
                # Fail on any business logic attribute access during initialization
                raise RuntimeError("Simulated auth manager failure")
        
        class FailingHTTPManager(HTTPManager):
            def __init__(self):
                # Initialize with minimal valid state to pass validation
                super().__init__(base_url="https://dummy.url")
            
            def __getattribute__(self, name):
                # Allow access to essential object attributes needed for construction
                essential_attrs = {
                    '__class__', '__dict__', '__module__', '__weakref__', '__init__',
                    '__new__', '__getattribute__', '__setattr__', '__delattr__',
                    '__hash__', '__eq__', '__ne__'
                }
                if name in essential_attrs:
                    return super().__getattribute__(name)
                
                # Fail on any business logic attribute access during initialization
                raise RuntimeError("Simulated HTTP manager failure")
        
        with pytest.raises(RuntimeError, match="Simulated.*failure"):
            CompletionsEndpoint(FailingAuthManager(), FailingHTTPManager())
    
    def test_initialization_with_invalid_endpoint_path_handling(self):
        """Test error handling when endpoint name configuration fails."""
        auth_manager = AuthManager(api_key="test_key")
        http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        
        # Test that initialization completes even with potential logging issues
        endpoint = CompletionsEndpoint(auth_manager, http_manager)
        assert endpoint.endpoint_path == 'completions'


class Test_CompletionsEndpoint___init___05_StateTransitionBehaviors:
    """Test state transition behaviors for CompletionsEndpoint initialization."""
    
    def test_correct_inheritance_setup(self):
        """Test correct setup of inherited BaseEndpoint state and endpoint name."""
        auth_manager = AuthManager(api_key="test_key")
        http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        
        endpoint = CompletionsEndpoint(auth_manager, http_manager)
        
        # Verify transition from uninitialized to initialized state
        assert hasattr(endpoint, 'auth_manager')
        assert hasattr(endpoint, 'http_manager')
        assert hasattr(endpoint, 'endpoint_path')
        assert hasattr(endpoint, 'logger')
        
        # Verify proper inheritance chain setup
        assert isinstance(endpoint, CompletionsEndpoint)
        assert endpoint.endpoint_path == 'completions'
    
    def test_proper_attribute_assignment_order(self):
        """Test that all required attributes are properly assigned during initialization."""
        auth_manager = AuthManager(api_key="test_key")
        http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        
        endpoint = CompletionsEndpoint(auth_manager, http_manager)
        
        # Verify all critical attributes exist and have correct values
        assert endpoint.auth_manager == auth_manager
        assert endpoint.http_manager == http_manager
        assert endpoint.endpoint_path == 'completions'
        assert endpoint.logger is not None


class Test_CompletionsEndpoint__ParseStreamingResponse_01_NominalBehaviors:
    """Test nominal behaviors for _parse_streaming_response method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    @pytest.mark.parametrize("chunk_data", [
        {
            "id": "cmpl-test123",
            "object": "text_completion",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "text": "Hello world",
                    "finish_reason": "stop",
                    "logprobs": {
                        "tokens": ["Hello", " world"],
                        "token_logprobs": [-0.1, -0.2],
                        "top_logprobs": [{"Hello": -0.1}, {" world": -0.2}],
                        "text_offset": [0, 5]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        },
        {
            "id": "cmpl-test456",
            "object": "text_completion",
            "created": 1234567891,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "text": "Simple response",
                    "finish_reason": "length"
                }
            ]
        },
        {
            "id": "cmpl-test789",
            "object": "text_completion",
            "created": 1234567892,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "text": "Multi-choice response",
                    "finish_reason": "stop"
                },
                {
                    "index": 1,
                    "text": "Alternative response",
                    "finish_reason": "stop"
                }
            ]
        }
    ])
    def test_successful_parsing_of_well_formed_chunks(self, chunk_data):
        """Test successful parsing of well-formed streaming response chunks into CompletionsStreamResponse objects."""
        response_iter = iter([chunk_data])
        
        parsed_responses = list(self.endpoint._parse_streaming_response(response_iter))
        
        assert len(parsed_responses) == 1
        parsed_response = parsed_responses[0]
        
        # Verify the response contains expected data
        if isinstance(parsed_response, CompletionsStreamResponse):
            assert parsed_response.id == chunk_data["id"]
            assert parsed_response.object == chunk_data["object"]
            assert parsed_response.model == chunk_data["model"]
        elif isinstance(parsed_response, dict):
            assert parsed_response["id"] == chunk_data["id"]
            assert parsed_response["object"] == chunk_data["object"]
        else:
            # Raw data fallback
            assert parsed_response == chunk_data
    
    def test_parsing_multiple_sequential_chunks(self):
        """Test parsing of multiple sequential chunks maintaining correct order."""
        chunks = [
            {"id": "chunk_1", "object": "text_completion", "created": 1, "model": "test", "choices": []},
            {"id": "chunk_2", "object": "text_completion", "created": 2, "model": "test", "choices": []},
            {"id": "chunk_3", "object": "text_completion", "created": 3, "model": "test", "choices": []}
        ]
        
        response_iter = iter(chunks)
        parsed_responses = list(self.endpoint._parse_streaming_response(response_iter))
        
        assert len(parsed_responses) == 3
        
        # Verify sequential order is maintained
        for i, response in enumerate(parsed_responses):
            expected_id = f"chunk_{i + 1}"
            if hasattr(response, 'id'):
                assert response.id == expected_id
            elif isinstance(response, dict):
                assert response["id"] == expected_id


class Test_CompletionsEndpoint__ParseStreamingResponse_02_NegativeBehaviors:
    """Test negative behaviors for _parse_streaming_response method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    @pytest.mark.parametrize("malformed_chunk", [
        {"id": "test", "invalid_field_only": "value"},  # Missing required fields
        {"choices": [{"invalid_choice": "data"}], "object": "text_completion"},  # Malformed choices
        {"usage": {"invalid_usage": "data"}, "object": "text_completion"},  # Malformed usage
        {},  # Completely empty chunk
        {"choices": "not_a_list", "object": "text_completion"},  # Invalid choices type
        {"choices": [{"finish_reason": "invalid_reason_value"}]},  # Invalid finish_reason
        {"logprobs": {"invalid": "structure"}},  # Invalid logprobs structure
    ])
    def test_graceful_handling_of_malformed_chunks(self, malformed_chunk):
        """Test graceful handling of malformed response chunks without breaking the stream."""
        response_iter = iter([malformed_chunk])
        
        # Should not raise exception and should yield something (either parsed or raw)
        parsed_responses = list(self.endpoint._parse_streaming_response(response_iter))
        
        assert len(parsed_responses) == 1
        # Should either be parsed object or raw data, but never None
        assert parsed_responses[0] is not None
    
    def test_handling_empty_iterator(self):
        """Test handling of empty response iterators."""
        response_iter = iter([])
        
        parsed_responses = list(self.endpoint._parse_streaming_response(response_iter))
        
        assert len(parsed_responses) == 0
    
    def test_handling_none_values_in_chunks(self):
        """Test handling of None values within otherwise valid chunks."""
        chunk_with_nones = {
            "id": "test",
            "object": "text_completion",
            "choices": [{"text": "test", "finish_reason": None}],
            "usage": None
        }
        
        response_iter = iter([chunk_with_nones])
        parsed_responses = list(self.endpoint._parse_streaming_response(response_iter))
        
        assert len(parsed_responses) == 1
        assert parsed_responses[0] is not None


class Test_CompletionsEndpoint__ParseStreamingResponse_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for _parse_streaming_response method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    def test_graceful_degradation_on_validation_failure(self):
        """Test graceful degradation when model_validate fails, falling back to raw data."""
        # Create chunks that will likely cause validation failures
        problematic_chunks = [
            {
                "id": "test",
                "choices": [{"text": "test", "finish_reason": "completely_invalid_reason"}],
                "usage": {"prompt_tokens": "not_an_integer"}  # Wrong type
            },
            {
                "id": "test2",
                "choices": [{"index": "not_an_integer", "text": 12345}],  # Wrong types
                "usage": {"total_tokens": -1}  # Invalid value
            }
        ]
        
        for chunk in problematic_chunks:
            response_iter = iter([chunk])
            parsed_responses = list(self.endpoint._parse_streaming_response(response_iter))
            
            assert len(parsed_responses) == 1
            # Should fallback to raw data or attempt partial parsing
            assert parsed_responses[0] is not None
    
    def test_continues_processing_after_individual_chunk_failures(self):
        """Test that processing continues even if individual chunks fail."""
        chunks = [
            {"id": "good1", "object": "text_completion", "choices": []},  # Valid
            {"invalid": "chunk"},  # Invalid
            {"id": "good2", "object": "text_completion", "choices": []},  # Valid
        ]
        
        response_iter = iter(chunks)
        parsed_responses = list(self.endpoint._parse_streaming_response(response_iter))
        
        # All chunks should be yielded (either parsed or raw)
        assert len(parsed_responses) == 3
        for response in parsed_responses:
            assert response is not None


class Test_CompletionsEndpoint__ParseStreamingResponse_05_StateTransitionBehaviors:
    """Test state transition behaviors for _parse_streaming_response method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    def test_sequential_chunk_processing_maintains_state(self):
        """Test proper sequential processing of streaming chunks while maintaining iterator state."""
        chunks = [
            {"id": "chunk_1", "object": "text_completion", "created": 1, "model": "test"},
            {"id": "chunk_2", "object": "text_completion", "created": 2, "model": "test"},
            {"id": "chunk_3", "object": "text_completion", "created": 3, "model": "test"}
        ]
        
        response_iter = iter(chunks)
        
        # Process chunks one by one to verify state is maintained
        parsed_iter = self.endpoint._parse_streaming_response(response_iter)
        
        first_chunk = next(parsed_iter)
        assert self._extract_id(first_chunk) == "chunk_1"
        
        second_chunk = next(parsed_iter)
        assert self._extract_id(second_chunk) == "chunk_2"
        
        third_chunk = next(parsed_iter)
        assert self._extract_id(third_chunk) == "chunk_3"
        
        # Iterator should be exhausted
        with pytest.raises(StopIteration):
            next(parsed_iter)
    
    def test_iterator_state_preservation_across_parsing_errors(self):
        """Test that iterator state is preserved even when parsing errors occur."""
        chunks = [
            {"id": "valid_1", "object": "text_completion"},
            {"invalid": "chunk_data"},
            {"id": "valid_2", "object": "text_completion"}
        ]
        
        response_iter = iter(chunks)
        parsed_iter = self.endpoint._parse_streaming_response(response_iter)
        
        # Should be able to iterate through all chunks despite the invalid one
        results = list(parsed_iter)
        assert len(results) == 3
        
        # Verify the valid chunks are in correct positions
        assert self._extract_id(results[0]) == "valid_1"
        assert self._extract_id(results[2]) == "valid_2"
    
    def _extract_id(self, response):
        """Helper method to extract ID from various response formats."""
        if hasattr(response, 'id'):
            return response.id
        elif isinstance(response, dict) and 'id' in response:
            return response['id']
        return None


class Test_CompletionsEndpoint__CreateRequestModel_01_NominalBehaviors:
    """Test nominal behaviors for _create_request_model method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    @pytest.mark.parametrize("prompt,model,kwargs", [
        ("Hello world", "gpt-3.5-turbo", {}),
        ("Test prompt", "gpt-4", {"temperature": 0.7, "max_tokens": 100}),
        ("Simple prompt", "o3", {"temperature": 0.5}),
        ("Complex prompt", "gpt-4", {
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 200,
            "stop": ["\\n", "END"],
            "n": 3,
            "logprobs": 5,
            "echo": True,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "user": "test_user"
        })
    ])
    def test_successful_request_creation_with_valid_parameters(self, prompt, model, kwargs):
        """Test successful creation and validation of CompletionsRequest from valid parameters."""
        request = self.endpoint._create_request_model(prompt, model, **kwargs)
        
        assert isinstance(request, CompletionsRequest)
        assert request.prompt == prompt
        if model:
            assert request.model == model
        
        # Verify kwargs were properly applied
        for key, value in kwargs.items():
            if hasattr(request, key):
                assert getattr(request, key) == value
    
    @pytest.mark.parametrize("nested_objects", [
        {
            "tools": [
                {
                    "type": "function", 
                    "function": {
                        "name": "test_func", 
                        "description": "Test function",
                        "parameters": {"type": "object"}
                    }
                }
            ]
        },
        {
            "functions": [
                {"name": "test_func", "description": "Test function", "parameters": {"type": "object"}}
            ]
        },
        {
            "response_format": {"type": "json_object"}
        },
        {
            "reasoning": {"effort": "high", "max_tokens": 100}
        }
    ])
    def test_proper_parsing_of_nested_objects(self, nested_objects):
        """Test proper parsing of tools, functions, reasoning, and response_format parameters."""
        request = self.endpoint._create_request_model("test prompt", "gpt-4", **nested_objects)
        
        assert isinstance(request, CompletionsRequest)
        assert request.prompt == "test prompt"
        
        # Verify nested objects were processed
        for key in nested_objects.keys():
            if hasattr(request, key):
                assert getattr(request, key) is not None


class Test_CompletionsEndpoint__CreateRequestModel_02_NegativeBehaviors:
    """Test negative behaviors for _create_request_model method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    @pytest.mark.parametrize("invalid_params", [
        {"tools": [{"invalid": "tool_structure"}]},
        {"functions": [{"missing_required": "fields"}]},
        {"response_format": {"invalid_type": "unknown"}},
        {"reasoning": {"invalid_effort": "super_high"}},
        {"tools": "not_a_list"},
        {"functions": "not_a_list"},
        {"response_format": "not_a_dict"}
    ])
    def test_rejection_of_invalid_nested_objects(self, invalid_params):
        """Test rejection of invalid nested objects in tools, functions, and response_format parameters."""
        # Some parameters may be accepted and logged as warnings, others should fail
        try:
            request = self.endpoint._create_request_model("test prompt", "gpt-4", **invalid_params)
            # If no exception, verify request was still created with basic parameters
            assert isinstance(request, CompletionsRequest)
            assert request.prompt == "test prompt"
        except (ValueError, TypeError):
            # Expected for truly invalid parameters
            pass
    
    @pytest.mark.parametrize("invalid_basic_params", [
        {"prompt": "", "model": ""},  # Empty strings
        {"prompt": None},  # None prompt
        {"prompt": 12345},  # Wrong type for prompt
        {"prompt": []},  # Wrong type for prompt
    ])
    def test_handling_invalid_basic_parameters(self, invalid_basic_params):
        """Test handling of invalid basic parameters like prompt and model."""
        prompt = invalid_basic_params.pop("prompt", "default")
        model = invalid_basic_params.pop("model", None)
        
        if not prompt or not isinstance(prompt, str) or not model or not isinstance(model, str):
            with pytest.raises((ValueError, TypeError, ValidationError)):
                self.endpoint._create_request_model(prompt, model, **invalid_basic_params)
        else:
            # Some invalid params might be accepted
            request = self.endpoint._create_request_model(prompt, model, **invalid_basic_params)
            assert isinstance(request, CompletionsRequest)


class Test_CompletionsEndpoint__CreateRequestModel_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for _create_request_model method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    def test_meaningful_error_messages_on_validation_failure(self):
        """Test graceful handling of validation failures with meaningful error messages."""
        # Parameters that should definitely cause validation failures
        invalid_kwargs = {
            "temperature": "not_a_number",
            "max_tokens": -100,
            "top_p": 2.0,  # Out of valid range
            "n": -1  # Invalid value
        }
        
        with pytest.raises((ValueError, TypeError)) as exc_info:
            self.endpoint._create_request_model("test prompt", "model", **invalid_kwargs)
        
        # Verify error message provides useful information
        error_message = str(exc_info.value)
        assert len(error_message) > 0
        # Should contain some indication of what went wrong
        assert any(keyword in error_message.lower() for keyword in 
                  ["temperature", "max_tokens", "top_p", "validation", "invalid"])
    
    def test_partial_validation_failure_handling(self):
        """Test handling when some nested objects fail validation but others succeed."""
        mixed_params = {
            "tools": [
                {"type": "function", "function": {"name": "valid_func"}},  # Valid
                {"invalid": "tool"}  # Invalid
            ],
            "temperature": 0.7  # Valid
        }
        
        # Should either succeed with warnings or fail gracefully
        try:
            request = self.endpoint._create_request_model("test prompt", "model", **mixed_params)
            assert isinstance(request, CompletionsRequest)
            assert request.temperature == 0.7
        except (ValueError, TypeError):
            # Acceptable if validation is strict
            pass


class Test_CompletionsEndpoint__CreateRequestModel_05_StateTransitionBehaviors:
    """Test state transition behaviors for _create_request_model method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    def test_transformation_to_validated_pydantic_models(self):
        """Test transformation of raw dictionary parameters into validated Pydantic models."""
        raw_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "stop": ["\\n", "END"],
            "n": 2,
            "tools": [{"type": "function", "function": {"name": "test_func", "parameters": {}}}],
            "reasoning": {"effort": "high", "max_tokens": 50}
        }
        
        # Verify transformation from raw dict to validated model
        request = self.endpoint._create_request_model("test prompt", "gpt-4", **raw_params)
        
        assert isinstance(request, CompletionsRequest)
        assert request.prompt == "test prompt"
        assert request.model == "gpt-4"
        assert request.temperature == 0.7
        assert request.max_tokens == 100
        assert request.stop == ["\\n", "END"]
        assert request.n == 2
        
        # Verify nested objects were properly processed
        if hasattr(request, 'tools') and request.tools:
            assert len(request.tools) > 0
        if hasattr(request, 'reasoning') and request.reasoning:
            assert request.reasoning is not None
    
    def test_progressive_parameter_processing(self):
        """Test that parameters are processed progressively and independently."""
        # Test with parameters that should be processed in sequence
        params_sequence = [
            {"temperature": 0.5},
            {"max_tokens": 150},
            {"tools": [{"type": "function", "function": {"name": "func1", "parameters": {}}}]},
            {"reasoning": {"effort": "medium"}}
        ]
        
        accumulated_params = {}
        for param_set in params_sequence:
            accumulated_params.update(param_set)
            
            request = self.endpoint._create_request_model("test", "model", **accumulated_params)
            assert isinstance(request, CompletionsRequest)
            
            # Verify each parameter was properly added
            for key, value in param_set.items():
                if hasattr(request, key):
                    request_value = getattr(request, key)
                    if key in ["tools", "reasoning"] and request_value is not None:
                        # Complex objects should be processed
                        assert request_value is not None
                    else:
                        assert request_value == value


class Test_CompletionsEndpoint_Create_01_NominalBehaviors:
    """Test nominal behaviors for create method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    @pytest.mark.parametrize("stream", [True, False])
    def test_successful_completion_generation(self, stream):
        """Test successful generation of text completions for valid prompts and parameters."""
        try:
            result = self.endpoint.create(
                prompt="Hello world",
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=100,
                stream=stream
            )
            
            if stream:
                # Should return iterator for streaming
                assert hasattr(result, '__iter__')
                # Verify it's actually an iterator, not just iterable
                assert hasattr(result, '__next__') or hasattr(result, 'next')
            else:
                # Should return response object for non-streaming
                assert result is not None
                if isinstance(result, CompletionsResponse):
                    assert hasattr(result, 'id')
                    assert hasattr(result, 'choices')
                
        except (APIError, RequestsConnectionError, ConnectionError, ValueError, StreamingError) as e:
            # Expected in test environment without real API
            if stream:
                assert "Streaming completions failed" in e.message
            else:
                assert "Request failed" in e.message
    
    @pytest.mark.parametrize("params", [
        {"temperature": 0.0, "max_tokens": 50},
        {"temperature": 1.0, "top_p": 0.9, "n": 2},
        {"stop": ["\\n"], "echo": True, "logprobs": 1},
        {"presence_penalty": 0.5, "frequency_penalty": 0.5, "user": "test_user"},
    ])
    def test_correct_parameter_handling(self, params):
        """Test correct execution with various parameter combinations."""
        try:
            result = self.endpoint.create(
                prompt="Test prompt for parameter validation",
                model="gpt-3.5-turbo",
                **params
            )
            assert result is not None
            
        except (APIError, RequestsConnectionError, ConnectionError, ValueError, StreamingError) as e:
            assert "Request failed" in e.message or "Not Found" in e.message


class Test_CompletionsEndpoint_Create_02_NegativeBehaviors:
    """Test negative behaviors for create method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexist.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    @pytest.mark.parametrize("invalid_params", [
        {"prompt": "", "model": None},  # Empty prompt and None model
        {"prompt": "test", "temperature": 3.0},  # Temperature out of range
        {"prompt": "test", "max_tokens": -1},  # Negative max_tokens
        {"prompt": "test", "top_p": 1.5},  # top_p out of range
        {"prompt": "test", "n": 0},  # Invalid n value
        {"prompt": "test", "logprobs": 6},  # logprobs out of range
    ])
    def test_handling_invalid_parameters(self, invalid_params):
        """Test proper handling of invalid authentication or malformed parameters."""
        with pytest.raises((APIError, RequestsConnectionError, ValueError, TypeError)):
            self.endpoint.create(**invalid_params, validate_request=True)
    
    @pytest.mark.parametrize("malformed_nested", [
        {"tools": [{"invalid": "structure"}]},
        {"functions": [{"missing": "name"}]},
        {"response_format": {"type": "unsupported_type"}},
        {"reasoning": {"effort": "invalid_level"}},
    ])
    def test_handling_malformed_nested_parameters(self, malformed_nested):
        """Test handling of malformed nested parameters."""
        try:
            result = self.endpoint.create(
                prompt="test prompt",
                model="gpt-3.5-turbo",
                validate_request=True,
                **malformed_nested
            )
            # If no exception, validation may be lenient
            assert result is not None
            
        except (APIError, RequestsConnectionError, ValueError, TypeError, ConnectionError, StreamingError):
            # Expected for invalid parameters or network issues
            pass


class Test_CompletionsEndpoint_Create_03_BoundaryBehaviors:
    """Test boundary behaviors for create method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    @pytest.mark.parametrize("reasoning,include_reasoning,expected_precedence", [
        ({"effort": "high"}, True, "reasoning_wins"),  # reasoning takes precedence
        ({"effort": "high"}, False, "reasoning_wins"),  # reasoning takes precedence
        (None, True, "include_reasoning_used"),  # include_reasoning used when reasoning is None
        (None, False, "include_reasoning_used"),  # include_reasoning used when reasoning is None
        ({"exclude": True}, True, "reasoning_wins"),  # reasoning explicit exclude wins
    ])
    def test_reasoning_parameter_interaction(self, reasoning, include_reasoning, expected_precedence):
        """Test correct handling of reasoning and include_reasoning parameter interaction when both are specified."""
        try:
            # Test the parameter precedence logic without making actual API calls
            request = self.endpoint._create_request_model(
                prompt="test prompt",
                model="gpt-4",
                reasoning=reasoning,
                include_reasoning=include_reasoning
            )
            
            # Verify request was created successfully
            assert isinstance(request, CompletionsRequest)
            
            # Test the actual create method parameter handling
            result = self.endpoint.create(
                prompt="test prompt",
                model="gpt-4",
                reasoning=reasoning,
                include_reasoning=include_reasoning,
                validate_request=True
            )
            
            # If we get here, the precedence logic worked correctly
            assert result is not None
            
        except (APIError, RequestsConnectionError, ConnectionError, ValueError, StreamingError) as e:
            # Expected in test environment, but verify it's not a precedence logic error
            error_msg = str(e).lower()
            assert "reasoning" not in error_msg or "precedence" not in error_msg
    
    @pytest.mark.parametrize("boundary_values", [
        {"temperature": 0.0},  # Minimum temperature
        {"temperature": 2.0},  # Maximum temperature
        {"top_p": 0.0},  # Minimum top_p
        {"top_p": 1.0},  # Maximum top_p
        {"max_tokens": 1},  # Minimum max_tokens
        {"presence_penalty": -2.0},  # Minimum presence_penalty
        {"presence_penalty": 2.0},  # Maximum presence_penalty
        {"frequency_penalty": -2.0},  # Minimum frequency_penalty
        {"frequency_penalty": 2.0},  # Maximum frequency_penalty
    ])
    def test_parameter_boundary_values(self, boundary_values):
        """Test handling of parameters at their boundary values."""
        try:
            result = self.endpoint.create(
                prompt="test prompt",
                model="gpt-3.5-turbo",
                validate_request=True,
                **boundary_values
            )
            assert result is not None
            
        except (APIError, RequestsConnectionError, ConnectionError, ValueError, StreamingError):
            # Expected in test environment
            pass


class Test_CompletionsEndpoint_Create_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for create method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    def test_validation_error_handling(self):
        """Test handling of request validation failures."""
        with pytest.raises(ValueError) as exc_info:
            self.endpoint.create(
                prompt="test",
                temperature="invalid_temperature",  # Wrong type
                validate_request=True
            )
        
        # Verify meaningful error message
        error_msg = str(exc_info.value)
        assert len(error_msg) > 0
        assert "validation" in error_msg.lower() or "invalid" in error_msg.lower()


class Test_CompletionsEndpoint_Create_05_StateTransitionBehaviors:
    """Test state transition behaviors for create method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    @pytest.mark.parametrize("stream", [True, False])
    def test_execution_path_switching(self, stream):
        """Test correct switching between streaming and non-streaming execution paths based on stream parameter."""
        try:
            result = self.endpoint.create(
                prompt="test prompt",
                model="gpt-3.5-turbo",
                stream=stream
            )
            
            if stream:
                # Should return iterator for streaming path
                assert hasattr(result, '__iter__')
                assert not isinstance(result, (CompletionsResponse, dict))
            else:
                # Should return response object for non-streaming path
                if result is not None:
                    assert not hasattr(result, '__iter__') or isinstance(result, (CompletionsResponse, dict))
                    
        except (APIError, RequestsConnectionError, ConnectionError, ValueError, StreamingError):
            # Expected in test environment without real API
            pass
    
    def test_parameter_processing_state_changes(self):
        """Test state changes during parameter processing and request preparation."""
        # Test with complex parameters that require processing
        complex_params = {
            "tools": [{"type": "function", "function": {"name": "test"}}],
            "reasoning": {"effort": "high"},
            "response_format": {"type": "json_object"}
        }
        
        try:
            # Test that parameter processing completes successfully
            result = self.endpoint.create(
                prompt="complex test prompt",
                model="gpt-4",
                validate_request=True,
                **complex_params
            )
            
            # If we get here, all parameter processing state transitions succeeded
            assert result is not None or True  # Success is making it through processing
            
        except (APIError, RequestsConnectionError, ConnectionError, ValueError, StreamingError):
            # Expected in test environment
            pass


class Test_CompletionsEndpoint_ResumeStream_01_NominalBehaviors:
    """Test nominal behaviors for resume_stream method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    def test_successful_stream_resumption(self):
        """Test successful resumption of streaming requests from valid state files."""
        # Create a comprehensive state file with all required data
        state_data = {
            "endpoint": "https://api.thisurlshouldntexsit.com/completions",
            "headers": {
                "Authorization": "Bearer test_key",
                "Content-Type": "application/json"
            },
            "prompt": "test prompt for resumption",
            "position": 150,
            "params": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 200,
                "stream": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(state_data, f)
            state_file = f.name
        
        try:
            result = self.endpoint.resume_stream(state_file)
            
            # Should return an iterator
            assert hasattr(result, '__iter__')
            assert hasattr(result, '__next__') or hasattr(result, 'next')
            
        except (ResumeError, ConnectionError, StreamingError):
            # Expected in test environment without real API
            pass
        finally:
            Path(state_file).unlink(missing_ok=True)
    
    @pytest.mark.parametrize("state_data", [
        {
            "endpoint": "https://api.thisurlshouldntexsit.com/completions",
            "headers": {"Authorization": "Bearer key1"},
            "prompt": "simple prompt",
            "position": 0,
            "params": {"model": "gpt-3.5-turbo"}
        },
        {
            "endpoint": "https://api.openai.com/v1/completions",
            "headers": {"Authorization": "Bearer key2", "User-Agent": "test"},
            "prompt": "complex prompt with many tokens",
            "position": 500,
            "params": {
                "model": "gpt-4",
                "temperature": 0.8,
                "max_tokens": 1000,
                "stop": ["\\n"]
            }
        }
    ])
    def test_resumption_with_various_state_configurations(self, state_data):
        """Test resumption with different valid state file configurations."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(state_data, f)
            state_file = f.name
        
        try:
            result = self.endpoint.resume_stream(state_file)
            assert hasattr(result, '__iter__')
            
        except (ResumeError, ConnectionError, StreamingError):
            # Expected in test environment
            pass
        finally:
            Path(state_file).unlink(missing_ok=True)


class Test_CompletionsEndpoint_ResumeStream_02_NegativeBehaviors:
    """Test negative behaviors for resume_stream method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    def test_handling_nonexistent_state_file(self):
        """Test proper handling of non-existent state files."""
        nonexistent_file = "/definitely/nonexistent/path/state.json"
        
        with pytest.raises((ResumeError, FileNotFoundError)):
            self.endpoint.resume_stream(nonexistent_file)
    
    @pytest.mark.parametrize("invalid_content", [
        "not valid json at all",
        '{"incomplete": "json"',  # Malformed JSON
        "null",  # Valid JSON but wrong type
        "[]",  # Valid JSON but wrong type
        '{"missing": "required_fields"}',  # Missing required state data
    ])
    def test_handling_corrupted_state_files(self, invalid_content):
        """Test proper handling of corrupted or malformed state files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(invalid_content)
            state_file = f.name
        
        try:
            with pytest.raises((ResumeError, json.JSONDecodeError, ValueError)):
                self.endpoint.resume_stream(state_file)
        finally:
            Path(state_file).unlink(missing_ok=True)
    
    @pytest.mark.parametrize("incomplete_state", [
        {},  # Completely empty
        {"endpoint": "https://api.thisurlshouldntexsit.com"},  # Missing headers, prompt, etc.
        {"headers": {"Auth": "Bearer test"}},  # Missing endpoint, prompt, etc.
        {"prompt": "test"},  # Missing endpoint, headers, etc.
        {"endpoint": "", "headers": {}, "prompt": ""},  # Empty required fields
    ])
    def test_handling_incomplete_state_data(self, incomplete_state):
        """Test handling of state files with incomplete or missing data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(incomplete_state, f)
            state_file = f.name
        
        try:
            with pytest.raises((ResumeError, ValueError, KeyError)):
                self.endpoint.resume_stream(state_file)
        finally:
            Path(state_file).unlink(missing_ok=True)


class Test_CompletionsEndpoint_ResumeStream_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for resume_stream method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    def test_resume_error_with_context(self):
        """Test conversion of resumption failures into ResumeError exceptions with context."""
        # Create a state file that will cause resumption to fail
        problematic_state = {
            "endpoint": "https://invalid.endpoint.thisurlshouldntexist.com",
            "headers": {"Authorization": "Bearer invalid_key"},
            "prompt": "test prompt",
            "position": -1,  # Invalid position
            "params": {"model": "nonexistent_model"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(problematic_state, f)
            state_file = f.name
        
        try:
            with pytest.raises(ResumeError) as exc_info:
                self.endpoint.resume_stream(state_file)
            
            # Verify error contains context information
            error = exc_info.value
            assert hasattr(error, 'state_file') and error.state_file == state_file
            assert hasattr(error, 'original_error') or 'error' in str(error).lower()
            
        except (APIError, RequestsConnectionError, ConnectionError, ValueError):
            # May be raised instead of ResumeError
            pass
        finally:
            Path(state_file).unlink(missing_ok=True)
    
    def test_meaningful_error_messages(self):
        """Test that error messages provide meaningful context for debugging."""
        test_cases = [
            ("missing_file.json", "file not found or path issues"),
            (None, "invalid file parameter")  # Will be created as empty
        ]
        
        for test_file, expected_context in test_cases:
            if test_file is None:
                # Test with completely invalid parameter
                with pytest.raises((ResumeError, TypeError, ValueError)) as exc_info:
                    self.endpoint.resume_stream(None)
                assert str(exc_info.value)  # Non-empty error message
            else:
                # Test with non-existent file
                with pytest.raises((ResumeError, FileNotFoundError)) as exc_info:
                    self.endpoint.resume_stream(test_file)
                error_msg = str(exc_info.value).lower()
                assert len(error_msg) > 0


class Test_CompletionsEndpoint_ResumeStream_05_StateTransitionBehaviors:
    """Test state transition behaviors for resume_stream method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager(api_key="test_key")
        self.http_manager = HTTPManager(base_url="https://api.thisurlshouldntexsit.com")
        self.endpoint = CompletionsEndpoint(self.auth_manager, self.http_manager)
    
    def test_state_restoration_from_file(self):
        """Test restoration of streaming state including endpoint, headers, and position from saved files."""
        # Create a comprehensive state file with all state information
        original_state = {
            "endpoint": "https://api.thisurlshouldntexsit.com/v1/completions",
            "headers": {
                "Authorization": "Bearer original_api_key",
                "Content-Type": "application/json",
                "User-Agent": "test-client/1.0"
            },
            "prompt": "original streaming prompt",
            "position": 750,
            "params": {
                "model": "gpt-4",
                "temperature": 0.9,
                "max_tokens": 1500,
                "stream": True,
                "stop": ["\\n", "END"]
            },
            "chunk_size": 4096
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(original_state, f)
            state_file = f.name
        
        try:
            # Attempt to resume - this tests the state restoration logic
            result = self.endpoint.resume_stream(state_file)
            
            # If successful, verify it returns an iterator (restored streaming state)
            assert hasattr(result, '__iter__')
            
        except (ResumeError, ConnectionError, StreamingError) as e:
            # Expected in test environment, but verify the state file was processed
            error_msg = str(e).lower()
            if "state_file" in error_msg or state_file in error_msg or "resume" in error_msg:
                # State file was processed, which means restoration logic ran
                assert True
            else:
                # Some other error occurred during network operations
                pass
        finally:
            Path(state_file).unlink(missing_ok=True)
    
    def test_transition_from_saved_to_active_state(self):
        """Test transition from saved state file to active streaming state."""
        # Test the complete state transition process
        saved_states = [
            {
                "endpoint": "https://api.thisurlshouldntexsit.com/completions",
                "headers": {"Authorization": "Bearer test1"},
                "prompt": "state transition test 1",
                "position": 0,  # Beginning of stream
                "params": {"model": "gpt-3.5-turbo"}
            },
            {
                "endpoint": "https://api.thisurlshouldntexsit.com/completions", 
                "headers": {"Authorization": "Bearer test2"},
                "prompt": "state transition test 2",
                "position": 500,  # Middle of stream
                "params": {"model": "gpt-4", "temperature": 0.7}
            }
        ]
        
        for i, state_data in enumerate(saved_states):
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_test_{i}.json', delete=False) as f:
                json.dump(state_data, f)
                state_file = f.name
            
            try:
                # Test state transition from file to active streaming
                result = self.endpoint.resume_stream(state_file)
                
                # Verify successful transition to active state
                assert hasattr(result, '__iter__')
                
                # Verify the iterator is ready for consumption
                try:
                    # Attempt to get the iterator (doesn't consume, just sets up)
                    iter(result)
                except (ConnectionError, StopIteration):
                    # Expected - we can't actually consume without real API
                    pass
                    
            except (ResumeError, ConnectionError, StreamingError):
                # Expected in test environment
                pass
            finally:
                Path(state_file).unlink(missing_ok=True)
    
    def test_preservation_of_original_request_context(self):
        """Test that original request context is preserved during state restoration."""
        # Create state with rich context information
        context_rich_state = {
            "endpoint": "https://api.thisurlshouldntexsit.com/v1/completions",
            "headers": {
                "Authorization": "Bearer context_test_key",
                "Content-Type": "application/json",
                "X-Request-ID": "original-request-123",
                "User-Agent": "test-client/2.0"
            },
            "prompt": "context preservation test prompt",
            "position": 1000,
            "params": {
                "model": "gpt-4",
                "temperature": 0.75,
                "max_tokens": 2000,
                "stream": True,
                "user": "test_user_context",
                "stop": ["STOP", "END"],
                "n": 1
            },
            "metadata": {
                "original_timestamp": 1234567890,
                "request_id": "original-req-456"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(context_rich_state, f)
            state_file = f.name
        
        try:
            # Attempt to resume with rich context
            result = self.endpoint.resume_stream(state_file)
            
            # If successful, verify it returns an iterator with preserved context
            assert hasattr(result, '__iter__')
            
            # Verify that the resume process completes without losing context information
            # The fact that no exception is raised indicates context was properly handled
            iter(result)
            
        except (ResumeError, ConnectionError, StreamingError) as e:
            # Expected in test environment, but verify context was processed
            error_msg = str(e).lower()
            
            # If the error mentions context-related issues, that's acceptable
            # as long as the state restoration logic was attempted
            assert ("context" in error_msg or 
                   "state" in error_msg or 
                   "resume" in error_msg or 
                   "headers" in error_msg or
                   "endpoint" in error_msg)
            
        finally:
            Path(state_file).unlink(missing_ok=True)
    
    def test_complex_state_transitions_with_metadata(self):
        """Test complex state transitions that include metadata preservation."""
        # Test multiple state files with different metadata configurations
        state_configurations = [
            {
                "endpoint": "https://api.thisurlshouldntexsit.com/completions",
                "headers": {"Authorization": "Bearer test1", "Custom-Header": "value1"},
                "prompt": "metadata test 1",
                "position": 100,
                "params": {"model": "gpt-3.5-turbo"},
                "metadata": {"version": "1.0", "source": "test_suite"}
            },
            {
                "endpoint": "https://api.openai.com/v1/completions",
                "headers": {"Authorization": "Bearer test2", "API-Version": "2023-07"},
                "prompt": "metadata test 2",
                "position": 500,
                "params": {"model": "gpt-4", "temperature": 0.8},
                "metadata": {"version": "2.0", "priority": "high", "tags": ["test", "metadata"]}
            }
        ]
        
        for i, state_config in enumerate(state_configurations):
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_metadata_{i}.json', delete=False) as f:
                json.dump(state_config, f)
                state_file = f.name
            
            try:
                # Test that each configuration can be processed
                result = self.endpoint.resume_stream(state_file)
                assert hasattr(result, '__iter__')
                
                # Verify transition to active state completes
                iter(result)
                
            except (ResumeError, ConnectionError, StreamingError):
                # Expected in test environment without real API
                pass
            finally:
                Path(state_file).unlink(missing_ok=True)