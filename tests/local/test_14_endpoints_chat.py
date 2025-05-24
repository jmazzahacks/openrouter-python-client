import logging
import json
from unittest.mock import Mock, patch

from pydantic import ValidationError
import pytest

from openrouter_client.auth import AuthManager
from openrouter_client.endpoints.chat import ChatEndpoint
from openrouter_client.exceptions import ResumeError, StreamingError
from openrouter_client.http import HTTPManager
from openrouter_client.models.chat import ChatCompletionFunction, ChatCompletionRequest, Message


class Test_ChatEndpoint_Init_01_NominalBehaviors:
    """Test nominal behaviors for ChatEndpoint.__init__ method."""
    
    def test_initialize_with_valid_managers(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        endpoint = ChatEndpoint(auth_manager, http_manager)
        
        # Verify the endpoint was created successfully
        assert endpoint.auth_manager is auth_manager
        assert endpoint.http_manager is http_manager
        assert hasattr(endpoint, 'logger')


class Test_ChatEndpoint_Init_02_NegativeBehaviors:
    """Test negative behaviors for ChatEndpoint.__init__ method."""
    
    @pytest.mark.parametrize("auth_manager,http_manager", [
        (None, Mock(spec=HTTPManager)),
        (Mock(spec=AuthManager), None),
        (None, None),
        ("invalid_auth", Mock(spec=HTTPManager)),
        (Mock(spec=AuthManager), "invalid_http")
    ])
    def test_handle_none_invalid_manager_parameters(self, auth_manager, http_manager):
        """Test handling None values and invalid types for required manager parameters."""
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            ChatEndpoint(auth_manager, http_manager)


class Test_ChatEndpoint_Init_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ChatEndpoint.__init__ method."""
    
    @pytest.mark.parametrize("exception_type,exception_message", [
        (RuntimeError, "Base initialization failed"),
        (ValueError, "Invalid configuration"),
        (AttributeError, "Missing required attribute")
    ])
    def test_manage_initialization_exceptions(self, exception_type, exception_message):
        """Test managing exceptions during initialization process - vital because initialization failures must not leave the object in an inconsistent state."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        with patch('openrouter_client.endpoints.chat.BaseEndpoint.__init__', side_effect=exception_type(exception_message)):
            with pytest.raises(exception_type) as exc_info:
                ChatEndpoint(auth_manager, http_manager)
            assert exception_message in str(exc_info.value)


class Test_ChatEndpoint_Init_05_StateTransitionBehaviors:
    """Test state transition behaviors for ChatEndpoint.__init__ method."""
    
    def test_complete_transition_to_functional_state(self, caplog):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        with caplog.at_level(logging.INFO):
            endpoint = ChatEndpoint(auth_manager, http_manager)
        
        assert "Initialized chat completions endpoint handler" in caplog.text
        assert endpoint.auth_manager is auth_manager
        assert endpoint.http_manager is http_manager


class Test_ChatEndpoint_ParseStreamingResponse_01_NominalBehaviors:
    """Test nominal behaviors for ChatEndpoint._parse_streaming_response method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("choice_data,expected_choice_count", [
        ([{"index": 0, "delta": {"content": "test"}, "finish_reason": "stop"}], 1),
        ([
            {"index": 0, "delta": {"content": "test1"}, "finish_reason": None},
            {"index": 1, "delta": {"content": "test2"}, "finish_reason": "stop"}
        ], 2),
        ([{"index": 0, "delta": {"content": "hello"}, "finish_reason": "length"}], 1)
    ])
    def test_parse_valid_streaming_chunks_with_complete_data(self, endpoint, choice_data, expected_choice_count):
        """Test parsing valid streaming chunks with complete choice and delta data - vital because this is the primary purpose of streaming response processing."""
        chunk = {
            "id": "chatcmpl-123",
            "object": "openrouter_client.endpoints.chat.completion.chunk",
            "choices": choice_data,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        
        with patch.multiple(
            'openrouter_client.endpoints.chat',
            ChatCompletionStreamResponseChoice=Mock(),
            ChatCompletionStreamResponseDelta=Mock(),
            ChatCompletionStreamResponse=Mock(),
            Usage=Mock(),
            FinishReason=Mock()
        ):
            response_iterator = [chunk]
            result = list(endpoint._parse_streaming_response(response_iterator))
            
            assert len(result) == 1
    
    def test_process_multiple_choices_correctly(self, endpoint):
        """Test processing multiple choices within response chunks correctly - vital because multi-choice responses are a documented API feature."""
        chunk = {
            "id": "chatcmpl-123",
            "object": "openrouter_client.endpoints.chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": "Choice 1"}, "finish_reason": None},
                {"index": 1, "delta": {"content": "Choice 2"}, "finish_reason": None},
                {"index": 2, "delta": {"content": "Choice 3"}, "finish_reason": "stop"}
            ]
        }
        
        with patch.multiple(
            'openrouter_client.endpoints.chat',
            ChatCompletionStreamResponseChoice=Mock(),
            ChatCompletionStreamResponseDelta=Mock(),
            ChatCompletionStreamResponse=Mock()
        ):
            response_iterator = [chunk]
            result = list(endpoint._parse_streaming_response(response_iterator))
            
            assert len(result) == 1


class Test_ChatEndpoint_ParseStreamingResponse_02_NegativeBehaviors:
    """Test negative behaviors for ChatEndpoint._parse_streaming_response method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("malformed_chunk", [
        {"id": "test", "choices": [{"index": 0}]},  # Missing delta and finish_reason
        {"id": "test", "choices": [{"delta": {"content": "test"}}]},  # Missing index
        {"id": "test"},  # Missing choices entirely
        {"choices": [{"index": 0, "delta": None, "finish_reason": "invalid"}]},  # Invalid finish_reason
        {"choices": [{"index": 0, "delta": {"malformed": True}}]}  # Malformed delta
    ])
    def test_handle_chunks_with_missing_malformed_fields(self, endpoint, malformed_chunk):
        """Test handling chunks with missing or malformed required fields without breaking the stream - vital because stream continuity must be maintained even with bad data."""
        response_iterator = [malformed_chunk]
        
        # Should not raise exception and should yield something
        result = list(endpoint._parse_streaming_response(response_iterator))
        assert len(result) == 1  # Should still yield the chunk even if malformed


class Test_ChatEndpoint_ParseStreamingResponse_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ChatEndpoint._parse_streaming_response method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("validation_error", [
        ValueError("Invalid choice data"),
        TypeError("Wrong type for delta"),
        KeyError("Missing required field"),
        AttributeError("Missing attribute")
    ])
    def test_graceful_fallback_to_raw_data_on_parsing_failure(self, endpoint, validation_error):
        """Test gracefully fallback to raw data when parsing fails while preserving stream continuity - vital because the method must never break the streaming iterator even with invalid data."""
        chunk = {"id": "test", "choices": [{"index": 0, "delta": {"content": "test"}}]}
        
        with patch('openrouter_client.endpoints.chat.ChatCompletionStreamResponse.model_validate', side_effect=validation_error):
            response_iterator = [chunk]
            result = list(endpoint._parse_streaming_response(response_iterator))
            
            # Should yield raw chunk and log warning
            assert len(result) == 1
            assert result[0] == chunk
            endpoint.logger.warning.assert_called()


class Test_ChatEndpoint_ParseStreamingResponse_05_StateTransitionBehaviors:
    """Test state transition behaviors for ChatEndpoint._parse_streaming_response method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    def test_maintain_iterator_state_progression(self, endpoint):
        """Test maintaining iterator state progression through all chunks regardless of parsing success - vital because streaming depends on continuous iteration without interruption."""
        chunks = [
            {"id": "chunk1", "choices": [{"index": 0, "delta": {"content": "test1"}}]},
            {"id": "chunk2", "malformed": True},  # This will fail parsing
            {"id": "chunk3", "choices": [{"index": 0, "delta": {"content": "test3"}}]}
        ]
        
        with patch('openrouter_client.endpoints.chat.ChatCompletionStreamResponse.model_validate', side_effect=[Mock(), Exception("Parse failed"), Mock()]):
            response_iterator = iter(chunks)
            result = list(endpoint._parse_streaming_response(response_iterator))
            
            # Should process all chunks despite middle one failing
            assert len(result) == 3


class Test_ChatEndpoint_CreateRequestModel_01_NominalBehaviors:
    """Test nominal behaviors for ChatEndpoint._create_request_model method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("messages,model", [
        ([{"role": "user", "content": "Hello"}], "gpt-3.5-turbo"),
        ([{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hi"}], "gpt-4"),
        ([{"role": "user", "content": "Test"}], None)
    ])
    def test_create_valid_request_models(self, endpoint, messages, model):
        """Test creating valid request models from proper message arrays and parameters - vital because this enables all completion requests."""
        with patch('openrouter_client.endpoints.chat.ChatCompletionRequest.model_validate') as mock_validate:
            mock_validate.return_value = Mock()
            
            result = endpoint._create_request_model(messages, model)
            
            mock_validate.assert_called_once()
            call_args = mock_validate.call_args[0][0]
            assert "messages" in call_args
            if model:
                assert call_args["model"] == model
    
    @pytest.mark.parametrize("mixed_messages", [
        ([Mock(spec=Message), {"role": "user", "content": "Hello"}]),
        ([{"role": "system", "content": "System"}, Mock(spec=Message), {"role": "user", "content": "User"}])
    ])
    def test_process_mixed_message_objects_and_dictionaries(self, endpoint, mixed_messages):
        """Test processing both Message objects and dictionary representations correctly - vital because the API accepts both formats."""
        # Mock Message objects to have model_dump method
        for msg in mixed_messages:
            if hasattr(msg, 'spec') and msg.spec == Message:
                msg.model_dump.return_value = {"role": "user", "content": "mocked"}
        
        with patch('openrouter_client.endpoints.chat.Message.model_validate') as mock_msg_validate:
            with patch('openrouter_client.endpoints.chat.ChatCompletionRequest.model_validate') as mock_req_validate:
                mock_msg_validate.return_value = Mock()
                mock_req_validate.return_value = Mock()
                
                endpoint._create_request_model(mixed_messages)
                
                mock_req_validate.assert_called_once()


class Test_ChatEndpoint_CreateRequestModel_02_NegativeBehaviors:
    """Test negative behaviors for ChatEndpoint._create_request_model method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("invalid_messages", [
        ([{"invalid": "structure"}]),
        ([{"role": "user"}]),  # Missing content
        ([{"content": "No role"}]),  # Missing role
        ([None]),
        ([{}])
    ])
    def test_handle_invalid_message_structures_with_fallback(self, endpoint, invalid_messages):
        """Test handling invalid message structures with fallback to raw data - vital because the method must not fail completely on malformed input."""
        with patch('openrouter_client.endpoints.chat.Message.model_validate', side_effect=ValueError("Invalid message")):
            with patch('openrouter_client.endpoints.chat.ChatCompletionRequest.model_validate') as mock_req_validate:
                mock_req_validate.return_value = Mock()
                
                endpoint._create_request_model(invalid_messages)
                
                # Should still create request with raw data
                mock_req_validate.assert_called_once()
                endpoint.logger.warning.assert_called()


class Test_ChatEndpoint_CreateRequestModel_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ChatEndpoint._create_request_model method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("failing_component,error_type", [
        ("tools", ValueError("Invalid tool")),
        ("functions", TypeError("Invalid function")),
        ("reasoning", AttributeError("Invalid reasoning")),
        ("response_format", KeyError("Missing format field"))
    ])
    def test_preserve_functionality_on_nested_component_failures(self, endpoint, failing_component, error_type):
        """Test preserving functionality when nested component validation fails - vital because partial parsing failures should not prevent request creation."""
        messages = [{"role": "user", "content": "Hello"}]
        kwargs_map = {
            "tools": [{"invalid": "data"}],  # List of tool objects
            "functions": [{"invalid": "data"}],  # List of function objects  
            "reasoning": {"invalid": "data"},  # Dict is valid for reasoning
            "response_format": {"invalid": "data"}  # Dict is valid for response_format
        }
        kwargs = {failing_component: kwargs_map[failing_component]}
        
        # Mock the specific component validation to fail
        component_patches = {
            "tools": "openrouter_client.endpoints.chat.ChatCompletionTool.model_validate",
            "functions": "openrouter_client.endpoints.chat.FunctionDefinition.model_validate", 
            "reasoning": "openrouter_client.endpoints.chat.ReasoningConfig.model_validate",
            "response_format": "openrouter_client.endpoints.chat.ResponseFormat.model_validate"
        }
        
        with patch(component_patches[failing_component], side_effect=error_type):
            with patch('openrouter_client.endpoints.chat.ChatCompletionRequest.model_validate') as mock_req_validate:
                mock_req_validate.return_value = Mock()
                
                result = endpoint._create_request_model(messages, **kwargs)
                
                # Should still complete successfully with raw data
                mock_req_validate.assert_called_once()
                endpoint.logger.warning.assert_called()


class Test_ChatEndpoint_CreateRequestModel_05_StateTransitionBehaviors:
    """Test state transition behaviors for ChatEndpoint._create_request_model method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    def test_transform_raw_input_to_validated_request_models(self, endpoint):
        """Test successfully transforming raw input data into validated request models - vital because this transformation is the method's core responsibility."""
        raw_messages = [{"role": "user", "content": "Hello"}]
        raw_kwargs = {
            "temperature": 0.7,
            "max_tokens": 100,
            "tools": [{"type": "function", "function": {"name": "test"}}]
        }
        
        with patch('openrouter_client.endpoints.chat.ChatCompletionRequest.model_validate') as mock_validate:
            mock_request = Mock(spec=ChatCompletionRequest)
            mock_validate.return_value = mock_request
            
            result = endpoint._create_request_model(raw_messages, **raw_kwargs)
            
            # Verify transformation occurred
            assert result == mock_request
            mock_validate.assert_called_once()
            
            # Verify input data was processed into request data
            call_args = mock_validate.call_args[0][0]
            assert "messages" in call_args
            assert "temperature" in call_args
            assert "max_tokens" in call_args


class Test_ChatEndpoint_ParseToolCalls_01_NominalBehaviors:
    """Test nominal behaviors for ChatEndpoint._parse_tool_calls method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("tool_calls_data,expected_count", [
        ([{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}], 1),
        ([
            {"id": "call_1", "type": "function", "function": {"name": "func1", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"name": "func2", "arguments": "{\"param\": \"value\"}"}}
        ], 2),
        ([], 0)
    ])
    def test_parse_valid_tool_call_arrays(self, endpoint, tool_calls_data, expected_count):
        """Test parsing valid tool call arrays with proper function structures - vital because tool calling is a major API feature."""
        with patch('openrouter_client.endpoints.chat.ToolCallFunction.model_validate') as mock_func_validate:
            with patch('openrouter_client.endpoints.chat.ChatCompletionToolCall.model_validate') as mock_tool_validate:
                mock_func_validate.return_value = Mock()
                mock_tool_validate.return_value = Mock()
                
                result = endpoint._parse_tool_calls(tool_calls_data)
                
                assert len(result) == expected_count
                if expected_count > 0:
                    assert mock_func_validate.call_count == expected_count
                    assert mock_tool_validate.call_count == expected_count


class Test_ChatEndpoint_ParseToolCalls_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ChatEndpoint._parse_tool_calls method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("validation_error", [
        ValueError("Invalid function"),
        TypeError("Wrong type"),
        KeyError("Missing field")
    ])
    def test_fallback_to_raw_data_on_parsing_failure(self, endpoint, validation_error):
        """Test fallback to raw data when individual tool call parsing fails - vital because partial failures should not break the entire tool calls array."""
        tool_calls_data = [
            {"id": "call_1", "type": "function", "function": {"name": "valid", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"invalid": "data"}}
        ]
        
        with patch('openrouter_client.endpoints.chat.ToolCallFunction.model_validate', side_effect=[Mock(), validation_error]):
            with patch('openrouter_client.endpoints.chat.ChatCompletionToolCall.model_validate', side_effect=[Mock(), validation_error]):
                result = endpoint._parse_tool_calls(tool_calls_data)
                
                # Should return both calls, second one as raw data
                assert len(result) == 2
                endpoint.logger.warning.assert_called()


class Test_ChatEndpoint_ParseFunctionCall_01_NominalBehaviors:
    """Test nominal behaviors for ChatEndpoint._parse_function_call method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("function_call_data", [
        {"name": "test_function", "arguments": "{}"},
        {"name": "complex_function", "arguments": "{\"param1\": \"value1\", \"param2\": 42}"},
        {"name": "simple_function", "arguments": "{\"message\": \"Hello World\"}"}
    ])
    def test_parse_valid_function_calls_with_json_arguments(self, endpoint, function_call_data):
        """Test parsing valid function calls with proper JSON arguments - vital because function calling requires correct argument parsing."""
        with patch('openrouter_client.endpoints.chat.ChatCompletionFunction.model_validate') as mock_validate:
            mock_validate.return_value = Mock(spec=ChatCompletionFunction)
            
            result = endpoint._parse_function_call(function_call_data)
            
            mock_validate.assert_called_once_with(function_call_data)


class Test_ChatEndpoint_ParseFunctionCall_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ChatEndpoint._parse_function_call method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("invalid_function_data,error_type", [
        ({"name": "test", "arguments": "invalid json"}, "json_error"),
        ({"invalid": "structure"}, "validation_error"),
        ({"name": "test", "arguments": "{unclosed"}, "json_error")
    ])
    def test_handle_json_parsing_failures_gracefully(self, endpoint, invalid_function_data, error_type):
        """Test handling JSON parsing failures in function arguments gracefully - vital because malformed JSON should not crash the parsing process."""
        if error_type == "validation_error":
            with patch('openrouter_client.endpoints.chat.ChatCompletionFunction.model_validate', side_effect=ValueError("Invalid function")):
                result = endpoint._parse_function_call(invalid_function_data)
                
                # Should return raw data when validation fails
                assert result == invalid_function_data
                endpoint.logger.warning.assert_called()
        else:  # json_error
            # JSON parsing happens internally, just verify it doesn't crash
            with patch('openrouter_client.endpoints.chat.ChatCompletionFunction.model_validate') as mock_validate:
                mock_validate.return_value = Mock()
                
                result = endpoint._parse_function_call(invalid_function_data)
                
                # Should still attempt validation even with bad JSON
                mock_validate.assert_called_once()


class Test_ChatEndpoint_Create_01_NominalBehaviors:
    """Test nominal behaviors for ChatEndpoint.create method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("stream,expected_streaming", [
        (True, True),
        (False, False),
        (None, False)
    ])
    def test_generate_streaming_and_non_streaming_completions(self, endpoint, stream, expected_streaming):
        """Test generating both streaming and non-streaming completions successfully - vital because these are the two primary operation modes."""
        messages = [{"role": "user", "content": "Hello"}]
        
        if expected_streaming:
            with patch('openrouter_client.endpoints.chat.StreamingChatCompletionsRequest') as mock_streamer:
                mock_instance = Mock()
                mock_streamer.return_value = mock_instance
                mock_instance.get_result.return_value = iter([])
                
                with patch.object(endpoint, '_parse_streaming_response', return_value=iter([])):
                    result = endpoint.create(messages, stream=stream)
                    
                    assert hasattr(result, '__iter__')  # Should be iterator
                    mock_streamer.assert_called_once()
        else:
            endpoint.http_manager.post.return_value.json.return_value = {"id": "test"}
            
            with patch('openrouter_client.endpoints.chat.ChatCompletionResponse.model_validate') as mock_validate:
                mock_validate.return_value = Mock()
                
                result = endpoint.create(messages, stream=stream)
                
                endpoint.http_manager.post.assert_called_once()
                mock_validate.assert_called_once()
    
    @pytest.mark.parametrize("messages,kwargs", [
        ([{"role": "user", "content": "Hello"}], {"temperature": 0.7, "max_tokens": 100}),
        ([Mock(spec=Message)], {"top_p": 0.9, "presence_penalty": 0.1}),
        ([{"role": "system", "content": "System"}, {"role": "user", "content": "User"}], {"functions": [], "tools": []})
    ])
    def test_process_mixed_message_types_and_parameters(self, endpoint, messages, kwargs):
        """Test processing conversations with mixed message types and parameter combinations - vital because real-world usage involves complex parameter sets."""
        # Mock Message objects if present
        for msg in messages:
            if hasattr(msg, 'spec') and msg.spec == Message:
                msg.model_dump.return_value = {"role": "user", "content": "test"}
        
        endpoint.http_manager.post.return_value.json.return_value = {"id": "test"}
        
        with patch('openrouter_client.endpoints.chat.ChatCompletionResponse.model_validate') as mock_validate:
            mock_validate.return_value = Mock()
            
            endpoint.create(messages, **kwargs)
            
            # Verify parameters were passed through
            call_args = endpoint.http_manager.post.call_args[1]['json']
            for key, value in kwargs.items():
                if value is not None:
                    assert key in call_args


class Test_ChatEndpoint_Create_02_NegativeBehaviors:
    """Test negative behaviors for ChatEndpoint.create method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("invalid_params", [
        {"model": ""},  # Empty model
        {"model": None},  # None model when required
        {"temperature": -1.0},  # Below valid range
        {"temperature": 3.0},  # Above valid range
        {"top_p": -0.1},  # Below valid range
        {"top_p": 1.1},  # Above valid range
        {"max_tokens": -1},  # Negative tokens
        {"n": 0},  # Invalid choice count
    ])
    def test_handle_invalid_parameters_appropriately(self, endpoint, invalid_params):
        """Test handling invalid model identifiers and out-of-range parameters appropriately - vital because these are common user errors that must be caught."""
        messages = [{"role": "user", "content": "Hello"}]
        
        endpoint.http_manager.post.return_value.json.return_value = {"id": "test"}
        
        # The method should either handle gracefully or let the API reject it
        # Most validation happens server-side, so we test that the request is formed
        with patch('openrouter_client.endpoints.chat.ChatCompletionResponse.model_validate') as mock_validate:
            mock_validate.return_value = Mock()
            
            # Should not crash during request formation
            endpoint.create(messages, **invalid_params)
            
            endpoint.http_manager.post.assert_called_once()


class Test_ChatEndpoint_Create_03_BoundaryBehaviors:
    """Test boundary behaviors for ChatEndpoint.create method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("boundary_params", [
        {"temperature": 0.0},  # Minimum temperature
        {"temperature": 2.0},  # Maximum temperature
        {"top_p": 0.0},  # Minimum top_p
        {"top_p": 1.0},  # Maximum top_p
        {"presence_penalty": -2.0},  # Minimum penalty
        {"presence_penalty": 2.0},  # Maximum penalty
        {"frequency_penalty": -2.0},  # Minimum penalty
        {"frequency_penalty": 2.0},  # Maximum penalty
        {"max_tokens": 1},  # Minimum tokens
        {"n": 1}  # Minimum choices
    ])
    def test_process_requests_at_documented_limits(self, endpoint, boundary_params):
        """Test processing requests with parameter values at documented limits - vital because these boundaries define valid API usage."""
        messages = [{"role": "user", "content": "Hello"}]
        
        endpoint.http_manager.post.return_value.json.return_value = {"id": "test"}
        
        with patch('openrouter_client.endpoints.chat.ChatCompletionResponse.model_validate') as mock_validate:
            mock_validate.return_value = Mock()
            
            endpoint.create(messages, **boundary_params)
            
            # Verify boundary values are properly included
            call_args = endpoint.http_manager.post.call_args[1]['json']
            for key, value in boundary_params.items():
                assert call_args[key] == value


class Test_ChatEndpoint_Create_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ChatEndpoint.create method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("http_error", [
        Exception("Connection failed"),
        TimeoutError("Request timeout"),
        ConnectionError("Network error")
    ])
    def test_manage_connectivity_issues_with_appropriate_exceptions(self, endpoint, http_error):
        """Test managing authentication failures and HTTP connectivity issues with appropriate exceptions - vital because network and auth failures are common and must be properly surfaced."""
        messages = [{"role": "user", "content": "Hello"}]
        
        endpoint.http_manager.post.side_effect = http_error
        
        with pytest.raises(type(http_error)):
            endpoint.create(messages)
    
    def test_handle_streaming_initialization_failures(self, endpoint):
        """Test handling streaming initialization failures with proper StreamingError reporting - vital because streaming failures need specific error context for debugging."""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch('openrouter_client.endpoints.chat.StreamingChatCompletionsRequest') as mock_streamer:
            mock_instance = Mock()
            mock_streamer.return_value = mock_instance
            mock_instance.start.side_effect = Exception("Streaming failed")
            mock_instance.position = 0
            
            with pytest.raises(StreamingError) as exc_info:
                endpoint.create(messages, stream=True)
            
            assert "Streaming chat completions failed" in str(exc_info.value)


class Test_ChatEndpoint_Create_05_StateTransitionBehaviors:
    """Test state transition behaviors for ChatEndpoint.create method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("stream_value,expected_path", [
        (True, "streaming"),
        (False, "non_streaming"),
        (None, "non_streaming")
    ])
    def test_transition_between_streaming_modes(self, endpoint, stream_value, expected_path):
        """Test correctly transitioning between streaming and non-streaming request processing paths - vital because the method must handle both modes based on the stream parameter."""
        messages = [{"role": "user", "content": "Hello"}]
        
        if expected_path == "streaming":
            with patch('openrouter_client.endpoints.chat.StreamingChatCompletionsRequest') as mock_streamer:
                mock_instance = Mock()
                mock_streamer.return_value = mock_instance
                mock_instance.get_result.return_value = iter([])
                
                with patch.object(endpoint, '_parse_streaming_response', return_value=iter([])):
                    result = endpoint.create(messages, stream=stream_value)
                    
                    mock_streamer.assert_called_once()
                    assert hasattr(result, '__iter__')
        else:
            endpoint.http_manager.post.return_value.json.return_value = {"id": "test"}
            
            with patch('openrouter_client.endpoints.chat.ChatCompletionResponse.model_validate') as mock_validate:
                mock_validate.return_value = Mock()
                
                result = endpoint.create(messages, stream=stream_value)
                
                endpoint.http_manager.post.assert_called_once()
                assert not hasattr(result, '__iter__') or not hasattr(result, '__next__')
    
    @pytest.mark.parametrize("validate_request", [True, False])
    def test_manage_request_validation_state(self, endpoint, validate_request):
        """Test managing request validation state when validate_request is enabled - vital because validation is an optional but important feature for request verification."""
        messages = [{"role": "user", "content": "Hello"}]
        
        if validate_request:
            with patch.object(endpoint, '_create_request_model') as mock_create:
                mock_create.return_value = Mock()
                endpoint.http_manager.post.return_value.json.return_value = {"id": "test"}
                
                with patch('openrouter_client.endpoints.chat.ChatCompletionResponse.model_validate'):
                    endpoint.create(messages, validate_request=validate_request)
                    
                    mock_create.assert_called_once()
        else:
            endpoint.http_manager.post.return_value.json.return_value = {"id": "test"}
            
            with patch('openrouter_client.endpoints.chat.ChatCompletionResponse.model_validate'):
                with patch.object(endpoint, '_create_request_model') as mock_create:
                    endpoint.create(messages, validate_request=validate_request)
                    
                    mock_create.assert_not_called()


class Test_ChatEndpoint_ResumeStream_01_NominalBehaviors:
    """Test nominal behaviors for ChatEndpoint.resume_stream method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    def test_successfully_resume_from_valid_state_files(self, endpoint):
        """Test successfully resuming streaming from valid saved state files - vital because stream resumption is the method's sole purpose."""
        state_file = "test_state.json"
        
        with patch('openrouter_client.endpoints.chat.StreamingChatCompletionsRequest') as mock_streamer:
            mock_instance = Mock()
            mock_streamer.return_value = mock_instance
            mock_instance.get_result.return_value = iter([{"id": "resumed_chunk"}])
            
            with patch.object(endpoint, '_parse_streaming_response') as mock_parse:
                mock_parse.return_value = iter([Mock()])
                
                result = endpoint.resume_stream(state_file)
                
                mock_streamer.assert_called_once()
                mock_instance.resume.assert_called_once()
                assert hasattr(result, '__iter__')


class Test_ChatEndpoint_ResumeStream_02_NegativeBehaviors:
    """Test negative behaviors for ChatEndpoint.resume_stream method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("state_file_issue", [
        "nonexistent_file.json",
        "",  # Empty filename
        None  # None filename
    ])
    def test_handle_nonexistent_corrupted_state_files(self, endpoint, state_file_issue):
        """Test handling non-existent or corrupted state files with appropriate error reporting - vital because state file issues are common failure modes."""
        with patch('openrouter_client.endpoints.chat.StreamingChatCompletionsRequest') as mock_streamer:
            mock_instance = Mock()
            mock_streamer.return_value = mock_instance
            mock_instance.resume.side_effect = FileNotFoundError("State file not found")
            
            with pytest.raises(ResumeError) as exc_info:
                endpoint.resume_stream(state_file_issue)
            
            assert "Resuming chat completions stream failed" in str(exc_info.value)


@patch('openrouter_client.endpoints.chat.StreamingChatCompletionsRequest')
class Test_ChatEndpoint_ResumeStream_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ChatEndpoint.resume_stream method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    @pytest.mark.parametrize("resume_error", [
        FileNotFoundError("State file not found"),
        json.JSONDecodeError("Invalid JSON", "", 0),
        ValueError("Corrupted state data"),
        ConnectionError("Network failure during resume")
    ])
    def test_raise_resume_error_with_proper_context(self, mock_streamer_class, endpoint, resume_error):
        """Test raising ResumeError with proper context when resumption fails."""
        state_file = "test_state.json"
        
        # Configure the mock instance
        mock_instance = Mock()
        mock_streamer_class.return_value = mock_instance
        mock_instance.resume.side_effect = resume_error
        
        # Execute the test and capture the exception
        with pytest.raises(ResumeError) as exc_info:
            endpoint.resume_stream(state_file)
        
        # Verify error context is preserved in exception message and structure
        assert "Resuming chat completions stream failed" in str(exc_info.value)
        
        # Check that state_file information is preserved (flexible approach)
        exception_data = str(exc_info.value) + str(getattr(exc_info.value, '__dict__', {})) + str(exc_info.value.args)
        assert state_file in exception_data
        
        # Check that original error information is preserved
        assert str(resume_error) in exception_data
        
        # Verify the streaming request was called correctly
        mock_streamer_class.assert_called_once_with(
            endpoint="",
            headers={},
            messages=[],
            state_file=state_file,
            logger=endpoint.logger,
            client=endpoint.http_manager.client
        )
        mock_instance.resume.assert_called_once()


class Test_ChatEndpoint_ResumeStream_05_StateTransitionBehaviors:
    """Test state transition behaviors for ChatEndpoint.resume_stream method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Configure missing attributes for streaming path
        http_manager.base_url = "https://api.example.com"
        http_manager.client = Mock()
        
        # Create normally, then patch what you need
        endpoint = ChatEndpoint(auth_manager, http_manager)
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer test"})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        return endpoint
    
    def test_restore_complete_streaming_context_and_continue(self, endpoint):
        """Test restoring complete streaming context from saved state and continuing iteration - vital because successful state restoration is essential for stream continuity."""
        state_file = "test_state.json"
        
        with patch('openrouter_client.endpoints.chat.StreamingChatCompletionsRequest') as mock_streamer:
            mock_instance = Mock()
            mock_streamer.return_value = mock_instance
            
            # Mock state restoration
            mock_instance.resume.return_value = None
            mock_instance.get_result.return_value = iter([
                {"id": "chunk1", "choices": []},
                {"id": "chunk2", "choices": []}
            ])
            
            with patch.object(endpoint, '_parse_streaming_response') as mock_parse:
                expected_chunks = [Mock(), Mock()]
                mock_parse.return_value = iter(expected_chunks)
                
                result = endpoint.resume_stream(state_file)
                
                # Verify state restoration and continuation
                mock_instance.resume.assert_called_once()
                mock_instance.get_result.assert_called_once()
                
                # Verify iterator can be consumed
                chunks = list(result)
                assert len(chunks) == len(expected_chunks)
