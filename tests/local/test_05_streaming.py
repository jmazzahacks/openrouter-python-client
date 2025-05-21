import pytest
import json
import base64
from datetime import datetime, timezone
from unittest.mock import MagicMock
from smartsurge import SmartSurgeClient
from smartsurge.exceptions import ResumeError, StreamingError

from openrouter_client.streaming import OpenRouterStreamingState, StreamingCompletionsRequest, StreamingChatCompletionsRequest


class Test_OpenRouterStreamingState_Init_01_NominalBehaviors:
    """Test nominal behaviors of the OpenRouterStreamingState.__init__ method."""
    
    def test_initialization_with_required_parameters(self):
        """Test that OpenRouterStreamingState initializes correctly with required parameters."""
        # Arrange
        endpoint = "/v1/completions"
        method = "POST"
        headers = {"Authorization": "Bearer test_token"}
        accumulated_data = b""
        last_position = 0
        
        # Act
        state = OpenRouterStreamingState(
            endpoint=endpoint,
            method=method,
            headers=headers,
            accumulated_data=accumulated_data,
            last_position=last_position
        )
        
        # Assert
        assert state.endpoint == endpoint
        assert state.method == method
        assert state.headers == headers
        assert state.accumulated_data == accumulated_data
        assert state.last_position == last_position
        assert state.params is None
        assert state.data is None
        assert state.total_size is None
        assert state.etag is None
        assert isinstance(state.last_updated, datetime)
        assert state.last_updated.tzinfo == timezone.utc
        assert len(state.request_id) > 0

class Test_OpenRouterStreamingState_Init_04_ErrorHandlingBehaviors:
    """Test error handling behaviors of the OpenRouterStreamingState.__init__ method."""
    
    @pytest.mark.parametrize("endpoint,method,headers,last_position,expected_error", [
        ("", "POST", {"Authorization": "Bearer test_token"}, 0, "String should have at least 1 character"),
        ("/v1/completions", "POST", "not_a_dict", 0, "Input should be a valid dictionary"),
        ("/v1/completions", "POST", {"Authorization": "Bearer test_token"}, -1, "greater than or equal to 0"),
    ])
    def test_parameter_validation(self, endpoint, method, headers, last_position, expected_error):
        """Test validation of parameters during initialization."""
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            OpenRouterStreamingState(
                endpoint=endpoint,
                method=method,
                headers=headers,
                accumulated_data=b"",
                last_position=last_position
            )
        assert expected_error in str(exc_info.value)

class Test_OpenRouterStreamingState_DecodeAccumulatedData_01_NominalBehaviors:
    """Test nominal behaviors of the OpenRouterStreamingState.decode_accumulated_data method."""
    
    def test_processing_sse_format_with_data_prefix(self):
        """Test processing of SSE format with data prefix."""
        # Arrange
        sse_data = 'data: {"id":"12345","choices":[{"delta":{"content":"Hello"}}]}\n\n'
        encoded_data = base64.b64encode(sse_data.encode('utf-8')).decode('ascii')
        
        # Act
        result = OpenRouterStreamingState.decode_accumulated_data(encoded_data)
        
        # Assert
        assert b'data: {"id":"12345","choices":[{"delta":{"content":"Hello"}}]}' in result
    
    def test_handling_openrouter_delta_format(self):
        """Test handling of OpenRouter delta format in streaming responses."""
        # Arrange
        sse_data = 'data: {"id":"12345","choices":[{"delta":{"content":"Hello"}}]}\n\n'
        encoded_data = base64.b64encode(sse_data.encode('utf-8')).decode('ascii')
        
        # Act
        result = OpenRouterStreamingState.decode_accumulated_data(encoded_data)
        
        # Assert
        decoded_result = result.decode('utf-8')
        assert 'data: {"id":"12345","choices":[{"delta":{"content":"Hello"}}]}' in decoded_result

class Test_OpenRouterStreamingState_DecodeAccumulatedData_04_ErrorHandlingBehaviors:
    """Test error handling behaviors of the OpenRouterStreamingState.decode_accumulated_data method."""
    
    def test_recovering_from_malformed_data(self):
        """Test recovery from malformed data in the stream."""
        # Arrange
        malformed_data = 'not valid base64 !'
        
        # Act
        result = OpenRouterStreamingState.decode_accumulated_data(malformed_data)
        
        # Assert
        assert isinstance(result, bytes)
        assert len(result) >= 0

class Test_OpenRouterStreamingState_DecodeAccumulatedData_03_BoundaryBehaviors:
    """Test boundary behaviors of the OpenRouterStreamingState.decode_accumulated_data method."""
    
    def test_processing_empty_input(self):
        """Test processing of empty input data."""
        # Arrange
        empty_data = ""
        
        # Act
        result = OpenRouterStreamingState.decode_accumulated_data(empty_data)
        
        # Assert
        assert isinstance(result, bytes)
        assert len(result) == 0

class Test_OpenRouterStreamingState_SerializeAccumulatedData_01_NominalBehaviors:
    """Test nominal behaviors of the OpenRouterStreamingState.serialize_accumulated_data method."""
    
    def test_encoding_bytes_to_base64_string(self):
        """Test encoding of bytes to base64 string for serialization."""
        # Arrange
        test_data = b"Test data to be encoded"
        state = MagicMock()
        
        # Act
        result = OpenRouterStreamingState.serialize_accumulated_data(state, test_data)
        
        # Assert
        decoded = base64.b64decode(result.encode('ascii'))
        assert decoded == test_data

class Test_StreamingCompletionsRequest_Init_01_NominalBehaviors:
    """Test nominal behaviors of the StreamingCompletionsRequest.__init__ method."""
    
    def test_initialization_with_api_parameters(self):
        """Test initialization with API parameters."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        endpoint = "/v1/completions"
        headers = {"Authorization": "Bearer test_token"}
        prompt = "Hello, world!"
        
        # Act
        request = StreamingCompletionsRequest(
            client=client,
            endpoint=endpoint,
            headers=headers,
            prompt=prompt
        )
        
        # Assert
        assert request.client == client
        assert request.endpoint == endpoint
        assert request.headers == headers
        assert request.prompt == prompt
        assert request.params == {"stream": True}
        assert request.completions == []
        assert request._cancelled is False
        assert request._response is None

class Test_StreamingCompletionsRequest_Init_05_StateTransitionBehaviors:
    """Test state transition behaviors of the StreamingCompletionsRequest.__init__ method."""
    
    def test_setting_stream_true(self):
        """Test that stream parameter is always set to True regardless of input."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        
        # Act - Case 1: params is None
        request1 = StreamingCompletionsRequest(
            client=client,
            endpoint="/v1/completions",
            headers={},
            prompt="Test"
        )
        
        # Case 2: params with stream=False
        request2 = StreamingCompletionsRequest(
            client=client,
            endpoint="/v1/completions",
            headers={},
            prompt="Test",
            params={"stream": False}
        )
        
        # Assert
        assert request1.params.get("stream") is True
        assert request2.params.get("stream") is True

class Test_StreamingCompletionsRequest_Stream_01_NominalBehaviors:
    """Test nominal behaviors of the StreamingCompletionsRequest.stream method."""
    
    def test_making_streaming_request_and_processing_chunks(self):
        """Test making a streaming request and processing chunks."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Mock the chunks to simulate SSE data
        chunks = [
            b'data: {"id": "1", "choices": [{"delta": {"content": "Hello"}}]}\n\n',
            b'data: {"id": "1", "choices": [{"delta": {"content": " world"}}]}\n\n',
            b'data: [DONE]\n\n'
        ]
        mock_response.iter_content.return_value = chunks
        client.post.return_value = (mock_response, {})
        
        request = StreamingCompletionsRequest(
            client=client,
            endpoint="/v1/completions",
            headers={"Authorization": "Bearer test_token"},
            prompt="Complete this: Hello"
        )
        
        # Mock process_chunk to return the expected format
        request.process_chunk = MagicMock(side_effect=[
            [{"id": "1", "choices": [{"delta": {"content": "Hello"}}]}],
            [{"id": "1", "choices": [{"delta": {"content": " world"}}]}],
            []
        ])
        
        # Act
        results = list(request.stream())
        
        # Assert
        assert client.post.called
        client.post.assert_called_with(
            "/v1/completions",
            headers={"Authorization": "Bearer test_token"},
            json={"prompt": "Complete this: Hello", "stream": True},
            stream=True
        )
        assert request.process_chunk.call_count == 3
        assert len(results) == 2  # [DONE] should not yield a result
        assert results == [
            {"id": "1", "choices": [{"delta": {"content": "Hello"}}]},
            {"id": "1", "choices": [{"delta": {"content": " world"}}]}
        ]

class Test_StreamingCompletionsRequest_Stream_04_ErrorHandlingBehaviors:
    """Test error handling behaviors of the StreamingCompletionsRequest.stream method."""
    
    def test_handling_network_and_api_errors(self):
        """Test handling of network and API errors."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        client.post.return_value = (mock_response, {})
        
        request = StreamingCompletionsRequest(
            client=client,
            endpoint="/v1/completions",
            headers={"Authorization": "Bearer test_token"},
            prompt="Complete this: Hello"
        )
        
        # Act & Assert
        with pytest.raises(StreamingError) as exc_info:
            list(request.stream())
        
        assert "Request failed with status 400" in str(exc_info.value)

class Test_StreamingCompletionsRequest_Stream_05_StateTransitionBehaviors:
    """Test state transition behaviors of the StreamingCompletionsRequest.stream method."""
    
    def test_updating_position_and_accumulated_data(self):
        """Test updating of position and accumulated data during streaming."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        chunk = b'data: {"id": "1", "choices": [{"delta": {"content": "Hello"}}]}\n\n'
        mock_response.iter_content.return_value = [chunk]
        client.post.return_value = (mock_response, {})
        
        request = StreamingCompletionsRequest(
            client=client,
            endpoint="/v1/completions",
            headers={},
            prompt="Test"
        )
        request.accumulated_data = bytearray()
        request.position = 0
        
        # Mock process_chunk to return test data
        request.process_chunk = MagicMock(return_value=[{"id": "1", "choices": [{"delta": {"content": "Hello"}}]}])
        
        # Act
        list(request.stream())
        
        # Assert
        assert request.position == len(chunk)
        assert bytes(request.accumulated_data) == chunk
        assert request.process_chunk.called

class Test_StreamingCompletionsRequest_Stream_02_NegativeBehaviors:
    """Test negative behaviors of the StreamingCompletionsRequest.stream method."""
    
    def test_checking_cancellation_status(self):
        """Test that streaming stops when cancellation is requested."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Set up for cancellation after first chunk
        chunks = [
            b'data: {"id": "1", "choices": [{"delta": {"content": "Hello"}}]}\n\n',
            b'data: {"id": "1", "choices": [{"delta": {"content": " world"}}]}\n\n'
        ]
        mock_response.iter_content.return_value = chunks
        client.post.return_value = (mock_response, {})
        
        request = StreamingCompletionsRequest(
            client=client,
            endpoint="/v1/completions",
            headers={},
            prompt="Test"
        )
        
        # Mock process_chunk and set up cancellation after first chunk
        def side_effect(chunk):
            if b"Hello" in chunk:
                request._cancelled = True
            return [{"id": "1"}]
        
        request.process_chunk = MagicMock(side_effect=side_effect)
        
        # Act
        results = list(request.stream())
        
        # Assert
        assert len(results) == 1  # Only one result before cancellation
        assert request.process_chunk.call_count == 1

class Test_StreamingCompletionsRequest_ProcessChunk_01_NominalBehaviors:
    """Test nominal behaviors of the StreamingCompletionsRequest.process_chunk method."""
    
    def test_parsing_sse_format_and_extracting_json(self):
        """Test parsing of SSE format and extraction of JSON data."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        request = StreamingCompletionsRequest(
            client=client,
            endpoint="/v1/completions",
            headers={},
            prompt="Test"
        )
        
        chunk = b'data: {"id": "1", "choices": [{"text": "Hello world"}]}\n\n'
        
        # Act
        result = request.process_chunk(chunk)
        
        # Assert
        assert len(result) == 1
        assert result[0]["id"] == "1"
        assert result[0]["choices"][0]["text"] == "Hello world"
        assert len(request.completions) == 1
        assert request.completions[0]["id"] == "1"

class Test_StreamingCompletionsRequest_ProcessChunk_04_ErrorHandlingBehaviors:
    """Test error handling behaviors of the StreamingCompletionsRequest.process_chunk method."""
    
    @pytest.mark.parametrize("chunk,expected_results", [
        (b'data: not_valid_json\n\n', []),  # Invalid JSON
        (b'\xc3\x28 invalid utf-8', []),  # Invalid UTF-8
        (b'random text without data prefix', []),  # No data prefix
        (b'data: {"id": "1"}\ndata: {"id": "2"}\n\n', [{"id": "1"}, {"id": "2"}]),  # Multiple data lines
    ])
    def test_handling_invalid_json_and_encoding_issues(self, chunk, expected_results):
        """Test handling of invalid JSON and encoding issues in chunks."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        request = StreamingCompletionsRequest(
            client=client,
            endpoint="/v1/completions",
            headers={},
            prompt="Test"
        )
        
        # Override the logger to avoid test output pollution
        request.logger = MagicMock()
        
        # Act
        result = request.process_chunk(chunk)
        
        # Assert
        assert len(result) == len(expected_results)

class Test_StreamingCompletionsRequest_Cancel_01_NominalBehaviors:
    """Test nominal behaviors of the StreamingCompletionsRequest.cancel method."""
    
    def test_closing_underlying_connection(self):
        """Test closing of underlying connection when cancelling."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        request = StreamingCompletionsRequest(
            client=client,
            endpoint="/v1/completions",
            headers={},
            prompt="Test"
        )
        
        # Mock response with close method
        mock_response = MagicMock()
        request._response = mock_response
        
        # Act
        request.cancel()
        
        # Assert
        assert request._cancelled is True
        assert mock_response.close.called

class Test_StreamingCompletionsRequest_Cancel_05_StateTransitionBehaviors:
    """Test state transition behaviors of the StreamingCompletionsRequest.cancel method."""
    
    def test_setting_cancelled_flag(self):
        """Test that cancelled flag is set appropriately."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        request = StreamingCompletionsRequest(
            client=client,
            endpoint="/v1/completions",
            headers={},
            prompt="Test"
        )
        request._cancelled = False
        
        # Act
        request.cancel()
        
        # Assert
        assert request._cancelled is True

class Test_StreamingCompletionsRequest_ResumeStream_01_NominalBehaviors:
    """Test nominal behaviors of the StreamingCompletionsRequest.resume_stream method."""
    
    def test_reconstructing_request_with_accumulated_data(self):
        """Test reconstruction of request with accumulated data for resumption."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b'data: {"id": "1", "choices": [{"delta": {"content": " continued"}}]}\n\n']
        client.post.return_value = (mock_response, {})
        
        request = StreamingCompletionsRequest(
            client=client,
            endpoint="/v1/completions",
            headers={"Authorization": "Bearer test_token"},
            prompt="Original prompt",
            state_file="test_state.json"
        )
        
        # Mock completions with some accumulated content
        request.completions = [
            {"choices": [{"delta": {"content": "Previous "}}]},
            {"choices": [{"delta": {"content": "content"}}]}
        ]
        
        # Mock load_state to return test state
        mock_state = MagicMock()
        mock_state.endpoint = "/v1/completions"
        mock_state.headers = {"Authorization": "Bearer test_token"}
        mock_state.data = {"prompt": "Original prompt"}
        request.load_state = MagicMock(return_value=mock_state)
        
        # Mock process_chunk
        request.process_chunk = MagicMock(return_value=[{"id": "1", "choices": [{"delta": {"content": " continued"}}]}])
        
        # Act
        results = list(request.resume_stream())
        
        # Assert
        assert client.post.called
        expected_prompt = "Original promptPrevious content"
        actual_prompt = client.post.call_args[1]["json"]["prompt"]
        assert actual_prompt == expected_prompt
        assert len(results) == 1
        assert results == [{"id": "1", "choices": [{"delta": {"content": " continued"}}]}]

class Test_StreamingCompletionsRequest_ResumeStream_04_ErrorHandlingBehaviors:
    """Test error handling behaviors of the StreamingCompletionsRequest.resume_stream method."""
    
    @pytest.mark.parametrize("state_scenario,error_msg", [
        (None, "No state file found"),
        (MagicMock(data=None), "State does not contain request data"),
    ])
    def test_validating_state_integrity(self, state_scenario, error_msg):
        """Test validation of state integrity during resumption."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        request = StreamingCompletionsRequest(
            client=client,
            endpoint="/v1/completions",
            headers={},
            prompt="Test",
            state_file="test_state.json"
        )
        
        # Mock load_state to return the test scenario
        request.load_state = MagicMock(return_value=state_scenario)
        
        # Act & Assert
        with pytest.raises(ResumeError) as exc_info:
            list(request.resume_stream())
        
        assert error_msg in str(exc_info.value)

class Test_StreamingCompletionsRequest_GetResult_01_NominalBehaviors:
    """Test nominal behaviors of the StreamingCompletionsRequest.get_result method."""
    
    def test_returning_accumulated_completions(self):
        """Test returning of accumulated completions."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        request = StreamingCompletionsRequest(
            client=client,
            endpoint="/v1/completions",
            headers={},
            prompt="Test"
        )
        
        # Setup test completions
        test_completions = [
            {"id": "1", "choices": [{"delta": {"content": "Part 1"}}]},
            {"id": "1", "choices": [{"delta": {"content": "Part 2"}}]}
        ]
        request.completions = test_completions
        request.completed = True
        
        # Act
        result = request.get_result()
        
        # Assert
        assert result == test_completions

class Test_StreamingChatCompletionsRequest_Init_01_NominalBehaviors:
    """Test nominal behaviors of the StreamingChatCompletionsRequest.__init__ method."""
    
    def test_initialization_with_chat_messages(self):
        """Test initialization with chat messages."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        endpoint = "/v1/chat/completions"
        headers = {"Authorization": "Bearer test_token"}
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        
        # Act
        request = StreamingChatCompletionsRequest(
            client=client,
            endpoint=endpoint,
            headers=headers,
            messages=messages
        )
        
        # Assert
        assert request.client == client
        assert request.endpoint == endpoint
        assert request.headers == headers
        assert request.messages == messages
        assert request.params == {"stream": True}
        assert request.completions == []
        assert request._cancelled is False
        assert request._response is None

class Test_StreamingChatCompletionsRequest_ResumeStream_01_NominalBehaviors:
    """Test nominal behaviors of the StreamingChatCompletionsRequest.resume_stream method."""
    
    def test_appending_accumulated_response_to_message_history(self):
        """Test appending accumulated response to message history for resumption."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b'data: {"id": "1", "choices": [{"delta": {"content": " continued"}}]}\n\n']
        client.post.return_value = (mock_response, {})
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a story."}
        ]
        
        request = StreamingChatCompletionsRequest(
            client=client,
            endpoint="/v1/chat/completions",
            headers={"Authorization": "Bearer test_token"},
            messages=messages,
            state_file="test_state.json"
        )
        
        # Mock completions with accumulated content
        request.completions = [
            {"choices": [{"delta": {"content": "Once upon "}}]},
            {"choices": [{"delta": {"content": "a time"}}]}
        ]
        
        # Mock load_state to return test state
        mock_state = MagicMock()
        mock_state.endpoint = "/v1/chat/completions"
        mock_state.headers = {"Authorization": "Bearer test_token"}
        mock_state.data = {"messages": messages}
        request.load_state = MagicMock(return_value=mock_state)
        
        # Mock process_chunk
        request.process_chunk = MagicMock(return_value=[{"id": "1", "choices": [{"delta": {"content": " continued"}}]}])
        
        # Act
        results = list(request.resume_stream())
        
        # Assert
        assert client.post.called
        posted_messages = client.post.call_args[1]["json"]["messages"]
        assert len(posted_messages) == 3  # Original 2 + assistant response
        assert posted_messages[2]["role"] == "assistant"
        assert posted_messages[2]["content"] == "Once upon a time"
        assert len(results) == 1
        assert results == [{"id": "1", "choices": [{"delta": {"content": " continued"}}]}]

class Test_StreamingChatCompletionsRequest_ResumeStream_04_ErrorHandlingBehaviors:
    """Test error handling behaviors of the StreamingChatCompletionsRequest.resume_stream method."""
    
    @pytest.mark.parametrize("state_scenario,error_msg", [
        (None, "No state file found or state loading failed"),
        (MagicMock(data=None), "State does not contain request data"),
    ])
    def test_handling_missing_or_corrupted_state_data(self, state_scenario, error_msg):
        """Test handling of missing or corrupted state data during resumption."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        request = StreamingChatCompletionsRequest(
            client=client,
            endpoint="/v1/chat/completions",
            headers={},
            messages=[{"role": "user", "content": "Hello"}],
            state_file="test_state.json"
        )
        
        # Mock load_state to return the test scenario
        request.load_state = MagicMock(return_value=state_scenario)
        
        # Act & Assert
        with pytest.raises(ResumeError) as exc_info:
            list(request.resume_stream())
        
        assert error_msg in str(exc_info.value)

class Test_StreamingChatCompletionsRequest_ProcessChunk_01_NominalBehaviors:
    """Test nominal behaviors of the StreamingChatCompletionsRequest.process_chunk method."""
    
    @pytest.mark.parametrize("chunk,expected", [
        (
            b'data: {"id": "1", "choices": [{"delta": {"content": "Hello"}}]}\n\n', 
            [{"id": "1", "choices": [{"delta": {"content": "Hello"}}]}]
        ),
        (
            b'data: {"id": "1", "choices": [{"message": {"content": "Hello"}}]}\n\n',
            [{"id": "1", "choices": [{"message": {"content": "Hello"}}]}]
        ),
    ])
    def test_processing_delta_and_message_content_formats(self, chunk, expected):
        """Test processing of both delta and message content formats in chat completions."""
        # Arrange
        client = MagicMock(spec=SmartSurgeClient)
        request = StreamingChatCompletionsRequest(
            client=client,
            endpoint="/v1/chat/completions",
            headers={},
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Override the logger to avoid test output pollution
        request.logger = MagicMock()
        
        # Act
        result = request.process_chunk(chunk)
        
        # Assert
        assert len(result) == 1
        assert result[0]["id"] == expected[0]["id"]
        assert result[0]["choices"] == expected[0]["choices"]
