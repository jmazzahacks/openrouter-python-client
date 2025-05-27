from typing import List, Union

from pydantic import ValidationError
import pytest

from openrouter_client.models.completions import (
    CompletionsRequest,
    CompletionsResponse,
    CompletionsResponseChoice,
    CompletionsStreamResponse,
    LogProbs,
    Usage
)


class TestLogProbs_NominalBehaviors:
    @pytest.mark.parametrize(
        "tokens, token_logprobs, text_offset",
        [
            (["token1", "token2"], [-0.5, -0.6], [0, 6]),
            (["a", "b", "c"], [-0.1, -0.2, -0.3], [0, 1, 2]),
        ],
        ids=["multiple_tokens", "short_tokens"],
    )
    def test_successful_instantiation(self, tokens: List[str], token_logprobs: List[float], text_offset: List[int]):
        """Tests successful instantiation of LogProbs with valid inputs."""
        log_probs = LogProbs(tokens=tokens, token_logprobs=token_logprobs, text_offset=text_offset)
        assert log_probs.tokens == tokens
        assert log_probs.token_logprobs == token_logprobs
        assert log_probs.text_offset == text_offset

    def test_proper_field_access(self):
        """Tests if fields are properly accessible."""
        tokens = ["test_token"]
        token_logprobs = [-0.7]
        top_logprobs = [{"test_token": -0.1}]
        text_offset = [0]
        log_probs = LogProbs(tokens=tokens, token_logprobs=token_logprobs, top_logprobs=top_logprobs, text_offset=text_offset)
        assert log_probs.tokens == tokens
        assert log_probs.token_logprobs == token_logprobs
        assert log_probs.top_logprobs == top_logprobs
        assert log_probs.text_offset == text_offset

class TestLogProbs_NegativeBehaviors:
    def test_mismatched_list_lengths(self):
        """Tests instantiation failure with mismatched list lengths."""
        with pytest.raises(ValueError):
            LogProbs(tokens=["token1", "token2"], token_logprobs=[-0.5], text_offset=[0, 5])

    @pytest.mark.parametrize(
        "tokens, token_logprobs, text_offset",
        [
            ([sum, id], [-0.5, -0.6], [0, 6]),
            (["a", "b"], ["negative 0.1", "negative 0.2"], [0, 1]),
            (["a", "b"], [-0.1, -0.2], [0.5, 1.5]),
        ],
        ids=["invalid_tokens_type", "invalid_logprobs_type", "invalid_offset_type"],
    )
    def test_invalid_data_types(self, tokens: List[str], token_logprobs: List[float], text_offset: List[int]):
        """Tests instantiation failure with invalid data types in fields."""
        with pytest.raises(ValueError):
            LogProbs(tokens=tokens, token_logprobs=token_logprobs, text_offset=text_offset)

class TestLogProbs_03_BoundaryBehaviors:
    def test_empty_lists(self):
        """Tests instantiation with empty lists for all fields."""
        log_probs = LogProbs(tokens=[], token_logprobs=[], text_offset=[])
        assert log_probs.tokens == []
        assert log_probs.token_logprobs == []
        assert log_probs.text_offset == []

class TestLogProbs_04_ErrorHandlingBehaviors:
    def test_type_validation_errors(self):
        """Tests that validation errors are raised for type mismatches."""
        with pytest.raises(ValueError):
            LogProbs(tokens=["a"], token_logprobs=["b"], text_offset=[1])

class TestLogProbs_05_StateTransitionBehaviors:
    def test_serialization_deserialization_consistency(self):
        """Tests that the model maintains data integrity across serialization/deserialization."""
        data = {"tokens": ["a", "b"], "token_logprobs": [-0.1, -0.2], "text_offset": [0, 1]}
        log_probs = LogProbs(**data)
        serialized = log_probs.model_dump_json()
        deserialized = LogProbs.model_validate_json(serialized)
        assert log_probs == deserialized

class TestCompletionsRequest_NominalBehaviors:
    @pytest.mark.parametrize(
        "prompt, model",
        [
            ("Test prompt", "test-model")
        ]
    )
    def test_successful_instantiation_minimal(self, prompt: Union[str, List[str]], model: str):
        """Tests successful instantiation with minimal required fields."""
        request = CompletionsRequest(prompt=prompt, model=model)
        assert request.prompt == prompt
        assert request.model == model

    def test_proper_constraint_validation(self):
        """Tests correct validation for bounded parameters."""
        request = CompletionsRequest(prompt="test", model="test", temperature=1.0, top_p=0.5, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, min_p = 0.5, top_a = 0.5)
        assert request.temperature == 1.0
        assert request.top_p == 0.5
        assert request.presence_penalty == 0.0
        assert request.frequency_penalty == 0.0

class TestCompletionsRequest_NegativeBehaviors:
    @pytest.mark.parametrize(
        "temperature, top_p, presence_penalty, frequency_penalty, repetition_penalty, min_p, top_a",
        [
            (2.1, 0.5, 0.0, 0.0, 1.0, 0.5, 0.5),
            (1.0, 1.1, 0.0, 0.0, 1.0, 0.5, 0.5),
            (1.0, 0.5, -2.1, 0.0, 1.0, 0.5, 0.5),
            (1.0, 0.5, 0.0, 2.1, 1.0, 0.5, 0.5),
            (1.0, 0.5, 0.0, 0.0, 2.1, 0.5, 0.5),
            (1.0, 0.5, 0.0, 0.0, 1.0, -0.1, 0.5),
            (1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 1.1),
        ],
        ids=["invalid_temperature", "invalid_top_p", "invalid_presence_penalty", "invalid_frequency_penalty", "invalid_repetition_penalty", "invalid_min_p", "invalid_top_a"],
    )
    def test_out_of_range_values(self, temperature: float, top_p: float, presence_penalty: float, frequency_penalty: float, repetition_penalty: float, min_p: float, top_a: float):
        """Tests instantiation failure with out-of-range parameter values."""
        with pytest.raises(ValueError):
            CompletionsRequest(prompt="test", model="test", temperature=temperature, top_p=top_p, presence_penalty=presence_penalty, frequency_penalty=frequency_penalty, repetition_penalty = repetition_penalty, min_p = min_p, top_a = top_a)

    def test_invalid_model_identifier(self):
        """Tests instantiation failure with an invalid model identifier."""
        with pytest.raises(ValueError):
            CompletionsRequest(prompt="test", model="")

class TestCompletionsRequest_03_BoundaryBehaviors:
    def test_parameter_values_at_minimum(self):
        """Tests instantiation with parameter values at exact min boundaries."""
        request = CompletionsRequest(
            prompt="test", 
            model="test", 
            temperature=0.0, 
            top_p=1e-10, 
            presence_penalty=-2.0, 
            frequency_penalty=-2.0, 
            repetition_penalty=1e-10, 
            min_p = 0.0, 
            top_a = 0.0
        )
        assert request.temperature == 0.0
        assert request.top_p == 1e-10
        assert request.presence_penalty == -2.0
        assert request.frequency_penalty == -2.0
        assert request.repetition_penalty == 1e-10
        assert request.min_p == 0.0
        assert request.top_a == 0.0
        
    def test_parameter_values_at_maximum(self):
        """Tests instantiation with parameter values at exact max boundaries."""
        request = CompletionsRequest(
            prompt="test", 
            model="test", 
            temperature=2.0, 
            top_p=1.0, 
            presence_penalty=2.0, 
            frequency_penalty=2.0, 
            repetition_penalty=2.0, 
            min_p = 1.0, 
            top_a = 1.0
        )
        assert request.temperature == 2.0
        assert request.top_p == 1.0
        assert request.presence_penalty == 2.0
        assert request.frequency_penalty == 2.0
        assert request.repetition_penalty == 2.0
        assert request.min_p == 1.0
        assert request.top_a == 1.0

    def test_minimum_max_tokens(self):
        """Tests that max_tokens must be at least 1."""
        request = CompletionsRequest(prompt="test", model="test", max_tokens=1)
        assert request.max_tokens == 1

class TestCompletionsRequest_04_ErrorHandlingBehaviors:
    def test_constraint_violation_error_handling(self):
        """Tests that validation errors are raised for constraint violations."""
        with pytest.raises(ValueError):
            CompletionsRequest(prompt="test", model="test", temperature=3.0)

class TestCompletionsRequest_05_StateTransitionBehaviors:
    def test_field_interdependency_validation_state(self):
        """Tests that the model maintains consistent validation state across field updates."""
        request = CompletionsRequest(prompt="test", model="test")
        with pytest.raises(ValueError):
            request.model_validate({"prompt": "test", "model": "test", "temperature": 3.0})

class TestCompletionsRequest_ValidateFunctionAndTools_01_NominalBehaviors:
    def test_successful_validation_functions_only(self):
        """Tests successful validation when only functions is specified."""
        request = CompletionsRequest(prompt="test", model="test", functions=[])
        assert request.validate_function_and_tools() == request

    def test_successful_validation_tools_only(self):
        """Tests successful validation when only tools is specified."""
        request = CompletionsRequest(prompt="test", model="test", tools=[])
        assert request.validate_function_and_tools() == request

    def test_successful_validation_neither(self):
        """Tests successful validation when neither functions nor tools is specified."""
        request = CompletionsRequest(prompt="test", model="test")
        assert request.validate_function_and_tools() == request

class TestCompletionsRequest_ValidateFunctionAndTools_02_NegativeBehaviors:
    def test_value_error_both_specified(self):
        """Tests ValueError is raised when both functions and tools are specified."""
        with pytest.raises(ValueError):
            CompletionsRequest(prompt="test", model="test", functions=[], tools=[]).validate_function_and_tools()

class TestCompletionsRequest_ValidateFunctionAndTools_03_BoundaryBehaviors:
    def test_validation_with_empty_lists(self):
        """Tests validation with empty lists for functions and tools."""
        request = CompletionsRequest(prompt="test", model="test")
        request.functions = []
        request.tools = []
        with pytest.raises(ValueError):
            request.validate_function_and_tools()

class TestCompletionsRequest_ValidateFunctionAndTools_04_ErrorHandlingBehaviors:
    def test_proper_exception_type(self):
        """Tests that the correct exception type (ValueError) is raised."""
        with pytest.raises(ValueError):
            CompletionsRequest(prompt="test", model="test", functions=[], tools=[]).validate_function_and_tools()

class TestCompletionsRequest_ValidateFunctionAndTools_05_StateTransitionBehaviors:
    def test_model_validation_state_consistency(self):
        """Tests that the model maintains consistent validation state."""
        request = CompletionsRequest(prompt="test", model="test")
        try:
            request.validate_function_and_tools()
        except ValueError:
            pass  # Expecting ValueError due to missing functions/tools

class TestCompletionsResponseChoice_NominalBehaviors:
    def test_successful_instantiation(self):
        """Tests successful instantiation with required fields."""
        choice = CompletionsResponseChoice(text="Test completion", index=0)
        assert choice.text == "Test completion"
        assert choice.index == 0

class TestCompletionsResponseChoice_NegativeBehaviors:
    def test_invalid_index_value(self):
        """Tests instantiation failure with a negative index."""
        with pytest.raises(ValidationError):
            CompletionsResponseChoice(text="Test", index=-1)

    def test_type_mismatch_text(self):
         """Tests instantiation failure with a non-string text value."""
         with pytest.raises(ValueError):
            CompletionsResponseChoice(text=lambda x: x, index=0)

class TestCompletionsResponseChoice_03_BoundaryBehaviors:
    def test_empty_string_text(self):
        """Tests instantiation with an empty string for the text field."""
        choice = CompletionsResponseChoice(text="", index=0)
        assert choice.text == ""

    def test_maximum_index_value(self):
        """Tests instantiation with a maximum reasonable index value."""
        choice = CompletionsResponseChoice(text="test", index=1000)
        assert choice.index == 1000

class TestCompletionsResponseChoice_04_ErrorHandlingBehaviors:
    def test_required_field_validation_errors(self):
        """Tests that validation errors are raised for missing required fields."""
        with pytest.raises(ValidationError):
            CompletionsResponseChoice(index=0)

class TestCompletionsResponseChoice_05_StateTransitionBehaviors:
    def test_optional_field_state_management(self):
        """Tests that optional fields can transition between None and a value."""
        choice = CompletionsResponseChoice(text="Test", index=0)
        choice.finish_reason = "stop"
        assert choice.finish_reason == "stop"
        choice.finish_reason = None
        assert choice.finish_reason is None

class TestCompletionsResponse_NominalBehaviors:
    def test_successful_instantiation(self):
        """Tests successful instantiation with all required fields."""
        choice = CompletionsResponseChoice(text="Test completion", index=0)
        response = CompletionsResponse(id="test-id", object="chat.completion", created=1678886400, model="test-model", choices=[choice])
        assert response.id == "test-id"
        assert response.object == "chat.completion"
        assert response.created == 1678886400
        assert response.model == "test-model"
        assert response.choices == [choice]

    def test_proper_timestamp_handling(self):
        """Tests proper handling of the created timestamp field."""
        response = CompletionsResponse(id="test-id", object="chat.completion", created=1678886400, model="test-model", choices=[CompletionsResponseChoice(text="test", index=0)])
        assert response.created == 1678886400

class TestCompletionsResponse_NegativeBehaviors:
    def test_invalid_timestamp_format(self):
        """Tests instantiation failure with an invalid timestamp format."""
        with pytest.raises(ValueError):
            CompletionsResponse(id="test", object="chat.completion", created="invalid", model="test", choices=[CompletionsResponseChoice(text="test", index=0)])

    def test_empty_choices_list(self):
        """Tests that an empty choices list is acceptable."""
        response = CompletionsResponse(id="test", object="chat.completion", created=1678886400, model="test", choices=[])
        assert response.choices == []

class TestCompletionsResponse_03_BoundaryBehaviors:
    def test_minimum_timestamp_value(self):
        """Tests instantiation with a minimum reasonable timestamp value."""
        response = CompletionsResponse(id="test", object="chat.completion", created=0, model="test", choices=[CompletionsResponseChoice(text="test", index=0)])
        assert response.created == 0

class TestCompletionsResponse_04_ErrorHandlingBehaviors:
    def test_required_field_validation_failures(self):
        """Tests that validation errors occur when required fields are missing."""
        with pytest.raises(ValidationError):
            CompletionsResponse(object="chat.completion", created=0, model="test", choices=[CompletionsResponseChoice(text="test", index=0)])

class TestCompletionsResponse_05_StateTransitionBehaviors:
    def test_response_object_consistency(self):
        """Tests that response objects maintain integrity during processing."""
        choice = CompletionsResponseChoice(text="Test completion", index=0)
        response = CompletionsResponse(id="test-id", object="chat.completion", created=1678886400, model="test-model", choices=[choice])
        response.model = "new-model"
        assert response.model == "new-model"

class TestCompletionStreamResponse_NominalBehaviors:
    def test_successful_instantiation(self):
        """Tests successful instantiation with streaming-specific object type."""
        choice = CompletionsResponseChoice(text="Test completion", index=0)
        response = CompletionsStreamResponse(id="stream-id", object="chat.completion.chunk", created=1678886400, model="test-model", choices=[choice])
        assert response.object == "chat.completion.chunk"

class TestCompletionStreamResponse_NegativeBehaviors:
    def test_invalid_object_type(self):
        """Tests instantiation failure with an invalid object type for streaming."""
        with pytest.raises(ValueError):
            CompletionsStreamResponse(id="test", object=lambda x: x, created=1678886400, model="test", choices=[CompletionsResponseChoice(text="test", index=0)])

class TestCompletionStreamResponse_03_BoundaryBehaviors:
    def test_single_chunk_sequence(self):
        """Tests that single-chunk sequences are handled correctly."""
        choice = CompletionsResponseChoice(text="Test completion", index=0)
        response = CompletionsStreamResponse(id="stream-id", object="chat.completion.chunk", created=1678886400, model="test-model", choices=[choice])
        assert response.choices == [choice]

class TestCompletionStreamResponse_04_ErrorHandlingBehaviors:
    def test_streaming_protocol_constraint_validation(self):
        """Tests validation of streaming-specific constraints."""
        choice = CompletionsResponseChoice(text="Test completion", index=0)
        with pytest.raises(ValueError):
            CompletionsStreamResponse(id="test", object="chat.completion", created=1678886400, model="test", choices=[choice], usage=Usage(request_tokens = 0, response_tokens = 0, total_tokens = 0, requests = 0))

class TestCompletionStreamResponse_05_StateTransitionBehaviors:
    def test_usage_field_state_transitions(self):
        """Tests that usage statistics appear correctly in final chunks."""
        choice = CompletionsResponseChoice(text="Test completion", index=0)
        usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        response = CompletionsStreamResponse(id="stream-id", object="chat.completion.chunk", created=1678886400, model="test-model", choices=[choice], usage=usage)
        assert response.usage == usage
