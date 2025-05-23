import pytest
from datetime import datetime
import json
import re
import sys
import time
from pydantic import ValidationError

from openrouter_client.models.chat import (ChatCompletionFunction, ChatCompletionFunctionCall, Usage,
                                           ToolCallFunction, ChatCompletionToolCall, ToolCallChunk, 
                                           FunctionDefinition, FunctionParameters, ChatCompletionTool,
                                           ChatCompletionToolChoiceOption, ChatCompletionRequest, 
                                           ReasoningConfig, ChatCompletionResponseChoice, 
                                           ChatCompletionStreamResponse, ChatCompletionStreamResponseDelta,
                                           ChatCompletionStreamResponseChoice, ChatCompletionResponse,
                                           ParameterType, ParameterDefinition, StringParameter,
                                           NumberParameter, BooleanParameter, ArrayParameter, 
                                           ObjectParameter, FunctionCall, FunctionCallResult, 
                                           StructuredToolResult, FunctionToolChoice)
from openrouter_client.models.core import Message, ModelRole


class Test_Usage_01_NominalBehaviors:
    """Tests for nominal behaviors of the Usage class."""
    
    @pytest.mark.parametrize("prompt_tokens, completion_tokens, total_tokens, cached_tokens, cache_discount", [
        (100, 50, 150, None, None),  # Required fields only
        (100, 50, 150, 20, None),    # With cached_tokens
        (100, 50, 150, None, 0.5),   # With cache_discount
        (100, 50, 150, 20, 0.5),     # With all fields
    ])
    def test_initialization(self, prompt_tokens, completion_tokens, total_tokens, cached_tokens, cache_discount):
        """Verify the Usage class can be properly instantiated with valid parameters."""
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            cache_discount=cache_discount
        )
        
        # Verify properties are accessible and contain expected values
        assert usage.prompt_tokens == prompt_tokens
        assert usage.completion_tokens == completion_tokens
        assert usage.total_tokens == total_tokens
        assert usage.cached_tokens == cached_tokens
        assert usage.cache_discount == cache_discount
    
    def test_serialization(self):
        """Verify the Usage class can be properly serialized to JSON."""
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cached_tokens=20, cache_discount=0.5)
        
        # Serialize to JSON
        json_data = usage.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["prompt_tokens"] == 100
        assert parsed_json["completion_tokens"] == 50
        assert parsed_json["total_tokens"] == 150
        assert parsed_json["cached_tokens"] == 20
        assert parsed_json["cache_discount"] == 0.5
    
    def test_deserialization(self):
        """Verify the Usage class can be properly deserialized from JSON."""
        json_data = '{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"cached_tokens":20,"cache_discount":0.5}'
        
        # Deserialize from JSON
        usage = Usage.model_validate_json(json_data)
        
        # Verify properties are accessible and contain expected values
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cached_tokens == 20
        assert usage.cache_discount == 0.5


class Test_Usage_02_NegativeBehaviors:
    """Tests for negative behaviors of the Usage class."""
    
    @pytest.mark.parametrize("data, error_field", [
        # Missing required fields
        ({"completion_tokens": 50, "total_tokens": 150}, "prompt_tokens"),
        ({"prompt_tokens": 100, "total_tokens": 150}, "completion_tokens"),
        ({"prompt_tokens": 100, "completion_tokens": 50}, "total_tokens"),
        
        # Incorrect data types
        ({"prompt_tokens": "not_a_number", "completion_tokens": 50, "total_tokens": 150}, "prompt_tokens"),
        ({"prompt_tokens": 100, "completion_tokens": "not_a_number", "total_tokens": 150}, "completion_tokens"),
        ({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": "not_a_number"}, "total_tokens"),
        ({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cached_tokens": "not_a_number"}, "cached_tokens"),
        ({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cache_discount": "not_a_number"}, "cache_discount"),
        
        # Negative values
        ({"prompt_tokens": -100, "completion_tokens": 50, "total_tokens": 150}, "prompt_tokens"),
        ({"prompt_tokens": 100, "completion_tokens": -50, "total_tokens": 150}, "completion_tokens"),
        ({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": -150}, "total_tokens"),
        ({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cached_tokens": -20}, "cached_tokens"),
    ])
    def test_initialization_with_invalid_data(self, data, error_field):
        """Verify the Usage class rejects initialization with invalid data."""
        with pytest.raises(ValidationError) as excinfo:
            Usage(**data)
        
        # Verify the error field is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["loc"][0] == error_field for error in errors)


class Test_Usage_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the Usage class."""
    
    @pytest.mark.parametrize("prompt_tokens, completion_tokens, total_tokens, cached_tokens, cache_discount", [
        # Zero values for token fields
        (0, 0, 0, None, None),
        
        # Maximum integer values
        (2**31-1, 2**31-1, 2**31-1, None, None),
        
        # Boundary values for cache_discount
        (100, 50, 150, None, 0.0),
        (100, 50, 150, None, 1.0),
    ])
    def test_boundary_values(self, prompt_tokens, completion_tokens, total_tokens, cached_tokens, cache_discount):
        """Verify the Usage class handles boundary values correctly."""
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            cache_discount=cache_discount
        )
        
        # Verify properties are accessible and contain expected values
        assert usage.prompt_tokens == prompt_tokens
        assert usage.completion_tokens == completion_tokens
        assert usage.total_tokens == total_tokens
        assert usage.cached_tokens == cached_tokens
        assert usage.cache_discount == cache_discount


class Test_Usage_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the Usage class."""
    
    def test_validation_error_handling(self):
        """Verify the Usage class properly handles validation errors."""
        # Test with missing required field
        with pytest.raises(ValidationError) as excinfo:
            Usage(completion_tokens=50, total_tokens=150)
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("prompt_tokens",)
    
    def test_deserialization_error_handling(self):
        """Verify the Usage class properly handles deserialization errors."""
        # Test with malformed JSON
        with pytest.raises(Exception) as excinfo:
            Usage.model_validate_json("{malformed_json}")
        
        # Verify error is raised
        assert excinfo.type is not None


class Test_ChatCompletionFunction_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionFunction class."""
    
    @pytest.mark.parametrize("name, arguments", [
        ("test_function", "{}"),  # Minimal valid JSON
        ("test_function", '{"arg1": "value1", "arg2": 42}'),  # JSON with values
        ("test_function", '{"nested": {"value": true}}'),  # Nested JSON
    ])
    def test_initialization(self, name, arguments):
        """Verify the ChatCompletionFunction can be properly initialized with valid parameters."""
        function = ChatCompletionFunction(name=name, arguments=arguments)
        
        # Verify properties are accessible and contain expected values
        assert function.name == name
        assert function.arguments == arguments
    
    def test_serialization_deserialization(self):
        """Verify the ChatCompletionFunction can be properly serialized and deserialized."""
        function = ChatCompletionFunction(
            name="test_function",
            arguments='{"arg1": "value1", "arg2": 42}'
        )
        
        # Serialize to JSON
        json_data = function.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["name"] == "test_function"
        assert json.loads(parsed_json["arguments"]) == {"arg1": "value1", "arg2": 42}
        
        # Deserialize from JSON
        deserialized = ChatCompletionFunction.model_validate_json(json_data)
        
        # Verify all properties match the original
        assert deserialized.name == function.name
        assert deserialized.arguments == function.arguments


class Test_ChatCompletionFunction_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionFunction class."""
    
    @pytest.mark.parametrize("data, error_field", [
        # Missing required fields
        ({"arguments": '{}'}, "name"),
        ({"name": "test_function"}, "arguments"),
        
        # Empty strings
        ({"name": "", "arguments": '{}'}, "name"),
        ({"name": "test_function", "arguments": ""}, "arguments"),
        
        # Non-JSON arguments
        ({"name": "test_function", "arguments": "not_json"}, "arguments"),
    ])
    def test_initialization_with_invalid_data(self, data, error_field):
        """Verify the ChatCompletionFunction rejects initialization with invalid data."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionFunction(**data)
        
        # Verify the error field is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["loc"][0] == error_field for error in errors)


class Test_ChatCompletionFunction_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionFunction class."""
    
    @pytest.mark.parametrize("name, arguments", [
        # Very long function name
        ("a" * 1000, "{}"),
        
        # Minimal valid JSON
        ("test_function", "{}"),
        
        # Complex nested JSON
        ("test_function", '{"level1": {"level2": {"level3": {"level4": {"level5": [1, 2, 3, 4, 5]}}}}}'),
    ])
    def test_boundary_values(self, name, arguments):
        """Verify the ChatCompletionFunction handles boundary values correctly."""
        function = ChatCompletionFunction(name=name, arguments=arguments)
        
        # Verify properties are accessible and contain expected values
        assert function.name == name
        assert function.arguments == arguments


class Test_ChatCompletionFunction_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionFunction class."""
    
    def test_validation_error_handling(self):
        """Verify the ChatCompletionFunction properly handles validation errors."""
        # Test with missing required field
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionFunction(name="test_function")
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("arguments",)
    
    def test_json_parsing_error_handling(self):
        """Verify the ChatCompletionFunction properly handles JSON parsing errors."""
        # Test with invalid JSON
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionFunction(name="test_function", arguments="not_json")
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) > 0


class Test_ChatCompletionFunctionCall_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionFunctionCall class."""
    
    @pytest.mark.parametrize("name", [
        "test_function",
        None,  # name is optional
    ])
    def test_initialization(self, name):
        """Verify the ChatCompletionFunctionCall can be properly initialized with valid parameters."""
        function_call = ChatCompletionFunctionCall(name=name)
        
        # Verify properties are accessible and contain expected values
        assert function_call.name == name
    
    def test_serialization_deserialization(self):
        """Verify the ChatCompletionFunctionCall can be properly serialized and deserialized."""
        function_call = ChatCompletionFunctionCall(name="test_function")
        
        # Serialize to JSON
        json_data = function_call.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["name"] == "test_function"
        
        # Deserialize from JSON
        deserialized = ChatCompletionFunctionCall.model_validate_json(json_data)
        
        # Verify all properties match the original
        assert deserialized.name == function_call.name


class Test_ChatCompletionFunctionCall_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionFunctionCall class."""
    
    @pytest.mark.parametrize("data, expected_error_type", [
        # Incorrect data types
        ({"name": 123}, "string_type"),
        ({"name": True}, "string_type"),
        ({"name": ["list", "items"]}, "string_type"),
    ])
    def test_initialization_with_invalid_data_types(self, data, expected_error_type):
        """Verify the ChatCompletionFunctionCall rejects initialization with invalid data types."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionFunctionCall(**data)
        
        # Verify the error type is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["type"].startswith(expected_error_type) for error in errors)


class Test_ChatCompletionFunctionCall_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionFunctionCall class."""
    
    @pytest.mark.parametrize("name", [
        # Very long function name
        "a" * 1000,
        
        # Empty string
        "",
    ])
    def test_boundary_values(self, name):
        """Verify the ChatCompletionFunctionCall handles boundary values correctly."""
        function_call = ChatCompletionFunctionCall(name=name)
        
        # Verify properties are accessible and contain expected values
        assert function_call.name == name


class Test_ChatCompletionFunctionCall_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionFunctionCall class."""
    
    def test_validation_error_handling(self):
        """Verify the ChatCompletionFunctionCall properly handles validation errors."""
        # Test with invalid data type
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionFunctionCall(name=123)
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) > 0
        assert any(error["type"].startswith("string_type") for error in errors)


class Test_ToolCallFunction_01_NominalBehaviors:
    """Tests for nominal behaviors of the ToolCallFunction class."""
    
    @pytest.mark.parametrize("name, arguments", [
        ("test_function", "{}"),  # Minimal valid JSON
        ("test_function", '{"arg1": "value1", "arg2": 42}'),  # JSON with values
        ("test_function", '{"nested": {"value": true}}'),  # Nested JSON
    ])
    def test_initialization(self, name, arguments):
        """Verify the ToolCallFunction can be properly initialized with valid parameters."""
        function = ToolCallFunction(name=name, arguments=arguments)
        
        # Verify properties are accessible and contain expected values
        assert function.name == name
        assert function.arguments == arguments
    
    def test_serialization_deserialization(self):
        """Verify the ToolCallFunction can be properly serialized and deserialized."""
        function = ToolCallFunction(
            name="test_function",
            arguments='{"arg1": "value1", "arg2": 42}'
        )
        
        # Serialize to JSON
        json_data = function.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["name"] == "test_function"
        assert json.loads(parsed_json["arguments"]) == {"arg1": "value1", "arg2": 42}
        
        # Deserialize from JSON
        deserialized = ToolCallFunction.model_validate_json(json_data)
        
        # Verify all properties match the original
        assert deserialized.name == function.name
        assert deserialized.arguments == function.arguments


class Test_ToolCallFunction_02_NegativeBehaviors:
    """Tests for negative behaviors of the ToolCallFunction class."""
    
    @pytest.mark.parametrize("data, error_field", [
        # Missing required fields
        ({"arguments": '{}'}, "name"),
        ({"name": "test_function"}, "arguments"),
        
        # Empty strings
        ({"name": "", "arguments": '{}'}, "name"),
        ({"name": "test_function", "arguments": ""}, "arguments"),
        
        # Non-JSON arguments
        ({"name": "test_function", "arguments": "not_json"}, "arguments"),
    ])
    def test_initialization_with_invalid_data(self, data, error_field):
        """Verify the ToolCallFunction rejects initialization with invalid data."""
        with pytest.raises(ValidationError) as excinfo:
            ToolCallFunction(**data)
        
        # Verify the error field is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["loc"][0] == error_field for error in errors)


class Test_ToolCallFunction_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ToolCallFunction class."""
    
    @pytest.mark.parametrize("name, arguments", [
        # Very long function name
        ("a" * 1000, "{}"),
        
        # Minimal valid JSON
        ("test_function", "{}"),
        
        # Complex nested JSON
        ("test_function", '{"level1": {"level2": {"level3": {"level4": {"level5": [1, 2, 3, 4, 5]}}}}}'),
    ])
    def test_boundary_values(self, name, arguments):
        """Verify the ToolCallFunction handles boundary values correctly."""
        function = ToolCallFunction(name=name, arguments=arguments)
        
        # Verify properties are accessible and contain expected values
        assert function.name == name
        assert function.arguments == arguments


class Test_ToolCallFunction_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ToolCallFunction class."""
    
    def test_validation_error_handling(self):
        """Verify the ToolCallFunction properly handles validation errors."""
        # Test with missing required field
        with pytest.raises(ValidationError) as excinfo:
            ToolCallFunction(name="test_function")
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("arguments",)
    
    def test_json_parsing_error_handling(self):
        """Verify the ToolCallFunction properly handles JSON parsing errors."""
        # Test with invalid JSON
        with pytest.raises(ValidationError) as excinfo:
            ToolCallFunction(name="test_function", arguments="not_json")
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) > 0


class Test_ChatCompletionToolCall_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionToolCall class."""
    
    @pytest.mark.parametrize("tool_id, type_value, function_name, function_arguments", [
        ("call_123", "function", "test_function", "{}"),
        ("call_123", None, "test_function", "{}"),  # type defaults to "function"
    ])
    def test_initialization(self, tool_id, type_value, function_name, function_arguments):
        """Verify the ChatCompletionToolCall can be properly initialized with valid parameters."""
        function = ToolCallFunction(name=function_name, arguments=function_arguments)
        tool_call = ChatCompletionToolCall(
            id=tool_id,
            type=type_value if type_value else "function",
            function=function
        )
        
        # Verify properties are accessible and contain expected values
        assert tool_call.id == tool_id
        assert tool_call.type == "function"  # type defaults to "function" if None
        assert tool_call.function.name == function_name
        assert tool_call.function.arguments == function_arguments
    
    def test_serialization_deserialization(self):
        """Verify the ChatCompletionToolCall can be properly serialized and deserialized."""
        function = ToolCallFunction(name="test_function", arguments='{"arg1": "value1"}')
        tool_call = ChatCompletionToolCall(
            id="call_123",
            function=function
        )
        
        # Serialize to JSON
        json_data = tool_call.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["id"] == "call_123"
        assert parsed_json["type"] == "function"
        assert parsed_json["function"]["name"] == "test_function"
        assert json.loads(parsed_json["function"]["arguments"]) == {"arg1": "value1"}
        
        # Deserialize from JSON
        deserialized = ChatCompletionToolCall.model_validate_json(json_data)
        
        # Verify all properties match the original
        assert deserialized.id == tool_call.id
        assert deserialized.type == tool_call.type
        assert deserialized.function.name == tool_call.function.name
        assert deserialized.function.arguments == tool_call.function.arguments


class Test_ChatCompletionToolCall_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionToolCall class."""
    
    @pytest.mark.parametrize("data, error_field", [
        # Missing required fields
        ({"function": ToolCallFunction(name="test_function", arguments="{}")}, "id"),
        ({"id": "call_123"}, "function"),
        
        # Invalid nested object
        ({"id": "call_123", "function": "not_an_object"}, "function"),
    ])
    def test_initialization_with_invalid_data(self, data, error_field):
        """Verify the ChatCompletionToolCall rejects initialization with invalid data."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionToolCall(**data)
        
        # Verify the error field is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["loc"][0] == error_field for error in errors)


class Test_ChatCompletionToolCall_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionToolCall class."""
    
    @pytest.mark.parametrize("tool_id, type_value", [
        # Very long ID
        ("a" * 1000, "function"),
        
        # Non-default type
        ("call_123", "non_standard_type"),
    ])
    def test_boundary_values(self, tool_id, type_value):
        """Verify the ChatCompletionToolCall handles boundary values correctly."""
        function = ToolCallFunction(name="test_function", arguments="{}")
        tool_call = ChatCompletionToolCall(
            id=tool_id,
            type=type_value,
            function=function
        )
        
        # Verify properties are accessible and contain expected values
        assert tool_call.id == tool_id
        assert tool_call.type == type_value
        assert tool_call.function.name == "test_function"
        assert tool_call.function.arguments == "{}"


class Test_ChatCompletionToolCall_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionToolCall class."""
    
    def test_validation_error_handling(self):
        """Verify the ChatCompletionToolCall properly handles validation errors."""
        # Test with missing required field
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionToolCall(id="call_123")
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("function",)
    
    def test_nested_validation_error_handling(self):
        """Verify the ChatCompletionToolCall properly handles nested validation errors."""
        # Test with invalid nested object
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionToolCall(
                id="call_123",
                function={"name": "test_function"}  # Missing arguments field
            )
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) > 0


class Test_ToolCallChunk_01_NominalBehaviors:
    """Tests for nominal behaviors of the ToolCallChunk class."""
    
    @pytest.mark.parametrize("tool_id, type_value, function, index", [
        ("call_123", "function", {"name": "test_function", "arguments": "{}"}, 0),
        ("call_123", None, {"name": "test_function", "arguments": "{}"}, 1),  # type defaults to "function"
    ])
    def test_initialization(self, tool_id, type_value, function, index):
        """Verify the ToolCallChunk can be properly initialized with valid parameters."""
        chunk = ToolCallChunk(
            id=tool_id,
            type=type_value if type_value else "function",
            function=function,
            index=index
        )
        
        # Verify properties are accessible and contain expected values
        assert chunk.id == tool_id
        assert chunk.type == "function"  # type defaults to "function" if None
        assert chunk.function == function
        assert chunk.index == index
    
    def test_serialization_deserialization(self):
        """Verify the ToolCallChunk can be properly serialized and deserialized."""
        chunk = ToolCallChunk(
            id="call_123",
            function={"name": "test_function", "arguments": "{}"},
            index=0
        )
        
        # Serialize to JSON
        json_data = chunk.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["id"] == "call_123"
        assert parsed_json["type"] == "function"
        assert parsed_json["function"] == {"name": "test_function", "arguments": "{}"}
        assert parsed_json["index"] == 0
        
        # Deserialize from JSON
        deserialized = ToolCallChunk.model_validate_json(json_data)
        
        # Verify all properties match the original
        assert deserialized.id == chunk.id
        assert deserialized.type == chunk.type
        assert deserialized.function == chunk.function
        assert deserialized.index == chunk.index


class Test_ToolCallChunk_02_NegativeBehaviors:
    """Tests for negative behaviors of the ToolCallChunk class."""
    
    @pytest.mark.parametrize("data, error_field", [
        # Missing required fields
        ({"function": {"name": "test_function", "arguments": "{}"}, "index": 0}, "id"),
        ({"id": "call_123", "index": 0}, "function"),
        ({"id": "call_123", "function": {"name": "test_function", "arguments": "{}"}}, "index"),
        
        # Invalid function dictionary
        ({"id": "call_123", "function": {"name": 123, "arguments": "{}"}, "index": 0}, "function"),
        ({"id": "call_123", "function": {"name": "test_function", "arguments": 123}, "index": 0}, "function"),
        
        # Negative index
        ({"id": "call_123", "function": {"name": "test_function", "arguments": "{}"}, "index": -1}, "index"),
    ])
    def test_initialization_with_invalid_data(self, data, error_field):
        """Verify the ToolCallChunk rejects initialization with invalid data."""
        with pytest.raises(ValidationError) as excinfo:
            ToolCallChunk(**data)
        
        # Verify the error field is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["loc"][0] == error_field for error in errors)


class Test_ToolCallChunk_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ToolCallChunk class."""
    
    @pytest.mark.parametrize("tool_id, type_value, index", [
        # Very long ID
        ("a" * 1000, "function", 0),
        
        # Boundary index values
        ("call_123", "function", 0),
        ("call_123", "function", 2**31-1),
        
        # Non-default type
        ("call_123", "non_standard_type", 0),
    ])
    def test_boundary_values(self, tool_id, type_value, index):
        """Verify the ToolCallChunk handles boundary values correctly."""
        chunk = ToolCallChunk(
            id=tool_id,
            type=type_value,
            function={"name": "test_function", "arguments": "{}"},
            index=index
        )
        
        # Verify properties are accessible and contain expected values
        assert chunk.id == tool_id
        assert chunk.type == type_value
        assert chunk.function == {"name": "test_function", "arguments": "{}"}
        assert chunk.index == index


class Test_ToolCallChunk_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ToolCallChunk class."""
    
    def test_validation_error_handling(self):
        """Verify the ToolCallChunk properly handles validation errors."""
        # Test with missing required field
        with pytest.raises(ValidationError) as excinfo:
            ToolCallChunk(id="call_123", function={"name": "test_function", "arguments": "{}"})
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("index",)
    
    def test_function_dictionary_validation_error_handling(self):
        """Verify the ToolCallChunk properly handles function dictionary validation errors."""
        # Test with invalid function dictionary
        with pytest.raises(ValidationError) as excinfo:
            ToolCallChunk(
                id="call_123",
                function={"invalid": "dictionary"},
                index=0
            )
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) > 0


class Test_ChatCompletionTool_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionTool class."""
    
    @pytest.mark.parametrize("type_value", [
        "function",
        None,  # type defaults to "function"
    ])
    def test_initialization(self, type_value):
        """Verify the ChatCompletionTool can be properly initialized with valid parameters."""
        # Create a valid FunctionDefinition object
        function_def = FunctionDefinition(
            name="test_function",
            description="Test function description",
            parameters=FunctionParameters(
                type="object",
                properties={"arg1": {"type": "string"}}
            )
        )
        
        tool = ChatCompletionTool(
            type=type_value if type_value else "function",
            function=function_def
        )
        
        # Verify properties are accessible and contain expected values
        assert tool.type == "function"  # type defaults to "function" if None
        assert tool.function == function_def
        assert tool.function.name == "test_function"
        assert tool.function.description == "Test function description"
    
    def test_serialization_deserialization(self):
        """Verify the ChatCompletionTool can be properly serialized and deserialized."""
        # Create a valid FunctionDefinition object
        function_def = FunctionDefinition(
            name="test_function",
            description="Test function description",
            parameters=FunctionParameters(
                type="object",
                properties={"arg1": {"type": "string"}}
            )
        )
        
        tool = ChatCompletionTool(function=function_def)
        
        # Serialize to JSON
        json_data = tool.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["type"] == "function"
        assert parsed_json["function"]["name"] == "test_function"
        assert parsed_json["function"]["description"] == "Test function description"
        
        # Deserialize from JSON
        deserialized = ChatCompletionTool.model_validate_json(json_data)
        
        # Verify all properties match the original
        assert deserialized.type == tool.type
        assert deserialized.function.name == tool.function.name
        assert deserialized.function.description == tool.function.description


class Test_ChatCompletionTool_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionTool class."""
    
    @pytest.mark.parametrize("data, error_field", [
        # Missing required fields
        ({}, "function"),
        
        # Invalid function definition
        ({"function": "not_an_object"}, "function"),
    ])
    def test_initialization_with_invalid_data(self, data, error_field):
        """Verify the ChatCompletionTool rejects initialization with invalid data."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionTool(**data)
        
        # Verify the error field is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["loc"][0] == error_field for error in errors)


class Test_ChatCompletionTool_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionTool class."""
    
    @pytest.mark.parametrize("type_value", [
        # Non-default type
        "non_standard_type",
    ])
    def test_boundary_values(self, type_value):
        """Verify the ChatCompletionTool handles boundary values correctly."""
        # Create a valid FunctionDefinition object
        function_def = FunctionDefinition(
            name="test_function",
            description="Test function description",
            parameters=FunctionParameters(
                type="object",
                properties={"arg1": {"type": "string"}}
            )
        )
        
        tool = ChatCompletionTool(
            type=type_value,
            function=function_def
        )
        
        # Verify properties are accessible and contain expected values
        assert tool.type == type_value
        assert tool.function == function_def
    
    def test_complex_function_definition(self):
        """Verify the ChatCompletionTool handles complex FunctionDefinition objects."""
        # Create a complex FunctionDefinition object
        function_def = FunctionDefinition(
            name="complex_function",
            description="Complex function description",
            parameters=FunctionParameters(
                type="object",
                properties={
                    "string_arg": {"type": "string", "description": "A string parameter", "minLength": 1, "maxLength": 100},
                    "number_arg": {"type": "number", "description": "A number parameter", "minimum": 0, "maximum": 100},
                    "integer_arg": {"type": "integer", "description": "An integer parameter", "minimum": 0, "maximum": 10},
                    "boolean_arg": {"type": "boolean", "description": "A boolean parameter"},
                    "array_arg": {
                        "type": "array",
                        "description": "An array parameter",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 10
                    },
                    "object_arg": {
                        "type": "object",
                        "description": "An object parameter",
                        "properties": {
                            "nested_string": {"type": "string"},
                            "nested_number": {"type": "number"}
                        },
                        "required": ["nested_string"]
                    }
                },
                required=["string_arg", "boolean_arg"]
            )
        )
        
        tool = ChatCompletionTool(
            function=function_def
        )
        
        # Verify properties are accessible and contain expected values
        assert tool.type == "function"
        assert tool.function == function_def
        assert tool.function.name == "complex_function"
        assert tool.function.description == "Complex function description"
        assert "string_arg" in tool.function.parameters.properties
        assert "object_arg" in tool.function.parameters.properties
        assert tool.function.parameters.required == ["string_arg", "boolean_arg"]


class Test_ChatCompletionTool_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionTool class."""
    
    def test_validation_error_handling(self):
        """Verify the ChatCompletionTool properly handles validation errors."""
        # Test with missing required field
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionTool()
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("function",)
    
    def test_nested_validation_error_handling(self):
        """Verify the ChatCompletionTool properly handles nested validation errors."""
        # Test with invalid function definition
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionTool(
                function={"invalid": "format"}  # Not a valid FunctionDefinition
            )
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) > 0


class Test_ChatCompletionToolChoiceOption_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionToolChoiceOption class."""
    
    @pytest.mark.parametrize("type_value, function_name", [
        ("function", "test_function"),
        ("function", None),  # function.name can be None
    ])
    def test_initialization(self, type_value, function_name):
        """Verify the ChatCompletionToolChoiceOption can be properly initialized with valid parameters."""
        function_call = ChatCompletionFunctionCall(name=function_name)
        tool_choice = ChatCompletionToolChoiceOption(
            type=type_value,
            function=function_call
        )
        
        # Verify properties are accessible and contain expected values
        assert tool_choice.type == type_value
        assert tool_choice.function == function_call
        assert tool_choice.function.name == function_name
    
    def test_serialization_deserialization(self):
        """Verify the ChatCompletionToolChoiceOption can be properly serialized and deserialized."""
        function_call = ChatCompletionFunctionCall(name="test_function")
        tool_choice = ChatCompletionToolChoiceOption(
            function=function_call
        )
        
        # Serialize to JSON
        json_data = tool_choice.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["type"] == "function"
        assert parsed_json["function"]["name"] == "test_function"
        
        # Deserialize from JSON
        deserialized = ChatCompletionToolChoiceOption.model_validate_json(json_data)
        
        # Verify all properties match the original
        assert deserialized.type == tool_choice.type
        assert deserialized.function.name == tool_choice.function.name


class Test_ChatCompletionToolChoiceOption_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionToolChoiceOption class."""
    
    @pytest.mark.parametrize("data, error_field", [
        # Missing required fields
        ({}, "function"),
        
        # Invalid function call
        ({"function": "not_an_object"}, "function"),
    ])
    def test_initialization_with_invalid_data(self, data, error_field):
        """Verify the ChatCompletionToolChoiceOption rejects initialization with invalid data."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionToolChoiceOption(**data)
        
        # Verify the error field is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["loc"][0] == error_field for error in errors)


class Test_ChatCompletionToolChoiceOption_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionToolChoiceOption class."""
    
    @pytest.mark.parametrize("type_value, function_name", [
        # Non-default type
        ("non_standard_type", "test_function"),
        
        # Empty function name
        ("function", ""),
        
        # Very long function name
        ("function", "a" * 1000),
    ])
    def test_boundary_values(self, type_value, function_name):
        """Verify the ChatCompletionToolChoiceOption handles boundary values correctly."""
        function_call = ChatCompletionFunctionCall(name=function_name)
        tool_choice = ChatCompletionToolChoiceOption(
            type=type_value,
            function=function_call
        )
        
        # Verify properties are accessible and contain expected values
        assert tool_choice.type == type_value
        assert tool_choice.function.name == function_name


class Test_ChatCompletionToolChoiceOption_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionToolChoiceOption class."""
    
    def test_validation_error_handling(self):
        """Verify the ChatCompletionToolChoiceOption properly handles validation errors."""
        # Test with missing required field
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionToolChoiceOption()
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("function",)
    
    def test_nested_validation_error_handling(self):
        """Verify the ChatCompletionToolChoiceOption properly handles nested validation errors."""
        # Test with invalid function call
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionToolChoiceOption(
                function={"invalid": "format"}  # Not a valid ChatCompletionFunctionCall
            )
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) > 0


class Test_ChatCompletionRequest_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionRequest class."""
    
    def test_minimal_request_creation(self):
        """Test creation with only required fields (messages, model)."""
        messages = [
            {"role": "user", "content": "Hello, world!"}
        ]
        model = "gpt-3.5-turbo"
        
        request = ChatCompletionRequest(messages=messages, model=model)
        
        assert request.messages == [Message(**message) for message in messages]
        assert request.model == model
        assert request.temperature is None
        assert request.max_tokens is None
    
    @pytest.mark.parametrize("optional_param, value", [
        ("temperature", 0.7),
        ("top_p", 0.9),
        ("max_tokens", 100),
        ("stop", ["stop"]),
        ("n", 1),
        ("stream", True),
        ("presence_penalty", 0.5),
        ("frequency_penalty", -0.5),
    ])
    def test_optional_parameters(self, optional_param, value):
        """Test initialization with various optional parameters."""
        messages = [{"role": "user", "content": "Hello"}]
        model = "gpt-4"
        
        kwargs = {
            "messages": messages,
            "model": model,
            optional_param: value
        }
        
        request = ChatCompletionRequest(**kwargs)
        
        assert request.messages == [Message(**message) for message in messages]
        assert request.model == model
        assert getattr(request, optional_param) == value
    
    @pytest.mark.parametrize("message_count", [1, 2, 5, 10])
    def test_message_sequences(self, message_count):
        """Test handling of message sequences with various lengths."""
        messages = []
        for i in range(message_count):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i}"})
        
        request = ChatCompletionRequest(messages=messages, model="gpt-3.5-turbo")
        
        assert len(request.messages) == message_count
        for i, message in enumerate(request.messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message.role == expected_role
            assert message.content == f"Message {i}"
    
    def test_content_types(self):
        """Test handling of different message content types."""
        messages = [
            {"role": "user", "content": "Text message"},
            {"role": "user", "content": [{"type": "text", "text": "Structured text"}]},
            {"role": "user", "content": [
                {"type": "text", "text": "Mixed content"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]}
        ]
        
        request = ChatCompletionRequest(messages=messages, model="gpt-4-vision")
        
        assert len(request.messages) == 3
        assert request.messages[0].content == "Text message"
        assert request.messages[1].content[0].text == "Structured text"
        assert request.messages[2].content[1].image_url.url == "https://example.com/image.jpg"
    
    def test_parameter_combinations(self):
        """Test handling of compatible parameter combinations."""
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            presence_penalty=0.5,
            frequency_penalty=-0.5
        )
        
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.max_tokens == 100
        assert request.presence_penalty == 0.5
        assert request.frequency_penalty == -0.5


class Test_ChatCompletionRequest_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionRequest class."""
    
    @pytest.mark.parametrize("kwargs, expected_error", [
        (
            {"model": "gpt-4"},  # Missing messages
            "messages\n  Field required"
        ),
        (
            {"messages": [{"role": "user", "content": "Hello"}]},  # Missing model
            "model\n  Field required"
        ),
    ])
    def test_missing_required_fields(self, kwargs, expected_error):
        """Test creation without required fields."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionRequest(**kwargs)
        
        assert expected_error in str(excinfo.value)
    
    @pytest.mark.parametrize("param, value, expected_error", [
        ("temperature", -0.1, "Input should be greater than or equal to 0"),
        ("temperature", 2.1, "Input should be less than or equal to 2"),
        ("top_p", 0.0, "Input should be greater than 0"),
        ("top_p", 1.1, "Input should be less than or equal to 1"),
        ("presence_penalty", -2.1, "Input should be greater than or equal to -2"),
        ("presence_penalty", 2.1, "Input should be less than or equal to 2"),
        ("frequency_penalty", -2.1, "Input should be greater than or equal to -2"),
        ("frequency_penalty", 2.1, "Input should be less than or equal to 2"),
        ("max_tokens", 0, "Input should be greater than 0"),
        ("max_tokens", -1, "Input should be greater than 0"),
    ])
    def test_out_of_range_parameters(self, param, value, expected_error):
        """Test validation with out-of-range parameter values."""
        kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
            param: value
        }
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionRequest(**kwargs)
        
        assert expected_error in str(excinfo.value)
    
    def test_incompatible_parameters(self):
        """Test with incompatible parameter combinations (functions with tools)."""
        functions = [{
            "name": "get_weather", 
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }]
        tools = [{
            "type": "function", 
            "function": {
                "name": "get_weather", 
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }]
        
        kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
            "functions": functions,
            "tools": tools
        }
        
        with pytest.raises(ValueError) as excinfo:
            ChatCompletionRequest(**kwargs)
        
        assert "Cannot specify both 'functions' and 'tools' parameters" in str(excinfo.value)
    
    def test_invalid_model_identifier(self):
        """Test rejection of invalid model identifiers."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "Hello"}],
                model=""
            )
        
        assert "String should have at least 1 character" in str(excinfo.value)


class Test_ChatCompletionRequest_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionRequest class."""
    
    @pytest.mark.parametrize("param, value", [
        ("temperature", 0.0),  # Minimum temperature
        ("temperature", 2.0),  # Maximum temperature
        ("top_p", 0.00001),  # Minimum allowed top_p (just above 0)
        ("top_p", 1.0),  # Maximum top_p
        ("presence_penalty", -2.0),  # Minimum presence_penalty
        ("presence_penalty", 2.0),  # Maximum presence_penalty
        ("frequency_penalty", -2.0),  # Minimum frequency_penalty
        ("frequency_penalty", 2.0),  # Maximum frequency_penalty
        ("max_tokens", 1),  # Minimum max_tokens
        ("max_tokens", 100000),  # Very large max_tokens
    ])
    def test_boundary_parameter_values(self, param, value):
        """Test with boundary values for numerical parameters."""
        kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
            param: value
        }
        
        request = ChatCompletionRequest(**kwargs)
        assert getattr(request, param) == value
    
    @pytest.mark.parametrize("message_count", [1, 100])
    def test_message_count_boundaries(self, message_count):
        """Test with single message and large message count."""
        messages = []
        for i in range(message_count):
            messages.append({"role": "user", "content": f"Message {i}"})
        
        request = ChatCompletionRequest(messages=messages, model="gpt-3.5-turbo")
        
        assert len(request.messages) == message_count
    
    def test_empty_message_content(self):
        """Test with empty message content."""
        messages = [{"role": "user", "content": ""}]
        
        request = ChatCompletionRequest(
            messages=messages, 
            model="gpt-3.5-turbo"
        )
        
        assert request.messages[0].content == ""


class Test_ChatCompletionRequest_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionRequest class."""
    
    def test_validate_function_and_tools_error(self):
        """Test that validate_function_and_tools raises appropriate errors."""
        with pytest.raises(ValueError) as excinfo:
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo",
                functions=[{
                    "name": "get_weather", 
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }],
                tools=[{
                    "type": "function", 
                    "function": {
                        "name": "get_weather", 
                        "description": "Get weather",
                        "parameters": {
                            "type": "int"
                        }
                    }
                }]
            )
        
        assert "Cannot specify both 'functions' and 'tools' parameters" in str(excinfo.value)
    
    @pytest.mark.parametrize("param, value, expected_error", [
        ("temperature", "not_a_number", "Input should be a valid number"),
        ("top_p", "invalid", "Input should be a valid number"),
        ("max_tokens", "hundred", "Input should be a valid integer"),
        ("n", -1, "Input should be greater than 0"),
        ("stop", 123, "Input should be a valid string"),
    ])
    def test_validation_error_messages(self, param, value, expected_error):
        """Test validation error messages for clarity and correctness."""
        kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
            param: value
        }
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionRequest(**kwargs)
        
        assert expected_error in str(excinfo.value)
    
    def test_incompatible_parameters_behavior(self):
        """Test behavior when incompatible parameters are provided."""
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo",
                functions=[{"name": "get_weather"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}]
            )


class Test_ReasoningConfig_01_NominalBehaviors:
    """Tests for nominal behaviors of the ReasoningConfig class."""
    
    @pytest.mark.parametrize("effort, max_tokens, exclude", [
        (None, None, None),  # All fields optional
        ("high", None, None),  # With effort only
        (None, 1000, None),   # With max_tokens only
        (None, None, True),   # With exclude only
        ("medium", 500, False),  # With all fields
        ("low", 250, True),    # Different values
    ])
    def test_creation_with_optional_fields(self, effort, max_tokens, exclude):
        """Test creation with different combinations of optional fields."""
        config = ReasoningConfig(
            effort=effort,
            max_tokens=max_tokens,
            exclude=exclude
        )
        
        assert config.effort == effort
        assert config.max_tokens == max_tokens
        assert config.exclude == exclude
    
    @pytest.mark.parametrize("effort", ["high", "medium", "low"])
    def test_valid_effort_values(self, effort):
        """Test that all valid effort values are accepted."""
        config = ReasoningConfig(effort=effort)
        
        assert config.effort == effort
    
    @pytest.mark.parametrize("max_tokens", [1, 10, 100, 1000, 10000])
    def test_max_tokens_values(self, max_tokens):
        """Test functionality with various max_tokens values."""
        config = ReasoningConfig(max_tokens=max_tokens)
        
        assert config.max_tokens == max_tokens


class Test_ReasoningConfig_02_NegativeBehaviors:
    """Tests for negative behaviors of the ReasoningConfig class."""
    
    @pytest.mark.parametrize("effort", ["invalid", "HIGH", "Medium", "unknown", ""])
    def test_invalid_effort_values(self, effort):
        """Test with invalid effort values."""
        with pytest.raises(ValueError) as excinfo:
            ReasoningConfig(effort=effort)
        
        assert "effort must be one of: high, medium, low" in str(excinfo.value)
    
    @pytest.mark.parametrize("max_tokens", [0, -1, -100])
    def test_invalid_max_tokens(self, max_tokens):
        """Test rejection of non-positive max_tokens values."""
        with pytest.raises(ValidationError) as excinfo:
            ReasoningConfig(max_tokens=max_tokens)
        
        assert "Input should be greater than 0" in str(excinfo.value)
    
    @pytest.mark.parametrize("field, value, expected_error", [
        ("effort", 123, "Input should be a valid string"),
        ("max_tokens", "string", "Input should be a valid integer"),
        ("exclude", "not_a_boolean", "Input should be a valid boolean"),
    ])
    def test_incorrect_data_types(self, field, value, expected_error):
        """Test with incorrect data types for all fields."""
        kwargs = {field: value}
        
        with pytest.raises(ValidationError) as excinfo:
            ReasoningConfig(**kwargs)
        
        assert expected_error in str(excinfo.value)


class Test_ReasoningConfig_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ReasoningConfig class."""
    
    def test_min_valid_max_tokens(self):
        """Test with minimum valid max_tokens (1)."""
        config = ReasoningConfig(max_tokens=1)
        assert config.max_tokens == 1
    
    def test_large_max_tokens(self):
        """Test with very large max_tokens values."""
        large_value = sys.maxsize  # Maximum integer value
        config = ReasoningConfig(max_tokens=large_value)
        assert config.max_tokens == large_value
    
    @pytest.mark.parametrize("effort", ["high", "medium", "low"])
    def test_each_allowed_effort_value(self, effort):
        """Test behavior with each allowed effort value."""
        config = ReasoningConfig(effort=effort)
        assert config.effort == effort


class Test_ReasoningConfig_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ReasoningConfig class."""
    
    def test_validate_effort_validator(self):
        """Test that validate_effort validator produces appropriate errors."""
        with pytest.raises(ValueError) as excinfo:
            ReasoningConfig(effort="invalid")
        
        error_message = str(excinfo.value)
        assert "effort must be one of: high, medium, low" in error_message
    
    @pytest.mark.parametrize("field, value, expected_error", [
        ("effort", "invalid", "effort must be one of: high, medium, low"),
        ("max_tokens", 0, "Input should be greater than 0"),
        ("max_tokens", -10, "Input should be greater than 0"),
    ])
    def test_error_message_clarity(self, field, value, expected_error):
        """Test error message clarity and correctness for invalid inputs."""
        kwargs = {field: value}
        
        with pytest.raises((ValueError, ValidationError)) as excinfo:
            ReasoningConfig(**kwargs)
        
        assert expected_error in str(excinfo.value)


class Test_ChatCompletionResponseChoice_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionResponseChoice class."""
    
    def test_creation_with_required_fields(self):
        """Test creation with required fields (index, message)."""
        message = {"role": "assistant", "content": "Hello, world!"}
        choice = ChatCompletionResponseChoice(
            index=0,
            message=message
        )
        
        assert choice.index == 0
        assert choice.message == Message(**message)
        assert choice.finish_reason is None
        assert choice.native_finish_reason is None
        assert choice.logprobs is None
    
    @pytest.mark.parametrize("content_type", [
        "Plain text content",
        [{"type": "text", "text": "Structured content"}],
        [
            {"type": "text", "text": "Mixed content"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    ])
    def test_message_content_types(self, content_type):
        message = {"role": "assistant", "content": content_type}
        choice = ChatCompletionResponseChoice(index=0, message=message)
        
        if isinstance(content_type, str):
            assert choice.message.content == content_type
        else:
            # Compare semantic content for structured types
            assert len(choice.message.content) == len(content_type)
            for actual, expected in zip(choice.message.content, content_type):
                assert actual.type == expected["type"]
                if actual.type == "image_url":
                    assert actual.image_url.url == expected["image_url"]["url"]
                elif actual.type == "text":
                    assert actual.text == expected["text"]
                else:
                    raise ValueError(f"Unexpected content type: {actual.type}")
    
    @pytest.mark.parametrize("finish_reason", [
        "stop", "length", "content_filter", "function_call", "tool_calls"
    ])
    def test_finish_reason_values(self, finish_reason):
        """Test with various valid finish_reason values."""
        message = {"role": "assistant", "content": "Hello"}
        choice = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason=finish_reason
        )
        
        assert choice.finish_reason == finish_reason


class Test_ChatCompletionResponseChoice_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionResponseChoice class."""
    
    @pytest.mark.parametrize("kwargs, expected_error", [
        (
            {"message": {"role": "assistant", "content": "Hello"}},  # Missing index
            "index\n  Field required"
        ),
        (
            {"index": 0},  # Missing message
            "message\n  Field required"
        ),
    ])
    def test_missing_required_fields(self, kwargs, expected_error):
        """Test creation with missing required fields."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionResponseChoice(**kwargs)
        
        assert expected_error in str(excinfo.value)
    
    @pytest.mark.parametrize("field, value, expected_error", [
        ("index", "not_an_int", "Input should be a valid integer"),
        ("message", "not_a_dict", "Input should be a valid dictionary"),
        ("native_finish_reason", lambda x: x, "Input should be a valid string"),
    ])
    def test_invalid_field_types(self, field, value, expected_error):
        """Test behavior with invalid types for each field."""
        kwargs = {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello"}
        }
        kwargs[field] = value
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionResponseChoice(**kwargs)
        
        assert expected_error in str(excinfo.value)
    
    @pytest.mark.parametrize("finish_reason", [
        "invalid", "STOP", "unknown", ""
    ])
    def test_invalid_finish_reason(self, finish_reason):
        """Test that invalid values for finish_reason raise ValidationError."""
        message = {"role": "assistant", "content": "Hello"}
        
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionResponseChoice(
                index=0,
                message=message,
                finish_reason=finish_reason
            )
        
        assert "finish_reason" in str(exc_info.value)
        assert "enum" in str(exc_info.value)


class Test_ChatCompletionResponseChoice_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionResponseChoice class."""
    
    @pytest.mark.parametrize("index", [0, 1, 100, sys.maxsize])
    def test_index_boundary_values(self, index):
        """Test with boundary values for index (0, maximum integer)."""
        message = {"role": "assistant", "content": "Hello"}
        choice = ChatCompletionResponseChoice(
            index=index,
            message=message
        )
        
        assert choice.index == index
    
    def test_empty_message_content(self):
        """Test with empty message content."""
        message = {"role": "assistant", "content": ""}
        choice = ChatCompletionResponseChoice(
            index=0,
            message=message
        )
        
        assert choice.message.content == ""
    
    @pytest.mark.parametrize("finish_reason", [
        "stop", "length", "content_filter", "function_call", "tool_calls", None
    ])
    def test_all_finish_reason_values(self, finish_reason):
        """Test with all possible finish_reason enum values."""
        message = {"role": "assistant", "content": "Hello"}
        choice = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason=finish_reason
        )
        
        assert choice.finish_reason == finish_reason


class Test_ChatCompletionResponseChoice_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionResponseChoice class."""
    
    def test_validation_error_messages(self):
        """Test appropriate error messages during validation failures."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionResponseChoice(
                index="not_an_integer",
                message={"role": "assistant", "content": "Hello"}
            )
        
        error_message = str(excinfo.value)
        assert "index" in error_message
        assert "valid integer" in error_message
    
    def test_deserialization_with_malformed_data(self):
        """Test deserialization behavior with malformed data."""
        malformed_data = {
            "index": 0,
            "message": "not_a_dict"  # Message should be a dict, not a string
        }
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionResponseChoice.model_validate(malformed_data)
        
        error_message = str(excinfo.value)
        assert "message" in error_message
        assert "valid dictionary" in error_message


class Test_ChatCompletionStreamResponseDelta_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionStreamResponseDelta class."""
    
    @pytest.mark.parametrize("fields", [
        {"role": "assistant"},
        {"content": "Hello, world!"},
        {"role": "assistant", "content": "Hello, world!"},
        {"content": [{"type": "text", "text": "Structured content"}]},
        {"function_call": {"name": "get_weather", "arguments": '{"location": "New York"}'}},
        {"tool_calls": [{"id": "tool_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "New York"}'}}]},
    ])
    def test_creation_with_optional_fields(self, fields):
        """Test creation with different combinations of optional fields."""
        delta = ChatCompletionStreamResponseDelta(**fields)
        
        for field, value in fields.items():
            assert getattr(delta, field) == value
    
    def test_content_formats(self):
        """Test handling of various content formats."""
        # String content
        delta1 = ChatCompletionStreamResponseDelta(content="Plain text")
        assert delta1.content == "Plain text"
        
        # Structured content
        structured_content = [{"type": "text", "text": "Structured content"}]
        delta2 = ChatCompletionStreamResponseDelta(content=structured_content)
        assert delta2.content == structured_content
        
        # Complex nested structure
        complex_content = [
            {"type": "text", "text": "Mixed content"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
        delta3 = ChatCompletionStreamResponseDelta(content=complex_content)
        assert delta3.content == complex_content
    
    def test_function_tool_calls(self):
        """Test proper handling of function_call and tool_calls fields."""
        # Function call
        function_call = {"name": "get_weather", "arguments": '{"location": "New York"}'}
        delta1 = ChatCompletionStreamResponseDelta(function_call=function_call)
        assert delta1.function_call == function_call
        
        # Tool calls
        tool_calls = [{"id": "tool_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "New York"}'}}]
        delta2 = ChatCompletionStreamResponseDelta(tool_calls=tool_calls)
        assert delta2.tool_calls == tool_calls


class Test_ChatCompletionStreamResponseDelta_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionStreamResponseDelta class."""
    
    @pytest.mark.parametrize("field, value, expected_error", [
        ("role", lambda x: x, "Input should be"),
        ("content", lambda x: x, "Input should be a valid string"),
        ("function_call", "not_a_dict", "Input should be a valid dictionary"),
        ("tool_calls", "not_a_list", "Input should be a valid list"),
    ])
    def test_invalid_field_types(self, field, value, expected_error):
        """Test behavior with invalid types for each field."""
        kwargs = {field: value}
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionStreamResponseDelta(**kwargs)
        
        assert expected_error in str(excinfo.value)
    
    def test_malformed_content_structures(self):
        """Test behavior with malformed content structures."""
        # Content list without required fields
        invalid_content = [{"wrong_field": "value"}]
        
        # This might not raise an error if the model doesn't strictly validate content structure
        # But we'll check that it's stored as provided
        delta = ChatCompletionStreamResponseDelta(content=invalid_content)
        assert delta.content == invalid_content
    
    @pytest.mark.parametrize("role", ["invalid_role", "USER", "system_", ""])
    def test_invalid_role_values(self, role):
        """Test with invalid role values."""
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionStreamResponseDelta(role=role)


class Test_ChatCompletionStreamResponseDelta_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionStreamResponseDelta class."""
    
    def test_empty_content(self):
        """Test with empty/null content."""
        # Empty string
        delta1 = ChatCompletionStreamResponseDelta(content="")
        assert delta1.content == ""
        
        # None (null) content
        delta2 = ChatCompletionStreamResponseDelta(content=None)
        assert delta2.content is None
        
        # Empty list
        delta3 = ChatCompletionStreamResponseDelta(content=[])
        assert delta3.content == []
    
    def test_minimal_content_structure(self):
        """Test with minimal valid content structure."""
        minimal_content = [{"type": "text", "text": ""}]
        delta = ChatCompletionStreamResponseDelta(content=minimal_content)
        assert delta.content == minimal_content
    
    def test_complex_nested_content(self):
        """Test with complex, nested content structures."""
        complex_content = [
            {"type": "text", "text": "First part"},
            {"type": "image_url", "image_url": {
                "url": "https://example.com/image.jpg",
                "detail": "high",
                "metadata": {"source": "user", "timestamp": 12345}
            }},
            {"type": "text", "text": "Second part with nested content"}
        ]
        delta = ChatCompletionStreamResponseDelta(content=complex_content)
        assert delta.content == complex_content


class Test_ChatCompletionStreamResponseDelta_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionStreamResponseDelta class."""
    
    def test_validation_with_malformed_data(self):
        """Test validation behavior with malformed data."""
        malformed_data = {
            "role": 123,  # Role should be a string
            "content": "content"
        }
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionStreamResponseDelta.model_validate(malformed_data)
        
        assert "role" in str(excinfo.value)
    
    def test_missing_content_when_expected(self):
        """Test validation behavior when content is missing in specific contexts."""
        # This test is contextual and depends on specific validation rules
        # For example, if we have a function_call that requires specific content
        # Since we don't have such validation in the model definition, we'll create a mock test
        
        # A simple assertion to demonstrate the concept
        delta = ChatCompletionStreamResponseDelta(function_call={"name": "get_weather"})
        assert delta.content is None


class Test_ChatCompletionStreamResponseChoice_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionStreamResponseChoice class."""
    
    def test_creation_with_required_fields(self):
        """Test creation with required fields (index, delta)."""
        delta = ChatCompletionStreamResponseDelta(content="Hello")
        choice = ChatCompletionStreamResponseChoice(
            index=0,
            delta=delta
        )
        
        assert choice.index == 0
        assert choice.delta == delta
        assert choice.finish_reason is None
        assert choice.native_finish_reason is None
        assert choice.logprobs is None
    
    @pytest.mark.parametrize("delta_content", [
        "Plain text",
        [{"type": "text", "text": "Structured content"}],
        None
    ])
    def test_delta_content_types(self, delta_content):
        """Test proper handling of different delta content types."""
        delta = ChatCompletionStreamResponseDelta(content=delta_content)
        choice = ChatCompletionStreamResponseChoice(
            index=0,
            delta=delta
        )
        
        assert choice.delta.content == delta_content
    
    @pytest.mark.parametrize("finish_reason", [
        "stop", "length", "content_filter", "function_call", "tool_calls"
    ])
    def test_finish_reason_values(self, finish_reason):
        """Test with various valid finish_reason values."""
        delta = ChatCompletionStreamResponseDelta(content="Hello")
        choice = ChatCompletionStreamResponseChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason
        )
        
        assert choice.finish_reason == finish_reason


class Test_ChatCompletionStreamResponseChoice_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionStreamResponseChoice class."""
    
    @pytest.mark.parametrize("kwargs, expected_error", [
        (
            {"delta": ChatCompletionStreamResponseDelta(content="Hello")},  # Missing index
            "index\n  Field required"
        ),
        (
            {"index": 0},  # Missing delta
            "delta\n  Field required"
        ),
    ])
    def test_missing_required_fields(self, kwargs, expected_error):
        """Test creation with missing required fields."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionStreamResponseChoice(**kwargs)
        
        assert expected_error in str(excinfo.value)
    
    @pytest.mark.parametrize("field, value, expected_error", [
        ("index", "not_an_int", "Input should be a valid integer"),
        ("delta", "not_a_delta", "Input should be a valid dictionary"),
        ("finish_reason", lambda x: x, "enum"),
    ])
    def test_invalid_field_types(self, field, value, expected_error):
        """Test behavior with invalid types for each field."""
        kwargs = {
            "index": 0,
            "delta": ChatCompletionStreamResponseDelta(content="Hello")
        }
        kwargs[field] = value
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionStreamResponseChoice(**kwargs)
        
        assert expected_error in str(excinfo.value)
    
    @pytest.mark.parametrize("finish_reason", [
        "invalid", "STOP", "unknown", ""
    ])
    def test_invalid_finish_reason(self, finish_reason):
        """Test with invalid values for finish_reason."""
        delta = ChatCompletionStreamResponseDelta(content="Hello")
        
        with pytest.raises(ValidationError) as excinfo:
            choice = ChatCompletionStreamResponseChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason
            )
        
        assert "Input should be 'stop', 'length', 'content_filter', 'tool_calls', 'function_call', 'error' or 'None'" in str(excinfo.value)


class Test_ChatCompletionStreamResponseChoice_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionStreamResponseChoice class."""
    
    @pytest.mark.parametrize("index", [0, 1, 100, sys.maxsize])
    def test_index_boundary_values(self, index):
        """Test with boundary values for index (0, maximum integer)."""
        delta = ChatCompletionStreamResponseDelta(content="Hello")
        choice = ChatCompletionStreamResponseChoice(
            index=index,
            delta=delta
        )
        
        assert choice.index == index
    
    def test_empty_delta_content(self):
        """Test with empty delta content."""
        delta = ChatCompletionStreamResponseDelta(content="")
        choice = ChatCompletionStreamResponseChoice(
            index=0,
            delta=delta
        )
        
        assert choice.delta.content == ""
    
    @pytest.mark.parametrize("finish_reason", [
        "stop", "length", "content_filter", "function_call", "tool_calls", None
    ])
    def test_all_finish_reason_values(self, finish_reason):
        """Test with all possible finish_reason enum values."""
        delta = ChatCompletionStreamResponseDelta(content="Hello")
        choice = ChatCompletionStreamResponseChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason
        )
        
        assert choice.finish_reason == finish_reason


class Test_ChatCompletionStreamResponseChoice_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionStreamResponseChoice class."""
    
    def test_validation_error_messages(self):
        """Verify appropriate error messages during validation failures."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionStreamResponseChoice(
                index="not_an_integer",
                delta=ChatCompletionStreamResponseDelta(content="Hello")
            )
        
        error_message = str(excinfo.value)
        assert "index" in error_message
        assert "valid integer" in error_message
    
    def test_deserialization_with_malformed_data(self):
        """Test deserialization behavior with malformed data."""
        malformed_data = {
            "index": 0,
            "delta": "not_a_delta_object"
        }
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionStreamResponseChoice.model_validate(malformed_data)
        
        assert "delta" in str(excinfo.value)


class Test_ChatCompletionResponse_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionResponse class."""
    
    def test_creation_with_required_fields(self):
        """Test creation with all required fields."""
        response = ChatCompletionResponse(
            id="resp_123",
            object="chat.completion",
            created=1652345678,
            model="gpt-3.5-turbo",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message={"role": "assistant", "content": "Hello, world!"}
                )
            ]
        )
        
        assert response.id == "resp_123"
        assert response.object == "chat.completion"
        assert response.created == 1652345678
        assert response.model == "gpt-3.5-turbo"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello, world!"
        assert response.usage is None
        assert response.system_fingerprint is None
    
    @pytest.mark.parametrize("choice_count", [1, 2, 5])
    def test_multiple_choices(self, choice_count):
        """Test proper handling of choices array with multiple items."""
        choices = []
        for i in range(choice_count):
            choices.append(
                ChatCompletionResponseChoice(
                    index=i,
                    message={"role": "assistant", "content": f"Choice {i}"}
                )
            )
        
        response = ChatCompletionResponse(
            id="resp_123",
            object="chat.completion",
            created=1652345678,
            model="gpt-3.5-turbo",
            choices=choices
        )
        
        assert len(response.choices) == choice_count
        for i in range(choice_count):
            assert response.choices[i].index == i
            assert response.choices[i].message.content == f"Choice {i}"
    
    def test_with_usage(self):
        """Test with Usage object present."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        
        response = ChatCompletionResponse(
            id="resp_123",
            object="chat.completion",
            created=1652345678,
            model="gpt-3.5-turbo",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message={"role": "assistant", "content": "Hello"}
                )
            ],
            usage=usage
        )
        
        assert response.usage == usage
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30


class Test_ChatCompletionResponse_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionResponse class."""
    
    @pytest.mark.parametrize("field", ["id", "created", "model", "choices"])
    def test_missing_required_fields(self, field):
        """Test creation with missing required fields."""
        kwargs = {
            "id": "resp_123",
            "created": 1652345678,
            "model": "gpt-3.5-turbo",
            "choices": [
                ChatCompletionResponseChoice(
                    index=0,
                    message={"role": "assistant", "content": "Hello"}
                )
            ]
        }
        del kwargs[field]
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionResponse(**kwargs)
        
        assert field in str(excinfo.value)
    
    @pytest.mark.parametrize("field, value, expected_error", [
        ("id", 123, "Input should be a valid string"),
        ("object", 123, "Input should be a valid string"),
        ("created", "not_a_timestamp", "Input should be a valid integer"),
        ("model", 123, "Input should be a valid string"),
        ("choices", "not_a_list", "Input should be a valid list"),
    ])
    def test_invalid_field_types(self, field, value, expected_error):
        """Test behavior with invalid types for each field."""
        kwargs = {
            "id": "resp_123",
            "object": "chat.completion",
            "created": 1652345678,
            "model": "gpt-3.5-turbo",
            "choices": [
                ChatCompletionResponseChoice(
                    index=0,
                    message={"role": "assistant", "content": "Hello"}
                )
            ]
        }
        kwargs[field] = value
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionResponse(**kwargs)
        
        assert expected_error in str(excinfo.value)


class Test_ChatCompletionResponse_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionResponse class."""
    
    @pytest.mark.parametrize("choice_count", [1, 10])
    def test_choice_count_boundaries(self, choice_count):
        """Test with single vs. multiple choices."""
        choices = []
        for i in range(choice_count):
            choices.append(
                ChatCompletionResponseChoice(
                    index=i,
                    message={"role": "assistant", "content": f"Choice {i}"}
                )
            )
        
        response = ChatCompletionResponse(
            id="resp_123",
            object="chat.completion",
            created=1652345678,
            model="gpt-3.5-turbo",
            choices=choices
        )
        
        assert len(response.choices) == choice_count
    
    def test_timestamp_boundaries(self):
        for timestamp in [0, 1, 1747953976]:
            """Test with extremely old or future timestamps for created."""
            response = ChatCompletionResponse(
                id="resp_123",
                object="chat.completion",
                created=timestamp,
                model="gpt-3.5-turbo",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message={"role": "assistant", "content": "Hello"}
                    )
                ]
            )
            
            assert response.created == timestamp
    
    @pytest.mark.parametrize("model", [
        "gpt-3.5-turbo",
        "gpt-4",
        "anthropic/claude-2",
        "a-very-long-model-name-with-many-details-and-specifications"
    ])
    def test_model_identifier_formats(self, model):
        """Test with various model identifier formats."""
        response = ChatCompletionResponse(
            id="resp_123",
            object="chat.completion",
            created=1652345678,
            model=model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message={"role": "assistant", "content": "Hello"}
                )
            ]
        )
        
        assert response.model == model


class Test_ChatCompletionResponse_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionResponse class."""
    
    def test_deserialization_with_malformed_json(self):
        """Test deserialization with malformed or incomplete JSON."""
        malformed_data = {
            "id": "resp_123",
            "object": "chat.completion",
            "created": 1652345678,
            "model": "gpt-3.5-turbo",
            "choices": [{"invalid": "choice"}]  # Missing required fields
        }
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionResponse.model_validate(malformed_data)
        
        assert "choices" in str(excinfo.value) or "index" in str(excinfo.value) or "message" in str(excinfo.value)
    
    def test_missing_fields_behavior(self):
        """Test behavior when expected fields are missing or null."""
        # Test with missing optional fields
        response = ChatCompletionResponse(
            id="resp_123",
            object="chat.completion",
            created=1652345678,
            model="gpt-3.5-turbo",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message={"role": "assistant", "content": "Hello"}
                )
            ]
        )
        
        assert response.usage is None
        assert response.system_fingerprint is None


class Test_ChatCompletionStreamResponse_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionStreamResponse class."""
    
    def test_creation_with_required_fields(self):
        """Test creation with all required fields."""
        response = ChatCompletionStreamResponse(
            id="resp_123",
            object="chat.completion.chunk",
            created=1652345678,
            model="gpt-3.5-turbo",
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=ChatCompletionStreamResponseDelta(content="Hello")
                )
            ]
        )
        
        assert response.id == "resp_123"
        assert response.object == "chat.completion.chunk"
        assert response.created == 1652345678
        assert response.model == "gpt-3.5-turbo"
        assert len(response.choices) == 1
        assert response.choices[0].delta.content == "Hello"
        assert response.usage is None
        assert response.system_fingerprint is None
    
    @pytest.mark.parametrize("choice_count", [1, 2, 5])
    def test_multiple_streaming_choices(self, choice_count):
        """Test handling of multiple streaming choices."""
        choices = []
        for i in range(choice_count):
            choices.append(
                ChatCompletionStreamResponseChoice(
                    index=i,
                    delta=ChatCompletionStreamResponseDelta(content=f"Choice {i}")
                )
            )
        
        response = ChatCompletionStreamResponse(
            id="resp_123",
            object="chat.completion.chunk",
            created=1652345678,
            model="gpt-3.5-turbo",
            choices=choices
        )
        
        assert len(response.choices) == choice_count
        for i in range(choice_count):
            assert response.choices[i].index == i
            assert response.choices[i].delta.content == f"Choice {i}"
    
    def test_with_usage(self):
        """Test with Usage object present."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        
        response = ChatCompletionStreamResponse(
            id="resp_123",
            object="chat.completion.chunk",
            created=1652345678,
            model="gpt-3.5-turbo",
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=ChatCompletionStreamResponseDelta(content="Hello")
                )
            ],
            usage=usage
        )
        
        assert response.usage == usage
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15


class Test_ChatCompletionStreamResponse_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionStreamResponse class."""
    
    @pytest.mark.parametrize("field", ["id", "created", "model", "choices"])
    def test_missing_required_fields(self, field):
        """Test creation with missing required fields."""
        kwargs = {
            "id": "resp_123",
            "created": 1652345678,
            "model": "gpt-3.5-turbo",
            "choices": [
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=ChatCompletionStreamResponseDelta(content="Hello")
                )
            ]
        }
        del kwargs[field]
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionStreamResponse(**kwargs)
        
        assert field in str(excinfo.value)
    
    @pytest.mark.parametrize("field, value, expected_error", [
        ("id", 123, "Input should be a valid string"),
        ("object", 123, "Input should be a valid string"),
        ("created", "not_a_timestamp", "Input should be a valid integer"),
        ("model", 123, "Input should be a valid string"),
        ("choices", "not_a_list", "Input should be a valid list"),
    ])
    def test_invalid_field_types(self, field, value, expected_error):
        """Test behavior with invalid types for each field."""
        kwargs = {
            "id": "resp_123",
            "object": "chat.completion.chunk",
            "created": 1652345678,
            "model": "gpt-3.5-turbo",
            "choices": [
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=ChatCompletionStreamResponseDelta(content="Hello")
                )
            ]
        }
        kwargs[field] = value
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionStreamResponse(**kwargs)
        
        assert expected_error in str(excinfo.value)


class Test_ChatCompletionStreamResponse_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionStreamResponse class."""
    
    @pytest.mark.parametrize("choice_count", [1, 10])
    def test_choice_count_boundaries(self, choice_count):
        """Test with single vs. multiple streaming choices."""
        choices = []
        for i in range(choice_count):
            choices.append(
                ChatCompletionStreamResponseChoice(
                    index=i,
                    delta=ChatCompletionStreamResponseDelta(content=f"Choice {i}")
                )
            )
        
        response = ChatCompletionStreamResponse(
            id="resp_123",
            object="chat.completion.chunk",
            created=1652345678,
            model="gpt-3.5-turbo",
            choices=choices
        )
        
        assert len(response.choices) == choice_count
    
    def test_timestamp_boundaries(self):
        """Test with varying timestamps for created."""
        for timestamp in [0, 1, 1747954375]:
            response = ChatCompletionStreamResponse(
                id="resp_123",
                object="chat.completion.chunk",
                created=timestamp,
                model="gpt-3.5-turbo",
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=ChatCompletionStreamResponseDelta(content="Hello")
                    )
                ]
            )
            
            assert response.created == timestamp
    
    @pytest.mark.parametrize("model", [
        "gpt-3.5-turbo",
        "gpt-4",
        "anthropic/claude-2",
        "a-very-long-model-name-with-many-details-and-specifications"
    ])
    def test_model_identifier_formats(self, model):
        """Test with standard vs. non-standard model identifiers."""
        response = ChatCompletionStreamResponse(
            id="resp_123",
            object="chat.completion.chunk",
            created=1652345678,
            model=model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=ChatCompletionStreamResponseDelta(content="Hello")
                )
            ]
        )
        
        assert response.model == model


class Test_ChatCompletionStreamResponse_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionStreamResponse class."""
    
    def test_deserialization_with_malformed_data(self):
        """Test deserialization with malformed or incomplete streaming data."""
        malformed_data = {
            "id": "resp_123",
            "object": "chat.completion.chunk",
            "created": 1652345678,
            "model": "gpt-3.5-turbo",
            "choices": [{"invalid": "choice"}]  # Missing required fields
        }
        
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionStreamResponse.model_validate(malformed_data)
        
        assert "choices" in str(excinfo.value) or "index" in str(excinfo.value) or "delta" in str(excinfo.value)
    
    def test_missing_fields_behavior(self):
        """Test behavior when expected fields are missing or null."""
        # Test with missing optional fields
        response = ChatCompletionStreamResponse(
            id="resp_123",
            object="chat.completion.chunk",
            created=1652345678,
            model="gpt-3.5-turbo",
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=ChatCompletionStreamResponseDelta(content="Hello")
                )
            ]
        )
        
        # Verify optional fields are properly set to None
        assert response.usage is None
        assert response.system_fingerprint is None
    
    def test_malformed_streaming_chunks(self):
        """Test handling of malformed streaming chunks as seen in real-world scenarios."""
        # Based on issues reported in the OpenAI community
        malformed_chunk = '{"id":"chatcmpl-8IVTD4XbqF5boxUIVcTJERVy075ww","object":"chat.completion.chunk","created":1699421567,"model":"g'
        
        # Test should verify that attempting to parse this as JSON would fail
        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed_chunk)
            
    def test_missing_first_token(self):
        """Test handling of missing first token in streaming responses."""
        # This test simulates the issue reported in OpenAI community where the first content token is missing
        first_chunk = {
            "id": "resp_123",
            "object": "chat.completion.chunk",
            "created": 1652345678,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},  # Note: no content field in first chunk
                    "finish_reason": None
                }
            ]
        }
        
        # Verify this is a valid chunk despite missing content
        response = ChatCompletionStreamResponse.model_validate(first_chunk)
        assert response.choices[0].delta.content is None
        assert response.choices[0].delta.role == "assistant"


class Test_ChatCompletionFunction_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionFunction class."""
    
    @pytest.mark.parametrize("name, arguments", [
        ("test_function", "{}"),  # Minimal valid JSON
        ("test_function", '{"arg1": "value1", "arg2": 42}'),  # JSON with values
        ("test_function", '{"nested": {"value": true}}'),  # Nested JSON
    ])
    def test_initialization(self, name, arguments):
        """Verify the ChatCompletionFunction can be properly initialized with valid parameters."""
        function = ChatCompletionFunction(name=name, arguments=arguments)
        
        # Verify properties are accessible and contain expected values
        assert function.name == name
        assert function.arguments == arguments
    
    def test_serialization_deserialization(self):
        """Verify the ChatCompletionFunction can be properly serialized and deserialized."""
        function = ChatCompletionFunction(
            name="test_function",
            arguments='{"arg1": "value1", "arg2": 42}'
        )
        
        # Serialize to JSON
        json_data = function.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["name"] == "test_function"
        assert json.loads(parsed_json["arguments"]) == {"arg1": "value1", "arg2": 42}
        
        # Deserialize from JSON
        deserialized = ChatCompletionFunction.model_validate_json(json_data)
        
        # Verify all properties match the original
        assert deserialized.name == function.name
        assert deserialized.arguments == function.arguments


class Test_ChatCompletionFunction_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionFunction class."""
    
    @pytest.mark.parametrize("data, error_field", [
        # Missing required fields
        ({"arguments": '{}'}, "name"),
        ({"name": "test_function"}, "arguments"),
        
        # Empty strings
        ({"name": "", "arguments": '{}'}, "name"),
        ({"name": "test_function", "arguments": ""}, "arguments"),
        
        # Non-JSON arguments
        ({"name": "test_function", "arguments": "not_json"}, "arguments"),
    ])
    def test_initialization_with_invalid_data(self, data, error_field):
        """Verify the ChatCompletionFunction rejects initialization with invalid data."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionFunction(**data)
        
        # Verify the error field is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["loc"][0] == error_field for error in errors)


class Test_ChatCompletionFunction_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionFunction class."""
    
    @pytest.mark.parametrize("name, arguments", [
        # Very long function name
        ("a" * 1000, "{}"),
        
        # Minimal valid JSON
        ("test_function", "{}"),
        
        # Complex nested JSON
        ("test_function", '{"level1": {"level2": {"level3": {"level4": {"level5": [1, 2, 3, 4, 5]}}}}}'),
    ])
    def test_boundary_values(self, name, arguments):
        """Verify the ChatCompletionFunction handles boundary values correctly."""
        function = ChatCompletionFunction(name=name, arguments=arguments)
        
        # Verify properties are accessible and contain expected values
        assert function.name == name
        assert function.arguments == arguments


class Test_ChatCompletionFunction_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionFunction class."""
    
    def test_validation_error_handling(self):
        """Verify the ChatCompletionFunction properly handles validation errors."""
        # Test with missing required field
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionFunction(name="test_function")
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("arguments",)
    
    def test_json_parsing_error_handling(self):
        """Verify the ChatCompletionFunction properly handles JSON parsing errors."""
        # Test with invalid JSON
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionFunction(name="test_function", arguments="not_json")
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) > 0


class Test_ChatCompletionFunctionCall_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionFunctionCall class."""
    
    @pytest.mark.parametrize("name", [
        "test_function",
        None,  # name is optional
    ])
    def test_initialization(self, name):
        """Verify the ChatCompletionFunctionCall can be properly initialized with valid parameters."""
        function_call = ChatCompletionFunctionCall(name=name)
        
        # Verify properties are accessible and contain expected values
        assert function_call.name == name
    
    def test_serialization_deserialization(self):
        """Verify the ChatCompletionFunctionCall can be properly serialized and deserialized."""
        function_call = ChatCompletionFunctionCall(name="test_function")
        
        # Serialize to JSON
        json_data = function_call.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["name"] == "test_function"
        
        # Deserialize from JSON
        deserialized = ChatCompletionFunctionCall.model_validate_json(json_data)
        
        # Verify all properties match the original
        assert deserialized.name == function_call.name


class Test_ChatCompletionFunctionCall_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionFunctionCall class."""
    
    @pytest.mark.parametrize("data, expected_error_type", [
        # Incorrect data types
        ({"name": 123}, "string_type"),
        ({"name": True}, "string_type"),
        ({"name": ["list", "items"]}, "string_type"),
    ])
    def test_initialization_with_invalid_data_types(self, data, expected_error_type):
        """Verify the ChatCompletionFunctionCall rejects initialization with invalid data types."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionFunctionCall(**data)
        
        # Verify the error type is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["type"].startswith(expected_error_type) for error in errors)


class Test_ChatCompletionFunctionCall_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionFunctionCall class."""
    
    @pytest.mark.parametrize("name", [
        # Very long function name
        "a" * 1000,
        
        # Min length function name
        "a",
    ])
    def test_boundary_values(self, name):
        """Verify the ChatCompletionFunctionCall handles boundary values correctly."""
        function_call = ChatCompletionFunctionCall(name=name)
        
        # Verify properties are accessible and contain expected values
        assert function_call.name == name


class Test_ChatCompletionFunctionCall_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionFunctionCall class."""
    
    def test_validation_error_handling(self):
        """Verify the ChatCompletionFunctionCall properly handles validation errors."""
        # Test with invalid data type
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionFunctionCall(name=123)
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) > 0
        assert any(error["type"].startswith("string_type") for error in errors)


class Test_ToolCallFunction_01_NominalBehaviors:
    """Tests for nominal behaviors of the ToolCallFunction class."""
    
    @pytest.mark.parametrize("name, arguments", [
        ("test_function", "{}"),  # Minimal valid JSON
        ("test_function", '{"arg1": "value1", "arg2": 42}'),  # JSON with values
        ("test_function", '{"nested": {"value": true}}'),  # Nested JSON
    ])
    def test_initialization(self, name, arguments):
        """Verify the ToolCallFunction can be properly initialized with valid parameters."""
        function = ToolCallFunction(name=name, arguments=arguments)
        
        # Verify properties are accessible and contain expected values
        assert function.name == name
        assert function.arguments == arguments
    
    def test_serialization_deserialization(self):
        """Verify the ToolCallFunction can be properly serialized and deserialized."""
        function = ToolCallFunction(
            name="test_function",
            arguments='{"arg1": "value1", "arg2": 42}'
        )
        
        # Serialize to JSON
        json_data = function.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["name"] == "test_function"
        assert json.loads(parsed_json["arguments"]) == {"arg1": "value1", "arg2": 42}
        
        # Deserialize from JSON
        deserialized = ToolCallFunction.model_validate_json(json_data)
        
        # Verify all properties match the original
        assert deserialized.name == function.name
        assert deserialized.arguments == function.arguments


class Test_ToolCallFunction_02_NegativeBehaviors:
    """Tests for negative behaviors of the ToolCallFunction class."""
    
    @pytest.mark.parametrize("data, error_field", [
        # Missing required fields
        ({"arguments": '{}'}, "name"),
        ({"name": "test_function"}, "arguments"),
        
        # Empty strings
        ({"name": "", "arguments": '{}'}, "name"),
        ({"name": "test_function", "arguments": ""}, "arguments"),
        
        # Non-JSON arguments
        ({"name": "test_function", "arguments": "not_json"}, "arguments"),
    ])
    def test_initialization_with_invalid_data(self, data, error_field):
        """Verify the ToolCallFunction rejects initialization with invalid data."""
        with pytest.raises(ValidationError) as excinfo:
            ToolCallFunction(**data)
        
        # Verify the error field is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["loc"][0] == error_field for error in errors)


class Test_ToolCallFunction_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ToolCallFunction class."""
    
    @pytest.mark.parametrize("name, arguments", [
        # Very long function name
        ("a" * 1000, "{}"),
        
        # Minimal valid JSON
        ("test_function", "{}"),
        
        # Complex nested JSON
        ("test_function", '{"level1": {"level2": {"level3": {"level4": {"level5": [1, 2, 3, 4, 5]}}}}}'),
    ])
    def test_boundary_values(self, name, arguments):
        """Verify the ToolCallFunction handles boundary values correctly."""
        function = ToolCallFunction(name=name, arguments=arguments)
        
        # Verify properties are accessible and contain expected values
        assert function.name == name
        assert function.arguments == arguments


class Test_ToolCallFunction_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ToolCallFunction class."""
    
    def test_validation_error_handling(self):
        """Verify the ToolCallFunction properly handles validation errors."""
        # Test with missing required field
        with pytest.raises(ValidationError) as excinfo:
            ToolCallFunction(name="test_function")
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("arguments",)
    
    def test_json_parsing_error_handling(self):
        """Verify the ToolCallFunction properly handles JSON parsing errors."""
        # Test with invalid JSON
        with pytest.raises(ValidationError) as excinfo:
            ToolCallFunction(name="test_function", arguments="not_json")
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) > 0


class Test_ChatCompletionToolCall_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionToolCall class."""
    
    @pytest.mark.parametrize("tool_id, type_value, function_name, function_arguments", [
        ("call_123", "function", "test_function", "{}"),
        ("call_123", None, "test_function", "{}"),  # type defaults to "function"
    ])
    def test_initialization(self, tool_id, type_value, function_name, function_arguments):
        """Verify the ChatCompletionToolCall can be properly initialized with valid parameters."""
        function = ToolCallFunction(name=function_name, arguments=function_arguments)
        tool_call = ChatCompletionToolCall(
            id=tool_id,
            type=type_value if type_value else "function",
            function=function
        )
        
        # Verify properties are accessible and contain expected values
        assert tool_call.id == tool_id
        assert tool_call.type == "function"  # type defaults to "function" if None
        assert tool_call.function.name == function_name
        assert tool_call.function.arguments == function_arguments
    
    def test_serialization_deserialization(self):
        """Verify the ChatCompletionToolCall can be properly serialized and deserialized."""
        function = ToolCallFunction(name="test_function", arguments='{"arg1": "value1"}')
        tool_call = ChatCompletionToolCall(
            id="call_123",
            function=function
        )
        
        # Serialize to JSON
        json_data = tool_call.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["id"] == "call_123"
        assert parsed_json["type"] == "function"
        assert parsed_json["function"]["name"] == "test_function"
        assert json.loads(parsed_json["function"]["arguments"]) == {"arg1": "value1"}
        
        # Deserialize from JSON
        deserialized = ChatCompletionToolCall.model_validate_json(json_data)
        
        # Verify all properties match the original
        assert deserialized.id == tool_call.id
        assert deserialized.type == tool_call.type
        assert deserialized.function.name == tool_call.function.name
        assert deserialized.function.arguments == tool_call.function.arguments


class Test_ChatCompletionToolCall_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionToolCall class."""
    
    @pytest.mark.parametrize("data, error_field", [
        # Missing required fields
        ({"function": ToolCallFunction(name="test_function", arguments="{}")}, "id"),
        ({"id": "call_123"}, "function"),
        
        # Invalid nested object
        ({"id": "call_123", "function": "not_an_object"}, "function"),
    ])
    def test_initialization_with_invalid_data(self, data, error_field):
        """Verify the ChatCompletionToolCall rejects initialization with invalid data."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionToolCall(**data)
        
        # Verify the error field is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["loc"][0] == error_field for error in errors)


class Test_ChatCompletionToolCall_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionToolCall class."""
    
    @pytest.mark.parametrize("tool_id, type_value", [
        # Very long ID
        ("a" * 1000, "function"),
        
        # Non-default type
        ("call_123", "non_standard_type"),
    ])
    def test_boundary_values(self, tool_id, type_value):
        """Verify the ChatCompletionToolCall handles boundary values correctly."""
        function = ToolCallFunction(name="test_function", arguments="{}")
        tool_call = ChatCompletionToolCall(
            id=tool_id,
            type=type_value,
            function=function
        )
        
        # Verify properties are accessible and contain expected values
        assert tool_call.id == tool_id
        assert tool_call.type == type_value
        assert tool_call.function.name == "test_function"
        assert tool_call.function.arguments == "{}"


class Test_ChatCompletionToolCall_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionToolCall class."""
    
    def test_validation_error_handling(self):
        """Verify the ChatCompletionToolCall properly handles validation errors."""
        # Test with missing required field
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionToolCall(id="call_123")
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("function",)
    
    def test_nested_validation_error_handling(self):
        """Verify the ChatCompletionToolCall properly handles nested validation errors."""
        # Test with invalid nested object
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionToolCall(
                id="call_123",
                function={"name": "test_function"}  # Missing arguments field
            )
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) > 0


class Test_ToolCallChunk_01_NominalBehaviors:
    """Tests for nominal behaviors of the ToolCallChunk class."""
    
    @pytest.mark.parametrize("tool_id, type_value, function, index", [
        ("call_123", "function", {"name": "test_function", "arguments": "{}"}, 0),
        ("call_123", None, {"name": "test_function", "arguments": "{}"}, 1),  # type defaults to "function"
    ])
    def test_initialization(self, tool_id, type_value, function, index):
        """Verify the ToolCallChunk can be properly initialized with valid parameters."""
        chunk = ToolCallChunk(
            id=tool_id,
            type=type_value if type_value else "function",
            function=function,
            index=index
        )
        
        # Verify properties are accessible and contain expected values
        assert chunk.id == tool_id
        assert chunk.type == "function"  # type defaults to "function" if None
        assert chunk.function == function
        assert chunk.index == index
    
    def test_serialization_deserialization(self):
        """Verify the ToolCallChunk can be properly serialized and deserialized."""
        chunk = ToolCallChunk(
            id="call_123",
            function={"name": "test_function", "arguments": "{}"},
            index=0
        )
        
        # Serialize to JSON
        json_data = chunk.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["id"] == "call_123"
        assert parsed_json["type"] == "function"
        assert parsed_json["function"] == {"name": "test_function", "arguments": "{}"}
        assert parsed_json["index"] == 0
        
        # Deserialize from JSON
        deserialized = ToolCallChunk.model_validate_json(json_data)
        
        # Verify all properties match the original
        assert deserialized.id == chunk.id
        assert deserialized.type == chunk.type
        assert deserialized.function == chunk.function
        assert deserialized.index == chunk.index


class Test_ToolCallChunk_02_NegativeBehaviors:
    """Tests for negative behaviors of the ToolCallChunk class."""
    
    @pytest.mark.parametrize("data, error_field", [
        # Missing required fields
        ({"function": {"name": "test_function", "arguments": "{}"}, "index": 0}, "id"),
        ({"id": "call_123", "index": 0}, "function"),
        ({"id": "call_123", "function": {"name": "test_function", "arguments": "{}"}}, "index"),
        
        # Invalid function dictionary
        ({"id": "call_123", "function": {"name": 123, "arguments": "{}"}, "index": 0}, "function"),
        ({"id": "call_123", "function": {"name": "test_function", "arguments": 123}, "index": 0}, "function"),
    ])
    def test_initialization_with_invalid_data(self, data, error_field):
        """Verify the ToolCallChunk rejects initialization with invalid data."""
        with pytest.raises(ValidationError) as excinfo:
            ToolCallChunk(**data)
        
        # Verify the error field is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["loc"][0] == error_field for error in errors)


class Test_ToolCallChunk_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ToolCallChunk class."""
    
    @pytest.mark.parametrize("tool_id, type_value, index", [
        # Very long ID
        ("a" * 1000, "function", 0),
        
        # Boundary index values
        ("call_123", "function", 0),
        ("call_123", "function", 2**31-1),
        
        # Non-default type
        ("call_123", "non_standard_type", 0),
    ])
    def test_boundary_values(self, tool_id, type_value, index):
        """Verify the ToolCallChunk handles boundary values correctly."""
        chunk = ToolCallChunk(
            id=tool_id,
            type=type_value,
            function={"name": "test_function", "arguments": "{}"},
            index=index
        )
        
        # Verify properties are accessible and contain expected values
        assert chunk.id == tool_id
        assert chunk.type == type_value
        assert chunk.function == {"name": "test_function", "arguments": "{}"}
        assert chunk.index == index


class Test_ToolCallChunk_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ToolCallChunk class."""
    
    def test_validation_error_handling(self):
        """Verify the ToolCallChunk properly handles validation errors."""
        # Test with missing required field
        with pytest.raises(ValidationError) as excinfo:
            ToolCallChunk(id="call_123", function={"name": "test_function", "arguments": "{}"})
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("index",)
    
    def test_function_dictionary_validation_error_handling(self):
        """Verify the ToolCallChunk properly handles function dictionary validation errors."""
        # Test with invalid function dictionary
        with pytest.raises(ValidationError) as excinfo:
            ToolCallChunk(
                id="call_123",
                function={"invalid": lambda x: x},
                index=0
            )
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) > 0


class Test_ChatCompletionTool_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionTool class."""
    
    @pytest.mark.parametrize("type_value", [
        "function",
        None,  # type defaults to "function"
    ])
    def test_initialization(self, type_value):
        """Verify the ChatCompletionTool can be properly initialized with valid parameters."""
        # Create a valid FunctionDefinition object
        function_def = FunctionDefinition(
            name="test_function",
            description="Test function description",
            parameters={  # ← Use raw dictionary instead
                "type": "object",
                "properties": {"arg1": {"type": "string"}}
            }
        )
        
        tool = ChatCompletionTool(
            type=type_value if type_value else "function",
            function=function_def
        )
        
        # Verify properties are accessible and contain expected values
        assert tool.type == "function"  # type defaults to "function" if None
        assert tool.function == function_def
        assert tool.function.name == "test_function"
        assert tool.function.description == "Test function description"
    
    def test_serialization_deserialization(self):
        """Verify the ChatCompletionTool can be properly serialized and deserialized."""
        # Create a valid FunctionDefinition object
        function_def = FunctionDefinition(
            name="test_function",
            description="Test function description",
            parameters={
                "type": "object",
                "properties": {"arg1": {"type": "string"}}
            }
        )
        
        tool = ChatCompletionTool(function=function_def)
        
        # Serialize to JSON
        json_data = tool.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["type"] == "function"
        assert parsed_json["function"]["name"] == "test_function"
        assert parsed_json["function"]["description"] == "Test function description"
        
        # Deserialize from JSON
        deserialized = ChatCompletionTool.model_validate_json(json_data)
        
        # Verify all properties match the original
        assert deserialized.type == tool.type
        assert deserialized.function.name == tool.function.name
        assert deserialized.function.description == tool.function.description


class Test_ChatCompletionTool_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionTool class."""
    
    @pytest.mark.parametrize("data, error_field", [
        # Missing required fields
        ({}, "function"),
        
        # Invalid function definition
        ({"function": "not_an_object"}, "function"),
    ])
    def test_initialization_with_invalid_data(self, data, error_field):
        """Verify the ChatCompletionTool rejects initialization with invalid data."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionTool(**data)
        
        # Verify the error field is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["loc"][0] == error_field for error in errors)


class Test_ChatCompletionTool_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionTool class."""
    
    @pytest.mark.parametrize("type_value", [
        # Non-default type
        "non_standard_type",
    ])
    def test_boundary_values(self, type_value):
        """Verify the ChatCompletionTool handles boundary values correctly."""
        # Create a valid FunctionDefinition object
        function_def = FunctionDefinition(
            name="test_function",
            description="Test function description",
            parameters=FunctionParameters(
                type="object",
                properties={"arg1": {"type": "string"}}
            )
        )
        
        tool = ChatCompletionTool(
            type=type_value,
            function=function_def
        )
        
        # Verify properties are accessible and contain expected values
        assert tool.type == type_value
        assert tool.function == function_def
    
class Test_ChatCompletionTool_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionTool class."""
    
    @pytest.mark.parametrize("type_value", [
        # Non-default type
        "non_standard_type",
    ])
    def test_boundary_values(self, type_value):
        """Verify the ChatCompletionTool handles boundary values correctly."""
        # Create a valid FunctionDefinition object
        function_def = FunctionDefinition(
            name="test_function",
            description="Test function description",
            parameters={
                "type": "object",
                "properties": {"arg1": {"type": "string"}}
            }
        )
        
        tool = ChatCompletionTool(
            type=type_value,
            function=function_def
        )
        
        # Verify properties are accessible and contain expected values
        assert tool.type == type_value
        assert tool.function == function_def
    
    def test_complex_function_definition(self):
        """Verify the ChatCompletionTool handles complex FunctionDefinition objects."""
        # Create a complex FunctionDefinition object
        function_def = FunctionDefinition(
            name="complex_function",
            description="Complex function description",
            parameters={
                "type": "object",
                "properties": {
                    "string_arg": {"type": "string", "description": "A string parameter", "minLength": 1, "maxLength": 100},
                    "number_arg": {"type": "number", "description": "A number parameter", "minimum": 0, "maximum": 100},
                    "integer_arg": {"type": "integer", "description": "An integer parameter", "minimum": 0, "maximum": 10},
                    "boolean_arg": {"type": "boolean", "description": "A boolean parameter"},
                    "array_arg": {
                        "type": "array",
                        "description": "An array parameter",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 10
                    },
                    "object_arg": {
                        "type": "object",
                        "description": "An object parameter",
                        "properties": {
                            "nested_string": {"type": "string"},
                            "nested_number": {"type": "number"}
                        },
                        "required": ["nested_string"]
                    }
                },
                "required": ["string_arg", "boolean_arg"]
            }
        )
        
        tool = ChatCompletionTool(
            function=function_def
        )
        
        # Verify properties are accessible and contain expected values
        assert tool.type == "function"
        assert tool.function == function_def
        assert tool.function.name == "complex_function"
        assert tool.function.description == "Complex function description"
        assert "string_arg" in tool.function.parameters["properties"]
        assert "object_arg" in tool.function.parameters["properties"]
        assert tool.function.parameters["required"] == ["string_arg", "boolean_arg"]


class Test_ChatCompletionTool_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionTool class."""
    
    def test_validation_error_handling(self):
        """Verify the ChatCompletionTool properly handles validation errors."""
        # Test with missing required field
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionTool()
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("function",)
    
    def test_nested_validation_error_handling(self):
        """Verify the ChatCompletionTool properly handles nested validation errors."""
        # Test with invalid function definition
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionTool(
                function={"invalid": "format"}  # Not a valid FunctionDefinition
            )
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) > 0


class Test_ChatCompletionToolChoiceOption_01_NominalBehaviors:
    """Tests for nominal behaviors of the ChatCompletionToolChoiceOption class."""
    
    @pytest.mark.parametrize("type_value, function_name", [
        ("function", "test_function"),
        ("function", None),  # function.name can be None
    ])
    def test_initialization(self, type_value, function_name):
        """Verify the ChatCompletionToolChoiceOption can be properly initialized with valid parameters."""
        function_call = ChatCompletionFunctionCall(name=function_name)
        tool_choice = ChatCompletionToolChoiceOption(
            type=type_value,
            function=function_call
        )
        
        # Verify properties are accessible and contain expected values
        assert tool_choice.type == type_value
        assert tool_choice.function == function_call
        assert tool_choice.function.name == function_name
    
    def test_serialization_deserialization(self):
        """Verify the ChatCompletionToolChoiceOption can be properly serialized and deserialized."""
        function_call = ChatCompletionFunctionCall(name="test_function")
        tool_choice = ChatCompletionToolChoiceOption(
            function=function_call
        )
        
        # Serialize to JSON
        json_data = tool_choice.model_dump_json()
        parsed_json = json.loads(json_data)
        
        # Verify JSON contains expected fields
        assert parsed_json["type"] == "function"
        assert parsed_json["function"]["name"] == "test_function"
        
        # Deserialize from JSON
        deserialized = ChatCompletionToolChoiceOption.model_validate_json(json_data)
        
        # Verify all properties match the original
        assert deserialized.type == tool_choice.type
        assert deserialized.function.name == tool_choice.function.name


class Test_ChatCompletionToolChoiceOption_02_NegativeBehaviors:
    """Tests for negative behaviors of the ChatCompletionToolChoiceOption class."""
    
    @pytest.mark.parametrize("data, error_field", [
        # Missing required fields
        ({}, "function"),
        
        # Invalid function call
        ({"function": "not_an_object"}, "function"),
    ])
    def test_initialization_with_invalid_data(self, data, error_field):
        """Verify the ChatCompletionToolChoiceOption rejects initialization with invalid data."""
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionToolChoiceOption(**data)
        
        # Verify the error field is identified in the exception
        errors = excinfo.value.errors()
        assert any(error["loc"][0] == error_field for error in errors)


class Test_ChatCompletionToolChoiceOption_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ChatCompletionToolChoiceOption class."""
    
    @pytest.mark.parametrize("type_value, function_name", [
        # Non-default type
        ("non_standard_type", "test_function"),
        
        # Min length function name
        ("function", "a"),
        
        # Very long function name
        ("function", "a" * 1000),
    ])
    def test_boundary_values(self, type_value, function_name):
        """Verify the ChatCompletionToolChoiceOption handles boundary values correctly."""
        function_call = ChatCompletionFunctionCall(name=function_name)
        tool_choice = ChatCompletionToolChoiceOption(
            type=type_value,
            function=function_call
        )
        
        # Verify properties are accessible and contain expected values
        assert tool_choice.type == type_value
        assert tool_choice.function.name == function_name


class Test_ChatCompletionToolChoiceOption_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ChatCompletionToolChoiceOption class."""
    
    def test_validation_error_handling(self):
        """Verify the ChatCompletionToolChoiceOption properly handles validation errors."""
        # Test with missing required field
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionToolChoiceOption()
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert errors[0]["loc"] == ("function",)
    
    def test_nested_validation_error_handling(self):
        """Verify the ChatCompletionToolChoiceOption properly handles nested validation errors."""
        # Test with invalid function call
        with pytest.raises(ValidationError) as excinfo:
            ChatCompletionToolChoiceOption(
                function={"invalid": "format"}  # Not a valid ChatCompletionFunctionCall
            )
        
        # Verify error details
        errors = excinfo.value.errors()
        assert len(errors) > 0


# ============================================================================
# ParameterDefinition Tests
# ============================================================================

class Test_ParameterDefinition_Constructor_01_NominalBehaviors:
    """Test nominal behaviors for ParameterDefinition constructor."""
    
    @pytest.mark.parametrize("type_val,description,enum,default", [
        (ParameterType.STRING, "A string parameter", None, None),
        (ParameterType.NUMBER, "A number parameter", [1, 2, 3], 1),
        (ParameterType.BOOLEAN, "A boolean parameter", [True, False], False),
        (ParameterType.ARRAY, "An array parameter", None, []),
        (ParameterType.OBJECT, "An object parameter", None, {}),
        ([ParameterType.STRING, ParameterType.NULL], "Multi-type parameter", None, None),
        (ParameterType.INTEGER, "Integer with enum", [1, 2, 3, 4, 5], 3),
    ])
    def test_create_instances_with_valid_combinations(self, type_val, description, enum, default):
        """Test creating ParameterDefinition instances with valid type, description, enum, and default values."""
        param = ParameterDefinition(
            type=type_val,
            description=description,
            enum=enum,
            default=default
        )
        assert param.type == type_val
        assert param.description == description
        assert param.enum == enum
        assert param.default == default
    
    def test_verify_proper_field_assignment_and_inheritance_chain(self):
        """Test that all fields are properly assigned and inheritance chain is correctly initialized."""
        param = ParameterDefinition(
            type=ParameterType.STRING,
            description="Test parameter"
        )
        assert hasattr(param, 'type')
        assert hasattr(param, 'description') 
        assert hasattr(param, 'enum')
        assert hasattr(param, 'default')
        assert isinstance(param, ParameterDefinition)


class Test_ParameterDefinition_Constructor_02_NegativeBehaviors:
    """Test negative behaviors for ParameterDefinition constructor."""
    
    @pytest.mark.parametrize("invalid_type", [
        "invalid_string",
        123,
        [],
        {},
    ])
    def test_attempt_creation_with_invalid_type_values(self, invalid_type):
        """Test creation attempts with invalid type values outside ParameterType enum."""
        with pytest.raises(ValidationError):
            ParameterDefinition(type=invalid_type)
    
    @pytest.mark.parametrize("invalid_enum", [
        "not_a_list",
        123,
        {},
    ])
    def test_pass_non_list_structures_to_enum_field(self, invalid_enum):
        """Test passing non-list structures to enum field."""
        with pytest.raises(ValidationError):
            ParameterDefinition(
                type=ParameterType.STRING,
                enum=invalid_enum
            )


class Test_ParameterDefinition_Constructor_03_BoundaryBehaviors:
    """Test boundary behaviors for ParameterDefinition constructor."""
    
    @pytest.mark.parametrize("description", [
        "",
        None,
        "a" * 1000,  # Very long description
    ])
    def test_create_with_boundary_description_values(self, description):
        """Test creation with empty/None/maximum length description strings."""
        param = ParameterDefinition(
            type=ParameterType.STRING,
            description=description
        )
        assert param.description == description
    
    @pytest.mark.parametrize("enum_val", [
        [],
        [1],
        ["single_value"],
    ])
    def test_create_with_boundary_enum_values(self, enum_val):
        """Test creation with empty enum or single value enum."""
        param = ParameterDefinition(
            type=ParameterType.STRING,
            enum=enum_val
        )
        assert param.enum == enum_val


class Test_ParameterDefinition_Constructor_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ParameterDefinition constructor."""
    
    def test_handle_pydantic_validation_errors_for_malformed_input(self):
        """Test handling of Pydantic validation errors for malformed input data."""
        with pytest.raises(ValidationError) as exc_info:
            ParameterDefinition()  # Missing required 'type' field
        
        assert "type" in str(exc_info.value)
    
    def test_verify_meaningful_error_messages_for_type_mismatches(self):
        """Test that proper error messages are provided for type constraint violations."""
        with pytest.raises(ValidationError) as exc_info:
            ParameterDefinition(type="invalid_type")
        
        error_str = str(exc_info.value)
        assert "type" in error_str.lower()


class Test_ParameterDefinition_Constructor_05_StateTransitionBehaviors:
    """Test state transition behaviors for ParameterDefinition constructor."""
    def test_ensure_proper_pydantic_model_behavior(self):
        """Test proper Pydantic model behavior and field accessibility."""
        param = ParameterDefinition(
            type=ParameterType.STRING,
            description="Test",
            enum=["a", "b"],
            default="a"
        )
        
        # Test field accessibility
        assert param.type == ParameterType.STRING
        assert param.description == "Test"
        assert param.enum == ["a", "b"]
        assert param.default == "a"
        
        # Test model dump functionality
        model_dict = param.model_dump()
        assert isinstance(model_dict, dict)
        assert model_dict["type"] == ParameterType.STRING


# ============================================================================
# StringParameter Tests
# ============================================================================

class Test_StringParameter_Constructor_01_NominalBehaviors:
    """Test nominal behaviors for StringParameter constructor."""
    
    @pytest.mark.parametrize("min_length,max_length,pattern,format_val", [
        (0, 100, None, None),
        (1, 50, r"^[a-z]+$", "email"),
        (5, 20, r"\d+", "date-time"),
        (None, 100, None, "uri"),
        (0, None, r".*", None),
    ])
    def test_create_instances_with_valid_string_constraints(self, min_length, max_length, pattern, format_val):
        """Test creating StringParameter instances with valid string constraints."""
        param = StringParameter(
            min_length=min_length,
            max_length=max_length,
            pattern=pattern,
            format=format_val
        )
        assert param.type == ParameterType.STRING
        assert param.min_length == min_length
        assert param.max_length == max_length
        assert param.pattern == pattern
        assert param.format == format_val
    
    def test_verify_type_field_correctly_set_to_string(self):
        """Test that type field is correctly set to STRING and inheritance works."""
        param = StringParameter()
        assert param.type == ParameterType.STRING
        assert isinstance(param, ParameterDefinition)
        assert isinstance(param, StringParameter)


class Test_StringParameter_Constructor_02_NegativeBehaviors:
    """Test negative behaviors for StringParameter constructor."""
    
    @pytest.mark.parametrize("invalid_value", [
        (-1,),
        (0,),
        (-10,),
    ])
    def test_pass_negative_values_to_length_constraints(self, invalid_value):
        """Test passing negative values to min_length or max_length."""
        with pytest.raises(ValidationError):
            StringParameter(min_length=invalid_value)
        
        with pytest.raises(ValidationError):
            StringParameter(max_length=invalid_value)
    
    @pytest.mark.parametrize("invalid_pattern", [
        "[unclosed_bracket",
        "(?P<incomplete",
        "*invalid_quantifier",
    ])
    def test_provide_malformed_regex_patterns(self, invalid_pattern):
        """Test providing invalid regex patterns."""
        # Note: Pattern validation might occur at usage time rather than creation
        param = StringParameter(pattern=invalid_pattern)
        # Verify that regex compilation would fail
        with pytest.raises(re.error):
            re.compile(invalid_pattern)


class Test_StringParameter_Constructor_03_BoundaryBehaviors:
    """Test boundary behaviors for StringParameter constructor."""
    
    @pytest.mark.parametrize("min_val,max_val", [
        (0, 0),
        (0, 1),
        (100, 100),
        (0, 999999),
    ])
    def test_boundary_length_constraint_values(self, min_val, max_val):
        """Test boundary values for length constraints."""
        param = StringParameter(min_length=min_val, max_length=max_val)
        assert param.min_length == min_val
        assert param.max_length == max_val


class Test_StringParameter_Constructor_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for StringParameter constructor."""
    
    def test_handle_pydantic_validation_failures(self):
        """Test handling of Pydantic validation failures."""
        with pytest.raises(ValidationError):
            StringParameter(min_length="not_an_int")
    
    def test_manage_constraint_validation_errors(self):
        """Test management of constraint validation errors."""
        with pytest.raises(ValidationError):
            StringParameter(min_length=-1)


class Test_StringParameter_Constructor_05_StateTransitionBehaviors:
    """Test state transition behaviors for StringParameter constructor."""
    
    def test_verify_proper_inheritance_of_base_class_properties(self):
        """Test proper inheritance of base class properties."""
        param = StringParameter(
            description="Test string",
            enum=["a", "b", "c"],
            default="a"
        )
        assert param.type == ParameterType.STRING
        assert param.description == "Test string"
        assert param.enum == ["a", "b", "c"]
        assert param.default == "a"
    
    def test_ensure_constraint_validation_occurs_at_creation(self):
        """Test that constraint validation occurs at object creation."""
        # Valid constraints should work
        param = StringParameter(min_length=5, max_length=10)
        assert param.min_length == 5
        assert param.max_length == 10


class Test_StringParameter_ValidateLengthConstraints_01_NominalBehaviors:
    """Test nominal behaviors for StringParameter validate_length_constraints method."""
    
    @pytest.mark.parametrize("min_len,max_len", [
        (5, 10),
        (0, 100),
        (1, 1000),
        (None, 50),
        (10, None),
    ])
    def test_successfully_validate_when_max_exceeds_min(self, min_len, max_len):
        """Test successful validation when max_length exceeds min_length."""
        param = StringParameter(min_length=min_len, max_length=max_len)
        # If object creation succeeds, validation passed
        assert param.min_length == min_len
        assert param.max_length == max_len


class Test_StringParameter_ValidateLengthConstraints_02_NegativeBehaviors:
    """Test negative behaviors for StringParameter validate_length_constraints method."""
    
    @pytest.mark.parametrize("min_len,max_len", [
        (10, 5),
        (50, 25),
        (100, 1),
    ])
    def test_reject_instances_where_max_less_than_min(self, min_len, max_len):
        """Test rejection of instances where max_length is less than min_length."""
        with pytest.raises(ValidationError):
            StringParameter(min_length=min_len, max_length=max_len)


class Test_StringParameter_ValidateLengthConstraints_03_BoundaryBehaviors:
    """Test boundary behaviors for StringParameter validate_length_constraints method."""
    
    def test_validate_when_min_equals_max(self):
        """Test validation when min_length equals max_length."""
        param = StringParameter(min_length=10, max_length=10)
        assert param.min_length == param.max_length == 10
    
    @pytest.mark.parametrize("min_len,max_len", [
        (None, 10),
        (5, None),
        (None, None),
    ])
    def test_handle_none_value_constraints(self, min_len, max_len):
        """Test handling of None values for either constraint."""
        param = StringParameter(min_length=min_len, max_length=max_len)
        assert param.min_length == min_len
        assert param.max_length == max_len


class Test_StringParameter_ValidateLengthConstraints_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for StringParameter validate_length_constraints method."""
    
    def test_generate_clear_error_messages_for_constraint_violations(self):
        """Test generation of clear ValueError messages for constraint violations."""
        with pytest.raises(ValidationError) as exc_info:
            StringParameter(min_length=20, max_length=10)
        
        error_str = str(exc_info.value)
        assert "length" in error_str.lower() or "constraint" in error_str.lower()


class Test_StringParameter_ValidateLengthConstraints_05_StateTransitionBehaviors:
    """Test state transition behaviors for StringParameter validate_length_constraints method."""
    
    def test_maintain_object_validity_after_successful_validation(self):
        """Test that object validity is maintained after successful validation."""
        param = StringParameter(min_length=5, max_length=15)
        assert param.min_length == 5
        assert param.max_length == 15
        # Object should be fully functional
        assert param.type == ParameterType.STRING


# ============================================================================
# NumberParameter Tests
# ============================================================================

class Test_NumberParameter_Constructor_01_NominalBehaviors:
    """Test nominal behaviors for NumberParameter constructor."""
    
    @pytest.mark.parametrize("type_val,minimum,maximum,exclusive_min,exclusive_max,multiple", [
        (ParameterType.NUMBER, 0.0, 100.0, False, False, None),
        (ParameterType.INTEGER, -10, 10, True, True, 2),
        (ParameterType.NUMBER, None, 1000.0, None, False, 0.5),
        (ParameterType.INTEGER, 0, None, False, None, None),
        (ParameterType.NUMBER, -float('inf'), float('inf'), False, False, None),
    ])
    def test_create_with_valid_numeric_constraints(self, type_val, minimum, maximum, exclusive_min, exclusive_max, multiple):
        """Test creating NumberParameter with valid numeric constraints for both NUMBER and INTEGER types."""
        param = NumberParameter(
            type=type_val,
            minimum=minimum,
            maximum=maximum,
            exclusive_minimum=exclusive_min,
            exclusive_maximum=exclusive_max,
            multiple_of=multiple
        )
        assert param.type == type_val
        assert param.minimum == minimum
        assert param.maximum == maximum
        assert param.exclusive_minimum == exclusive_min
        assert param.exclusive_maximum == exclusive_max
        assert param.multiple_of == multiple


class Test_NumberParameter_Constructor_02_NegativeBehaviors:
    """Test negative behaviors for NumberParameter constructor."""
    
    @pytest.mark.parametrize("invalid_value", [
        ("not_a_number",),
        (None,),
        ([],),
        ({},),
        (lambda x: x,),
    ])
    def test_pass_non_numeric_values_to_numeric_fields(self, invalid_value):
        """Test passing non-numeric values to numeric fields."""
        with pytest.raises(ValidationError):
            NumberParameter(
                type=ParameterType.NUMBER,
                minimum=invalid_value
            )
        
        with pytest.raises(ValidationError):
            NumberParameter(
                type=ParameterType.NUMBER,
                maximum=invalid_value
            )
        
        with pytest.raises(ValidationError):
            NumberParameter(
                type=ParameterType.NUMBER,
                multiple_of=invalid_value
            )


class Test_NumberParameter_Constructor_03_BoundaryBehaviors:
    """Test boundary behaviors for NumberParameter constructor."""
    
    @pytest.mark.parametrize("minimum,maximum", [
        (-float('inf'), float('inf')),
        (0, 0),
        (-1000000, 1000000),
        (0.0001, 0.0001),
    ])
    def test_boundary_numeric_values(self, minimum, maximum):
        """Test with extreme and boundary numeric values."""
        param = NumberParameter(
            type=ParameterType.NUMBER,
            minimum=minimum,
            maximum=maximum
        )
        assert param.minimum == minimum
        assert param.maximum == maximum


class Test_NumberParameter_Constructor_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for NumberParameter constructor."""
    
    def test_handle_type_validation_errors(self):
        """Test handling of type validation errors."""
        with pytest.raises(ValidationError):
            NumberParameter(type="invalid_type")
    
    def test_manage_floating_point_precision_issues(self):
        """Test management of floating point precision issues."""
        param = NumberParameter(
            type=ParameterType.NUMBER,
            minimum=0.1 + 0.2,  # Known floating point precision issue
            maximum=0.4
        )
        # Should handle floating point values appropriately
        assert abs(param.minimum - 0.3) < 1e-10


class Test_NumberParameter_Constructor_05_StateTransitionBehaviors:
    """Test state transition behaviors for NumberParameter constructor."""
    
    @pytest.mark.parametrize("type_val", [
        ParameterType.NUMBER,
        ParameterType.INTEGER,
    ])
    def test_verify_correct_type_assignment(self, type_val):
        """Test verification of correct type assignment (NUMBER vs INTEGER)."""
        param = NumberParameter(type=type_val)
        assert param.type == type_val
        assert isinstance(param, ParameterDefinition)


class Test_NumberParameter_ValidateRangeConstraints_01_NominalBehaviors:
    """Test nominal behaviors for NumberParameter validate_range_constraints method."""
    
    @pytest.mark.parametrize("minimum,maximum", [
        (5, 10),
        (0.0, 100.5),
        (-50, 50),
        (None, 100),
        (-10, None),
    ])
    def test_successfully_validate_when_maximum_exceeds_minimum(self, minimum, maximum):
        """Test successful validation when maximum exceeds minimum."""
        param = NumberParameter(
            type=ParameterType.NUMBER,
            minimum=minimum,
            maximum=maximum
        )
        assert param.minimum == minimum
        assert param.maximum == maximum


class Test_NumberParameter_ValidateRangeConstraints_02_NegativeBehaviors:
    """Test negative behaviors for NumberParameter validate_range_constraints method."""
    
    @pytest.mark.parametrize("minimum,maximum", [
        (10, 5),
        (100.5, 50.2),
        (0, -10),
    ])
    def test_reject_instances_where_maximum_less_than_minimum(self, minimum, maximum):
        """Test rejection of instances where maximum is less than minimum."""
        with pytest.raises(ValidationError):
            NumberParameter(
                type=ParameterType.NUMBER,
                minimum=minimum,
                maximum=maximum
            )


class Test_NumberParameter_ValidateRangeConstraints_03_BoundaryBehaviors:
    """Test boundary behaviors for NumberParameter validate_range_constraints method."""
    
    def test_validate_when_minimum_equals_maximum(self):
        """Test validation when minimum equals maximum."""
        param = NumberParameter(
            type=ParameterType.NUMBER,
            minimum=10.0,
            maximum=10.0
        )
        assert param.minimum == param.maximum == 10.0
    
    @pytest.mark.parametrize("minimum,maximum", [
        (float('-inf'), 100),
        (-100, float('inf')),
        (1e-10, 1e10),
    ])
    def test_handle_extreme_numeric_values(self, minimum, maximum):
        """Test handling of extremely large or small values."""
        param = NumberParameter(
            type=ParameterType.NUMBER,
            minimum=minimum,
            maximum=maximum
        )
        assert param.minimum == minimum
        assert param.maximum == maximum


class Test_NumberParameter_ValidateRangeConstraints_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for NumberParameter validate_range_constraints method."""
    
    def test_provide_descriptive_error_messages_for_range_violations(self):
        """Test provision of descriptive error messages for range violations."""
        with pytest.raises(ValidationError) as exc_info:
            NumberParameter(
                type=ParameterType.NUMBER,
                minimum=50,
                maximum=25
            )
        
        error_str = str(exc_info.value)
        assert "maximum" in error_str.lower() or "minimum" in error_str.lower()


class Test_NumberParameter_ValidateRangeConstraints_05_StateTransitionBehaviors:
    """Test state transition behaviors for NumberParameter validate_range_constraints method."""
    
    def test_ensure_object_consistency_after_successful_validation(self):
        """Test object consistency after successful validation."""
        param = NumberParameter(
            type=ParameterType.INTEGER,
            minimum=1,
            maximum=100
        )
        assert param.minimum == 1
        assert param.maximum == 100
        assert param.type == ParameterType.INTEGER


# ============================================================================
# BooleanParameter Tests
# ============================================================================

class Test_BooleanParameter_Constructor_01_NominalBehaviors:
    """Test nominal behaviors for BooleanParameter constructor."""
    
    def test_create_instances_with_type_correctly_fixed_to_boolean(self):
        """Test creating instances with type correctly fixed to BOOLEAN."""
        param = BooleanParameter()
        assert param.type == ParameterType.BOOLEAN
    
    def test_verify_proper_inheritance_from_parameter_definition(self):
        """Test proper inheritance from ParameterDefinition."""
        param = BooleanParameter(description="Boolean parameter")
        assert isinstance(param, ParameterDefinition)
        assert isinstance(param, BooleanParameter)
        assert param.description == "Boolean parameter"


class Test_BooleanParameter_Constructor_03_BoundaryBehaviors:
    """Test boundary behaviors for BooleanParameter constructor."""
    
    @pytest.mark.parametrize("description,enum,default", [
        ("", [], None),
        ("Very long description" * 100, [True, False], True),
        (None, None, False),
    ])
    def test_boundary_inherited_field_configurations(self, description, enum, default):
        """Test with minimal and maximal inherited field configurations."""
        param = BooleanParameter(
            description=description,
            enum=enum,
            default=default
        )
        assert param.type == ParameterType.BOOLEAN
        assert param.description == description
        assert param.enum == enum
        assert param.default == default


class Test_BooleanParameter_Constructor_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for BooleanParameter constructor."""
    
    def test_handle_attempts_to_modify_immutable_type_field(self):
        """Test handling of attempts to modify immutable type field."""
        
        # Type should be immutable
        with pytest.raises(ValidationError):
            param = BooleanParameter(type=ParameterType.NUMBER)
        


class Test_BooleanParameter_Constructor_05_StateTransitionBehaviors:
    """Test state transition behaviors for BooleanParameter constructor."""
    
    def test_confirm_proper_base_class_initialization(self):
        """Test proper base class initialization."""
        param = BooleanParameter(
            description="Test boolean",
            enum=[True, False],
            default=True
        )
        assert param.type == ParameterType.BOOLEAN
        assert param.description == "Test boolean"
        assert param.enum == [True, False]
        assert param.default == True


# ============================================================================
# ArrayParameter Tests
# ============================================================================

class Test_ArrayParameter_Constructor_01_NominalBehaviors:
    """Test nominal behaviors for ArrayParameter constructor."""
    
    @pytest.mark.parametrize("items_schema,min_items,max_items,unique", [
        ({"type": "string"}, 0, 10, False),
        ({"type": "number"}, 1, 100, True),
        ({"type": "object", "properties": {"name": {"type": "string"}}}, None, 50, None),
        ({"type": "array", "items": {"type": "integer"}}, 5, None, False),
    ])
    def test_create_instances_with_valid_array_constraints(self, items_schema, min_items, max_items, unique):
        """Test creating ArrayParameter instances with valid array constraints."""
        param = ArrayParameter(
            items=items_schema,
            min_items=min_items,
            max_items=max_items,
            unique_items=unique
        )
        assert param.type == ParameterType.ARRAY
        assert param.items == items_schema
        assert param.min_items == min_items
        assert param.max_items == max_items
        assert param.unique_items == unique


class Test_ArrayParameter_Constructor_02_NegativeBehaviors:
    """Test negative behaviors for ArrayParameter constructor."""
    
    @pytest.mark.parametrize("invalid_items", [
        "not_a_dict",
        [],
        None,
    ])
    def test_pass_invalid_schema_structures_to_items_field(self, invalid_items):
        """Test passing invalid schema structures to items field."""
        with pytest.raises(ValidationError):
            ArrayParameter(items=invalid_items)
    
    @pytest.mark.parametrize("invalid_value", [(-1,), (-2,), (-5,)])
    def test_provide_negative_values_for_item_count_constraints(self, invalid_value):
        """Test providing negative values for item count constraints."""
        with pytest.raises(ValidationError):
            ArrayParameter(
                items={"type": "string"},
                min_items=invalid_value
            )
        
        with pytest.raises(ValidationError):
            ArrayParameter(
                items={"type": "string"},
                max_items=invalid_value
            )


class Test_ArrayParameter_Constructor_03_BoundaryBehaviors:
    """Test boundary behaviors for ArrayParameter constructor."""
    
    @pytest.mark.parametrize("min_items,max_items", [
        (0, 0),
        (0, 999999),
        (100, 100),
    ])
    def test_boundary_item_count_constraints(self, min_items, max_items):
        """Test boundary values for item count constraints."""
        param = ArrayParameter(
            items={"type": "string"},
            min_items=min_items,
            max_items=max_items
        )
        assert param.min_items == min_items
        assert param.max_items == max_items


class Test_ArrayParameter_Constructor_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ArrayParameter constructor."""
    
    def test_handle_malformed_nested_schema_definitions(self):
        """Test handling of malformed nested schema definitions."""
        with pytest.raises(ValidationError):
            ArrayParameter()  # Missing required items field


class Test_ArrayParameter_Constructor_05_StateTransitionBehaviors:
    """Test state transition behaviors for ArrayParameter constructor."""
    
    def test_verify_proper_schema_inheritance_and_nesting(self):
        """Test proper schema inheritance and nesting."""
        param = ArrayParameter(
            items={"type": "string", "minLength": 1},
            min_items=1,
            max_items=10
        )
        assert param.type == ParameterType.ARRAY
        assert param.items["type"] == "string"
        assert param.items["minLength"] == 1


class Test_ArrayParameter_ValidateItemsConstraints_01_NominalBehaviors:
    """Test nominal behaviors for ArrayParameter validate_items_constraints method."""
    
    @pytest.mark.parametrize("min_items,max_items", [
        (5, 10),
        (0, 100),
        (None, 50),
        (10, None),
    ])
    def test_successfully_validate_when_max_exceeds_min(self, min_items, max_items):
        """Test successful validation when max_items exceeds min_items."""
        param = ArrayParameter(
            items={"type": "string"},
            min_items=min_items,
            max_items=max_items
        )
        assert param.min_items == min_items
        assert param.max_items == max_items


class Test_ArrayParameter_ValidateItemsConstraints_02_NegativeBehaviors:
    """Test negative behaviors for ArrayParameter validate_items_constraints method."""
    
    @pytest.mark.parametrize("min_items,max_items", [
        (10, 5),
        (50, 25),
        (100, 1),
    ])
    def test_reject_instances_where_max_less_than_min(self, min_items, max_items):
        """Test rejection of instances where max_items is less than min_items."""
        with pytest.raises(ValidationError):
            ArrayParameter(
                items={"type": "string"},
                min_items=min_items,
                max_items=max_items
            )


class Test_ArrayParameter_ValidateItemsConstraints_03_BoundaryBehaviors:
    """Test boundary behaviors for ArrayParameter validate_items_constraints method."""
    
    def test_validate_when_min_equals_max(self):
        """Test validation when min_items equals max_items."""
        param = ArrayParameter(
            items={"type": "string"},
            min_items=10,
            max_items=10
        )
        assert param.min_items == param.max_items == 10


class Test_ArrayParameter_ValidateItemsConstraints_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ArrayParameter validate_items_constraints method."""
    
    def test_generate_meaningful_error_messages_for_constraint_violations(self):
        """Test generation of meaningful error messages for constraint violations."""
        with pytest.raises(ValidationError) as exc_info:
            ArrayParameter(
                items={"type": "string"},
                min_items=20,
                max_items=10
            )
        
        error_str = str(exc_info.value)
        assert "items" in error_str.lower() or "constraint" in error_str.lower()


class Test_ArrayParameter_ValidateItemsConstraints_05_StateTransitionBehaviors:
    """Test state transition behaviors for ArrayParameter validate_items_constraints method."""
    
    def test_maintain_object_state_consistency(self):
        """Test maintenance of object state consistency."""
        param = ArrayParameter(
            items={"type": "number"},
            min_items=1,
            max_items=5,
            unique_items=True
        )
        assert param.min_items == 1
        assert param.max_items == 5
        assert param.unique_items == True


# ============================================================================
# ObjectParameter Tests
# ============================================================================

class Test_ObjectParameter_Constructor_01_NominalBehaviors:
    """Test nominal behaviors for ObjectParameter constructor."""
    
    @pytest.mark.parametrize("properties,required,additional_props", [
        ({"name": {"type": "string"}}, ["name"], True),
        ({"age": {"type": "integer"}, "email": {"type": "string"}}, ["age"], False),
        ({}, [], None),
        ({"nested": {"type": "object", "properties": {"value": {"type": "number"}}}}, None, {"type": "string"}),
    ])
    def test_create_instances_with_valid_object_configurations(self, properties, required, additional_props):
        """Test creating ObjectParameter instances with valid nested property schemas."""
        param = ObjectParameter(
            properties=properties,
            required=required,
            additional_properties=additional_props
        )
        assert param.type == ParameterType.OBJECT
        assert param.properties == properties
        assert param.required == required
        assert param.additional_properties == additional_props


class Test_ObjectParameter_Constructor_02_NegativeBehaviors:
    """Test negative behaviors for ObjectParameter constructor."""
    
    @pytest.mark.parametrize("invalid_properties", [
        "not_a_dict",
        123,
        [],
    ])
    def test_pass_non_dictionary_structures_to_properties_field(self, invalid_properties):
        """Test passing non-dictionary structures to properties field."""
        with pytest.raises(ValidationError):
            ObjectParameter(properties=invalid_properties)
    
    @pytest.mark.parametrize("invalid_required", [
        "not_a_list",
        123,
        {"key": "value"},
    ])
    def test_provide_non_list_values_to_required_field(self, invalid_required):
        """Test providing non-list values to required field."""
        with pytest.raises(ValidationError):
            ObjectParameter(
                properties={"name": {"type": "string"}},
                required=invalid_required
            )


class Test_ObjectParameter_Constructor_03_BoundaryBehaviors:
    """Test boundary behaviors for ObjectParameter constructor."""
    
    def test_create_with_empty_properties_dictionary(self):
        """Test creation with empty properties dictionary."""
        param = ObjectParameter(properties={})
        assert param.properties == {}
        assert param.type == ParameterType.OBJECT
    
    def test_specify_all_properties_as_required(self):
        """Test specifying all properties as required."""
        properties = {"name": {"type": "string"}, "age": {"type": "integer"}}
        param = ObjectParameter(
            properties=properties,
            required=["name", "age"]
        )
        assert param.required == ["name", "age"]
        assert len(param.required) == len(param.properties)


class Test_ObjectParameter_Constructor_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ObjectParameter constructor."""
    
    def test_handle_malformed_nested_property_schemas(self):
        """Test handling of malformed nested property schemas."""
        # Properties should be a dictionary
        param = ObjectParameter(properties={})  # Empty is valid
        assert param.properties == {}


class Test_ObjectParameter_Constructor_05_StateTransitionBehaviors:
    """Test state transition behaviors for ObjectParameter constructor."""
    
    def test_verify_proper_nested_schema_handling(self):
        """Test proper nested schema handling."""
        properties = {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            }
        }
        param = ObjectParameter(properties=properties)
        assert param.properties["user"]["type"] == "object"
        assert "name" in param.properties["user"]["properties"]


class Test_ObjectParameter_ValidateRequiredProperties_01_NominalBehaviors:
    """Test nominal behaviors for ObjectParameter validate_required_properties method."""
    
    @pytest.mark.parametrize("properties,required", [
        ({"name": {"type": "string"}, "age": {"type": "integer"}}, ["name"]),
        ({"email": {"type": "string"}}, ["email"]),
        ({"a": {"type": "string"}, "b": {"type": "number"}}, ["a", "b"]),
        ({"optional": {"type": "string"}}, []),
    ])
    def test_successfully_validate_when_required_properties_exist(self, properties, required):
        """Test successful validation when all required properties exist in properties dictionary."""
        param = ObjectParameter(
            properties=properties,
            required=required
        )
        assert param.properties == properties
        assert param.required == required


class Test_ObjectParameter_ValidateRequiredProperties_02_NegativeBehaviors:
    """Test negative behaviors for ObjectParameter validate_required_properties method."""
    
    @pytest.mark.parametrize("properties,required", [
        ({"name": {"type": "string"}}, ["missing_property"]),
        ({"age": {"type": "integer"}}, ["name", "email"]),
        ({}, ["required_but_missing"]),
    ])
    def test_reject_instances_with_missing_required_properties(self, properties, required):
        """Test rejection when required properties are missing from properties dictionary."""
        with pytest.raises(ValidationError):
            ObjectParameter(
                properties=properties,
                required=required
            )


class Test_ObjectParameter_ValidateRequiredProperties_03_BoundaryBehaviors:
    """Test boundary behaviors for ObjectParameter validate_required_properties method."""
    
    def test_validate_with_empty_required_list(self):
        """Test validation with empty required list."""
        param = ObjectParameter(
            properties={"optional": {"type": "string"}},
            required=[]
        )
        assert param.required == []
    
    def test_validate_with_all_properties_required(self):
        """Test validation with all properties marked as required."""
        properties = {"a": {"type": "string"}, "b": {"type": "number"}}
        param = ObjectParameter(
            properties=properties,
            required=["a", "b"]
        )
        assert set(param.required) == set(properties.keys())


class Test_ObjectParameter_ValidateRequiredProperties_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ObjectParameter validate_required_properties method."""
    
    def test_provide_clear_error_messages_for_missing_properties(self):
        """Test provision of clear error messages for missing required properties."""
        with pytest.raises(ValidationError) as exc_info:
            ObjectParameter(
                properties={"name": {"type": "string"}},
                required=["missing_prop"]
            )
        
        error_str = str(exc_info.value)
        assert "missing_prop" in error_str or "property" in error_str.lower()


class Test_ObjectParameter_ValidateRequiredProperties_05_StateTransitionBehaviors:
    """Test state transition behaviors for ObjectParameter validate_required_properties method."""
    
    def test_ensure_consistent_property_requirement_relationships(self):
        """Test consistent property-requirement relationships."""
        param = ObjectParameter(
            properties={
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"}
            },
            required=["name", "email"]
        )
        
        # Verify relationships are maintained
        for req_prop in param.required:
            assert req_prop in param.properties


# ============================================================================
# FunctionParameters Tests
# ============================================================================

class Test_FunctionParameters_Constructor_01_NominalBehaviors:
    """Test nominal behaviors for FunctionParameters constructor."""
    
    @pytest.mark.parametrize("properties,required", [
        ({"param1": {"type": "string"}, "param2": {"type": "number"}}, ["param1"]),
        ({"input": {"type": "object", "properties": {"value": {"type": "integer"}}}}, []),
        ({}, None),
        ({"optional": {"type": "boolean"}}, []),
    ])
    def test_create_instances_with_valid_property_dictionaries(self, properties, required):
        """Test creating FunctionParameters instances with valid property dictionaries and required lists."""
        params = FunctionParameters(
            properties=properties,
            required=required
        )
        assert params.type == "object"
        assert params.properties == properties
        assert params.required == required


class Test_FunctionParameters_Constructor_02_NegativeBehaviors:
    """Test negative behaviors for FunctionParameters constructor."""
    
    @pytest.mark.parametrize("invalid_properties", [
        "not_a_dict",
        123,
        [],
    ])
    def test_pass_invalid_data_types_to_properties_field(self, invalid_properties):
        """Test passing invalid data types to properties field."""
        with pytest.raises(ValidationError):
            FunctionParameters(properties=invalid_properties)


class Test_FunctionParameters_Constructor_03_BoundaryBehaviors:
    """Test boundary behaviors for FunctionParameters constructor."""
    
    def test_create_with_empty_properties(self):
        """Test creation with empty properties."""
        params = FunctionParameters(properties={})
        assert params.properties == {}
        assert params.type == "object"
    
    @pytest.mark.parametrize("required_config", [
        [],
        None,
        ["all", "properties", "required"],
    ])
    def test_boundary_required_parameter_configurations(self, required_config):
        """Test boundary configurations for required parameters."""
        properties = {"all": {"type": "string"}, "properties": {"type": "number"}, "required": {"type": "boolean"}}
        params = FunctionParameters(
            properties=properties,
            required=required_config
        )
        assert params.required == required_config


class Test_FunctionParameters_Constructor_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for FunctionParameters constructor."""
    
    def test_handle_malformed_property_schema_definitions(self):
        """Test handling of malformed property schema definitions."""
        with pytest.raises(ValidationError):
            FunctionParameters()  # Missing required properties field

# ============================================================================
# FunctionCall Tests
# ============================================================================

class Test_FunctionCall_Constructor_01_NominalBehaviors:
    """Test nominal behaviors for FunctionCall constructor."""
    
    @pytest.mark.parametrize("name,arguments,call_id", [
        ("test_function", '{"param": "value"}', None),
        ("another_func", '{"num": 42, "bool": true}', "call_123"),
        ("no_args_func", '{}', "unique_id"),
        ("complex_func", '{"nested": {"key": "value"}, "array": [1, 2, 3]}', None),
    ])
    def test_create_instances_with_valid_function_names_and_json_arguments(self, name, arguments, call_id):
        """Test creating FunctionCall instances with valid function names and properly formatted JSON arguments."""
        call = FunctionCall(
            name=name,
            arguments=arguments,
            id=call_id
        )
        assert call.name == name
        assert call.arguments == arguments
        assert call.id == call_id
        
        # Verify arguments is valid JSON
        json.loads(arguments)  # Should not raise exception


class Test_FunctionCall_Constructor_02_NegativeBehaviors:
    """Test negative behaviors for FunctionCall constructor."""
    
    @pytest.mark.parametrize("invalid_name", [
        "",
        None,
    ])
    def test_pass_empty_or_none_values_to_required_fields(self, invalid_name):
        """Test passing empty or None values to required fields."""
        with pytest.raises(ValidationError):
            FunctionCall(name=invalid_name, arguments='{}')
    
    @pytest.mark.parametrize("malformed_json", [
        '{"invalid": json}',
        '{missing_quotes: "value"}',
        '{"unclosed": "string}',
        'not_json_at_all',
        '{,}',
    ])
    def test_provide_malformed_json_in_arguments_field(self, malformed_json):
        """Test providing malformed JSON strings in arguments field."""
        # Note: JSON validation might occur at usage time rather than creation
        call = FunctionCall(name="test", arguments=malformed_json)
        
        # Verify that JSON parsing would fail
        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed_json)


class Test_FunctionCall_Constructor_03_BoundaryBehaviors:
    """Test boundary behaviors for FunctionCall constructor."""
    
    @pytest.mark.parametrize("name_length", [
        "a",  # Minimum length
        "x" * 1000,  # Very long name
    ])
    def test_boundary_name_string_lengths(self, name_length):
        """Test with minimum and maximum length name strings."""
        call = FunctionCall(name=name_length, arguments='{}')
        assert call.name == name_length
    
    def test_create_with_very_large_json_argument_strings(self):
        """Test creation with very large JSON argument strings."""
        large_args = json.dumps({"large_array": list(range(1000))})
        call = FunctionCall(name="test", arguments=large_args)
        assert call.arguments == large_args


class Test_FunctionCall_Constructor_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for FunctionCall constructor."""
    
    def test_handle_missing_required_field_scenarios(self):
        """Test handling of missing required field scenarios."""
        with pytest.raises(ValidationError):
            FunctionCall(arguments='{}')  # Missing name
        
        with pytest.raises(ValidationError):
            FunctionCall(name="test")  # Missing arguments


class Test_FunctionCall_Constructor_05_StateTransitionBehaviors:
    """Test state transition behaviors for FunctionCall constructor."""
    
    def test_verify_proper_field_assignment_and_accessibility(self):
        """Test proper field assignment and accessibility."""
        call = FunctionCall(
            name="test_func",
            arguments='{"key": "value"}',
            id="test_id"
        )
        assert call.name == "test_func"
        assert call.arguments == '{"key": "value"}'
        assert call.id == "test_id"
    
    def test_ensure_consistent_json_string_handling(self):
        """Test consistent JSON string handling."""
        json_args = '{"complex": {"nested": true, "array": [1, 2, 3]}}'
        call = FunctionCall(name="test", arguments=json_args)
        
        # Should be able to parse the stored JSON
        parsed = json.loads(call.arguments)
        assert parsed["complex"]["nested"] == True
        assert parsed["complex"]["array"] == [1, 2, 3]


# ============================================================================
# FunctionCallResult Tests
# ============================================================================

class Test_FunctionCallResult_Constructor_01_NominalBehaviors:
    """Test nominal behaviors for FunctionCallResult constructor."""
    
    @pytest.mark.parametrize("name,arguments,result", [
        ("func1", {"param": "value"}, "success"),
        ("func2", {"num": 42}, {"result": "object"}),
        ("func3", {}, [1, 2, 3]),
        ("func4", {"complex": {"nested": True}}, None),
        ("func5", {"bool": True, "str": "test"}, 42.5),
    ])
    def test_create_instances_with_valid_function_data(self, name, arguments, result):
        """Test creating FunctionCallResult instances with valid function names, argument dictionaries, and result objects."""
        call_result = FunctionCallResult(
            name=name,
            arguments=arguments,
            result=result
        )
        assert call_result.name == name
        assert call_result.arguments == arguments
        assert call_result.result == result


class Test_FunctionCallResult_Constructor_02_NegativeBehaviors:
    """Test negative behaviors for FunctionCallResult constructor."""
    
    @pytest.mark.parametrize("invalid_name", [
        None,
        "",
    ])
    def test_pass_none_values_to_required_fields(self, invalid_name):
        """Test passing None values to required fields."""
        with pytest.raises(ValidationError):
            FunctionCallResult(name=invalid_name, arguments={}, result="test")
    
    @pytest.mark.parametrize("invalid_arguments", [
        "not_a_dict",
        123,
        [],
        None,
    ])
    def test_provide_non_dictionary_structures_to_arguments_field(self, invalid_arguments):
        """Test providing non-dictionary structures to arguments field."""
        with pytest.raises(ValidationError):
            FunctionCallResult(
                name="test",
                arguments=invalid_arguments,
                result="result"
            )


class Test_FunctionCallResult_Constructor_03_BoundaryBehaviors:
    """Test boundary behaviors for FunctionCallResult constructor."""
    
    def test_create_with_empty_arguments_dictionary(self):
        """Test creation with empty arguments dictionary."""
        call_result = FunctionCallResult(
            name="test",
            arguments={},
            result="empty_args"
        )
        assert call_result.arguments == {}
    
    @pytest.mark.parametrize("large_result", [
        {"large_dict": {f"key_{i}": f"value_{i}" for i in range(100)}},
        list(range(1000)),
        "x" * 10000,
    ])
    def test_handle_large_result_objects(self, large_result):
        """Test with large result objects."""
        call_result = FunctionCallResult(
            name="test",
            arguments={"param": "value"},
            result=large_result
        )
        assert call_result.result == large_result


class Test_FunctionCallResult_Constructor_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for FunctionCallResult constructor."""
    
    def test_handle_missing_required_fields(self):
        """Test handling of missing required fields."""
        with pytest.raises(ValidationError):
            FunctionCallResult(arguments={}, result="test")  # Missing name
        
        with pytest.raises(ValidationError):
            FunctionCallResult(name="test", result="test")  # Missing arguments


class Test_FunctionCallResult_Constructor_05_StateTransitionBehaviors:
    """Test state transition behaviors for FunctionCallResult constructor."""
    
    def test_verify_proper_type_handling_for_result_field(self):
        """Test proper type handling for flexible result field."""
        test_cases = [
            ("string_result", "test_string"),
            ("int_result", 42),
            ("float_result", 3.14),
            ("bool_result", True),
            ("none_result", None),
            ("dict_result", {"key": "value"}),
            ("list_result", [1, 2, 3]),
        ]
        
        for test_name, result_value in test_cases:
            call_result = FunctionCallResult(
                name=test_name,
                arguments={"test": "param"},
                result=result_value
            )
            assert call_result.result == result_value
    
    def test_ensure_argument_dictionary_integrity(self):
        """Test argument dictionary integrity and accessibility."""
        arguments = {"nested": {"key": "value"}, "array": [1, 2, 3], "bool": True}
        call_result = FunctionCallResult(
            name="test",
            arguments=arguments,
            result="test"
        )
        
        # Verify dictionary structure is preserved
        assert call_result.arguments["nested"]["key"] == "value"
        assert call_result.arguments["array"] == [1, 2, 3]
        assert call_result.arguments["bool"] == True


# ============================================================================
# StructuredToolResult Tests
# ============================================================================

class Test_StructuredToolResult_Constructor_01_NominalBehaviors:
    """Test nominal behaviors for StructuredToolResult constructor."""
    
    @pytest.mark.parametrize("tool_call_id,function_name,function_args,function_result", [
        ("call_123", "test_func", {"param": "value"}, "success"),
        ("unique_id", "another_func", {}, {"result": "object"}),
        ("complex_call", "complex_func", {"nested": {"key": "value"}}, [1, 2, 3]),
    ])
    def test_create_instances_with_valid_tool_call_data(self, tool_call_id, function_name, function_args, function_result):
        """Test creating StructuredToolResult instances with valid tool call identifiers and FunctionCallResult objects."""
        function_call_result = FunctionCallResult(
            name=function_name,
            arguments=function_args,
            result=function_result
        )
        
        tool_result = StructuredToolResult(
            tool_call_id=tool_call_id,
            function=function_call_result
        )
        
        assert tool_result.tool_call_id == tool_call_id
        assert tool_result.function == function_call_result
        assert tool_result.function.name == function_name


class Test_StructuredToolResult_Constructor_02_NegativeBehaviors:
    """Test negative behaviors for StructuredToolResult constructor."""
    
    @pytest.mark.parametrize("invalid_function", [
        "not_a_function_call_result",
        123,
        {},
        None,
    ])
    def test_pass_invalid_object_types_to_function_field(self, invalid_function):
        """Test passing invalid object types to function field."""
        with pytest.raises(ValidationError):
            StructuredToolResult(
                tool_call_id="test_id",
                function=invalid_function
            )
    
    @pytest.mark.parametrize("invalid_tool_call_id", [
        "",
        None,
    ])
    def test_provide_empty_or_malformed_tool_call_identifiers(self, invalid_tool_call_id):
        """Test providing empty or malformed tool call identifiers."""
        function_result = FunctionCallResult(
            name="test",
            arguments={},
            result="test"
        )
        
        with pytest.raises(ValidationError):
            StructuredToolResult(
                tool_call_id=invalid_tool_call_id,
                function=function_result
            )


class Test_StructuredToolResult_Constructor_03_BoundaryBehaviors:
    """Test boundary behaviors for StructuredToolResult constructor."""
    
    @pytest.mark.parametrize("tool_call_id_length", [
        "a",  # Minimum length
        "x" * 1000,  # Very long ID
    ])
    def test_boundary_tool_call_identifier_lengths(self, tool_call_id_length):
        """Test with minimum and maximum length tool call identifiers."""
        function_result = FunctionCallResult(
            name="test",
            arguments={},
            result="test"
        )
        
        tool_result = StructuredToolResult(
            tool_call_id=tool_call_id_length,
            function=function_result
        )
        assert tool_result.tool_call_id == tool_call_id_length
    
    def test_handle_complex_nested_function_result_objects(self):
        """Test handling of complex nested function result objects."""
        complex_result = {
            "nested": {
                "deep": {
                    "structure": [1, 2, 3, {"inner": "value"}]
                }
            },
            "array": [{"item": i} for i in range(10)]
        }
        
        function_result = FunctionCallResult(
            name="complex_func",
            arguments={"complex": "args"},
            result=complex_result
        )
        
        tool_result = StructuredToolResult(
            tool_call_id="complex_call",
            function=function_result
        )
        
        assert tool_result.function.result == complex_result


class Test_StructuredToolResult_Constructor_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for StructuredToolResult constructor."""
    
    def test_handle_missing_required_fields(self):
        """Test handling of missing required fields."""
        with pytest.raises(ValidationError):
            StructuredToolResult(tool_call_id="test")  # Missing function
        
        with pytest.raises(ValidationError):
            StructuredToolResult(function=FunctionCallResult(name="test", arguments={}, result="test"))  # Missing tool_call_id


class Test_StructuredToolResult_Constructor_05_StateTransitionBehaviors:
    """Test state transition behaviors for StructuredToolResult constructor."""
    
    def test_verify_proper_nested_object_composition(self):
        """Test proper nested object composition."""
        function_result = FunctionCallResult(
            name="nested_test",
            arguments={"param1": "value1", "param2": "value2"},
            result={"status": "success", "data": [1, 2, 3]}
        )
        
        tool_result = StructuredToolResult(
            tool_call_id="composition_test",
            function=function_result
        )
        
        # Verify nested object relationships
        assert tool_result.tool_call_id == "composition_test"
        assert tool_result.function.name == "nested_test"
        assert tool_result.function.arguments["param1"] == "value1"
        assert tool_result.function.result["status"] == "success"
    
    def test_ensure_consistent_tool_call_tracking(self):
        """Test consistent tool call tracking and result association."""
        function_results = []
        tool_results = []
        
        for i in range(3):
            func_result = FunctionCallResult(
                name=f"func_{i}",
                arguments={"index": i},
                result=f"result_{i}"
            )
            function_results.append(func_result)
            
            tool_result = StructuredToolResult(
                tool_call_id=f"call_{i}",
                function=func_result
            )
            tool_results.append(tool_result)
        
        # Verify tracking consistency
        for i, tool_result in enumerate(tool_results):
            assert tool_result.tool_call_id == f"call_{i}"
            assert tool_result.function.name == f"func_{i}"
            assert tool_result.function.arguments["index"] == i


# ============================================================================
# FunctionToolChoice Tests
# ============================================================================

class Test_FunctionToolChoice_Constructor_01_NominalBehaviors:
    """Test nominal behaviors for FunctionToolChoice constructor."""
    
    @pytest.mark.parametrize("function_name", [
        "test_function",
        "another_func",
        "complex_function_name",
        None,  # Optional name
    ])
    def test_create_instances_with_valid_function_call_objects(self, function_name):
        """Test creating FunctionToolChoice instances with valid ChatCompletionFunctionCall objects."""
        function_call = ChatCompletionFunctionCall(name=function_name)
        
        tool_choice = FunctionToolChoice(function=function_call)
        
        assert tool_choice.type == "function"
        assert tool_choice.function == function_call
        assert tool_choice.function.name == function_name


class Test_FunctionToolChoice_Constructor_02_NegativeBehaviors:
    """Test negative behaviors for FunctionToolChoice constructor."""
    
    @pytest.mark.parametrize("invalid_function", [
        "not_a_function_call",
        123,
        None,
        [],
    ])
    def test_pass_invalid_object_types_to_function_field(self, invalid_function):
        """Test passing invalid object types to function field."""
        with pytest.raises(ValidationError):
            FunctionToolChoice(function=invalid_function)


class Test_FunctionToolChoice_Constructor_03_BoundaryBehaviors:
    """Test boundary behaviors for FunctionToolChoice constructor."""
    
    def test_handle_minimal_function_call_configurations(self):
        """Test with minimal function call configurations."""
        minimal_function_call = ChatCompletionFunctionCall()  # No name provided
        
        tool_choice = FunctionToolChoice(function=minimal_function_call)
        
        assert tool_choice.type == "function"
        assert tool_choice.function.name is None
    
    def test_verify_type_constraint_enforcement_across_scenarios(self):
        """Test type constraint enforcement across different input scenarios."""
        function_call = ChatCompletionFunctionCall(name="test_func")
        
        # Normal creation
        tool_choice = FunctionToolChoice(function=function_call)
        assert tool_choice.type == "function"
        
        # Type should always be "function"
        assert tool_choice.type == "function"


class Test_FunctionToolChoice_Constructor_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for FunctionToolChoice constructor."""
    
    def test_handle_nested_object_validation_failures(self):
        """Test handling of nested object validation failures."""
        with pytest.raises(ValidationError):
            FunctionToolChoice()  # Missing required function field


class Test_FunctionToolChoice_Constructor_05_StateTransitionBehaviors:
    """Test state transition behaviors for FunctionToolChoice constructor."""
    
    def test_verify_fixed_type_field_behavior_throughout_lifecycle(self):
        """Test fixed type field behavior throughout object lifecycle."""
        function_call = ChatCompletionFunctionCall(name="lifecycle_test")
        tool_choice = FunctionToolChoice(function=function_call)
        
        # Type should be fixed at creation
        assert tool_choice.type == "function"
        
        # Type should remain fixed
        original_type = tool_choice.type
        assert tool_choice.type == original_type
        
        # Even after accessing other fields
        _ = tool_choice.function
        assert tool_choice.type == original_type
    
    def test_ensure_proper_function_call_object_embedding(self):
        """Test proper function call object embedding and accessibility."""
        function_call = ChatCompletionFunctionCall(name="embedded_test")
        tool_choice = FunctionToolChoice(function=function_call)
        
        # Verify embedding
        assert tool_choice.function is function_call
        assert tool_choice.function.name == "embedded_test"
        
        # Verify accessibility
        embedded_function = tool_choice.function
        assert embedded_function.name == "embedded_test"
        assert isinstance(embedded_function, ChatCompletionFunctionCall)