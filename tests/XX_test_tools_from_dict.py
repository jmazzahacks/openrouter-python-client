"""Tests for creating tools from Python built-in types (dicts, lists, etc.)."""

import json
from typing import Dict, List, Optional

import pytest
from pydantic import BaseModel

from openrouter_client.models.chat import (
    ParameterType, StringParameter, NumberParameter, ArrayParameter,
    ObjectParameter, FunctionCall, ChatCompletionToolCall, ChatCompletionTool
)
from openrouter_client.models.core import FunctionDefinition, ToolDefinition
from openrouter_client.tools import (
    create_parameter_schema_from_value,
    create_function_definition_from_dict,
    create_tool_definition_from_dict,
    create_chat_completion_tool_from_dict,
    create_function_call_from_dict,
    create_tool_call_from_dict
)


def test_create_parameter_schema_from_value():
    """Test creating parameter schema from Python values."""
    # Test string parameter
    string_param = create_parameter_schema_from_value("example", "test_string", "A test string")
    assert isinstance(string_param, StringParameter)
    assert string_param.type == ParameterType.STRING
    assert string_param.description == "A test string"
    assert string_param.default == "example"
    
    # Test number parameter (int)
    int_param = create_parameter_schema_from_value(42, "test_int", "A test integer")
    assert isinstance(int_param, NumberParameter)
    assert int_param.type == ParameterType.INTEGER
    assert int_param.description == "A test integer"
    assert int_param.default == 42
    
    # Test number parameter (float)
    float_param = create_parameter_schema_from_value(3.14, "test_float", "A test float")
    assert isinstance(float_param, NumberParameter)
    assert float_param.type == ParameterType.NUMBER
    assert float_param.description == "A test float"
    assert float_param.default == 3.14
    
    # Test array parameter
    array_param = create_parameter_schema_from_value([1, 2, 3], "test_array", "A test array")
    assert isinstance(array_param, ArrayParameter)
    assert array_param.type == ParameterType.ARRAY
    assert array_param.description == "A test array"
    assert array_param.default == [1, 2, 3]
    assert array_param.items["type"] == ParameterType.INTEGER
    
    # Test object parameter
    obj_param = create_parameter_schema_from_value(
        {"name": "test", "value": 42}, "test_object", "A test object"
    )
    assert isinstance(obj_param, ObjectParameter)
    assert obj_param.type == ParameterType.OBJECT
    assert obj_param.description == "A test object"
    assert obj_param.properties["name"]["type"] == ParameterType.STRING
    assert obj_param.properties["value"]["type"] == ParameterType.INTEGER
    assert obj_param.required == ["name", "value"]


def test_create_function_definition_from_dict():
    """Test creating function definition from dictionary of parameters."""
    # Define a simple function with parameters
    parameters = {
        "query": "test query",
        "max_results": 10,
        "filters": {"year": 2023, "published": True}
    }
    
    func_def = create_function_definition_from_dict(
        "search", parameters, "Search for documents"
    )
    
    # Verify basic properties
    assert isinstance(func_def, FunctionDefinition)
    assert func_def.name == "search"
    assert func_def.description == "Search for documents"
    
    # Verify parameters
    params = func_def.parameters
    assert params["type"] == "object"
    assert set(params["properties"].keys()) == {"query", "max_results", "filters"}
    assert params["properties"]["query"]["type"] == ParameterType.STRING
    assert params["properties"]["max_results"]["type"] == ParameterType.INTEGER
    assert params["properties"]["filters"]["type"] == ParameterType.OBJECT
    
    # Verify required parameters
    assert set(params["required"]) == {"query", "max_results", "filters"}
    
    # Test with explicit required parameters
    func_def = create_function_definition_from_dict(
        "search", parameters, "Search for documents", ["query"]
    )
    assert params["required"] != ["query"]  # This checks that the previous test's required params are different


def test_create_tool_definition_from_dict():
    """Test creating tool definition from dictionary of parameters."""
    parameters = {
        "message": "Hello, world!",
        "channel": "general",
        "mentions": ["user1", "user2"]
    }
    
    tool_def = create_tool_definition_from_dict(
        "send_message", parameters, "Send a message to a channel"
    )
    
    # Verify basic properties
    assert isinstance(tool_def, ToolDefinition)
    assert tool_def.type == "function"
    assert isinstance(tool_def.function, FunctionDefinition)
    assert tool_def.function.name == "send_message"
    assert tool_def.function.description == "Send a message to a channel"
    
    # Verify parameters
    params = tool_def.function.parameters
    assert params["properties"]["message"]["type"] == ParameterType.STRING
    assert params["properties"]["channel"]["type"] == ParameterType.STRING
    assert params["properties"]["mentions"]["type"] == ParameterType.ARRAY
    assert params["properties"]["mentions"]["items"]["type"] == ParameterType.STRING


def test_create_chat_completion_tool_from_dict():
    """Test creating chat completion tool from dictionary of parameters."""
    parameters = {
        "query": "weather",
        "location": {"city": "New York", "country": "US"},
        "units": "metric"
    }
    
    chat_tool = create_chat_completion_tool_from_dict(
        "get_weather", parameters, "Get weather information"
    )
    
    # Verify basic properties
    assert isinstance(chat_tool, ChatCompletionTool)
    assert chat_tool.type == "function"
    assert isinstance(chat_tool.function, FunctionDefinition)
    assert chat_tool.function.name == "get_weather"
    assert chat_tool.function.description == "Get weather information"
    
    # Verify parameters
    params = chat_tool.function.parameters
    assert params["properties"]["query"]["type"] == ParameterType.STRING
    assert params["properties"]["location"]["type"] == ParameterType.OBJECT
    assert params["properties"]["location"]["properties"]["city"]["type"] == ParameterType.STRING


def test_create_function_call_from_dict():
    """Test creating function call from dictionary of arguments."""
    args = {
        "query": "test query",
        "max_results": 10,
        "filters": {"year": 2023}
    }
    
    func_call = create_function_call_from_dict(
        "search", args, "call_123"
    )
    
    # Verify basic properties
    assert isinstance(func_call, FunctionCall)
    assert func_call.name == "search"
    assert func_call.id == "call_123"
    
    # Verify arguments
    args_dict = json.loads(func_call.arguments)
    assert args_dict["query"] == "test query"
    assert args_dict["max_results"] == 10
    assert args_dict["filters"]["year"] == 2023


def test_create_tool_call_from_dict():
    """Test creating tool call from dictionary of arguments."""
    args = {
        "message": "Hello, world!",
        "channel": "general",
        "mentions": ["user1", "user2"]
    }
    
    tool_call = create_tool_call_from_dict(
        "send_message", args, "tool_call_456"
    )
    
    # Verify basic properties
    assert isinstance(tool_call, ChatCompletionToolCall)
    assert tool_call.id == "tool_call_456"
    assert tool_call.type == "function"
    assert tool_call.function.name == "send_message"
    
    # Verify arguments
    args_dict = json.loads(tool_call.function.arguments)
    assert args_dict["message"] == "Hello, world!"
    assert args_dict["channel"] == "general"
    assert args_dict["mentions"] == ["user1", "user2"]