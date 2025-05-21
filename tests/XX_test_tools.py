"""Tests for the tools module."""

import json
from enum import Enum
from typing import Dict, List, Optional, Union

import pytest
from pydantic import BaseModel

from openrouter_client.models.chat import (
    ParameterType, FunctionCall, StringParameter, NumberParameter,
    BooleanParameter, ArrayParameter, ObjectParameter, ChatCompletionTool
)
from openrouter_client.models.core import FunctionDefinition, ToolDefinition
from openrouter_client.tools import (
    infer_parameter_type,
    build_parameter_schema,
    build_function_definition,
    build_tool_definition,
    build_chat_completion_tool,
    build_function_call,
    build_tool_call,
    tool
)


class TestEnum(Enum):
    """Test enum for parameter type inference."""
    VALUE1 = "value1"
    VALUE2 = "value2"


class TestModel(BaseModel):
    """Test model for parameter type inference."""
    name: str
    value: int


def test_infer_parameter_type_primitives():
    """Test parameter type inference for primitive types."""
    assert infer_parameter_type(str) == ParameterType.STRING
    assert infer_parameter_type(int) == ParameterType.INTEGER
    assert infer_parameter_type(float) == ParameterType.NUMBER
    assert infer_parameter_type(bool) == ParameterType.BOOLEAN
    assert infer_parameter_type(list) == ParameterType.ARRAY
    assert infer_parameter_type(dict) == ParameterType.OBJECT
    assert infer_parameter_type(type(None)) == ParameterType.NULL


def test_infer_parameter_type_complex():
    """Test parameter type inference for complex types."""
    assert infer_parameter_type(List[str]) == ParameterType.ARRAY
    assert infer_parameter_type(Dict[str, int]) == ParameterType.OBJECT
    # Note: Implementation returns a union type for Optional[str], not just STRING
    result = infer_parameter_type(Optional[str]) 
    assert isinstance(result, list) and ParameterType.STRING in result
    assert isinstance(infer_parameter_type(Union[str, int]), list)
    assert infer_parameter_type(TestEnum) == ParameterType.STRING
    assert infer_parameter_type(TestModel) == ParameterType.OBJECT


def test_build_parameter_schema():
    """Test building parameter schema from Python type."""
    # Test simple string parameter
    string_param = build_parameter_schema("name", str, "User's name", "default_name")
    assert isinstance(string_param, StringParameter)
    assert string_param.type == ParameterType.STRING
    assert string_param.description == "User's name"
    assert string_param.default == "default_name"
    
    # Test enum parameter
    enum_param = build_parameter_schema("status", TestEnum, "Status code")
    assert isinstance(enum_param, StringParameter)
    assert enum_param.type == ParameterType.STRING
    assert enum_param.description == "Status code"
    assert enum_param.enum == ["value1", "value2"]
    
    # Test array parameter
    array_param = build_parameter_schema("items", List[str], "List of items")
    assert isinstance(array_param, ArrayParameter)
    assert array_param.type == ParameterType.ARRAY
    assert array_param.description == "List of items"
    assert array_param.items["type"] == ParameterType.STRING
    
    # Test object parameter with Pydantic model
    model_param = build_parameter_schema("user", TestModel, "User details")
    assert isinstance(model_param, ObjectParameter)
    assert model_param.type == ParameterType.OBJECT
    assert model_param.description == "User details"
    assert model_param.properties is not None
    assert "name" in model_param.properties
    assert "value" in model_param.properties


def test_build_function_definition():
    """Test building function definition from Python function."""
    def test_function(name: str, age: int, options: Optional[List[str]] = None):
        """Test function for definition building.
        
        Args:
            name: User's name.
            age: User's age.
            options: Optional list of settings.
        """
        return {"name": name, "age": age, "options": options}
    
    func_def = build_function_definition(test_function)
    
    assert isinstance(func_def, FunctionDefinition)
    assert func_def.name == "test_function"
    assert "Test function for definition building" in func_def.description
    
    # Check parameters
    params = func_def.parameters
    assert params["type"] == "object"
    assert "name" in params["properties"]
    assert "age" in params["properties"]
    assert "options" in params["properties"]
    assert params["required"] == ["name", "age"]
    
    # Check parameter descriptions
    assert params["properties"]["name"]["description"] == "User's name."
    assert params["properties"]["age"]["description"] == "User's age."


def test_build_tool_definition():
    """Test building tool definition from Python function."""
    def calculator(operation: str, a: float, b: float):
        """Perform basic math operations.
        
        Args:
            operation: Math operation to perform (add, subtract, multiply, divide).
            a: First number.
            b: Second number.
        """
        operations = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else None
        }
        return operations.get(operation)
    
    tool_def = build_tool_definition(calculator)
    
    assert isinstance(tool_def, ToolDefinition)
    assert tool_def.type == "function"
    assert tool_def.function.name == "calculator"
    assert "Perform basic math operations" in tool_def.function.description
    
    # Check function parameters
    func_params = tool_def.function.parameters
    assert "operation" in func_params["properties"]
    assert "a" in func_params["properties"]
    assert "b" in func_params["properties"]
    assert func_params["required"] == ["operation", "a", "b"]


def test_build_function_call():
    """Test building function call from function and arguments."""
    def greet(name: str, formal: bool = False):
        """Greet a user."""
        return "Hello, " + ("Mr./Ms. " if formal else "") + name
    
    args = {"name": "Alice", "formal": True}
    func_call = build_function_call(greet, args, "call123")
    
    assert isinstance(func_call, FunctionCall)
    assert func_call.name == "greet"
    assert func_call.id == "call123"
    
    # Parse arguments JSON
    args_dict = json.loads(func_call.arguments)
    assert args_dict["name"] == "Alice"
    assert args_dict["formal"] is True


def test_tool_decorator():
    """Test the @tool decorator for Python functions."""
    @tool(name="weather", description="Get weather information")
    def get_weather(location: str, units: str = "metric"):
        """Get weather information for a location.
        
        Args:
            location: City or location name.
            units: Temperature units (metric or imperial).
        """
        return {"location": location, "temp": 22.5, "units": units}
    
    # Check that function still works normally
    result = get_weather("London")
    assert result["location"] == "London"
    assert result["units"] == "metric"
    
    # Check attached function definition
    assert hasattr(get_weather, "as_function_definition")
    func_def = get_weather.as_function_definition
    assert isinstance(func_def, FunctionDefinition)
    assert func_def.name == "weather"
    assert func_def.description == "Get weather information"
    
    # Check attached tool definition
    assert hasattr(get_weather, "as_tool_definition")
    tool_def = get_weather.as_tool_definition
    assert isinstance(tool_def, ToolDefinition)
    assert tool_def.type == "function"
    assert tool_def.function.name == "weather"
    
    # Check attached chat completion tool
    assert hasattr(get_weather, "as_chat_completion_tool")
    chat_tool = get_weather.as_chat_completion_tool
    assert isinstance(chat_tool, ChatCompletionTool)
    assert chat_tool.type == "function"
    assert chat_tool.function.name == "weather"