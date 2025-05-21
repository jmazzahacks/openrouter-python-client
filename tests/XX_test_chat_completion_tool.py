"""Tests for the ChatCompletionTool builder functions."""

import json
from typing import Dict, List, Optional

import pytest
from pydantic import BaseModel

from openrouter_client.models.chat import (
    ChatCompletionTool, ChatCompletionToolCall, ToolCallFunction
)
from openrouter_client.models.core import FunctionDefinition
from openrouter_client.tools import (
    build_chat_completion_tool,
    build_tool_call
)


class User(BaseModel):
    """User model for testing."""
    name: str
    age: int
    preferences: Optional[List[str]] = None


def test_build_chat_completion_tool():
    """Test building ChatCompletionTool from a Python function."""
    def process_user(user: User, update_profile: bool = False):
        """Process a user profile.
        
        Args:
            user: The user to process.
            update_profile: Whether to update the user's profile.
        """
        return {"processed": True, "user_name": user.name, "updated": update_profile}
    
    # Build a chat completion tool
    chat_tool = build_chat_completion_tool(process_user)
    
    # Verify the basic properties
    assert isinstance(chat_tool, ChatCompletionTool)
    assert chat_tool.type == "function"
    assert isinstance(chat_tool.function, FunctionDefinition)
    assert chat_tool.function.name == "process_user"
    
    # Check that the parameters were correctly extracted
    params = chat_tool.function.parameters
    assert "type" in params and params["type"] == "object"
    assert "properties" in params
    assert "user" in params["properties"]
    assert "update_profile" in params["properties"]
    assert params["properties"]["update_profile"]["type"] == "boolean"
    
    # Check that the user parameter is correctly defined as an object
    user_param = params["properties"]["user"]
    assert user_param["type"] == "object"
    assert "properties" in user_param
    assert "name" in user_param["properties"]
    assert "age" in user_param["properties"]
    assert "preferences" in user_param["properties"]


def test_build_tool_call():
    """Test building ChatCompletionToolCall from a function and arguments."""
    def calculate(operation: str, x: float, y: float):
        """Perform calculation on two numbers."""
        return {"result": x + y if operation == "add" else x - y}
    
    # Build a tool call
    args = {"operation": "add", "x": 10.5, "y": 5.2}
    tool_call = build_tool_call(calculate, args, "call_123")
    
    # Verify the basic properties
    assert isinstance(tool_call, ChatCompletionToolCall)
    assert tool_call.id == "call_123"
    assert tool_call.type == "function"
    assert isinstance(tool_call.function, ToolCallFunction)
    assert tool_call.function.name == "calculate"
    
    # Verify the arguments were correctly serialized
    args_dict = json.loads(tool_call.function.arguments)
    assert args_dict["operation"] == "add"
    assert args_dict["x"] == 10.5
    assert args_dict["y"] == 5.2