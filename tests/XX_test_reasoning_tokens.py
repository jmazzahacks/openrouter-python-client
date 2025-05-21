"""Tests for reasoning tokens support in the OpenRouter client."""

import pytest
from unittest.mock import patch, MagicMock

from openrouter_client.endpoints.chat import ChatEndpoint
from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager


def test_reasoning_parameter_chat():
    """Test that the reasoning parameter is correctly passed to the API."""
    # Create mock objects
    auth_manager = MagicMock(spec=AuthManager)
    http_manager = MagicMock(spec=HTTPManager)
    
    # Create response mock
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response",
                    "reasoning": "Reasoning tokens here"
                },
                "finish_reason": "stop"
            }
        ]
    }
    http_manager.post.return_value = mock_response
    
    # Create endpoint
    chat_endpoint = ChatEndpoint(auth_manager, http_manager)
    
    # Test with effort parameter
    chat_endpoint.create(
        messages=[{"role": "user", "content": "Hello"}],
        reasoning={"effort": "high"}
    )
    
    # Check that the reasoning parameter was correctly passed
    call_args = http_manager.post.call_args
    assert call_args is not None
    _, kwargs = call_args
    assert "json" in kwargs
    assert "reasoning" in kwargs["json"]
    assert kwargs["json"]["reasoning"] == {"effort": "high"}
    
    # Reset mock
    http_manager.post.reset_mock()
    
    # Test with max_tokens parameter
    chat_endpoint.create(
        messages=[{"role": "user", "content": "Hello"}],
        reasoning={"max_tokens": 2000}
    )
    
    # Check that the reasoning parameter was correctly passed
    call_args = http_manager.post.call_args
    assert call_args is not None
    _, kwargs = call_args
    assert "json" in kwargs
    assert "reasoning" in kwargs["json"]
    assert kwargs["json"]["reasoning"] == {"max_tokens": 2000}
    
    # Reset mock
    http_manager.post.reset_mock()
    
    # Test with exclude parameter
    chat_endpoint.create(
        messages=[{"role": "user", "content": "Hello"}],
        reasoning={"effort": "high", "exclude": True}
    )
    
    # Check that the reasoning parameter was correctly passed
    call_args = http_manager.post.call_args
    assert call_args is not None
    _, kwargs = call_args
    assert "json" in kwargs
    assert "reasoning" in kwargs["json"]
    assert kwargs["json"]["reasoning"] == {"effort": "high", "exclude": True}


def test_legacy_include_reasoning_parameter_chat():
    """Test that the include_reasoning parameter is correctly translated to the reasoning parameter."""
    # Create mock objects
    auth_manager = MagicMock(spec=AuthManager)
    http_manager = MagicMock(spec=HTTPManager)
    
    # Create response mock
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                },
                "finish_reason": "stop"
            }
        ]
    }
    http_manager.post.return_value = mock_response
    
    # Create endpoint
    chat_endpoint = ChatEndpoint(auth_manager, http_manager)
    
    # Test with include_reasoning=True
    chat_endpoint.create(
        messages=[{"role": "user", "content": "Hello"}],
        include_reasoning=True
    )
    
    # Check that the reasoning parameter was correctly set
    call_args = http_manager.post.call_args
    assert call_args is not None
    _, kwargs = call_args
    assert "json" in kwargs
    assert "reasoning" in kwargs["json"]
    assert kwargs["json"]["reasoning"] == {}
    
    # Reset mock
    http_manager.post.reset_mock()
    
    # Test with include_reasoning=False
    chat_endpoint.create(
        messages=[{"role": "user", "content": "Hello"}],
        include_reasoning=False
    )
    
    # Check that the reasoning parameter was correctly set
    call_args = http_manager.post.call_args
    assert call_args is not None
    _, kwargs = call_args
    assert "json" in kwargs
    assert "reasoning" in kwargs["json"]
    assert kwargs["json"]["reasoning"] == {"exclude": True}
    
    # Reset mock
    http_manager.post.reset_mock()
    
    # Test precedence when both parameters are specified
    chat_endpoint.create(
        messages=[{"role": "user", "content": "Hello"}],
        reasoning={"effort": "medium"},
        include_reasoning=True  # This should be ignored
    )
    
    # Check that the reasoning parameter takes precedence
    call_args = http_manager.post.call_args
    assert call_args is not None
    _, kwargs = call_args
    assert "json" in kwargs
    assert "reasoning" in kwargs["json"]
    assert kwargs["json"]["reasoning"] == {"effort": "medium"}