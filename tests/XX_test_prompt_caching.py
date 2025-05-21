"""Tests for the prompt caching functionality."""

import unittest
from unittest.mock import MagicMock, patch

from openrouter_client.models.core import TextContent, CacheControl
from openrouter_client.endpoints.chat import ChatEndpoint
from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager


class TestPromptCaching(unittest.TestCase):
    """Test cases for the prompt caching functionality."""

    def setUp(self):
        self.auth_manager = MagicMock(spec=AuthManager)
        self.http_manager = MagicMock(spec=HTTPManager)
        
        # Mock response from http_manager
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
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cached_tokens": 50,
                "cache_discount": -0.25
            }
        }
        self.http_manager.post.return_value = mock_response
        
        # Create chat endpoint
        self.chat_endpoint = ChatEndpoint(self.auth_manager, self.http_manager)
    
    def test_include_usage_parameter(self):
        """Test that the include parameter with usage=true returns cache information."""
        # Call the create method with include={"usage": True}
        response = self.chat_endpoint.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
            include={"usage": True}
        )
        
        # Check that the include parameter was passed correctly
        request_data = self.http_manager.post.call_args[1]["json"]
        self.assertIn("include", request_data)
        self.assertEqual(request_data["include"], {"usage": True})
        
        # Check that the response contains usage with cache information
        self.assertIn("usage", response)
        self.assertIn("cached_tokens", response["usage"])
        self.assertIn("cache_discount", response["usage"])
        self.assertEqual(response["usage"]["cached_tokens"], 50)
        self.assertEqual(response["usage"]["cache_discount"], -0.25)
    
    def test_content_with_cache_control(self):
        """Test that messages with cache_control are properly structured."""
        # Create a message with cache control
        content = [
            {"type": "text", "text": "Regular content"},
            {"type": "text", "text": "Content to cache", "cache_control": {"type": "ephemeral"}}
        ]
        
        # Call the create method with the message
        self.chat_endpoint.create(
            messages=[{"role": "user", "content": content}],
            model="test-model"
        )
        
        # Check that the message was structured correctly
        request_data = self.http_manager.post.call_args[1]["json"]
        sent_message = request_data["messages"][0]
        
        self.assertEqual(sent_message["role"], "user")
        self.assertEqual(len(sent_message["content"]), 2)
        
        # Check first content part (without cache control)
        self.assertEqual(sent_message["content"][0]["type"], "text")
        self.assertEqual(sent_message["content"][0]["text"], "Regular content")
        self.assertNotIn("cache_control", sent_message["content"][0])
        
        # Check second content part (with cache control)
        self.assertEqual(sent_message["content"][1]["type"], "text")
        self.assertEqual(sent_message["content"][1]["text"], "Content to cache")
        self.assertIn("cache_control", sent_message["content"][1])
        self.assertEqual(sent_message["content"][1]["cache_control"]["type"], "ephemeral")


if __name__ == "__main__":
    unittest.main()