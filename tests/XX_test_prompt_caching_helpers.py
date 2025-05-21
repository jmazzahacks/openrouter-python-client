"""Tests for the prompt caching helper functions."""

import unittest

from openrouter_client.tools import cache_control, create_cached_content, string_param_with_cache_control
from openrouter_client.models.core import TextContent, CacheControl


class TestPromptCachingHelpers(unittest.TestCase):
    """Test cases for the prompt caching helper functions."""

    def test_cache_control(self):
        """Test that cache_control creates a valid CacheControl object."""
        # Default type
        control = cache_control()
        self.assertIsInstance(control, CacheControl)
        self.assertEqual(control.type, "ephemeral")
        
        # Custom type
        control = cache_control(type="custom_type")
        self.assertEqual(control.type, "custom_type")
    
    def test_create_cached_content(self):
        """Test that create_cached_content creates a valid TextContent object with cache control."""
        # With default cache control
        text = "This is test content to be cached"
        content = create_cached_content(text)
        
        self.assertIsInstance(content, TextContent)
        self.assertEqual(content.type, "text")
        self.assertEqual(content.text, text)
        self.assertIsNotNone(content.cache_control)
        self.assertEqual(content.cache_control.type, "ephemeral")
        
        # With custom cache control
        custom_control = cache_control(type="custom_type")
        content = create_cached_content(text, custom_control)
        
        self.assertEqual(content.cache_control.type, "custom_type")
    
    def test_string_param_with_cache_control(self):
        """Test that string_param_with_cache_control creates a valid parameter schema."""
        # With description that doesn't mention cache control
        param = string_param_with_cache_control(
            description="A text parameter",
            required=True
        )
        
        self.assertIsInstance(param, dict)
        self.assertEqual(param["type"], "string")
        self.assertIn("cache_control", param["description"].lower())
        
        # With description that already mentions cache control
        param = string_param_with_cache_control(
            description="A text parameter with cache_control support",
            required=True
        )
        
        self.assertEqual(param["description"], "A text parameter with cache_control support")


if __name__ == "__main__":
    unittest.main()