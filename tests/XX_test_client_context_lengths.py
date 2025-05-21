"""Tests for the refresh_context_lengths functionality in OpenRouterClient."""

import unittest
from unittest.mock import MagicMock, patch

from openrouter_client import OpenRouterClient
from openrouter_client.models.core import _CONTEXT_LENGTHS

class TestClientContextLengths(unittest.TestCase):
    """Test cases for context length refreshing from models endpoint."""
    
    def setUp(self):
        # Mock the auth and http managers
        with patch('openrouter_client.auth.AuthManager'), \
             patch('openrouter_client.http.HTTPManager'):
            self.client = OpenRouterClient(api_key="test_key")
        
        # Mock the models endpoint
        self.client.models = MagicMock()
        
        # Set up a sample response with detailed model information
        self.mock_models_response = {
            'data': [
                {
                    'id': 'model1',
                    'context_length': 8192,
                    'other_field': 'value'
                },
                {
                    'id': 'model2',
                    'context_length': 16384,
                    'other_field': 'value'
                },
                {
                    # Missing context_length
                    'id': 'model3'
                }
            ]
        }
        
        # Reset context lengths between tests
        _CONTEXT_LENGTHS.clear()
    
    def test_refresh_context_lengths_with_dict_response(self):
        """Test refresh_context_lengths when models.list returns a dict response."""
        # Configure the mock to return our sample response
        self.client.models.list.return_value = self.mock_models_response
        
        # Call the method
        result = self.client.refresh_context_lengths()
        
        # Verify models.list was called with details=True
        self.client.models.list.assert_called_once_with(details=True)
        
        # Verify the correct context lengths were extracted
        self.assertEqual(len(result), 2)  # Only 2 models have context_length
        self.assertEqual(result['model1'], 8192)
        self.assertEqual(result['model2'], 16384)
        
        # Verify global context lengths were updated
        self.assertEqual(_CONTEXT_LENGTHS['model1'], 8192)
        self.assertEqual(_CONTEXT_LENGTHS['model2'], 16384)
        self.assertNotIn('model3', _CONTEXT_LENGTHS)  # Should not include model3
    
    def test_refresh_context_lengths_with_list_response(self):
        """Test refresh_context_lengths when models.list returns a list response."""
        # Configure the mock to return a list response
        self.client.models.list.return_value = self.mock_models_response['data']
        
        # Call the method
        result = self.client.refresh_context_lengths()
        
        # Verify models.list was called with details=True
        self.client.models.list.assert_called_once_with(details=True)
        
        # Verify the correct context lengths were extracted
        self.assertEqual(len(result), 2)  # Only 2 models have context_length
        self.assertEqual(result['model1'], 8192)
        self.assertEqual(result['model2'], 16384)

if __name__ == "__main__":
    unittest.main()