"""Tests for the client integration with models endpoint."""

import unittest
from unittest.mock import MagicMock, patch

from openrouter_client import OpenRouterClient
from openrouter_client.endpoints.models import ModelData, ModelPricing


class TestClientModelsIntegration(unittest.TestCase):
    """Test cases for client integration with models endpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the auth and http managers
        with patch('openrouter_client.auth.AuthManager'), \
             patch('openrouter_client.http.HTTPManager'):
            self.client = OpenRouterClient(api_key="test_key")
        
        # Mock the models endpoint response for list
        mock_models_response = {
            "data": [
                {
                    "id": "model1",
                    "name": "Model 1",
                    "created": 1609459200,
                    "context_length": 8192,
                    "pricing": {
                        "prompt": "0.000008",
                        "completion": "0.000024"
                    }
                },
                {
                    "id": "model2",
                    "name": "Model 2",
                    "created": 1609459200,
                    "context_length": 16384,
                    "pricing": {
                        "prompt": "0.000012",
                        "completion": "0.000036"
                    }
                }
            ]
        }
        
        # Set up the mock for client.models.list
        self.client.models.list = MagicMock(return_value=mock_models_response)
        
        # Set up the mock for client.models.get
        self.client.models.get = MagicMock(return_value=mock_models_response["data"][0])
    
    def test_refresh_context_lengths(self):
        """Test that refresh_context_lengths correctly processes model data."""
        # Call the method
        result = self.client.refresh_context_lengths()
        
        # Verify models.list was called correctly
        self.client.models.list.assert_called_once_with(details=True)
        
        # Check that the context lengths were extracted correctly
        self.assertEqual(len(result), 2)
        self.assertEqual(result["model1"], 8192)
        self.assertEqual(result["model2"], 16384)
    
    def test_client_with_models_endpoint(self):
        """Test client integration with models endpoint."""
        # Call the list method through the client
        models = self.client.models.list(details=True)
        
        # Verify response format
        self.assertIsInstance(models, dict)
        self.assertIn("data", models)
        self.assertEqual(len(models["data"]), 2)
        
        # Test getting a single model
        model = self.client.models.get("model1")
        self.assertEqual(model["id"], "model1")
        self.assertEqual(model["context_length"], 8192)


if __name__ == "__main__":
    unittest.main()