"""Tests for the models endpoint handler."""

import unittest
from unittest.mock import MagicMock, patch

from openrouter_client.endpoints.models import ModelsEndpoint
from openrouter_client.models.models import ModelData, ModelPricing, ModelsResponse
from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager


class Test_ModelsEndpoint_01_NominalBehavior(unittest.TestCase):
    """Test cases for nominal behaviors of the models endpoint."""

    def setUp(self):
        """Set up test fixtures."""
        self.auth_manager = MagicMock(spec=AuthManager)
        self.http_manager = MagicMock(spec=HTTPManager)
        # Mock the _get_headers method
        self.auth_manager._get_headers = MagicMock(return_value={"Authorization": "Bearer test_key"})
        
        # Mock response for list endpoint
        self.list_response_mock = MagicMock()
        self.list_response_mock.json.return_value = {
            "data": [
                {
                    "id": "anthropic/claude-3-opus",
                    "name": "Claude 3 Opus",
                    "created": 1709596800,
                    "description": "Anthropic's most capable model",
                    "context_length": 200000,
                    "max_completion_tokens": 4096,
                    "quantization": "fp16",
                    "pricing": {
                        "prompt": "0.000015",
                        "completion": "0.000075"
                    }
                },
                {
                    "id": "anthropic/claude-3-sonnet",
                    "name": "Claude 3 Sonnet",
                    "created": 1709596800,
                    "description": "Anthropic's balanced model",
                    "context_length": 180000,
                    "max_completion_tokens": 4096,
                    "quantization": "fp16",
                    "pricing": {
                        "prompt": "0.000003",
                        "completion": "0.000015"
                    }
                }
            ]
        }
        
        # Mock response for get endpoint
        self.get_response_mock = MagicMock()
        self.get_response_mock.json.return_value = {
            "id": "anthropic/claude-3-opus",
            "name": "Claude 3 Opus",
            "created": 1709596800,
            "description": "Anthropic's most capable model",
            "context_length": 200000,
            "max_completion_tokens": 4096,
            "quantization": "fp16",
            "pricing": {
                "prompt": "0.000015",
                "completion": "0.000075"
            }
        }
        
        # Create models endpoint
        self.models_endpoint = ModelsEndpoint(self.auth_manager, self.http_manager)
    
    def test_list_with_details(self):
        """Test listing models with details=True."""
        # Set up the mock
        self.http_manager.get.return_value = self.list_response_mock
        
        # Call the method with details=True
        response = self.models_endpoint.list(details=True)
        
        # Assert endpoint URL and params were correct
        self.http_manager.get.assert_called_once()
        call_args = self.http_manager.get.call_args
        self.assertEqual(call_args[0][0], "models")  # Endpoint URL
        self.assertEqual(call_args[1]["params"], {"details": "true"})
        # Don't assert exact headers as they're handled internally
        
        # Check response format
        self.assertIsInstance(response, ModelsResponse)
        self.assertEqual(len(response.data), 2)
        
        # Check model data fields
        model = response.data[0]
        self.assertEqual(model.id, "anthropic/claude-3-opus")
        self.assertEqual(model.context_length, 200000)
        self.assertIsNotNone(model.pricing)
        self.assertEqual(model.pricing.prompt, "0.000015")
    
    def test_list_without_details(self):
        """Test listing models with details=False."""
        # Set up the mock
        self.http_manager.get.return_value = self.list_response_mock
        
        # Call the method with details=False
        response = self.models_endpoint.list(details=False)
        
        # Assert endpoint URL and params were correct
        self.http_manager.get.assert_called_once()
        call_args = self.http_manager.get.call_args
        self.assertEqual(call_args[0][0], "models")  # Endpoint URL
        self.assertEqual(call_args[1]["params"], {"details": "false"})
        # Don't assert exact headers as they're handled internally
        
        # Check response format - should just be a list of IDs
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 2)
        self.assertEqual(response[0], "anthropic/claude-3-opus")
        self.assertEqual(response[1], "anthropic/claude-3-sonnet")
    
    def test_get_model(self):
        """Test getting a specific model."""
        # Set up the mock
        self.http_manager.get.return_value = self.get_response_mock
        
        # Call the method
        response = self.models_endpoint.get("anthropic/claude-3-opus")
        
        # Assert correct URL was used
        self.http_manager.get.assert_called_once()
        call_args = self.http_manager.get.call_args
        self.assertEqual(call_args[0][0], "models/anthropic/claude-3-opus")
        # Don't assert exact headers as they're handled internally
        
        # Check response
        self.assertEqual(response.id, "anthropic/claude-3-opus")
        self.assertEqual(response.context_length, 200000)
    
    def test_get_context_length(self):
        """Test getting context length for a model."""
        # Set up the mock for HTTP get
        self.http_manager.get.return_value = self.get_response_mock
        
        # Mock the get method directly
        original_get = self.models_endpoint.get
        try:
            # Replace get method with a mock
            model_data = ModelData(
                id="anthropic/claude-3-opus",
                name="Claude 3 Opus",
                created=1709596800,
                context_length=200000,
                quantization="fp16",
                pricing=ModelPricing(prompt="0.000015", completion="0.000075")
            )
            self.models_endpoint.get = MagicMock(return_value=model_data)
            
            # Call the method
            context_length = self.models_endpoint.get_context_length("anthropic/claude-3-opus")
            
            # Check that mock get method was called with the right ID
            self.models_endpoint.get.assert_called_once_with("anthropic/claude-3-opus")
            
            # Assert correct value was returned
            self.assertEqual(context_length, 200000)
        finally:
            # Restore original method
            self.models_endpoint.get = original_get
    
    def test_get_model_pricing(self):
        """Test getting pricing information for a model."""
        # Set up the mock for HTTP get
        self.http_manager.get.return_value = self.get_response_mock
        
        # Mock the get method directly
        original_get = self.models_endpoint.get
        try:
            # Replace get method with a mock
            model_data = ModelData(
                id="anthropic/claude-3-opus",
                name="Claude 3 Opus",
                created=1709596800,
                context_length=200000,
                quantization="fp16",
                pricing=ModelPricing(prompt="0.000015", completion="0.000075")
            )
            self.models_endpoint.get = MagicMock(return_value=model_data)
            
            # Call the method
            pricing = self.models_endpoint.get_model_pricing("anthropic/claude-3-opus")
            
            # Check that mock get method was called with the right ID
            self.models_endpoint.get.assert_called_once_with("anthropic/claude-3-opus")
            
            # Assert correct pricing was returned
            self.assertIsInstance(pricing, ModelPricing)
            self.assertEqual(pricing.prompt, "0.000015")
            self.assertEqual(pricing.completion, "0.000075")
        finally:
            # Restore original method
            self.models_endpoint.get = original_get


class Test_ModelsEndpoint_02_NegativeBehavior(unittest.TestCase):
    """Test cases for negative behaviors of the models endpoint."""

    def setUp(self):
        """Set up test fixtures."""
        self.auth_manager = MagicMock(spec=AuthManager)
        self.http_manager = MagicMock(spec=HTTPManager)
        # Mock the _get_headers method
        self.auth_manager._get_headers = MagicMock(return_value={"Authorization": "Bearer test_key"})
        
        # Create models endpoint
        self.models_endpoint = ModelsEndpoint(self.auth_manager, self.http_manager)
    
    def test_get_context_length_missing(self):
        """Test getting context length when it's not available."""
        # Mock a response with missing required fields
        response_mock = MagicMock()
        response_mock.json.return_value = {
            "id": "test-model",
            "name": "Test Model"
            # Missing required fields like context_length, created, etc.
        }
        self.http_manager.get.return_value = response_mock
        
        # Call should raise validation error
        with self.assertRaises(Exception) as context:
            self.models_endpoint.get_context_length("test-model")
        
        # Should contain validation error about required fields
        self.assertIn("validation error", str(context.exception).lower())
    
    def test_get_pricing_missing(self):
        """Test getting pricing when it's not available."""
        # Mock a response with missing required fields
        response_mock = MagicMock()
        response_mock.json.return_value = {
            "id": "test-model",
            "name": "Test Model"
            # Missing required fields like pricing, created, etc.
        }
        self.http_manager.get.return_value = response_mock
        
        # Call should raise validation error
        with self.assertRaises(Exception) as context:
            self.models_endpoint.get_model_pricing("test-model")
        
        # Should contain validation error about required fields
        self.assertIn("validation error", str(context.exception).lower())


if __name__ == "__main__":
    unittest.main()