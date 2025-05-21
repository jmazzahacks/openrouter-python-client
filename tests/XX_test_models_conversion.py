"""Tests for the ModelData and ModelsResponse Pydantic models."""

import unittest
from typing import Dict, Any

from openrouter_client.models.models import ModelData, ModelPricing, ModelsResponse


class TestModelsPydanticConversion(unittest.TestCase):
    """Test the conversion of API responses to Pydantic models."""
    
    def test_model_data_validation(self):
        """Test that ModelData validates properly with different inputs."""
        # Sample model data from API
        model_json: Dict[str, Any] = {
            "id": "anthropic/claude-3-opus",
            "name": "Anthropic: Claude 3 Opus",
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
        
        # Validate as ModelData
        model = ModelData.model_validate(model_json)
        
        # Check model properties
        self.assertEqual(model.id, "anthropic/claude-3-opus")
        self.assertEqual(model.name, "Anthropic: Claude 3 Opus")
        self.assertEqual(model.context_length, 200000)
        self.assertEqual(model.max_completion_tokens, 4096)
        
        # Check pricing
        self.assertEqual(model.pricing.prompt, "0.000015")
        self.assertEqual(model.pricing.completion, "0.000075")
        
        # Check optional fields
        self.assertEqual(model.description, "Anthropic's most capable model")
        self.assertEqual(model.quantization, "fp16")
    
    def test_models_response_validation(self):
        """Test that ModelsResponse validates properly with a list of models."""
        # Sample API response
        response_json = {
            "data": [
                {
                    "id": "anthropic/claude-3-opus",
                    "name": "Anthropic: Claude 3 Opus",
                    "created": 1709596800,
                    "context_length": 200000,
                    "quantization": "fp16",
                    "pricing": {
                        "prompt": "0.000015",
                        "completion": "0.000075"
                    }
                },
                {
                    "id": "anthropic/claude-3-sonnet",
                    "name": "Anthropic: Claude 3 Sonnet",
                    "created": 1709596800,
                    "context_length": 180000,
                    "quantization": "fp16",
                    "pricing": {
                        "prompt": "0.000003",
                        "completion": "0.000015"
                    }
                }
            ]
        }
        
        # Validate as ModelsResponse
        response = ModelsResponse.model_validate(response_json)
        
        # Check response properties
        self.assertEqual(len(response.data), 2)
        self.assertEqual(response.data[0].id, "anthropic/claude-3-opus")
        self.assertEqual(response.data[1].id, "anthropic/claude-3-sonnet")
        
        # Check nested model properties
        self.assertEqual(response.data[0].context_length, 200000)
        self.assertEqual(response.data[1].context_length, 180000)
        
        # Check pricing in nested models
        self.assertEqual(response.data[0].pricing.prompt, "0.000015")
        self.assertEqual(response.data[1].pricing.prompt, "0.000003")


if __name__ == "__main__":
    unittest.main()