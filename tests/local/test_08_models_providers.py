import pytest
import json
from pydantic import ValidationError

from openrouter_client.models.providers import ProviderPreferences, ProviderMaxPrice


class Test_ProviderMaxPrice_01_NominalBehaviors:
    """Test nominal behaviors for ProviderMaxPrice class."""

    @pytest.mark.parametrize("test_case, input_values, expected_values", [
        (
            "all valid positive values",
            {"prompt": 0.5, "completion": 0.7, "request": 0.1, "image": 0.8},
            {"prompt": 0.5, "completion": 0.7, "request": 0.1, "image": 0.8}
        ),
        (
            "all zero values (minimum)",
            {"prompt": 0.0, "completion": 0.0, "request": 0.0, "image": 0.0},
            {"prompt": 0.0, "completion": 0.0, "request": 0.0, "image": 0.0}
        ),
        (
            "all None values (default)",
            {},
            {"prompt": None, "completion": None, "request": None, "image": None}
        ),
        (
            "mixed None and valid values",
            {"prompt": 0.5, "image": 0.8},
            {"prompt": 0.5, "completion": None, "request": None, "image": 0.8}
        ),
    ])
    def test_init_with_valid_configurations(self, test_case, input_values, expected_values):
        """Verify the class can initialize with various valid price configurations."""
        model = ProviderMaxPrice(**input_values)
        
        for field, expected in expected_values.items():
            assert getattr(model, field) == expected, f"Field {field} should be {expected} but got {getattr(model, field)}"

    @pytest.mark.parametrize("test_case, input_values", [
        (
            "all valid positive values",
            {"prompt": 0.5, "completion": 0.7, "request": 0.1, "image": 0.8}
        ),
        (
            "all None values (default)",
            {}
        ),
        (
            "mixed None and valid values",
            {"prompt": 0.5, "image": 0.8}
        ),
    ])
    def test_serialization_and_deserialization(self, test_case, input_values):
        """Verify the model correctly serializes to and deserializes from JSON."""
        original_model = ProviderMaxPrice(**input_values)
        
        # Serialize to JSON
        json_data = original_model.json()
        
        # Deserialize from JSON
        deserialized_model = ProviderMaxPrice.parse_raw(json_data)
        
        # Verify the deserialized model matches the original
        assert deserialized_model == original_model, f"Deserialized model should match original model"
        
        # Also verify individual fields
        for field in ["prompt", "completion", "request", "image"]:
            assert getattr(deserialized_model, field) == getattr(original_model, field), \
                f"Field {field} mismatch after serialization/deserialization"


class Test_ProviderMaxPrice_02_NegativeBehaviors:
    """Test negative behaviors for ProviderMaxPrice class."""

    @pytest.mark.parametrize("test_case, field, value", [
        ("negative prompt", "prompt", -0.1),
        ("negative completion", "completion", -0.01),
        ("negative request", "request", -1.0),
        ("negative image", "image", -0.001),
    ])
    def test_reject_negative_values(self, test_case, field, value):
        """Verify the model rejects negative price values for all price attributes."""
        input_data = {field: value}
        
        with pytest.raises(ValidationError) as exc_info:
            ProviderMaxPrice(**input_data)
        
        # Verify the error message mentions the field and the constraint
        error_str = str(exc_info.value)
        assert field in error_str, f"Validation error should mention field name '{field}'"
        assert "greater than or equal to 0" in error_str or "ge=" in error_str, \
            "Validation error should mention the constraint (ge=0.0)"

    @pytest.mark.parametrize("test_case, field, value", [
        ("string for prompt", "prompt", "not_a_float"),
        ("boolean for completion", "completion", True),
        ("list for request", "request", [1, 2, 3]),
        ("dict for image", "image", {"nested": "value"}),
    ])
    def test_reject_invalid_data_types(self, test_case, field, value):
        """Verify the model rejects invalid data types for price attributes."""
        input_data = {field: value}
        
        with pytest.raises(ValidationError) as exc_info:
            ProviderMaxPrice(**input_data)
        
        # Verify the error message mentions the field and type error
        error_str = str(exc_info.value)
        assert field in error_str, f"Validation error should mention field name '{field}'"
        assert "type" in error_str.lower(), "Validation error should mention type error"

    @pytest.mark.parametrize("test_case, input_data", [
        (
            "malformed json string",
            '{"prompt": 0.5, "completion": invalid}'
        ),
        (
            "array instead of object",
            '[0.5, 0.7, 0.1, 0.8]'
        ),
    ])
    def test_handle_malformed_input(self, test_case, input_data):
        """Verify the model gracefully handles malformed input data."""
        with pytest.raises((ValidationError, json.JSONDecodeError)):
            ProviderMaxPrice.parse_raw(input_data)


class Test_ProviderMaxPrice_03_BoundaryBehaviors:
    """Test boundary behaviors for ProviderMaxPrice class."""

    @pytest.mark.parametrize("test_case, field, value", [
        ("zero prompt", "prompt", 0.0),
        ("zero completion", "completion", 0.0),
        ("zero request", "request", 0.0),
        ("zero image", "image", 0.0),
    ])
    def test_accept_minimum_boundary_values(self, test_case, field, value):
        """Verify the model accepts price values at minimum boundary (0.0)."""
        input_data = {field: value}
        model = ProviderMaxPrice(**input_data)
        
        assert getattr(model, field) == value, f"Field {field} should accept minimum value {value}"

    @pytest.mark.parametrize("test_case, field, value", [
        ("slightly negative prompt", "prompt", -0.000001),
        ("slightly negative completion", "completion", -1e-10),
        ("slightly negative request", "request", -0.000000000001),
        ("slightly negative image", "image", -1e-100),
    ])
    def test_reject_below_minimum_boundary(self, test_case, field, value):
        """Verify the model rejects price values below minimum boundary (< 0.0)."""
        input_data = {field: value}
        
        with pytest.raises(ValidationError) as exc_info:
            ProviderMaxPrice(**input_data)
        
        # Verify the error message mentions the field and the constraint
        error_str = str(exc_info.value)
        assert field in error_str, f"Validation error should mention field name '{field}'"
        assert "greater than or equal to 0" in error_str or "ge=" in error_str, \
            "Validation error should mention the constraint (ge=0.0)"

    @pytest.mark.parametrize("test_case, field, value", [
        ("very large prompt", "prompt", 1e100),
        ("very small but positive completion", "completion", 1e-100),
        ("float precision limit request", "request", float('1e-323')),
        ("maximum float image", "image", float('1.7976931348623157e+308')),
    ])
    def test_handle_extreme_float_values(self, test_case, field, value):
        """Verify the model correctly handles extreme but valid float values."""
        input_data = {field: value}
        model = ProviderMaxPrice(**input_data)
        
        assert getattr(model, field) == value, f"Field {field} should accept extreme value {value}"


class Test_ProviderMaxPrice_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ProviderMaxPrice class."""

    @pytest.mark.parametrize("test_case, input_data, expected_errors", [
        (
            "multiple negative values",
            {"prompt": -0.1, "completion": -0.2, "request": -0.3, "image": -0.4},
            ["prompt", "completion", "request", "image"]
        ),
        (
            "mixed negative and invalid types",
            {"prompt": -0.1, "completion": "invalid", "request": -0.3},
            ["prompt", "completion", "request"]
        ),
    ])
    def test_multiple_validation_errors(self, test_case, input_data, expected_errors):
        """Verify the model handles multiple validation errors consistently."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderMaxPrice(**input_data)
        
        # Verify all expected errors are reported
        error_str = str(exc_info.value)
        for field in expected_errors:
            assert field in error_str, f"Validation error should mention field name '{field}'"


## ProviderPreferences Tests

class Test_ProviderPreferences_01_NominalBehaviors:
    """Test nominal behaviors for ProviderPreferences class."""

    def test_init_with_defaults(self):
        """Verify the class initializes with proper default values when no values are provided."""
        model = ProviderPreferences()
        
        # Check default values
        assert model.order is None
        assert model.allow_fallbacks is True
        assert model.require_parameters is False
        assert model.data_collection == "allow"
        assert model.only is None
        assert model.ignore is None
        assert model.quantizations is None
        assert model.sort is None
        assert model.max_price is None

    @pytest.mark.parametrize("test_case, input_values, expected_values", [
        (
            "custom provider routing order",
            {"order": ["provider1", "provider2", "provider3"]},
            {"order": ["provider1", "provider2", "provider3"]}
        ),
        (
            "disable fallbacks",
            {"allow_fallbacks": False},
            {"allow_fallbacks": False}
        ),
        (
            "require parameters",
            {"require_parameters": True},
            {"require_parameters": True}
        ),
        (
            "deny data collection",
            {"data_collection": "deny"},
            {"data_collection": "deny"}
        ),
        (
            "filter by allowed providers",
            {"only": ["provider1", "provider2"]},
            {"only": ["provider1", "provider2"]}
        ),
        (
            "filter by ignored providers",
            {"ignore": ["provider3", "provider4"]},
            {"ignore": ["provider3", "provider4"]}
        ),
        (
            "filter by quantizations",
            {"quantizations": ["4bit", "8bit"]},
            {"quantizations": ["4bit", "8bit"]}
        ),
        (
            "sort by price",
            {"sort": "price"},
            {"sort": "price"}
        ),
        (
            "sort by throughput",
            {"sort": "throughput"},
            {"sort": "throughput"}
        ),
        (
            "sort by latency",
            {"sort": "latency"},
            {"sort": "latency"}
        ),
    ])
    def test_init_with_custom_routing_settings(self, test_case, input_values, expected_values):
        """Verify the class correctly initializes with custom provider routing settings."""
        model = ProviderPreferences(**input_values)
        
        for field, expected in expected_values.items():
            assert getattr(model, field) == expected, f"Field {field} should be {expected} but got {getattr(model, field)}"

    def test_init_with_nested_max_price(self):
        """Verify the class correctly handles nested max_price object."""
        max_price_data = {
            "prompt": 0.5,
            "completion": 0.7,
            "request": 0.1,
            "image": 0.8
        }
        model = ProviderPreferences(max_price=max_price_data)
        
        # Verify max_price is an instance of ProviderMaxPrice
        assert isinstance(model.max_price, ProviderMaxPrice)
        
        # Verify max_price values
        assert model.max_price.prompt == 0.5
        assert model.max_price.completion == 0.7
        assert model.max_price.request == 0.1
        assert model.max_price.image == 0.8

    @pytest.mark.parametrize("test_case, input_values", [
        (
            "complex configuration",
            {
                "order": ["provider1", "provider2"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
                "only": ["provider1"],
                "ignore": ["provider3"],
                "quantizations": ["4bit"],
                "sort": "price",
                "max_price": {"prompt": 0.5, "completion": 0.7}
            }
        ),
        (
            "minimal configuration",
            {
                "order": ["provider1"]
            }
        ),
    ])
    def test_serialization_and_deserialization(self, test_case, input_values):
        """Verify the model correctly serializes to and deserializes from JSON."""
        original_model = ProviderPreferences(**input_values)
        
        # Serialize to JSON
        json_data = original_model.json()
        
        # Deserialize from JSON
        deserialized_model = ProviderPreferences.parse_raw(json_data)
        
        # Verify the deserialized model matches the original
        assert deserialized_model == original_model, f"Deserialized model should match original model"
        
        # If max_price is in the input, verify it's properly deserialized
        if "max_price" in input_values:
            assert isinstance(deserialized_model.max_price, ProviderMaxPrice)
            for field in ["prompt", "completion", "request", "image"]:
                if field in input_values["max_price"]:
                    assert getattr(deserialized_model.max_price, field) == input_values["max_price"][field]


class Test_ProviderPreferences_02_NegativeBehaviors:
    """Test negative behaviors for ProviderPreferences class."""

    @pytest.mark.parametrize("test_case, field, value", [
        ("string instead of list for order", "order", "provider1,provider2"),
        ("integer instead of list for only", "only", 123),
        ("dict instead of list for ignore", "ignore", {"provider": "name"}),
        ("tuple instead of list for quantizations", "quantizations", ("4bit", "8bit")),
    ])
    def test_reject_invalid_list_data_types(self, test_case, field, value):
        """Verify the model rejects invalid data types for list fields."""
        input_data = {field: value}
        
        with pytest.raises(ValidationError) as exc_info:
            ProviderPreferences(**input_data)
        
        # Verify the error message mentions the field and type error
        error_str = str(exc_info.value)
        assert field in error_str, f"Validation error should mention field name '{field}'"
        assert "type" in error_str.lower(), "Validation error should mention type error"

    @pytest.mark.parametrize("test_case, field, value", [
        ("string instead of bool for allow_fallbacks", "allow_fallbacks", "yes"),
        ("integer instead of bool for require_parameters", "require_parameters", 1),
        ("list instead of bool for allow_fallbacks", "allow_fallbacks", [True]),
        ("dict instead of bool for require_parameters", "require_parameters", {"value": True}),
    ])
    def test_reject_invalid_boolean_data_types(self, test_case, field, value):
        """Verify the model rejects invalid data types for boolean fields."""
        input_data = {field: value}
        
        with pytest.raises(ValidationError) as exc_info:
            ProviderPreferences(**input_data)
        
        # Verify the error message mentions the field and type error
        error_str = str(exc_info.value)
        assert field in error_str, f"Validation error should mention field name '{field}'"
        assert "type" in error_str.lower(), "Validation error should mention type error"

    @pytest.mark.parametrize("test_case, field, value", [
        ("invalid value for data_collection", "data_collection", "store"),
        ("invalid value for data_collection", "data_collection", "disallow"),
        ("invalid value for sort", "sort", "cost"),
        ("invalid value for sort", "sort", "speed"),
        ("invalid value for sort", "sort", "performance"),
    ])
    def test_reject_invalid_literal_values(self, test_case, field, value):
        """Verify the model rejects invalid values for literal fields."""
        input_data = {field: value}
        
        with pytest.raises(ValidationError) as exc_info:
            ProviderPreferences(**input_data)
        
        # Verify the error message mentions the field and permitted values
        error_str = str(exc_info.value)
        assert field in error_str, f"Validation error should mention field name '{field}'"
        assert "permitted" in error_str.lower() or "allowed" in error_str.lower() or "valid" in error_str.lower(), \
            "Validation error should mention permitted values"

    @pytest.mark.parametrize("test_case, max_price_data", [
        ("negative prompt in max_price", {"prompt": -0.1}),
        ("negative completion in max_price", {"completion": -0.2}),
        ("invalid type for request in max_price", {"request": "invalid"}),
        ("invalid type for image in max_price", {"image": True}),
    ])
    def test_reject_invalid_nested_max_price(self, test_case, max_price_data):
        """Verify the model rejects invalid nested max_price structure."""
        input_data = {"max_price": max_price_data}
        
        with pytest.raises(ValidationError) as exc_info:
            ProviderPreferences(**input_data)
        
        # Verify the error message mentions max_price and the specific issue
        error_str = str(exc_info.value)
        assert "max_price" in error_str, "Validation error should mention 'max_price'"

    @pytest.mark.parametrize("test_case, input_data", [
        (
            "malformed json string",
            '{"order": ["provider1"], "allow_fallbacks": invalid}'
        ),
        (
            "array instead of object",
            '["provider1", "provider2", "provider3"]'
        ),
    ])
    def test_handle_malformed_input(self, test_case, input_data):
        """Verify the model gracefully handles malformed input data."""
        with pytest.raises((ValidationError, json.JSONDecodeError)):
            ProviderPreferences.parse_raw(input_data)


class Test_ProviderPreferences_03_BoundaryBehaviors:
    """Test boundary behaviors for ProviderPreferences class."""

    @pytest.mark.parametrize("test_case, field, value", [
        ("empty list for order", "order", []),
        ("empty list for only", "only", []),
        ("empty list for ignore", "ignore", []),
        ("empty list for quantizations", "quantizations", []),
    ])
    def test_accept_empty_lists(self, test_case, field, value):
        """Verify the model accepts empty lists for list fields."""
        input_data = {field: value}
        model = ProviderPreferences(**input_data)
        
        assert getattr(model, field) == value, f"Field {field} should accept empty list"

    def test_handle_boundary_conditions_in_nested_max_price(self):
        """Verify the model handles boundary conditions in nested max_price object."""
        # Test with all zero values (minimum allowed)
        max_price_data = {
            "prompt": 0.0,
            "completion": 0.0,
            "request": 0.0,
            "image": 0.0
        }
        model = ProviderPreferences(max_price=max_price_data)
        
        assert model.max_price.prompt == 0.0
        assert model.max_price.completion == 0.0
        assert model.max_price.request == 0.0
        assert model.max_price.image == 0.0
        
        # Test with extreme values
        max_price_data = {
            "prompt": 1e100,
            "completion": 1e-100,
            "request": float('1e-323'),
            "image": float('1.7976931348623157e+308')
        }
        model = ProviderPreferences(max_price=max_price_data)
        
        assert model.max_price.prompt == 1e100
        assert model.max_price.completion == 1e-100
        assert model.max_price.request == float('1e-323')
        assert model.max_price.image == float('1.7976931348623157e+308')


class Test_ProviderPreferences_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ProviderPreferences class."""

    @pytest.mark.parametrize("test_case, input_data, expected_errors", [
        (
            "multiple invalid types",
            {
                "order": "not_a_list",
                "allow_fallbacks": "not_a_bool",
                "data_collection": "invalid"
            },
            ["order", "allow_fallbacks", "data_collection"]
        ),
        (
            "invalid literal and nested max_price",
            {
                "sort": "invalid",
                "max_price": {"prompt": -0.1}
            },
            ["sort", "max_price"]
        ),
    ])
    def test_multiple_validation_errors(self, test_case, input_data, expected_errors):
        """Verify the model raises appropriate validation errors for invalid inputs."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderPreferences(**input_data)
        
        # Verify all expected errors are reported
        error_str = str(exc_info.value)
        for field in expected_errors:
            assert field in error_str, f"Validation error should mention field name '{field}'"

    def test_nested_max_price_validation_errors(self):
        """Verify the model handles validation errors in nested max_price object."""
        input_data = {
            "max_price": {
                "prompt": -0.1,
                "completion": "invalid",
                "request": -0.3,
                "image": True
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ProviderPreferences(**input_data)
        
        # Verify the error message contains all the nested validation errors
        error_str = str(exc_info.value)
        assert "max_price" in error_str, "Validation error should mention 'max_price'"
        assert "prompt" in error_str, "Validation error should mention 'prompt'"
        assert "completion" in error_str, "Validation error should mention 'completion'"
        assert "request" in error_str, "Validation error should mention 'request'"
        assert "image" in error_str, "Validation error should mention 'image'"
    
    @pytest.mark.parametrize("test_case, input_json", [
        (
            "missing closing brace",
            '{"order": ["provider1", "provider2"'
        ),
        (
            "extra commas in array",
            '{"order": ["provider1", , "provider2"]}'
        ),
        (
            "missing quotes around property name",
            '{order: ["provider1", "provider2"]}'
        ),
        (
            "invalid boolean format",
            '{"allow_fallbacks": TRUE}'
        ),
    ])
    def test_json_parsing_errors(self, test_case, input_json):
        """Verify the model handles JSON parsing errors gracefully."""
        with pytest.raises((ValidationError, json.JSONDecodeError)):
            ProviderPreferences.parse_raw(input_json)
            
    def test_error_details_in_nested_max_price(self):
        """Verify that errors in the nested max_price include detailed information."""
        input_data = {
            "max_price": {"prompt": -1.0}  # Invalid negative value
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ProviderPreferences(**input_data)
        
        error_str = str(exc_info.value)
        
        # Check that the error message provides enough context
        assert "max_price" in error_str, "Error should mention the parent field (max_price)"
        assert "prompt" in error_str, "Error should mention the specific field with issue (prompt)"
        assert "-1.0" in error_str or "greater than or equal to 0" in error_str, \
            "Error should mention the invalid value or constraint"
            
    def test_validation_with_multiple_errors_precedence(self):
        """Verify the model prioritizes and reports all validation errors correctly."""
        # Create input with various types of validation errors
        input_data = {
            "order": "not_a_list",  # Type error
            "data_collection": "invalid_value",  # Literal value error
            "max_price": {"prompt": -0.1}  # Nested validation error
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ProviderPreferences(**input_data)
            
        error_str = str(exc_info.value)
        
        # All errors should be reported
        assert "order" in error_str, "Type error in 'order' should be reported"
        assert "data_collection" in error_str, "Literal value error in 'data_collection' should be reported"
        assert "max_price" in error_str, "Nested validation error in 'max_price' should be reported"
