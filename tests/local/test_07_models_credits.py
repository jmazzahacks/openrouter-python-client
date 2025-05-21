import pytest
from pydantic import ValidationError

from openrouter_client.models.credits import CreditsResponse


class Test_CreditsResponse_01_NominalBehaviors:
    """Tests that verify the core expected functionality of CreditsResponse works correctly."""
    
    @pytest.mark.parametrize("credits,used,optional_fields", [
        (100.0, 50.0, {}),
        (100.0, 50.0, {"purchase_credits": 75.0}),
        (100.0, 50.0, {"gifted_credits": 25.0}),
        (100.0, 50.0, {"remaining_free_credits": 50.0}),
        (100.0, 50.0, {"purchase_credits": 75.0, "gifted_credits": 25.0, "remaining_free_credits": 50.0})
    ])
    def test_successful_instantiation(self, credits, used, optional_fields):
        """Test that CreditsResponse can be instantiated with required fields and optional fields."""
        # Arrange
        data = {"credits": credits, "used": used, **optional_fields}
        
        # Act
        response = CreditsResponse(**data)
        
        # Assert
        assert response.credits == credits
        assert response.used == used
        for field, value in optional_fields.items():
            assert getattr(response, field) == value
    
    @pytest.mark.parametrize("field_name,field_value", [
        ("credits", 100.0),
        ("used", 50.0),
        ("purchase_credits", 75.0),
        ("gifted_credits", 25.0),
        ("remaining_free_credits", 50.0)
    ])
    def test_field_value_accessibility(self, field_name, field_value):
        """Test that field values can be accessed as attributes."""
        # Arrange
        data = {
            "credits": 100.0,
            "used": 50.0,
            "purchase_credits": 75.0,
            "gifted_credits": 25.0,
            "remaining_free_credits": 50.0
        }
        response = CreditsResponse(**data)
        
        # Act & Assert
        assert getattr(response, field_name) == field_value


class Test_CreditsResponse_02_NegativeBehaviors:
    """Tests that verify CreditsResponse handles invalid inputs appropriately."""
    
    @pytest.mark.parametrize("missing_field,remaining_data", [
        ("credits", {"used": 50.0}),
        ("used", {"credits": 100.0})
    ])
    def test_missing_required_fields(self, missing_field, remaining_data):
        """Test that ValidationError is raised when required fields are missing."""
        # Act & Assert
        with pytest.raises(ValidationError) as excinfo:
            CreditsResponse(**remaining_data)
        
        # Check that the error message contains the missing field
        assert missing_field in str(excinfo.value)
    
    @pytest.mark.parametrize("field,invalid_value", [
        ("credits", "100"),
        ("credits", None),
        ("used", "50"),
        ("used", None),
        ("purchase_credits", "75"),
        ("gifted_credits", "25"),
        ("remaining_free_credits", "50")
    ])
    def test_type_validation_for_numeric_fields(self, field, invalid_value):
        """Test that type validation is enforced for numeric fields."""
        # Arrange
        data = {
            "credits": 100.0,
            "used": 50.0
        }
        data[field] = invalid_value
        
        # Act & Assert
        # For required fields that are None or non-numeric, should raise ValidationError
        # For optional fields that are non-numeric (but not None), should also raise ValidationError
        if field in ["credits", "used"] or (invalid_value is not None and not isinstance(invalid_value, (int, float))):
            with pytest.raises(ValidationError) as excinfo:
                CreditsResponse(**data)
            assert field in str(excinfo.value)
        else:
            # Optional fields can be None
            response = CreditsResponse(**data)
            assert getattr(response, field) is None


class Test_CreditsResponse_03_BoundaryBehaviors:
    """Tests that verify CreditsResponse handles edge cases correctly."""
    
    @pytest.mark.parametrize("field,zero_value", [
        ("credits", 0.0),
        ("used", 0.0),
        ("purchase_credits", 0.0),
        ("gifted_credits", 0.0),
        ("remaining_free_credits", 0.0)
    ])
    def test_handling_of_zero_values(self, field, zero_value):
        """Test that zero values are handled correctly for all numeric fields."""
        # Arrange
        data = {
            "credits": 100.0,
            "used": 50.0
        }
        data[field] = zero_value
        
        # Act
        response = CreditsResponse(**data)
        
        # Assert
        assert getattr(response, field) == 0.0
    
    @pytest.mark.parametrize("field,negative_value", [
        ("credits", -100.0),
        ("used", -50.0),
        ("purchase_credits", -75.0),
        ("gifted_credits", -25.0),
        ("remaining_free_credits", -50.0)
    ])
    def test_handling_of_negative_values(self, field, negative_value):
        """Test that negative values are handled correctly for all numeric fields."""
        # Arrange
        data = {
            "credits": 100.0,
            "used": 50.0
        }
        data[field] = negative_value
        
        # Act
        response = CreditsResponse(**data)
        
        # Assert
        assert getattr(response, field) == negative_value
    
    @pytest.mark.parametrize("optional_field", [
        "purchase_credits",
        "gifted_credits",
        "remaining_free_credits"
    ])
    def test_optional_field_presence_and_absence(self, optional_field):
        """Test that optional fields can be present or absent."""
        # Case 1: Field is absent
        data_absent = {
            "credits": 100.0,
            "used": 50.0
        }
        response_absent = CreditsResponse(**data_absent)
        assert getattr(response_absent, optional_field) is None
        
        # Case 2: Field is present
        data_present = {
            "credits": 100.0,
            "used": 50.0,
            optional_field: 25.0
        }
        response_present = CreditsResponse(**data_present)
        assert getattr(response_present, optional_field) == 25.0


class Test_CreditsResponse_04_ErrorHandlingBehaviors:
    """Tests that verify CreditsResponse produces appropriate error messages."""
    
    @pytest.mark.parametrize("missing_field", [
        "credits",
        "used"
    ])
    def test_validation_error_messages_for_missing_required_fields(self, missing_field):
        """Test that validation error messages clearly indicate which required field is missing."""
        # Arrange
        data = {
            "credits": 100.0,
            "used": 50.0
        }
        del data[missing_field]
        
        # Act & Assert
        with pytest.raises(ValidationError) as excinfo:
            CreditsResponse(**data)
        
        error_msg = str(excinfo.value)
        assert missing_field in error_msg
        assert "field required" in error_msg.lower()
    
    @pytest.mark.parametrize("field,invalid_value,expected_error_text", [
        ("credits", "100", "should be a valid number"),
        ("used", "50", "should be a valid number"),
        ("purchase_credits", "75", "should be a valid number"),
        ("gifted_credits", "25", "should be a valid number"),
        ("remaining_free_credits", "50", "should be a valid number"),
        ("credits", None, "should be a valid number"),
        ("used", None, "should be a valid number")
    ])
    def test_validation_error_messages_for_type_errors(self, field, invalid_value, expected_error_text):
        """Test that validation error messages clearly indicate type errors."""
        # Arrange
        data = {
            "credits": 100.0,
            "used": 50.0
        }
        data[field] = invalid_value
        
        # Act & Assert
        if field in ["credits", "used"] or (invalid_value is not None and not isinstance(invalid_value, (int, float))):
            with pytest.raises(ValidationError) as excinfo:
                CreditsResponse(**data)
            
            error_msg = str(excinfo.value).lower()
            assert field in error_msg
            assert expected_error_text.lower() in error_msg
        else:
            # Optional fields can be None
            response = CreditsResponse(**data)
            assert getattr(response, field) is None
