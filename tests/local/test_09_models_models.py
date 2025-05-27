import pytest
from pydantic import ValidationError
from typing import List, Optional, Dict, Any

from openrouter_client.models.models import (
    ModelPermission, ModelDataPolicy, ModelPricing, ModelQuantization, 
    ModelContextWindow, ModelProvider, Model, ModelData, ModelsResponse, 
    ModelList, ModelEndpoint, ModelEndpointsRequest, ModelEndpointsResponse
)

@pytest.fixture
def valid_model_permission_data():
    """Returns valid data for a ModelPermission instance."""
    return {
        "id": "permission_123",
        "created": 1680000000,
        "allow_create_engine": True,
        "allow_sampling": True,
        "allow_logprobs": False,
        "allow_search_indices": False,
        "allow_view": True,
        "allow_fine_tuning": False,
        "organization": "org_123",
        "is_blocking": False
    }

@pytest.fixture
def valid_model_data_policy_data():
    """Returns valid data for a ModelDataPolicy instance."""
    return {
        "retention": "7d",
        "logging": True,
        "training": False
    }

@pytest.fixture
def valid_model_pricing_data():
    """Returns valid data for a ModelPricing instance."""
    return {
        "prompt": "0.01",
        "completion": "0.02",
        "request": "0.03",
        "image": "0.04"
    }

@pytest.fixture
def valid_model_quantization_data():
    """Returns valid data for a ModelQuantization instance."""
    return {
        "bits": 8,
        "method": "linear",
        "type": "int"
    }

@pytest.fixture
def valid_model_context_window_data():
    """Returns valid data for a ModelContextWindow instance."""
    return {
        "default": 2048,
        "maximum": 4096
    }

@pytest.fixture
def valid_model_provider_data(valid_model_pricing_data, valid_model_context_window_data, 
                             valid_model_quantization_data, valid_model_data_policy_data):
    """Returns valid data for a ModelProvider instance."""
    return {
        "id": "provider_1",
        "name": "ProviderName",
        "status": "operational",
        "latency": 0.5,
        "throughput": 20.0,
        "pricing": valid_model_pricing_data,
        "context_window": valid_model_context_window_data,
        "quantization": valid_model_quantization_data,
        "data_policy": valid_model_data_policy_data
    }

@pytest.fixture
def valid_model_data(valid_model_permission_data, valid_model_provider_data, valid_model_pricing_data):
    """Returns valid data for a Model instance."""
    return {
        "id": "model_1",
        "created": 1680000000,
        "owned_by": "owner_1",
        "permissions": [valid_model_permission_data],
        "providers": [valid_model_provider_data],
        "pricing": valid_model_pricing_data,
        "context_window": 4096,
        "variants": ["variant1", "variant2"],
        "description": "Model description",
        "features": ["feature1", "feature2"],
        "formats": ["format1", "format2"],
        "tags": ["tag1", "tag2"]
    }

@pytest.fixture
def valid_model_data_api_data(valid_model_pricing_data):
    """Returns valid data for a ModelData instance."""
    return {
        "id": "provider/model-name",
        "name": "Model Name",
        "created": 1680000000,
        "description": "Model description",
        "context_length": 2048,
        "max_completion_tokens": 1024,
        "quantization": "fp16",
        "pricing": valid_model_pricing_data
    }

@pytest.fixture
def valid_models_response_data(valid_model_data_api_data):
    """Returns valid data for a ModelsResponse instance."""
    return {
        "data": [valid_model_data_api_data]
    }

@pytest.fixture
def valid_model_list_data(valid_model_data):
    """Returns valid data for a ModelList instance."""
    return {
        "object": "list",
        "data": [valid_model_data]
    }

@pytest.fixture
def valid_model_endpoint_data():
    """Returns valid data for a ModelEndpoint instance."""
    return {
        "id": "endpoint_1",
        "name": "Endpoint Name",
        "description": "Endpoint description",
        "url": "https://api.example.com/endpoint",
        "method": "GET",
        "parameters": ["param1", "param2"]
    }

@pytest.fixture
def valid_model_endpoints_request_data():
    """Returns valid data for a ModelEndpointsRequest instance."""
    return {
        "author": "author_1",
        "slug": "slug_1"
    }

@pytest.fixture
def valid_model_endpoints_response_data(valid_model_endpoint_data):
    """Returns valid data for a ModelEndpointsResponse instance."""
    return {
        "object": "list",
        "data": [valid_model_endpoint_data]
    }


class Test_ModelPermission_01_NominalBehaviors:
    """Tests for nominal behaviors of the ModelPermission class."""
    
    @pytest.mark.parametrize("field_name,expected_value", [
        ("id", "permission_123"),
        ("object", "model_permission"),  # Test default value
        ("created", 1680000000),
        ("allow_create_engine", True),
        ("allow_sampling", True),
        ("allow_logprobs", False),
        ("allow_search_indices", False),
        ("allow_view", True),
        ("allow_fine_tuning", False),
        ("organization", "org_123"),
        ("is_blocking", False)
    ])
    def test_field_values(self, valid_model_permission_data, field_name, expected_value):
        """Test that field values are properly set during instantiation."""
        model = ModelPermission(**valid_model_permission_data)
        assert getattr(model, field_name) == expected_value


class Test_ModelPermission_02_NegativeBehaviors:
    """Tests for negative behaviors of the ModelPermission class."""
    
    @pytest.mark.parametrize("field_to_remove", [
        "id", "created", "allow_create_engine", "allow_sampling", "allow_logprobs", 
        "allow_search_indices", "allow_view", "allow_fine_tuning", "organization", "is_blocking"
    ])
    def test_missing_required_field(self, valid_model_permission_data, field_to_remove):
        """Test that instantiation fails when a required field is missing."""
        data = valid_model_permission_data.copy()
        del data[field_to_remove]
        
        with pytest.raises(ValidationError) as exc_info:
            ModelPermission(**data)
            
        errors = exc_info.value.errors()
        assert any(error["loc"] == (field_to_remove,) for error in errors)
    
    @pytest.mark.parametrize("field_name,invalid_value,expected_error_type", [
        ("id", 123, "string_type"),
        ("created", "not_an_int", "int_parsing"),
        ("allow_create_engine", "not_a_bool", "bool_parsing"),
        ("organization", 123, "string_type")
    ])
    def test_invalid_field_type(self, valid_model_permission_data, field_name, invalid_value, expected_error_type):
        """Test that instantiation fails when a field has an invalid type."""
        data = valid_model_permission_data.copy()
        data[field_name] = invalid_value
        
        with pytest.raises(ValidationError) as exc_info:
            ModelPermission(**data)
            
        errors = exc_info.value.errors()
        assert any(error["type"].startswith(expected_error_type) for error in errors)


class Test_ModelPermission_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ModelPermission class."""
    
    def test_validation_error_messages(self, valid_model_permission_data):
        """Test that validation errors provide clear error messages."""
        # Test with missing required field
        data = valid_model_permission_data.copy()
        del data["id"]
        
        with pytest.raises(ValidationError) as exc_info:
            ModelPermission(**data)
            
        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any(error["loc"] == ("id",) for error in errors)
        
        # Test that error message contains helpful information
        error_message = str(exc_info.value)
        assert "id" in error_message


class Test_ModelDataPolicy_01_NominalBehaviors:
    """Tests for nominal behaviors of the ModelDataPolicy class."""
    
    def test_instantiation_with_none_fields(self):
        """Test instantiation with all fields as None - vital since all fields are optional."""
        model = ModelDataPolicy()
        assert model.retention is None
        assert model.logging is None
        assert model.training is None


class Test_ModelDataPolicy_02_NegativeBehaviors:
    """Tests for negative behaviors of the ModelDataPolicy class."""
    
    @pytest.mark.parametrize("field_name,invalid_value,expected_error_type", [
        ("retention", 123, "string_type"),
        ("logging", "not_a_bool", "bool_parsing"),
        ("training", "not_a_bool", "bool_parsing")
    ])
    def test_invalid_field_type(self, field_name, invalid_value, expected_error_type):
        """Test that instantiation fails when a field has an invalid type."""
        data = {field_name: invalid_value}
        
        with pytest.raises(ValidationError) as exc_info:
            ModelDataPolicy(**data)
            
        errors = exc_info.value.errors()
        assert any(error["type"].startswith(expected_error_type) for error in errors)


class Test_ModelPricing_01_NominalBehaviors:
    """Tests for nominal behaviors of the ModelPricing class."""
    
    def test_instantiation_with_none_fields(self):
        """Test instantiation with all fields as None - vital for API flexibility."""
        model = ModelPricing()
        assert model.prompt is None
        assert model.completion is None
        assert model.request is None
        assert model.image is None


class Test_ModelPricing_02_NegativeBehaviors:
    """Tests for negative behaviors of the ModelPricing class."""
    
    @pytest.mark.parametrize("field_name,invalid_value,expected_error_type", [
        ("prompt", 0.01, "string_type"),  # Number instead of string
        ("completion", 0.02, "string_type"),
        ("request", 0.03, "string_type"),
        ("image", 0.04, "string_type")
    ])
    def test_invalid_field_type(self, field_name, invalid_value, expected_error_type):
        """Test that instantiation fails when a field has an invalid type."""
        data = {field_name: invalid_value}
        
        with pytest.raises(ValidationError) as exc_info:
            ModelPricing(**data)
            
        errors = exc_info.value.errors()
        assert any(error["type"].startswith(expected_error_type) for error in errors)


class Test_ModelQuantization_01_NominalBehaviors:
    """Tests for nominal behaviors of the ModelQuantization class."""
    
    def test_instantiation_with_none_fields(self):
        """Test instantiation with all fields as None - vital since all fields are optional."""
        model = ModelQuantization()
        assert model.bits is None
        assert model.method is None
        assert model.type is None


class Test_ModelQuantization_02_NegativeBehaviors:
    """Tests for negative behaviors of the ModelQuantization class."""
    
    @pytest.mark.parametrize("field_name,invalid_value,expected_error_type", [
        ("bits", "not_an_int", "int_parsing"),
        ("method", 123, "string_type"),
        ("type", 123, "string_type")
    ])
    def test_invalid_field_type(self, field_name, invalid_value, expected_error_type):
        """Test that instantiation fails when a field has an invalid type."""
        data = {field_name: invalid_value}
        
        with pytest.raises(ValidationError) as exc_info:
            ModelQuantization(**data)
            
        errors = exc_info.value.errors()
        assert any(error["type"].startswith(expected_error_type) for error in errors)


class Test_ModelContextWindow_01_NominalBehaviors:
    """Tests for nominal behaviors of the ModelContextWindow class."""
    
    def test_instantiation_with_required_field(self):
        """Test instantiation with only the required default field - vital for API contract."""
        model = ModelContextWindow(default=2048)
        assert model.default == 2048
        assert model.maximum is None


class Test_ModelContextWindow_02_NegativeBehaviors:
    """Tests for negative behaviors of the ModelContextWindow class."""
    
    def test_missing_required_field(self):
        """Test that instantiation fails when the required default field is missing."""
        with pytest.raises(ValidationError) as exc_info:
            ModelContextWindow()
            
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("default",) for error in errors)
    
    @pytest.mark.parametrize("field_name,invalid_value,expected_error_type", [
        ("default", "not_an_int", "int_parsing"),
        ("maximum", "not_an_int", "int_parsing")
    ])
    def test_invalid_field_type(self, field_name, invalid_value, expected_error_type):
        """Test that instantiation fails when a field has an invalid type."""
        data = {"default": 2048} if field_name != "default" else {}
        data[field_name] = invalid_value
        
        with pytest.raises(ValidationError) as exc_info:
            ModelContextWindow(**data)
            
        errors = exc_info.value.errors()
        assert any(error["type"].startswith(expected_error_type) for error in errors)


class Test_ModelProvider_01_NominalBehaviors:
    """Tests for nominal behaviors of the ModelProvider class."""
    
    def test_instantiation_with_required_fields_only(self):
        """Test instantiation with only the required fields - vital for minimum API contract."""
        model = ModelProvider(id="provider_1", name="ProviderName")
        assert model.id == "provider_1"
        assert model.name == "ProviderName"
        assert model.status is None
        assert model.pricing is None
    
    @pytest.mark.parametrize("nested_field,nested_type", [
        ("pricing", ModelPricing),
        ("context_window", ModelContextWindow),
        ("quantization", ModelQuantization),
        ("data_policy", ModelDataPolicy)
    ])
    def test_nested_objects(self, valid_model_provider_data, nested_field, nested_type):
        """Test that nested objects are properly instantiated - vital for complex object relationships."""
        model = ModelProvider(**valid_model_provider_data)
        nested_object = getattr(model, nested_field)
        assert isinstance(nested_object, nested_type)


class Test_ModelProvider_02_NegativeBehaviors:
    """Tests for negative behaviors of the ModelProvider class."""
    
    @pytest.mark.parametrize("field_to_remove", [
        "id", "name"
    ])
    def test_missing_required_field(self, valid_model_provider_data, field_to_remove):
        """Test that instantiation fails when a required field is missing."""
        data = valid_model_provider_data.copy()
        del data[field_to_remove]
        
        with pytest.raises(ValidationError) as exc_info:
            ModelProvider(**data)
            
        errors = exc_info.value.errors()
        assert any(error["loc"] == (field_to_remove,) for error in errors)
    
    @pytest.mark.parametrize("nested_field,invalid_value", [
        ("pricing", {"prompt": 0.01}),  # Number instead of string
        ("context_window", {"default": "not_an_int"}),
        ("quantization", {"bits": "not_an_int"}),
        ("data_policy", {"logging": "not_a_bool"})
    ])
    def test_invalid_nested_object(self, valid_model_provider_data, nested_field, invalid_value):
        """Test that instantiation fails when a nested object has invalid data."""
        data = valid_model_provider_data.copy()
        data[nested_field] = invalid_value
        
        with pytest.raises(ValidationError) as exc_info:
            ModelProvider(**data)
            
        errors = exc_info.value.errors()
        assert any(error["loc"][0] == nested_field for error in errors)


class Test_Model_01_NominalBehaviors:
    """Tests for nominal behaviors of the Model class."""
    
    def test_instantiation_with_required_fields(self, valid_model_permission_data):
        """Test instantiation with only the required fields - vital for API contract."""
        model = Model(
            id="model_1",
            created=1680000000,
            owned_by="owner_1",
            permissions=[ModelPermission(**valid_model_permission_data)]
        )
        assert model.id == "model_1"
        assert model.object == "model"  # Default value
        assert model.created == 1680000000
        assert model.owned_by == "owner_1"
        assert len(model.permissions) == 1
    
    @pytest.mark.parametrize("list_field", [
        "permissions", "providers", "variants", "features", "formats", "tags"
    ])
    def test_list_fields(self, valid_model_data, list_field):
        """Test that list fields are properly handled - vital for complex data structures."""
        model = Model(**valid_model_data)
        list_value = getattr(model, list_field)
        if list_value is not None:  # Some lists might be optional and None
            assert isinstance(list_value, list)


class Test_Model_02_NegativeBehaviors:
    """Tests for negative behaviors of the Model class."""
    
    @pytest.mark.parametrize("field_to_remove", [
        "id", "created", "owned_by", "permissions"
    ])
    def test_missing_required_field(self, valid_model_data, field_to_remove):
        """Test that instantiation fails when a required field is missing."""
        data = valid_model_data.copy()
        del data[field_to_remove]
        
        with pytest.raises(ValidationError) as exc_info:
            Model(**data)
            
        errors = exc_info.value.errors()
        assert any(error["loc"] == (field_to_remove,) for error in errors)
    
    @pytest.mark.parametrize("field_name,invalid_value,expected_error_type", [
        ("id", 123, "string_type"),
        ("created", "not_an_int", "int_parsing"),
        ("owned_by", 123, "string_type"),
        ("permissions", "not_a_list", "list_type")
    ])
    def test_invalid_field_type(self, valid_model_data, field_name, invalid_value, expected_error_type):
        """Test that instantiation fails when a field has an invalid type."""
        data = valid_model_data.copy()
        data[field_name] = invalid_value
        
        with pytest.raises(ValidationError) as exc_info:
            Model(**data)
            
        errors = exc_info.value.errors()
        assert any(error["type"].startswith(expected_error_type) for error in errors)


class Test_ModelData_01_NominalBehaviors:
    """Tests for nominal behaviors of the ModelData class."""
    
    def test_instantiation_with_required_fields(self, valid_model_pricing_data):
        """Test instantiation with only the required fields - vital for API contract."""
        model = ModelData(
            id="provider/model-name",
            name="Model Name",
            created=1680000000,
            context_length=2048,
            quantization="fp16",
            pricing=ModelPricing(**valid_model_pricing_data)
        )
        assert model.id == "provider/model-name"
        assert model.name == "Model Name"
        assert model.created == 1680000000
        assert model.context_length == 2048
        assert model.quantization == "fp16"
        assert isinstance(model.pricing, ModelPricing)
    
    def test_nested_pricing_object(self, valid_model_data_api_data):
        """Test that the nested pricing object is properly instantiated - vital for complex object relationships."""
        model = ModelData(**valid_model_data_api_data)
        assert isinstance(model.pricing, ModelPricing)


class Test_ModelData_02_NegativeBehaviors:
    """Tests for negative behaviors of the ModelData class."""
    
    @pytest.mark.parametrize("field_to_remove", [
        "id", "name", "created", "context_length", "pricing"
    ])
    def test_missing_required_field(self, valid_model_data_api_data, field_to_remove):
        """Test that instantiation fails when a required field is missing."""
        data = valid_model_data_api_data.copy()
        del data[field_to_remove]
        
        with pytest.raises(ValidationError) as exc_info:
            ModelData(**data)
            
        errors = exc_info.value.errors()
        assert any(error["loc"] == (field_to_remove,) for error in errors)
    
    @pytest.mark.parametrize("field_name,invalid_value,expected_error_type", [
        ("id", 123, "string_type"),
        ("name", 123, "string_type"),
        ("created", "not_an_int", "int_parsing"),
        ("context_length", "not_an_int", "int_parsing"),
        ("quantization", 123, "string_type"),
        ("pricing", "not_an_object", "model_type")
    ])
    def test_invalid_field_type(self, valid_model_data_api_data, field_name, invalid_value, expected_error_type):
        """Test that instantiation fails when a field has an invalid type."""
        data = valid_model_data_api_data.copy()
        data[field_name] = invalid_value
        
        with pytest.raises(ValidationError) as exc_info:
            ModelData(**data)
            
        errors = exc_info.value.errors()
        assert any(error["type"].startswith(expected_error_type) for error in errors)


class Test_ModelsResponse_01_NominalBehaviors:
    """Tests for nominal behaviors of the ModelsResponse class."""
    
    def test_instantiation_with_valid_data_list(self, valid_models_response_data, valid_model_data_api_data):
        """Test instantiation with a valid list of ModelData objects - vital for API response parsing."""
        model = ModelsResponse(**valid_models_response_data)
        assert isinstance(model.data, list)
        assert len(model.data) == 1
        assert isinstance(model.data[0], ModelData)


class Test_ModelsResponse_02_NegativeBehaviors:
    """Tests for negative behaviors of the ModelsResponse class."""
    
    def test_missing_required_data_field(self):
        """Test that instantiation fails when the required data field is missing."""
        with pytest.raises(ValidationError) as exc_info:
            ModelsResponse()
            
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("data",) for error in errors)
    
    @pytest.mark.parametrize("invalid_value,expected_error_type", [
        ("not_a_list", "list_type"),
        ([{"id": 123}], "string_type")  # Invalid ModelData object
    ])
    def test_invalid_data_field(self, invalid_value, expected_error_type):
        """Test that instantiation fails when the data field has an invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            ModelsResponse(data=invalid_value)
            
        errors = exc_info.value.errors()
        assert any(error["type"].startswith(expected_error_type) for error in errors)


class Test_ModelsResponse_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ModelsResponse class."""
    
    def test_empty_data_list(self):
        """Test model behavior with an empty data list - vital for handling API responses with no results."""
        model = ModelsResponse(data=[])
        assert isinstance(model.data, list)
        assert len(model.data) == 0


class Test_ModelList_01_NominalBehaviors:
    """Tests for nominal behaviors of the ModelList class."""
    
    def test_instantiation_with_valid_data_list(self, valid_model_list_data, valid_model_data):
        """Test instantiation with a valid list of Model objects - vital for API response parsing."""
        model = ModelList(**valid_model_list_data)
        assert model.object == "list"
        assert isinstance(model.data, list)
        assert len(model.data) == 1
        assert isinstance(model.data[0], Model)


class Test_ModelList_02_NegativeBehaviors:
    """Tests for negative behaviors of the ModelList class."""
    
    def test_missing_required_data_field(self):
        """Test that instantiation fails when the required data field is missing."""
        with pytest.raises(ValidationError) as exc_info:
            ModelList(object="list")
            
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("data",) for error in errors)
    
    @pytest.mark.parametrize("invalid_value,expected_error_type", [
        ("not_a_list", "list_type"),
        ([{"id": 123}], "string_type")  # Invalid Model object
    ])
    def test_invalid_data_field(self, invalid_value, expected_error_type):
        """Test that instantiation fails when the data field has an invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            ModelList(object="list", data=invalid_value)
            
        errors = exc_info.value.errors()
        assert any(error["type"].startswith(expected_error_type) for error in errors)


class Test_ModelList_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ModelList class."""
    
    def test_empty_data_list(self):
        """Test model behavior with an empty data list - vital for handling API responses with no results."""
        model = ModelList(object="list", data=[])
        assert model.object == "list"
        assert isinstance(model.data, list)
        assert len(model.data) == 0


class Test_ModelEndpoint_01_NominalBehaviors:
    """Tests for nominal behaviors of the ModelEndpoint class."""
    
    def test_instantiation_with_required_fields(self):
        """Test instantiation with only the required fields - vital for API contract."""
        model = ModelEndpoint(
            id="endpoint_1",
            name="Endpoint Name",
            url="https://api.example.com/endpoint",
            method="GET"
        )
        assert model.id == "endpoint_1"
        assert model.name == "Endpoint Name"
        assert model.url == "https://api.example.com/endpoint"
        assert model.method == "GET"
        assert model.description is None
        assert model.parameters is None


class Test_ModelEndpoint_02_NegativeBehaviors:
    """Tests for negative behaviors of the ModelEndpoint class."""
    
    @pytest.mark.parametrize("field_to_remove", [
        "id", "name", "url", "method"
    ])
    def test_missing_required_field(self, valid_model_endpoint_data, field_to_remove):
        """Test that instantiation fails when a required field is missing."""
        data = valid_model_endpoint_data.copy()
        del data[field_to_remove]
        
        with pytest.raises(ValidationError) as exc_info:
            ModelEndpoint(**data)
            
        errors = exc_info.value.errors()
        assert any(error["loc"] == (field_to_remove,) for error in errors)
    
    @pytest.mark.parametrize("field_name,invalid_value,expected_error_type", [
        ("id", 123, "string_type"),
        ("name", 123, "string_type"),
        ("url", 123, "string_type"),
        ("method", 123, "string_type"),
        ("parameters", "not_a_list", "list_type")
    ])
    def test_invalid_field_type(self, valid_model_endpoint_data, field_name, invalid_value, expected_error_type):
        """Test that instantiation fails when a field has an invalid type."""
        data = valid_model_endpoint_data.copy()
        data[field_name] = invalid_value
        
        with pytest.raises(ValidationError) as exc_info:
            ModelEndpoint(**data)
            
        errors = exc_info.value.errors()
        assert any(error["type"].startswith(expected_error_type) for error in errors)


class Test_ModelEndpointsRequest_01_NominalBehaviors:
    """Tests for nominal behaviors of the ModelEndpointsRequest class."""
    
    def test_instantiation_with_required_fields(self, valid_model_endpoints_request_data):
        """Test instantiation with the required fields - vital for API request formatting."""
        model = ModelEndpointsRequest(**valid_model_endpoints_request_data)
        assert model.author == "author_1"
        assert model.slug == "slug_1"


class Test_ModelEndpointsRequest_02_NegativeBehaviors:
    """Tests for negative behaviors of the ModelEndpointsRequest class."""
    
    @pytest.mark.parametrize("field_to_remove", [
        "author", "slug"
    ])
    def test_missing_required_field(self, valid_model_endpoints_request_data, field_to_remove):
        """Test that instantiation fails when a required field is missing."""
        data = valid_model_endpoints_request_data.copy()
        del data[field_to_remove]
        
        with pytest.raises(ValidationError) as exc_info:
            ModelEndpointsRequest(**data)
            
        errors = exc_info.value.errors()
        assert any(error["loc"] == (field_to_remove,) for error in errors)
    
    @pytest.mark.parametrize("field_name,invalid_value,expected_error_type", [
        ("author", 123, "string_type"),
        ("slug", 123, "string_type")
    ])
    def test_invalid_field_type(self, valid_model_endpoints_request_data, field_name, invalid_value, expected_error_type):
        """Test that instantiation fails when a field has an invalid type."""
        data = valid_model_endpoints_request_data.copy()
        data[field_name] = invalid_value
        
        with pytest.raises(ValidationError) as exc_info:
            ModelEndpointsRequest(**data)
            
        errors = exc_info.value.errors()
        assert any(error["type"].startswith(expected_error_type) for error in errors)


class Test_ModelEndpointsResponse_01_NominalBehaviors:
    """Tests for nominal behaviors of the ModelEndpointsResponse class."""
    
    def test_instantiation_with_valid_data_list(self, valid_model_endpoints_response_data, valid_model_endpoint_data):
        """Test instantiation with a valid list of ModelEndpoint objects - vital for API response parsing."""
        model = ModelEndpointsResponse(**valid_model_endpoints_response_data)
        assert model.object == "list"
        assert isinstance(model.data, list)
        assert len(model.data) == 1
        assert isinstance(model.data[0], ModelEndpoint)


class Test_ModelEndpointsResponse_02_NegativeBehaviors:
    """Tests for negative behaviors of the ModelEndpointsResponse class."""
    
    def test_missing_required_data_field(self):
        """Test that instantiation fails when the required data field is missing."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEndpointsResponse(object="list")
            
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("data",) for error in errors)
    
    @pytest.mark.parametrize("invalid_value,expected_error_type", [
        ("not_a_list", "list_type"),
        ([{"id": 123}], "string_type")  # Invalid ModelEndpoint object
    ])
    def test_invalid_data_field(self, invalid_value, expected_error_type):
        """Test that instantiation fails when the data field has an invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEndpointsResponse(object="list", data=invalid_value)
            
        errors = exc_info.value.errors()
        assert any(error["type"].startswith(expected_error_type) for error in errors)


class Test_ModelEndpointsResponse_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the ModelEndpointsResponse class."""
    
    def test_empty_data_list(self):
        """Test model behavior with an empty data list - vital for handling API responses with no results."""
        model = ModelEndpointsResponse(object="list", data=[])
        assert model.object == "list"
        assert isinstance(model.data, list)
        assert len(model.data) == 0
