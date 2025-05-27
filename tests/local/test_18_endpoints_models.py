import pytest
from unittest.mock import Mock, patch
import json

from openrouter_client.endpoints.models import ModelsEndpoint
from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager
from openrouter_client.models.models import ModelData, ModelPricing, ModelsResponse, ModelEndpointsResponse
from openrouter_client.exceptions import APIError
from pydantic import ValidationError


@pytest.fixture
def mock_auth_manager():
    """Fixture providing a properly configured AuthManager mock."""
    return Mock(spec=AuthManager)


@pytest.fixture
def mock_http_manager():
    """Fixture providing a properly configured HTTPManager mock."""
    return Mock(spec=HTTPManager)


@pytest.fixture
def mock_response():
    """Fixture providing a configurable mock HTTP response."""
    response = Mock()
    response.json = Mock()
    response.status_code = 200
    return response


@pytest.fixture
def models_endpoint_base(mock_auth_manager, mock_http_manager):
    """Fixture providing a ModelsEndpoint instance with base mocking."""
    # Configure auth_manager to return proper headers by default
    mock_auth_manager.get_auth_headers.return_value = {
        "Authorization": "Bearer test-token"
    }
    return ModelsEndpoint(mock_auth_manager, mock_http_manager)


class Test_ModelsEndpoint_Init_01_NominalBehaviors:
    """Test nominal behaviors for ModelsEndpoint.__init__ method."""

    def test_successful_initialization_with_valid_auth_and_http_managers(self):
        """Test successful initialization with valid AuthManager and HTTPManager instances."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Act
        with patch.object(ModelsEndpoint, '_get_headers'), patch.object(ModelsEndpoint, '_get_endpoint_url'):
            endpoint = ModelsEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint.auth_manager is auth_manager
        assert endpoint.http_manager is http_manager
        assert hasattr(endpoint, 'logger')


class Test_ModelsEndpoint_Init_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ModelsEndpoint.__init__ method."""

    @pytest.mark.parametrize("auth_manager,http_manager", [
        (None, Mock(spec=HTTPManager)),
        (Mock(spec=AuthManager), None),
        (None, None),
        ("invalid_auth", Mock(spec=HTTPManager)),
        (Mock(spec=AuthManager), "invalid_http"),
    ])
    def test_initialization_fails_with_invalid_dependencies(self, auth_manager, http_manager):
        """Test that initialization fails appropriately with None or invalid dependencies."""
        # Arrange, Act, Assert
        with pytest.raises(ValidationError):
            ModelsEndpoint(auth_manager, http_manager)


class Test_ModelsEndpoint_Init_05_StateTransitionBehaviors:
    """Test state transition behaviors for ModelsEndpoint.__init__ method."""

    def test_object_transitions_from_uninitialized_to_operational_state(self):
        """Test that object properly transitions to fully operational state during initialization."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Act
        with patch.object(ModelsEndpoint, '_get_headers'), patch.object(ModelsEndpoint, '_get_endpoint_url'):
            endpoint = ModelsEndpoint(auth_manager, http_manager)
        
        # Assert - verify all required attributes for operation are present
        assert hasattr(endpoint, 'auth_manager')
        assert hasattr(endpoint, 'http_manager')
        assert hasattr(endpoint, 'endpoint_path')
        assert hasattr(endpoint, 'logger')
        assert endpoint.endpoint_path == "models"


class Test_ModelsEndpoint_List_01_NominalBehaviors:
    """Test nominal behaviors for ModelsEndpoint.list method."""

    def test_successful_retrieval_with_details_false_returns_model_ids(self, models_endpoint_base, mock_response):
        """Test successful model listing with details=False returning list of model IDs."""
        # Arrange
        response_data = {
            "data": [
                {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus"},
                {"id": "openai/gpt-4", "name": "GPT-4"}
            ]
        }
        mock_response.json.return_value = response_data
        models_endpoint_base.http_manager.get.return_value = mock_response
        
        # Act
        result = models_endpoint_base.list(details=False)
        
        # Assert
        assert result == ["anthropic/claude-3-opus", "openai/gpt-4"]
        models_endpoint_base.http_manager.get.assert_called_once()

    def test_successful_retrieval_with_details_true_returns_models_response(self, models_endpoint_base, mock_response):
        """Test successful model listing with details=True returning complete ModelsResponse."""
        # Arrange
        response_data = {
            "object": "list",
            "data": [
                {
                    "id": "anthropic/claude-3-opus",
                    "name": "Claude 3 Opus",
                    "context_length": 200000,
                    "pricing": {"prompt": "0.000015", "completion": "0.000075"}
                }
            ]
        }
        mock_response.json.return_value = response_data
        models_endpoint_base.http_manager.get.return_value = mock_response
        
        with patch.object(ModelsResponse, 'model_validate') as mock_validate:
            mock_models_response = Mock(spec=ModelsResponse)
            mock_validate.return_value = mock_models_response
            
            # Act
            result = models_endpoint_base.list(details=True)
            
            # Assert
            mock_validate.assert_called_once_with(response_data)
            assert result is mock_models_response


class Test_ModelsEndpoint_List_02_NegativeBehaviors:
    """Test negative behaviors for ModelsEndpoint.list method."""

    def test_api_response_missing_data_field_when_details_false(self, models_endpoint_base, mock_response):
        """Test behavior when API response missing expected data field with details=False."""
        # Arrange
        response_data = {"object": "list"}  # Missing 'data' field
        mock_response.json.return_value = response_data
        models_endpoint_base.http_manager.get.return_value = mock_response
        
        with patch.object(ModelsResponse, 'model_validate') as mock_validate:
            mock_models_response = Mock(spec=ModelsResponse)
            mock_validate.return_value = mock_models_response
            
            # Act
            result = models_endpoint_base.list(details=False)
            
            # Assert - should fall back to ModelsResponse validation
            mock_validate.assert_called_once_with(response_data)
            assert result is mock_models_response  # Returns the ModelsResponse object

    @pytest.mark.parametrize("invalid_data,expected_result", [
        ([{"name": "model1"}], [{"name": "model1"}]),  # Missing 'id' key - returns item as-is
        (["string_item"], ["string_item"]),  # Non-dict item - returns as-is
        ([None], [None]),  # None item - returns as-is
        ([{"id": None}], [None]),  # None id value - extracts None
        ([{"id": "valid-id"}, {"name": "no-id"}, {"id": "another-valid"}], ["valid-id", {"name": "no-id"}, "another-valid"]),  # Mixed valid/invalid
    ])
    def test_api_response_data_contains_invalid_items_when_extracting_ids(self, models_endpoint_base, mock_response, invalid_data, expected_result):
        """Test handling of API response data field containing non-dictionary items when extracting IDs."""
        # Arrange
        mock_response.json.return_value = {"data": invalid_data}
        models_endpoint_base.http_manager.get.return_value = mock_response
        
        # Act
        result = models_endpoint_base.list(details=False)
        
        # Assert - based on implementation, it returns items as-is when not a dict with 'id'
        assert result == expected_result


class Test_ModelsEndpoint_List_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ModelsEndpoint.list method."""

    @pytest.mark.parametrize("exception_type", [
        ConnectionError,
        TimeoutError,
        Exception,
    ])
    def test_network_connectivity_failures_during_http_request(self, models_endpoint_base, exception_type):
        """Test proper handling of network connectivity failures during HTTP request."""
        # Arrange
        models_endpoint_base.http_manager.get.side_effect = exception_type("Network error")
        
        # Act & Assert
        with pytest.raises(exception_type):
            models_endpoint_base.list()

    @pytest.mark.parametrize("invalid_json", [
        "invalid json",
        "",
    ])
    def test_json_parsing_failures_from_malformed_responses(self, models_endpoint_base, mock_response, invalid_json):
        """Test handling of JSON parsing failures from malformed API responses."""
        # Arrange
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", invalid_json, 0)
        models_endpoint_base.http_manager.get.return_value = mock_response
        
        # Act & Assert
        with pytest.raises(json.JSONDecodeError):
            models_endpoint_base.list()


class Test_ModelsEndpoint_Get_01_NominalBehaviors:
    """Test nominal behaviors for ModelsEndpoint.get method."""

    def test_successful_retrieval_and_model_data_construction_with_valid_model_id(self, models_endpoint_base, mock_response):
        """Test successful model retrieval and ModelData construction with valid model_id."""
        # Arrange
        model_data = {
            "id": "anthropic/claude-3-opus",
            "name": "Claude 3 Opus",
            "context_length": 200000,
            "pricing": {"prompt": "0.000015", "completion": "0.000075"}
        }
        mock_response.json.return_value = model_data
        models_endpoint_base.http_manager.get.return_value = mock_response
        
        with patch.object(ModelData, 'model_validate') as mock_validate:
            mock_model = Mock(spec=ModelData)
            mock_validate.return_value = mock_model
            
            # Act
            result = models_endpoint_base.get("anthropic/claude-3-opus")
            
            # Assert
            mock_validate.assert_called_once_with(model_data)
            assert result is mock_model


class Test_ModelsEndpoint_Get_02_NegativeBehaviors:
    """Test negative behaviors for ModelsEndpoint.get method."""

    def test_non_existent_model_id_resulting_in_http_404(self, models_endpoint_base):
        """Test handling of non-existent model_id resulting in HTTP 404 error."""
        # Arrange
        # Configure HTTP manager to raise 404 error
        models_endpoint_base.http_manager.get.side_effect = APIError(
            "Model not found",
            status_code=404
        )
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            models_endpoint_base.get("non-existent-model")
        
        assert "Model not found" in str(exc_info.value)


class Test_ModelsEndpoint_Get_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ModelsEndpoint.get method."""

    @pytest.mark.parametrize("exception_type,error_message", [
        (ConnectionError, "Network connection failed"),
        (TimeoutError, "Request timed out"),
        (APIError, "Authentication failed"),
    ])
    def test_network_failures_and_authentication_errors_during_model_retrieval(self, models_endpoint_base, exception_type, error_message):
        """Test handling of network failures and authentication errors during model retrieval."""
        # Arrange
        # Configure HTTP manager to raise the exception
        if isinstance(exception_type, APIError):
            models_endpoint_base.http_manager.get.side_effect = APIError(error_message, status_code=401)
        else:
            models_endpoint_base.http_manager.get.side_effect = exception_type(error_message)
        
        # Act & Assert
        with pytest.raises(exception_type) as exc_info:
            models_endpoint_base.get("test-model")
        
        assert error_message in str(exc_info.value)

    @pytest.mark.parametrize("error_type", [
        json.JSONDecodeError("Invalid JSON", "", 0),
        ValueError("Validation failed"),
        KeyError("Required field missing"),
    ])
    def test_json_parsing_and_model_data_validation_failures(self, models_endpoint_base, mock_response, error_type):
        """Test handling of JSON parsing and ModelData validation failures."""
        # Arrange
        if isinstance(error_type, json.JSONDecodeError):
            mock_response.json.side_effect = error_type
            models_endpoint_base.http_manager.get.return_value = mock_response
            
            # Act & Assert
            with pytest.raises(json.JSONDecodeError):
                models_endpoint_base.get("test-model")
        else:
            mock_response.json.return_value = {"id": "test-model"}
            models_endpoint_base.http_manager.get.return_value = mock_response
            
            with patch.object(ModelData, 'model_validate', side_effect=error_type):
                # Act & Assert
                with pytest.raises(type(error_type)):
                    models_endpoint_base.get("test-model")


class Test_ModelsEndpoint_GetContextLength_01_NominalBehaviors:
    """Test nominal behaviors for ModelsEndpoint.get_context_length method."""

    def test_successful_extraction_of_context_length_integer_from_model_data(self, models_endpoint_base, mock_response):
        """Test successful extraction of context_length integer from model data."""
        # Arrange
        model_data = {
            "id": "test-model",
            "name": "Test Model",
            "context_length": 200000,
            "pricing": {"prompt": "0.000015", "completion": "0.000075"}
        }
        mock_response.json.return_value = model_data
        models_endpoint_base.http_manager.get.return_value = mock_response
        
        with patch.object(ModelData, 'model_validate') as mock_validate:
            mock_model_data = Mock(spec=ModelData)
            mock_model_data.context_length = 200000
            mock_validate.return_value = mock_model_data
            
            # Act
            result = models_endpoint_base.get_context_length("test-model")
            
            # Assert
            assert result == 200000
            mock_validate.assert_called_once_with(model_data)


class Test_ModelsEndpoint_GetContextLength_02_NegativeBehaviors:
    """Test negative behaviors for ModelsEndpoint.get_context_length method."""

    def test_model_data_missing_context_length_attribute(self, models_endpoint_base, mock_response):
        """Test behavior when model data missing context_length attribute raises AttributeError."""
        # Arrange
        model_data = {
            "id": "test-model",
            "name": "Test Model",
            # Missing context_length
            "pricing": {"prompt": "0.000015", "completion": "0.000075"}
        }
        mock_response.json.return_value = model_data
        models_endpoint_base.http_manager.get.return_value = mock_response
        
        with patch.object(ModelData, 'model_validate') as mock_validate:
            mock_model_data = Mock(spec=['id', 'name', 'pricing'])  # spec without context_length
            mock_validate.return_value = mock_model_data
            
            # Act & Assert
            with pytest.raises(AttributeError):
                models_endpoint_base.get_context_length("test-model")


class Test_ModelsEndpoint_GetContextLength_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ModelsEndpoint.get_context_length method."""

    @pytest.mark.parametrize("exception_type,error_message", [
        (ConnectionError, "Network error"),
        (ValueError, "Invalid model data"),
        (APIError, "API error"),
    ])
    def test_error_propagation_from_underlying_get_method(self, models_endpoint_base, exception_type, error_message):
        """Test that errors from underlying get method are properly propagated."""
        # Arrange
        models_endpoint_base.http_manager.get.side_effect = exception_type(error_message)
        
        # Act & Assert
        with pytest.raises(exception_type) as exc_info:
            models_endpoint_base.get_context_length("test-model")
        
        assert error_message in str(exc_info.value)


class Test_ModelsEndpoint_GetModelPricing_01_NominalBehaviors:
    """Test nominal behaviors for ModelsEndpoint.get_model_pricing method."""

    def test_successful_extraction_of_model_pricing_object_from_model_data(self, models_endpoint_base, mock_response):
        """Test successful extraction of ModelPricing object from model data."""
        # Arrange
        model_data = {
            "id": "test-model",
            "name": "Test Model",
            "context_length": 200000,
            "pricing": {"prompt": "0.000015", "completion": "0.000075"}
        }
        mock_response.json.return_value = model_data
        models_endpoint_base.http_manager.get.return_value = mock_response
        
        with patch.object(ModelData, 'model_validate') as mock_validate:
            mock_pricing = Mock(spec=ModelPricing)
            mock_model_data = Mock(spec=ModelData)
            mock_model_data.pricing = mock_pricing
            mock_validate.return_value = mock_model_data
            
            # Act
            result = models_endpoint_base.get_model_pricing("test-model")
            
            # Assert
            assert result is mock_pricing
            mock_validate.assert_called_once_with(model_data)


class Test_ModelsEndpoint_GetModelPricing_02_NegativeBehaviors:
    """Test negative behaviors for ModelsEndpoint.get_model_pricing method."""

    def test_model_data_missing_pricing_attribute(self, models_endpoint_base, mock_response):
        """Test behavior when model data missing pricing attribute raises AttributeError."""
        # Arrange
        model_data = {
            "id": "test-model",
            "name": "Test Model",
            "context_length": 200000
            # Missing pricing
        }
        mock_response.json.return_value = model_data
        models_endpoint_base.http_manager.get.return_value = mock_response
        
        with patch.object(ModelData, 'model_validate') as mock_validate:
            mock_model_data = Mock(spec=['id', 'name', 'context_length'])  # spec without pricing
            mock_validate.return_value = mock_model_data
            
            # Act & Assert
            with pytest.raises(AttributeError):
                models_endpoint_base.get_model_pricing("test-model")


class Test_ModelsEndpoint_GetModelPricing_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ModelsEndpoint.get_model_pricing method."""

    @pytest.mark.parametrize("exception_type,error_message", [
        (ConnectionError, "Network failure"),
        (ValueError, "Model validation failed"),
        (APIError, "API request failed"),
    ])
    def test_error_propagation_from_underlying_get_method(self, models_endpoint_base, exception_type, error_message):
        """Test that errors from underlying get method are properly propagated."""
        # Arrange
        models_endpoint_base.http_manager.get.side_effect = exception_type(error_message)
        
        # Act & Assert
        with pytest.raises(exception_type) as exc_info:
            models_endpoint_base.get_model_pricing("test-model")
        
        assert error_message in str(exc_info.value)


class Test_ModelsEndpoint_ListEndpoints_01_NominalBehaviors:
    """Test nominal behaviors for ModelsEndpoint.list_endpoints method."""

    def test_successful_endpoint_listing_with_valid_author_and_slug_parameters(self, models_endpoint_base, mock_response):
        """Test successful endpoint listing with valid author and slug parameters."""
        # Arrange
        endpoints_data = {
            "object": "list",
            "data": [
                {
                    "id": "chat",
                    "name": "Chat Completion",
                    "url": "/api/v1/chat/completions",
                    "method": "POST"
                }
            ]
        }
        mock_response.json.return_value = endpoints_data
        models_endpoint_base.http_manager.get.return_value = mock_response
        
        with patch.object(ModelEndpointsResponse, 'model_validate') as mock_validate:
            mock_endpoints_response = Mock(spec=ModelEndpointsResponse)
            mock_validate.return_value = mock_endpoints_response
            
            # Act
            result = models_endpoint_base.list_endpoints("anthropic", "claude-3-opus")
            
            # Assert
            mock_validate.assert_called_once_with(endpoints_data)
            assert result is mock_endpoints_response
            models_endpoint_base.http_manager.get.assert_called_once()


class Test_ModelsEndpoint_ListEndpoints_02_NegativeBehaviors:
    """Test negative behaviors for ModelsEndpoint.list_endpoints method."""

    def test_non_existent_author_slug_combination_resulting_in_http_404(self, models_endpoint_base):
        """Test handling of non-existent author/slug combination resulting in HTTP 404."""
        # Arrange
        # Configure HTTP manager to raise 404 error
        models_endpoint_base.http_manager.get.side_effect = APIError(
            "Model endpoints not found",
            status_code=404
        )
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            models_endpoint_base.list_endpoints("non-existent", "model")
        
        assert "not found" in str(exc_info.value).lower()


class Test_ModelsEndpoint_ListEndpoints_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ModelsEndpoint.list_endpoints method."""

    @pytest.mark.parametrize("exception_type,error_message", [
        (ConnectionError, "Authentication failed"),
        (ValueError, "Invalid response format"),
        (TimeoutError, "Network timeout"),
    ])
    def test_authentication_failures_and_response_validation_errors(self, models_endpoint_base, mock_response, exception_type, error_message):
        """Test handling of authentication failures and response validation errors."""
        # Arrange
        if exception_type in [ConnectionError, TimeoutError]:
            models_endpoint_base.http_manager.get.side_effect = exception_type(error_message)
            
            # Act & Assert
            with pytest.raises(exception_type) as exc_info:
                models_endpoint_base.list_endpoints("author", "slug")
            
            assert error_message in str(exc_info.value)
        else:
            # For validation errors, mock successful HTTP but failed validation
            mock_response.json.return_value = {"invalid": "data"}
            models_endpoint_base.http_manager.get.return_value = mock_response
            
            with patch.object(ModelEndpointsResponse, 'model_validate', side_effect=exception_type(error_message)):
                # Act & Assert
                with pytest.raises(exception_type) as exc_info:
                    models_endpoint_base.list_endpoints("author", "slug")
                
                assert error_message in str(exc_info.value)

    def test_json_parsing_and_model_endpoints_response_validation_failures(self, models_endpoint_base, mock_response):
        """Test handling of JSON parsing and ModelEndpointsResponse validation failures."""
        # Arrange
        # Test JSON parsing failure
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        models_endpoint_base.http_manager.get.return_value = mock_response
        
        # Act & Assert
        with pytest.raises(json.JSONDecodeError):
            models_endpoint_base.list_endpoints("author", "slug")
        
        # Reset for next test
        mock_response.json.side_effect = None
        mock_response.json.return_value = {"invalid": "data"}
        
        # Test validation failure
        with patch.object(ModelEndpointsResponse, 'model_validate', side_effect=ValueError("Validation failed")):
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                models_endpoint_base.list_endpoints("author", "slug")
            
            assert "Validation failed" in str(exc_info.value)
