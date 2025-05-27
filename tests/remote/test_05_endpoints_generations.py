import pytest
import os
from datetime import datetime, timedelta
from typing import Optional, Union
import time

from openrouter_client import OpenRouterClient
from openrouter_client.http import HTTPManager
from openrouter_client.auth import AuthManager
from openrouter_client.endpoints.generations import GenerationsEndpoint
from openrouter_client.exceptions import APIError


@pytest.fixture(scope="session")
def http_manager():
    """Shared OpenRouter client instance for all tests."""
    return HTTPManager(base_url="https://openrouter.ai/api/v1")

@pytest.fixture(scope="session")
def auth_manager():
    """Shared AuthManager instance for all tests."""
    return AuthManager()


@pytest.fixture
def generations_endpoint(http_manager, auth_manager):
    """Generations endpoint instance for testing."""
    return GenerationsEndpoint(auth_manager, http_manager)


class Test_GenerationsEndpoint_Init_01_NominalBehaviors:
    """Test nominal behaviors for GenerationsEndpoint initialization."""
    
    def test_successful_initialization_with_valid_managers(self, auth_manager, http_manager):
        """Test successful initialization with valid AuthManager and HTTPManager instances."""
        # Act
        endpoint = GenerationsEndpoint(auth_manager, http_manager)
        
        # Assert
        assert isinstance(endpoint.auth_manager, AuthManager)
        assert isinstance(endpoint.http_manager, HTTPManager)
        assert endpoint.endpoint_path == "generation"  # Note: singular, not plural
        assert hasattr(endpoint, 'logger')

    def test_proper_endpoint_path_configuration(self, auth_manager, http_manager):
        """Test proper endpoint path configuration to 'generation'."""
        # Arrange & Act
        endpoint = GenerationsEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint.endpoint_path == "generation"
        assert endpoint._get_endpoint_url() == "generation"


class Test_GenerationsEndpoint_Init_02_NegativeBehaviors:
    """Test negative behaviors for GenerationsEndpoint initialization."""
    
    @pytest.mark.parametrize("auth_manager,http_manager", [
        (None, None),
        (None, "invalid_http_manager"),
        ("invalid_auth_manager", None),
        ("invalid_auth_manager", "invalid_http_manager"),
    ])
    def test_initialization_with_invalid_manager_instances(self, auth_manager, http_manager):
        """Test initialization with None or invalid manager instances."""
        # Arrange & Act & Assert
        with pytest.raises((TypeError, AttributeError, ValueError)):
            GenerationsEndpoint(auth_manager, http_manager)


class Test_GenerationsEndpoint_Get_01_NominalBehaviors:
    """Test nominal behaviors for the get method."""
    
    def test_get_with_valid_generation_id(self, generations_endpoint):
        """Test retrieving generation metadata with a valid generation ID."""
        # Arrange
        # Note: We need a valid generation ID from a previous request
        # Since we can't guarantee one exists, we'll skip this test
        pytest.skip("Requires a valid generation ID from a previous request")
    
    def test_proper_query_parameter_construction(self, auth_manager):
        """Test that the generation ID is properly included as a query parameter."""
        # Arrange
        from unittest.mock import Mock, MagicMock
        
        mock_http = Mock(spec=HTTPManager)
        mock_response = Mock()
        mock_response.json.return_value = {"id": "test-id", "status": "completed"}
        mock_http.get.return_value = mock_response
        
        endpoint = GenerationsEndpoint(auth_manager, mock_http)
        
        # Act
        result = endpoint.get("test-generation-id")
        
        # Assert
        mock_http.get.assert_called_once()
        call_args = mock_http.get.call_args
        assert call_args[1]['params'] == {"id": "test-generation-id"}
        assert result == {"id": "test-id", "status": "completed"}


class Test_GenerationsEndpoint_Get_02_NegativeBehaviors:
    """Test negative behaviors for the get method."""
    
    def test_get_with_invalid_generation_id(self, generations_endpoint):
        """Test retrieving generation metadata with an invalid generation ID."""
        # Arrange
        invalid_id = "invalid-generation-id-12345"
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            generations_endpoint.get(invalid_id)
        # Should get a 404 or similar error for invalid ID
        assert "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()
    
    def test_get_with_empty_generation_id(self, generations_endpoint):
        """Test retrieving generation metadata with an empty generation ID."""
        # Arrange
        empty_id = ""
        
        # Act & Assert
        with pytest.raises(APIError):
            generations_endpoint.get(empty_id)
    
    @pytest.mark.parametrize("invalid_id", [
        None,
        123,  # Wrong type
        [],   # Wrong type
        {},   # Wrong type
    ])
    def test_get_with_wrong_type_generation_id(self, generations_endpoint, invalid_id):
        """Test retrieving generation metadata with wrong type for generation ID."""
        # Act & Assert
        with pytest.raises((TypeError, AttributeError, APIError)):
            generations_endpoint.get(invalid_id)


class Test_GenerationsEndpoint_Get_03_BoundaryBehaviors:
    """Test boundary behaviors for the get method."""
    
    def test_get_with_very_long_generation_id(self, generations_endpoint):
        """Test retrieving generation metadata with a very long generation ID."""
        # Arrange
        long_id = "a" * 1000
        
        # Act & Assert
        with pytest.raises(APIError):
            generations_endpoint.get(long_id)
    
    def test_get_with_special_characters_in_id(self, generations_endpoint):
        """Test retrieving generation metadata with special characters in ID."""
        # Arrange
        special_id = "test!@#$%^&*()_+-=[]{}|;':\",./<>?"
        
        # Act & Assert
        with pytest.raises(APIError):
            generations_endpoint.get(special_id)


class Test_GenerationsEndpoint_Get_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for the get method."""
    
    def test_network_connectivity_failures(self, auth_manager):
        """Test handling of network connectivity failures."""
        # Arrange
        from unittest.mock import Mock
        
        mock_http = Mock(spec=HTTPManager)
        mock_http.get.side_effect = APIError("Network connectivity failure")
        
        endpoint = GenerationsEndpoint(auth_manager, mock_http)
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            endpoint.get("test-id")
        assert "Network connectivity failure" in str(exc_info.value)
    
    def test_malformed_json_response(self, auth_manager):
        """Test handling of malformed JSON responses."""
        # Arrange
        from unittest.mock import Mock
        
        mock_http = Mock(spec=HTTPManager)
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_http.get.return_value = mock_response
        
        endpoint = GenerationsEndpoint(auth_manager, mock_http)
        
        # Act & Assert
        with pytest.raises(ValueError):
            endpoint.get("test-id")
    
    @pytest.mark.parametrize("status_code,error_type", [
        (400, "Bad Request"),
        (401, "Unauthorized"),
        (403, "Forbidden"),
        (404, "Not Found"),
        (500, "Internal Server Error"),
        (502, "Bad Gateway"),
        (503, "Service Unavailable"),
    ])
    def test_http_error_status_codes(self, auth_manager, status_code, error_type):
        """Test handling of various HTTP error status codes."""
        # Arrange
        from unittest.mock import Mock
        
        mock_http = Mock(spec=HTTPManager)
        mock_http.get.side_effect = APIError(f"HTTP {status_code}: {error_type}")
        
        endpoint = GenerationsEndpoint(auth_manager, mock_http)
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            endpoint.get("test-id")
        assert str(status_code) in str(exc_info.value)


class Test_GenerationsEndpoint_Get_05_StateTransitionBehaviors:
    """Test state transition behaviors for the get method."""
    
    def test_endpoint_state_consistency_after_successful_call(self, auth_manager):
        """Test that endpoint maintains consistent state after successful call."""
        # Arrange
        from unittest.mock import Mock
        
        mock_http = Mock(spec=HTTPManager)
        mock_response = Mock()
        mock_response.json.return_value = {"id": "test-id", "status": "completed"}
        mock_http.get.return_value = mock_response
        
        endpoint = GenerationsEndpoint(auth_manager, mock_http)
        
        # Act
        result1 = endpoint.get("test-id-1")
        result2 = endpoint.get("test-id-2")
        
        # Assert
        assert mock_http.get.call_count == 2
        assert endpoint.endpoint_path == "generation"  # State unchanged
    
    def test_endpoint_state_consistency_after_failed_call(self, auth_manager):
        """Test that endpoint maintains consistent state after failed call."""
        # Arrange
        from unittest.mock import Mock
        
        mock_http = Mock(spec=HTTPManager)
        mock_http.get.side_effect = [
            APIError("First call fails"),
            Mock(json=Mock(return_value={"id": "test-id", "status": "completed"}))
        ]
        
        endpoint = GenerationsEndpoint(auth_manager, mock_http)
        
        # Act & Assert
        with pytest.raises(APIError):
            endpoint.get("test-id-1")
        
        # Endpoint should still work after error
        result = endpoint.get("test-id-2")
        assert result == {"id": "test-id", "status": "completed"}