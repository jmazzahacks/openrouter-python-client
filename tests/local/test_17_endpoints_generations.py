import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from pydantic import ValidationError

from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager
from openrouter_client.endpoints.generations import GenerationsEndpoint


class TestGenerationsEndpointInit01NominalBehaviors:
    """Test nominal behaviors for GenerationsEndpoint.__init__()"""
    
    def test_successful_initialization_with_valid_managers(self):
        """Test successful initialization with valid AuthManager and HTTPManager instances"""
        # Arrange
        auth_manager= AuthManager()
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesnotexist.com")
        
        # Act
        endpoint = GenerationsEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint.auth_manager is auth_manager
        assert endpoint.http_manager is http_manager
        assert endpoint.endpoint_path == "generation"  # Note: singular, not plural


class TestGenerationsEndpointInit02NegativeBehaviors:
    """Test negative behaviors for GenerationsEndpoint.__init__()"""
    
    @pytest.mark.parametrize("auth_manager,http_manager", [
        (None, Mock()),
        (Mock(), None),
        (None, None),
        ("invalid_type", Mock()),
        (Mock(), "invalid_type"),
        (123, Mock()),
        (Mock(), 456),
    ])
    def test_initialization_with_invalid_parameters(self, auth_manager, http_manager):
        """Test initialization fails with None or invalid object types"""
        # Arrange, Act & Assert
        with pytest.raises((TypeError, AttributeError, ValueError)):
            GenerationsEndpoint(auth_manager, http_manager)


class TestGenerationsEndpointInit04ErrorHandlingBehaviors:
    """Test error handling behaviors for GenerationsEndpoint.__init__()"""
    
    def test_logger_creation_handles_exceptions_gracefully(self):
        """Test logger creation handles exceptions gracefully"""
        # Arrange
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://api.openrouter.ai/v1")
        
        # Act - Should succeed even if logger creation fails
        endpoint = GenerationsEndpoint(auth_manager, http_manager)
        
        # Assert
        assert hasattr(endpoint, 'logger')
        assert endpoint.logger is not None


class TestGenerationsEndpointGet01NominalBehaviors:
    """Test nominal behaviors for GenerationsEndpoint.get()"""
    
    def test_successful_api_call_with_valid_generation_id(self):
        """Test successful API call with valid generation ID"""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test_key"}
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "gen-12345",
            "model": "gpt-4",
            "status": "completed",
            "tokens": 150,
            "cost": 0.005
        }
        http_manager.get.return_value = mock_response
        
        endpoint = GenerationsEndpoint(auth_manager, http_manager)
        
        # Act
        result = endpoint.get("gen-12345")
        
        # Assert
        assert result["id"] == "gen-12345"
        assert result["status"] == "completed"
        http_manager.get.assert_called_once_with(
            "generation",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer test_key"
            },
            params={"id": "gen-12345"}
        )
    
    def test_proper_query_parameter_construction(self):
        """Test that generation ID is properly passed as query parameter"""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {}
        
        mock_response = Mock()
        mock_response.json.return_value = {"id": "test-id"}
        http_manager.get.return_value = mock_response
        
        endpoint = GenerationsEndpoint(auth_manager, http_manager)
        
        # Act
        endpoint.get("test-generation-id")
        
        # Assert
        call_args = http_manager.get.call_args
        assert call_args[1]['params'] == {"id": "test-generation-id"}


class TestGenerationsEndpointGet02NegativeBehaviors:
    """Test negative behaviors for GenerationsEndpoint.get()"""
    
    def test_handle_api_error_for_invalid_generation_id(self):
        """Test handling of API error for invalid generation ID"""
        # Arrange
        from openrouter_client.exceptions import APIError
        
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {}
        http_manager.get.side_effect = APIError("Generation not found", code=404)
        
        endpoint = GenerationsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            endpoint.get("invalid-id")
        assert "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()
    
    def test_handle_none_generation_id(self):
        """Test handling of None as generation ID"""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        endpoint = GenerationsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises((TypeError, AttributeError)):
            endpoint.get(None)
    
    def test_handle_empty_generation_id(self):
        """Test handling of empty string as generation ID"""
        # Arrange
        from openrouter_client.exceptions import APIError
        
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {}
        http_manager.get.side_effect = APIError("Invalid generation ID", code=400)
        
        endpoint = GenerationsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError):
            endpoint.get("")


class TestGenerationsEndpointGet04ErrorHandlingBehaviors:
    """Test error handling behaviors for GenerationsEndpoint.get()"""
    
    def test_handle_json_parsing_errors(self):
        """Test handling of JSON parsing errors from response"""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {}
        
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        http_manager.get.return_value = mock_response
        
        endpoint = GenerationsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(ValueError):
            endpoint.get("test-id")
    
    def test_propagate_http_errors_without_modification(self):
        """Test that HTTP errors are propagated without modification"""
        # Arrange
        from openrouter_client.exceptions import APIError
        
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {}
        
        original_error = APIError("Server error", code=500)
        http_manager.get.side_effect = original_error
        
        endpoint = GenerationsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            endpoint.get("test-id")
        assert exc_info.value is original_error


class TestGenerationsEndpointGet05StateTransitionBehaviors:
    """Test state transition behaviors for GenerationsEndpoint.get()"""
    
    def test_consistent_state_after_successful_call(self):
        """Test endpoint maintains consistent state after successful call"""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {}
        
        mock_response = Mock()
        mock_response.json.return_value = {"id": "test-id", "status": "completed"}
        http_manager.get.return_value = mock_response
        
        endpoint = GenerationsEndpoint(auth_manager, http_manager)
        
        # Act
        result1 = endpoint.get("id-1")
        result2 = endpoint.get("id-2")
        
        # Assert
        assert http_manager.get.call_count == 2
        assert endpoint.endpoint_path == "generation"  # State unchanged
    
    def test_consistent_state_after_failed_call(self):
        """Test endpoint maintains consistent state after failed call"""
        # Arrange
        from openrouter_client.exceptions import APIError
        
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {}
        
        http_manager.get.side_effect = [
            APIError("Error"),
            Mock(json=Mock(return_value={"id": "test-id"}))
        ]
        
        endpoint = GenerationsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError):
            endpoint.get("id-1")
        
        # Endpoint should still work after error
        result = endpoint.get("id-2")
        assert result == {"id": "test-id"}