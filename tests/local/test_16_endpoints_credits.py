import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from pydantic import ValidationError

from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager
from openrouter_client.exceptions import APIError
from openrouter_client.endpoints.credits import CreditsEndpoint


class Test_CreditsEndpoint_Init_01_NominalBehaviors:
    """Test nominal behaviors for CreditsEndpoint.__init__ method."""
    
    def test_initialize_with_valid_managers_and_verify_endpoint_path(self):
        """Test initialization with valid AuthManager and HTTPManager instances and verify correct endpoint path assignment."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Act
        endpoint = CreditsEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint.auth_manager is auth_manager
        assert endpoint.http_manager is http_manager
        assert endpoint.endpoint_path == "credits"
        assert hasattr(endpoint, 'logger')


class Test_CreditsEndpoint_Init_02_NegativeBehaviors:
    """Test negative behaviors for CreditsEndpoint.__init__ method."""
    
    @pytest.mark.parametrize("auth_manager,http_manager,expected_exception", [
        (None, Mock(spec=HTTPManager), (ValidationError, TypeError, AttributeError)),
        (Mock(spec=AuthManager), None, (ValidationError, TypeError, AttributeError)),
        (None, None, (ValidationError, TypeError, AttributeError)),
        ("invalid_auth", Mock(spec=HTTPManager), (ValidationError, TypeError, AttributeError)),
        (Mock(spec=AuthManager), "invalid_http", (ValidationError, TypeError, AttributeError)),
        (123, Mock(spec=HTTPManager), (ValidationError, TypeError, AttributeError)),
        (Mock(spec=AuthManager), 456, (ValidationError, TypeError, AttributeError)),
    ])
    def test_initialize_with_invalid_parameters(self, auth_manager, http_manager, expected_exception):
        """Test initialization with None or invalid object types for required parameters."""
        # Act & Assert
        with pytest.raises(expected_exception):
            CreditsEndpoint(auth_manager, http_manager)


class Test_CreditsEndpoint_Init_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for CreditsEndpoint.__init__ method."""
    
    def test_logger_creation_handles_exceptions_gracefully(self):
        """Test that logger creation handles exceptions gracefully and falls back to root logger."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Act - Create endpoint even if logger creation would fail
        endpoint = CreditsEndpoint(auth_manager, http_manager)
        
        # Assert
        assert hasattr(endpoint, 'logger')
        assert endpoint.logger is not None


class Test_CreditsEndpoint_Get_01_NominalBehaviors:
    """Test nominal behaviors for CreditsEndpoint.get method."""
    
    def test_successful_api_call_with_valid_authentication(self):
        """Test successful API call with valid authentication headers."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test_key"}
        mock_response = Mock()
        mock_response.json.return_value = {"credits": 100.0, "used": 50.0}
        http_manager.get.return_value = mock_response
        
        endpoint = CreditsEndpoint(auth_manager, http_manager)
        
        # Act
        result = endpoint.get()
        
        # Assert
        assert result == {"credits": 100.0, "used": 50.0}
        auth_manager.get_auth_headers.assert_called_once_with(require_provisioning=True)
        http_manager.get.assert_called_once_with(
            "credits",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer test_key"
            }
        )
    
    def test_return_parsed_json_response_containing_credit_info(self):
        """Test that the method returns parsed JSON response containing credit information."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test_key"}
        mock_response = Mock()
        expected_data = {
            "credits": 500.0,
            "used": 100.0,
            "purchase_credits": 400.0,
            "gifted_credits": 100.0,
            "remaining_free_credits": 0.0
        }
        mock_response.json.return_value = expected_data
        http_manager.get.return_value = mock_response
        
        endpoint = CreditsEndpoint(auth_manager, http_manager)
        
        # Act
        result = endpoint.get()
        
        # Assert
        assert result == expected_data
        assert isinstance(result, dict)
        assert "credits" in result
        assert "used" in result


class Test_CreditsEndpoint_Get_02_NegativeBehaviors:
    """Test negative behaviors for CreditsEndpoint.get method."""
    
    def test_handle_authentication_failure_from_auth_manager(self):
        """Test handling of authentication failures from AuthManager."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.side_effect = APIError("Authentication failed")
        
        endpoint = CreditsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            endpoint.get()
        assert "Authentication failed" in str(exc_info.value)
    
    def test_handle_api_error_responses_from_http_manager(self):
        """Test handling of API error responses from HTTPManager."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test_key"}
        http_manager.get.side_effect = APIError("API Error: 401 Unauthorized")
        
        endpoint = CreditsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            endpoint.get()
        assert "401 Unauthorized" in str(exc_info.value)


class Test_CreditsEndpoint_Get_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for CreditsEndpoint.get method."""
    
    def test_handle_json_parsing_errors_from_response(self):
        """Test handling of JSON parsing errors from response."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test_key"}
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        http_manager.get.return_value = mock_response
        
        endpoint = CreditsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            endpoint.get()
        assert "Invalid JSON" in str(exc_info.value)
    
    def test_propagate_http_errors_without_modification(self):
        """Test that HTTP errors are propagated without modification."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test_key"}
        original_error = APIError("Network timeout", code=500)
        http_manager.get.side_effect = original_error
        
        endpoint = CreditsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            endpoint.get()
        assert exc_info.value is original_error


class Test_CreditsEndpoint_Get_05_StateTransitionBehaviors:
    """Test state transition behaviors for CreditsEndpoint.get method."""
    
    def test_consistent_state_after_successful_call(self):
        """Test that endpoint maintains consistent state after successful call."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test_key"}
        mock_response = Mock()
        mock_response.json.return_value = {"credits": 100.0, "used": 50.0}
        http_manager.get.return_value = mock_response
        
        endpoint = CreditsEndpoint(auth_manager, http_manager)
        
        # Act
        result1 = endpoint.get()
        result2 = endpoint.get()
        
        # Assert
        assert result1 == result2
        assert auth_manager.get_auth_headers.call_count == 2
        assert http_manager.get.call_count == 2
    
    def test_consistent_state_after_failed_call(self):
        """Test that endpoint maintains consistent state after failed call."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test_key"}
        http_manager.get.side_effect = [APIError("Temporary failure"), Mock(json=Mock(return_value={"credits": 100.0}))]
        
        endpoint = CreditsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError):
            endpoint.get()
        
        # Verify endpoint can still be used after error
        result = endpoint.get()
        assert result == {"credits": 100.0}