"""
Comprehensive test suite for KeysEndpoint using Pytest.

This module provides exhaustive testing of the KeysEndpoint class,
covering all vital behaviors identified through behavioral analysis.
"""

import pytest
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager
from openrouter_client.endpoints.keys import KeysEndpoint
from openrouter_client.exceptions import APIError


class TestKeysEndpointInit01NominalBehaviors:
    """Test nominal behaviors for KeysEndpoint.__init__ method."""

    def test_successful_initialization_with_valid_managers(self):
        """Test successful initialization with valid AuthManager and HTTPManager instances."""
        # Arrange
        auth_manager = AuthManager(api_key="test_key")
        http_manager = HTTPManager(
            base_url="https://invalid.thisurldoesnotexist.com", timeout=1)

        # Act
        endpoint = KeysEndpoint(auth_manager, http_manager)

        # Assert
        assert endpoint.auth_manager is auth_manager
        assert endpoint.http_manager is http_manager
        assert hasattr(endpoint, 'logger')
        assert isinstance(endpoint.logger, logging.Logger)

    def test_correct_inheritance_and_endpoint_path_configuration(self):
        """Test correct inheritance and endpoint path configuration."""
        # Arrange
        auth_manager = AuthManager(api_key="test_key")
        http_manager = HTTPManager(
            base_url="https://invalid.thisurldoesnotexist.com", timeout=1)

        # Act
        endpoint = KeysEndpoint(auth_manager, http_manager)

        # Assert
        assert endpoint.endpoint_path == "keys"
        assert hasattr(endpoint, '_get_endpoint_url')
        assert hasattr(endpoint, '_get_headers')


class TestKeysEndpointInit02NegativeBehaviors:
    """Test negative behaviors for KeysEndpoint.__init__ method."""

    @pytest.mark.parametrize("auth_manager,http_manager", [
        (None, HTTPManager(base_url="https://invalid.thisurldoesnotexist.com", timeout=1)),
        (AuthManager(api_key="test"), None),
        (None, None),
        ("invalid_auth", HTTPManager(
            base_url="https://invalid.thisurldoesnotexist.com", timeout=1)),
        (AuthManager(api_key="test"), "invalid_http"),
        (123, HTTPManager(base_url="https://invalid.thisurldoesnotexist.com", timeout=1)),
        (AuthManager(api_key="test"), 456),
    ])
    def test_initialization_with_invalid_parameters(self, auth_manager, http_manager):
        """Test initialization with None or invalid types for required parameters."""
        # Arrange & Act & Assert
        with pytest.raises((TypeError, AttributeError, ValueError)):
            KeysEndpoint(auth_manager, http_manager)


class TestKeysEndpointInit04ErrorHandlingBehaviors:
    """Test error handling behaviors for KeysEndpoint.__init__ method."""

    @patch('openrouter_client.http.HTTPManager.get')
    def test_exception_propagation_from_parent_initialization(self, mock_get):
        """Test proper exception propagation from auth manager failures during method calls."""
        # Arrange - Mock successful HTTP response
        mock_response = Mock()
        mock_response.json.return_value = {"data": [{"hash": "key1", "name": "test_key"}]}
        mock_get.return_value = mock_response
        
        class FailingAuthManager(AuthManager):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
            def get_auth_headers(self, require_provisioning=False, **kwargs):
                # Simulate auth manager failure at the right point
                raise RuntimeError("Auth manager initialization failed")

        failing_auth = FailingAuthManager(api_key="test")
        http_manager = HTTPManager(
            base_url="https://api.example.com", timeout=1)
        
        endpoint = KeysEndpoint(failing_auth, http_manager)
        
        # Assert - Exception should propagate from auth manager
        with pytest.raises(RuntimeError, match="Auth manager initialization failed"):
            endpoint.list()
        
        # Verify HTTP request was never made due to auth failure
        mock_get.assert_not_called()


class TestKeysEndpointInit05StateTransitionBehaviors:
    """Test state transition behaviors for KeysEndpoint.__init__ method."""

    def test_object_state_transition_to_operational(self):
        """Test object state transition from uninitialized to fully operational."""
        # Arrange
        auth_manager = AuthManager(api_key="test_key")
        http_manager = HTTPManager(
            base_url="https://invalid.thisurldoesnotexist.com", timeout=1)

        # Act
        endpoint = KeysEndpoint(auth_manager, http_manager)

        # Assert
        assert endpoint.auth_manager is not None
        assert endpoint.http_manager is not None
        assert endpoint.endpoint_path == "keys"
        assert endpoint.logger is not None
        # Verify the object is in a fully operational state
        assert hasattr(endpoint, 'list')
        assert hasattr(endpoint, 'get')
        assert hasattr(endpoint, 'create')
        assert hasattr(endpoint, 'update')
        assert hasattr(endpoint, 'delete')
        assert hasattr(endpoint, 'get_current')


class TestKeysEndpointList01NominalBehaviors:
    """Test nominal behaviors for KeysEndpoint.list method."""

    def test_successful_retrieval_with_proper_authentication(self):
        """Test successful retrieval of API key list with proper authentication."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.get') as mock_get:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer test-provisioning-key"}
            expected_keys = {
                "data": [
                    {
                        "created_at": "2025-02-19T20:52:27.363244+00:00",
                        "updated_at": "2025-02-19T21:24:11.708154+00:00",
                        "hash": "key1",
                        "label": "sk-or-v1-customkey1",
                        "name": "Test Key 1",
                        "disabled": False,
                        "limit": 10,
                        "usage": 0
                    },
                    {
                        "created_at": "2025-02-20T20:52:27.363244+00:00",
                        "updated_at": "2025-02-20T21:24:11.708154+00:00",
                        "hash": "key2",
                        "label": "sk-or-v1-customkey2",
                        "name": "Test Key 2",
                        "disabled": False,
                        "limit": 20,
                        "usage": 5
                    }
                ]
            }
            mock_response = Mock()
            mock_response.json.return_value = expected_keys
            mock_get.return_value = mock_response

            auth_manager = AuthManager(api_key="valid_provisioning_key")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act
            result = endpoint.list()

            # Assert
            assert isinstance(result, dict)
            assert "data" in result
            assert len(result["data"]) == 2
            assert all(isinstance(key, dict) for key in result["data"])
            assert all("hash" in key for key in result["data"])
            assert result["data"][0]["hash"] == "key1"
            assert result["data"][1]["hash"] == "key2"

            # Verify authentication was called correctly
            mock_auth.assert_called_once_with(require_provisioning=True)
            mock_get.assert_called_once()

    def test_list_with_pagination_parameters(self):
        """Test list method with pagination parameters."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.get') as mock_get:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer test-provisioning-key"}
            expected_keys = {"data": []}
            mock_response = Mock()
            mock_response.json.return_value = expected_keys
            mock_get.return_value = mock_response

            auth_manager = AuthManager(api_key="valid_provisioning_key")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act
            result = endpoint.list(offset=100, include_disabled=True)

            # Assert
            assert isinstance(result, dict)
            
            # Verify the correct parameters were passed
            mock_get.assert_called_once_with(
                "keys",
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json', 
                    'Authorization': 'Bearer test-provisioning-key'
                },
                params={"offset": 100, "includeDisabled": "true"}
            )

    def test_handling_empty_key_lists(self):
        """Test correct handling of empty key lists."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.get') as mock_get:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer test-provisioning-key"}
            mock_response = Mock()
            mock_response.json.return_value = {"data": []}
            mock_get.return_value = mock_response

            auth_manager = AuthManager(api_key="valid_provisioning_key")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act
            result = endpoint.list()

            # Assert
            assert isinstance(result, dict)
            assert "data" in result
            assert len(result["data"]) == 0

            # Verify authentication was called correctly
            mock_auth.assert_called_once_with(require_provisioning=True)
            mock_get.assert_called_once()


class TestKeysEndpointList02NegativeBehaviors:
    """Test negative behaviors for KeysEndpoint.list method."""

    def test_unauthorized_access_without_provisioning_key(self):
        """Test unauthorized access attempts without proper provisioning key."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.get') as mock_get:

            # Arrange
            mock_auth.side_effect = APIError("Unauthorized")

            auth_manager = AuthManager(api_key="invalid_key")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act & Assert
            with pytest.raises(APIError):
                endpoint.list()


class TestKeysEndpointList03BoundaryBehaviors:
    """Test boundary behaviors for KeysEndpoint.list method."""

    def test_empty_response_handling(self):
        """Test empty response handling."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.get') as mock_get:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer test-provisioning-key"}
            mock_response = Mock()
            mock_response.json.return_value = {"data": []}
            mock_get.return_value = mock_response

            auth_manager = AuthManager(api_key="valid_provisioning_key")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act
            result = endpoint.list()

            # Assert
            assert isinstance(result, dict)
            assert result["data"] == []

            # Verify authentication was called correctly
            mock_auth.assert_called_once_with(require_provisioning=True)
            mock_get.assert_called_once()


class TestKeysEndpointList04ErrorHandlingBehaviors:
    """Test error handling behaviors for KeysEndpoint.list method."""

    @pytest.mark.parametrize("status_code,error_type", [
        (500, "http_500_error"),
        (404, "http_404_error"),
        (403, "http_403_error")
    ])
    def test_network_and_http_error_handling(self, status_code, error_type):
        """Test network failure and HTTP error status code handling."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.get') as mock_get:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer test-provisioning-key"}
            mock_get.side_effect = APIError(f"HTTP {status_code} error")

            auth_manager = AuthManager(api_key="valid_provisioning_key")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act & Assert
            with pytest.raises(APIError):
                endpoint.list()

    def test_json_parsing_failures(self):
        """Test JSON parsing failures from malformed responses."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.get') as mock_get:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer test-provisioning-key"}
            mock_response = Mock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_get.return_value = mock_response

            auth_manager = AuthManager(api_key="valid_provisioning_key")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act & Assert
            with pytest.raises(ValueError):
                endpoint.list()


class TestKeysEndpointGet01NominalBehaviors:
    """Test nominal behaviors for KeysEndpoint.get method."""

    def test_successful_get_with_valid_key_hash(self):
        """Test successful retrieval of a specific API key."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.get') as mock_get:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer test-provisioning-key"}
            expected_key = {
                "created_at": "2025-02-19T20:52:27.363244+00:00",
                "updated_at": "2025-02-19T21:24:11.708154+00:00",
                "hash": "test_hash",
                "label": "sk-or-v1-customkey",
                "name": "Test Key",
                "disabled": False,
                "limit": 10,
                "usage": 0
            }
            mock_response = Mock()
            mock_response.json.return_value = expected_key
            mock_get.return_value = mock_response

            auth_manager = AuthManager(api_key="valid_provisioning_key")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act
            result = endpoint.get("test_hash")

            # Assert
            assert isinstance(result, dict)
            assert result["hash"] == "test_hash"
            assert result["name"] == "Test Key"

            # Verify authentication was called correctly
            mock_auth.assert_called_once_with(require_provisioning=True)
            mock_get.assert_called_once_with(
                "keys/test_hash",
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json', 
                    'Authorization': 'Bearer test-provisioning-key'
                }
            )


class TestKeysEndpointCreate01NominalBehaviors:
    """Test nominal behaviors for KeysEndpoint.create method."""

    def test_successful_creation_with_all_parameters(self):
        """Test successful creation with all parameters."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.post') as mock_post:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer test-provisioning-key"}
            mock_response = Mock()
            mock_response.json.return_value = {
                "hash": "test_key_123",
                "key": "sk-or-v1-test123",
                "name": "Test Key",
                "label": "customer-123",
                "limit": 1000,
                "created_at": "2025-01-15T10:30:00Z"
            }
            mock_post.return_value = mock_response

            auth_manager = AuthManager(api_key="valid_provisioning_key")
            http_manager = HTTPManager(
                base_url="https://invalid.thisurldoesnotexist.com", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act
            result = endpoint.create(name="Test Key", label="customer-123", limit=1000)

            # Assert - verify the result structure
            assert isinstance(result, dict)
            assert "key" in result
            assert "hash" in result
            assert result["name"] == "Test Key"
            assert result["label"] == "customer-123"
            assert result["limit"] == 1000

            # Assert - verify authentication was called correctly
            mock_auth.assert_called_once_with(require_provisioning=True)

            # Assert - verify the HTTP request was made with correct data
            mock_post.assert_called_once_with(
                "keys",
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json', 
                    'Authorization': 'Bearer test-provisioning-key'
                },
                json={"name": "Test Key", "label": "customer-123", "limit": 1000}
            )

    def test_creation_with_minimal_parameters(self):
        """Test creation with only required parameters."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.post') as mock_post:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer test-provisioning-key"}
            mock_response = Mock()
            mock_response.json.return_value = {
                "hash": "minimal_key",
                "key": "sk-or-v1-minimal",
                "name": "Minimal Key"
            }
            mock_post.return_value = mock_response

            auth_manager = AuthManager(api_key="valid_provisioning_key")
            http_manager = HTTPManager(
                base_url="https://invalid.thisurldoesnotexist.com", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act
            result = endpoint.create(name="Minimal Key")

            # Assert
            assert isinstance(result, dict)
            assert "key" in result
            assert result["name"] == "Minimal Key"

            # Verify only the name was sent
            mock_post.assert_called_once_with(
                "keys",
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json', 
                    'Authorization': 'Bearer test-provisioning-key'
                },
                json={"name": "Minimal Key"}
            )


class TestKeysEndpointCreate02NegativeBehaviors:
    """Test negative behaviors for KeysEndpoint.create method."""

    @patch('openrouter_client.auth.AuthManager.get_auth_headers')
    @patch('openrouter_client.http.HTTPManager.post')
    def test_unauthorized_key_creation(self, mock_post, mock_auth):
        """Test unauthorized key creation attempts."""
        # Arrange
        auth_manager = AuthManager(api_key="invalid_key")
        http_manager = HTTPManager(
            base_url="https://openrouter.ai/api/v1", timeout=1)
        endpoint = KeysEndpoint(auth_manager, http_manager)

        mock_auth.return_value = {"Authorization": "Bearer invalid_key"}
        mock_post.side_effect = APIError("Unauthorized")

        # Act & Assert
        with pytest.raises(APIError):
            endpoint.create(name="Test Key")

    @pytest.mark.parametrize("invalid_params", [
        {"name": 123},  # Wrong type for name
        {"name": "Test", "label": 123},  # Wrong type for label
        {"name": "Test", "limit": "not_a_number"},  # Wrong type for limit
    ])
    @patch('openrouter_client.auth.AuthManager.get_auth_headers')
    @patch('openrouter_client.http.HTTPManager.post')
    def test_invalid_parameter_type_handling(self, mock_post, mock_auth, invalid_params):
        """Test invalid parameter type handling."""
        # Arrange
        auth_manager = AuthManager(api_key="valid_provisioning_key")
        http_manager = HTTPManager(
            base_url="https://openrouter.ai/api/v1", timeout=1)
        endpoint = KeysEndpoint(auth_manager, http_manager)

        mock_auth.return_value = {
            "Authorization": "Bearer valid_provisioning_key"}
        mock_post.side_effect = TypeError("Invalid parameter type")

        # Act & Assert
        with pytest.raises((TypeError, APIError)):
            endpoint.create(**invalid_params)


class TestKeysEndpointCreate03BoundaryBehaviors:
    """Test boundary behaviors for KeysEndpoint.create method."""

    @pytest.mark.parametrize("limit_value", [
        0,     # Zero limit
        0.1,   # Small decimal
        1000000,  # Large limit
    ])
    @patch('openrouter_client.auth.AuthManager.get_auth_headers')
    @patch('openrouter_client.http.HTTPManager.post')
    def test_limit_boundary_handling(self, mock_post, mock_auth, limit_value):
        """Test limit parameter boundary handling."""
        # Arrange
        auth_manager = AuthManager(api_key="valid_provisioning_key")
        http_manager = HTTPManager(
            base_url="https://openrouter.ai/api/v1", timeout=1)
        endpoint = KeysEndpoint(auth_manager, http_manager)

        expected_result = {
            "hash": "boundary_key_123",
            "key": "sk-or-v1-boundary123",
            "name": "Test Key",
            "limit": limit_value
        }
        mock_auth.return_value = {
            "Authorization": "Bearer valid_provisioning_key"}
        mock_response = Mock()
        mock_response.json.return_value = expected_result
        mock_post.return_value = mock_response

        # Act
        result = endpoint.create(name="Test Key", limit=limit_value)

        # Assert
        assert isinstance(result, dict)
        assert result["limit"] == limit_value
        
        mock_post.assert_called_once_with(
            "keys",
            headers={
                'Content-Type': 'application/json', 
                'Accept': 'application/json', 
                'Authorization': 'Bearer valid_provisioning_key'
            },
            json={"name": "Test Key", "limit": limit_value}
        )


class TestKeysEndpointCreate04ErrorHandlingBehaviors:
    """Test error handling behaviors for KeysEndpoint.create method."""

    @patch('openrouter_client.auth.AuthManager.get_auth_headers')
    @patch('openrouter_client.http.HTTPManager.post')
    def test_network_failures_and_authentication_errors(self, mock_post, mock_auth):
        """Test network failures and authentication errors."""
        # Arrange
        auth_manager = AuthManager(api_key="valid_key")
        http_manager = HTTPManager(
            base_url="https://openrouter.ai/api/v1", timeout=1)
        endpoint = KeysEndpoint(auth_manager, http_manager)

        mock_auth.return_value = {"Authorization": "Bearer valid_key"}
        mock_post.side_effect = APIError("Authentication failed")

        # Act & Assert
        with pytest.raises(APIError):
            endpoint.create(name="Test Key")

    @patch('openrouter_client.auth.AuthManager.get_auth_headers')
    @patch('openrouter_client.http.HTTPManager.post')
    def test_server_side_validation_errors(self, mock_post, mock_auth):
        """Test server-side validation errors."""
        # Arrange
        auth_manager = AuthManager(api_key="valid_provisioning_key")
        http_manager = HTTPManager(
            base_url="https://openrouter.ai/api/v1", timeout=1)
        endpoint = KeysEndpoint(auth_manager, http_manager)

        mock_auth.return_value = {
            "Authorization": "Bearer valid_provisioning_key"}
        mock_post.side_effect = APIError("Name cannot be empty")

        # Act & Assert
        with pytest.raises(APIError):
            endpoint.create(name="")


class TestKeysEndpointUpdate01NominalBehaviors:
    """Test nominal behaviors for KeysEndpoint.update method."""

    def test_successful_update_with_name_and_disabled_status(self):
        """Test successful update of API key name and disabled status."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.patch') as mock_patch:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer test-provisioning-key"}
            mock_response = Mock()
            mock_response.json.return_value = {
                "hash": "test_hash",
                "name": "Updated Key Name",
                "disabled": True,
                "updated_at": "2025-01-16T10:30:00Z"
            }
            mock_patch.return_value = mock_response

            auth_manager = AuthManager(api_key="valid_provisioning_key")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act
            result = endpoint.update("test_hash", name="Updated Key Name", disabled=True)

            # Assert
            assert isinstance(result, dict)
            assert result["name"] == "Updated Key Name"
            assert result["disabled"] is True

            # Verify authentication was called correctly
            mock_auth.assert_called_once_with(require_provisioning=True)
            mock_patch.assert_called_once_with(
                "keys/test_hash",
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json', 
                    'Authorization': 'Bearer test-provisioning-key'
                },
                json={"name": "Updated Key Name", "disabled": True}
            )

    def test_update_only_name(self):
        """Test updating only the name."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.patch') as mock_patch:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer test-provisioning-key"}
            mock_response = Mock()
            mock_response.json.return_value = {
                "hash": "test_hash",
                "name": "New Name Only"
            }
            mock_patch.return_value = mock_response

            auth_manager = AuthManager(api_key="valid_provisioning_key")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act
            result = endpoint.update("test_hash", name="New Name Only")

            # Assert
            assert result["name"] == "New Name Only"
            
            # Verify only name was sent
            mock_patch.assert_called_once_with(
                "keys/test_hash",
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json', 
                    'Authorization': 'Bearer test-provisioning-key'
                },
                json={"name": "New Name Only"}
            )


class TestKeysEndpointDelete01NominalBehaviors:
    """Test nominal behaviors for KeysEndpoint.delete method."""

    @patch('openrouter_client.auth.AuthManager.get_auth_headers')
    @patch('openrouter_client.http.HTTPManager.delete')
    def test_successful_deletion_of_existing_key(self, mock_delete, mock_auth):
        """Test successful deletion of existing API key."""
        # Arrange
        auth_manager = AuthManager(api_key="valid_provisioning_key")
        http_manager = HTTPManager(
            base_url="https://openrouter.ai/api/v1", timeout=1)
        endpoint = KeysEndpoint(auth_manager, http_manager)
        key_hash = "existing_key_123"

        expected_result = {
            "success": True,
            "message": "API key deleted successfully"
        }
        
        # Mock the auth headers
        mock_auth.return_value = {
            "Authorization": "Bearer valid_provisioning_key"
        }
        
        # Expected headers that will be passed to the delete method
        expected_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json", 
            "Authorization": "Bearer valid_provisioning_key"
        }
        
        mock_response = Mock()
        mock_response.json.return_value = expected_result
        mock_delete.return_value = mock_response

        # Act
        result = endpoint.delete(key_hash)

        # Assert
        assert isinstance(result, dict)
        assert result["success"] is True
        
        # Assert the delete method was called with correct URL and headers
        mock_delete.assert_called_once_with(
            f"keys/{key_hash}",  # URL without leading slash
            headers=expected_headers
        )


class TestKeysEndpointDelete02NegativeBehaviors:
    """Test negative behaviors for KeysEndpoint.delete method."""

    @pytest.mark.parametrize("invalid_key_hash", [
        "non_existent_key",
        "invalid_format_key",
        "already_deleted_key"
    ])
    @patch('openrouter_client.auth.AuthManager.get_auth_headers')
    @patch('openrouter_client.http.HTTPManager.delete')
    def test_deletion_with_invalid_key_hashes(self, mock_delete, mock_auth, invalid_key_hash):
        """Test deletion attempts with non-existent or invalid key_hash."""
        # Arrange
        auth_manager = AuthManager(api_key="valid_provisioning_key")
        http_manager = HTTPManager(
            base_url="https://openrouter.ai/api/v1", timeout=1)
        endpoint = KeysEndpoint(auth_manager, http_manager)

        mock_auth.return_value = {
            "Authorization": "Bearer valid_provisioning_key"}
        mock_delete.side_effect = APIError(f"Key {invalid_key_hash} not found")

        # Act & Assert
        with pytest.raises(APIError):
            endpoint.delete(invalid_key_hash)

    @patch('openrouter_client.auth.AuthManager.get_auth_headers')
    @patch('openrouter_client.http.HTTPManager.delete')
    def test_unauthorized_deletion_attempts(self, mock_delete, mock_auth):
        """Test unauthorized deletion attempts."""
        # Arrange
        auth_manager = AuthManager(api_key="invalid_key")
        http_manager = HTTPManager(
            base_url="https://openrouter.ai/api/v1", timeout=1)
        endpoint = KeysEndpoint(auth_manager, http_manager)

        mock_auth.return_value = {"Authorization": "Bearer invalid_key"}
        mock_delete.side_effect = APIError("Unauthorized")

        # Act & Assert
        with pytest.raises(APIError):
            endpoint.delete("some_key_hash")


class TestKeysEndpointDelete04ErrorHandlingBehaviors:
    """Test error handling behaviors for KeysEndpoint.delete method."""

    @patch('openrouter_client.auth.AuthManager.get_auth_headers')
    @patch('openrouter_client.http.HTTPManager.delete')
    def test_network_failures_and_authentication_errors(self, mock_delete, mock_auth):
        """Test network failures and authentication errors."""
        # Arrange
        auth_manager = AuthManager(api_key="valid_key")
        http_manager = HTTPManager(
            base_url="https://openrouter.ai/api/v1", timeout=1)
        endpoint = KeysEndpoint(auth_manager, http_manager)

        mock_auth.return_value = {"Authorization": "Bearer valid_key"}
        mock_delete.side_effect = APIError("Internal server error")

        # Act & Assert
        with pytest.raises(APIError):
            endpoint.delete("test_key_hash")


class TestKeysEndpointDelete05StateTransitionBehaviors:
    """Test state transition behaviors for KeysEndpoint.delete method."""

    @patch('openrouter_client.auth.AuthManager.get_auth_headers')
    @patch('openrouter_client.http.HTTPManager.post')
    @patch('openrouter_client.http.HTTPManager.delete')
    @patch('openrouter_client.http.HTTPManager.get')
    def test_key_state_transition_to_deleted(self, mock_get, mock_delete, mock_post, mock_auth):
        """Test API key state transition from active to deleted."""
        # Arrange
        auth_manager = AuthManager(api_key="valid_provisioning_key")
        http_manager = HTTPManager(
            base_url="https://openrouter.ai/api/v1", timeout=1)
        endpoint = KeysEndpoint(auth_manager, http_manager)

        mock_auth.return_value = {
            "Authorization": "Bearer valid_provisioning_key"}

        # Setup create mock
        created_key = {
            "hash": "test_key_for_deletion",
            "key": "sk-or-v1-testdeletion",
            "name": "Test Key for Deletion"
        }
        mock_post_response = Mock()
        mock_post_response.json.return_value = created_key
        mock_post.return_value = mock_post_response

        # Setup delete mock
        delete_result = {
            "success": True,
            "message": "Key deleted successfully"
        }
        mock_delete_response = Mock()
        mock_delete_response.json.return_value = delete_result
        mock_delete.return_value = mock_delete_response

        # Setup list mock - empty after deletion
        mock_get_response = Mock()
        mock_get_response.json.return_value = {"data": []}
        mock_get.return_value = mock_get_response

        # Create a key first
        created_key = endpoint.create(name="Test Key for Deletion")
        key_hash = created_key.get("hash")

        # Act
        delete_result = endpoint.delete(key_hash)

        # Assert
        assert isinstance(delete_result, dict)
        assert delete_result["success"] is True

        # Verify the key is no longer in the list
        keys_list = endpoint.list()
        assert len(keys_list["data"]) == 0


class TestKeysEndpointWarningBehaviors:
    """Test warning behaviors for create method."""

    @patch('openrouter_client.auth.AuthManager.get_auth_headers')
    @patch('openrouter_client.http.HTTPManager.post')
    def test_one_time_key_display_warning_for_create(self, mock_post, mock_auth, caplog):
        """Test one-time key display warning generation for create."""
        # Arrange
        auth_manager = AuthManager(api_key="valid_provisioning_key")
        http_manager = HTTPManager(
            base_url="https://openrouter.ai/api/v1", timeout=1)
        endpoint = KeysEndpoint(auth_manager, http_manager)

        mock_auth.return_value = {
            "Authorization": "Bearer valid_provisioning_key"}

        # Setup create mock
        created_key = {
            "hash": "warning_test_key",
            "key": "sk-or-v1-warning123",
            "name": "Warning Test Key"
        }

        mock_create_response = Mock()
        mock_create_response.json.return_value = created_key
        mock_post.return_value = mock_create_response

        with caplog.at_level(logging.WARNING):
            # Act
            result = endpoint.create(name="Warning Test Key")

            # Assert
            assert "key" in result
            # Check for warning in logs
            warning_found = any("API key will only be shown once" in record.message 
                              for record in caplog.records)
            assert warning_found


class TestKeysEndpointGetCurrent01NominalBehaviors:
    """Test nominal behaviors for KeysEndpoint.get_current method."""

    def test_successful_get_current_with_valid_api_key(self):
        """Test successful retrieval of current API key information."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.get') as mock_get:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer sk-or-v1-test-key"}
            expected_response = {
                "data": {
                    "label": "My Test Key",
                    "usage": 150.5,
                    "limit": 1000,
                    "is_free_tier": False,
                    "rate_limit": {
                        "requests": 200,
                        "interval": "10s"
                    }
                }
            }
            mock_response = Mock()
            mock_response.json.return_value = expected_response
            mock_get.return_value = mock_response

            auth_manager = AuthManager(api_key="sk-or-v1-test-key")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act
            result = endpoint.get_current()

            # Assert
            assert isinstance(result, dict)
            assert "data" in result
            assert result["data"]["label"] == "My Test Key"
            assert result["data"]["usage"] == 150.5
            assert result["data"]["limit"] == 1000
            assert result["data"]["is_free_tier"] is False
            assert result["data"]["rate_limit"]["requests"] == 200
            assert result["data"]["rate_limit"]["interval"] == "10s"

            # Verify authentication was called correctly (without provisioning requirement)
            mock_auth.assert_called_once_with(require_provisioning=False)
            mock_get.assert_called_once_with(
                "auth/key",
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json', 
                    'Authorization': 'Bearer sk-or-v1-test-key'
                }
            )

    def test_get_current_with_unlimited_key(self):
        """Test get_current with an unlimited API key (null limit)."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.get') as mock_get:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer sk-or-v1-unlimited"}
            expected_response = {
                "data": {
                    "label": "Unlimited Key",
                    "usage": 5000.0,
                    "limit": None,  # Unlimited
                    "is_free_tier": False,
                    "rate_limit": {
                        "requests": 1000,
                        "interval": "10s"
                    }
                }
            }
            mock_response = Mock()
            mock_response.json.return_value = expected_response
            mock_get.return_value = mock_response

            auth_manager = AuthManager(api_key="sk-or-v1-unlimited")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act
            result = endpoint.get_current()

            # Assert
            assert isinstance(result, dict)
            assert result["data"]["limit"] is None
            assert result["data"]["usage"] == 5000.0


class TestKeysEndpointGetCurrent02NegativeBehaviors:
    """Test negative behaviors for KeysEndpoint.get_current method."""

    def test_get_current_with_invalid_api_key(self):
        """Test get_current fails with invalid API key."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.get') as mock_get:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer invalid-key"}
            mock_get.side_effect = APIError("Invalid API key")

            auth_manager = AuthManager(api_key="invalid-key")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act & Assert
            with pytest.raises(APIError):
                endpoint.get_current()

    def test_get_current_with_expired_api_key(self):
        """Test get_current fails with expired API key."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.get') as mock_get:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer sk-or-v1-expired"}
            mock_get.side_effect = APIError("API key has expired")

            auth_manager = AuthManager(api_key="sk-or-v1-expired")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act & Assert
            with pytest.raises(APIError):
                endpoint.get_current()


class TestKeysEndpointGetCurrent03BoundaryBehaviors:
    """Test boundary behaviors for KeysEndpoint.get_current method."""

    def test_get_current_with_zero_usage(self):
        """Test get_current with a key that has zero usage."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.get') as mock_get:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer sk-or-v1-new"}
            expected_response = {
                "data": {
                    "label": "Brand New Key",
                    "usage": 0,
                    "limit": 100,
                    "is_free_tier": True,
                    "rate_limit": {
                        "requests": 50,
                        "interval": "10s"
                    }
                }
            }
            mock_response = Mock()
            mock_response.json.return_value = expected_response
            mock_get.return_value = mock_response

            auth_manager = AuthManager(api_key="sk-or-v1-new")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act
            result = endpoint.get_current()

            # Assert
            assert result["data"]["usage"] == 0
            assert result["data"]["is_free_tier"] is True

    def test_get_current_at_usage_limit(self):
        """Test get_current with a key at its usage limit."""
        with patch('openrouter_client.auth.AuthManager.get_auth_headers') as mock_auth, \
                patch('openrouter_client.http.HTTPManager.get') as mock_get:

            # Arrange
            mock_auth.return_value = {
                "Authorization": "Bearer sk-or-v1-maxed"}
            expected_response = {
                "data": {
                    "label": "Maxed Out Key",
                    "usage": 1000,
                    "limit": 1000,
                    "is_free_tier": False,
                    "rate_limit": {
                        "requests": 200,
                        "interval": "10s"
                    }
                }
            }
            mock_response = Mock()
            mock_response.json.return_value = expected_response
            mock_get.return_value = mock_response

            auth_manager = AuthManager(api_key="sk-or-v1-maxed")
            http_manager = HTTPManager(
                base_url="https://openrouter.ai/api/v1", timeout=1)
            endpoint = KeysEndpoint(auth_manager, http_manager)

            # Act
            result = endpoint.get_current()

            # Assert
            assert result["data"]["usage"] == result["data"]["limit"]
            assert result["data"]["usage"] == 1000


class TestKeysEndpointGetCurrent04ErrorHandlingBehaviors:
    """Test error handling behaviors for KeysEndpoint.get_current method."""

    @patch('openrouter_client.auth.AuthManager.get_auth_headers')
    @patch('openrouter_client.http.HTTPManager.get')
    def test_network_failure_during_get_current(self, mock_get, mock_auth):
        """Test network failure during get_current request."""
        # Arrange
        auth_manager = AuthManager(api_key="sk-or-v1-valid")
        http_manager = HTTPManager(
            base_url="https://openrouter.ai/api/v1", timeout=1)
        endpoint = KeysEndpoint(auth_manager, http_manager)

        mock_auth.return_value = {"Authorization": "Bearer sk-or-v1-valid"}
        mock_get.side_effect = APIError("Network timeout")

        # Act & Assert
        with pytest.raises(APIError):
            endpoint.get_current()

    @patch('openrouter_client.auth.AuthManager.get_auth_headers')
    @patch('openrouter_client.http.HTTPManager.get')
    def test_malformed_response_from_get_current(self, mock_get, mock_auth):
        """Test handling of malformed response from get_current."""
        # Arrange
        auth_manager = AuthManager(api_key="sk-or-v1-valid")
        http_manager = HTTPManager(
            base_url="https://openrouter.ai/api/v1", timeout=1)
        endpoint = KeysEndpoint(auth_manager, http_manager)

        mock_auth.return_value = {"Authorization": "Bearer sk-or-v1-valid"}
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        # Act & Assert
        with pytest.raises(ValueError):
            endpoint.get_current()


class TestKeysEndpointGetCurrent05StateTransitionBehaviors:
    """Test state transition behaviors for KeysEndpoint.get_current method."""

    @patch('openrouter_client.auth.AuthManager.get_auth_headers')
    @patch('openrouter_client.http.HTTPManager.get')
    @patch('openrouter_client.http.HTTPManager.post')
    def test_get_current_reflects_usage_changes(self, mock_post, mock_get, mock_auth):
        """Test that get_current reflects usage changes after API calls."""
        # Arrange
        auth_manager = AuthManager(api_key="sk-or-v1-tracking")
        http_manager = HTTPManager(
            base_url="https://openrouter.ai/api/v1", timeout=1)
        endpoint = KeysEndpoint(auth_manager, http_manager)

        # Initial state
        initial_response = {
            "data": {
                "label": "Usage Tracking Key",
                "usage": 100,
                "limit": 1000,
                "is_free_tier": False,
                "rate_limit": {
                    "requests": 200,
                    "interval": "10s"
                }
            }
        }
        
        # After usage state
        after_usage_response = {
            "data": {
                "label": "Usage Tracking Key",
                "usage": 150,  # Usage increased
                "limit": 1000,
                "is_free_tier": False,
                "rate_limit": {
                    "requests": 200,
                    "interval": "10s"
                }
            }
        }

        mock_auth.return_value = {"Authorization": "Bearer sk-or-v1-tracking"}
        
        # Set up responses
        mock_response_1 = Mock()
        mock_response_1.json.return_value = initial_response
        mock_response_2 = Mock()
        mock_response_2.json.return_value = after_usage_response
        mock_get.side_effect = [mock_response_1, mock_response_2]

        # Act
        initial_state = endpoint.get_current()
        # Simulate some API usage here
        after_state = endpoint.get_current()

        # Assert
        assert initial_state["data"]["usage"] == 100
        assert after_state["data"]["usage"] == 150
        assert after_state["data"]["usage"] > initial_state["data"]["usage"]