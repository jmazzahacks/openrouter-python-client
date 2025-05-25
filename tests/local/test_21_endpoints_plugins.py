import pytest
from unittest.mock import Mock, patch
import json

from openrouter_client.endpoints.plugins import PluginsEndpoint
from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager
from openrouter_client.exceptions import APIError
from pydantic import ValidationError


class Test_PluginsEndpoint_Init_01_NominalBehaviors:
    """Test nominal behaviors for PluginsEndpoint initialization."""
    
    def test_successful_initialization_with_valid_managers(self):
        """Test successful initialization with valid AuthManager and HTTPManager instances."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Act
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint.auth_manager is auth_manager
        assert endpoint.http_manager is http_manager
        assert endpoint.endpoint_path == 'plugins'


class Test_PluginsEndpoint_Init_02_NegativeBehaviors:
    """Test negative behaviors for PluginsEndpoint initialization."""
    
    @pytest.mark.parametrize("auth_manager,http_manager", [
        (None, Mock(spec=HTTPManager)),
        (Mock(spec=AuthManager), None),
        (None, None),
        ("invalid_auth", Mock(spec=HTTPManager)),
        (Mock(spec=AuthManager), "invalid_http"),
    ])
    def test_initialization_with_invalid_managers(self, auth_manager, http_manager):
        """Test initialization fails with None or invalid manager types."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError):
            PluginsEndpoint(auth_manager, http_manager)


class Test_PluginsEndpoint_Init_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for PluginsEndpoint initialization."""
    
    @patch('openrouter_client.endpoints.plugins.BaseEndpoint.__init__')
    def test_handling_parent_class_initialization_exception(self, mock_base_init):
        """Test handling exceptions during parent class initialization."""
        # Arrange
        mock_base_init.side_effect = Exception("Parent initialization failed")
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Act & Assert
        with pytest.raises(Exception, match="Parent initialization failed"):
            PluginsEndpoint(auth_manager, http_manager)


class Test_PluginsEndpoint_List_01_NominalBehaviors:
    """Test nominal behaviors for PluginsEndpoint list method."""
    
    def test_successful_plugin_list_retrieval(self):
        """Test successful plugin list retrieval with valid authentication."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        expected_plugins = [
            {"id": "plugin1", "name": "Test Plugin 1"},
            {"id": "plugin2", "name": "Test Plugin 2"}
        ]
        
        mock_response = Mock()
        mock_response.json.return_value = expected_plugins
        http_manager.get.return_value = mock_response
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act
        result = endpoint.list()
        
        # Assert
        assert result == expected_plugins
        http_manager.get.assert_called_once_with(
            endpoint="plugins",
            headers={"Content-Type": "application/json", "Accept": "application/json", "Authorization": "Bearer test-token"}
        )


class Test_PluginsEndpoint_List_02_NegativeBehaviors:
    """Test negative behaviors for PluginsEndpoint list method."""
    
    def test_authentication_failures_unauthorized_access(self):
        """Test authentication failures resulting in unauthorized access."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer invalid-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.get.side_effect = APIError("Unauthorized", status_code=401)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Unauthorized"):
            endpoint.list()


class Test_PluginsEndpoint_List_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for PluginsEndpoint list method."""
    
    def test_network_connectivity_failures(self):
        """Test network connectivity failures during HTTP requests."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.get.side_effect = APIError("Network connection failed", status_code=503)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Network connection failed"):
            endpoint.list()
    
    def test_json_parsing_failures_malformed_response(self):
        """Test JSON parsing failures from malformed server responses."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        http_manager.get.return_value = mock_response
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(json.JSONDecodeError):
            endpoint.list()


class Test_PluginsEndpoint_Get_01_NominalBehaviors:
    """Test nominal behaviors for PluginsEndpoint get method."""
    
    @pytest.mark.parametrize("plugin_id,expected_plugin", [
        ("plugin1", {"id": "plugin1", "name": "Test Plugin 1", "version": "1.0.0"}),
        ("plugin2", {"id": "plugin2", "name": "Test Plugin 2", "version": "2.1.0"}),
        ("complex-plugin-id", {"id": "complex-plugin-id", "name": "Complex Plugin", "config": {"enabled": True}})
    ])
    def test_successful_plugin_detail_retrieval(self, plugin_id, expected_plugin):
        """Test successful plugin detail retrieval with valid plugin_id."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        mock_response = Mock()
        mock_response.json.return_value = expected_plugin
        http_manager.get.return_value = mock_response
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act
        result = endpoint.get(plugin_id)
        
        # Assert
        assert result == expected_plugin
        http_manager.get.assert_called_once_with(
            endpoint=f"plugins/{plugin_id}",
            headers={"Content-Type": "application/json", "Accept": "application/json", "Authorization": "Bearer test-token"}
        )


class Test_PluginsEndpoint_Get_02_NegativeBehaviors:
    """Test negative behaviors for PluginsEndpoint get method."""
    
    @pytest.mark.parametrize("plugin_id", [
        "nonexistent_plugin",
        "invalid-plugin-id",
        "deleted_plugin"
    ])
    def test_requests_for_nonexistent_plugin_ids(self, plugin_id):
        """Test requests for non-existent plugin_id values."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.get.side_effect = APIError("Plugin not found", status_code=404)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Plugin not found"):
            endpoint.get(plugin_id)
    
    def test_authentication_failures_for_restricted_plugin_access(self):
        """Test authentication failures for restricted plugin access."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.get.side_effect = APIError("Forbidden", status_code=403)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Forbidden"):
            endpoint.get("restricted_plugin")


class Test_PluginsEndpoint_Get_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for PluginsEndpoint get method."""
    
    def test_network_failures_during_plugin_detail_requests(self):
        """Test network failures during plugin detail requests."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.get.side_effect = APIError("Connection timeout", status_code=504)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Connection timeout"):
            endpoint.get("plugin1")
    
    def test_json_parsing_exceptions_from_server_responses(self):
        """Test JSON parsing exceptions from server responses."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Malformed JSON", "", 5)
        http_manager.get.return_value = mock_response
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(json.JSONDecodeError):
            endpoint.get("plugin1")


class Test_PluginsEndpoint_Register_01_NominalBehaviors:
    """Test nominal behaviors for PluginsEndpoint register method."""
    
    @pytest.mark.parametrize("manifest_url,auth,kwargs,expected_response", [
        (
            "https://example.com/plugin.json",
            None,
            {},
            {"id": "new_plugin", "status": "registered"}
        ),
        (
            "https://example.com/plugin.json",
            {"api_key": "secret"},
            {},
            {"id": "auth_plugin", "status": "registered"}
        ),
        (
            "https://example.com/plugin.json",
            {"api_key": "secret"},
            {"description": "Test plugin", "version": "1.0.0"},
            {"id": "complex_plugin", "status": "registered", "version": "1.0.0"}
        ),
    ])
    def test_successful_plugin_registration(self, manifest_url, auth, kwargs, expected_response):
        """Test successful plugin registration with valid manifest_url."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        mock_response = Mock()
        mock_response.json.return_value = expected_response
        http_manager.post.return_value = mock_response
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act
        result = endpoint.register(manifest_url, auth, **kwargs)
        
        # Assert
        assert result == expected_response
        # Build expected data
        expected_data = {"manifest_url": manifest_url}
        if auth is not None:
            expected_data["auth"] = auth
        expected_data.update(kwargs)
        
        http_manager.post.assert_called_once_with(
            endpoint="plugins",
            headers={"Content-Type": "application/json", "Accept": "application/json", "Authorization": "Bearer test-token"},
            json=expected_data
        )


class Test_PluginsEndpoint_Register_02_NegativeBehaviors:
    """Test negative behaviors for PluginsEndpoint register method."""
    
    @pytest.mark.parametrize("manifest_url,error_message,status_code", [
        ("https://invalid.example.com/nonexistent.json", "Manifest not found", 404),
        ("invalid-url", "Invalid URL format", 400),
        ("https://malformed.example.com/plugin.json", "Invalid manifest format", 422),
    ])
    def test_registration_with_invalid_inaccessible_manifest_url(self, manifest_url, error_message, status_code):
        """Test registration with invalid or inaccessible manifest_url."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.post.side_effect = APIError(error_message, status_code=status_code)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match=error_message):
            endpoint.register(manifest_url)
    
    def test_duplicate_plugin_registration_attempts(self):
        """Test duplicate plugin registration attempts."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.post.side_effect = APIError("Plugin already registered", status_code=409)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Plugin already registered"):
            endpoint.register("https://example.com/existing-plugin.json")


class Test_PluginsEndpoint_Register_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for PluginsEndpoint register method."""
    
    def test_network_failures_when_accessing_manifest_urls(self):
        """Test network failures when accessing manifest URLs."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.post.side_effect = APIError("Network unreachable", status_code=503)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Network unreachable"):
            endpoint.register("https://unreachable.example.com/plugin.json")
    
    def test_server_side_registration_processing_errors(self):
        """Test server-side registration processing errors."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.post.side_effect = APIError("Internal server error during registration", status_code=500)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Internal server error during registration"):
            endpoint.register("https://example.com/plugin.json")


class Test_PluginsEndpoint_Unregister_01_NominalBehaviors:
    """Test nominal behaviors for PluginsEndpoint unregister method."""
    
    @pytest.mark.parametrize("plugin_id", [
        "plugin1",
        "test-plugin",
        "complex_plugin_id_123"
    ])
    def test_successful_plugin_removal(self, plugin_id):
        """Test successful plugin removal with valid plugin_id."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        expected_response = {"status": "unregistered", "plugin_id": plugin_id}
        mock_response = Mock()
        mock_response.json.return_value = expected_response
        http_manager.delete.return_value = mock_response
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act
        result = endpoint.unregister(plugin_id)
        
        # Assert
        assert result == expected_response
        http_manager.delete.assert_called_once_with(
            endpoint=f"plugins/{plugin_id}",
            headers={"Content-Type": "application/json", "Accept": "application/json", "Authorization": "Bearer test-token"}
        )


class Test_PluginsEndpoint_Unregister_02_NegativeBehaviors:
    """Test negative behaviors for PluginsEndpoint unregister method."""
    
    @pytest.mark.parametrize("plugin_id", [
        "nonexistent_plugin",
        "already_deleted_plugin",
        "invalid-id-format"
    ])
    def test_unregistration_attempts_for_nonexistent_plugins(self, plugin_id):
        """Test unregistration attempts for non-existent plugins."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.delete.side_effect = APIError("Plugin not found", status_code=404)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Plugin not found"):
            endpoint.unregister(plugin_id)


class Test_PluginsEndpoint_Unregister_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for PluginsEndpoint unregister method."""
    
    def test_network_failures_during_unregistration_requests(self):
        """Test network failures during unregistration requests."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.delete.side_effect = APIError("Connection failed", status_code=503)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Connection failed"):
            endpoint.unregister("plugin1")
    
    def test_server_side_deletion_processing_errors(self):
        """Test server-side deletion processing errors."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.delete.side_effect = APIError("Internal error during deletion", status_code=500)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Internal error during deletion"):
            endpoint.unregister("plugin1")


class Test_PluginsEndpoint_Invoke_01_NominalBehaviors:
    """Test nominal behaviors for PluginsEndpoint invoke method."""
    
    @pytest.mark.parametrize("plugin_id,action,parameters,kwargs,expected_response", [
        (
            "plugin1",
            "execute",
            None,
            {},
            {"status": "success", "result": {"output": "executed"}}
        ),
        (
            "plugin2",
            "process",
            {"input": "test"},
            {},
            {"status": "success", "result": {"processed": "test"}}
        ),
        (
            "plugin3",
            "analyze",
            {"data": [1, 2, 3], "mode": "deep"},
            {"timeout": 30, "priority": "high"},
            {"status": "success", "result": {"analysis": "complete"}}
        ),
    ])
    def test_successful_plugin_action_execution(self, plugin_id, action, parameters, kwargs, expected_response):
        """Test successful plugin action execution with valid plugin_id and action."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        mock_response = Mock()
        mock_response.json.return_value = expected_response
        http_manager.post.return_value = mock_response
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act
        result = endpoint.invoke(plugin_id, action, parameters, **kwargs)
        
        # Assert
        assert result == expected_response
        # Build expected data
        expected_data = {"action": action}
        if parameters is not None:
            expected_data["parameters"] = parameters
        expected_data.update(kwargs)
        
        http_manager.post.assert_called_once_with(
            endpoint=f"plugins/{plugin_id}/invoke",
            headers={"Content-Type": "application/json", "Accept": "application/json", "Authorization": "Bearer test-token"},
            json=expected_data
        )


class Test_PluginsEndpoint_Invoke_02_NegativeBehaviors:
    """Test negative behaviors for PluginsEndpoint invoke method."""
    
    @pytest.mark.parametrize("plugin_id", [
        "nonexistent_plugin",
        "inactive_plugin",
        "unregistered_plugin"
    ])
    def test_invocation_with_nonexistent_plugin_id(self, plugin_id):
        """Test invocation with non-existent plugin_id."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.post.side_effect = APIError("Plugin not found", status_code=404)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Plugin not found"):
            endpoint.invoke(plugin_id, "execute")
    
    @pytest.mark.parametrize("action", [
        "undefined_action",
        "deprecated_action",
        "invalid-action-name"
    ])
    def test_requests_for_undefined_invalid_action_names(self, action):
        """Test requests for undefined or invalid action names."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.post.side_effect = APIError("Action not found", status_code=400)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Action not found"):
            endpoint.invoke("plugin1", action)


class Test_PluginsEndpoint_Invoke_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for PluginsEndpoint invoke method."""
    
    def test_plugin_execution_failures_and_error_propagation(self):
        """Test plugin execution failures and error propagation."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.post.side_effect = APIError("Plugin execution failed", status_code=500)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Plugin execution failed"):
            endpoint.invoke("plugin1", "execute")
    
    def test_network_failures_during_plugin_communication(self):
        """Test network failures during plugin communication."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.post.side_effect = APIError("Network timeout", status_code=504)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match="Network timeout"):
            endpoint.invoke("plugin1", "execute")
    
    @pytest.mark.parametrize("parameters,error_message", [
        ({"invalid": "format"}, "Parameter validation failed"),
        ({"missing_required": None}, "Required parameter missing"),
        ({"too_large": "x" * 1000}, "Parameter too large"),
    ])
    def test_parameter_validation_failures_within_plugin_execution(self, parameters, error_message):
        """Test parameter validation failures within plugin execution."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        auth_manager.get_auth_headers.return_value = {"Authorization": "Bearer test-token"}
        http_manager = Mock(spec=HTTPManager)
        
        http_manager.post.side_effect = APIError(error_message, status_code=422)
        
        endpoint = PluginsEndpoint(auth_manager, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError, match=error_message):
            endpoint.invoke("plugin1", "execute", parameters)
