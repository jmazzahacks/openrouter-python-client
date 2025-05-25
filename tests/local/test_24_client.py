import pytest
from unittest.mock import Mock, patch
import logging
from contextlib import contextmanager

from openrouter_client.client import OpenRouterClient
from openrouter_client.exceptions import APIError, AuthenticationError


@contextmanager
def mock_all_endpoints():
    """Context manager to mock all endpoint classes to avoid Pydantic validation errors."""
    with patch.multiple(
        'openrouter_client.client',
        CompletionsEndpoint=Mock(),
        ChatEndpoint=Mock(),
        ModelsEndpoint=Mock(),
        ImagesEndpoint=Mock(),
        GenerationsEndpoint=Mock(),
        CreditsEndpoint=Mock(),
        KeysEndpoint=Mock(),
        PluginsEndpoint=Mock(),
        WebEndpoint=Mock()
    ):
        yield

class Test_OpenRouterClient_Init_01_NominalBehaviors:
    """Test nominal behaviors for OpenRouterClient.__init__ method."""
    
    @pytest.mark.parametrize("api_key,provisioning_key,base_url,org_id,ref_id,timeout,retries", [
        ("test-key", None, "https://openrouter.ai/api/v1", None, None, 60.0, 3),
        ("test-key", "prov-key", "https://openrouter.ai/api/v1", "org-123", "ref-456", 60.0, 3),
        ("test-key", None, "https://custom.api.com/v1", "org-123", None, 30.0, 5),
        (None, None, "https://openrouter.ai/api/v1", None, "ref-789", 120.0, 1),
    ])
    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_initialize_client_with_various_valid_parameter_combinations(
        self, mock_logging, mock_http_manager, mock_auth_manager,
        api_key, provisioning_key, base_url, org_id, ref_id, timeout, retries
    ):
        """Test successful client initialization with different valid parameter combinations."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        mock_auth_instance = Mock()
        mock_auth_manager.return_value = mock_auth_instance
        mock_http_instance = Mock()
        mock_http_manager.return_value = mock_http_instance
        
        # Mock endpoint classes
        with patch.multiple(
            'openrouter_client.client',
            CompletionsEndpoint=Mock(),
            ChatEndpoint=Mock(),
            ModelsEndpoint=Mock(),
            ImagesEndpoint=Mock(),
            GenerationsEndpoint=Mock(),
            CreditsEndpoint=Mock(),
            KeysEndpoint=Mock(),
            PluginsEndpoint=Mock(),
            WebEndpoint=Mock()
        ):
            # Act
            client = OpenRouterClient(
                api_key=api_key,
                provisioning_api_key=provisioning_key,
                base_url=base_url,
                organization_id=org_id,
                reference_id=ref_id,
                timeout=timeout,
                retries=retries
            )
            
            # Assert
            assert client.auth_manager == mock_auth_instance
            assert client.http_manager == mock_http_instance
            assert client.base_url == base_url
            assert isinstance(client._context_lengths, dict)
            assert len(client._context_lengths) == 0
            mock_auth_manager.assert_called_once_with(
                api_key=api_key,
                provisioning_api_key=provisioning_key,
                organization_id=org_id,
                reference_id=ref_id,
                secrets_manager=None
            )
            mock_http_manager.assert_called_once()


class Test_OpenRouterClient_Init_02_NegativeBehaviors:
    """Test negative behaviors for OpenRouterClient.__init__ method."""
    
    @pytest.mark.parametrize("invalid_params,expected_exception", [
        ({"base_url": "not-a-url"}, Exception),
        ({"timeout": -1}, Exception),
        ({"retries": -1}, Exception),
        ({"api_key": 123}, Exception),
        ({"organization_id": 456}, Exception),
    ])
    @patch('openrouter_client.client.AuthManager')
    def test_initialize_client_with_invalid_parameter_values_or_types(
        self, mock_auth_manager, invalid_params, expected_exception
    ):
        """Test initialization fails with invalid parameter values or types."""
        # Arrange
        mock_auth_manager.side_effect = expected_exception("Invalid parameter")
        
        # Act & Assert
        with pytest.raises(expected_exception):
            OpenRouterClient(**invalid_params)


class Test_OpenRouterClient_Init_03_BoundaryBehaviors:
    """Test boundary behaviors for OpenRouterClient.__init__ method."""
    
    @pytest.mark.parametrize("boundary_params", [
        {"api_key": ""},
        {"organization_id": ""},
        {"reference_id": ""},
        {"organization_id": "x" * 1000},
        {"reference_id": "y" * 1000},
        {"timeout": 0.1},
        {"timeout": 3600.0},
        {"retries": 0},
        {"retries": 100},
    ])
    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_initialize_client_with_boundary_values_for_all_parameter_types(
        self, mock_logging, mock_http_manager, mock_auth_manager, boundary_params
    ):
        """Test initialization with boundary values for different parameter types."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        mock_auth_instance = Mock()
        mock_auth_manager.return_value = mock_auth_instance
        mock_http_instance = Mock()
        mock_http_manager.return_value = mock_http_instance
        
        with patch.multiple(
            'openrouter_client.client',
            CompletionsEndpoint=Mock(),
            ChatEndpoint=Mock(),
            ModelsEndpoint=Mock(),
            ImagesEndpoint=Mock(),
            GenerationsEndpoint=Mock(),
            CreditsEndpoint=Mock(),
            KeysEndpoint=Mock(),
            PluginsEndpoint=Mock(),
            WebEndpoint=Mock()
        ):
            # Act
            client = OpenRouterClient(**boundary_params)
            
            # Assert
            assert client is not None
            assert hasattr(client, '_context_lengths')


class Test_OpenRouterClient_Init_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for OpenRouterClient.__init__ method."""
    
    @pytest.mark.parametrize("failure_component,exception_type", [
        ("auth_manager", AuthenticationError),
        ("http_manager", Exception),
        ("endpoints", Exception),
    ])
    @patch('openrouter_client.client.configure_logging')
    def test_handle_initialization_failures_from_dependency_components(
        self, mock_logging, failure_component, exception_type
    ):
        """Test handling of initialization failures from dependency components."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        
        if failure_component == "auth_manager":
            with patch('openrouter_client.client.AuthManager', side_effect=exception_type("Auth failed")):
                # Act & Assert
                with pytest.raises(exception_type):
                    OpenRouterClient(api_key="test")
        elif failure_component == "http_manager":
            with patch('openrouter_client.client.AuthManager'), \
                 patch('openrouter_client.client.HTTPManager', side_effect=exception_type("HTTP failed")):
                # Act & Assert
                with pytest.raises(exception_type):
                    OpenRouterClient(api_key="test")
        elif failure_component == "endpoints":
            with patch('openrouter_client.client.AuthManager'), \
                 patch('openrouter_client.client.HTTPManager'), \
                 patch('openrouter_client.client.CompletionsEndpoint', side_effect=exception_type("Endpoint failed")):
                # Act & Assert
                with pytest.raises(exception_type):
                    OpenRouterClient(api_key="test")


class Test_OpenRouterClient_Init_05_StateTransitionBehaviors:
    """Test state transition behaviors for OpenRouterClient.__init__ method."""
    
    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_verify_proper_initialization_of_all_client_state_components(
        self, mock_logging, mock_http_manager, mock_auth_manager
    ):
        """Test that client properly establishes all internal state during initialization."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        mock_auth_instance = Mock()
        mock_auth_manager.return_value = mock_auth_instance
        mock_http_instance = Mock()
        mock_http_manager.return_value = mock_http_instance
        
        endpoint_mocks = {}
        endpoint_names = [
            'CompletionsEndpoint', 'ChatEndpoint', 'ModelsEndpoint', 'ImagesEndpoint',
            'GenerationsEndpoint', 'CreditsEndpoint', 'KeysEndpoint', 'PluginsEndpoint', 'WebEndpoint'
        ]
        
        for name in endpoint_names:
            endpoint_mocks[name] = Mock()
            
        with patch.multiple('openrouter_client.client', **endpoint_mocks):
            # Act
            client = OpenRouterClient(api_key="test", log_level=logging.DEBUG)
            
            # Assert - Verify all state components are properly initialized
            assert client.auth_manager == mock_auth_instance
            assert client.http_manager == mock_http_instance
            assert client.logger == mock_logger
            assert isinstance(client._context_lengths, dict)
            assert len(client._context_lengths) == 0
            assert hasattr(client, 'completions')
            assert hasattr(client, 'chat')
            assert hasattr(client, 'models')
            assert hasattr(client, 'images')
            assert hasattr(client, 'generations')
            assert hasattr(client, 'credits')
            assert hasattr(client, 'keys')
            assert hasattr(client, 'plugins')
            assert hasattr(client, 'web')


class Test_OpenRouterClient_InitializeEndpoints_01_NominalBehaviors:
    """Test nominal behaviors for OpenRouterClient._initialize_endpoints method."""
    
    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_successfully_initialize_all_endpoint_handlers_with_shared_managers(
        self, mock_logging, mock_http_manager, mock_auth_manager
    ):
        """Test successful initialization of all endpoint handlers with shared managers."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        mock_auth_instance = Mock()
        mock_auth_manager.return_value = mock_auth_instance
        mock_http_instance = Mock()
        mock_http_manager.return_value = mock_http_instance
        
        endpoint_mocks = {}
        endpoint_instances = {}
        endpoint_names = [
            'CompletionsEndpoint', 'ChatEndpoint', 'ModelsEndpoint', 'ImagesEndpoint',
            'GenerationsEndpoint', 'CreditsEndpoint', 'KeysEndpoint', 'PluginsEndpoint', 'WebEndpoint'
        ]
        
        for name in endpoint_names:
            instance = Mock()
            endpoint_mocks[name] = Mock(return_value=instance)
            endpoint_instances[name] = instance
            
        with patch.multiple('openrouter_client.client', **endpoint_mocks):
            # Act
            client = OpenRouterClient(api_key="test")
            
            # Assert - Verify all endpoints initialized with shared managers
            for name in endpoint_names:
                endpoint_mocks[name].assert_called_once_with(
                    auth_manager=mock_auth_instance,
                    http_manager=mock_http_instance
                )
            
            # Verify debug logging occurred
            mock_logger.debug.assert_called_with("All endpoint handlers initialized successfully")


class Test_OpenRouterClient_InitializeEndpoints_02_NegativeBehaviors:
    """Test negative behaviors for OpenRouterClient._initialize_endpoints method."""
    
    @pytest.mark.parametrize("invalid_manager_scenario", [
        "auth_manager_none",
        "http_manager_none",
        "endpoint_creation_failure"
    ])
    @patch('openrouter_client.client.configure_logging')
    def test_handle_invalid_manager_dependencies_during_endpoint_creation(
        self, mock_logging, invalid_manager_scenario
    ):
        """Test handling of invalid manager dependencies during endpoint creation."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        
        if invalid_manager_scenario == "auth_manager_none":
            with patch('openrouter_client.client.AuthManager', return_value=None), \
                 patch('openrouter_client.client.HTTPManager'):
                # Act & Assert
                with pytest.raises(Exception):
                    OpenRouterClient(api_key="test")
                    
        elif invalid_manager_scenario == "http_manager_none":
            with patch('openrouter_client.client.AuthManager'), \
                 patch('openrouter_client.client.HTTPManager', return_value=None):
                # Act & Assert
                with pytest.raises(Exception):
                    OpenRouterClient(api_key="test")
                    
        elif invalid_manager_scenario == "endpoint_creation_failure":
            with patch('openrouter_client.client.AuthManager'), \
                 patch('openrouter_client.client.HTTPManager'), \
                 patch('openrouter_client.client.CompletionsEndpoint', side_effect=Exception("Endpoint failed")):
                # Act & Assert
                with pytest.raises(Exception):
                    OpenRouterClient(api_key="test")


class Test_OpenRouterClient_InitializeEndpoints_03_BoundaryBehaviors:
    """Test boundary behaviors for OpenRouterClient._initialize_endpoints method."""
    
    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_initialize_endpoints_when_managers_have_minimal_valid_configuration(
        self, mock_logging, mock_http_manager, mock_auth_manager
    ):
        """Test endpoint initialization when managers have minimal valid configuration."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        
        # Create minimal valid managers
        minimal_auth = Mock()
        minimal_auth.api_key = "minimal"
        mock_auth_manager.return_value = minimal_auth
        
        minimal_http = Mock()
        minimal_http.base_url = "https://api.test"
        mock_http_manager.return_value = minimal_http
        
        endpoint_mocks = {}
        endpoint_names = [
            'CompletionsEndpoint', 'ChatEndpoint', 'ModelsEndpoint', 'ImagesEndpoint',
            'GenerationsEndpoint', 'CreditsEndpoint', 'KeysEndpoint', 'PluginsEndpoint', 'WebEndpoint'
        ]
        
        for name in endpoint_names:
            endpoint_mocks[name] = Mock()
            
        with patch.multiple('openrouter_client.client', **endpoint_mocks):
            # Act
            client = OpenRouterClient(api_key="minimal")
            
            # Assert
            assert client.auth_manager == minimal_auth
            assert client.http_manager == minimal_http
            for name in endpoint_names:
                endpoint_mocks[name].assert_called_once()


class Test_OpenRouterClient_InitializeEndpoints_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for OpenRouterClient._initialize_endpoints method."""
    
    @pytest.mark.parametrize("failing_endpoint", [
        'CompletionsEndpoint', 'ChatEndpoint', 'ModelsEndpoint', 'ImagesEndpoint',
        'GenerationsEndpoint', 'CreditsEndpoint', 'KeysEndpoint', 'PluginsEndpoint', 'WebEndpoint'
    ])
    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_handle_and_propagate_endpoint_creation_failures(
        self, mock_logging, mock_http_manager, mock_auth_manager, failing_endpoint
    ):
        """Test handling and propagation of endpoint creation failures."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        mock_auth_manager.return_value = Mock()
        mock_http_manager.return_value = Mock()
        
        patch_dict = {}
        for endpoint in ['CompletionsEndpoint', 'ChatEndpoint', 'ModelsEndpoint', 'ImagesEndpoint',
                        'GenerationsEndpoint', 'CreditsEndpoint', 'KeysEndpoint', 'PluginsEndpoint', 'WebEndpoint']:
            if endpoint == failing_endpoint:
                patch_dict[endpoint] = Mock(side_effect=Exception(f"{endpoint} failed"))
            else:
                patch_dict[endpoint] = Mock()
        
        with patch.multiple('openrouter_client.client', **patch_dict):
            # Act & Assert
            with pytest.raises(Exception) as exc_info:
                OpenRouterClient(api_key="test")
            
            assert f"{failing_endpoint} failed" in str(exc_info.value)


class Test_OpenRouterClient_InitializeEndpoints_05_StateTransitionBehaviors:
    """Test state transition behaviors for OpenRouterClient._initialize_endpoints method."""
    
    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_verify_complete_endpoint_initialization_state_transition(
        self, mock_logging, mock_http_manager, mock_auth_manager
    ):
        """Test complete endpoint initialization state transition."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        mock_auth_instance = Mock()
        mock_auth_manager.return_value = mock_auth_instance
        mock_http_instance = Mock()
        mock_http_manager.return_value = mock_http_instance
        
        endpoint_instances = {}
        endpoint_mocks = {}
        endpoint_attributes = ['completions', 'chat', 'models', 'images', 'generations',
                              'credits', 'keys', 'plugins', 'web']
        endpoint_classes = ['CompletionsEndpoint', 'ChatEndpoint', 'ModelsEndpoint', 'ImagesEndpoint',
                           'GenerationsEndpoint', 'CreditsEndpoint', 'KeysEndpoint', 'PluginsEndpoint', 'WebEndpoint']
        
        for cls_name, attr_name in zip(endpoint_classes, endpoint_attributes):
            instance = Mock()
            endpoint_instances[attr_name] = instance
            endpoint_mocks[cls_name] = Mock(return_value=instance)
            
        with patch.multiple('openrouter_client.client', **endpoint_mocks):
            # Act
            client = OpenRouterClient(api_key="test")
            
            # Assert - Verify state transition from no endpoints to all endpoints
            for attr_name in endpoint_attributes:
                assert hasattr(client, attr_name)
                assert getattr(client, attr_name) == endpoint_instances[attr_name]


class Test_OpenRouterClient_RefreshContextLengths_01_NominalBehaviors:
    """Test nominal behaviors for OpenRouterClient.refresh_context_lengths method."""
    
    @pytest.mark.parametrize("api_response_format,expected_models", [
        # Test data array format
        (
            {"data": [
                {"id": "model1", "context_length": 4096},
                {"id": "model2", "context_length": 8192}
            ]},
            {"model1": 4096, "model2": 8192}
        ),
        # Test direct list format
        (
            [
                {"id": "model3", "context_length": 2048},
                {"id": "model4", "context_length": 16384}
            ],
            {"model3": 2048, "model4": 16384}
        ),
        # Test empty valid response
        (
            {"data": []},
            {}
        ),
        # Test mixed valid/invalid models
        (
            {"data": [
                {"id": "model5", "context_length": 1024},
                {"id": "", "context_length": 2048},  # Invalid: empty id
                {"id": "model6"},  # Invalid: missing context_length
                {"id": "model7", "context_length": 0},  # Invalid: zero context_length
                {"id": "model8", "context_length": 32768}
            ]},
            {"model5": 1024, "model8": 32768}
        )
    ])
    def test_successfully_process_various_valid_api_response_formats_and_update_registry(
        self, api_response_format, expected_models
    ):
        """Test successful processing of various valid API response formats and registry updates."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            client.models = Mock()
            client.models.list.return_value = api_response_format
            client.logger = Mock()
            
            # Act
            result = client.refresh_context_lengths()
            
            # Assert
            assert result == expected_models
            assert client._context_lengths == expected_models
            client.models.list.assert_called_once_with(details=True)
            client.logger.info.assert_any_call("Refreshing model context lengths from API")
            client.logger.info.assert_any_call(f"Successfully refreshed context lengths for {len(expected_models)} models")


class Test_OpenRouterClient_RefreshContextLengths_02_NegativeBehaviors:
    """Test negative behaviors for OpenRouterClient.refresh_context_lengths method."""
    
    @pytest.mark.parametrize("malformed_response", [
        None,
        {"invalid": "structure"},
        {"data": None},
        {"data": "not_a_list"},
        "invalid_string_response",
        123,
        {"data": [None, {"invalid": "model"}]}
    ])
    def test_handle_malformed_or_incomplete_api_response_data(self, malformed_response):
        """Test handling of malformed or incomplete API response data."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            client.models = Mock()
            client.models.list.return_value = malformed_response
            client.logger = Mock()
            
            # Act
            result = client.refresh_context_lengths()
            
            # Assert - Should handle gracefully and return empty registry
            assert isinstance(result, dict)
            assert len(result) == 0
            client.logger.info.assert_any_call("Refreshing model context lengths from API")


class Test_OpenRouterClient_RefreshContextLengths_03_BoundaryBehaviors:
    """Test boundary behaviors for OpenRouterClient.refresh_context_lengths method."""
    
    @pytest.mark.parametrize("boundary_scenario,response_data,expected_count", [
        # Zero models
        ("zero_models", {"data": []}, 0),
        # Large number of models
        ("many_models", {"data": [{"id": f"model{i}", "context_length": 4096} for i in range(1000)]}, 1000),
        # Zero context length (should be filtered out)
        ("zero_context", {"data": [{"id": "model1", "context_length": 0}]}, 0),
        # Maximum context length
        ("max_context", {"data": [{"id": "model1", "context_length": 2**31-1}]}, 1),
    ])
    def test_handle_boundary_cases_in_model_count_and_context_length_values(
        self, boundary_scenario, response_data, expected_count
    ):
        """Test handling of boundary cases in model count and context length values."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            client.models = Mock()
            client.models.list.return_value = response_data
            client.logger = Mock()
            
            # Act
            result = client.refresh_context_lengths()
            
            # Assert
            assert len(result) == expected_count
            if expected_count > 0:
                for model_id, context_length in result.items():
                    assert isinstance(model_id, str)
                    assert isinstance(context_length, int)
                    assert context_length > 0


class Test_OpenRouterClient_RefreshContextLengths_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for OpenRouterClient.refresh_context_lengths method."""
    
    @pytest.mark.parametrize("error_scenario,exception_type", [
        ("api_request_failure", Exception),
        ("network_timeout", Exception),
        ("json_decode_error", Exception),
    ])
    def test_handle_api_communication_failures_and_convert_to_appropriate_exceptions(
        self, error_scenario, exception_type
    ):
        """Test handling of API communication failures and conversion to appropriate exceptions."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            client.models = Mock()
            client.models.list.side_effect = exception_type(f"API {error_scenario}")
            client.logger = Mock()
            
            # Act & Assert
            with pytest.raises(APIError) as exc_info:
                client.refresh_context_lengths()
            
            assert "Failed to retrieve model context lengths" in str(exc_info.value)
            client.logger.error.assert_called_once()
            assert f"API {error_scenario}" in client.logger.error.call_args[0][0]


class Test_OpenRouterClient_RefreshContextLengths_05_StateTransitionBehaviors:
    """Test state transition behaviors for OpenRouterClient.refresh_context_lengths method."""
    
    @pytest.mark.parametrize("initial_state,new_data,expected_final_state", [
        # Empty to populated
        ({}, {"data": [{"id": "model1", "context_length": 4096}]}, {"model1": 4096}),
        # Update existing
        ({"model1": 2048}, {"data": [{"id": "model1", "context_length": 4096}]}, {"model1": 4096}),
        # Partial update (some valid, some invalid)
        (
            {"model1": 2048}, 
            {"data": [
                {"id": "model1", "context_length": 4096},
                {"id": "model2"},  # Invalid
                {"id": "model3", "context_length": 8192}
            ]}, 
            {"model1": 4096, "model3": 8192}
        ),
        # Add to existing
        (
            {"model1": 2048}, 
            {"data": [{"id": "model2", "context_length": 4096}]}, 
            {"model1": 2048, "model2": 4096}
        )
    ])
    def test_verify_registry_state_updates_under_various_data_validity_conditions(
        self, initial_state, new_data, expected_final_state
    ):
        """Test registry state updates under various data validity conditions."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            client._context_lengths = initial_state.copy()
            client.models = Mock()
            client.models.list.return_value = new_data
            client.logger = Mock()
            
            # Act
            result = client.refresh_context_lengths()
            
            # Assert
            assert client._context_lengths == expected_final_state
            assert result == expected_final_state


class Test_OpenRouterClient_GetContextLength_01_NominalBehaviors:
    """Test nominal behaviors for OpenRouterClient.get_context_length method."""
    
    @pytest.mark.parametrize("model_id,registry_state,expected_result", [
        ("existing_model", {"existing_model": 8192}, 8192),
        ("nonexistent_model", {"existing_model": 8192}, 4096),
        ("another_model", {"model1": 2048, "model2": 4096, "another_model": 16384}, 16384),
        ("missing", {}, 4096),
    ])
    def test_return_appropriate_context_length_values_for_various_model_lookup_scenarios(
        self, model_id, registry_state, expected_result
    ):
        """Test returning appropriate context length values for various model lookup scenarios."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            client._context_lengths = registry_state
            
            # Act
            result = client.get_context_length(model_id)
            
            # Assert
            assert result == expected_result
            assert isinstance(result, int)


class Test_OpenRouterClient_GetContextLength_02_NegativeBehaviors:
    """Test negative behaviors for OpenRouterClient.get_context_length method."""
    
    @pytest.mark.parametrize("invalid_model_id", [
        None,
        123,
        [],
        {},
        object(),
    ])
    def test_handle_invalid_input_types_for_model_identifier(self, invalid_model_id):
        """Test handling of invalid input types for model identifier."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            client._context_lengths = {"valid_model": 4096}
            
            # Act
            result = client.get_context_length(invalid_model_id)
            
            # Assert - Should return default value gracefully
            assert result == 4096


class Test_OpenRouterClient_GetContextLength_03_BoundaryBehaviors:
    """Test boundary behaviors for OpenRouterClient.get_context_length method."""
    
    @pytest.mark.parametrize("boundary_model_id,registry_state,expected_result", [
        ("", {"": 2048}, 2048),  # Empty string exists
        ("", {"model1": 4096}, 4096),  # Empty string doesn't exist
        ("x" * 1000, {"x" * 1000: 8192}, 8192),  # Very long string exists
        ("x" * 1000, {"model1": 4096}, 4096),  # Very long string doesn't exist
        ("model-with-特殊chars-éñü", {"model-with-特殊chars-éñü": 1024}, 1024), # Special chars exist
        ("model-with-特殊chars-éñü", {"model1": 4096}, 4096), # Special chars don't exist
    ])
    def test_handle_edge_cases_in_model_identifier_format_and_length(
        self, boundary_model_id, registry_state, expected_result
    ):
        """Test handling of edge cases in model identifier format and length."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            client._context_lengths = registry_state
            
            # Act
            result = client.get_context_length(boundary_model_id)
            
            # Assert
            assert result == expected_result


class Test_OpenRouterClient_GetContextLength_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for OpenRouterClient.get_context_length method."""
    
    def test_verify_method_never_raises_exceptions(self):
        """Verify method never raises exceptions, ensuring graceful degradation."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            client._context_lengths = {"valid_model": 4096}
            
            # Act & Assert
            try:
                client.get_context_length(None)  # Test with invalid input
                client.get_context_length("nonexistent_model") # Test with valid but non-existent
            except Exception as e:
                pytest.fail(f"get_context_length raised an unexpected exception: {e}")


class Test_OpenRouterClient_GetContextLength_05_StateTransitionBehaviors:
    """Test state transition behaviors for OpenRouterClient.get_context_length method."""
    
    def test_verify_method_does_not_modify_client_state(self):
        """Verify method does not modify client state (read-only behavior)."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            initial_registry = {"model1": 4096, "model2": 8192}
            client._context_lengths = initial_registry.copy()
            
            # Act
            client.get_context_length("model1")
            client.get_context_length("nonexistent_model")
            
            # Assert
            assert client._context_lengths == initial_registry, "Registry was modified"


class Test_OpenRouterClient_CalculateRateLimits_01_NominalBehaviors:
    """Test nominal behaviors for OpenRouterClient.calculate_rate_limits method."""
    
    @pytest.mark.parametrize("credits_response,expected_limits", [
        # Sufficient credits
        (
            {"remaining": 1000, "refresh_rate": {"seconds": 3600}}, 
            {"requests": 100, "period": 60, "cooldown": 0}
        ),
        # Low credits, cooldown active
        (
            {"remaining": 5, "refresh_rate": {"seconds": 1800}}, 
            {"requests": 1, "period": 60, "cooldown": 1800}
        ),
        # Minimal credits
        (
            {"remaining": 10, "refresh_rate": {"seconds": 3600}},
            {"requests": 1, "period": 60, "cooldown": 0}
        ),
        # No refresh_rate info
        (
            {"remaining": 500},
            {"requests": 50, "period": 60, "cooldown": 0}
        )
    ])
    def test_successfully_process_credit_information_and_generate_rate_limit_configuration(
        self, credits_response, expected_limits
    ):
        """Test successful processing of credit info and rate limit generation."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            client.credits = Mock()
            client.credits.get.return_value = credits_response
            client.logger = Mock()
            
            # Act
            result = client.calculate_rate_limits()
            
            # Assert
            assert result == expected_limits
            client.credits.get.assert_called_once()
            client.logger.info.assert_any_call("Calculating rate limits based on remaining credits")
            client.logger.info.assert_any_call(
                f"Calculated rate limits: {expected_limits['requests']} requests per minute, "
                f"cooldown: {expected_limits['cooldown']} seconds"
            )


class Test_OpenRouterClient_CalculateRateLimits_02_NegativeBehaviors:
    """Test negative behaviors for OpenRouterClient.calculate_rate_limits method."""
    
    @pytest.mark.parametrize("invalid_response", [
        None,
        {"invalid": "structure"},
        {"remaining": "not_a_number"},
        {"refresh_rate": "not_a_dict"},
        {"refresh_rate": {"seconds": "not_a_number"}},
    ])
    def test_handle_invalid_or_incomplete_credit_api_responses(self, invalid_response):
        """Test handling of invalid or incomplete credit API responses."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            client.credits = Mock()
            client.credits.get.return_value = invalid_response
            client.logger = Mock()
            
            # Act & Assert
            # Default behavior when response is malformed: assumes 0 credits, 3600s refresh
            expected_limits = {"requests": 1, "period": 60, "cooldown": 3600}
            if invalid_response is None or not isinstance(invalid_response, dict):
                result = client.calculate_rate_limits()
                assert result == expected_limits
            elif isinstance(invalid_response.get("remaining"), str):
                 result = client.calculate_rate_limits() # Should default to 0 remaining
                 assert result == expected_limits
            elif isinstance(invalid_response.get("refresh_rate"), str):
                 result = client.calculate_rate_limits() # Should default to 3600s
                 assert result['cooldown'] == 3600
            else: # Other specific malformed cases
                result = client.calculate_rate_limits()
                # Check general structure, as specific values depend on how defaults are handled
                assert "requests" in result
                assert "period" in result
                assert "cooldown" in result
            
            client.logger.info.assert_any_call("Calculating rate limits based on remaining credits")


class Test_OpenRouterClient_CalculateRateLimits_03_BoundaryBehaviors:
    """Test boundary behaviors for OpenRouterClient.calculate_rate_limits method."""
    
    @pytest.mark.parametrize("boundary_scenario,credits_data,expected_requests,expected_cooldown", [
        ("zero_credits", {"remaining": 0, "refresh_rate": {"seconds": 3600}}, 1, 3600),
        ("max_credits", {"remaining": 10**9, "refresh_rate": {"seconds": 3600}}, 10**8, 0),
        ("negative_credits", {"remaining": -100, "refresh_rate": {"seconds": 3600}}, 1, 3600), # Treated as 0
        ("missing_refresh_rate", {"remaining": 100}, 10, 0), # Defaults to 3600s for cooldown logic, but 0 if >10 credits
        ("missing_seconds_in_refresh", {"remaining": 5, "refresh_rate": {}}, 1, 3600), # Defaults to 3600s
    ])
    def test_handle_boundary_values_in_credit_amounts_and_missing_optional_fields(
        self, boundary_scenario, credits_data, expected_requests, expected_cooldown
    ):
        """Test handling of boundary values in credit data and optional fields."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            client.credits = Mock()
            client.credits.get.return_value = credits_data
            client.logger = Mock()
            
            # Act
            result = client.calculate_rate_limits()
            
            # Assert
            assert result["requests"] == expected_requests
            assert result["cooldown"] == expected_cooldown
            assert result["period"] == 60


class Test_OpenRouterClient_CalculateRateLimits_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for OpenRouterClient.calculate_rate_limits method."""
    
    @pytest.mark.parametrize("error_scenario,exception_type", [
        ("api_failure", Exception),
        ("network_issue", Exception),
    ])
    def test_handle_api_communication_failures_and_convert_to_appropriate_exceptions(
        self, error_scenario, exception_type
    ):
        """Test handling of API communication failures and conversion to APIError."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            client.credits = Mock()
            client.credits.get.side_effect = exception_type(f"Credits API {error_scenario}")
            client.logger = Mock()
            
            # Act & Assert
            with pytest.raises(APIError) as exc_info:
                client.calculate_rate_limits()
            
            assert "Failed to retrieve credit information for rate limiting" in str(exc_info.value)
            client.logger.error.assert_called_once()
            assert f"Credits API {error_scenario}" in client.logger.error.call_args[0][0]


class Test_OpenRouterClient_CalculateRateLimits_05_StateTransitionBehaviors:
    """Test state transition behaviors for OpenRouterClient.calculate_rate_limits method."""
    
    @pytest.mark.parametrize("initial_credits,final_credits,expected_initial_req,expected_final_req", [
        # High to low credits
        ({"remaining": 1000, "refresh_rate": {"seconds": 3600}}, {"remaining": 5, "refresh_rate": {"seconds": 1800}}, 100, 1),
        # Low to high credits
        ({"remaining": 5, "refresh_rate": {"seconds": 1800}}, {"remaining": 1000, "refresh_rate": {"seconds": 3600}}, 1, 100),
    ])
    def test_verify_rate_limit_calculations_adapt_to_varying_credit_states(
        self, initial_credits, final_credits, expected_initial_req, expected_final_req
    ):
        """Test that rate limit calculations adapt to varying credit states."""
        # Arrange
        with patch('openrouter_client.client.AuthManager'), \
             patch('openrouter_client.client.HTTPManager'), \
             patch('openrouter_client.client.configure_logging'), \
             mock_all_endpoints():
            
            client = OpenRouterClient(api_key="test")
            client.credits = Mock()
            client.logger = Mock()
            
            # Act - Initial calculation
            client.credits.get.return_value = initial_credits
            initial_limits = client.calculate_rate_limits()
            
            # Act - Final calculation (simulating credit change)
            client.credits.get.return_value = final_credits
            final_limits = client.calculate_rate_limits()
            
            # Assert
            assert initial_limits["requests"] == expected_initial_req
            assert final_limits["requests"] == expected_final_req


class Test_OpenRouterClient_Close_01_NominalBehaviors:
    """Test nominal behaviors for OpenRouterClient.close method."""
    
    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_successfully_release_all_client_resources_and_log_completion(
        self, mock_logging, mock_http_manager, mock_auth_manager
    ):
        """Test successful release of all client resources and logging."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        mock_auth_instance = Mock()
        mock_auth_manager.return_value = mock_auth_instance
        mock_http_instance = Mock()
        mock_http_manager.return_value = mock_http_instance
        
        endpoint_names = [
            'CompletionsEndpoint', 'ChatEndpoint', 'ModelsEndpoint', 'ImagesEndpoint',
            'GenerationsEndpoint', 'CreditsEndpoint', 'KeysEndpoint', 'PluginsEndpoint', 'WebEndpoint'
        ]
        endpoint_attributes = [name.lower().replace('endpoint', '') for name in endpoint_names]

        with patch.multiple('openrouter_client.client', **{name: Mock() for name in endpoint_names}):
            client = OpenRouterClient(api_key="test")
        
            # Act
            client.close()
            
            # Assert
            mock_http_instance.close.assert_called_once()
            for attr in endpoint_attributes:
                assert getattr(client, attr) is None
            
            mock_logger.info.assert_any_call("Shutting down OpenRouterClient")
            mock_logger.info.assert_any_call("OpenRouterClient shut down successfully")


class Test_OpenRouterClient_Close_02_NegativeBehaviors:
    """Test negative behaviors for OpenRouterClient.close method."""
    
    @pytest.mark.parametrize("scenario", ["already_closed", "missing_endpoints"])
    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_handle_repeated_or_partial_closure_scenarios_gracefully(
        self, mock_logging, mock_http_manager, mock_auth_manager, scenario
    ):
        """Test graceful handling of repeated or partial closure scenarios."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        mock_http_instance = Mock()
        mock_http_manager.return_value = mock_http_instance

        with patch.multiple('openrouter_client.client', CompletionsEndpoint=Mock(), ChatEndpoint=Mock(), ModelsEndpoint=Mock(), 
                           ImagesEndpoint=Mock(), GenerationsEndpoint=Mock(), CreditsEndpoint=Mock(), 
                           KeysEndpoint=Mock(), PluginsEndpoint=Mock(), WebEndpoint=Mock()):
            client = OpenRouterClient(api_key="test")

        if scenario == "already_closed":
            client.close() # Close once
            mock_http_instance.close.reset_mock() # Reset mock for second call check
            mock_logger.reset_mock()

        elif scenario == "missing_endpoints":
            # Simulate some endpoints not being initialized (e.g., due to earlier error)
            del client.chat
            del client.models

        # Act
        try:
            client.close() # Attempt to close again or with missing parts
        except Exception as e:
            pytest.fail(f"client.close() raised an unexpected exception: {e}")

        # Assert
        if scenario == "already_closed":
            # HTTPManager.close might be called again if not idempotent, or not.
            # The key is no error and endpoints are None.
            pass 
        
        endpoint_attributes = ['completions', 'images', 'generations', 'credits', 'keys', 'plugins', 'web']
        if scenario == "missing_endpoints": # chat and models were deleted
            endpoint_attributes.extend(['chat', 'models'])

        for attr in endpoint_attributes:
            if hasattr(client, attr): # Check if it existed to be nulled
                 assert getattr(client, attr) is None
            
        mock_logger.info.assert_any_call("Shutting down OpenRouterClient")
        mock_logger.info.assert_any_call("OpenRouterClient shut down successfully")


class Test_OpenRouterClient_Close_03_BoundaryBehaviors:
    """Test boundary behaviors for OpenRouterClient.close method."""
    
    @pytest.mark.parametrize("initialization_level", ["minimal", "partial_endpoints"])
    @patch('openrouter_client.client.configure_logging')
    def test_handle_closure_with_varying_levels_of_client_initialization(
        self, mock_logging, initialization_level
    ):
        """Test closure with varying levels of client initialization."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        client = OpenRouterClient.__new__(OpenRouterClient) # Create instance without calling __init__
        client.logger = mock_logger # Assign logger manually

        if initialization_level == "minimal":
            # Only http_manager might exist, or not even that.
            # Let's assume http_manager was created but endpoints were not.
            client.http_manager = Mock()
        
        elif initialization_level == "partial_endpoints":
            client.http_manager = Mock()
            client.auth_manager = Mock() # Needed by _initialize_endpoints
            # Simulate only some endpoints initialized
            with patch('openrouter_client.client.CompletionsEndpoint', Mock()) as MockCompletions, \
                 patch('openrouter_client.client.ChatEndpoint', Mock()) as MockChat:
                 # Manually call parts of _initialize_endpoints or set attributes
                 client.completions = MockCompletions(auth_manager=client.auth_manager, http_manager=client.http_manager)
                 client.chat = MockChat(auth_manager=client.auth_manager, http_manager=client.http_manager)
                 # Other endpoints (models, images etc.) are not set on client


        # Act
        try:
            client.close()
        except Exception as e:
            pytest.fail(f"client.close() raised an unexpected exception: {e}")

        # Assert
        if hasattr(client, 'http_manager') and client.http_manager is not None:
            client.http_manager.close.assert_called_once()
        
        # Check that initialized endpoints were nulled
        if hasattr(client, 'completions'):
             assert client.completions is None
        if hasattr(client, 'chat'):
             assert client.chat is None
        
        mock_logger.info.assert_any_call("Shutting down OpenRouterClient")
        mock_logger.info.assert_any_call("OpenRouterClient shut down successfully")


class Test_OpenRouterClient_Close_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for OpenRouterClient.close method."""

    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_handle_component_cleanup_failures_and_ensure_best_effort_resource_release(
        self, mock_logging, mock_http_manager, mock_auth_manager
    ):
        """Test resilient cleanup that continues even if individual components fail."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        mock_http_instance = Mock()
        mock_http_instance.close.side_effect = Exception("HTTP close failed")
        mock_http_manager.return_value = mock_http_instance

        with patch.multiple('openrouter_client.client', CompletionsEndpoint=Mock(), ChatEndpoint=Mock(), ModelsEndpoint=Mock(), 
                           ImagesEndpoint=Mock(), GenerationsEndpoint=Mock(), CreditsEndpoint=Mock(), 
                           KeysEndpoint=Mock(), PluginsEndpoint=Mock(), WebEndpoint=Mock()):
            client = OpenRouterClient(api_key="test")
        
        # Act
        try:
            client.close() # HTTPManager close will raise an exception
        except Exception as e:
            assert "HTTP close failed" in str(e), "Expected HTTP close failure"

        # Assert
        # Ensure that even if http_manager.close fails, endpoints are still nulled
        endpoint_attributes = [
            'completions', 'chat', 'models', 'images', 'generations',
            'credits', 'keys', 'plugins', 'web'
        ]
        for attr in endpoint_attributes:
            assert getattr(client, attr) is None, f"Endpoint {attr} was not nulled"
        
        mock_logger.info.assert_any_call("Shutting down OpenRouterClient")
        # The "successfully" message might not be logged if an error occurs mid-way,
        # depending on exact logging placement in the original code.
        # If the "successfully" log is after all operations, it won't be hit.
        # If it's before the http_manager.close(), it would be.
        # For this test, we assume errors might prevent the "successful" log.


class Test_OpenRouterClient_Close_05_StateTransitionBehaviors:
    """Test state transition behaviors for OpenRouterClient.close method."""

    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_verify_complete_transition_to_closed_state_with_resource_nullification(
        self, mock_logging, mock_http_manager, mock_auth_manager
    ):
        """Test that client properly transitions to a closed state with all resources released."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        mock_http_instance = Mock()
        mock_http_manager.return_value = mock_http_instance
        
        endpoint_attributes = [
            'completions', 'chat', 'models', 'images', 'generations',
            'credits', 'keys', 'plugins', 'web'
        ]
        endpoint_classes = [
            'CompletionsEndpoint', 'ChatEndpoint', 'ModelsEndpoint', 'ImagesEndpoint',
            'GenerationsEndpoint', 'CreditsEndpoint', 'KeysEndpoint', 'PluginsEndpoint', 'WebEndpoint'
        ]

        with patch.multiple('openrouter_client.client', **{cls: Mock() for cls in endpoint_classes}):
            client = OpenRouterClient(api_key="test")
            
            # Verify initial state (endpoints are not None)
            for attr in endpoint_attributes:
                assert getattr(client, attr) is not None
            assert client.http_manager is not None

            # Act
            client.close()
            
            # Assert - Verify transition to closed state
            for attr in endpoint_attributes:
                assert getattr(client, attr) is None, f"Endpoint {attr} was not nulled"
            
            # http_manager itself might not be set to None, but its resources are closed.
            # The code sets endpoint attributes to None, but not http_manager attribute on client.
            mock_http_instance.close.assert_called_once()


class Test_OpenRouterClient_Enter_01_NominalBehaviors:
    """Test nominal behaviors for OpenRouterClient.__enter__ method."""

    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_return_self_instance_for_context_manager_usage(
        self, mock_logging, mock_http_manager, mock_auth_manager
    ):
        """Test that __enter__ returns the client instance itself."""
        # Arrange
        with patch.multiple('openrouter_client.client', CompletionsEndpoint=Mock(), ChatEndpoint=Mock(), ModelsEndpoint=Mock(), 
                           ImagesEndpoint=Mock(), GenerationsEndpoint=Mock(), CreditsEndpoint=Mock(), 
                           KeysEndpoint=Mock(), PluginsEndpoint=Mock(), WebEndpoint=Mock()):
            client_instance = OpenRouterClient(api_key="test")
        
            # Act
            returned_instance = client_instance.__enter__()
            
            # Assert
            assert returned_instance is client_instance


# __enter__ is too simple for Negative, Boundary, ErrorHandling, StateTransition tests.

class Test_OpenRouterClient_Exit_01_NominalBehaviors:
    """Test nominal behaviors for OpenRouterClient.__exit__ method."""

    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_successfully_complete_context_manager_lifecycle(
        self, mock_logging, mock_http_manager, mock_auth_manager
    ):
        """Test successful completion of the context manager lifecycle (normal exit)."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        mock_http_instance = Mock()
        mock_http_manager.return_value = mock_http_instance
        
        with patch.multiple('openrouter_client.client', CompletionsEndpoint=Mock(), ChatEndpoint=Mock(), ModelsEndpoint=Mock(), 
                           ImagesEndpoint=Mock(), GenerationsEndpoint=Mock(), CreditsEndpoint=Mock(), 
                           KeysEndpoint=Mock(), PluginsEndpoint=Mock(), WebEndpoint=Mock()):
            client = OpenRouterClient(api_key="test")
        
            # Act
            client.__exit__(None, None, None) # Simulate normal exit
            
            # Assert
            mock_http_instance.close.assert_called_once()
            mock_logger.error.assert_not_called() # No error should be logged
            mock_logger.info.assert_any_call("Shutting down OpenRouterClient")
            mock_logger.info.assert_any_call("OpenRouterClient shut down successfully")


class Test_OpenRouterClient_Exit_02_NegativeBehaviors:
    """Test negative behaviors for OpenRouterClient.__exit__ method."""
    # __exit__ is designed to handle all exception scenarios as part of its contract.
    # Specific negative behaviors would relate to how `close()` handles issues,
    # which are tested in `Test_OpenRouterClient_Close_...` tests.
    # This test focuses on ensuring `close` is called even with exceptions.
    
    @pytest.mark.parametrize("exc_type, exc_val, exc_tb", [
        (ValueError, ValueError("Test error"), Mock()),
        (TypeError, TypeError("Another error"), Mock()),
        (None, None, None) # For completeness, also test normal exit path here
    ])
    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_handle_all_exception_scenarios_during_context_exit(
        self, mock_logging, mock_http_manager, mock_auth_manager, exc_type, exc_val, exc_tb
    ):
        """Test that __exit__ handles various exception scenarios by calling close."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        mock_http_instance = Mock()
        mock_http_manager.return_value = mock_http_instance
        
        with patch.multiple('openrouter_client.client', CompletionsEndpoint=Mock(), ChatEndpoint=Mock(), ModelsEndpoint=Mock(), 
                           ImagesEndpoint=Mock(), GenerationsEndpoint=Mock(), CreditsEndpoint=Mock(), 
                           KeysEndpoint=Mock(), PluginsEndpoint=Mock(), WebEndpoint=Mock()):
            client = OpenRouterClient(api_key="test")
        
            # Act
            client.__exit__(exc_type, exc_val, exc_tb)
            
            # Assert
            mock_http_instance.close.assert_called_once() # Ensure close is always called
            if exc_type is not None:
                mock_logger.error.assert_called_once_with(
                    f"Exception occurred in OpenRouterClient context: {exc_type.__name__}: {exc_val}"
                )
            else:
                mock_logger.error.assert_not_called()


# Boundary behaviors for __exit__ are implicitly covered by parameterizing exception types.

class Test_OpenRouterClient_Exit_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for OpenRouterClient.__exit__ method."""

    @pytest.mark.parametrize("exception_in_context", [True, False])
    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_handle_exceptions_during_context_usage_and_ensure_resource_cleanup(
        self, mock_logging, mock_http_manager, mock_auth_manager, exception_in_context
    ):
        """Test exception logging and resource cleanup during context exit."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        mock_http_instance = Mock()
        mock_http_manager.return_value = mock_http_instance
        
        exc_type, exc_val, exc_tb = (None, None, None)
        if exception_in_context:
            exc_type, exc_val, exc_tb = (RuntimeError, RuntimeError("Context error"), Mock())

        with patch.multiple('openrouter_client.client', CompletionsEndpoint=Mock(), ChatEndpoint=Mock(), ModelsEndpoint=Mock(), 
                           ImagesEndpoint=Mock(), GenerationsEndpoint=Mock(), CreditsEndpoint=Mock(), 
                           KeysEndpoint=Mock(), PluginsEndpoint=Mock(), WebEndpoint=Mock()):
            client = OpenRouterClient(api_key="test")
        
            # Act
            client.__exit__(exc_type, exc_val, exc_tb)
            
            # Assert
            mock_http_instance.close.assert_called_once() # Cleanup must occur
            if exception_in_context:
                mock_logger.error.assert_called_once_with(
                    f"Exception occurred in OpenRouterClient context: {exc_type.__name__}: {exc_val}"
                )
            else:
                mock_logger.error.assert_not_called()


class Test_OpenRouterClient_Exit_05_StateTransitionBehaviors:
    """Test state transition behaviors for OpenRouterClient.__exit__ method."""

    @pytest.mark.parametrize("exit_with_exception", [True, False])
    @patch('openrouter_client.client.AuthManager')
    @patch('openrouter_client.client.HTTPManager')
    @patch('openrouter_client.client.configure_logging')
    def test_verify_consistent_state_transition_regardless_of_exit_conditions(
        self, mock_logging, mock_http_manager, mock_auth_manager, exit_with_exception
    ):
        """Test that client consistently transitions to closed state."""
        # Arrange
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        mock_http_instance = Mock()
        mock_http_manager.return_value = mock_http_instance
        
        exc_info = (None, None, None)
        if exit_with_exception:
            exc_info = (TypeError, TypeError("Exit error"), Mock())

        endpoint_attributes = [
            'completions', 'chat', 'models', 'images', 'generations',
            'credits', 'keys', 'plugins', 'web'
        ]
        endpoint_classes = [
            'CompletionsEndpoint', 'ChatEndpoint', 'ModelsEndpoint', 'ImagesEndpoint',
            'GenerationsEndpoint', 'CreditsEndpoint', 'KeysEndpoint', 'PluginsEndpoint', 'WebEndpoint'
        ]
        
        with patch.multiple('openrouter_client.client', **{cls: Mock() for cls in endpoint_classes}):
            client = OpenRouterClient(api_key="test")

            # Verify initial state (endpoints are not None)
            for attr in endpoint_attributes:
                assert getattr(client, attr) is not None

            # Act
            client.__exit__(*exc_info)
            
            # Assert - Verify transition to closed state (endpoints nulled, http closed)
            mock_http_instance.close.assert_called_once()
            for attr in endpoint_attributes:
                assert getattr(client, attr) is None, f"Endpoint {attr} was not nulled on exit"
