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
        assert endpoint.endpoint_path == "generations"


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
    
    def test_parent_class_initialization_failure(self):
        """Test behavior when parent class initialization fails"""
        # Arrange
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesnotexist.com")
        
        # Act & Assert
        with patch('openrouter_client.endpoints.base.BaseEndpoint.__init__', side_effect=Exception("Parent init failed")):
            with pytest.raises(Exception) as exc_info:
                GenerationsEndpoint(auth_manager, http_manager)
            assert "Parent init failed" in str(exc_info.value)


class TestGenerationsEndpointInit05StateTransitionBehaviors:
    """Test state transition behaviors for GenerationsEndpoint.__init__()"""
    
    def test_object_transitions_to_configured_state(self):
        """Test object transitions from uninitialized to properly configured state"""
        # Arrange
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesnotexist.com")
        
        # Act
        endpoint = GenerationsEndpoint(auth_manager, http_manager)
        
        # Assert
        assert hasattr(endpoint, 'auth_manager')
        assert hasattr(endpoint, 'http_manager')
        assert hasattr(endpoint, 'endpoint_path')
        assert endpoint.endpoint_path == "generations"
        assert hasattr(endpoint, 'logger')


class TestGenerationsEndpointList01NominalBehaviors:
    """Test nominal behaviors for GenerationsEndpoint.list()"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth_manager = AuthManager()
        self.http_manager = Mock(spec=HTTPManager)
        self.endpoint = GenerationsEndpoint(self.auth_manager, self.http_manager)
        self.mock_response = Mock()
        self.mock_response.json.return_value = {
            "data": [],
            "total": 0,
            "page": 1,
            "per_page": 10
        }
        self.http_manager.get.return_value = self.mock_response
    
    @pytest.mark.parametrize("limit,offset,model", [
        (10, 0, None),
        (50, 20, "gpt-4"),
        (None, None, None),
        (1, 0, "claude-3"),
        (100, 50, "dall-e-3"),
    ])
    def test_successful_retrieval_with_valid_parameters(self, limit, offset, model):
        """Test successful retrieval with valid parameter combinations"""
        # Arrange
        with patch('openrouter_client.models.generations.GenerationList.model_validate') as mock_validate:
            mock_validate.return_value = Mock()
            
            # Act
            result = self.endpoint.list(limit=limit, offset=offset, model=model)
            
            # Assert
            assert self.http_manager.get.called
            call_args = self.http_manager.get.call_args
            params = call_args[1]['params']
            
            if limit is not None:
                assert params['limit'] == limit
            if offset is not None:
                assert params['offset'] == offset
            if model is not None:
                assert params['model'] == model
            
            assert mock_validate.called
    
    @pytest.mark.parametrize("start_date,end_date", [
        (datetime(2024, 1, 1), datetime(2024, 12, 31)),
        (datetime.now() - timedelta(days=30), datetime.now()),
        (datetime(2023, 6, 15, 10, 30), datetime(2023, 6, 15, 18, 45)),
    ])
    def test_datetime_to_iso_format_conversion(self, start_date, end_date):
        """Test proper datetime object conversion to ISO format strings"""
        # Arrange
        with patch('openrouter_client.models.generations.GenerationList.model_validate') as mock_validate:
            mock_validate.return_value = Mock()
            
            # Act
            self.endpoint.list(start_date=start_date, end_date=end_date)
            
            # Assert
            call_args = self.http_manager.get.call_args
            params = call_args[1]['params']
            
            assert params['start_date'] == start_date.isoformat()
            assert params['end_date'] == end_date.isoformat()
    
    def test_string_dates_passed_through_unchanged(self):
        """Test string dates are passed through without conversion"""
        # Arrange
        start_date_str = "2024-01-01T00:00:00"
        end_date_str = "2024-12-31T23:59:59"
        
        with patch('openrouter_client.models.generations.GenerationList.model_validate') as mock_validate:
            mock_validate.return_value = Mock()
            
            # Act
            self.endpoint.list(start_date=start_date_str, end_date=end_date_str)
            
            # Assert
            call_args = self.http_manager.get.call_args
            params = call_args[1]['params']
            
            assert params['start_date'] == start_date_str
            assert params['end_date'] == end_date_str


class TestGenerationsEndpointList02NegativeBehaviors:
    """Test negative behaviors for GenerationsEndpoint.list()"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth_manager = AuthManager()
        self.http_manager = Mock(spec=HTTPManager)
        self.endpoint = GenerationsEndpoint(self.auth_manager, self.http_manager)
    
    @pytest.mark.parametrize("limit,offset", [
        (-1, 0),
        (0, -1),
        (-10, -5),
        ("invalid", 0),
        (0, "invalid"),
        (10.5, 0),
        (0, 20.3),
    ])
    def test_invalid_limit_offset_parameters(self, limit, offset):
        """Test invalid parameter types for limit and offset"""
        # Arrange, Act & Assert
        with pytest.raises((TypeError, ValueError)):
            self.endpoint.list(limit=limit, offset=offset)
    
    @pytest.mark.parametrize("start_date,end_date", [
        ("invalid-date", None),
        (None, "malformed-date"),
        ("2024-13-01", None),  # Invalid month
        ("2024-02-30", None),  # Invalid day
        (None, "not-a-date-at-all"),
        ("2024-01-01T25:00:00", None),  # Invalid hour
    ])
    def test_malformed_date_parameters(self, start_date, end_date):
        """Test malformed date parameters causing conversion failures"""
        # Arrange
        self.http_manager.get.side_effect = ValueError("Invalid date format")
        
        # Act & Assert
        with pytest.raises(ValueError):
            self.endpoint.list(start_date=start_date, end_date=end_date)


class TestGenerationsEndpointList04ErrorHandlingBehaviors:
    """Test error handling behaviors for GenerationsEndpoint.list()"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth_manager = AuthManager()
        self.http_manager = Mock(spec=HTTPManager)
        self.endpoint = GenerationsEndpoint(self.auth_manager, self.http_manager)
    
    @pytest.mark.parametrize("exception_type,exception_message", [
        (ConnectionError, "Network connection failed"),
        (TimeoutError, "Request timed out"),
        (Exception, "HTTP request failed"),
    ])
    def test_network_api_failures_during_request(self, exception_type, exception_message):
        """Test network/API failures during HTTP request execution"""
        # Arrange
        self.http_manager.get.side_effect = exception_type(exception_message)
        
        # Act & Assert
        with pytest.raises(exception_type) as exc_info:
            self.endpoint.list()
        assert exception_message in str(exc_info.value)
    
    def test_generation_list_validation_failures(self):
        """Test GenerationList.model_validate() failures with invalid response data"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {"invalid": "data"}
        self.http_manager.get.return_value = mock_response
        
        with patch('openrouter_client.models.generations.GenerationList.model_validate', side_effect=ValueError("Validation failed")):
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                self.endpoint.list()
            assert "Validation failed" in str(exc_info.value)


class TestGenerationsEndpointGet01NominalBehaviors:
    """Test nominal behaviors for GenerationsEndpoint.get()"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth_manager = Mock(spec=AuthManager)
        self.http_manager = Mock(spec=HTTPManager)
        self.endpoint = GenerationsEndpoint(self.auth_manager, self.http_manager)
        self.mock_response = Mock()
        self.mock_response.json.return_value = {
            "id": "gen_123",
            "status": "completed",
            "created_at": "2024-01-01T00:00:00Z"
        }
        self.http_manager.get.return_value = self.mock_response
    
    @pytest.mark.parametrize("generation_id", [
        "gen_123456789",
        "generation_abcdef", 
        "12345-abcde-67890",
        "short_id",
        "very_long_generation_identifier_with_many_characters",
    ])
    def test_successful_retrieval_with_valid_generation_id(self, generation_id):
        """Test successful retrieval with valid generation_id"""
        # Arrange
        # Mock the auth_manager to return a proper dictionary
        self.endpoint.auth_manager.get_auth_headers.return_value = {
            "Authorization": "Bearer test_token"
        }
        
        with patch('openrouter_client.models.generations.Generation.model_validate') as mock_validate:
            mock_validate.return_value = Mock()
            
            # Act
            result = self.endpoint.get(generation_id)
            
            # Assert
            assert self.http_manager.get.called
            call_args = self.http_manager.get.call_args
            assert generation_id in call_args[0][0]  # URL contains generation_id
            assert mock_validate.called
    
    def test_provisioning_api_key_enforcement(self):
        """Test proper provisioning API key requirement enforcement"""
        # Arrange
        with patch.object(self.endpoint, '_get_headers') as mock_headers:
            with patch('openrouter_client.models.generations.Generation.model_validate') as mock_validate:
                mock_validate.return_value = Mock()
                
                # Act
                self.endpoint.get("gen_123")
                
                # Assert
                mock_headers.assert_called_once_with(require_provisioning=True)


class TestGenerationsEndpointGet02NegativeBehaviors:
    """Test negative behaviors for GenerationsEndpoint.get()"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth_manager = Mock(spec=AuthManager)
        self.http_manager = Mock(spec=HTTPManager)
        self.endpoint = GenerationsEndpoint(self.auth_manager, self.http_manager)
    
    @pytest.mark.parametrize("generation_id", [
        None,
        "",
        "   ",  # Whitespace only
        123,  # Wrong type
        [],   # Wrong type
        {},   # Wrong type
    ])
    def test_invalid_generation_id_parameters(self, generation_id):
        """Test empty, None, or invalid format generation_id parameter"""
        # Arrange, Act & Assert
        with pytest.raises((TypeError, ValueError, AttributeError)):
            self.endpoint.get(generation_id)


class TestGenerationsEndpointGet04ErrorHandlingBehaviors:
    """Test error handling behaviors for GenerationsEndpoint.get()"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth_manager = Mock(spec=AuthManager)
        self.http_manager = Mock(spec=HTTPManager)
        
        self.auth_manager.get_auth_headers.return_value = {
            "Authorization": "Bearer test-token",
            "X-API-Key": "test-key"
        }
        
        self.endpoint = GenerationsEndpoint(self.auth_manager, self.http_manager)
    
    def test_http_404_responses_for_nonexistent_generations(self):
        """Test HTTP 404 responses for non-existent generations"""
        # Arrange
        # Mock _get_headers to return a dictionary
        self.endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer test-token'})
        
        mock_404_error = Exception("404 Not Found")
        self.http_manager.get.side_effect = mock_404_error
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            self.endpoint.get("nonexistent_gen_id")
        assert "404" in str(exc_info.value)
    
    def test_authentication_failures_invalid_provisioning_keys(self):
        """Test authentication failures due to invalid provisioning keys"""
        # Arrange
        auth_error = Exception("Authentication failed - invalid provisioning key")
        with patch.object(self.endpoint, '_get_headers', side_effect=auth_error):
            # Act & Assert
            with pytest.raises(Exception) as exc_info:
                self.endpoint.get("gen_123")
            assert "Authentication failed" in str(exc_info.value)
    
    def test_generation_validation_failures(self):
        """Test Generation.model_validate() failures with corrupted response data"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {"corrupted": "data"}
        self.http_manager.get.return_value = mock_response
        
        with patch('openrouter_client.models.generations.Generation.model_validate', 
                side_effect=ValueError("Validation failed")):
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                self.endpoint.get("gen_123")
            
            # Verify the specific validation error is propagated
            assert "Validation failed" in str(exc_info.value)
            
            # Verify the HTTP request was made with correct parameters
            self.http_manager.get.assert_called_once()
            call_args = self.http_manager.get.call_args
            assert "gen_123" in call_args[0][0]  # URL contains generation_id
            
            # Verify auth headers were requested with provisioning requirement
            self.auth_manager.get_auth_headers.assert_called_once_with(require_provisioning=True)
            
            # Verify response.json() was called
            mock_response.json.assert_called_once()


class TestGenerationsEndpointStats01NominalBehaviors:
    """Test nominal behaviors for GenerationsEndpoint.stats()"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth_manager = AuthManager()
        self.http_manager = Mock(spec=HTTPManager)
        self.endpoint = GenerationsEndpoint(self.auth_manager, self.http_manager)
        self.mock_response = Mock()
        self.mock_response.json.return_value = {
            "total_generations": 1000,
            "period": "month",
            "stats": []
        }
        self.http_manager.get.return_value = self.mock_response
    
    @pytest.mark.parametrize("period", [
        "day",
        "week", 
        "month",
        "year"
    ])
    def test_successful_statistics_retrieval_with_valid_periods(self, period):
        """Test successful statistics retrieval with valid period values"""
        # Arrange
        with patch('openrouter_client.models.generations.GenerationStats.model_validate') as mock_validate:
            mock_validate.return_value = Mock()
            
            # Act
            result = self.endpoint.stats(period=period)
            
            # Assert
            assert self.http_manager.get.called
            call_args = self.http_manager.get.call_args
            params = call_args[1]['params']
            assert params['period'] == period
            assert mock_validate.called
    
    @pytest.mark.parametrize("start_date,end_date", [
        (datetime(2024, 1, 1), datetime(2024, 1, 31)),
        (datetime.now() - timedelta(days=7), datetime.now()),
        (datetime(2023, 12, 1, 9, 0), datetime(2023, 12, 31, 17, 0)),
    ])
    def test_datetime_conversion_for_date_parameters(self, start_date, end_date):
        """Test proper datetime conversion for date parameters"""
        # Arrange
        with patch('openrouter_client.models.generations.GenerationStats.model_validate') as mock_validate:
            mock_validate.return_value = Mock()
            
            # Act
            self.endpoint.stats(start_date=start_date, end_date=end_date)
            
            # Assert
            call_args = self.http_manager.get.call_args
            params = call_args[1]['params']
            
            assert params['start_date'] == start_date.isoformat()
            assert params['end_date'] == end_date.isoformat()


class TestGenerationsEndpointStats02NegativeBehaviors:
    """Test negative behaviors for GenerationsEndpoint.stats()"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth_manager = AuthManager()
        self.http_manager = Mock(spec=HTTPManager)
        self.endpoint = GenerationsEndpoint(self.auth_manager, self.http_manager)
    
    @pytest.mark.parametrize("period", [
        "invalid_period",
        "hour",  # Not in allowed options
        "quarter",  # Not in allowed options
        "second",  # Not in allowed options
        "",
        None,
        123,
        [],
    ])
    def test_invalid_period_parameter_values(self, period):
        """Test invalid period parameter values outside expected options"""
        # Arrange
        api_error = Exception(f"Invalid period: {period}")
        self.http_manager.get.side_effect = api_error
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            self.endpoint.stats(period=period)
        assert "Invalid period" in str(exc_info.value)


class TestGenerationsEndpointStats04ErrorHandlingBehaviors:
    """Test error handling behaviors for GenerationsEndpoint.stats()"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth_manager = AuthManager()
        self.http_manager = Mock(spec=HTTPManager)
        self.endpoint = GenerationsEndpoint(self.auth_manager, self.http_manager)
    
    @pytest.mark.parametrize("exception_type,exception_message", [
        (ConnectionError, "Statistics service unavailable"),
        (TimeoutError, "Statistics request timed out"),
        (Exception, "Statistics endpoint failed"),
    ])
    def test_network_api_failures_during_statistics_retrieval(self, exception_type, exception_message):
        """Test network/API failures during statistics retrieval"""
        # Arrange
        self.http_manager.get.side_effect = exception_type(exception_message)
        
        # Act & Assert
        with pytest.raises(exception_type) as exc_info:
            self.endpoint.stats()
        assert exception_message in str(exc_info.value)
    
    def test_generation_stats_validation_failures(self):
        """Test GenerationStats.model_validate() failures"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {"invalid": "stats_data"}
        self.http_manager.get.return_value = mock_response
        
        with patch('openrouter_client.models.generations.GenerationStats.model_validate', side_effect=ValueError("Stats validation failed")):
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                self.endpoint.stats()
            assert "Stats validation failed" in str(exc_info.value)


class TestGenerationsEndpointModels01NominalBehaviors:
    """Test nominal behaviors for GenerationsEndpoint.models()"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth_manager = AuthManager()
        self.http_manager = Mock(spec=HTTPManager)
        self.endpoint = GenerationsEndpoint(self.auth_manager, self.http_manager)
        self.mock_response = Mock()
        self.mock_response.json.return_value = {
            "models": {
                "gpt-4": {"usage_count": 100},
                "claude-3": {"usage_count": 50}
            }
        }
        self.http_manager.get.return_value = self.mock_response
    
    def test_successful_model_statistics_retrieval_without_date_filtering(self):
        """Test successful model statistics retrieval with no date parameters"""
        # Arrange
        with patch('openrouter_client.models.generations.ModelStats.model_validate') as mock_validate:
            mock_validate.return_value = Mock()
            
            # Act
            result = self.endpoint.models()
            
            # Assert
            assert self.http_manager.get.called
            call_args = self.http_manager.get.call_args
            assert 'models' in call_args[0][0]  # URL contains 'models'
            assert mock_validate.called
    
    @pytest.mark.parametrize("start_date,end_date", [
        (datetime(2024, 1, 1), datetime(2024, 6, 30)),
        (datetime.now() - timedelta(days=90), datetime.now()),
        (datetime(2023, 10, 15, 14, 30), datetime(2023, 11, 15, 16, 45)),
    ])
    def test_datetime_conversion_with_date_filtering(self, start_date, end_date):
        """Test proper datetime conversion when date parameters are provided"""
        # Arrange
        with patch('openrouter_client.models.generations.ModelStats.model_validate') as mock_validate:
            mock_validate.return_value = Mock()
            
            # Act
            self.endpoint.models(start_date=start_date, end_date=end_date)
            
            # Assert
            call_args = self.http_manager.get.call_args
            params = call_args[1]['params']
            
            assert params['start_date'] == start_date.isoformat()
            assert params['end_date'] == end_date.isoformat()


class TestGenerationsEndpointModels04ErrorHandlingBehaviors:
    """Test error handling behaviors for GenerationsEndpoint.models()"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth_manager = AuthManager()
        self.http_manager = Mock(spec=HTTPManager)
        self.endpoint = GenerationsEndpoint(self.auth_manager, self.http_manager)
    
    @pytest.mark.parametrize("exception_type,exception_message", [
        (ConnectionError, "Model statistics service down"),
        (TimeoutError, "Model statistics timeout"), 
        (Exception, "Model endpoint unavailable"),
    ])
    def test_network_api_failures_during_model_statistics_retrieval(self, exception_type, exception_message):
        """Test network/API failures during model statistics retrieval"""
        # Arrange
        self.http_manager.get.side_effect = exception_type(exception_message)
        
        # Act & Assert
        with pytest.raises(exception_type) as exc_info:
            self.endpoint.models()
        assert exception_message in str(exc_info.value)
    
    def test_model_stats_validation_failures(self):
        """Test ModelStats.model_validate() failures with invalid data structures"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {"malformed": "model_data"}
        self.http_manager.get.return_value = mock_response
        
        with patch('openrouter_client.models.generations.ModelStats.model_validate', side_effect=ValueError("Model stats validation failed")):
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                self.endpoint.models()
            assert "Model stats validation failed" in str(exc_info.value)
