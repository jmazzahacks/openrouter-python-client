import pytest
import os
from datetime import datetime, timedelta
from typing import Optional, Union
import time

from openrouter_client import OpenRouterClient
from openrouter_client.http import HTTPManager
from openrouter_client.auth import AuthManager
from openrouter_client.endpoints.generations import GenerationsEndpoint
from openrouter_client.models.generations import Generation, GenerationList, GenerationStats, ModelStats
from openrouter_client.exceptions import APIError


@pytest.fixture(scope="session")
def http_manager():
    """Shared OpenRouter client instance for all tests."""
    return HTTPManager()

@pytest.fixture(scope="session")
def auth_manager():
    """Shared AuthManager instance for all tests."""
    return AuthManager()


@pytest.fixture
def generations_endpoint(http_manager, auth_manager):
    """Generations endpoint instance for testing."""
    return GenerationsEndpoint(auth_manager, http_manager)


@pytest.fixture
def sample_dates():
    """Sample date ranges for testing."""
    now = datetime.now()
    return {
        'yesterday': now - timedelta(days=1),
        'week_ago': now - timedelta(days=7),
        'month_ago': now - timedelta(days=30),
        'year_ago': now - timedelta(days=365),
        'now': now
    }


class Test_GenerationsEndpoint_List_01_NominalBehaviors:
    """Test nominal behaviors for the list method."""
    
    @pytest.mark.parametrize("limit,offset,model", [
        (10, 0, None),
        (50, 20, "gpt-4"),
        (1, 0, "claude-3"),
        (100, 0, None),
        (25, 10, "anthropic/claude-3-opus"),
    ])
    def test_list_with_valid_parameters(self, generations_endpoint, limit, offset, model):
        """Test successful GET request with various valid parameter combinations."""
        # Arrange
        expected_limit = limit
        expected_offset = offset
        
        # Act
        result = generations_endpoint.list(limit=limit, offset=offset, model=model)
        
        # Assert
        assert isinstance(result, GenerationList)
        assert hasattr(result, 'data')
        assert hasattr(result, 'total')
        assert len(result.data) <= expected_limit
    
    @pytest.mark.parametrize("start_date,end_date", [
        (datetime.now() - timedelta(days=7), datetime.now()),
        ("2024-01-01T00:00:00Z", "2024-01-31T23:59:59Z"),
        (datetime.now() - timedelta(days=1), None),
        (None, datetime.now()),
    ])
    def test_list_with_date_filtering(self, generations_endpoint, start_date, end_date):
        """Test successful requests with various date filtering scenarios."""
        # Arrange
        # Act
        result = generations_endpoint.list(start_date=start_date, end_date=end_date)
        
        # Assert
        assert isinstance(result, GenerationList)
        assert hasattr(result, 'data')
        
    def test_list_with_datetime_objects(self, generations_endpoint, sample_dates):
        """Test proper datetime to ISO format conversion."""
        # Arrange
        start_date = sample_dates['week_ago']
        end_date = sample_dates['yesterday']
        
        # Act
        result = generations_endpoint.list(start_date=start_date, end_date=end_date)
        
        # Assert
        assert isinstance(result, GenerationList)
        
    def test_list_with_additional_kwargs(self, generations_endpoint):
        """Test handling of additional kwargs parameters."""
        # Arrange
        custom_param = "test_value"
        
        # Act
        result = generations_endpoint.list(custom_parameter=custom_param, limit=10)
        
        # Assert
        assert isinstance(result, GenerationList)


class Test_GenerationsEndpoint_List_02_NegativeBehaviors:
    """Test negative behaviors for the list method."""
    
    def test_list_with_invalid_authentication(self):
        """Test HTTP request with invalid authentication credentials."""
        # Arrange
        invalid_client = OpenRouterClient(api_key="invalid_key")
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            invalid_client.generations.list()
        assert exc_info.value.status_code in [401, 403]
    
    @pytest.mark.parametrize("invalid_date", [
        "not-a-date",
        "2024-13-01",  # Invalid month
        "2024-02-30",  # Invalid day
        "invalid-format",
    ])
    def test_list_with_malformed_dates(self, generations_endpoint, invalid_date):
        """Test request with malformed date strings."""
        # Arrange
        # Act & Assert
        with pytest.raises(APIError):
            generations_endpoint.list(start_date=invalid_date)
    
    @pytest.mark.parametrize("invalid_model", [
        "nonexistent-model-12345",
        "invalid/model/name",
        "",
    ])
    def test_list_with_invalid_model_names(self, generations_endpoint, invalid_model):
        """Test request with invalid model names."""
        # Arrange
        # Act
        result = generations_endpoint.list(model=invalid_model)
        
        # Assert
        assert isinstance(result, GenerationList)
        assert len(result.data) == 0  # Should return empty list for invalid models


class Test_GenerationsEndpoint_List_03_BoundaryBehaviors:
    """Test boundary behaviors for the list method."""
    
    @pytest.mark.parametrize("limit", [0, 1, 1000])
    def test_list_with_boundary_limits(self, generations_endpoint, limit):
        """Test requests with limit values at boundaries."""
        # Arrange
        # Act
        result = generations_endpoint.list(limit=limit)
        
        # Assert
        assert isinstance(result, GenerationList)
        assert len(result.data) <= limit
    
    @pytest.mark.parametrize("offset", [0, 999999])
    def test_list_with_boundary_offsets(self, generations_endpoint, offset):
        """Test requests with offset at boundaries."""
        # Arrange
        # Act
        result = generations_endpoint.list(offset=offset, limit=10)
        
        # Assert
        assert isinstance(result, GenerationList)
    
    def test_list_with_same_start_end_date(self, generations_endpoint, sample_dates):
        """Test date ranges spanning exactly one day."""
        # Arrange
        single_date = sample_dates['yesterday']
        
        # Act
        result = generations_endpoint.list(start_date=single_date, end_date=single_date)
        
        # Assert
        assert isinstance(result, GenerationList)
    
    def test_list_with_maximum_date_span(self, generations_endpoint, sample_dates):
        """Test requests with maximum allowed time spans."""
        # Arrange
        start_date = sample_dates['year_ago']
        end_date = sample_dates['now']
        
        # Act
        result = generations_endpoint.list(start_date=start_date, end_date=end_date)
        
        # Assert
        assert isinstance(result, GenerationList)
    
    def test_list_with_all_none_parameters(self, generations_endpoint):
        """Test requests with all optional parameters set to None."""
        # Arrange
        # Act
        result = generations_endpoint.list(
            limit=None, offset=None, start_date=None, end_date=None, model=None
        )
        
        # Assert
        assert isinstance(result, GenerationList)


class Test_GenerationsEndpoint_List_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for the list method."""
    
    def test_list_network_connectivity_failure(self, generations_endpoint, monkeypatch):
        """Test network connectivity failures during HTTP request."""
        # Arrange
        def mock_network_error(*args, **kwargs):
            raise ConnectionError("Network unreachable")
        
        monkeypatch.setattr(generations_endpoint.http_manager, 'get', mock_network_error)
        
        # Act & Assert
        with pytest.raises(ConnectionError):
            generations_endpoint.list()
    
    def test_list_server_error_response(self, generations_endpoint, monkeypatch):
        """Test API server returning 5xx status codes."""
        # Arrange
        class MockResponse:
            status_code = 500
            def json(self):
                return {"error": "Internal server error"}
        
        def mock_server_error(*args, **kwargs):
            raise APIError("Server error", response=MockResponse())
        
        monkeypatch.setattr(generations_endpoint.http_manager, 'get', mock_server_error)
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            generations_endpoint.list()
        assert "Server error" in str(exc_info.value)
    
    def test_list_malformed_json_response(self, generations_endpoint, monkeypatch):
        """Test malformed JSON responses from API server."""
        # Arrange
        class MockResponse:
            def json(self):
                raise ValueError("Invalid JSON")
        
        monkeypatch.setattr(generations_endpoint.http_manager, 'get', lambda *args, **kwargs: MockResponse())
        
        # Act & Assert
        with pytest.raises(ValueError):
            generations_endpoint.list()


class Test_GenerationsEndpoint_List_05_StateTransitionBehaviors:
    """Test state transition behaviors for the list method."""
    
    def test_list_pagination_consistency(self, generations_endpoint):
        """Test sequential requests with different pagination parameters."""
        # Arrange
        limit = 10
        
        # Act
        first_page = generations_endpoint.list(limit=limit, offset=0)
        second_page = generations_endpoint.list(limit=limit, offset=limit)
        
        # Assert
        assert isinstance(first_page, GenerationList)
        assert isinstance(second_page, GenerationList)
        # Verify no overlap in data between pages
        first_ids = {gen.id for gen in first_page.data}
        second_ids = {gen.id for gen in second_page.data}
        assert first_ids.isdisjoint(second_ids)
    
    def test_list_time_filtered_consistency(self, generations_endpoint, sample_dates):
        """Test requests reflecting changes in generation history over time."""
        # Arrange
        recent_cutoff = sample_dates['yesterday']
        older_cutoff = sample_dates['week_ago']
        
        # Act
        recent_results = generations_endpoint.list(start_date=recent_cutoff)
        older_results = generations_endpoint.list(start_date=older_cutoff)
        
        # Assert
        assert isinstance(recent_results, GenerationList)
        assert isinstance(older_results, GenerationList)
        # Older results should include more generations
        assert older_results.total >= recent_results.total


class Test_GenerationsEndpoint_Get_01_NominalBehaviors:
    """Test nominal behaviors for the get method."""
    
    def test_get_with_valid_generation_id(self, generations_endpoint):
        """Test successful GET request with valid generation_id."""
        # Arrange
        # First get a valid generation ID from list
        generations_list = generations_endpoint.list(limit=1)
        if not generations_list.data:
            pytest.skip("No generations available for testing")
        generation_id = generations_list.data[0].id
        
        # Act
        result = generations_endpoint.get(generation_id)
        
        # Assert
        assert isinstance(result, Generation)
        assert result.id == generation_id
    
    def test_get_requires_provisioning_key(self, client):
        """Test proper authentication headers with provisioning API key requirement."""
        # Arrange
        if not client.provisioning_key:
            pytest.skip("Provisioning key not available")
        
        generations_list = client.generations.list(limit=1)
        if not generations_list.data:
            pytest.skip("No generations available for testing")
        generation_id = generations_list.data[0].id
        
        # Act
        result = client.generations.get(generation_id)
        
        # Assert
        assert isinstance(result, Generation)


class Test_GenerationsEndpoint_Get_02_NegativeBehaviors:
    """Test negative behaviors for the get method."""
    
    def test_get_with_nonexistent_id(self, generations_endpoint):
        """Test HTTP request with non-existent generation_id."""
        # Arrange
        nonexistent_id = "nonexistent-generation-id-12345"
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            generations_endpoint.get(nonexistent_id)
        assert exc_info.value.status_code == 404
    
    def test_get_without_provisioning_key(self):
        """Test request without required provisioning API key."""
        # Arrange
        client_without_provisioning = OpenRouterClient(
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            client_without_provisioning.generations.get("any-id")
        assert exc_info.value.status_code in [401, 403]


class Test_GenerationsEndpoint_Get_03_BoundaryBehaviors:
    """Test boundary behaviors for the get method."""
    
    @pytest.mark.parametrize("generation_id", [
        "a",  # Minimum length
        "a" * 100,  # Long ID
        "id-with-special-chars_123",  # Special characters
    ])
    def test_get_with_boundary_id_lengths(self, generations_endpoint, generation_id):
        """Test requests with minimum and maximum length generation_id strings."""
        # Arrange
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            generations_endpoint.get(generation_id)
        # Should return 404 for invalid IDs
        assert exc_info.value.status_code == 404
    
    @pytest.mark.parametrize("invalid_id", ["", None])
    def test_get_with_empty_or_null_id(self, generations_endpoint, invalid_id):
        """Test requests with empty string or null generation_id values."""
        # Arrange
        # Act & Assert
        with pytest.raises((APIError, ValueError, TypeError)):
            generations_endpoint.get(invalid_id)


class Test_GenerationsEndpoint_Get_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for the get method."""
    
    def test_get_network_failure(self, generations_endpoint, monkeypatch):
        """Test network connectivity failures during specific generation lookup."""
        # Arrange
        def mock_network_error(*args, **kwargs):
            raise ConnectionError("Network unreachable")
        
        monkeypatch.setattr(generations_endpoint.http_manager, 'get', mock_network_error)
        
        # Act & Assert
        with pytest.raises(ConnectionError):
            generations_endpoint.get("any-id")
    
    def test_get_authentication_failure(self):
        """Test authentication failures when provisioning key is invalid."""
        # Arrange
        invalid_client = OpenRouterClient(
            api_key="invalid-key",
            provisioning_key="invalid-provisioning-key"
        )
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            invalid_client.generations.get("any-id")
        assert exc_info.value.status_code in [401, 403]


class Test_GenerationsEndpoint_Get_05_StateTransitionBehaviors:
    """Test state transition behaviors for the get method."""
    
    def test_get_generation_state_consistency(self, generations_endpoint):
        """Test requests for generation maintaining consistent state."""
        # Arrange
        generations_list = generations_endpoint.list(limit=1)
        if not generations_list.data:
            pytest.skip("No generations available for testing")
        generation_id = generations_list.data[0].id
        
        # Act
        first_fetch = generations_endpoint.get(generation_id)
        time.sleep(1)  # Small delay
        second_fetch = generations_endpoint.get(generation_id)
        
        # Assert
        assert isinstance(first_fetch, Generation)
        assert isinstance(second_fetch, Generation)
        assert first_fetch.id == second_fetch.id


class Test_GenerationsEndpoint_Stats_01_NominalBehaviors:
    """Test nominal behaviors for the stats method."""
    
    @pytest.mark.parametrize("period", ["day", "week", "month", "year"])
    def test_stats_with_valid_periods(self, generations_endpoint, period):
        """Test successful GET request with valid period values."""
        # Arrange
        # Act
        result = generations_endpoint.stats(period=period)
        
        # Assert
        assert isinstance(result, GenerationStats)
        assert hasattr(result, 'period')
    
    @pytest.mark.parametrize("start_date,end_date", [
        (datetime.now() - timedelta(days=30), datetime.now()),
        ("2024-01-01T00:00:00Z", "2024-01-31T23:59:59Z"),
    ])
    def test_stats_with_date_filtering(self, generations_endpoint, start_date, end_date):
        """Test successful requests with date filtering."""
        # Arrange
        # Act
        result = generations_endpoint.stats(
            period="day", start_date=start_date, end_date=end_date
        )
        
        # Assert
        assert isinstance(result, GenerationStats)
    
    def test_stats_datetime_conversion(self, generations_endpoint, sample_dates):
        """Test proper datetime to ISO format conversion."""
        # Arrange
        start_date = sample_dates['week_ago']
        end_date = sample_dates['yesterday']
        
        # Act
        result = generations_endpoint.stats(
            period="day", start_date=start_date, end_date=end_date
        )
        
        # Assert
        assert isinstance(result, GenerationStats)


class Test_GenerationsEndpoint_Stats_02_NegativeBehaviors:
    """Test negative behaviors for the stats method."""
    
    @pytest.mark.parametrize("invalid_period", [
        "invalid", "hour", "second", "", "MONTH"
    ])
    def test_stats_with_invalid_periods(self, generations_endpoint, invalid_period):
        """Test HTTP request with invalid period values."""
        # Arrange
        # Act & Assert
        with pytest.raises(APIError):
            generations_endpoint.stats(period=invalid_period)
    
    def test_stats_authentication_failure(self):
        """Test authentication failures for stats endpoint access."""
        # Arrange
        invalid_client = OpenRouterClient(api_key="invalid-key")
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            invalid_client.generations.stats()
        assert exc_info.value.status_code in [401, 403]
    
    @pytest.mark.parametrize("invalid_date", [
        "not-a-date", "2024-13-01", "invalid-format"
    ])
    def test_stats_with_invalid_date_format(self, generations_endpoint, invalid_date):
        """Test request with invalid date format strings."""
        # Arrange
        # Act & Assert
        with pytest.raises(APIError):
            generations_endpoint.stats(start_date=invalid_date)


class Test_GenerationsEndpoint_Stats_03_BoundaryBehaviors:
    """Test boundary behaviors for the stats method."""
    
    @pytest.mark.parametrize("period", ["day", "year"])
    def test_stats_with_granularity_extremes(self, generations_endpoint, period):
        """Test requests with minimum and maximum period granularity."""
        # Arrange
        # Act
        result = generations_endpoint.stats(period=period)
        
        # Assert
        assert isinstance(result, GenerationStats)
    
    def test_stats_with_equal_start_end_dates(self, generations_endpoint, sample_dates):
        """Test requests with start_date equal to end_date."""
        # Arrange
        single_date = sample_dates['yesterday']
        
        # Act
        result = generations_endpoint.stats(
            period="day", start_date=single_date, end_date=single_date
        )
        
        # Assert
        assert isinstance(result, GenerationStats)
    
    def test_stats_with_reversed_date_range(self, generations_endpoint, sample_dates):
        """Test empty date range where start_date is after end_date."""
        # Arrange
        start_date = sample_dates['yesterday']
        end_date = sample_dates['week_ago']
        
        # Act
        result = generations_endpoint.stats(
            period="day", start_date=start_date, end_date=end_date
        )
        
        # Assert
        assert isinstance(result, GenerationStats)


class Test_GenerationsEndpoint_Stats_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for the stats method."""
    
    def test_stats_network_failure(self, generations_endpoint, monkeypatch):
        """Test network failures during statistics requests."""
        # Arrange
        def mock_network_error(*args, **kwargs):
            raise ConnectionError("Network unreachable")
        
        monkeypatch.setattr(generations_endpoint.http_manager, 'get', mock_network_error)
        
        # Act & Assert
        with pytest.raises(ConnectionError):
            generations_endpoint.stats()
    
    def test_stats_json_parsing_failure(self, generations_endpoint, monkeypatch):
        """Test JSON parsing failures for malformed statistics responses."""
        # Arrange
        class MockResponse:
            def json(self):
                raise ValueError("Invalid JSON")
        
        monkeypatch.setattr(generations_endpoint.http_manager, 'get', lambda *args, **kwargs: MockResponse())
        
        # Act & Assert
        with pytest.raises(ValueError):
            generations_endpoint.stats()


class Test_GenerationsEndpoint_Stats_05_StateTransitionBehaviors:
    """Test state transition behaviors for the stats method."""
    
    def test_stats_period_comparison(self, generations_endpoint):
        """Test statistics showing different patterns across time periods."""
        # Arrange
        # Act
        daily_stats = generations_endpoint.stats(period="day")
        monthly_stats = generations_endpoint.stats(period="month")
        
        # Assert
        assert isinstance(daily_stats, GenerationStats)
        assert isinstance(monthly_stats, GenerationStats)
    
    def test_stats_time_range_progression(self, generations_endpoint, sample_dates):
        """Test statistics reflecting changes over different time ranges."""
        # Arrange
        recent_range = generations_endpoint.stats(
            period="day", 
            start_date=sample_dates['yesterday'],
            end_date=sample_dates['now']
        )
        longer_range = generations_endpoint.stats(
            period="day",
            start_date=sample_dates['week_ago'],
            end_date=sample_dates['now']
        )
        
        # Act & Assert
        assert isinstance(recent_range, GenerationStats)
        assert isinstance(longer_range, GenerationStats)


class Test_GenerationsEndpoint_Models_01_NominalBehaviors:
    """Test nominal behaviors for the models method."""
    
    def test_models_without_date_filters(self, generations_endpoint):
        """Test successful requests without date filters (all-time statistics)."""
        # Arrange
        # Act
        result = generations_endpoint.models()
        
        # Assert
        assert isinstance(result, ModelStats)
        assert hasattr(result, 'models')
    
    @pytest.mark.parametrize("start_date,end_date", [
        (datetime.now() - timedelta(days=7), datetime.now()),
        ("2024-01-01T00:00:00Z", "2024-01-31T23:59:59Z"),
    ])
    def test_models_with_date_filtering(self, generations_endpoint, start_date, end_date):
        """Test successful requests with date filtering."""
        # Arrange
        # Act
        result = generations_endpoint.models(start_date=start_date, end_date=end_date)
        
        # Assert
        assert isinstance(result, ModelStats)
    
    def test_models_datetime_conversion(self, generations_endpoint, sample_dates):
        """Test proper datetime to ISO format conversion."""
        # Arrange
        start_date = sample_dates['week_ago']
        end_date = sample_dates['yesterday']
        
        # Act
        result = generations_endpoint.models(start_date=start_date, end_date=end_date)
        
        # Assert
        assert isinstance(result, ModelStats)


class Test_GenerationsEndpoint_Models_02_NegativeBehaviors:
    """Test negative behaviors for the models method."""
    
    def test_models_authentication_failure(self):
        """Test authentication failures for model statistics endpoint."""
        # Arrange
        invalid_client = OpenRouterClient(api_key="invalid-key")
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            invalid_client.generations.models()
        assert exc_info.value.status_code in [401, 403]
    
    @pytest.mark.parametrize("invalid_date", [
        "not-a-date", "2024-13-01", "invalid-format"
    ])
    def test_models_with_malformed_dates(self, generations_endpoint, invalid_date):
        """Test request with malformed date parameters."""
        # Arrange
        # Act & Assert
        with pytest.raises(APIError):
            generations_endpoint.models(start_date=invalid_date)


class Test_GenerationsEndpoint_Models_03_BoundaryBehaviors:
    """Test boundary behaviors for the models method."""
    
    def test_models_single_day_range(self, generations_endpoint, sample_dates):
        """Test requests with date ranges covering single day."""
        # Arrange
        single_date = sample_dates['yesterday']
        
        # Act
        result = generations_endpoint.models(
            start_date=single_date, end_date=single_date
        )
        
        # Assert
        assert isinstance(result, ModelStats)
    
    def test_models_maximum_timespan(self, generations_endpoint, sample_dates):
        """Test requests with maximum available timespan."""
        # Arrange
        start_date = sample_dates['year_ago']
        end_date = sample_dates['now']
        
        # Act
        result = generations_endpoint.models(start_date=start_date, end_date=end_date)
        
        # Assert
        assert isinstance(result, ModelStats)
    
    def test_models_current_timestamp_boundary(self, generations_endpoint):
        """Test requests with end_date at current timestamp boundary."""
        # Arrange
        current_time = datetime.now()
        
        # Act
        result = generations_endpoint.models(end_date=current_time)
        
        # Assert
        assert isinstance(result, ModelStats)


class Test_GenerationsEndpoint_Models_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for the models method."""
    
    def test_models_network_failure(self, generations_endpoint, monkeypatch):
        """Test network connectivity issues during model statistics retrieval."""
        # Arrange
        def mock_network_error(*args, **kwargs):
            raise ConnectionError("Network unreachable")
        
        monkeypatch.setattr(generations_endpoint.http_manager, 'get', mock_network_error)
        
        # Act & Assert
        with pytest.raises(ConnectionError):
            generations_endpoint.models()
    
    def test_models_json_parsing_failure(self, generations_endpoint, monkeypatch):
        """Test JSON parsing exceptions for corrupted model statistics responses."""
        # Arrange
        class MockResponse:
            def json(self):
                raise ValueError("Invalid JSON")
        
        monkeypatch.setattr(generations_endpoint.http_manager, 'get', lambda *args, **kwargs: MockResponse())
        
        # Act & Assert
        with pytest.raises(ValueError):
            generations_endpoint.models()


class Test_GenerationsEndpoint_Models_05_StateTransitionBehaviors:
    """Test state transition behaviors for the models method."""
    
    def test_models_usage_pattern_changes(self, generations_endpoint, sample_dates):
        """Test statistics reflecting changes in model usage patterns over time periods."""
        # Arrange
        recent_stats = generations_endpoint.models(
            start_date=sample_dates['yesterday'],
            end_date=sample_dates['now']
        )
        historical_stats = generations_endpoint.models(
            start_date=sample_dates['month_ago'],
            end_date=sample_dates['week_ago']
        )
        
        # Act & Assert
        assert isinstance(recent_stats, ModelStats)
        assert isinstance(historical_stats, ModelStats)
    
    def test_models_comprehensive_timespan(self, generations_endpoint, sample_dates):
        """Test statistics showing transition patterns across comprehensive time periods."""
        # Arrange
        # Act
        all_time_stats = generations_endpoint.models()
        recent_stats = generations_endpoint.models(
            start_date=sample_dates['week_ago'],
            end_date=sample_dates['now']
        )
        
        # Assert
        assert isinstance(all_time_stats, ModelStats)
        assert isinstance(recent_stats, ModelStats)
