import pytest
from datetime import datetime
from pydantic import ValidationError

from openrouter_client.models.generations import (
    GenerationUsage, GenerationCost, Generation, GenerationListMeta,
    GenerationList, GenerationListParams, StatsPoint, GenerationStats,
    GenerationStatsParams, ModelStatsPoint, ModelStats, ModelStatsParams
)


class Test_GenerationUsage_01_NominalBehaviors:
    """Tests for nominal behaviors of GenerationUsage."""
    
    @pytest.mark.parametrize("prompt_tokens, completion_tokens, total_tokens", [
        (100, 50, 150),  # All positive values
        (1, 1, 2),  # Minimum positive values
        (999999, 999999, 1999998)  # Large values
    ])
    def test_creation_with_valid_integer_values(self, prompt_tokens, completion_tokens, total_tokens):
        """Test creating GenerationUsage with valid integer values for token fields."""
        usage = GenerationUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
        
        assert usage.prompt_tokens == prompt_tokens
        assert usage.completion_tokens == completion_tokens
        assert usage.total_tokens == total_tokens


class Test_GenerationUsage_02_NegativeBehaviors:
    """Tests for negative behaviors of GenerationUsage."""
    
    @pytest.mark.parametrize("prompt_tokens, completion_tokens, total_tokens, expected_error", [
        ("one hundred", 50, 150, ValidationError),  # Non-integer prompt_tokens
        (100, "fifty", 150, ValidationError),  # Non-integer completion_tokens
        (100, 50, "one hundred fifty", ValidationError),  # Non-integer total_tokens
        (None, 50, 150, ValidationError),   # None for prompt_tokens
        (100, None, 150, ValidationError),  # None for completion_tokens
        (100, 50, None, ValidationError),   # None for total_tokens
    ])
    def test_handling_invalid_input_types(self, prompt_tokens, completion_tokens, total_tokens, expected_error):
        """Test handling of invalid input types for token fields."""
        with pytest.raises(expected_error):
            GenerationUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )


class Test_GenerationUsage_01_NominalBehaviors:
    """Tests for nominal behaviors of GenerationUsage."""
    
    @pytest.mark.parametrize("prompt_tokens, completion_tokens, total_tokens", [
        (100, 50, 150),  # All positive values
        (1, 1, 2),  # Minimum positive values
        (999999, 999999, 1999998)  # Large values
    ])
    def test_creation_with_valid_integer_values(self, prompt_tokens, completion_tokens, total_tokens):
        """Test creating GenerationUsage with valid integer values for token fields."""
        usage = GenerationUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
        
        assert usage.prompt_tokens == prompt_tokens
        assert usage.completion_tokens == completion_tokens
        assert usage.total_tokens == total_tokens

class Test_GenerationUsage_03_BoundaryBehaviors:
    """Tests for boundary behaviors of GenerationUsage."""
    
    @pytest.mark.parametrize("prompt_tokens, completion_tokens, total_tokens", [
        (0, 0, 0),  # All zeros
        (0, 1, 1),  # Zero prompt tokens
        (1, 0, 1),  # Zero completion tokens
    ])
    def test_handling_zero_values(self, prompt_tokens, completion_tokens, total_tokens):
        """Test creating GenerationUsage with zero values for token fields."""
        usage = GenerationUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
        
        assert usage.prompt_tokens == prompt_tokens
        assert usage.completion_tokens == completion_tokens
        assert usage.total_tokens == total_tokens

class Test_GenerationUsage_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of GenerationUsage."""
    
    @pytest.mark.parametrize("prompt_tokens, completion_tokens, total_tokens, expected_error", [
        (-1, 50, 49, ValidationError),  # Negative prompt_tokens
        (100, -50, 50, ValidationError),  # Negative completion_tokens
        (100, 50, -150, ValidationError),  # Negative total_tokens
    ])
    def test_validation_error_handling(self, prompt_tokens, completion_tokens, total_tokens, expected_error):
        """Test validation error handling for required fields with invalid values."""
        with pytest.raises(expected_error):
            GenerationUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )


class Test_GenerationCost_01_NominalBehaviors:
    """Tests for nominal behaviors of GenerationCost."""
    
    @pytest.mark.parametrize("prompt_tokens, completion_tokens, total_tokens", [
        (0.5, 0.3, 0.8),  # Typical positive values
        (1.0, 1.0, 2.0),  # Equal values
        (10.99, 20.99, 31.98),  # Larger values
    ])
    def test_creation_with_valid_float_values(self, prompt_tokens, completion_tokens, total_tokens):
        """Test creating GenerationCost with valid float values for cost fields."""
        cost = GenerationCost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
        
        assert cost.prompt_tokens == prompt_tokens
        assert cost.completion_tokens == completion_tokens
        assert cost.total_tokens == total_tokens

class Test_GenerationCost_02_NegativeBehaviors:
    """Tests for negative behaviors of GenerationCost."""
    
    @pytest.mark.parametrize("prompt_tokens, prompt_tokens, completion_tokens, expected_error", [
        (-0.5, 0.3, -0.2, ValidationError),  # Negative prompt_tokens
        (0.5, -0.3, 0.2, ValidationError),  # Negative completion_tokens
        (0.5, 0.3, -0.8, ValidationError),  # Negative total_tokens
    ])
    def test_handling_negative_and_invalid_cost_values(self, prompt_tokens, completion_tokens, total_tokens, expected_error):
        """Test handling of negative and invalid input types for cost fields."""
        with pytest.raises(expected_error):
            GenerationCost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )

class Test_GenerationCost_03_BoundaryBehaviors:
    """Tests for boundary behaviors of GenerationCost."""
    
    @pytest.mark.parametrize("prompt_tokens, completion_tokens, total_tokens", [
        (0.000001, 0.000001, 0.000002),  # Very small decimal values
        (0.0, 0.0, 0.0),  # All zeros
        (0.0, 0.1, 0.1),  # Zero prompt_tokens
        (0.1, 0.0, 0.1),  # Zero completion_tokens
    ])
    def test_handling_very_small_decimal_values(self, prompt_tokens, completion_tokens, total_tokens):
        """Test creating GenerationCost with very small decimal values."""
        cost = GenerationCost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
        
        assert cost.prompt_tokens == prompt_tokens
        assert cost.completion_tokens == completion_tokens
        assert cost.total_tokens == total_tokens

class Test_GenerationCost_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of GenerationCost."""
    
    @pytest.mark.parametrize("prompt_tokens, completion_tokens, completion_tokens, expected_error", [
        (None, 0.3, 0.8, ValidationError),  # None for prompt_tokens
        (0.5, None, 0.8, ValidationError),  # None for completion_tokens
        (0.5, 0.3, None, ValidationError),  # None for total_tokens
    ])
    def test_validation_error_handling_for_required_fields(self, prompt_tokens, completion_tokens, total_tokens, expected_error):
        """Test validation error handling for required cost fields."""
        with pytest.raises(expected_error):
            GenerationCost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )


class Test_Generation_01_NominalBehaviors:
    """Tests for nominal behaviors of Generation."""
    
    @pytest.mark.parametrize("id, model, status, created, usage_data, cost_data", [
        (
            "gen_123",
            "gpt-4",
            "completed",
            datetime(2023, 1, 1).timestamp(),
            GenerationUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            GenerationCost(prompt_tokens=0.5, completion_tokens=0.3, total_tokens=0.8)
        ),
        (
            "gen_456",
            "claude-3",
            "completed",
            datetime(2023, 2, 1).timestamp(),
            GenerationUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300),
            GenerationCost(prompt_tokens=1.0, completion_tokens=0.5, total_tokens=1.5)
        ),
    ])
    def test_creation_with_all_required_fields(self, id, model, status, created, usage_data, cost_data):
        """Test creating Generation with all required fields populated correctly."""
        generation = Generation(
            id=id,
            model=model,
            status=status,
            created=created,
            usage=usage_data,
            cost=cost_data
        )
        
        assert generation.id == id
        assert generation.model == model
        assert generation.status == status
        assert generation.created == created
        assert generation.usage == usage_data
        assert generation.cost == cost_data

class Test_Generation_02_NegativeBehaviors:
    """Tests for negative behaviors of Generation."""
    
    @pytest.mark.parametrize("id, model, status, created, usage_data, cost_data, expected_error", [
        (
            None, "gpt-4", "completed", datetime(2023, 1, 1).timestamp(),
            GenerationUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150), 
            GenerationCost(prompt_tokens=0.5, completion_tokens=0.3, total_tokens=0.8), 
            ValidationError
        ),  # Missing id
        (
            "gen_123", None, "completed", datetime(2023, 1, 1).timestamp(),
            GenerationUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150), 
            GenerationCost(prompt_tokens=0.5, completion_tokens=0.3, total_tokens=0.8), 
            ValidationError
        ),  # Missing model
    ])
    def test_handling_missing_required_fields(self, id, model, status, created, usage_data, cost_data, expected_error):
        """Test handling of missing required fields in Generation."""
        with pytest.raises(expected_error):
            Generation(
                id=id,
                model=model,
                status=status,
                created=created,
                usage=usage_data,
                cost=cost_data
            )

class Test_Generation_03_BoundaryBehaviors:
    """Tests for boundary behaviors of Generation."""
    
    @pytest.mark.parametrize("created", [
        0,  # Unix epoch start
        int(datetime(2100, 12, 31).timestamp()),  # Far future date
    ])
    def test_unusual_timestamp_values(self, created):
        """Test creating Generation with unusual timestamp values."""
        generation = Generation(
            id="gen_123",
            model="gpt-4",
            status="completed",
            created=created,
            usage=GenerationUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150), 
            cost=GenerationCost(prompt_tokens=0.5, completion_tokens=0.3, total_tokens=0.8)
        )
        
        assert generation.created == created

class Test_Generation_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of Generation."""
    
    @pytest.mark.parametrize("usage_data, cost_data, expected_error", [
        (
            None, GenerationCost(prompt_tokens=0.5, completion_tokens=0.3, total_tokens=0.8), ValidationError
        ),  # Missing usage
        (
            GenerationUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150), None, ValidationError
        ),  # Missing cost
    ])
    def test_validation_of_complex_nested_structures(self, usage_data, cost_data, expected_error):
        """Test validation of complex nested structures in Generation."""
        with pytest.raises(expected_error):
            Generation(
                id="gen_123",
                model="gpt-4",
                status="completed",
                created=int(datetime(2023, 1, 1).timestamp()),
                usage=usage_data,
                cost=cost_data
            )

class Test_Generation_05_StateTransitionBehaviors:
    """Tests for state transition behaviors of Generation."""
    
    @pytest.mark.parametrize("initial_status, new_status, error_info", [
        ("pending", "completed", None),  # Successful completion
        ("pending", "failed", {"reason": "timeout"}),  # Failure with error info
    ])
    def test_status_transitions(self, initial_status, new_status, error_info):
        """Test status transitions and associated error information handling."""
        generation = Generation(
            id="gen_123",
            model="gpt-4",
            status=initial_status,
            created=int(datetime(2023, 1, 1).timestamp()),
            usage=GenerationUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150), 
            cost=GenerationCost(prompt_tokens=0.5, completion_tokens=0.3, total_tokens=0.8)
        )
        
        if new_status == "failed" and initial_status != "completed":
            generation.status = new_status
            generation.error = error_info
            assert generation.status == new_status
            assert generation.error == error_info
        elif new_status == "completed":
            generation.status = new_status
            assert generation.status == new_status
            assert generation.error is None
        else:
            with pytest.raises(ValidationError):
                generation.status = new_status


class Test_GenerationListMeta_01_NominalBehaviors:
    """Tests for nominal behaviors of GenerationListMeta."""
    
    @pytest.mark.parametrize("limit, offset, total", [
        (10, 0, 100),  # Standard pagination
        (20, 20, 100),  # Offset pagination
        (50, 50, 100),  # Near end pagination
    ])
    def test_creation_with_valid_pagination_values(self, limit, offset, total):
        """Test creating GenerationListMeta with valid pagination values."""
        meta = GenerationListMeta(limit=limit, offset=offset, total=total)
        assert meta.limit == limit
        assert meta.offset == offset
        assert meta.total == total

class Test_GenerationListMeta_02_NegativeBehaviors:
    """Tests for negative behaviors of GenerationListMeta."""
    
    @pytest.mark.parametrize("limit, offset, total, expected_error", [
        (10, 101, 100, ValidationError),  # Offset > total
        (-1, 0, 100, ValidationError),  # Negative limit
        (10, -1, 100, ValidationError),  # Negative offset
        (10, 0, -1, ValidationError),  # Negative total
    ])
    def test_handling_logically_inconsistent_values(self, limit, offset, total, expected_error):
        """Test handling of logically inconsistent pagination values."""
        with pytest.raises(expected_error):
            GenerationListMeta(limit=limit, offset=offset, total=total)


class Test_GenerationListMeta_03_BoundaryBehaviors:
    """Tests for boundary behaviors of GenerationListMeta."""
    
    @pytest.mark.parametrize("limit, offset, total", [
        (50, 50, 100),  # Limit + offset equals total
        (0, 0, 0),  # All zeros
    ])
    def test_limit_offset_equal_total(self, limit, offset, total):
        """Test pagination when limit + offset equals total."""
        meta = GenerationListMeta(limit=limit, offset=offset, total=total)
        assert meta.limit == limit
        assert meta.offset == offset
        assert meta.total == total


class Test_GenerationListMeta_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of GenerationListMeta."""
    
    @pytest.mark.parametrize("limit, offset, total, expected_error", [
        (None, 0, 100, ValidationError),  # None for limit
        (10, None, 100, ValidationError),  # None for offset
        (10, 0, None, ValidationError),  # None for total
    ])
    def test_validation_error_handling(self, limit, offset, total, expected_error):
        """Test validation error handling for pagination parameters."""
        with pytest.raises(expected_error):
            GenerationListMeta(limit=limit, offset=offset, total=total)


class Test_GenerationList_01_NominalBehaviors:
    """Tests for nominal behaviors of GenerationList."""
    
    @pytest.mark.parametrize("data, meta", [
        (
            [
                Generation(
                    id="gen_1", model="gpt-4", status="completed",
                    created=int(datetime(2023, 1, 1).timestamp()),
                    usage=GenerationUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150), 
                    cost=GenerationCost(prompt_tokens=0.5, completion_tokens=0.3, total_tokens=0.8)
                )
            ],
            GenerationListMeta(limit=10, offset=0, total=1)
        ),
        (
            [
                Generation(
                    id="gen_1", model="gpt-4", status="completed",
                    created=int(datetime(2023, 1, 1).timestamp()),
                    usage=GenerationUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150), 
                    cost=GenerationCost(prompt_tokens=0.5, completion_tokens=0.3, total_tokens=0.8)
                ),
                Generation(
                    id="gen_2", model="claude-3", status="completed",
                    created=int(datetime(2023, 1, 2).timestamp()),
                    usage=GenerationUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300), 
                    cost=GenerationCost(prompt_tokens=1.0, completion_tokens=0.5, total_tokens=1.5)
                )
            ],
            GenerationListMeta(limit=10, offset=0, total=2)
        ),
    ])
    def test_creation_with_valid_generation_array(self, data, meta):
        """Test creating GenerationList with valid array of Generation objects."""
        gen_list = GenerationList(data=data, meta=meta)
        assert gen_list.data == data
        assert gen_list.meta == meta
        assert len(gen_list.data) <= meta.total


class Test_GenerationList_02_NegativeBehaviors:
    """Tests for negative behaviors of GenerationList."""
    
    def test_creation_with_inconsistent_metadata(self):
        """Test creating GenerationList with inconsistent metadata."""
        data = [
            Generation(
                id="gen_1", model="gpt-4", status="completed",
                created=int(datetime(2023, 1, 1).timestamp()),
                usage=GenerationUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150), 
                cost=GenerationCost(prompt_tokens=0.5, completion_tokens=0.3, total_tokens=0.8)
            )
        ]
        meta = GenerationListMeta(limit=10, offset=0, total=5)  # Total inconsistent with data length
        
        with pytest.raises(ValidationError):
            GenerationList(data=data, meta=meta)


class Test_GenerationList_03_BoundaryBehaviors:
    """Tests for boundary behaviors of GenerationList."""
    
    def test_handling_empty_data_arrays(self):
        """Test creating GenerationList with empty data arrays."""
        data = []
        meta = GenerationListMeta(limit=10, offset=0, total=0)
        gen_list = GenerationList(data=data, meta=meta)
        assert len(gen_list.data) == 0
        assert gen_list.meta.total == 0

class Test_GenerationList_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of GenerationList."""
    
    def test_handling_validation_errors_in_nested_generations(self):
        """Test handling validation errors in nested Generation objects."""
        data = [
            Generation(
                id="gen_1", model="gpt-4", status="completed",
                created=int(datetime(2023, 1, 1).timestamp()),
                usage=GenerationUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),  # Invalid usage
                cost=GenerationCost(prompt_tokens=0.5, completion_tokens=0.3, total_tokens=0.8)
            ),
            None
        ]
        meta = GenerationListMeta(limit=10, offset=0, total=1)
        
        with pytest.raises(ValidationError):
            GenerationList(data=data, meta=meta)


class Test_GenerationListParams_01_NominalBehaviors:
    """Tests for nominal behaviors of GenerationListParams."""
    
    @pytest.mark.parametrize("model, start_date, end_date, limit, offset", [
        (None, None, None, 10, 0),  # No filters
        ("gpt-4", datetime(2023, 1, 1), datetime(2023, 12, 31), 20, 0),  # All filters
        ("gpt-3.5", datetime(2023, 1, 1), None, 10, 10),  # Partial filters
    ])
    def test_creation_with_various_filter_combinations(self, model, start_date, end_date, limit, offset):
        """Test creating GenerationListParams with various combinations of optional filter parameters."""
        params = GenerationListParams(
            model=model, start_date=start_date,
            end_date=end_date, limit=limit, offset=offset
        )
        assert params.model == model
        assert params.start_date == start_date
        assert params.end_date == end_date
        assert params.limit == limit
        assert params.offset == offset


class Test_GenerationListParams_02_NegativeBehaviors:
    """Tests for negative behaviors of GenerationListParams."""
    
    def test_creation_with_end_date_before_start_date(self):
        """Test creating GenerationListParams with end_date before start_date."""
        with pytest.raises(ValidationError):
            GenerationListParams(
                model=None,
                start_date=datetime(2023, 12, 31),
                end_date=datetime(2023, 1, 1),
                limit=10,
                offset=0
            )


class Test_GenerationListParams_03_BoundaryBehaviors:
    """Tests for boundary behaviors of GenerationListParams."""
    
    def test_identical_start_and_end_date(self):
        """Test creating GenerationListParams with identical start_date and end_date."""
        params = GenerationListParams(
            model="gpt-4",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 1),
            limit=10,
            offset=0
        )
        assert params.start_date == datetime(2023, 1, 1)
        assert params.end_date == datetime(2023, 1, 1)

class Test_GenerationListParams_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of GenerationListParams."""
    
    @pytest.mark.parametrize("model, start_date, end_date, limit, offset, expected_error", [
        ("gpt-4", "invalid_date", None, 10, 0, ValidationError),  # Invalid start_date format
        ("gpt-4", datetime(2023, 1, 1), "invalid_date", 10, 0, ValidationError),  # Invalid end_date format
        ("gpt-4", datetime(2023, 1, 1), None, -1, 0, ValidationError),  # Negative limit
        ("gpt-4", datetime(2023, 1, 1), None, 10, -1, ValidationError),  # Negative offset
    ])
    def test_date_parsing_validation(self, model, start_date, end_date, limit, offset, expected_error):
        """Test date parsing validation and invalid pagination parameters."""
        with pytest.raises(expected_error):
            GenerationListParams(
                model=model,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                offset=offset
            )


class Test_StatsPoint_01_NominalBehaviors:
    """Tests for nominal behaviors of StatsPoint."""
    
    @pytest.mark.parametrize("date, count, tokens, cost", [
        ("2023-01-01", 10, 100, 0.8),  # Standard values
        ("2023-12-31", 1, 3, 0.02),  # Small values
    ])
    def test_creation_with_valid_date_and_statistics(self, date, count, tokens, cost):
        """Test creating StatsPoint with valid date string and statistics values."""
        stats_point = StatsPoint(
            date=date,
            count=count,
            tokens=tokens,
            cost=cost
        )
        assert stats_point.date == date
        assert stats_point.count == count
        assert stats_point.tokens == tokens
        assert stats_point.cost == cost

class Test_StatsPoint_02_NegativeBehaviors:
    """Tests for negative behaviors of StatsPoint."""
    
    @pytest.mark.parametrize("date, count, tokens, cost, expected_error", [
        ("2023-01-01", -1, 100, 0.8, ValidationError),  # Negative count
        ("2023-01-01", 10, -100, 0.8, ValidationError),  # Negative tokens
        ("2023-01-01", 10, 100, -0.8, ValidationError),  # Negative cost
    ])
    def test_creation_with_negative_values(self, date, count, tokens, cost, expected_error):
        """Test creating StatsPoint with negative values for count, tokens, or cost."""
        with pytest.raises(expected_error):
            StatsPoint(
                date=date,
                count=count,
                tokens=tokens,
                cost=cost
            )

class Test_StatsPoint_03_BoundaryBehaviors:
    """Tests for boundary behaviors of StatsPoint."""
    
    @pytest.mark.parametrize("date, count, tokens, cost", [
        ("2023-01-01", 0, 0, 0.0),  # All zero values
    ])
    def test_creation_with_zero_values(self, date, count, tokens, cost):
        """Test creating StatsPoint with zero values for statistics."""
        stats_point = StatsPoint(
            date=date,
            count=count,
            tokens=tokens,
            cost=cost
        )
        assert stats_point.count == 0
        assert stats_point.tokens == 0
        assert stats_point.cost == 0.0

class Test_StatsPoint_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of StatsPoint."""
    
    @pytest.mark.parametrize("date, expected_error", [
        ("invalid-date", ValidationError),  # Invalid date format
        ("2023|01|01", ValidationError),  # Invalid separator
        ("", ValidationError),  # Empty string
    ])
    def test_date_format_validation(self, date, expected_error):
        """Test date format validation for StatsPoint."""
        with pytest.raises(expected_error):
            StatsPoint(
                date=date,
                count=10,
                tokens=100,
                cost=0.8
            )


class Test_GenerationStats_01_NominalBehaviors:
    """Tests for nominal behaviors of GenerationStats."""
    
    @pytest.mark.parametrize("period, data", [
        (
            "day",
            [
                StatsPoint(
                    date="2023-01-01", 
                    count=10, 
                    tokens=100, 
                    cost=0.8
                ),
                StatsPoint(
                    date="2023-01-02",
                    count=5,
                    tokens=75,
                    cost=0.4
                )
            ]
        ),
        (
            "month",
            [
                StatsPoint(
                    date="2023-01",
                    count=100,
                    tokens=1500,
                    cost=8.0
                )
            ]
        ),
    ])
    def test_creation_with_valid_period_and_data(self, period, data):
        """Test creating GenerationStats with valid period value and data array."""
        stats = GenerationStats(period=period, data=data)
        assert stats.period == period
        assert stats.data == data

class Test_GenerationStats_02_NegativeBehaviors:
    """Tests for negative behaviors of GenerationStats."""
    
    @pytest.mark.parametrize("period, expected_error", [
        ("invalid", ValidationError),  # Invalid period string
        ("", ValidationError),  # Empty period string
    ])
    def test_creation_with_invalid_period(self, period, expected_error):
        """Test creating GenerationStats with invalid period strings."""
        with pytest.raises(expected_error):
            GenerationStats(
                period=period,
                data=[StatsPoint(
                    date="2023-01-01",
                    count=10,
                    tokens=100,
                    cost=0.8
                )]
            )

class Test_GenerationStats_03_BoundaryBehaviors:
    """Tests for boundary behaviors of GenerationStats."""
    
    @pytest.mark.parametrize("data", [
        ([]),  # Empty data array
        (
            [
                StatsPoint(
                    date="2023-01-01",
                    count=10,
                    tokens=100,
                    cost=0.8
                ),
                StatsPoint(
                    date="2023-01-02",
                    count=5,
                    tokens=75,
                    cost=0.4
                ),
                StatsPoint(
                    date="2023-01-03",
                    count=15,
                    tokens=225,
                    cost=1.2
                )
            ]
        ),  # Multiple data points
    ])
    def test_varying_data_array_sizes(self, data):
        """Test creating GenerationStats with varying data array sizes."""
        stats = GenerationStats(period="day", data=data)
        assert stats.data == data
        assert len(stats.data) == len(data)

class Test_GenerationStats_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of GenerationStats."""
    
    def test_validation_of_period_field(self):
        """Test validation of period field with incorrect values."""
        with pytest.raises(ValidationError):
            GenerationStats(period="decade", data=[])  # Assuming 'week' is not a valid period


class Test_GenerationStatsParams_01_NominalBehaviors:
    """Tests for nominal behaviors of GenerationStatsParams."""
    
    @pytest.mark.parametrize("period, start_date, end_date", [
        ("day", datetime(2023, 1, 1), datetime(2023, 1, 31)),  # Daily stats
        ("month", datetime(2023, 1, 1), datetime(2023, 12, 31)),  # Monthly stats
        ("year", datetime(2022, 1, 1), datetime(2023, 12, 31)),  # Yearly stats
    ])
    def test_creation_with_different_period_values(self, period, start_date, end_date):
        """Test creating GenerationStatsParams with different period values."""
        params = GenerationStatsParams(period=period, start_date=start_date, end_date=end_date)
        assert params.period == period
        assert params.start_date == start_date
        assert params.end_date == end_date

class Test_GenerationStatsParams_02_NegativeBehaviors:
    """Tests for negative behaviors of GenerationStatsParams."""
    
    def test_creation_with_end_date_before_start_date(self):
        """Test creating GenerationStatsParams with end_date before start_date."""
        with pytest.raises(ValidationError):
            GenerationStatsParams(
                period="day",
                start_date=datetime(2023, 12, 31),
                end_date=datetime(2023, 1, 1)
            )

class Test_GenerationStatsParams_03_BoundaryBehaviors:
    """Tests for boundary behaviors of GenerationStatsParams."""
    
    @pytest.mark.parametrize("period, start_date, end_date", [
        ("month", datetime(2023, 1, 1), datetime(2023, 2, 1)),  # Month boundary
        ("day", datetime(2023, 12, 31), datetime(2024, 1, 1)),  # Year boundary
    ])
    def test_period_transitions(self, period, start_date, end_date):
        """Test period transitions like month or year boundaries."""
        params = GenerationStatsParams(period=period, start_date=start_date, end_date=end_date)
        assert params.start_date == start_date
        assert params.end_date == end_date

class Test_GenerationStatsParams_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of GenerationStatsParams."""
    
    @pytest.mark.parametrize("period, start_date, end_date, expected_error", [
        ("invalid", datetime(2023, 1, 1), datetime(2023, 1, 31), ValidationError),  # Invalid period
        ("day", "invalid_date", datetime(2023, 1, 31), ValidationError),  # Invalid start_date
        ("day", datetime(2023, 1, 1), "invalid_date", ValidationError),  # Invalid end_date
    ])
    def test_validation_for_inconsistent_date_ranges(self, period, start_date, end_date, expected_error):
        """Test handling validation for inconsistent date ranges and invalid inputs."""
        with pytest.raises(expected_error):
            GenerationStatsParams(period=period, start_date=start_date, end_date=end_date)


class Test_ModelStatsPoint_01_NominalBehaviors:
    """Tests for nominal behaviors of ModelStatsPoint."""
    
    @pytest.mark.parametrize("model, count, tokens, cost", [
        ("gpt-4", 10, 150, 0.8),  # Standard values
        ("claude-3", 5, 75, 0.4),  # Different model
    ])
    def test_creation_with_valid_model_and_statistics(self, model, count, tokens, cost):
        """Test creating ModelStatsPoint with valid model identifier and statistics values."""
        point = ModelStatsPoint(
            model=model,
            count=count,
            tokens=tokens,
            cost=cost
        )
        assert point.model == model
        assert point.count == count
        assert point.tokens == tokens
        assert point.cost == cost

class Test_ModelStatsPoint_02_NegativeBehaviors:
    """Tests for negative behaviors of ModelStatsPoint."""
    
    @pytest.mark.parametrize("model, expected_error", [
        ("", ValidationError),  # Empty model identifier
        (None, ValidationError),  # None model identifier
    ])
    def test_creation_with_empty_model_identifier(self, model, expected_error):
        """Test creating ModelStatsPoint with empty or None model identifier."""
        with pytest.raises(expected_error):
            ModelStatsPoint(
                model=model,
                count=10,
                tokens=150,
                cost=0.8
            )

class Test_ModelStatsPoint_03_BoundaryBehaviors:
    """Tests for boundary behaviors of ModelStatsPoint."""
    
    @pytest.mark.parametrize("model, count, tokens, cost", [
        ("gpt-4", 0, 0, 0.0),  # Zero statistics values
    ])
    def test_creation_with_zero_statistics_values(self, model, count, tokens, cost):
        """Test creating ModelStatsPoint with zero values for statistics."""
        point = ModelStatsPoint(
            model=model,
            count=count,
            tokens=tokens,
            cost=cost
        )
        assert point.count == 0
        assert point.tokens == 0
        assert point.cost == 0.0

class Test_ModelStatsPoint_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of ModelStatsPoint."""
    
    @pytest.mark.parametrize("model, count, tokens, cost, expected_error", [
        ("gpt-4", None, 150, 0.8, ValidationError),  # Missing count
        ("gpt-4", 10, -150, 0.8, ValidationError),  # Negative tokens
        ("gpt-4", 10, 150, -0.8, ValidationError),  # Negative cost
    ])
    def test_validation_errors_for_missing_or_invalid_fields(self, model, count, tokens, cost, expected_error):
        """Test handling validation errors when required fields are missing or invalid."""
        with pytest.raises(expected_error):
            ModelStatsPoint(
                model=model,
                count=count,
                tokens=tokens,
                cost=cost
            )


class Test_ModelStats_01_NominalBehaviors:
    """Tests for nominal behaviors of ModelStats."""
    
    @pytest.mark.parametrize("data", [
        (
            [
                ModelStatsPoint(
                    model="gpt-4",
                    count=10,
                    tokens=150,
                    cost=0.8
                ),
                ModelStatsPoint(
                    model="claude-3",
                    count=5,
                    tokens=75,
                    cost=0.4
                )
            ]
        ),
        (
            [
                ModelStatsPoint(
                    model="gpt-4",
                    count=10,
                    tokens=150,
                    cost=0.8
                )
            ]
        ),
    ])
    def test_creation_with_valid_array_of_model_stats_points(self, data):
        """Test creating ModelStats with valid array of ModelStatsPoint objects."""
        stats = ModelStats(data=data)
        assert stats.data == data

class Test_ModelStats_02_NegativeBehaviors:
    """Tests for negative behaviors of ModelStats."""
    
    def test_creation_with_invalid_data(self):
        """Test creating ModelStats with empty data array."""
        with pytest.raises(ValidationError):
            ModelStats(data="[]")


class Test_ModelStats_03_BoundaryBehaviors:
    """Tests for boundary behaviors of ModelStats."""
    
    @pytest.mark.parametrize("data", [
        (
            [
                ModelStatsPoint(
                    model="gpt-4",
                    count=10,
                    tokens=150,
                    cost=0.8
                ),
                ModelStatsPoint(
                    model="gpt-4",
                    count=5,
                    tokens=75,
                    cost=0.4
                )
            ]
        ),  # Multiple entries for same model
    ])
    def test_data_with_multiple_entries_for_same_model(self, data):
        """Test creating ModelStats with multiple entries for the same model."""
        stats = ModelStats(data=data)
        assert stats.data == data
        assert len(stats.data) == 2


class Test_ModelStats_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of ModelStats."""
    
    def test_validation_of_invalid_data(self):
        """Test validation of nested ModelStatsPoint objects with invalid data."""
        data = [
            ModelStatsPoint(
                model="gpt-4",
                count=10,
                tokens=150,
                cost=0.8
            ),
            None,  # Invalid ModelStatsPoint object
        ]
        with pytest.raises(ValidationError):
            ModelStats(data=data)


class Test_ModelStatsParams_01_NominalBehaviors:
    """Tests for nominal behaviors of ModelStatsParams."""
    
    @pytest.mark.parametrize("start_date, end_date", [
        (datetime(2023, 1, 1), datetime(2023, 12, 31)),  # Full year range
        (datetime(2023, 6, 1), datetime(2023, 6, 30)),  # Single month range
        (None, None),  # No date filters
    ])
    def test_creation_with_different_combinations_of_date_filters(self, start_date, end_date):
        """Test creating ModelStatsParams with different combinations of date filters."""
        params = ModelStatsParams(start_date=start_date, end_date=end_date)
        assert params.start_date == start_date
        assert params.end_date == end_date

class Test_ModelStatsParams_02_NegativeBehaviors:
    """Tests for negative behaviors of ModelStatsParams."""
    
    def test_creation_with_end_date_before_start_date(self):
        """Test creating ModelStatsParams with end_date before start_date."""
        with pytest.raises(ValidationError):
            ModelStatsParams(
                start_date=datetime(2023, 12, 31),
                end_date=datetime(2023, 1, 1)
            )

class Test_ModelStatsParams_03_BoundaryBehaviors:
    """Tests for boundary behaviors of ModelStatsParams."""
    
    def test_creation_with_identical_start_and_end_date(self):
        """Test creating ModelStatsParams with identical start_date and end_date values."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 1)
        params = ModelStatsParams(start_date=start_date, end_date=end_date)
        assert params.start_date == start_date
        assert params.end_date == end_date

class Test_ModelStatsParams_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of ModelStatsParams."""
    
    @pytest.mark.parametrize("start_date, end_date, expected_error", [
        ("invalid_date", datetime(2023, 12, 31), ValidationError),  # Invalid start_date format
        (datetime(2023, 1, 1), "invalid_date", ValidationError),  # Invalid end_date format
    ])
    def test_date_parsing_validation(self, start_date, end_date, expected_error):
        """Test date parsing validation for ModelStatsParams."""
        with pytest.raises(expected_error):
            ModelStatsParams(start_date=start_date, end_date=end_date)
