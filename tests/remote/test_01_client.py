import pytest
import os

from openrouter_client.client import OpenRouterClient
from openrouter_client.exceptions import APIError, AuthenticationError

# Skip all tests if API key is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY environment variable not set"
)

@pytest.fixture(scope="session")
def client():
    """
    Shared real client instance using the environment API key.
    """
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY environment variable not set")
    return OpenRouterClient()

@pytest.fixture(scope="session")
def client_with_provisioning(provisioning_api_key):
    """
    Client instance with provisioning API key for credits operations.
    """
    return OpenRouterClient()

@pytest.fixture
def bad_client():
    """
    Client with an invalid base_url to trigger network errors.
    """
    # Temporarily set a dummy API key if none exists to avoid AuthenticationError
    original_key = os.environ.get("OPENROUTER_API_KEY")
    if not original_key:
        os.environ["OPENROUTER_API_KEY"] = "test-key"
    try:
        client = OpenRouterClient(base_url="https://invalid.openrouter.ai/api/v1")
        return client
    finally:
        if not original_key:
            del os.environ["OPENROUTER_API_KEY"]


# ----------------------------------------
# Tests for refresh_context_lengths()
# ----------------------------------------

class Test_OpenRouterClient_RefreshContextLengths_01_NominalBehaviors:
    def test_returns_non_empty_dict_with_int_values(self, client):
        # Arrange (client fixture)
        # Act
        result = client.refresh_context_lengths()
        # Assert
        assert isinstance(result, dict)
        assert result, "Expected at least one model context length"
        assert all(isinstance(k, str) for k in result.keys())
        assert all(isinstance(v, int) for v in result.values())

class Test_OpenRouterClient_RefreshContextLengths_02_NegativeBehaviors:
    @pytest.mark.parametrize("invalid_key", [None, ""])
    def test_invalid_api_key_raises_auth_error(self, invalid_key):
        # Arrange
        original_key = os.environ.get("OPENROUTER_API_KEY")
        try:
            if original_key:
                del os.environ["OPENROUTER_API_KEY"]
            if invalid_key == "":
                os.environ["OPENROUTER_API_KEY"] = invalid_key
            # None or empty string causes AuthenticationError during init
            with pytest.raises(AuthenticationError):
                c = OpenRouterClient()
        finally:
            # Restore original env
            if original_key:
                os.environ["OPENROUTER_API_KEY"] = original_key
            elif "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]
                
    def test_invalid_api_key_still_returns_models(self):
        # Arrange
        original_key = os.environ.get("OPENROUTER_API_KEY")
        try:
            # Set an invalid key in env
            os.environ["OPENROUTER_API_KEY"] = "invalid_api_key"
            c = OpenRouterClient()
            # Act - OpenRouter API allows listing models without valid auth
            result = c.refresh_context_lengths()
            # Assert - Should still return model data
            assert isinstance(result, dict)
            assert len(result) > 0, "Expected models even with invalid key"
        finally:
            # Restore original env
            if original_key:
                os.environ["OPENROUTER_API_KEY"] = original_key
            elif "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]

class Test_OpenRouterClient_RefreshContextLengths_03_BoundaryBehaviors:
    def test_preserves_catalog_context_length_boundaries(self, client):
        # Catalog sizes change as models retire; compare with the live catalog.
        models = [m for m in client.models.list(details=True).data if m.context_length]
        assert models
        lengths = client.refresh_context_lengths()
        for model in (
            min(models, key=lambda m: m.context_length),
            max(models, key=lambda m: m.context_length),
        ):
            assert lengths[model.id] == model.context_length

class Test_OpenRouterClient_RefreshContextLengths_04_ErrorHandlingBehaviors:
    def test_network_error_raises_api_error(self, bad_client):
        # Arrange (bad_client fixture)
        # Act / Assert
        with pytest.raises(APIError):
            bad_client.refresh_context_lengths()

class Test_OpenRouterClient_RefreshContextLengths_05_StateTransitionBehaviors:
    def test_registry_persists_between_calls(self, client):
        # Arrange
        first = client.refresh_context_lengths()
        # Act
        second = client.refresh_context_lengths()
        # Assert
        assert first == second, "Registry should remain consistent across refreshes"
        assert first is not second, "Should return a fresh copy each time"


# ----------------------------------------
# Tests for calculate_rate_limits()
# ----------------------------------------

class Test_OpenRouterClient_CalculateRateLimits_01_NominalBehaviors:
    def test_returns_rate_limit_config_with_expected_keys_and_types(self, client_with_provisioning):
        # Arrange / Act
        rl = client_with_provisioning.calculate_rate_limits()
        # Assert
        assert isinstance(rl, dict)
        assert set(rl.keys()) == {"requests", "period", "cooldown"}
        assert isinstance(rl["requests"], int) and rl["requests"] >= 1
        assert rl["period"] == 60
        assert isinstance(rl["cooldown"], (int, float)) and rl["cooldown"] >= 0
    
    def test_requires_provisioning_key(self):
        # The library reads account credits, which require provisioning auth.
        with OpenRouterClient(provisioning_api_key="") as client:
            with pytest.raises(APIError) as exc_info:
                client.calculate_rate_limits()
        assert isinstance(exc_info.value.__cause__, AuthenticationError)

class Test_OpenRouterClient_CalculateRateLimits_02_NegativeBehaviors:
    @pytest.mark.parametrize("invalid_key", [None, "", "invalid_api_key"])
    def test_invalid_api_key_raises_api_error(self, invalid_key):
        # Arrange
        original_key = os.environ.get("OPENROUTER_API_KEY")
        try:
            if invalid_key is None or invalid_key == "":
                # Clear env var to test None/empty behavior
                if original_key:
                    del os.environ["OPENROUTER_API_KEY"]
                if invalid_key == "":
                    os.environ["OPENROUTER_API_KEY"] = invalid_key
                # None or empty string causes AuthenticationError during init
                with pytest.raises(AuthenticationError):
                    c = OpenRouterClient()
            else:
                # Set the invalid key in env
                os.environ["OPENROUTER_API_KEY"] = invalid_key
                with OpenRouterClient(provisioning_api_key="invalid_key") as c:
                    with pytest.raises(APIError):
                        c.calculate_rate_limits()
        finally:
            # Restore original env
            if original_key:
                os.environ["OPENROUTER_API_KEY"] = original_key
            elif "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]

class Test_OpenRouterClient_CalculateRateLimits_03_BoundaryBehaviors:
    def test_requests_per_period_matches_credits_division_rule(self, client_with_provisioning):
        # Arrange
        credits_info = client_with_provisioning.credits.get()
        remaining = credits_info.get("remaining", 0)
        # Act
        rl = client_with_provisioning.calculate_rate_limits()
        expected = max(1, int(remaining / 10))
        # Assert
        assert rl["requests"] == expected

class Test_OpenRouterClient_CalculateRateLimits_04_ErrorHandlingBehaviors:
    def test_network_error_raises_api_error(self, bad_client):
        # Arrange (bad_client fixture)
        # Act / Assert
        with pytest.raises(APIError):
            bad_client.calculate_rate_limits()

class Test_OpenRouterClient_CalculateRateLimits_05_StateTransitionBehaviors:
    def test_rate_limit_dicts_are_fresh_each_call(self, client_with_provisioning):
        # Arrange
        first = client_with_provisioning.calculate_rate_limits()
        # Act
        second = client_with_provisioning.calculate_rate_limits()
        # Assert
        assert first.keys() == second.keys()
        assert first is not second, "Each call should return a new dict object"
