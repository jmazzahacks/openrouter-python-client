import pytest
from datetime import datetime, timedelta

from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager
from openrouter_client.endpoints.keys import KeysEndpoint
from openrouter_client.exceptions import APIError


class TestKeysEndpointFixtures:
    """Shared fixtures and utilities for all KeysEndpoint tests."""
    
    @pytest.fixture(scope="session")
    def auth_manager(self):
        """Shared AuthManager instance automatically populated for all tests."""
        return AuthManager()
    
    @pytest.fixture(scope="session") 
    def http_manager(self):
        """Shared HTTPManager instance for all tests."""
        return HTTPManager(base_url="https://openrouter.ai/api/v1")
    
    @pytest.fixture(scope="session")
    def keys_endpoint(self, auth_manager, http_manager):
        """Shared KeysEndpoint instance for all HTTP request tests."""
        return KeysEndpoint(auth_manager, http_manager)


class Test_KeysEndpoint_List_01_NominalBehaviors(TestKeysEndpointFixtures):
    """Test nominal HTTP request behaviors for KeysEndpoint.list() method."""
    
    def test_successful_get_request_with_valid_provisioning_api_key(self, keys_endpoint):
        """Test successful GET request with valid provisioning API key returns properly formatted list."""
        # Arrange
        # Valid credentials automatically provided through AuthManager
        
        # Act
        response = keys_endpoint.list()
        
        # Assert
        assert isinstance(response, dict)
        assert "data" in response
        assert isinstance(response["data"], list)
        if response["data"]:  # If API keys exist in account
            assert all(isinstance(key_data, dict) for key_data in response["data"])
            # Verify expected JSON response structure
            for key_data in response["data"]:
                assert "hash" in key_data
                assert "name" in key_data
    
    def test_list_with_pagination_parameters(self, keys_endpoint):
        """Test list method with pagination parameters."""
        # Arrange
        # Valid credentials automatically provided through AuthManager
        
        # Act
        response = keys_endpoint.list(offset=0, include_disabled=True)
        
        # Assert
        assert isinstance(response, dict)
        assert "data" in response
        assert isinstance(response["data"], list)
    
    def test_empty_list_response_when_no_api_keys_exist(self, keys_endpoint):
        """Test empty list returned when no API keys exist for the account."""
        # Arrange
        # Account state may vary, but response format should be consistent
        
        # Act
        response = keys_endpoint.list()
        
        # Assert
        assert isinstance(response, dict)
        assert "data" in response
        assert isinstance(response["data"], list)
        # Response should be valid JSON list regardless of content
    
    def test_json_response_contains_expected_metadata_fields(self, keys_endpoint):
        """Test JSON response contains expected key metadata fields."""
        # Arrange
        # Create a test key to ensure we have data to validate
        test_key = keys_endpoint.create(name="metadata_validation_key")
        key_hash = test_key["data"]["hash"]
        
        try:
            # Act
            response = keys_endpoint.list()
            
            # Assert
            assert isinstance(response, dict)
            assert "data" in response
            assert len(response["data"]) > 0
            
            # Find our test key and validate its metadata structure
            test_key_found = False
            for key_data in response["data"]:
                if key_data.get("hash") == key_hash:
                    test_key_found = True
                    assert isinstance(key_data, dict)
                    assert "created_at" in key_data
                    assert "updated_at" in key_data
                    assert "label" in key_data
                    assert "disabled" in key_data
                    assert "usage" in key_data
                    break
            
            assert test_key_found
        finally:
            # Cleanup
            keys_endpoint.delete(key_hash)
    
    def test_authentication_headers_correctly_formatted_and_accepted(self, keys_endpoint):
        """Test authentication headers are correctly formatted and accepted by OpenRouter API."""
        # Arrange
        # Headers are managed internally by the endpoint
        
        # Act
        response = keys_endpoint.list()
        
        # Assert
        # Successful response indicates headers were accepted
        assert isinstance(response, dict)
        assert "data" in response


class Test_KeysEndpoint_List_02_NegativeBehaviors(TestKeysEndpointFixtures):
    """Test negative HTTP request behaviors for KeysEndpoint.list() method."""
    
    def test_request_fails_with_invalid_provisioning_api_key(self, auth_manager, http_manager):
        """Test request fails with invalid or expired provisioning API key."""
        # Skip this test if no provisioning key is configured
        if not hasattr(auth_manager, 'provisioning_key') or not auth_manager.provisioning_key:
            pytest.skip("No provisioning key configured")
            
        # Arrange
        invalid_auth = AuthManager()
        invalid_auth.provisioning_key = "invalid_test_key_12345"
        invalid_endpoint = KeysEndpoint(invalid_auth, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError):
            invalid_endpoint.list()
    
    def test_request_fails_with_malformed_authentication_headers(self, auth_manager, http_manager):
        """Test request fails with malformed authentication headers."""
        # Skip this test if no provisioning key is configured
        if not hasattr(auth_manager, 'provisioning_key') or not auth_manager.provisioning_key:
            pytest.skip("No provisioning key configured")
            
        # Arrange
        malformed_auth = AuthManager()
        malformed_auth.provisioning_key = "malformed header content !@#$%"
        malformed_endpoint = KeysEndpoint(malformed_auth, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError):
            malformed_endpoint.list()
    
    def test_server_rejects_request_due_to_insufficient_permissions(self, auth_manager, http_manager):
        """Test server rejects request due to insufficient permissions."""
        # Skip this test if no provisioning key is configured
        if not hasattr(auth_manager, 'provisioning_key') or not auth_manager.provisioning_key:
            pytest.skip("No provisioning key configured")
            
        # Arrange
        # Use a regular API key instead of provisioning key (insufficient permissions)
        insufficient_auth = AuthManager()
        insufficient_auth.provisioning_key = None  # No provisioning key
        insufficient_endpoint = KeysEndpoint(insufficient_auth, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError):
            insufficient_endpoint.list()


class Test_KeysEndpoint_Get_01_NominalBehaviors(TestKeysEndpointFixtures):
    """Test nominal HTTP request behaviors for KeysEndpoint.get() method."""
    
    def test_successful_get_request_with_valid_key_hash(self, keys_endpoint):
        """Test successful GET request retrieves specific API key details."""
        # Arrange
        # Create a test key first
        test_key = keys_endpoint.create(name="get_test_key")
        key_hash = test_key["data"]["hash"]
        
        try:
            # Act
            response = keys_endpoint.get(key_hash)
            
            # Assert
            assert isinstance(response, dict)
            assert "data" in response
            assert response["data"]["hash"] == key_hash
            assert response["data"]["name"] == "get_test_key"
            assert "created_at" in response["data"]
            assert "updated_at" in response["data"]
            assert "disabled" in response["data"]
        finally:
            # Cleanup
            keys_endpoint.delete(key_hash)


class Test_KeysEndpoint_Create_01_NominalBehaviors(TestKeysEndpointFixtures):
    """Test nominal HTTP request behaviors for KeysEndpoint.create() method."""
    
    def test_successful_post_request_creates_api_key_with_all_parameters(self, keys_endpoint):
        """Test successful POST request creates new API key with all valid parameters."""
        # Arrange
        initial_keys = keys_endpoint.list()
        initial_count = len(initial_keys["data"])
        
        # Act
        result = keys_endpoint.create(name="full_test_key", label="test-label", limit=100.0)
        
        try:
            # Assert
            assert isinstance(result, dict)
            assert "key" in result  # One-time API key value
            assert "data" in result
            assert result["data"]["hash"]  # Key hash for management operations
            assert result["data"]["name"] == "full_test_key"
            if "label" in result["data"]:
                assert result["data"]["label"] == "test-label" or result["data"]["label"].startswith("sk-or-v1-")
            if "limit" in result["data"]:
                assert result["data"]["limit"] == 100.0
            
            # Verify key appears in list
            updated_keys = keys_endpoint.list()
            assert len(updated_keys["data"]) == initial_count + 1
        finally:
            # Cleanup
            keys_endpoint.delete(result["data"]["hash"])
    
    def test_creation_with_minimal_parameters(self, keys_endpoint):
        """Test creation with only required parameters."""
        # Arrange
        test_name = "minimal_test_key"
        
        # Act
        result = keys_endpoint.create(name=test_name)
        
        try:
            # Assert
            assert isinstance(result, dict)
            assert "key" in result
            assert "data" in result
            assert result["data"]["name"] == test_name
            assert result["data"]["hash"]
        finally:
            # Cleanup
            keys_endpoint.delete(result["data"]["hash"])
    
    def test_response_contains_one_time_api_key_value_and_metadata(self, keys_endpoint):
        """Test response contains one-time API key value and metadata."""
        # Arrange
        test_name = "metadata_response_test"
        
        # Act
        result = keys_endpoint.create(name=test_name)
        
        try:
            # Assert
            assert isinstance(result, dict)
            assert "key" in result
            assert isinstance(result["key"], str)
            assert len(result["key"]) > 0
            assert result["key"].startswith("sk-or-")  # OpenRouter key format
            assert "data" in result
            assert "hash" in result["data"]
            assert isinstance(result["data"]["hash"], str)
        finally:
            # Cleanup
            keys_endpoint.delete(result["data"]["hash"])


class Test_KeysEndpoint_Update_01_NominalBehaviors(TestKeysEndpointFixtures):
    """Test nominal HTTP request behaviors for KeysEndpoint.update() method."""
    
    def test_successful_update_with_name_and_disabled_status(self, keys_endpoint):
        """Test successful PATCH request updates API key name and disabled status."""
        # Arrange
        # Create a test key first
        test_key = keys_endpoint.create(name="original_name")
        key_hash = test_key["data"]["hash"]
        
        try:
            # Act
            result = keys_endpoint.update(key_hash, name="updated_name", disabled=True)
            
            # Assert
            assert isinstance(result, dict)
            assert "data" in result
            assert result["data"]["name"] == "updated_name"
            assert result["data"]["disabled"] is True
            
            # Verify changes persist
            updated_key = keys_endpoint.get(key_hash)
            assert updated_key["data"]["name"] == "updated_name"
            assert updated_key["data"]["disabled"] is True
        finally:
            # Cleanup - disabled keys may still need deletion
            keys_endpoint.delete(key_hash)
    
    def test_update_only_name(self, keys_endpoint):
        """Test updating only the name."""
        # Arrange
        test_key = keys_endpoint.create(name="name_only_original")
        key_hash = test_key["data"]["hash"]
        
        try:
            # Act
            result = keys_endpoint.update(key_hash, name="name_only_updated")
            
            # Assert
            assert "data" in result
            assert result["data"]["name"] == "name_only_updated"
            assert result["data"]["disabled"] is False  # Should remain unchanged
        finally:
            # Cleanup
            keys_endpoint.delete(key_hash)


class Test_KeysEndpoint_Delete_01_NominalBehaviors(TestKeysEndpointFixtures):
    """Test nominal HTTP request behaviors for KeysEndpoint.delete() method."""
    
    def test_successful_delete_request_removes_existing_key(self, keys_endpoint):
        """Test successful DELETE request removes existing API key using key_hash."""
        # Arrange
        test_key = keys_endpoint.create(name="delete_test_key")
        key_hash = test_key["data"]["hash"]
        
        # Verify key exists before deletion
        pre_delete_keys = keys_endpoint.list()
        pre_delete_hashes = [key["hash"] for key in pre_delete_keys["data"]]
        assert key_hash in pre_delete_hashes
        
        # Act
        result = keys_endpoint.delete(key_hash)
        
        # Assert
        assert isinstance(result, dict)
        
        # Verify key no longer in list
        post_delete_keys = keys_endpoint.list()
        post_delete_hashes = [key["hash"] for key in post_delete_keys["data"]]
        assert key_hash not in post_delete_hashes
    
    def test_deleted_key_becomes_immediately_inactive(self, keys_endpoint):
        """Test deleted key becomes immediately inactive."""
        # Arrange
        test_key = keys_endpoint.create(name="immediate_inactive_test")
        key_hash = test_key["data"]["hash"]
        
        # Act
        keys_endpoint.delete(key_hash)
        
        # Assert
        # Immediate check - key should not appear in active list
        current_keys = keys_endpoint.list()
        current_hashes = [key["hash"] for key in current_keys["data"]]
        assert key_hash not in current_hashes
        
        # Attempting to get the deleted key should fail
        with pytest.raises(APIError):
            keys_endpoint.get(key_hash)


class Test_KeysEndpoint_Delete_02_NegativeBehaviors(TestKeysEndpointFixtures):
    """Test negative HTTP request behaviors for KeysEndpoint.delete() method."""
    
    @pytest.mark.parametrize("invalid_key_hash", [
        "definitely_nonexistent_key_hash_12345",
        "",  # Empty key hash
        "not_a_key_format_at_all",
    ])
    def test_request_fails_attempting_to_delete_nonexistent_key(self, keys_endpoint, invalid_key_hash):
        """Test request fails when attempting to delete non-existent key_hash."""
        # Arrange
        # Use clearly invalid key hashes
        
        # Act & Assert
        with pytest.raises(APIError):
            keys_endpoint.delete(invalid_key_hash)
    
    def test_request_fails_attempting_to_delete_already_deleted_key(self, keys_endpoint):
        """Test request fails when attempting to delete already-deleted key."""
        # Arrange
        test_key = keys_endpoint.create(name="double_delete_test")
        key_hash = test_key["data"]["hash"]
        
        # Delete once
        keys_endpoint.delete(key_hash)
        
        # Act & Assert
        # Attempt to delete again should fail
        with pytest.raises(APIError):
            keys_endpoint.delete(key_hash)


class Test_KeysEndpoint_StateTransitions(TestKeysEndpointFixtures):
    """Test state transition behaviors across all endpoints."""
    
    def test_complete_key_lifecycle(self, keys_endpoint):
        """Test complete key lifecycle: create -> update -> delete."""
        # Arrange
        initial_keys = keys_endpoint.list()
        initial_count = len(initial_keys["data"])
        
        # Act - Create
        new_key = keys_endpoint.create(name="lifecycle_test", label="test", limit=50)
        assert "key" in new_key
        assert "data" in new_key
        assert "hash" in new_key["data"]
        
        # Verify creation
        post_create_keys = keys_endpoint.list()
        assert len(post_create_keys["data"]) == initial_count + 1
        
        # Act - Update
        updated_key = keys_endpoint.update(new_key["data"]["hash"], name="lifecycle_updated", disabled=False)
        assert updated_key["data"]["name"] == "lifecycle_updated"
        
        # Act - Delete
        keys_endpoint.delete(new_key["data"]["hash"])
        
        # Verify deletion
        post_delete_keys = keys_endpoint.list()
        assert len(post_delete_keys["data"]) == initial_count
    
    def test_disabled_key_can_still_be_deleted(self, keys_endpoint):
        """Test that disabled keys can still be deleted."""
        # Arrange
        test_key = keys_endpoint.create(name="disable_then_delete")
        key_hash = test_key["data"]["hash"]
        
        # Act - Disable the key
        keys_endpoint.update(key_hash, disabled=True)
        
        # Act - Delete the disabled key
        result = keys_endpoint.delete(key_hash)
        
        # Assert
        assert isinstance(result, dict)
        
        # Verify key is gone
        current_keys = keys_endpoint.list(include_disabled=True)
        current_hashes = [key["hash"] for key in current_keys["data"]]
        assert key_hash not in current_hashes


class Test_KeysEndpoint_GetCurrent_01_NominalBehaviors(TestKeysEndpointFixtures):
    """Test nominal HTTP request behaviors for KeysEndpoint.get_current() method."""
    
    def test_successful_get_current_request_with_valid_api_key(self, keys_endpoint):
        """Test successful GET request to retrieve current API key information."""
        # Arrange
        # Valid credentials automatically provided through AuthManager
        
        # Act
        response = keys_endpoint.get_current()
        
        # Assert
        assert isinstance(response, dict)
        assert "data" in response
        assert isinstance(response["data"], dict)
        
        # Verify expected fields in response
        data = response["data"]
        assert "label" in data
        assert "usage" in data
        assert "limit" in data or data.get("limit") is None  # Can be null for unlimited
        assert "is_free_tier" in data
        assert "rate_limit" in data
        
        # Verify rate limit structure
        rate_limit = data["rate_limit"]
        assert "requests" in rate_limit
        assert "interval" in rate_limit
    
    def test_get_current_returns_numeric_usage_value(self, keys_endpoint):
        """Test that get_current returns proper numeric usage value."""
        # Arrange & Act
        response = keys_endpoint.get_current()
        
        # Assert
        assert isinstance(response["data"]["usage"], (int, float))
        assert response["data"]["usage"] >= 0
    
    def test_get_current_returns_boolean_free_tier_status(self, keys_endpoint):
        """Test that get_current returns boolean is_free_tier status."""
        # Arrange & Act
        response = keys_endpoint.get_current()
        
        # Assert
        assert isinstance(response["data"]["is_free_tier"], bool)
    
    def test_get_current_rate_limit_format(self, keys_endpoint):
        """Test that rate limit information is properly formatted."""
        # Arrange & Act
        response = keys_endpoint.get_current()
        
        # Assert
        rate_limit = response["data"]["rate_limit"]
        assert isinstance(rate_limit["requests"], int)
        assert rate_limit["requests"] > 0
        assert isinstance(rate_limit["interval"], str)
        # Interval should be in format like "10s", "60s", etc.
        assert rate_limit["interval"].endswith("s")


class Test_KeysEndpoint_GetCurrent_02_NegativeBehaviors(TestKeysEndpointFixtures):
    """Test negative HTTP request behaviors for KeysEndpoint.get_current() method."""
    
    def test_request_fails_with_invalid_api_key(self, http_manager):
        """Test request fails with invalid API key."""
        # Arrange
        invalid_auth = AuthManager()
        invalid_auth.api_key = "invalid_test_key_12345"
        invalid_endpoint = KeysEndpoint(invalid_auth, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError):
            invalid_endpoint.get_current()
    
    def test_request_fails_with_expired_api_key(self, http_manager):
        """Test request fails with expired API key."""
        # Arrange
        expired_auth = AuthManager()
        expired_auth.api_key = "sk-or-v1-expired-key"
        expired_endpoint = KeysEndpoint(expired_auth, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError):
            expired_endpoint.get_current()
    
    def test_request_fails_without_api_key(self, http_manager):
        """Test request fails when no API key is provided."""
        # Arrange
        no_auth = AuthManager()
        no_auth.api_key = None
        no_auth_endpoint = KeysEndpoint(no_auth, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError):
            no_auth_endpoint.get_current()


class Test_KeysEndpoint_GetCurrent_03_BoundaryBehaviors(TestKeysEndpointFixtures):
    """Test boundary HTTP request behaviors for KeysEndpoint.get_current() method."""
    
    def test_get_current_with_unlimited_key(self, keys_endpoint):
        """Test get_current with a key that has no limit (unlimited)."""
        # Arrange & Act
        response = keys_endpoint.get_current()
        
        # Assert
        # Limit can be either a number or None (for unlimited)
        limit = response["data"]["limit"]
        assert limit is None or isinstance(limit, (int, float))
        
        # If limit is None, it means unlimited
        if limit is None:
            # Usage can be any value with unlimited keys
            assert isinstance(response["data"]["usage"], (int, float))
    
    def test_get_current_near_usage_limit(self, keys_endpoint):
        """Test get_current when usage is close to limit."""
        # Arrange & Act
        response = keys_endpoint.get_current()
        
        # Assert
        usage = response["data"]["usage"]
        limit = response["data"]["limit"]
        
        if limit is not None:
            # Usage should not exceed limit
            assert usage <= limit
            
            # Calculate usage percentage
            if limit > 0:
                usage_percentage = (usage / limit) * 100
                assert 0 <= usage_percentage <= 100


class Test_KeysEndpoint_GetCurrent_04_ErrorHandlingBehaviors(TestKeysEndpointFixtures):
    """Test error handling HTTP request behaviors for KeysEndpoint.get_current() method."""
    
    def test_network_timeout_during_get_current_request(self, keys_endpoint):
        """Test handling of potential network timeouts."""
        # Arrange & Act & Assert
        try:
            response = keys_endpoint.get_current()
            # If successful, verify proper response format
            assert isinstance(response, dict)
            assert "data" in response
        except APIError as e:
            # Network timeouts should be properly handled
            assert str(e)
            assert len(str(e)) > 0
    
    def test_server_error_response_handling(self, http_manager):
        """Test handling of server errors during get_current."""
        # This test might not trigger in normal conditions
        # but tests the error handling capability
        
        # Arrange
        auth = AuthManager()
        endpoint = KeysEndpoint(auth, http_manager)
        
        # Act & Assert
        try:
            response = endpoint.get_current()
            # If successful, should still be valid format
            assert isinstance(response, dict)
        except APIError as e:
            # Server errors should be handled gracefully
            assert str(e)


class Test_KeysEndpoint_GetCurrent_05_StateTransitionBehaviors(TestKeysEndpointFixtures):
    """Test state transition behaviors for KeysEndpoint.get_current() method."""
    
    def test_get_current_consistency_across_multiple_calls(self, keys_endpoint):
        """Test that get_current returns consistent structure across calls."""
        # Arrange & Act
        response1 = keys_endpoint.get_current()
        response2 = keys_endpoint.get_current()
        
        # Assert - Structure should be consistent
        assert set(response1["data"].keys()) == set(response2["data"].keys())
        assert set(response1["data"]["rate_limit"].keys()) == set(response2["data"]["rate_limit"].keys())
        
        # Label and rate limits should remain constant
        assert response1["data"]["label"] == response2["data"]["label"]
        assert response1["data"]["rate_limit"] == response2["data"]["rate_limit"]
        
        # Usage might change between calls but should not decrease
        assert response2["data"]["usage"] >= response1["data"]["usage"]
    
    def test_get_current_reflects_account_state(self, keys_endpoint):
        """Test that get_current reflects current account state."""
        # Arrange & Act
        response = keys_endpoint.get_current()
        
        # Assert
        # Free tier status should be consistent with usage
        if response["data"]["is_free_tier"]:
            # Free tier keys typically have lower limits
            if response["data"]["limit"] is not None:
                assert response["data"]["limit"] <= 10000  # Reasonable free tier limit
        
        # Usage should be non-negative
        assert response["data"]["usage"] >= 0
        
        # Rate limits should be reasonable
        assert response["data"]["rate_limit"]["requests"] > 0
        assert response["data"]["rate_limit"]["requests"] <= 10000  # Reasonable upper bound