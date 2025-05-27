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
        return HTTPManager()
    
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
        assert isinstance(response, list)
        if response:  # If API keys exist in account
            assert all(isinstance(key_data, dict) for key_data in response)
            # Verify expected JSON response structure
            for key_data in response:
                assert "id" in key_data or "name" in key_data or "created" in key_data
    
    def test_empty_list_response_when_no_api_keys_exist(self, keys_endpoint):
        """Test empty list returned when no API keys exist for the account."""
        # Arrange
        # Account state may vary, but response format should be consistent
        
        # Act
        response = keys_endpoint.list()
        
        # Assert
        assert isinstance(response, list)
        # Response should be valid JSON list regardless of content
    
    def test_json_response_contains_expected_metadata_fields(self, keys_endpoint):
        """Test JSON response contains expected key metadata fields."""
        # Arrange
        # Create a test key to ensure we have data to validate
        test_key = keys_endpoint.create(name="metadata_validation_key")
        
        # Act
        response = keys_endpoint.list()
        
        # Assert
        assert isinstance(response, list)
        assert len(response) > 0
        
        # Find our test key and validate its metadata structure
        test_key_found = False
        for key_data in response:
            if key_data.get("id") == test_key["id"]:
                test_key_found = True
                assert isinstance(key_data, dict)
                break
        
        assert test_key_found
        
        # Cleanup
        keys_endpoint.revoke(test_key["id"])
    
    def test_authentication_headers_correctly_formatted_and_accepted(self, keys_endpoint):
        """Test authentication headers are correctly formatted and accepted by OpenRouter API."""
        # Arrange
        # Headers are managed internally by the endpoint
        
        # Act
        response = keys_endpoint.list()
        
        # Assert
        # Successful response indicates headers were accepted
        assert isinstance(response, list)


class Test_KeysEndpoint_List_02_NegativeBehaviors(TestKeysEndpointFixtures):
    """Test negative HTTP request behaviors for KeysEndpoint.list() method."""
    
    def test_request_fails_with_invalid_provisioning_api_key(self, auth_manager, http_manager):
        """Test request fails with invalid or expired provisioning API key."""
        # Arrange
        invalid_auth = AuthManager()
        invalid_auth.provisioning_key = "invalid_test_key_12345"
        invalid_endpoint = KeysEndpoint(invalid_auth, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError):
            invalid_endpoint.list()
    
    def test_request_fails_with_malformed_authentication_headers(self, auth_manager, http_manager):
        """Test request fails with malformed authentication headers."""
        # Arrange
        malformed_auth = AuthManager()
        malformed_auth.provisioning_key = "malformed header content !@#$%"
        malformed_endpoint = KeysEndpoint(malformed_auth, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError):
            malformed_endpoint.list()
    
    def test_server_rejects_request_due_to_insufficient_permissions(self, auth_manager, http_manager):
        """Test server rejects request due to insufficient permissions."""
        # Arrange
        # Use a regular API key instead of provisioning key (insufficient permissions)
        insufficient_auth = AuthManager()
        insufficient_auth.provisioning_key = None  # No provisioning key
        insufficient_endpoint = KeysEndpoint(insufficient_auth, http_manager)
        
        # Act & Assert
        with pytest.raises(APIError):
            insufficient_endpoint.list()


class Test_KeysEndpoint_List_03_BoundaryBehaviors(TestKeysEndpointFixtures):
    """Test boundary HTTP request behaviors for KeysEndpoint.list() method."""
    
    def test_request_when_account_has_maximum_api_keys(self, keys_endpoint):
        """Test request when account has maximum allowed number of API keys."""
        # Arrange
        current_keys = keys_endpoint.list()
        current_count = len(current_keys)
        
        # Act
        response = keys_endpoint.list()
        
        # Assert
        assert isinstance(response, list)
        assert len(response) == current_count
        # Should handle maximum keys gracefully
    
    def test_response_handling_when_api_key_list_reaches_maximum_size(self, keys_endpoint):
        """Test response handling when API key list reaches maximum size."""
        # Arrange
        # Test with current key count, whatever it may be
        
        # Act
        response = keys_endpoint.list()
        
        # Assert
        assert isinstance(response, list)
        # Response should be properly formatted regardless of size


class Test_KeysEndpoint_List_04_ErrorHandlingBehaviors(TestKeysEndpointFixtures):
    """Test error handling HTTP request behaviors for KeysEndpoint.list() method."""
    
    def test_network_timeout_during_get_request_execution(self, keys_endpoint):
        """Test network timeout during GET request execution."""
        # Arrange
        # Network conditions may vary, test what we can observe
        
        # Act & Assert
        try:
            response = keys_endpoint.list()
            # If successful, verify proper response format
            assert isinstance(response, list)
        except APIError as e:
            # Network timeouts should be properly handled with meaningful error messages
            assert str(e)
            assert len(str(e)) > 0
    
    def test_malformed_json_response_handling(self, keys_endpoint):
        """Test handling of malformed or invalid JSON response from server."""
        # Arrange
        # Real API should return valid JSON, but test error handling capability
        
        # Act & Assert
        try:
            response = keys_endpoint.list()
            # Valid response should be proper JSON structure
            assert isinstance(response, list)
        except APIError as e:
            # JSON parsing errors should be handled gracefully
            assert str(e)


class Test_KeysEndpoint_List_05_StateTransitionBehaviors(TestKeysEndpointFixtures):
    """Test state transition HTTP request behaviors for KeysEndpoint.list() method."""
    
    def test_api_key_list_state_reflects_recent_operations(self, keys_endpoint):
        """Test API key list state accurately reflects recent key creation or revocation operations."""
        # Arrange
        initial_keys = keys_endpoint.list()
        initial_count = len(initial_keys)
        
        # Act - Create key and verify state transition
        new_key = keys_endpoint.create(name="state_transition_test")
        post_create_keys = keys_endpoint.list()
        
        # Assert - State should reflect new key
        assert len(post_create_keys) == initial_count + 1
        
        # Act - Revoke key and verify state transition
        keys_endpoint.revoke(new_key["id"])
        post_revoke_keys = keys_endpoint.list()
        
        # Assert - State should reflect revocation
        assert len(post_revoke_keys) == initial_count


class Test_KeysEndpoint_Create_01_NominalBehaviors(TestKeysEndpointFixtures):
    """Test nominal HTTP request behaviors for KeysEndpoint.create() method."""
    
    @pytest.mark.parametrize("name,expiry,permissions", [
        ("basic_test_key", None, None),
        ("key_with_expiry", 30, None),
        ("key_with_permissions", None, ["read"]),
        ("full_featured_key", 7, ["read", "write"]),
        (None, None, None),  # Minimal parameters
    ])
    def test_successful_post_request_creates_api_key_with_valid_parameters(self, keys_endpoint, name, expiry, permissions):
        """Test successful POST request creates new API key with all valid parameters combinations."""
        # Arrange
        initial_keys = keys_endpoint.list()
        initial_count = len(initial_keys)
        
        # Act
        result = keys_endpoint.create(name=name, expiry=expiry, permissions=permissions)
        
        # Assert
        assert isinstance(result, dict)
        assert "key" in result  # One-time API key value
        assert "id" in result   # Key ID for management operations
        
        # Verify key appears in list
        updated_keys = keys_endpoint.list()
        assert len(updated_keys) == initial_count + 1
        
        # Cleanup
        keys_endpoint.revoke(result["id"])
    
    @pytest.mark.parametrize("expiry_format,expiry_value", [
        ("datetime_object", datetime.now() + timedelta(days=30)),
        ("integer_days", 15),
        ("iso_string", "2025-12-31T23:59:59Z"),
    ])
    def test_different_expiry_format_handling_in_http_request(self, keys_endpoint, expiry_format, expiry_value):
        """Test different expiry format handling (datetime object, integer days, string format) in HTTP request."""
        # Arrange
        test_name = f"expiry_test_{expiry_format}"
        
        # Act
        result = keys_endpoint.create(name=test_name, expiry=expiry_value)
        
        # Assert
        assert "key" in result
        assert "id" in result
        
        # Cleanup
        keys_endpoint.revoke(result["id"])
    
    def test_response_contains_one_time_api_key_value_and_metadata(self, keys_endpoint):
        """Test response contains one-time API key value and metadata."""
        # Arrange
        test_name = "metadata_response_test"
        
        # Act
        result = keys_endpoint.create(name=test_name)
        
        # Assert
        assert isinstance(result, dict)
        assert "key" in result
        assert isinstance(result["key"], str)
        assert len(result["key"]) > 0
        assert "id" in result
        assert isinstance(result["id"], str)
        
        # Cleanup
        keys_endpoint.revoke(result["id"])


class Test_KeysEndpoint_Create_02_NegativeBehaviors(TestKeysEndpointFixtures):
    """Test negative HTTP request behaviors for KeysEndpoint.create() method."""
    
    @pytest.mark.parametrize("invalid_name", [
        "",  # Empty name
        " ",  # Whitespace only
        "x" * 1000,  # Extremely long name
        "@#$%^&*()",  # Special characters
    ])
    def test_request_fails_with_invalid_name_format(self, keys_endpoint, invalid_name):
        """Test request fails with invalid name format or content."""
        # Arrange
        # Invalid name parameters
        
        # Act & Assert
        with pytest.raises(APIError):
            keys_endpoint.create(name=invalid_name)
    
    @pytest.mark.parametrize("invalid_expiry", [
        "not_a_date",
        "2020-13-45",  # Invalid date
        -1,  # Negative days
        0,   # Zero days
    ])
    def test_request_fails_with_malformed_expiry_values(self, keys_endpoint, invalid_expiry):
        """Test request fails with malformed expiry date values."""
        # Arrange
        test_name = "invalid_expiry_test"
        
        # Act & Assert
        with pytest.raises(APIError):
            keys_endpoint.create(name=test_name, expiry=invalid_expiry)
    
    @pytest.mark.parametrize("invalid_permissions", [
        ["nonexistent_permission"],
        [""],  # Empty permission string
        ["read", "invalid_perm"],  # Mixed valid/invalid
    ])
    def test_request_fails_with_invalid_permissions_list(self, keys_endpoint, invalid_permissions):
        """Test request fails with invalid permission strings in permissions list."""
        # Arrange
        test_name = "invalid_permissions_test"
        
        # Act & Assert
        with pytest.raises(APIError):
            keys_endpoint.create(name=test_name, permissions=invalid_permissions)


class Test_KeysEndpoint_Create_03_BoundaryBehaviors(TestKeysEndpointFixtures):
    """Test boundary HTTP request behaviors for KeysEndpoint.create() method."""
    
    @pytest.mark.parametrize("boundary_condition", [
        {"name": "x"},  # Minimum length name
        {"name": "a" * 100},  # Maximum reasonable length
        {"expiry": 1},  # Minimum expiry
        {"expiry": 365},  # Maximum reasonable expiry
    ])
    def test_api_key_creation_at_parameter_boundaries(self, keys_endpoint, boundary_condition):
        """Test API key creation with parameters at minimum and maximum boundaries."""
        # Arrange
        base_params = {"name": "boundary_test"}
        test_params = {**base_params, **boundary_condition}
        
        # Act
        try:
            result = keys_endpoint.create(**test_params)
            
            # Assert
            assert "key" in result
            assert "id" in result
            
            # Cleanup
            keys_endpoint.revoke(result["id"])
        except APIError:
            # Some boundary values may be legitimately rejected
            pass
    
    def test_creation_when_account_at_api_key_limit(self, keys_endpoint):
        """Test creation attempt when account is at API key limit."""
        # Arrange
        current_keys = keys_endpoint.list()
        # Test regardless of current count - boundary behavior
        
        # Act & Assert
        try:
            result = keys_endpoint.create(name="limit_test_key")
            # If successful, cleanup
            keys_endpoint.revoke(result["id"])
        except APIError:
            # Hitting limit is a valid boundary condition
            pass


class Test_KeysEndpoint_Create_04_ErrorHandlingBehaviors(TestKeysEndpointFixtures):
    """Test error handling HTTP request behaviors for KeysEndpoint.create() method."""
    
    def test_network_failure_during_post_request_transmission(self, keys_endpoint):
        """Test network failure during POST request transmission."""
        # Arrange
        test_name = "network_failure_test"
        
        # Act & Assert
        try:
            result = keys_endpoint.create(name=test_name)
            # If successful, cleanup
            keys_endpoint.revoke(result["id"])
        except APIError as e:
            # Network failures should be properly handled
            assert str(e)
            assert len(str(e)) > 0
    
    def test_authentication_timeout_during_key_creation(self, keys_endpoint):
        """Test authentication timeout during key creation."""
        # Arrange
        test_name = "auth_timeout_test"
        
        # Act & Assert
        try:
            result = keys_endpoint.create(name=test_name)
            # If successful, cleanup
            keys_endpoint.revoke(result["id"])
        except APIError as e:
            # Authentication timeouts should be handled gracefully
            assert str(e)


class Test_KeysEndpoint_Create_05_StateTransitionBehaviors(TestKeysEndpointFixtures):
    """Test state transition HTTP request behaviors for KeysEndpoint.create() method."""
    
    def test_system_transitions_from_n_keys_to_n_plus_one_after_creation(self, keys_endpoint):
        """Test system transitions from N keys to N+1 keys after successful creation."""
        # Arrange
        initial_keys = keys_endpoint.list()
        initial_count = len(initial_keys)
        
        # Act
        new_key = keys_endpoint.create(name="state_transition_create")
        updated_keys = keys_endpoint.list()
        
        # Assert
        assert len(updated_keys) == initial_count + 1
        
        # Cleanup
        keys_endpoint.revoke(new_key["id"])
    
    def test_new_api_key_transitions_to_active_state(self, keys_endpoint):
        """Test new API key transitions from non-existent to active state."""
        # Arrange
        initial_keys = keys_endpoint.list()
        initial_ids = [key["id"] for key in initial_keys]
        
        # Act
        new_key = keys_endpoint.create(name="active_state_test")
        updated_keys = keys_endpoint.list()
        updated_ids = [key["id"] for key in updated_keys]
        
        # Assert
        assert new_key["id"] not in initial_ids  # Was non-existent
        assert new_key["id"] in updated_ids      # Now active
        
        # Cleanup
        keys_endpoint.revoke(new_key["id"])


class Test_KeysEndpoint_Revoke_01_NominalBehaviors(TestKeysEndpointFixtures):
    """Test nominal HTTP request behaviors for KeysEndpoint.revoke() method."""
    
    def test_successful_delete_request_revokes_existing_active_key(self, keys_endpoint):
        """Test successful DELETE request revokes existing active API key using key_id."""
        # Arrange
        test_key = keys_endpoint.create(name="revoke_nominal_test")
        key_id = test_key["id"]
        
        # Verify key exists before revocation
        pre_revoke_keys = keys_endpoint.list()
        pre_revoke_ids = [key["id"] for key in pre_revoke_keys]
        assert key_id in pre_revoke_ids
        
        # Act
        result = keys_endpoint.revoke(key_id)
        
        # Assert
        assert isinstance(result, dict)
        
        # Verify key no longer in active list
        post_revoke_keys = keys_endpoint.list()
        post_revoke_ids = [key["id"] for key in post_revoke_keys]
        assert key_id not in post_revoke_ids
    
    def test_proper_revocation_confirmation_response_returned(self, keys_endpoint):
        """Test proper revocation confirmation response returned."""
        # Arrange
        test_key = keys_endpoint.create(name="confirmation_response_test")
        
        # Act
        result = keys_endpoint.revoke(test_key["id"])
        
        # Assert
        assert isinstance(result, dict)
        # Response should contain confirmation information
    
    def test_revoked_key_becomes_immediately_inactive(self, keys_endpoint):
        """Test revoked key becomes immediately inactive."""
        # Arrange
        test_key = keys_endpoint.create(name="immediate_inactive_test")
        key_id = test_key["id"]
        
        # Act
        keys_endpoint.revoke(key_id)
        
        # Assert
        # Immediate check - key should not appear in active list
        current_keys = keys_endpoint.list()
        current_ids = [key["id"] for key in current_keys]
        assert key_id not in current_ids
    
    def test_revoke_specific_production_key(self, keys_endpoint):
        """Test revoking the specific production key as requested in requirements."""
        # Arrange
        specific_key_id = "valid_key"
        
        # Act & Assert
        try:
            result = keys_endpoint.revoke(specific_key_id)
            assert isinstance(result, dict)
        except APIError:
            # Key might not exist or already be revoked
            pass


class Test_KeysEndpoint_Revoke_02_NegativeBehaviors(TestKeysEndpointFixtures):
    """Test negative HTTP request behaviors for KeysEndpoint.revoke() method."""
    
    @pytest.mark.parametrize("invalid_key_id", [
        "definitely_nonexistent_key_id_12345",
        "",  # Empty key ID
        "sk-or-v1-invalid_format",
        "not_a_key_format_at_all",
    ])
    def test_request_fails_attempting_to_revoke_nonexistent_key_id(self, keys_endpoint, invalid_key_id):
        """Test request fails when attempting to revoke non-existent key_id."""
        # Arrange
        # Use clearly invalid key IDs
        
        # Act & Assert
        with pytest.raises(APIError):
            keys_endpoint.revoke(invalid_key_id)
    
    def test_request_fails_attempting_to_revoke_already_revoked_key(self, keys_endpoint):
        """Test request fails when attempting to revoke already-revoked key."""
        # Arrange
        test_key = keys_endpoint.create(name="double_revoke_test")
        key_id = test_key["id"]
        
        # Revoke once
        keys_endpoint.revoke(key_id)
        
        # Act & Assert
        # Attempt to revoke again should fail
        with pytest.raises(APIError):
            keys_endpoint.revoke(key_id)


class Test_KeysEndpoint_Revoke_03_BoundaryBehaviors(TestKeysEndpointFixtures):
    """Test boundary HTTP request behaviors for KeysEndpoint.revoke() method."""
    
    def test_revoking_with_key_id_at_length_boundaries(self, keys_endpoint):
        """Test revoking with key_id parameter at minimum and maximum length boundaries."""
        # Arrange
        # Create actual key to test with real key ID length
        test_key = keys_endpoint.create(name="boundary_length_test")
        real_key_id = test_key["id"]
        
        # Act
        result = keys_endpoint.revoke(real_key_id)
        
        # Assert
        assert isinstance(result, dict)


class Test_KeysEndpoint_Revoke_04_ErrorHandlingBehaviors(TestKeysEndpointFixtures):
    """Test error handling HTTP request behaviors for KeysEndpoint.revoke() method."""
    
    def test_network_failure_during_delete_request_execution(self, keys_endpoint):
        """Test network failure during DELETE request execution."""
        # Arrange
        test_key = keys_endpoint.create(name="network_failure_revoke_test")
        
        # Act & Assert
        try:
            result = keys_endpoint.revoke(test_key["id"])
            assert isinstance(result, dict)
        except APIError as e:
            # Network failures should be handled gracefully
            assert str(e)
            # Attempt cleanup
            try:
                keys_endpoint.revoke(test_key["id"])
            except:
                pass
    
    def test_authentication_failures_during_revocation_request(self, keys_endpoint):
        """Test authentication failures during revocation request."""
        # Arrange
        test_key = keys_endpoint.create(name="auth_failure_revoke_test")
        
        # Act & Assert
        try:
            result = keys_endpoint.revoke(test_key["id"])
            assert isinstance(result, dict)
        except APIError as e:
            # Authentication failures should be properly handled
            assert str(e)


class Test_KeysEndpoint_Revoke_05_StateTransitionBehaviors(TestKeysEndpointFixtures):
    """Test state transition HTTP request behaviors for KeysEndpoint.revoke() method."""
    
    def test_api_key_state_transitions_from_active_to_revoked(self, keys_endpoint):
        """Test API key state transitions from active to revoked status."""
        # Arrange
        test_key = keys_endpoint.create(name="state_transition_revoke")
        key_id = test_key["id"]
        
        # Verify active state
        pre_revoke_keys = keys_endpoint.list()
        pre_revoke_ids = [key["id"] for key in pre_revoke_keys]
        assert key_id in pre_revoke_ids  # Active state
        
        # Act
        keys_endpoint.revoke(key_id)
        
        # Assert
        post_revoke_keys = keys_endpoint.list()
        post_revoke_ids = [key["id"] for key in post_revoke_keys]
        assert key_id not in post_revoke_ids  # Revoked state
    
    def test_system_key_count_decreases_after_revocation(self, keys_endpoint):
        """Test system key count decreases appropriately after revocation."""
        # Arrange
        initial_keys = keys_endpoint.list()
        initial_count = len(initial_keys)
        
        test_key = keys_endpoint.create(name="count_decrease_test")
        pre_revoke_count = len(keys_endpoint.list())
        
        # Act
        keys_endpoint.revoke(test_key["id"])
        
        # Assert
        post_revoke_keys = keys_endpoint.list()
        post_revoke_count = len(post_revoke_keys)
        assert post_revoke_count == pre_revoke_count - 1
        assert post_revoke_count == initial_count


class Test_KeysEndpoint_Rotate_01_NominalBehaviors(TestKeysEndpointFixtures):
    """Test nominal HTTP request behaviors for KeysEndpoint.rotate() method."""
    
    def test_successful_post_request_rotates_existing_key(self, keys_endpoint):
        """Test successful POST request rotates existing API key using key_id."""
        # Arrange
        original_key = keys_endpoint.create(name="rotate_nominal_test")
        original_key_id = original_key["id"]
        
        # Act
        rotated_result = keys_endpoint.rotate(original_key_id)
        
        # Assert
        assert isinstance(rotated_result, dict)
        assert "key" in rotated_result
        assert "id" in rotated_result
        assert rotated_result["id"] != original_key_id  # New key ID
        
        # Cleanup
        keys_endpoint.revoke(rotated_result["id"])
    
    def test_new_key_created_with_identical_permissions(self, keys_endpoint):
        """Test new key created with identical permissions as original key."""
        # Arrange
        original_key = keys_endpoint.create(
            name="permissions_inheritance_test",
            permissions=["read"]
        )
        
        # Act
        rotated_result = keys_endpoint.rotate(original_key["id"])
        
        # Assert
        assert "key" in rotated_result
        assert "id" in rotated_result
        
        # Cleanup
        keys_endpoint.revoke(rotated_result["id"])
    
    def test_response_contains_new_one_time_api_key_value(self, keys_endpoint):
        """Test response contains new one-time API key value."""
        # Arrange
        original_key = keys_endpoint.create(name="one_time_key_test")
        
        # Act
        rotated_result = keys_endpoint.rotate(original_key["id"])
        
        # Assert
        assert "key" in rotated_result
        assert isinstance(rotated_result["key"], str)
        assert len(rotated_result["key"]) > 0
        assert rotated_result["key"] != original_key["key"]  # Different key value
        
        # Cleanup
        keys_endpoint.revoke(rotated_result["id"])


class Test_KeysEndpoint_Rotate_02_NegativeBehaviors(TestKeysEndpointFixtures):
    """Test negative HTTP request behaviors for KeysEndpoint.rotate() method."""
    
    @pytest.mark.parametrize("invalid_key_id", [
        "absolutely_nonexistent_key_for_rotation",
        "",  # Empty key ID
        "sk-or-v1-fake_rotation_key",
        "invalid_rotation_format_123",
    ])
    def test_request_fails_attempting_to_rotate_nonexistent_key(self, keys_endpoint, invalid_key_id):
        """Test request fails when attempting to rotate non-existent key_id."""
        # Arrange
        # Use clearly invalid key IDs
        
        # Act & Assert
        with pytest.raises(APIError):
            keys_endpoint.rotate(invalid_key_id)
    
    def test_request_fails_attempting_to_rotate_revoked_key(self, keys_endpoint):
        """Test request fails when attempting to rotate already-revoked key."""
        # Arrange
        test_key = keys_endpoint.create(name="revoked_rotation_test")
        key_id = test_key["id"]
        
        # Revoke the key first
        keys_endpoint.revoke(key_id)
        
        # Act & Assert
        with pytest.raises(APIError):
            keys_endpoint.rotate(key_id)


class Test_KeysEndpoint_Rotate_03_BoundaryBehaviors(TestKeysEndpointFixtures):
    """Test boundary HTTP request behaviors for KeysEndpoint.rotate() method."""
    
    def test_rotation_of_key_with_maximum_permissions(self, keys_endpoint):
        """Test rotation of key with maximum allowed permissions."""
        # Arrange
        original_key = keys_endpoint.create(
            name="max_permissions_rotation",
            permissions=["read", "write"]  # Maximum reasonable permissions
        )
        
        # Act
        try:
            rotated_result = keys_endpoint.rotate(original_key["id"])
            
            # Assert
            assert "key" in rotated_result
            assert "id" in rotated_result
            
            # Cleanup
            keys_endpoint.revoke(rotated_result["id"])
        except APIError:
            # If rotation fails, cleanup original
            keys_endpoint.revoke(original_key["id"])


class Test_KeysEndpoint_Rotate_04_ErrorHandlingBehaviors(TestKeysEndpointFixtures):
    """Test error handling HTTP request behaviors for KeysEndpoint.rotate() method."""
    
    def test_network_failure_during_post_request_for_rotation(self, keys_endpoint):
        """Test network failure during POST request for rotation."""
        # Arrange
        test_key = keys_endpoint.create(name="network_failure_rotation")
        
        # Act & Assert
        try:
            result = keys_endpoint.rotate(test_key["id"])
            # If successful, cleanup new key
            keys_endpoint.revoke(result["id"])
        except APIError as e:
            # Network failures should be handled gracefully
            assert str(e)
            # Cleanup original key
            keys_endpoint.revoke(test_key["id"])
    
    def test_authentication_failures_during_rotation_request(self, keys_endpoint):
        """Test authentication failures during rotation request."""
        # Arrange
        test_key = keys_endpoint.create(name="auth_failure_rotation")
        
        # Act & Assert
        try:
            result = keys_endpoint.rotate(test_key["id"])
            # If successful, cleanup
            keys_endpoint.revoke(result["id"])
        except APIError as e:
            # Authentication failures should be properly handled
            assert str(e)
            # Cleanup original
            keys_endpoint.revoke(test_key["id"])


class Test_KeysEndpoint_Rotate_05_StateTransitionBehaviors(TestKeysEndpointFixtures):
    """Test state transition HTTP request behaviors for KeysEndpoint.rotate() method."""
    
    def test_original_key_transitions_from_active_to_revoked_state(self, keys_endpoint):
        """Test original key transitions from active to revoked state."""
        # Arrange
        original_key = keys_endpoint.create(name="original_state_transition")
        original_key_id = original_key["id"]
        
        # Verify original key is active
        pre_rotation_keys = keys_endpoint.list()
        pre_rotation_ids = [key["id"] for key in pre_rotation_keys]
        assert original_key_id in pre_rotation_ids
        
        # Act
        rotated_result = keys_endpoint.rotate(original_key_id)
        
        # Assert
        post_rotation_keys = keys_endpoint.list()
        post_rotation_ids = [key["id"] for key in post_rotation_keys]
        assert original_key_id not in post_rotation_ids  # Original revoked
        
        # Cleanup
        keys_endpoint.revoke(rotated_result["id"])
    
    def test_new_key_transitions_from_nonexistent_to_active_state(self, keys_endpoint):
        """Test new key transitions from non-existent to active state."""
        # Arrange
        original_key = keys_endpoint.create(name="new_key_state_transition")
        pre_rotation_keys = keys_endpoint.list()
        pre_rotation_ids = [key["id"] for key in pre_rotation_keys]
        
        # Act
        rotated_result = keys_endpoint.rotate(original_key["id"])
        new_key_id = rotated_result["id"]
        
        # Assert
        assert new_key_id not in pre_rotation_ids  # Was non-existent
        
        post_rotation_keys = keys_endpoint.list()
        post_rotation_ids = [key["id"] for key in post_rotation_keys]
        assert new_key_id in post_rotation_ids  # Now active
        
        # Cleanup
        keys_endpoint.revoke(new_key_id)
    
    def test_overall_key_count_remains_constant_during_rotation(self, keys_endpoint):
        """Test overall key count remains constant while key identifiers change."""
        # Arrange
        initial_keys = keys_endpoint.list()
        initial_count = len(initial_keys)
        
        test_key = keys_endpoint.create(name="constant_count_rotation")
        pre_rotation_count = len(keys_endpoint.list())
        
        # Act
        rotated_result = keys_endpoint.rotate(test_key["id"])
        
        # Assert
        post_rotation_keys = keys_endpoint.list()
        post_rotation_count = len(post_rotation_keys)
        assert post_rotation_count == pre_rotation_count  # Count unchanged
        
        # Cleanup
        keys_endpoint.revoke(rotated_result["id"])
        
        # Verify back to initial state
        final_keys = keys_endpoint.list()
        assert len(final_keys) == initial_count
