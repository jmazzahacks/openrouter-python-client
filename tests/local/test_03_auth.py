import os
import pytest
from unittest.mock import call, patch, MagicMock

from openrouter_client.auth import AuthManager, EnvironmentSecretsManager, SecretsManager
from openrouter_client.exceptions import AuthenticationError


# Mock PyNaCl for testing encryption functionality
class MockNacl:
    class secret:
        class SecretBox:
            KEY_SIZE = 32
            
            def __init__(self, key):
                self.key = key
                
            def encrypt(self, data):
                return b"encrypted_" + data
                
            def decrypt(self, data):
                if not data.startswith(b"encrypted_"):
                    raise ValueError("Invalid encrypted data")
                return data[len(b"encrypted_"):]
    
    class utils:
        @staticmethod
        def random(size):
            return b"\x00" * size


## Tests for EnvironmentSecretsManager

class Test_EnvironmentSecretsManager_Init_01_NominalBehaviors:
    """Tests for nominal behaviors of the EnvironmentSecretsManager.__init__ method."""
    
    def test_logger_initialization(self):
        """Test that the logger is correctly initialized."""
        manager = EnvironmentSecretsManager()
        assert manager.logger.name == "openrouter_client.auth.env"


class Test_EnvironmentSecretsManager_GetKey_01_NominalBehaviors:
    """Tests for nominal behaviors of the EnvironmentSecretsManager.get_key method."""
    
    @pytest.mark.parametrize("key_name, key_value", [
        ("TEST_API_KEY", "test_value"),
        ("OPENROUTER_API_KEY", "api_key_123"),
        ("OPENROUTER_PROVISIONING_API_KEY", "prov_key_456")
    ])
    def test_get_existing_key(self, key_name, key_value):
        """Test retrieving existing environment variables."""
        with patch.dict(os.environ, {key_name: key_value}):
            manager = EnvironmentSecretsManager()
            key = manager.get_key(key_name)
            assert isinstance(key, bytearray)
            assert bytes(key) == key_value.encode('utf-8')


class Test_EnvironmentSecretsManager_GetKey_02_NegativeBehaviors:
    """Tests for negative behaviors of the EnvironmentSecretsManager.get_key method."""
    
    @pytest.mark.parametrize("key_name", [
        "NON_EXISTENT_KEY",
        "MISSING_API_KEY",
        ""
    ])
    def test_get_nonexistent_key(self, key_name):
        """Test behavior when requesting non-existent environment variables."""
        # Ensure the key doesn't exist
        if key_name in os.environ:
            del os.environ[key_name]
            
        manager = EnvironmentSecretsManager()
        with pytest.raises(AuthenticationError) as excinfo:
            manager.get_key(key_name)
        assert key_name in str(excinfo.value)
    
    def test_get_empty_key(self):
        """Test behavior when requesting an environment variable with empty value."""
        key_name = "EMPTY_API_KEY"
        with patch.dict(os.environ, {key_name: ""}):
            manager = EnvironmentSecretsManager()
            with pytest.raises(AuthenticationError) as excinfo:
                manager.get_key(key_name)
            assert key_name in str(excinfo.value)
            assert "empty" in str(excinfo.value).lower()


class Test_EnvironmentSecretsManager_GetKey_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the EnvironmentSecretsManager.get_key method."""
    
    @pytest.mark.parametrize("key_value", [
        "a" * 1,                # Single character
        "a" * 1024,             # Very long string
        "!@#$%^&*()",           # Special characters
        "ñáéíóúüç"              # Non-ASCII characters
    ])
    def test_key_content_variations(self, key_value):
        """Test retrieving environment variables with various content types and lengths."""
        key_name = "TEST_BOUNDARY_KEY"
        
        with patch.dict(os.environ, {key_name: key_value}):
            manager = EnvironmentSecretsManager()
            key = manager.get_key(key_name)
            assert isinstance(key, bytearray)
            assert bytes(key) == key_value.encode('utf-8')


class Test_EnvironmentSecretsManager_GetKey_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the EnvironmentSecretsManager.get_key method."""
    
    def test_informative_error_message(self):
        """Test that error messages include the name of the missing environment variable."""
        key_name = "MISSING_ENV_VAR"
        # Ensure the key doesn't exist
        if key_name in os.environ:
            del os.environ[key_name]
            
        manager = EnvironmentSecretsManager()
        with pytest.raises(AuthenticationError) as excinfo:
            manager.get_key(key_name)
        assert key_name in str(excinfo.value)
        assert "not found" in str(excinfo.value).lower() or "empty" in str(excinfo.value).lower()


## Tests for AuthManager

class Test_AuthManager_Init_01_NominalBehaviors:
    """Tests for nominal behaviors of the AuthManager.__init__ method."""
    
    def test_api_key_from_parameter(self):
        """Test that API key is correctly set from parameter."""
        api_key = "test_api_key"
        auth_manager = AuthManager(api_key=api_key)
        # Use _get_secure_key to decrypt the stored key
        assert auth_manager._get_secure_key(auth_manager.api_key) == api_key

    def test_api_key_from_secrets_manager(self):
        """Test that API key is correctly retrieved from secrets manager."""
        api_key = "secrets_manager_api_key"
        
        # Create a mock secrets manager
        mock_secrets_manager = MagicMock(spec=SecretsManager)
        mock_secrets_manager.get_key.return_value = bytearray(api_key.encode('utf-8'))
        
        auth_manager = AuthManager(secrets_manager=mock_secrets_manager)
        
        # Test that the mock secrets manager was used
        mock_secrets_manager.get_key.assert_has_calls([
            call("OPENROUTER_API_KEY"),
            call("OPENROUTER_PROVISIONING_API_KEY")
        ])
        
        # Check the key is correctly used in authorization headers
        headers = auth_manager.get_auth_headers()
        assert headers["Authorization"] == f"Bearer {api_key}"

    def test_api_key_from_environment(self):
        """Test that API key is correctly retrieved from environment variable."""
        api_key = "env_api_key"
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": api_key}):
            auth_manager = AuthManager()
            assert auth_manager._get_secure_key(auth_manager.api_key) == api_key

    @patch("openrouter_client.auth.NACL_AVAILABLE", True)
    @patch("openrouter_client.auth.nacl", MockNacl)
    def test_encryption_when_pynacl_available(self):
        """Test that API keys are encrypted when PyNaCl is available."""
        api_key = "test_api_key"
        prov_key = "test_prov_key"
        
        auth_manager = AuthManager(
            api_key=api_key,
            provisioning_api_key=prov_key
        )
        
        # Verify the keys are encrypted (not stored as plaintext)
        assert auth_manager.api_key != api_key
        assert auth_manager.provisioning_api_key != prov_key
        assert isinstance(auth_manager.api_key, bytes)
        assert isinstance(auth_manager.provisioning_api_key, bytes)


class Test_AuthManager_Init_02_NegativeBehaviors:
    """Tests for negative behaviors of the AuthManager.__init__ method."""
    
    def test_no_api_key_available(self):
        """Test that AuthenticationError is raised when no API key is available."""
        # Ensure OPENROUTER_API_KEY is not in environment
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(AuthenticationError) as excinfo:
                AuthManager()
            assert "API key is required" in str(excinfo.value)
    
    def test_failed_secrets_manager_fallback(self):
        """Test fallback to environment variable when secrets manager fails."""
        api_key = "fallback_api_key"
        
        # Create a mock secrets manager that raises an exception
        mock_secrets_manager = MagicMock(spec=SecretsManager)
        mock_secrets_manager.get_key.side_effect = Exception("Secrets manager failure")
        
        # Set the environment variable
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": api_key}):
            auth_manager = AuthManager(secrets_manager=mock_secrets_manager)
            assert auth_manager._get_secure_key(auth_manager.api_key) == api_key


class Test_AuthManager_Init_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the AuthManager.__init__ method."""
    
    def test_secrets_manager_exception_handling(self):
        """Test that exceptions during key retrieval from secrets manager are properly handled."""
        api_key = "fallback_api_key"
        
        # Create a mock secrets manager that raises an exception
        mock_secrets_manager = MagicMock(spec=SecretsManager)
        mock_secrets_manager.get_key.side_effect = Exception("Secrets manager failure")
        
        # Set the environment variable
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": api_key}):
            # This should not raise an exception
            auth_manager = AuthManager(secrets_manager=mock_secrets_manager)
            assert auth_manager._get_secure_key(auth_manager.api_key) == api_key


class Test_AuthManager_Init_05_StateTransitionBehaviors:
    """Tests for state transition behaviors of the AuthManager.__init__ method."""
    
    @patch("openrouter_client.auth.NACL_AVAILABLE", True)
    @patch("openrouter_client.auth.nacl", MockNacl)
    def test_transition_to_secure_state(self):
        """Test that AuthManager transitions to a secure state with encrypted keys when PyNaCl is available."""
        api_key = "test_api_key"
        
        auth_manager = AuthManager(api_key=api_key)
        
        # Verify the secure box was created
        assert auth_manager._secure_box is not None
        # Verify the key is encrypted (not stored as plaintext)
        assert auth_manager.api_key != api_key
        assert isinstance(auth_manager.api_key, bytes)


class Test_AuthManager_GetAuthHeaders_01_NominalBehaviors:
    """Tests for nominal behaviors of the AuthManager.get_auth_headers method."""
    
    def test_standard_authorization_header(self):
        """Test that the correct Authorization header is generated with the API key."""
        api_key = "test_api_key"
        auth_manager = AuthManager(api_key=api_key)
        
        headers = auth_manager.get_auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {api_key}"
    
    def test_provisioning_authorization_header(self):
        """Test that the correct Authorization header is generated with the provisioning API key when required."""
        api_key = "test_api_key"
        prov_key = "test_prov_key"
        auth_manager = AuthManager(api_key=api_key, provisioning_api_key=prov_key)
        
        headers = auth_manager.get_auth_headers(require_provisioning=True)
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {prov_key}"
    
    @pytest.mark.parametrize("org_id,ref_id,expected_headers", [
        (None, None, {}),
        ("org123", None, {"HTTP-OpenRouter-Organization": "org123"}),
        (None, "ref123", {"X-Request-Reference-ID": "ref123"}),
        ("org123", "ref123", {
            "HTTP-OpenRouter-Organization": "org123",
            "X-Request-Reference-ID": "ref123"
        })
    ])
    def test_tracking_headers(self, org_id, ref_id, expected_headers):
        """Test that organization_id and reference_id are correctly included in headers."""
        api_key = "test_api_key"
        auth_manager = AuthManager(
            api_key=api_key,
            organization_id=org_id,
            reference_id=ref_id
        )
        
        headers = auth_manager.get_auth_headers()
        
        for key, value in expected_headers.items():
            assert headers[key] == value


class Test_AuthManager_GetAuthHeaders_02_NegativeBehaviors:
    """Tests for negative behaviors of the AuthManager.get_auth_headers method."""
    
    def test_provisioning_key_required_but_not_available(self):
        """Test that AuthenticationError is raised when provisioning key is required but not available."""
        api_key = "test_api_key"
        auth_manager = AuthManager(api_key=api_key)  # No provisioning key
        
        with pytest.raises(AuthenticationError) as excinfo:
            auth_manager.get_auth_headers(require_provisioning=True)
        assert "Provisioning API key is required" in str(excinfo.value)


class Test_AuthManager_GetAuthHeaders_03_BoundaryBehaviors:
    """Tests for boundary behaviors of the AuthManager.get_auth_headers method."""
    
    @pytest.mark.parametrize("api_key", [
        "a",                    # Minimal length
        "a" * 1024,             # Very long key
        "!@#$%^&*()",           # Special characters
        "ñáéíóúüç"              # Non-ASCII characters
    ])
    def test_api_key_variations(self, api_key):
        """Test handling of API keys with various formats and lengths."""
        auth_manager = AuthManager(api_key=api_key)
        headers = auth_manager.get_auth_headers()
        assert headers["Authorization"] == f"Bearer {api_key}"


class Test_AuthManager_GetAuthHeaders_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the AuthManager.get_auth_headers method."""
    
    def test_sensitive_information_masked_in_logs(self):
        """Test that sensitive information is masked in logs."""
        api_key = "test_api_key"
        auth_manager = AuthManager(api_key=api_key)
        
        # Capture log output
        with patch.object(auth_manager.logger, "debug") as mock_debug:
            headers = auth_manager.get_auth_headers()
            
            # Check that at least one debug call was made
            assert mock_debug.called
            
            # Check that API key is not visible in any log messages
            for call in mock_debug.call_args_list:
                log_message = str(call)
                assert api_key not in log_message
                # Check for masked Authorization header
                if "Authorization" in log_message:
                    assert "***" in log_message


class Test_AuthManager_InitializeEncryption_01_NominalBehaviors:
    """Tests for nominal behaviors of the AuthManager._initialize_encryption method."""
    
    @patch("openrouter_client.auth.NACL_AVAILABLE", True)
    @patch("openrouter_client.auth.nacl", MockNacl)
    def test_encryption_key_generation(self):
        """Test that a secure random encryption key is generated."""
        auth_manager = AuthManager(api_key="test_api_key")
        
        # Re-initialize encryption to test the method directly
        auth_manager._initialize_encryption()
        
        # Verify encryption key was created with correct size
        assert auth_manager._encryption_key is not None
        assert len(auth_manager._encryption_key) == MockNacl.secret.SecretBox.KEY_SIZE


class Test_AuthManager_InitializeEncryption_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the AuthManager._initialize_encryption method."""
    
    @patch("openrouter_client.auth.NACL_AVAILABLE", True)
    def test_encryption_initialization_failure_handling(self):
        """Test that the method handles initialization failures gracefully."""
        # Create a mock that raises an exception when SecretBox is instantiated
        mock_nacl = MagicMock()
        mock_nacl.utils.random.return_value = b"\x00" * 32
        mock_nacl.secret.SecretBox.side_effect = Exception("SecretBox initialization failure")
        
        with patch("openrouter_client.auth.nacl", mock_nacl):
            auth_manager = AuthManager(api_key="test_api_key")
            
            # Check that _secure_box is None due to failure
            assert auth_manager._secure_box is None
            # Check that _encryption_key is None due to wiping
            assert auth_manager._encryption_key is None


class Test_AuthManager_EncryptSensitiveData_01_NominalBehaviors:
    """Tests for nominal behaviors of the AuthManager._encrypt_sensitive_data method."""
    
    @patch("openrouter_client.auth.NACL_AVAILABLE", True)
    def test_successful_encryption(self):
        """Test that sensitive data is properly encrypted."""
        # Mock nacl module
        mock_nacl = MagicMock()
        mock_secure_box = MagicMock()
        mock_nacl.secret.SecretBox.return_value = mock_secure_box
        mock_nacl.utils.random.return_value = b"\x00" * 32
        mock_secure_box.encrypt.side_effect = lambda data: b"encrypted_" + data
        
        with patch("openrouter_client.auth.nacl", mock_nacl):
            auth_manager = AuthManager(api_key="dummy")  # Need to create the manager with PyNaCl available
            
            # Reset mock calls to ensure we're testing the method's behavior
            mock_secure_box.encrypt.reset_mock()
            
            # Test encryption
            sensitive_data = "secret_data"
            encrypted = auth_manager._encrypt_sensitive_data(sensitive_data)
            
            # Verify encryption was called with correct data
            mock_secure_box.encrypt.assert_called_once()
            # Verify the result is the encrypted data
            assert encrypted.startswith(b"encrypted_")


class Test_AuthManager_EncryptSensitiveData_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the AuthManager._encrypt_sensitive_data method."""
    
    @patch("openrouter_client.auth.NACL_AVAILABLE", True)
    def test_encryption_error_handling(self):
        """Test that encryption errors are handled gracefully."""
        # Mock nacl module
        mock_nacl = MagicMock()
        mock_secure_box = MagicMock()
        mock_nacl.secret.SecretBox.return_value = mock_secure_box
        mock_nacl.utils.random.return_value = b"\x00" * 32
        mock_secure_box.encrypt.side_effect = Exception("Encryption error")
        
        with patch("openrouter_client.auth.nacl", mock_nacl):
            auth_manager = AuthManager(api_key="dummy")  # Need to create the manager with PyNaCl available
            
            # Test encryption with error
            sensitive_data = "secret_data"
            result = auth_manager._encrypt_sensitive_data(sensitive_data)
            
            # Verify that original data is returned on error
            assert result == sensitive_data


class Test_AuthManager_DecryptSensitiveData_01_NominalBehaviors:
    """Tests for nominal behaviors of the AuthManager._decrypt_sensitive_data method."""
    
    @patch("openrouter_client.auth.NACL_AVAILABLE", True)
    def test_successful_decryption(self):
        """Test that encrypted data is properly decrypted."""
        # Mock nacl module
        mock_nacl = MagicMock()
        mock_secure_box = MagicMock()
        mock_nacl.secret.SecretBox.return_value = mock_secure_box
        mock_nacl.utils.random.return_value = b"\x00" * 32
        mock_secure_box.decrypt.side_effect = lambda data: data[len(b"encrypted_"):]
        
        with patch("openrouter_client.auth.nacl", mock_nacl):
            auth_manager = AuthManager(api_key="dummy")  # Need to create the manager with PyNaCl available
            
            # Test decryption
            encrypted_data = b"encrypted_secret_data"
            decrypted = auth_manager._decrypt_sensitive_data(encrypted_data)
            
            # Verify decryption was called with correct data
            mock_secure_box.decrypt.assert_called_once_with(encrypted_data)
            # Verify the result is the decrypted data
            assert decrypted == "secret_data"


class Test_AuthManager_DecryptSensitiveData_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the AuthManager._decrypt_sensitive_data method."""
    
    @patch("openrouter_client.auth.NACL_AVAILABLE", True)
    def test_decryption_error_handling(self):
        """Test that decryption errors are handled gracefully."""
        # Mock nacl module
        mock_nacl = MagicMock()
        mock_secure_box = MagicMock()
        mock_nacl.secret.SecretBox.return_value = mock_secure_box
        mock_nacl.utils.random.return_value = b"\x00" * 32
        mock_secure_box.decrypt.side_effect = Exception("Decryption error")
        
        with patch("openrouter_client.auth.nacl", mock_nacl):
            auth_manager = AuthManager(api_key="dummy")  # Need to create the manager with PyNaCl available
            
            # Test decryption with error
            encrypted_data = b"encrypted_data"
            result = auth_manager._decrypt_sensitive_data(encrypted_data)
            
            # Verify that string representation of encrypted data is returned on error
            assert result == str(encrypted_data)

    def test_non_bytes_data_handling(self):
        """Test that non-bytes data is handled correctly."""
        auth_manager = AuthManager(api_key="dummy")
        
        # Test with string data (should return as-is)
        non_bytes_data = "plain_string"
        result = auth_manager._decrypt_sensitive_data(non_bytes_data)
        assert result == non_bytes_data


class Test_AuthManager_GetSecureKey_01_NominalBehaviors:
    """Tests for nominal behaviors of the AuthManager._get_secure_key method."""
    
    @pytest.mark.parametrize("key_type,key_value,expected_result", [
        ("str", "plain_key", "plain_key"),                    # Plain string key
        ("bytes", b"encrypted_secure_key", "secure_key"),     # Encrypted bytes key
        ("bytearray", bytearray(b"encrypted_test"), "test")   # Encrypted bytearray key
    ])
    def test_key_handling_by_type(self, key_type, key_value, expected_result):
        """Test handling of different key types."""
        # Mock nacl module for encrypted cases
        mock_nacl = MagicMock()
        mock_secure_box = MagicMock()
        mock_nacl.secret.SecretBox.return_value = mock_secure_box
        mock_nacl.utils.random.return_value = b"\x00" * 32
        mock_secure_box.decrypt.side_effect = lambda data: data[len(b"encrypted_"):]
        
        with patch("openrouter_client.auth.NACL_AVAILABLE", True), \
             patch("openrouter_client.auth.nacl", mock_nacl):
            
            auth_manager = AuthManager(api_key="dummy")
            result = auth_manager._get_secure_key(key_value)
            assert result == expected_result


class Test_AuthManager_SecureWipe_01_NominalBehaviors:
    """Tests for nominal behaviors of the AuthManager._secure_wipe method."""
    
    def test_bytearray_wiping(self):
        """Test that bytearray data is properly wiped."""
        auth_manager = AuthManager(api_key="dummy")
        
        # Create test data
        test_data = bytearray(b"sensitive_data")
        original_data = bytes(test_data)  # Keep a copy for comparison
        
        # Wipe the data
        auth_manager._secure_wipe(test_data)
        
        # Verify data has been wiped (all bytes are zero)
        assert all(b == 0 for b in test_data)
        # Verify the data is different from original
        assert bytes(test_data) != original_data

    @pytest.mark.parametrize("test_data", [
        "string data",
        123,
        None
    ])
    def test_non_bytearray_handling(self, test_data):
        """Test handling of non-bytearray data types."""
        auth_manager = AuthManager(api_key="dummy")
        
        # This should not raise an exception
        auth_manager._secure_wipe(test_data)


class Test_AuthManager_Del_01_NominalBehaviors:
    """Tests for nominal behaviors of the AuthManager.__del__ method."""
    
    @patch("openrouter_client.auth.NACL_AVAILABLE", True)
    def test_encryption_key_wiping(self):
        """Test that the encryption key is wiped when the object is destroyed."""
        # Mock nacl module
        mock_nacl = MagicMock()
        mock_secure_box = MagicMock()
        mock_nacl.secret.SecretBox.return_value = mock_secure_box
        mock_nacl.utils.random.return_value = b"\x00" * 32
        
        with patch("openrouter_client.auth.nacl", mock_nacl):
            auth_manager = AuthManager(api_key="dummy")
            
            # Create a spy on _secure_wipe method
            with patch.object(auth_manager, "_secure_wipe") as mock_wipe:
                # Creata a copy of the encryption key before deletion
                encryption_key_before_del = auth_manager._encryption_key
                
                # Delete the encryption key
                auth_manager.__del__()
                
                # Verify that _secure_wipe was called with the encryption key
                mock_wipe.assert_called_once_with(encryption_key_before_del)


class Test_AuthManager_Del_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the AuthManager.__del__ method."""
    
    @patch("openrouter_client.auth.NACL_AVAILABLE", True)
    def test_exception_handling_during_cleanup(self):
        """Test that exceptions during cleanup don't crash the program."""
        # Mock nacl module
        mock_nacl = MagicMock()
        mock_secure_box = MagicMock()
        mock_nacl.secret.SecretBox.return_value = mock_secure_box
        mock_nacl.utils.random.return_value = b"\x00" * 32
        
        with patch("openrouter_client.auth.nacl", mock_nacl):
            auth_manager = AuthManager(api_key="dummy")
            
            # Make _secure_wipe raise an exception
            with patch.object(auth_manager, "_secure_wipe", side_effect=Exception("Cleanup error")):
                try:
                    # This should not raise an exception outside the method
                    auth_manager.__del__()
                except Exception:
                    pytest.fail("__del__ method should handle exceptions gracefully")
