"""
Authentication management for OpenRouter Client.

This module handles authentication credentials, API key validation, and
organization information for requests to the OpenRouter API.

Exported:
- AuthManager: Authentication and API key manager
- SecretsManager: Abstract base class for secrets management
- PhaseSecretsManager: Phase implementation of SecretsManager
"""

import os
import logging
from typing import Dict, Optional, Union, Protocol, runtime_checkable

# Try to import PyNaCl for secure in-memory encryption
try:
    import nacl.secret
    import nacl.utils
    NACL_AVAILABLE = True
except ImportError:
    NACL_AVAILABLE = False

from .exceptions import AuthenticationError


@runtime_checkable
class SecretsManager(Protocol):
    """
    Protocol defining the interface for a secrets manager.
    
    All secret managers must implement the get_key method for retrieving secrets.
    """
    
    def get_key(self, name: str) -> bytearray:
        """
        Retrieve a secret key by name.
        
        Args:
            name (str): Name of the key to retrieve.
            
        Returns:
            bytearray: The secret key value.
            
        Raises:
            AuthenticationError: If the key cannot be retrieved.
        """
        ...

class EnvironmentSecretsManager:
    """
    Environment variables implementation of the SecretsManager protocol.
    
    Uses environment variables to retrieve secrets.
    """
    
    def __init__(self):
        """
        Initialize the environment secrets manager.
        """
        self.logger = logging.getLogger("openrouter_client.auth.env")
    
    def get_key(self, name: str) -> bytearray:
        """
        Retrieve a secret key from environment variables by name.
        
        Args:
            name (str): Name of the key to retrieve.
            
        Returns:
            bytearray: The key value.
            
        Raises:
            AuthenticationError: If the key cannot be retrieved.
        """
        value = os.environ.get(name, "")
        if not value:
            self.logger.warning(f"Environment variable {name} not found")
            raise AuthenticationError(f"Environment variable {name} not found or empty")
        
        self.logger.debug(f"Successfully retrieved {name} from environment variables")
        return bytearray(value.encode('utf-8'))


class AuthManager:
    """
    Manages authentication and API keys for OpenRouter API requests.
    
    Uses PyNaCl for secure handling of API keys in memory when available.
    Keys are stored in encrypted form and decrypted only when needed,
    with secure wiping of sensitive data from memory after use.
    
    Attributes:
        api_key (Union[str, ByteString]): OpenRouter API key (encrypted when PyNaCl is available).
        provisioning_api_key (Optional[Union[str, ByteString]]): API key for provisioning operations (encrypted when PyNaCl is available).
        organization_id (Optional[str]): Organization ID for request tracking.
        reference_id (Optional[str]): Reference ID for request tracking.
        logger (logging.Logger): Authentication logger.
        _secure_box (Optional[nacl.secret.SecretBox]): Encryption box when PyNaCl is available.
        _encryption_key (Optional[bytearray]): Encryption key for API keys (securely managed).
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 provisioning_api_key: Optional[str] = None,
                 organization_id: Optional[str] = None,
                 reference_id: Optional[str] = None,
                 secrets_manager: Optional[SecretsManager] = None):
        """
        Initialize the authentication manager.
        
        Args:
            api_key (Optional[str]): API key for authentication. If None, tries to retrieve from secrets manager,
                then falls back to OPENROUTER_API_KEY environment variable.
            provisioning_api_key (Optional[str]): API key for provisioning operations. If None, tries to retrieve
                from secrets manager, then falls back to OPENROUTER_PROVISIONING_API_KEY environment variable.
            organization_id (Optional[str]): Organization ID for request tracking.
            reference_id (Optional[str]): Reference ID for request tracking.
            secrets_manager (Optional[SecretsManager]): Custom secrets manager for retrieving keys.
        
        Raises:
            AuthenticationError: If no valid API key is available.
        """
        # Set up logger for authentication operations
        self.logger = logging.getLogger("openrouter_client.auth")
        
        # Initialize secrets managers
        self.secrets_manager = secrets_manager
        
        # Initialize secure encryption if PyNaCl is available
        self._secure_box = None
        self._encryption_key = None
        if NACL_AVAILABLE:
            self._initialize_encryption()
        
        # Get API key with priority: passed param > secrets manager > environment variable
        if api_key is None:
            # If no secrets manager, create an environment secrets manager
            if self.secrets_manager is None:
                self.secrets_manager = EnvironmentSecretsManager()
            
            # Try to retrieve API key from secrets manager
            try:
                self.logger.debug("Attempting to retrieve API key from secrets manager")
                key_bytes = self.secrets_manager.get_key("OPENROUTER_API_KEY")
                api_key = key_bytes.decode('utf-8')
                self.logger.debug("Successfully retrieved API key from secrets manager")
            except Exception as e:
                self.logger.warning(f"Failed to retrieve API key from secrets manager: {str(e)}")
            # If still None, fall back to environment variable
            if not api_key:
                api_key = os.environ.get("OPENROUTER_API_KEY", "")
                
            # If still no API key, raise error
            if not api_key:
                self.logger.error("No API key available from any source")
                raise AuthenticationError(
                    "API key is required. Either pass api_key parameter, use a secrets manager, "
                    "or set OPENROUTER_API_KEY environment variable"
                )
        
        # Securely store the API key if PyNaCl is available, otherwise store as plaintext
        if NACL_AVAILABLE and self._secure_box is not None:
            self.api_key = self._encrypt_sensitive_data(api_key)
            self.logger.debug("API key stored with secure encryption")
        else:
            self.api_key = api_key
        
        # Get provisioning API key with similar priority
        if provisioning_api_key is None:
            # If no secrets manager, create an environment secrets manager
            if self.secrets_manager is None:
                self.secrets_manager = EnvironmentSecretsManager()
            
            # Try to retrieve provisioning API key from secrets manager
            try:
                self.logger.debug("Attempting to retrieve provisioning API key from secrets manager")
                key_bytes = self.secrets_manager.get_key("OPENROUTER_PROVISIONING_API_KEY")
                provisioning_api_key = key_bytes.decode('utf-8')
                self.logger.debug("Successfully retrieved provisioning API key from secrets manager")
            except Exception as e:
                self.logger.warning(f"Failed to retrieve provisioning API key from secrets manager: {str(e)}")
            
            # If still None, fall back to environment variable
            if not provisioning_api_key:
                provisioning_api_key = os.environ.get("OPENROUTER_PROVISIONING_API_KEY", "")
        
        # Securely store the provisioning API key if PyNaCl is available, otherwise store as plaintext
        if NACL_AVAILABLE and self._secure_box is not None and provisioning_api_key:
            self.provisioning_api_key = self._encrypt_sensitive_data(provisioning_api_key)
            self.logger.debug("Provisioning API key stored with secure encryption")
        else:
            self.provisioning_api_key = provisioning_api_key
        
        # Store tracking IDs
        self.organization_id = organization_id
        self.reference_id = reference_id
        
        # Log successful initialization
        self.logger.debug(
            f"Auth manager initialized with API key, "
            f"provisioning key {'available' if self.provisioning_api_key else 'not available'}"
        )
        
    def get_auth_headers(self, require_provisioning: bool = False, http_referrer: Optional[str] = None, x_title: Optional[str] = None) -> Dict[str, str]:
        """
        Get authentication headers for API requests.
        
        Args:
            require_provisioning (bool): Whether provisioning API key is required. Defaults to False.
            http_referrer (Optional[str]): HTTP referrer URL to include in request headers. Defaults to None.
            x_title (Optional[str]): Title of the page or document to include in request headers. Defaults to None.
        
        Returns:
            Dict[str, str]: Headers containing authentication information.
        
        Raises:
            AuthenticationError: If provisioning API key is required but not available.
        """
        # Initialize empty headers dictionary
        headers = {}
        
        # Add authorization header with appropriate API key
        if require_provisioning:
            if not self.provisioning_api_key:
                self.logger.error("Provisioning API key required but not available")
                raise AuthenticationError(
                    "Provisioning API key is required for this operation. "
                    "Either pass provisioning_api_key parameter, use a secrets manager, "
                    "or set OPENROUTER_PROVISIONING_API_KEY environment variable"
                )
            # Get the provisioning API key securely if encrypted
            headers["Authorization"] = f"Bearer {self._get_secure_key(self.provisioning_api_key)}"
        else:
            # Get the API key securely if encrypted
            headers["Authorization"] = f"Bearer {self._get_secure_key(self.api_key)}"
        
        # Add organization ID header if provided
        if self.organization_id:
            headers["HTTP-OpenRouter-Organization"] = self.organization_id
        
        # Add reference ID header if provided
        if self.reference_id:
            headers["X-Request-Reference-ID"] = self.reference_id
            
        # Add standard headers for OpenRouter API
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        
        # Add HTTP-Referrer header if provided
        if http_referrer:
            # "HTTP-Referer" is not a typo -- it's in the OpenRouter API spec
            headers["HTTP-Referer"] = http_referrer
            
        # Add X-Title header if provided
        if x_title:
            headers["X-Title"] = x_title
        
        # Log headers (excluding sensitive information)
        log_headers = headers.copy()
        if "Authorization" in log_headers:
            log_headers["Authorization"] = "Bearer ***"
        self.logger.debug(f"Generated auth headers: {log_headers}")
        
        return headers
    
    def _initialize_encryption(self):
        """
        Initialize the PyNaCl encryption for secure key handling.
        Creates a secure random encryption key and SecretBox for encryption/decryption.
        """
        if not NACL_AVAILABLE:
            return
        
        try:
            # Create a secure random encryption key
            self._encryption_key = bytearray(nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE))
            # Create encryption box
            self._secure_box = nacl.secret.SecretBox(bytes(self._encryption_key))
            self.logger.debug("Secure encryption initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize secure encryption: {str(e)}")
            self._secure_box = None
            # Wipe the key if it was created
            if self._encryption_key:
                self._secure_wipe(self._encryption_key)
                self._encryption_key = None
    
    def _encrypt_sensitive_data(self, data: str) -> bytes:
        """
        Encrypt sensitive data using PyNaCl.
        
        Args:
            data (str): Sensitive data to encrypt.
            
        Returns:
            bytes: Encrypted data.
        """
        if not NACL_AVAILABLE or not self._secure_box:
            return data
        
        try:
            # Convert to bytearray for secure handling
            sensitive_data = bytearray(data.encode('utf-8'))
            # Encrypt the data
            encrypted = self._secure_box.encrypt(bytes(sensitive_data))
            # Securely wipe the plaintext data
            self._secure_wipe(sensitive_data)
            return encrypted
        except Exception as e:
            self.logger.warning(f"Failed to encrypt sensitive data: {str(e)}")
            return data
    
    def _decrypt_sensitive_data(self, encrypted_data: Union[bytes, bytearray]) -> str:
        """
        Decrypt sensitive data using PyNaCl.
        
        Args:
            encrypted_data (bytes): Encrypted data.
            
        Returns:
            str: Decrypted data.
        """
        if not NACL_AVAILABLE or not self._secure_box or not isinstance(encrypted_data, (bytes, bytearray)):
            return encrypted_data
        
        try:
            # Ensure encrypted_data is a bytes object
            if isinstance(encrypted_data, bytearray):
                encrypted_data = bytes(encrypted_data)
            # Decrypt the data
            decrypted = self._secure_box.decrypt(encrypted_data)
            # Convert to string
            result = decrypted.decode('utf-8')
            return result
        except Exception as e:
            self.logger.warning(f"Failed to decrypt sensitive data: {str(e)}")
            return str(encrypted_data)
    
    def _get_secure_key(self, key: Union[str, bytes, bytearray]) -> str:
        """
        Get a decrypted key if it's encrypted, or return as-is if not.
        
        Args:
            key (Union[str, bytes, bytearray]): The key to retrieve securely.
            
        Returns:
            str: The decrypted key.
        """
        if not NACL_AVAILABLE or not self._secure_box or isinstance(key, str):
            return key
        
        return self._decrypt_sensitive_data(key)
    
    def _secure_wipe(self, data: Union[bytearray, str]) -> None:
        """
        Securely wipe sensitive data from memory.
        
        Args:
            data (Union[bytearray, str]): Data to wipe.
        """
        if isinstance(data, bytearray):
            for i in range(len(data)):
                data[i] = 0
        elif isinstance(data, str) and hasattr(data, '_sa_instance_state'):
            # For SQLAlchemy or other mutable string-like objects
            if hasattr(data, 'replace'):
                data = '0' * len(data)
    
    def __del__(self):
        """
        Destructor to ensure secure cleanup of sensitive data.
        """
        try:
            # Wipe the encryption key if it was created
            if self._encryption_key:
                self._secure_wipe(self._encryption_key)
                self._encryption_key = None
        except Exception as e:
            # Log the exception if logger is available
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Exception during cleanup in __del__: {str(e)}")