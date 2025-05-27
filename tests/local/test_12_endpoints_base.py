import pytest
import logging
from pydantic import ValidationError

from openrouter_client.endpoints.base import BaseEndpoint
from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager


class TestBaseEndpointInit01NominalBehaviors:
    @pytest.mark.parametrize(
        "endpoint_path, expected_path",
        [
            ("api/v1/models", "api/v1/models"),
            ("/api/v1/models", "api/v1/models"),
            ("///api/v1/models", "api/v1/models"),
            ("", ""),
            ("/", ""),
        ],
        ids=[
            "no_leading_slash",
            "single_leading_slash",
            "multiple_leading_slashes",
            "empty_string",
            "only_slash",
        ],
    )
    def test_initialization(self, endpoint_path: str, expected_path: str, mocker):
        """
        Tests that the BaseEndpoint initializes correctly with various endpoint paths,
        ensuring leading slashes are stripped and attributes are set as expected.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        # Mock the logger to prevent actual logging during tests
        real_logger = logging.getLogger("test")
        mocker.patch.object(logging, 'getLogger', return_value=real_logger)
        endpoint = BaseEndpoint(auth_manager, http_manager, endpoint_path)
        assert endpoint.auth_manager == auth_manager
        assert endpoint.http_manager == http_manager
        assert endpoint.endpoint_path == expected_path
        logging.getLogger.assert_called_once()

class TestBaseEndpointInit02NegativeBehaviors:
    def test_invalid_argument_types(self):
        """
        Tests that BaseEndpoint raises a TypeError when initialized with incorrect argument types.
        """
        with pytest.raises(ValidationError):
            BaseEndpoint(None, None, sum)

    def test_insufficient_arguments(self):
        """
        Tests that BaseEndpoint raises a TypeError when initialized with too few arguments.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        with pytest.raises(TypeError):
            BaseEndpoint(auth_manager, http_manager)

    def test_too_many_arguments(self):
        """
        Tests that BaseEndpoint raises a TypeError when initialized with too many arguments.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        with pytest.raises(TypeError):
            BaseEndpoint(auth_manager, http_manager, "path", "extra")

class TestBaseEndpointInit03BoundaryBehaviors:
    def test_long_endpoint_path(self, mocker):
        """
        Tests that BaseEndpoint handles a very long endpoint_path without errors.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        long_path = "a" * 2048  # A very long string
        real_logger = logging.getLogger("test")
        mocker.patch.object(logging, 'getLogger', return_value=real_logger)
        endpoint = BaseEndpoint(auth_manager, http_manager, long_path)
        assert endpoint.endpoint_path == long_path
        logging.getLogger.assert_called_once()

class TestBaseEndpointInit04ErrorHandlingBehaviors:
    def test_logger_creation_failure(self, mocker):
        """
        Tests that BaseEndpoint handles exceptions during logger creation gracefully,
        falling back through the logger hierarchy until a working logger is found.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        
        # Save reference to original getLogger function before mocking
        original_getLogger = logging.getLogger
        
        # Create a selective mock that fails specific logger calls but allows root logger
        def selective_getLogger(name=None):
            if name is None:  # Allow root logger (final fallback) to work
                return original_getLogger(name)  # Call original, not mocked version
            elif name == "__main__" or "endpoints" in str(name):
                # Fail the first two attempts (endpoint-specific and module-level)
                raise Exception("Failed to create logger")
            else:
                # Allow other loggers to work normally
                return original_getLogger(name)  # Call original, not mocked version
        
        # Apply the selective mock
        mock_getLogger = mocker.patch.object(
            logging, 'getLogger',
            side_effect=selective_getLogger
        )
        
        # Test that initialization succeeds despite logger creation failures
        endpoint = BaseEndpoint(auth_manager, http_manager, "test_path")
        
        # Verify that the endpoint was created successfully
        assert endpoint.auth_manager == auth_manager
        assert endpoint.http_manager == http_manager
        assert endpoint.endpoint_path == "test_path"
        
        # Verify that a logger was eventually assigned (the root logger fallback)
        assert hasattr(endpoint, 'logger')
        assert endpoint.logger is not None
        
        # Verify that getLogger was called multiple times (showing fallback attempts)
        assert mock_getLogger.call_count >= 2
        
        # Verify the expected call pattern
        actual_calls = mock_getLogger.call_args_list
        assert len(actual_calls) == 3  # endpoint-specific, module-level, root
        
        # Check that we got the expected progression of calls
        call_args = [call[0][0] if call[0] else None for call in actual_calls]
        assert "baseendpoint" in call_args[0]  # First call should be endpoint-specific
        assert "base" in call_args[1]  # Second call should be module-level  
        assert call_args[2] is None  # Third call should be root logger (no args)

class TestBaseEndpointGetHeaders01NominalBehaviors:
    def test_headers_returned(self, mocker):
        """
        Tests that _get_headers combines authentication headers with standard headers correctly.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        endpoint = BaseEndpoint(auth_manager, http_manager, "test_path")
        auth_headers = {"Authorization": "Bearer test_token"}
        mocker.patch.object(auth_manager, 'get_auth_headers', return_value=auth_headers)
        headers = endpoint._get_headers()
        assert "Content-Type" in headers
        assert "Accept" in headers
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_token"
        auth_manager.get_auth_headers.assert_called_once_with(require_provisioning=False)

    def test_headers_returned_require_provisioning(self, mocker):
        """
        Tests that _get_headers combines authentication headers with standard headers correctly
        when require_provisioning is True.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        endpoint = BaseEndpoint(auth_manager, http_manager, "test_path")
        auth_headers = {"Authorization": "Bearer test_token"}
        mocker.patch.object(auth_manager, 'get_auth_headers', return_value=auth_headers)
        headers = endpoint._get_headers(require_provisioning=True)
        assert "Content-Type" in headers
        assert "Accept" in headers
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_token"
        auth_manager.get_auth_headers.assert_called_once_with(require_provisioning=True)

class TestBaseEndpointGetHeaders02NegativeBehaviors:
    def test_auth_manager_returns_empty_dict(self, mocker):
        """
        Tests that _get_headers handles the case where auth_manager returns an empty dictionary.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        endpoint = BaseEndpoint(auth_manager, http_manager, "test_path")
        mocker.patch.object(auth_manager, 'get_auth_headers', return_value={})
        headers = endpoint._get_headers()
        assert "Content-Type" in headers
        assert "Accept" in headers
        assert "Authorization" not in headers

class TestBaseEndpointGetHeaders03BoundaryBehaviors:
    @pytest.mark.parametrize("require_provisioning", [True, False], ids=["provisioning", "no_provisioning"])
    def test_require_provisioning_flag(self, mocker, require_provisioning: bool):
        """
        Tests that _get_headers correctly passes the require_provisioning flag to auth_manager.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        endpoint = BaseEndpoint(auth_manager, http_manager, "test_path")
        mocker.patch.object(auth_manager, 'get_auth_headers', return_value={})
        endpoint._get_headers(require_provisioning=require_provisioning)
        auth_manager.get_auth_headers.assert_called_once_with(require_provisioning=require_provisioning)

    def test_long_header_values(self, mocker):
        """
        Tests that _get_headers handles long header values without errors.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        endpoint = BaseEndpoint(auth_manager, http_manager, "test_path")
        long_header = {"Authorization": "Bearer " + "a" * 2048}
        mocker.patch.object(auth_manager, 'get_auth_headers', return_value=long_header)
        headers = endpoint._get_headers()
        assert headers["Authorization"] == long_header["Authorization"]

class TestBaseEndpointGetHeaders04ErrorHandlingBehaviors:
    def test_auth_manager_raises_exception(self, mocker):
        """
        Tests that _get_headers handles exceptions raised by auth_manager gracefully.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        endpoint = BaseEndpoint(auth_manager, http_manager, "test_path")
        mocker.patch.object(auth_manager, 'get_auth_headers', side_effect=Exception("Auth failed"))
        with pytest.raises(Exception, match="Auth failed"):
            endpoint._get_headers()

class TestBaseEndpointGetEndpointUrl01NominalBehaviors:
    @pytest.mark.parametrize(
        "endpoint_path, path, expected_url",
        [
            ("api/v1", "models", "api/v1/models"),
            ("api/v1/", "models", "api/v1/models"),
            ("api/v1", "/models", "api/v1/models"),
            ("api/v1/", "/models", "api/v1/models"),
            ("", "models", "models"),
            ("api/v1", "", "api/v1"),
            ("", "", ""),
        ],
        ids=[
            "both_paths",
            "endpoint_path_trailing_slash",
            "path_leading_slash",
            "both_slashes",
            "only_path",
            "only_endpoint_path",
            "neither_path",
        ],
    )
    def test_url_generation(self, endpoint_path: str, path: str, expected_url: str):
        """
        Tests that _get_endpoint_url combines the endpoint_path with the provided path correctly,
        handling various combinations of slashes and empty strings.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        endpoint = BaseEndpoint(auth_manager, http_manager, endpoint_path)
        url = endpoint._get_endpoint_url(path)
        assert url == expected_url

class TestBaseEndpointGetEndpointUrl02NegativeBehaviors:
    def test_invalid_url_characters(self):
        """
        Tests that _get_endpoint_url handles invalid URL characters in endpoint_path or path,
        though URL validation is not explicitly performed in the provided code.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        endpoint = BaseEndpoint(auth_manager, http_manager, "api/v1")
        path_with_invalid_chars = "models?param=value#fragment"
        url = endpoint._get_endpoint_url(path_with_invalid_chars)
        assert url == "api/v1/models?param=value#fragment"

class TestBaseEndpointGetEndpointUrl03BoundaryBehaviors:
    def test_path_starts_with_slash(self):
        """
        Tests that _get_endpoint_url correctly handles a path that starts with a slash.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        endpoint = BaseEndpoint(auth_manager, http_manager, "api/v1")
        url = endpoint._get_endpoint_url("/models")
        assert url == "api/v1/models"

    def test_long_paths(self):
        """
        Tests that _get_endpoint_url handles very long endpoint_path and path strings without errors.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        long_path = "a" * 2048
        endpoint = BaseEndpoint(auth_manager, http_manager, long_path)
        url = endpoint._get_endpoint_url(long_path)
        assert url == f"{long_path}/{long_path}"

    def test_empty_path(self):
        """
        Tests that _get_endpoint_url handles an empty path correctly.
        """
        auth_manager = AuthManager()
        http_manager = HTTPManager(base_url="https://example.com")
        endpoint = BaseEndpoint(auth_manager, http_manager, "api/v1")
        url = endpoint._get_endpoint_url()
        assert url == "api/v1"
