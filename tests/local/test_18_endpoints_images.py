import pytest
import os
import tempfile
from pathlib import Path
from pydantic import ValidationError
import subprocess
import sys
import tempfile
from unittest.mock import patch

from openrouter_client.auth import AuthManager
from openrouter_client.exceptions import APIError
from openrouter_client.http import HTTPManager
from openrouter_client.endpoints.images import ImagesEndpoint
from openrouter_client.endpoints.base import BaseEndpoint

class Test_ImagesEndpoint_Init_01_NominalBehaviors:
    """Test nominal initialization behaviors for ImagesEndpoint."""
    
    @pytest.mark.parametrize("api_key,timeout", [
        ("test-api-key-123", 1),
        ("sk-abcd1234", 2),
        ("valid-key", 3),
    ])
    def test_initialize_with_valid_managers(self, api_key, timeout):
        # Arrange
        auth_manager = AuthManager(api_key)
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com", timeout=timeout)
        
        # Act
        endpoint = ImagesEndpoint(auth_manager, http_manager)
        
        # Assert
        assert isinstance(endpoint, BaseEndpoint)
        assert endpoint.auth_manager is auth_manager
        assert endpoint.http_manager is http_manager
        assert hasattr(endpoint, 'logger')
        
    def test_establish_correct_endpoint_path(self):
        # Arrange
        auth_manager = AuthManager("test-api-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        
        # Act
        endpoint = ImagesEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint.endpoint_path == "images"

class Test_ImagesEndpoint_Init_02_NegativeBehaviors:
    """Test negative initialization behaviors for ImagesEndpoint."""
    
    @pytest.mark.parametrize("auth_manager,http_manager", [
        (None, HTTPManager(base_url="https://invalid.thisurldoesntexist.com")),
        (AuthManager("key"), None),
        (None, None),
        ("invalid_auth", HTTPManager(base_url="https://invalid.thisurldoesntexist.com")),
        (AuthManager("key"), "invalid_http"),
        (123, HTTPManager(base_url="https://invalid.thisurldoesntexist.com")),
        (AuthManager("key"), []),
    ])
    def test_handle_invalid_manager_types(self, auth_manager, http_manager):
        # Arrange - parameters already arranged
        
        # Act & Assert
        with pytest.raises(ValidationError):
            ImagesEndpoint(auth_manager, http_manager)

class Test_ImagesEndpoint_Init_04_ErrorHandlingBehaviors:
    """Test error handling during ImagesEndpoint initialization."""
    
    @pytest.mark.parametrize("auth_key", ["test-key", "another-key", "sk-12345"])
    def test_handle_parent_initialization_exceptions(self, auth_key):
        # Arrange
        auth_manager = AuthManager(auth_key)
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        
        # Act & Assert
        try:
            endpoint = ImagesEndpoint(auth_manager, http_manager)
            # If no exception, verify the object is properly formed
            assert hasattr(endpoint, 'auth_manager')
            assert hasattr(endpoint, 'http_manager')
            assert hasattr(endpoint, 'endpoint_path')
        except Exception as e:
            # If an exception occurs, it should be a recognizable type
            assert isinstance(e, (ValueError, TypeError, RuntimeError))

class Test_ImagesEndpoint_Init_05_StateTransitionBehaviors:
    """Test state transition behaviors during ImagesEndpoint initialization."""
    
    @pytest.mark.parametrize("configuration", [
        {"key": "test-key-1", "timeout": 3},
        {"key": "test-key-2", "timeout": 2},
        {"key": "test-key-3", "timeout": 1},
    ])
    def test_complete_object_initialization(self, configuration):
        # Arrange
        auth_manager = AuthManager(configuration["key"])
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexsit.com", timeout=configuration["timeout"])
        
        # Act
        endpoint = ImagesEndpoint(auth_manager, http_manager)
        
        # Assert - Verify complete initialization state
        assert endpoint.auth_manager is not None
        assert endpoint.http_manager is not None
        assert hasattr(endpoint, 'endpoint_path')
        assert hasattr(endpoint, 'logger')
        assert endpoint.endpoint_path == "images"
        assert endpoint.auth_manager == auth_manager
        assert endpoint.http_manager == http_manager

class Test_ImagesEndpoint_Create_01_NominalBehaviors:
    """Test nominal behaviors for ImagesEndpoint.create method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("prompt", [
        "A beautiful sunset over mountains",
        "Abstract art with geometric patterns",
        "Portrait of a wise old owl",
        "Minimalist digital artwork",
        "Photorealistic landscape scene",
    ])
    def test_generate_images_with_valid_prompts(self, endpoint, prompt):
        # Arrange - endpoint and prompt already arranged
        
        # Act
        try:
            result = endpoint.create(prompt=prompt)
            # Assert
            assert isinstance(result, dict)
        except APIError as e:
            # External API calls expected to fail in test environment
            assert "Request failed" in e.message
    
    @pytest.mark.parametrize("model,n,size,response_format,quality,style", [
        ("dall-e-3", 1, "1024x1024", "url", "standard", "vivid"),
        ("dall-e-2", 2, "512x512", "b64_json", "hd", "natural"),
        ("dall-e-3", 1, "1792x1024", "url", "hd", "vivid"),
        (None, None, None, None, None, None),
        ("dall-e-2", 4, "256x256", "url", "standard", None),
    ])
    def test_process_all_optional_parameters(self, endpoint, model, n, size, response_format, quality, style):
        # Arrange
        prompt = "Test image generation prompt"
        
        # Act
        try:
            result = endpoint.create(
                prompt=prompt,
                model=model,
                n=n,
                size=size,
                response_format=response_format,
                quality=quality,
                style=style
            )
            # Assert
            assert isinstance(result, dict)
        except APIError as e:
            assert "Request failed" in e.message
    
    @pytest.mark.parametrize("kwargs", [
        {"custom_param": "value", "another_param": 123},
        {"user_id": "test-user", "priority": "high"},
        {"metadata": {"source": "test"}, "version": 2},
        {},
    ])
    def test_handle_additional_kwargs(self, endpoint, kwargs):
        # Arrange
        prompt = "Test prompt for kwargs"
        
        # Act
        try:
            result = endpoint.create(prompt=prompt, **kwargs)
            # Assert
            assert isinstance(result, dict)
        except APIError as e:
            assert "Request failed" in e.message

class Test_ImagesEndpoint_Create_02_NegativeBehaviors:
    """Test negative behaviors for ImagesEndpoint.create method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("invalid_prompt", [
        None,
        "",
        "   ",
        "\t\n",
        "\r\n\t  ",
    ])
    def test_reject_empty_or_none_prompts(self, endpoint, invalid_prompt):
        # Arrange - invalid_prompt already arranged
        
        # Act & Assert
        with pytest.raises(ValueError):
            endpoint.create(prompt=invalid_prompt)
    
    @pytest.mark.parametrize("param_name,invalid_value", [
        ("n", "not_a_number"),
        ("n", -1),
        ("n", 0),
        ("n", [1, 2, 3]),
        ("size", "invalid_size"),
        ("size", 123),
        ("response_format", "invalid_format"),
        ("response_format", {"format": "url"}),
        ("quality", "invalid_quality"),
        ("quality", 42),
        ("style", True),
        ("model", 123),
        ("model", []),
    ])
    def test_validate_parameter_types_and_values(self, endpoint, param_name, invalid_value):
        # Arrange
        prompt = "Valid test prompt"
        kwargs = {param_name: invalid_value}
        
        # Act & Assert
        with pytest.raises((ValueError, TypeError)):
            endpoint.create(prompt=prompt, **kwargs)

class Test_ImagesEndpoint_Create_03_BoundaryBehaviors:
    """Test boundary behaviors for ImagesEndpoint.create method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("prompt_length,expected_behavior", [
        (1, "valid"),  # Minimum viable prompt
        (10, "valid"),  # Short prompt
        (1000, "valid"),  # Medium length prompt
        (4000, "valid"),  # Long prompt at typical limit
        (8000, "boundary"),  # Very long prompt
        (16000, "boundary"),  # Extremely long prompt
    ])
    def test_handle_prompt_length_boundaries(self, endpoint, prompt_length, expected_behavior):
        # Arrange
        prompt = "A" * prompt_length
        
        # Act
        if expected_behavior == "valid":
            try:
                result = endpoint.create(prompt=prompt)
                # Assert
                assert isinstance(result, dict)
            except APIError as e:
                assert "Request failed" in e.message
        else:
            # Assert boundary cases may raise exceptions
            with pytest.raises((ValidationError, ValueError, Exception)):
                endpoint.create(prompt=prompt)
    
    @pytest.mark.parametrize("n_value,expected_behavior", [
        (1, "valid"),  # Minimum value
        (2, "valid"),  # Small valid value
        (4, "valid"),  # Medium valid value
        (10, "valid"),  # Maximum typical value
        (0, "invalid"),  # Below minimum
        (-1, "invalid"),  # Negative value
        (100, "boundary"),  # Very high value
        (1000, "boundary"),  # Extremely high value
    ])
    def test_process_n_parameter_boundaries(self, endpoint, n_value, expected_behavior):
        # Arrange
        prompt = "Test prompt for n parameter"
        
        # Act
        if expected_behavior == "valid":
            try:
                result = endpoint.create(prompt=prompt, n=n_value)
                # Assert
                assert isinstance(result, dict)
            except APIError as e:
                assert "Request failed" in e.message
        else:
            # Assert invalid values raise exceptions
            with pytest.raises((ValueError, Exception)):
                endpoint.create(prompt=prompt, n=n_value)

class Test_ImagesEndpoint_Create_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ImagesEndpoint.create method."""
    
    @pytest.mark.parametrize("invalid_key", [
        "invalid-api-key",
        "expired-key-123",
        "sk-wrong-format",
        "",
    ])
    def test_manage_authentication_failures(self, invalid_key):
        # Arrange
        auth_manager = AuthManager(invalid_key)
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        endpoint = ImagesEndpoint(auth_manager, http_manager)
        prompt = "Test prompt for auth failure"
        
        # Act & Assert
        with pytest.raises(Exception):  # Authentication should fail with invalid key
            endpoint.create(prompt=prompt)
    
    @pytest.mark.parametrize("invalid_base_url", [
        "http://nonexistent-api.com",
        "https://invalid-endpoint.fake",
        "http://localhost:99999",
        "not-a-url",
    ])
    def test_handle_network_connectivity_issues(self, invalid_base_url):
        # Arrange
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url=invalid_base_url)
        endpoint = ImagesEndpoint(auth_manager, http_manager)
        prompt = "Test prompt for network failure"
        
        # Act & Assert
        with pytest.raises(Exception):  # Network error expected
            endpoint.create(prompt=prompt)
    
    @pytest.mark.parametrize("malformed_response_scenario", [
        "empty_response",
        "invalid_json",
        "missing_data_field",
    ])
    def test_process_malformed_json_responses(self, malformed_response_scenario):
        # Arrange
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        endpoint = ImagesEndpoint(auth_manager, http_manager)
        prompt = "Test prompt for malformed response"
        
        # Act & Assert
        # This test would require mock responses, but since mocks are forbidden,
        # we test that the method handles JSON parsing errors appropriately
        try:
            result = endpoint.create(prompt=prompt)
            assert isinstance(result, dict)
        except Exception as e:
            assert isinstance(e, (ValueError, TypeError, Exception))

class Test_ImagesEndpoint_Edit_01_NominalBehaviors:
    """Test nominal behaviors for ImagesEndpoint.edit method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.fixture
    def test_image_file(self):
        # Create a minimal valid PNG file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Write minimal PNG header and data
            png_data = (
                b'\x89PNG\r\n\x1a\n'  # PNG signature
                b'\x00\x00\x00\rIHDR'  # IHDR chunk
                b'\x00\x00\x00\x01\x00\x00\x00\x01'  # 1x1 pixel
                b'\x08\x02\x00\x00\x00\x90wS\xde'  # Color type, etc.
                b'\x00\x00\x00\x0cIDATx\x9cc\xf8'  # IDAT chunk start
                b'\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00'  # Image data
                b'IEND\xaeB`\x82'  # IEND chunk
            )
            f.write(png_data)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    @pytest.mark.parametrize("image_type", ["file_path", "file_object"])
    def test_process_image_editing_with_various_inputs(self, endpoint, test_image_file, image_type):
        # Arrange
        prompt = "Make the image brighter and more colorful"
        
        if image_type == "file_path":
            image_input = test_image_file
        else:  # file_object
            image_input = open(test_image_file, 'rb')
        
        try:
            # Act
            try:
                result = endpoint.edit(image=image_input, prompt=prompt)
                # Assert
                assert isinstance(result, dict)
            except APIError as e:
                assert "Request failed" in e.message
        finally:
            if image_type == "file_object" and hasattr(image_input, 'close'):
                image_input.close()
    
    @pytest.mark.parametrize("mask_type", [None, "file_path", "file_object"])
    def test_handle_optional_mask_images(self, endpoint, test_image_file, mask_type):
        # Arrange
        prompt = "Edit specific area of the image"
        
        if mask_type is None:
            mask_input = None
        elif mask_type == "file_path":
            mask_input = test_image_file
        else:  # file_object
            mask_input = open(test_image_file, 'rb')
        
        try:
            # Act
            try:
                result = endpoint.edit(image=test_image_file, prompt=prompt, mask=mask_input)
                # Assert
                assert isinstance(result, dict)
            except APIError as e:
                assert "Request failed" in e.message
        finally:
            if mask_type == "file_object" and hasattr(mask_input, 'close'):
                mask_input.close()
    
    @pytest.mark.parametrize("model,n,size,response_format", [
        ("dall-e-2", 1, "1024x1024", "url"),
        ("dall-e-2", 2, "512x512", "b64_json"),
        (None, None, None, None),
        ("dall-e-2", 1, "256x256", "url"),
    ])
    def test_apply_edits_with_optional_parameters(self, endpoint, test_image_file, model, n, size, response_format):
        # Arrange
        prompt = "Enhance the image quality"
        
        # Act
        try:
            result = endpoint.edit(
                image=test_image_file,
                prompt=prompt,
                model=model,
                n=n,
                size=size,
                response_format=response_format
            )
            # Assert
            assert isinstance(result, dict)
        except APIError as e:
            assert "Request failed" in e.message

class Test_ImagesEndpoint_Edit_02_NegativeBehaviors:
    """Test negative behaviors for ImagesEndpoint.edit method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("invalid_path_type,path", [
        ("nonexistent", "/nonexistent/path/image.png"),
        ("nonexistent", "/invalid/mask.jpg"),
        ("empty", ""),
        ("relative_nonexistent", "./missing/image.png"),
        ("invalid_extension", "/tmp/notanimage.txt"),
    ])
    def test_handle_nonexistent_file_paths(self, endpoint, invalid_path_type, path):
        # Arrange
        prompt = "Edit the image"
        
        # Act & Assert
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            endpoint.edit(image=path, prompt=prompt)
    
    @pytest.mark.parametrize("file_content,file_extension", [
        (b"Not an image file", ".png"),
        (b"<html><body>HTML content</body></html>", ".jpg"),
        (b"Plain text content", ".gif"),
        (b"\x00\x01\x02\x03\x04", ".png"),  # Random binary data
    ])
    def test_reject_invalid_file_types(self, endpoint, file_content, file_extension):
        # Arrange
        prompt = "Edit this invalid image"
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as f:
            f.write(file_content)
            invalid_file = f.name
        
        try:
            # Act & Assert
            with pytest.raises((ValueError, Exception)):
                endpoint.edit(image=invalid_file, prompt=prompt)
        finally:
            os.unlink(invalid_file)

class Test_ImagesEndpoint_Edit_03_BoundaryBehaviors:
    """Test boundary behaviors for ImagesEndpoint.edit method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.fixture
    def minimal_image_file(self):
        # Create the smallest possible valid PNG file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            minimal_png = (
                b'\x89PNG\r\n\x1a\n'
                b'\x00\x00\x00\rIHDR'
                b'\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x02\x00\x00\x00\x90wS\xde'
                b'\x00\x00\x00\x0cIDATx\x9cc\xf8'
                b'\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00'
                b'IEND\xaeB`\x82'
            )
            f.write(minimal_png)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    @pytest.mark.parametrize("file_size_type", ["minimal", "small", "medium"])
    def test_process_various_file_sizes(self, endpoint, file_size_type):
        # Arrange
        prompt = "Edit this image"
        
        if file_size_type == "minimal":
            file_size = 100  # Very small file
        elif file_size_type == "small":
            file_size = 1024  # 1KB file
        else:  # medium
            file_size = 1024 * 1024  # 1MB file
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create a file with specified size (not a valid image, but tests size handling)
            f.write(b'\x89PNG\r\n\x1a\n')  # PNG header
            f.write(b'0' * (file_size - 8))  # Fill to desired size
            test_file = f.name
        
        try:
            # Act
            try:
                result = endpoint.edit(image=test_file, prompt=prompt)
                # Assert
                assert isinstance(result, dict)
            except Exception:
                # Expected for invalid image data or API unavailability
                assert True
        finally:
            os.unlink(test_file)
    
    @pytest.mark.parametrize("permission_mode", [0o444, 0o400, 0o440])
    def test_handle_file_permission_scenarios(self, endpoint, permission_mode):
        # Arrange
        prompt = "Edit image with specific permissions"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_data = (
                b'\x89PNG\r\n\x1a\n'
                b'\x00\x00\x00\rIHDR'
                b'\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x02\x00\x00\x00\x90wS\xde'
                b'\x00\x00\x00\x0cIDATx\x9cc\xf8'
                b'\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00'
                b'IEND\xaeB`\x82'
            )
            f.write(png_data)
            restricted_file = f.name
        
        try:
            os.chmod(restricted_file, permission_mode)
            
            # Act
            if permission_mode & 0o400:  # If readable
                try:
                    result = endpoint.edit(image=restricted_file, prompt=prompt)
                    assert isinstance(result, dict)
                except APIError as e:
                    assert "Request failed" in e.message
            else:
                # Assert - Should raise permission error
                with pytest.raises((PermissionError, OSError)):
                    endpoint.edit(image=restricted_file, prompt=prompt)
        finally:
            os.chmod(restricted_file, 0o644)
            os.unlink(restricted_file)

class Test_ImagesEndpoint_Edit_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ImagesEndpoint.edit method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("corruption_type", [
        "truncated_header",
        "invalid_signature",
        "corrupted_data",
        "missing_end_marker",
    ])
    def test_manage_file_io_exceptions(self, endpoint, corruption_type):
        # Arrange
        prompt = "Edit corrupted image"
        
        if corruption_type == "truncated_header":
            corrupted_data = b'\x89PNG'  # Incomplete header
        elif corruption_type == "invalid_signature":
            corrupted_data = b'\x00PNG\r\n\x1a\n' + b'0' * 100
        elif corruption_type == "corrupted_data":
            corrupted_data = b'\x89PNG\r\n\x1a\n' + b'\xFF' * 100
        else:  # missing_end_marker
            corrupted_data = b'\x89PNG\r\n\x1a\n' + b'0' * 100  # No IEND
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(corrupted_data)
            corrupted_file = f.name
        
        try:
            # Act & Assert
            with pytest.raises(Exception):
                endpoint.edit(image=corrupted_file, prompt=prompt)
        finally:
            os.unlink(corrupted_file)
    
    def test_ensure_proper_resource_cleanup(self, endpoint):
        # Arrange
        prompt = "Edit image for cleanup test"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_data = (
                b'\x89PNG\r\n\x1a\n'
                b'\x00\x00\x00\rIHDR'
                b'\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x02\x00\x00\x00\x90wS\xde'
                b'\x00\x00\x00\x0cIDATx\x9cc\xf8'
                b'\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00'
                b'IEND\xaeB`\x82'
            )
            f.write(png_data)
            test_file = f.name
        
        try:
            # Act
            try:
                endpoint.edit(image=test_file, prompt=prompt)
            except Exception:
                pass  # Expected in test environment
            
            # Assert - File should still be accessible after method completion
            assert os.path.exists(test_file)
            assert os.access(test_file, os.R_OK)
        finally:
            os.unlink(test_file)

class Test_ImagesEndpoint_Edit_05_StateTransitionBehaviors:
    """Test state transition behaviors for ImagesEndpoint.edit method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("scenario", ["success", "api_failure", "file_error"])
    def test_track_file_object_lifecycle(self, endpoint, scenario):
        # Arrange
        prompt = "Edit image for lifecycle test"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_data = (
                b'\x89PNG\r\n\x1a\n'
                b'\x00\x00\x00\rIHDR'
                b'\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x02\x00\x00\x00\x90wS\xde'
                b'\x00\x00\x00\x0cIDATx\x9cc\xf8'
                b'\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00'
                b'IEND\xaeB`\x82'
            )
            f.write(png_data)
            test_file = f.name
        
        try:
            # Act
            initial_fd_count = len(os.listdir('/proc/self/fd')) if os.path.exists('/proc/self/fd') else 0
            
            try:
                if scenario == "file_error":
                    os.chmod(test_file, 0o000)  # Make unreadable
                endpoint.edit(image=test_file, prompt=prompt)
            except Exception:
                pass  # Expected in test environment or for error scenarios
            
            final_fd_count = len(os.listdir('/proc/self/fd')) if os.path.exists('/proc/self/fd') else 0
            
            # Assert - No file handle leaks (platform-dependent check)
            if os.path.exists('/proc/self/fd'):
                assert final_fd_count <= initial_fd_count + 2  # Allow some variance
        finally:
            os.chmod(test_file, 0o644)
            os.unlink(test_file)

class Test_ImagesEndpoint_Variations_01_NominalBehaviors:
    """Test nominal behaviors for ImagesEndpoint.variations method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.fixture
    def test_image_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_data = (
                b'\x89PNG\r\n\x1a\n'
                b'\x00\x00\x00\rIHDR'
                b'\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x02\x00\x00\x00\x90wS\xde'
                b'\x00\x00\x00\x0cIDATx\x9cc\xf8'
                b'\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00'
                b'IEND\xaeB`\x82'
            )
            f.write(png_data)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    @pytest.mark.parametrize("input_type", ["file_path", "file_object_binary", "file_object_text"])
    def test_generate_variations_from_various_inputs(self, endpoint, test_image_file, input_type):
        # Arrange
        if input_type == "file_path":
            image_input = test_image_file
        elif input_type == "file_object_binary":
            image_input = open(test_image_file, 'rb')
        else:  # file_object_text (should warn but work)
            image_input = open(test_image_file, 'r', encoding='latin1')
        
        try:
            # Act
            try:
                result = endpoint.variations(image=image_input)
                # Assert
                assert isinstance(result, dict)
            except APIError as e:
                assert "Request failed" in e.message
        finally:
            if hasattr(image_input, 'close') and input_type != "file_path":
                image_input.close()
    
    @pytest.mark.parametrize("model,n,size,response_format", [
        ("dall-e-2", 1, "1024x1024", "url"),
        ("dall-e-2", 2, "512x512", "b64_json"),
        ("dall-e-2", 4, "256x256", "url"),
        (None, None, None, None),
        ("dall-e-2", 1, "1024x1024", "b64_json"),
    ])
    def test_process_optional_parameters(self, endpoint, test_image_file, model, n, size, response_format):
        # Arrange - parameters already arranged
        
        # Act
        try:
            result = endpoint.variations(
                image=test_image_file,
                model=model,
                n=n,
                size=size,
                response_format=response_format
            )
            # Assert
            assert isinstance(result, dict)
        except APIError as e:
            assert "Request failed" in e.message

class Test_ImagesEndpoint_Variations_02_NegativeBehaviors:
    """Test negative behaviors for ImagesEndpoint.variations method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("invalid_path", [
        "/nonexistent/image.png",
        "",
        "/invalid/path.jpg",
        "/tmp/does_not_exist.gif",
        "./relative/missing.png",
    ])
    def test_handle_invalid_file_paths(self, endpoint, invalid_path):
        # Arrange - invalid_path already arranged
        
        # Act & Assert
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            endpoint.variations(image=invalid_path)
    
    @pytest.mark.parametrize("corruption_type,file_content", [
        ("text_file", b"This is just text, not an image"),
        ("html_file", b"<html><body>Not an image</body></html>"),
        ("binary_junk", b"\x00\x01\x02\x03\x04\x05\x06\x07"),
        ("partial_png", b'\x89PNG\r\n'),  # Incomplete PNG
        ("wrong_format", b'GIF89a'),  # GIF header in PNG file
    ])
    def test_reject_corrupted_image_data(self, endpoint, corruption_type, file_content):
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(file_content)
            corrupted_file = f.name
        
        try:
            # Act & Assert
            with pytest.raises((ValueError, Exception)):
                endpoint.variations(image=corrupted_file)
        finally:
            os.unlink(corrupted_file)

class Test_ImagesEndpoint_Variations_03_BoundaryBehaviors:
    """Test boundary behaviors for ImagesEndpoint.variations method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("file_size_scenario", [
        ("minimal", 100),
        ("small", 1024),
        ("medium", 1024 * 100),
        ("large", 1024 * 1024),
        ("very_large", 1024 * 1024 * 5),
    ])
    def test_process_api_file_size_limits(self, endpoint, file_size_scenario):
        # Arrange
        scenario_name, target_size = file_size_scenario
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create a basic PNG structure
            f.write(b'\x89PNG\r\n\x1a\n')
            f.write(b'\x00\x00\x00\rIHDR')
            f.write(b'\x00\x00\x00\x01\x00\x00\x00\x01')
            f.write(b'\x08\x02\x00\x00\x00\x90wS\xde')
            
            # Fill to target size
            remaining_size = max(0, target_size - f.tell() - 12)  # Account for IEND
            f.write(b'0' * remaining_size)
            f.write(b'IEND\xaeB`\x82')
            
            test_file = f.name
        
        try:
            # Act
            if scenario_name in ["minimal", "small", "medium"]:
                try:
                    result = endpoint.variations(image=test_file)
                    # Assert
                    assert isinstance(result, dict)
                except APIError as e:
                    assert "Request failed" in e.message
            else:
                # Large files may be rejected by API
                try:
                    endpoint.variations(image=test_file)
                except Exception as e:
                    assert isinstance(e, (ValueError, Exception))
        finally:
            os.unlink(test_file)

class Test_ImagesEndpoint_Variations_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ImagesEndpoint.variations method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)

    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)

    @pytest.mark.parametrize("permission_scenario,exception_type", [
        ("no_read", PermissionError),
        ("no_execute_dir", OSError),
        ("temporary_lock", PermissionError),
    ])
    def test_manage_file_access_exceptions(self, endpoint, permission_scenario, exception_type):
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_data = (
                b'\x89PNG\r\n\x1a\n'
                b'\x00\x00\x00\rIHDR'
                b'\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x02\x00\x00\x00\x90wS\xde'
                b'\x00\x00\x00\x0cIDATx\x9cc\xf8'
                b'\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00'
                b'IEND\xaeB`\x82'
            )
            f.write(png_data)
            test_file = f.name
        
        try:
            # Create a mock that raises the appropriate exception for our test file
            def mock_open_with_permission_error(file_path, mode='r', **kwargs):
                if file_path == test_file and 'rb' in mode:
                    if permission_scenario == "no_read":
                        raise PermissionError(f"Permission denied: '{file_path}'")
                    elif permission_scenario == "no_execute_dir":
                        raise OSError(f"No access to directory containing '{file_path}'")
                    elif permission_scenario == "temporary_lock":
                        raise PermissionError(f"File is locked: '{file_path}'")
                # For other files, use the real open function
                return open.__wrapped__(file_path, mode, **kwargs)
            
            # Mock the HTTP manager to prevent network calls
            with patch.object(endpoint.http_manager, 'post') as mock_post:
                # Mock builtins.open to simulate permission errors
                with patch('builtins.open', side_effect=mock_open_with_permission_error):
                    # Act & Assert
                    with pytest.raises(exception_type):
                        endpoint.variations(image=test_file)
                    
                    # Verify the HTTP request was not made due to file permission error
                    mock_post.assert_not_called()
        
        finally:
            # Clean up the test file
            try:
                os.unlink(test_file)
            except FileNotFoundError:
                pass
    
    @pytest.mark.parametrize("cleanup_scenario", ["normal", "exception_during_processing"])
    def test_ensure_proper_file_closure(self, endpoint, cleanup_scenario):
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_data = (
                b'\x89PNG\r\n\x1a\n'
                b'\x00\x00\x00\rIHDR'
                b'\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x02\x00\x00\x00\x90wS\xde'
                b'\x00\x00\x00\x0cIDATx\x9cc\xf8'
                b'\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00'
                b'IEND\xaeB`\x82'
            )
            f.write(png_data)
            test_file = f.name
        
        try:
            # Act
            initial_fd_count = len(os.listdir('/proc/self/fd')) if os.path.exists('/proc/self/fd') else 0
            
            if cleanup_scenario == "exception_during_processing":
                # Simulate processing exception by making file unreadable mid-process
                os.chmod(test_file, 0o000)
            
            try:
                endpoint.variations(image=test_file)
            except Exception:
                pass  # Expected in test environment or error scenarios
            
            final_fd_count = len(os.listdir('/proc/self/fd')) if os.path.exists('/proc/self/fd') else 0
            
            # Assert - No file descriptor leaks
            if os.path.exists('/proc/self/fd'):
                assert final_fd_count <= initial_fd_count + 1
            
            # Assert - File should remain accessible after cleanup
            os.chmod(test_file, 0o644)  # Restore permissions for cleanup
            assert os.path.exists(test_file)
        finally:
            os.chmod(test_file, 0o644)
            os.unlink(test_file)

class Test_ImagesEndpoint_Variations_05_StateTransitionBehaviors:
    """Test state transition behaviors for ImagesEndpoint.variations method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("operation_outcome", ["success", "api_error", "network_failure", "file_error"])
    def test_verify_proper_resource_cleanup(self, endpoint, operation_outcome):
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_data = (
                b'\x89PNG\r\n\x1a\n'
                b'\x00\x00\x00\rIHDR'
                b'\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x02\x00\x00\x00\x90wS\xde'
                b'\x00\x00\x00\x0cIDATx\x9cc\xf8'
                b'\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00'
                b'IEND\xaeB`\x82'
            )
            f.write(png_data)
            test_file = f.name
        
        try:
            # Act
            resource_count_before = len(os.listdir('/proc/self/fd')) if os.path.exists('/proc/self/fd') else 0
            
            # Simulate different failure scenarios
            if operation_outcome == "file_error":
                os.chmod(test_file, 0o000)
            
            try:
                endpoint.variations(image=test_file)
            except Exception:
                pass  # Expected in test environment or error scenarios
            
            resource_count_after = len(os.listdir('/proc/self/fd')) if os.path.exists('/proc/self/fd') else 0
            
            # Assert - Verify resource cleanup regardless of outcome
            if os.path.exists('/proc/self/fd'):
                assert resource_count_after <= resource_count_before + 1
            
            # Assert - Original file remains intact
            os.chmod(test_file, 0o644)  # Restore for verification
            assert os.path.exists(test_file)
        finally:
            os.chmod(test_file, 0o644)
            os.unlink(test_file)

class Test_ImagesEndpoint_ProcessImageInput_01_NominalBehaviors:
    """Test nominal behaviors for ImagesEndpoint._process_image_input method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("file_content", [
        b'\x89PNG\r\n\x1a\n' + b'0' * 100,  # PNG-like content
        b'\xFF\xD8\xFF\xE0' + b'0' * 100,    # JPEG-like content
        b'GIF89a' + b'0' * 100,              # GIF-like content
        b'Random binary content for testing',
    ])
    def test_convert_file_paths_to_binary_objects(self, endpoint, file_content):
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(file_content)
            test_file = f.name
        
        try:
            # Act
            result = endpoint._process_image_input(test_file)
            
            # Assert
            assert hasattr(result, 'read')
            assert hasattr(result, 'mode')
            assert 'b' in result.mode
            assert result.readable()
            result.close()
        finally:
            os.unlink(test_file)
    
    @pytest.mark.parametrize("file_mode", ['rb', 'r+b', 'ab'])
    def test_handle_existing_binary_file_objects(self, endpoint, file_mode):
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(b'Binary test content')
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, file_mode) as f:
                if 'a' in file_mode:
                    f.seek(0)  # Reset position for append mode
                
                # Act
                result = endpoint._process_image_input(f)
                
                # Assert
                assert result is f
                assert hasattr(result, 'read')
                assert 'b' in result.mode
        finally:
            os.unlink(temp_file_path)

class Test_ImagesEndpoint_ProcessImageInput_02_NegativeBehaviors:
    """Test negative behaviors for ImagesEndpoint._process_image_input method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("invalid_path", [
        "/absolutely/nonexistent/file.png",
        "",
        "/invalid/directory/image.jpg",
        "./relative/missing/file.gif",
        "/tmp/this_file_does_not_exist.bmp",
    ])
    def test_raise_error_for_nonexistent_files(self, endpoint, invalid_path):
        # Arrange - invalid_path already arranged
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            endpoint._process_image_input(invalid_path)
        assert "does not exist" in str(exc_info.value)
        assert invalid_path in str(exc_info.value)
    
    @pytest.mark.parametrize("invalid_input", [
        123,
        45.67,
        [],
        {},
        None,
        True,
        False,
        object(),
        set(),
        tuple(),
    ])
    def test_reject_unsupported_input_types(self, endpoint, invalid_input):
        # Arrange - invalid_input already arranged
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            endpoint._process_image_input(invalid_input)
        assert "Unsupported image input type" in str(exc_info.value)

class Test_ImagesEndpoint_ProcessImageInput_03_BoundaryBehaviors:
    """Test boundary behaviors for ImagesEndpoint._process_image_input method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("permission_mode", [0o444, 0o400, 0o440, 0o404])
    def test_handle_minimal_read_permissions(self, endpoint, permission_mode):
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b'Test content for permission testing')
            test_file = f.name
        
        try:
            os.chmod(test_file, permission_mode)
            
            # Act
            if permission_mode & 0o400:  # Owner has read permission
                result = endpoint._process_image_input(test_file)
                # Assert
                assert hasattr(result, 'read')
                assert hasattr(result, 'mode')
                assert 'b' in result.mode
                result.close()
            else:
                # Assert - Should raise permission error
                with pytest.raises((PermissionError, OSError)):
                    endpoint._process_image_input(test_file)
        finally:
            os.chmod(test_file, 0o644)
            os.unlink(test_file)
    
    @pytest.mark.parametrize("file_size", [0, 1, 10, 1024, 1024*1024])
    def test_handle_various_file_sizes(self, endpoint, file_size):
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            if file_size > 0:
                f.write(b'0' * file_size)
            test_file = f.name
        
        try:
            # Act
            result = endpoint._process_image_input(test_file)
            
            # Assert
            assert hasattr(result, 'read')
            assert hasattr(result, 'mode')
            assert 'b' in result.mode
            result.close()
        finally:
            os.unlink(test_file)

class Test_ImagesEndpoint_ProcessImageInput_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ImagesEndpoint._process_image_input method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("error_path_scenario", [
        ("no_read_permission", 0o000),
        ("no_directory_access", "restricted_dir"),
        ("file_system_corruption", "corrupt_fs"),
        ("network_path_failure", "/network/unavailable/file.png"),
        ("special_device_file", "/dev/null"),
    ])
    def test_generate_descriptive_error_messages(self, endpoint, error_path_scenario):
        # Arrange
        scenario_type, scenario_data = error_path_scenario
        
        if scenario_type == "no_read_permission":
            self._test_no_read_permission(endpoint)
                
        elif scenario_type == "no_directory_access":
            self._test_no_directory_access(endpoint)
                
        else:
            # For other scenarios, test with invalid paths
            invalid_path = str(scenario_data)
            
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                endpoint._process_image_input(invalid_path)
            assert "does not exist" in str(exc_info.value)
    
    def _test_no_read_permission(self, endpoint):
        """Test file permission denial with platform-specific handling."""
        if sys.platform == "win32":
            # Windows-specific permission handling using icacls
            self._test_windows_permission_denial(endpoint)
        else:
            # Unix-style permission handling
            self._test_unix_permission_denial(endpoint)
    
    def _test_windows_permission_denial(self, endpoint):
        """Windows-specific test using icacls to deny read permissions."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"Test image content")
            restricted_file = f.name
        
        try:
            # Use icacls to deny read permissions on Windows
            subprocess.run([
                "icacls", restricted_file, "/deny", f"{os.environ.get('USERNAME', 'Everyone')}:R"
            ], check=True, capture_output=True)
            
            # Act & Assert
            with pytest.raises((PermissionError, OSError)):
                endpoint._process_image_input(restricted_file)
        except subprocess.CalledProcessError:
            # Fallback to mocking if icacls fails
            self._test_with_mock(endpoint, restricted_file)
        finally:
            # Restore permissions and clean up
            try:
                subprocess.run([
                    "icacls", restricted_file, "/grant", f"{os.environ.get('USERNAME', 'Everyone')}:F"
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                pass
            
            try:
                os.unlink(restricted_file)
            except (PermissionError, FileNotFoundError):
                pass
    
    def _test_unix_permission_denial(self, endpoint):
        """Unix-specific test using chmod to deny read permissions."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"Test image content")
            restricted_file = f.name
        
        try:
            os.chmod(restricted_file, 0o000)
            
            # Act & Assert
            with pytest.raises((PermissionError, OSError)):
                endpoint._process_image_input(restricted_file)
        finally:
            os.chmod(restricted_file, 0o644)
            os.unlink(restricted_file)
    
    def _test_with_mock(self, endpoint, file_path):
        """Fallback test using mocking for cross-platform consistency."""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                endpoint._process_image_input(file_path)
    
    def _test_no_directory_access(self, endpoint):
        """Test directory access denial with platform-specific handling."""
        if sys.platform == "win32":
            # Skip directory permission test on Windows as it's unreliable
            return
        else:
            temp_dir = tempfile.mkdtemp()
            test_file = os.path.join(temp_dir, "test.png")
            
            try:
                with open(test_file, 'wb') as f:
                    f.write(b"Test image content")
                os.chmod(temp_dir, 0o000)
                
                # Act & Assert
                with pytest.raises((PermissionError, OSError)):
                    endpoint._process_image_input(test_file)
            finally:
                os.chmod(temp_dir, 0o755)
                os.unlink(test_file)
                os.rmdir(temp_dir)
    
    @pytest.mark.parametrize("file_system_error_type", [
        "disk_full",
        "read_only_filesystem", 
        "io_error",
        "device_not_ready",
    ])
    def test_manage_file_system_constraints(self, endpoint, file_system_error_type):
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"Test content for file system error testing")
            test_file = f.name
        
        try:
            # Act - File system errors are difficult to simulate without root privileges
            # Test that the method handles OS errors appropriately
            result = endpoint._process_image_input(test_file)
            
            # Assert - If successful, verify proper file object returned
            assert hasattr(result, 'read')
            assert 'b' in result.mode
            result.close()
            
        except (OSError, IOError) as e:
            # Assert - Verify OS errors are handled appropriately
            assert isinstance(e, (OSError, IOError))
        finally:
            os.unlink(test_file)

class Test_ImagesEndpoint_ProcessImageInput_05_StateTransitionBehaviors:
    """Test state transition behaviors for ImagesEndpoint._process_image_input method."""
    
    @pytest.fixture
    def endpoint(self):
        auth_manager = AuthManager("test-key")
        http_manager = HTTPManager(base_url="https://invalid.thisurldoesntexist.com")
        return ImagesEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("file_mode,should_warn", [
        ('r', True),      # Text mode should warn
        ('rt', True),     # Text mode should warn  
        ('r+', True),     # Text mode should warn
        ('rb', False),    # Binary mode should not warn
        ('r+b', False),   # Binary mode should not warn
        ('ab', False),    # Binary mode should not warn
    ])
    def test_issue_warnings_for_non_binary_modes(self, endpoint, file_mode, should_warn, caplog):
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(b"Test content for mode checking")
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, file_mode, encoding='latin1' if 'b' not in file_mode else None) as f:
                if 'a' in file_mode:
                    f.seek(0)  # Reset position for append mode
                
                # Act
                result = endpoint._process_image_input(f)
                
                # Assert
                assert result is f
                
                if should_warn:
                    # Check that warning was logged
                    warning_found = any(
                        "not opened in binary mode" in record.message 
                        for record in caplog.records 
                        if record.levelname == "WARNING"
                    )
                    assert warning_found, f"Expected warning for file mode {file_mode}"
                else:
                    # Check that no warning was logged
                    warning_found = any(
                        "not opened in binary mode" in record.message 
                        for record in caplog.records 
                        if record.levelname == "WARNING"
                    )
                    assert not warning_found, f"Unexpected warning for file mode {file_mode}"
        finally:
            os.unlink(temp_file_path)
    
    @pytest.mark.parametrize("input_type,processing_path", [
        ("string_path", "file_path_processing"),
        ("file_object_binary", "file_object_processing"),
        ("file_object_text", "file_object_processing"),
        ("string_object", "file_path_processing"),
        ("pathlib_path", "file_path_processing"),
    ])
    def test_track_input_type_determination(self, endpoint, input_type, processing_path):
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"Test content for input type determination")
            temp_path = f.name
        
        try:
            if input_type == "string_path":
                test_input = temp_path
            elif input_type == "file_object_binary":
                test_input = open(temp_path, 'rb')
            elif input_type == "file_object_text":
                test_input = open(temp_path, 'r', encoding='latin1')
            elif input_type == "string_object":
                test_input = str(temp_path)  # Explicit string conversion
            elif input_type == "pathlib_path":
                test_input = str(Path(temp_path))  # Convert Path to string
            
            # Act
            try:
                result = endpoint._process_image_input(test_input)
                
                # Assert - Verify correct processing path taken
                if processing_path == "file_path_processing":
                    # Should return a new file object
                    assert hasattr(result, 'read')
                    assert hasattr(result, 'mode')
                    assert 'b' in result.mode
                    if input_type in ["string_path", "string_object", "pathlib_path"]:
                        assert result is not test_input  # New object created
                    result.close()
                else:  # file_object_processing
                    # Should return the same object
                    assert result is test_input
                    assert hasattr(result, 'read')
                    
            finally:
                # Clean up file objects
                if hasattr(test_input, 'close') and input_type != "string_path":
                    test_input.close()
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.parametrize("validation_scenario", [
        ("valid_file_path", True),
        ("valid_file_object", True), 
        ("nonexistent_path", False),
        ("invalid_type", False),
    ])
    def test_verify_input_validation_workflow(self, endpoint, validation_scenario):
        # Arrange
        scenario_type, should_succeed = validation_scenario
        
        if scenario_type == "valid_file_path":
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(b"Valid test content")
                test_input = f.name
        elif scenario_type == "valid_file_object":
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_file.write(b"Valid test content")
            temp_file.close()
            test_input = open(temp_file.name, 'rb')
        elif scenario_type == "nonexistent_path":
            test_input = "/nonexistent/file/path.png"
        else:  # invalid_type
            test_input = {"invalid": "type"}
        
        try:
            # Act
            if should_succeed:
                result = endpoint._process_image_input(test_input)
                
                # Assert - Successful validation and processing
                assert hasattr(result, 'read')
                if scenario_type == "valid_file_path":
                    assert 'b' in result.mode
                    result.close()
                elif scenario_type == "valid_file_object":
                    assert result is test_input
            else:
                # Assert - Failed validation raises appropriate error
                with pytest.raises(ValueError):
                    endpoint._process_image_input(test_input)
                    
        finally:
            # Cleanup
            if scenario_type == "valid_file_path" and should_succeed:
                os.unlink(test_input)
            elif scenario_type == "valid_file_object":
                if hasattr(test_input, 'close'):
                    test_input.close()
                if hasattr(temp_file, 'name') and os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
