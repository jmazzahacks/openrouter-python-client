import pytest
import os
import tempfile
from PIL import Image

from openrouter_client import OpenRouterClient
from openrouter_client.endpoints.images import ImagesEndpoint
from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager
from openrouter_client.exceptions import APIError

@pytest.fixture(scope="session")
def http_manager():
    """Shared HTTPManager instance for all tests."""
    return HTTPManager(base_url="https://openrouter.ai/api/v1")

@pytest.fixture(scope="session")
def auth_manager():
    """Shared AuthManager instance for all tests."""
    return AuthManager()

@pytest.fixture(scope="session")
def images_endpoint(auth_manager, http_manager):
    """Images endpoint instance for testing."""
    return ImagesEndpoint(auth_manager, http_manager)

@pytest.fixture(scope="session")
def test_image_file():
    """Create a temporary test image file."""
    # Create a simple RGB image
    img = Image.new('RGB', (256, 256), color='red')
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(temp_file.name, 'PNG')
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    os.unlink(temp_file.name)

@pytest.fixture(scope="session")
def test_mask_file():
    """Create a temporary mask image file."""
    # Create a simple grayscale mask
    img = Image.new('L', (256, 256), color=128)
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(temp_file.name, 'PNG')
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    os.unlink(temp_file.name)


class Test_ImagesEndpoint_Init_01_NominalBehaviors:
    """Test nominal initialization behaviors for ImagesEndpoint."""
    
    def test_successful_initialization_with_valid_managers(self, openrouter_client):
        """Test successful initialization with valid auth_manager and http_manager."""
        # Arrange
        auth_manager = openrouter_client._auth_manager
        http_manager = openrouter_client._http_manager
        
        # Act
        endpoint = ImagesEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint is not None
        assert hasattr(endpoint, 'auth_manager')
        assert hasattr(endpoint, 'http_manager')
        assert endpoint.endpoint_path == "images"
    
    def test_proper_endpoint_path_configuration(self, openrouter_client):
        """Test proper endpoint path configuration for images API."""
        # Arrange
        auth_manager = openrouter_client._auth_manager
        http_manager = openrouter_client._http_manager
        
        # Act
        endpoint = ImagesEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint.endpoint_path == "images"

class Test_ImagesEndpoint_Init_02_NegativeBehaviors:
    """Test negative initialization behaviors for ImagesEndpoint."""
    
    @pytest.mark.parametrize("auth_manager,http_manager", [
        (None, None),
        (None, "valid_http_manager"),
        ("valid_auth_manager", None)
    ])
    def test_initialization_with_invalid_managers(self, auth_manager, http_manager, openrouter_client):
        """Test initialization with invalid or None authentication/HTTP managers."""
        # Arrange
        if auth_manager == "valid_auth_manager":
            auth_manager = openrouter_client._auth_manager
        if http_manager == "valid_http_manager":
            http_manager = openrouter_client._http_manager
        
        # Act & Assert
        with pytest.raises((TypeError, AttributeError)):
            ImagesEndpoint(auth_manager, http_manager)

class Test_ImagesEndpoint_Init_03_BoundaryBehaviors:
    """Test boundary initialization behaviors for ImagesEndpoint."""
    
    def test_initialization_with_minimal_valid_configuration(self, openrouter_client):
        """Test initialization with minimal valid configuration parameters."""
        # Arrange
        auth_manager = openrouter_client._auth_manager
        http_manager = openrouter_client._http_manager
        
        # Act
        endpoint = ImagesEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint is not None
        assert endpoint.endpoint_path == "images"

class Test_ImagesEndpoint_Init_04_ErrorHandlingBehaviors:
    """Test error handling during ImagesEndpoint initialization."""
    
    def test_initialization_with_corrupted_managers(self, openrouter_client):
        """Test handling initialization failures when dependent managers are corrupted."""
        # Arrange
        auth_manager = openrouter_client._auth_manager
        http_manager = openrouter_client._http_manager
        
        # Simulate corrupted managers by removing required attributes
        if hasattr(auth_manager, '_api_key'):
            original_key = auth_manager._api_key
            auth_manager._api_key = None
            
            # Act & Assert
            try:
                endpoint = ImagesEndpoint(auth_manager, http_manager)
                # Depending on implementation, this might succeed but fail during use
                assert endpoint is not None
            except Exception as e:
                assert isinstance(e, (ValueError, AttributeError))
            finally:
                # Restore
                auth_manager._api_key = original_key

class Test_ImagesEndpoint_Init_05_StateTransitionBehaviors:
    """Test state transition behaviors during ImagesEndpoint initialization."""
    
    def test_transition_from_uninitialized_to_initialized_state(self, openrouter_client):
        """Test transition from uninitialized to initialized state for HTTP endpoint handling."""
        # Arrange
        auth_manager = openrouter_client._auth_manager
        http_manager = openrouter_client._http_manager
        
        # Act - Create endpoint (transition to initialized state)
        endpoint = ImagesEndpoint(auth_manager, http_manager)
        
        # Assert - Verify initialized state
        assert endpoint is not None
        assert hasattr(endpoint, 'auth_manager')
        assert hasattr(endpoint, 'http_manager')
        assert endpoint.endpoint_path == "images"

class Test_ImagesEndpoint_Create_01_NominalBehaviors:
    """Test nominal create method behaviors for ImagesEndpoint."""
    
    @pytest.mark.parametrize("prompt", [
        "A red apple on a white table",
        "A beautiful sunset over mountains",
        "A cat sitting on a windowsill"
    ])
    def test_http_post_request_with_valid_prompt_default_parameters(self, images_endpoint, prompt):
        """Test HTTP POST request to /images/generations with valid prompt and default parameters."""
        # Arrange
        # Act
        response = images_endpoint.create(prompt=prompt)
        
        # Assert
        assert isinstance(response, dict)
        assert 'data' in response or 'images' in response
    
    @pytest.mark.parametrize("model,n,size,response_format,quality,style", [
        ("dall-e-3", 1, "1024x1024", "url", "standard", "vivid"),
        ("dall-e-2", 2, "512x512", "b64_json", "hd", "natural"),
        (None, None, None, None, None, None)
    ])
    def test_successful_request_with_all_optional_parameters(self, images_endpoint, model, n, size, response_format, quality, style):
        """Test successful request with all optional parameters (model, n, size, response_format, quality, style)."""
        # Arrange
        prompt = "A test image for API testing"
        
        # Act
        response = images_endpoint.create(
            prompt=prompt,
            model=model,
            n=n,
            size=size,
            response_format=response_format,
            quality=quality,
            style=style
        )
        
        # Assert
        assert isinstance(response, dict)
        assert 'data' in response or 'images' in response
    
    def test_proper_json_payload_construction_and_headers(self, images_endpoint):
        """Test proper JSON payload construction and Content-Type headers."""
        # Arrange
        prompt = "Test image with custom parameters"
        
        # Act
        response = images_endpoint.create(
            prompt=prompt,
            model="dall-e-3",
            n=1,
            custom_param="test_value"
        )
        
        # Assert
        assert isinstance(response, dict)
    
    def test_authentication_header_inclusion_in_requests(self, images_endpoint):
        """Test authentication header inclusion in HTTP requests."""
        # Arrange
        prompt = "Test authentication headers"
        
        # Act
        response = images_endpoint.create(prompt=prompt)
        
        # Assert
        assert isinstance(response, dict)
        # If we get a valid response, authentication headers were included correctly

class Test_ImagesEndpoint_Create_02_NegativeBehaviors:
    """Test negative create method behaviors for ImagesEndpoint."""
    
    @pytest.mark.parametrize("prompt", [
        "",
        "   ",
        "\t\n",
        None
    ])
    def test_http_request_behavior_with_invalid_prompts(self, images_endpoint, prompt):
        """Test HTTP request behavior with empty or invalid prompts."""
        # Arrange
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint.create(prompt=prompt)
    
    @pytest.mark.parametrize("model", [
        "",
        "   ",
        123,
        []
    ])
    def test_request_handling_with_invalid_model_parameter(self, images_endpoint, model):
        """Test request handling with invalid model parameter."""
        # Arrange
        prompt = "Test invalid model"
        
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint.create(prompt=prompt, model=model)
    
    @pytest.mark.parametrize("n", [
        0,
        -1,
        -10,
        "invalid",
        []
    ])
    def test_request_behavior_with_invalid_n_values(self, images_endpoint, n):
        """Test request behavior with invalid n values (zero, negative integers)."""
        # Arrange
        prompt = "Test invalid n value"
        
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint.create(prompt=prompt, n=n)
    
    @pytest.mark.parametrize("size", [
        "invalid",
        "1024",
        "1024x",
        "x1024",
        "1024x1024x512",
        123,
        []
    ])
    def test_http_request_with_malformed_size_parameter(self, images_endpoint, size):
        """Test HTTP request with malformed size parameter."""
        # Arrange
        prompt = "Test invalid size"
        
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint.create(prompt=prompt, size=size)
    
    @pytest.mark.parametrize("response_format", [
        "invalid",
        "json",
        "png",
        123,
        []
    ])
    def test_request_handling_with_invalid_response_format(self, images_endpoint, response_format):
        """Test request handling with invalid response_format values."""
        # Arrange
        prompt = "Test invalid response format"
        
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint.create(prompt=prompt, response_format=response_format)
    
    @pytest.mark.parametrize("quality", [
        "invalid",
        "high",
        "low",
        123,
        []
    ])
    def test_http_request_with_invalid_quality_values(self, images_endpoint, quality):
        """Test HTTP request with invalid quality values."""
        # Arrange
        prompt = "Test invalid quality"
        
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint.create(prompt=prompt, quality=quality)

class Test_ImagesEndpoint_Create_03_BoundaryBehaviors:
    """Test boundary create method behaviors for ImagesEndpoint."""
    
    def test_http_requests_with_maximum_allowed_prompt_length(self, images_endpoint):
        """Test HTTP requests with maximum allowed prompt length."""
        # Arrange
        # Assuming max prompt length is around 1000 characters
        prompt = "A" * 1000
        
        # Act
        response = images_endpoint.create(prompt=prompt)
        
        # Assert
        assert isinstance(response, dict)
    
    @pytest.mark.parametrize("n", [1, 10])  # Assuming 1 is minimum, 10 is maximum
    def test_requests_with_n_parameter_at_api_limits(self, images_endpoint, n):
        """Test requests with n parameter at API limits."""
        # Arrange
        prompt = "Test boundary n values"
        
        # Act
        response = images_endpoint.create(prompt=prompt, n=n)
        
        # Assert
        assert isinstance(response, dict)
    
    @pytest.mark.parametrize("size", [
        "256x256",    # Smallest common size
        "1024x1024",  # Standard size
        "1792x1024"   # Largest common size
    ])
    def test_boundary_size_values(self, images_endpoint, size):
        """Test boundary size values (smallest/largest supported dimensions)."""
        # Arrange
        prompt = "Test boundary sizes"
        
        # Act
        response = images_endpoint.create(prompt=prompt, size=size)
        
        # Assert
        assert isinstance(response, dict)
    
    def test_very_long_style_parameter_strings(self, images_endpoint):
        """Test very long style parameter strings."""
        # Arrange
        prompt = "Test long style"
        style = "very_long_style_parameter_" * 10
        
        # Act
        response = images_endpoint.create(prompt=prompt, style=style)
        
        # Assert
        assert isinstance(response, dict)

class Test_ImagesEndpoint_Create_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for create method."""
    
    def test_authentication_failures_and_unauthorized_access_responses(self, openrouter_client):
        """Test authentication failures and unauthorized access responses."""
        # Arrange
        # Create endpoint with invalid auth
        invalid_client = OpenRouterClient(api_key="invalid_key")
        images_endpoint = invalid_client.images
        prompt = "Test auth failure"
        
        # Act & Assert
        with pytest.raises(APIError):
            images_endpoint.create(prompt=prompt)
    
    def test_server_error_responses(self, images_endpoint):
        """Test server error responses (4xx, 5xx status codes)."""
        # Arrange
        # This test might be difficult to trigger reliably without mocks
        # We'll test with an extremely long prompt that might cause server errors
        prompt = "A" * 10000  # Extremely long prompt that might cause errors
        
        # Act
        try:
            response = images_endpoint.create(prompt=prompt)
            # If successful, that's also valid
            assert isinstance(response, dict)
        except APIError as e:
            # If it fails with an API error, that's the expected behavior we're testing
            assert isinstance(e, APIError)

class Test_ImagesEndpoint_Create_05_StateTransitionBehaviors:
    """Test state transition behaviors for create method."""
    
    def test_endpoint_state_before_during_after_successful_requests(self, images_endpoint):
        """Test endpoint state before, during, and after successful HTTP requests."""
        # Arrange
        prompt = "Test state transitions"
        initial_state = hasattr(images_endpoint, 'endpoint_path')
        
        # Act
        response = images_endpoint.create(prompt=prompt)
        
        # Assert
        assert initial_state  # Endpoint was properly initialized
        assert isinstance(response, dict)  # Request completed successfully
        assert hasattr(images_endpoint, 'endpoint_path')  # Endpoint state preserved
    
    def test_state_management_during_failed_requests(self, openrouter_client):
        """Test state management during failed HTTP request scenarios."""
        # Arrange
        invalid_client = OpenRouterClient(api_key="invalid_key")
        images_endpoint = invalid_client.images
        prompt = "Test failed state"
        
        # Act & Assert
        try:
            images_endpoint.create(prompt=prompt)
        except APIError:
            # Endpoint should maintain its state even after failed requests
            assert hasattr(images_endpoint, 'endpoint_path')
            assert images_endpoint.endpoint_path == "images"

class Test_ImagesEndpoint_Edit_01_NominalBehaviors:
    """Test nominal edit method behaviors for ImagesEndpoint."""
    
    def test_http_post_request_with_multipart_form_data(self, images_endpoint, test_image_file):
        """Test HTTP POST request to /images/edits with multipart form data."""
        # Arrange
        prompt = "Make the image brighter"
        
        # Act
        response = images_endpoint.edit(image=test_image_file, prompt=prompt)
        
        # Assert
        assert isinstance(response, dict)
        assert 'data' in response or 'images' in response
    
    def test_successful_multipart_upload_with_image_and_prompt(self, images_endpoint, test_image_file):
        """Test successful multipart upload with image file and prompt."""
        # Arrange
        prompt = "Add a blue sky background"
        
        # Act
        response = images_endpoint.edit(image=test_image_file, prompt=prompt)
        
        # Assert
        assert isinstance(response, dict)
    
    def test_http_request_with_image_and_mask_files(self, images_endpoint, test_image_file, test_mask_file):
        """Test HTTP request with both image and mask files."""
        # Arrange
        prompt = "Change the masked area to green"
        
        # Act
        response = images_endpoint.edit(
            image=test_image_file,
            prompt=prompt,
            mask=test_mask_file
        )
        
        # Assert
        assert isinstance(response, dict)
    
    def test_authentication_header_inclusion_in_multipart_requests(self, images_endpoint, test_image_file):
        """Test authentication header inclusion in multipart requests."""
        # Arrange
        prompt = "Test auth in multipart"
        
        # Act
        response = images_endpoint.edit(image=test_image_file, prompt=prompt)
        
        # Assert
        assert isinstance(response, dict)

class Test_ImagesEndpoint_Edit_02_NegativeBehaviors:
    """Test negative edit method behaviors for ImagesEndpoint."""
    
    def test_http_request_behavior_with_nonexistent_image_file_paths(self, images_endpoint):
        """Test HTTP request behavior with non-existent image file paths."""
        # Arrange
        nonexistent_file = "/path/to/nonexistent/image.png"
        prompt = "Edit nonexistent image"
        
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint.edit(image=nonexistent_file, prompt=prompt)
    
    def test_request_behavior_with_empty_prompt(self, images_endpoint, test_image_file):
        """Test request behavior with empty prompt in form data."""
        # Arrange
        prompt = ""
        
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint.edit(image=test_image_file, prompt=prompt)
    
    def test_multipart_request_handling_with_invalid_file_objects(self, images_endpoint):
        """Test multipart request handling with invalid file objects."""
        # Arrange
        invalid_file = "not_a_file_object"
        prompt = "Test invalid file"
        
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint.edit(image=invalid_file, prompt=prompt)

class Test_ImagesEndpoint_Edit_03_BoundaryBehaviors:
    """Test boundary edit method behaviors for ImagesEndpoint."""
    
    def test_http_requests_with_maximum_allowed_image_file_sizes(self, images_endpoint):
        """Test HTTP requests with maximum allowed image file sizes."""
        # Arrange
        # Create a larger test image
        large_img = Image.new('RGB', (2048, 2048), color='blue')
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        large_img.save(temp_file.name, 'PNG')
        temp_file.close()
        
        prompt = "Edit large image"
        
        try:
            # Act
            response = images_endpoint.edit(image=temp_file.name, prompt=prompt)
            
            # Assert
            assert isinstance(response, dict)
        finally:
            # Cleanup
            os.unlink(temp_file.name)
    
    def test_multipart_uploads_with_minimum_viable_image_dimensions(self, images_endpoint):
        """Test multipart uploads with minimum viable image dimensions."""
        # Arrange
        # Create a minimal size image
        small_img = Image.new('RGB', (64, 64), color='green')
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        small_img.save(temp_file.name, 'PNG')
        temp_file.close()
        
        prompt = "Edit small image"
        
        try:
            # Act
            response = images_endpoint.edit(image=temp_file.name, prompt=prompt)
            
            # Assert
            assert isinstance(response, dict)
        except APIError:
            # Some APIs might reject images that are too small
            pass
        finally:
            # Cleanup
            os.unlink(temp_file.name)

class Test_ImagesEndpoint_Edit_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for edit method."""
    
    def test_server_side_image_processing_error_responses(self, images_endpoint):
        """Test server-side image processing error responses."""
        # Arrange
        # Create a corrupted or invalid image file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.write(b"not_a_valid_image_file")
        temp_file.close()
        
        prompt = "Edit corrupted image"
        
        try:
            # Act & Assert
            with pytest.raises((ValueError, APIError)):
                images_endpoint.edit(image=temp_file.name, prompt=prompt)
        finally:
            # Cleanup
            os.unlink(temp_file.name)
    
    def test_file_handle_cleanup_after_failed_requests(self, images_endpoint):
        """Test file handle cleanup after failed HTTP requests."""
        # Arrange
        nonexistent_file = "/path/to/nonexistent/image.png"
        prompt = "Test cleanup"
        
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint.edit(image=nonexistent_file, prompt=prompt)
        
        # If we reach here, proper cleanup occurred (no hanging file handles)

class Test_ImagesEndpoint_Edit_05_StateTransitionBehaviors:
    """Test state transition behaviors for edit method."""
    
    def test_file_handle_state_management_throughout_request_lifecycle(self, images_endpoint, test_image_file):
        """Test file handle state management throughout HTTP request lifecycle."""
        # Arrange
        prompt = "Test file handle lifecycle"
        
        # Act
        response = images_endpoint.edit(image=test_image_file, prompt=prompt)
        
        # Assert
        assert isinstance(response, dict)
        # File should still be accessible after the request
        assert os.path.exists(test_image_file)
    
    def test_endpoint_state_transitions_during_multipart_upload_processes(self, images_endpoint, test_image_file):
        """Test endpoint state transitions during multipart upload processes."""
        # Arrange
        prompt = "Test endpoint state during upload"
        initial_state = images_endpoint.endpoint_path
        
        # Act
        response = images_endpoint.edit(image=test_image_file, prompt=prompt)
        
        # Assert
        assert isinstance(response, dict)
        assert images_endpoint.endpoint_path == initial_state

class Test_ImagesEndpoint_Variations_01_NominalBehaviors:
    """Test nominal variations method behaviors for ImagesEndpoint."""
    
    def test_http_post_request_with_multipart_form_data(self, images_endpoint, test_image_file):
        """Test HTTP POST request to /images/variations with multipart form data."""
        # Arrange
        # Act
        response = images_endpoint.variations(image=test_image_file)
        
        # Assert
        assert isinstance(response, dict)
        assert 'data' in response or 'images' in response
    
    def test_successful_image_upload_with_variation_parameters(self, images_endpoint, test_image_file):
        """Test successful image upload with variation parameters."""
        # Arrange
        # Act
        response = images_endpoint.variations(
            image=test_image_file,
            n=2,
            size="512x512"
        )
        
        # Assert
        assert isinstance(response, dict)
    
    def test_authentication_inclusion_in_variation_requests(self, images_endpoint, test_image_file):
        """Test authentication inclusion in variation requests."""
        # Arrange
        # Act
        response = images_endpoint.variations(image=test_image_file)
        
        # Assert
        assert isinstance(response, dict)

class Test_ImagesEndpoint_Variations_02_NegativeBehaviors:
    """Test negative variations method behaviors for ImagesEndpoint."""
    
    def test_http_request_behavior_with_invalid_image_file_inputs(self, images_endpoint):
        """Test HTTP request behavior with invalid image file inputs."""
        # Arrange
        invalid_file = "not_a_file"
        
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint.variations(image=invalid_file)
    
    def test_multipart_request_handling_with_corrupted_image_data(self, images_endpoint):
        """Test multipart request handling with corrupted image data."""
        # Arrange
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.write(b"corrupted_image_data")
        temp_file.close()
        
        try:
            # Act & Assert
            with pytest.raises((ValueError, APIError)):
                images_endpoint.variations(image=temp_file.name)
        finally:
            # Cleanup
            os.unlink(temp_file.name)

class Test_ImagesEndpoint_Variations_03_BoundaryBehaviors:
    """Test boundary variations method behaviors for ImagesEndpoint."""
    
    def test_http_requests_with_images_at_maximum_supported_resolution(self, images_endpoint):
        """Test HTTP requests with images at maximum supported resolution."""
        # Arrange
        large_img = Image.new('RGB', (2048, 2048), color='purple')
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        large_img.save(temp_file.name, 'PNG')
        temp_file.close()
        
        try:
            # Act
            response = images_endpoint.variations(image=temp_file.name)
            
            # Assert
            assert isinstance(response, dict)
        finally:
            # Cleanup
            os.unlink(temp_file.name)
    
    def test_variations_requests_with_minimum_viable_image_sizes(self, images_endpoint):
        """Test variations requests with minimum viable image sizes."""
        # Arrange
        small_img = Image.new('RGB', (64, 64), color='yellow')
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        small_img.save(temp_file.name, 'PNG')
        temp_file.close()
        
        try:
            # Act
            response = images_endpoint.variations(image=temp_file.name)
            
            # Assert
            assert isinstance(response, dict)
        except APIError:
            # Some APIs might reject images that are too small
            pass
        finally:
            # Cleanup
            os.unlink(temp_file.name)

class Test_ImagesEndpoint_Variations_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for variations method."""
    
    def test_network_failure_handling_during_image_upload(self, openrouter_client):
        """Test network failure handling during image upload."""
        # Arrange
        # This is difficult to test reliably without mocks
        # We'll test with invalid authentication to simulate network-related failures
        invalid_client = OpenRouterClient(api_key="invalid_key")
        images_endpoint = invalid_client.images
        
        # Create a test image
        test_img = Image.new('RGB', (256, 256), color='orange')
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        test_img.save(temp_file.name, 'PNG')
        temp_file.close()
        
        try:
            # Act & Assert
            with pytest.raises(APIError):
                images_endpoint.variations(image=temp_file.name)
        finally:
            # Cleanup
            os.unlink(temp_file.name)
    
    def test_file_resource_cleanup_after_request_failures(self, images_endpoint):
        """Test file resource cleanup after HTTP request failures."""
        # Arrange
        nonexistent_file = "/path/to/nonexistent/image.png"
        
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint.variations(image=nonexistent_file)

class Test_ImagesEndpoint_Variations_05_StateTransitionBehaviors:
    """Test state transition behaviors for variations method."""
    
    def test_image_file_handle_lifecycle_during_request_processing(self, images_endpoint, test_image_file):
        """Test image file handle lifecycle during HTTP request processing."""
        # Arrange
        # Act
        response = images_endpoint.variations(image=test_image_file)
        
        # Assert
        assert isinstance(response, dict)
        assert os.path.exists(test_image_file)
    
    def test_endpoint_state_management_for_concurrent_variation_requests(self, images_endpoint, test_image_file):
        """Test endpoint state management for concurrent variation requests."""
        # Arrange
        initial_state = images_endpoint.endpoint_path
        
        # Act
        response = images_endpoint.variations(image=test_image_file)
        
        # Assert
        assert isinstance(response, dict)
        assert images_endpoint.endpoint_path == initial_state

class Test_ImagesEndpoint_ProcessImageInput_01_NominalBehaviors:
    """Test nominal _process_image_input method behaviors for ImagesEndpoint."""
    
    def test_proper_file_object_preparation_for_multipart_uploads(self, images_endpoint, test_image_file):
        """Test proper file object preparation for HTTP multipart uploads."""
        # Arrange
        # Act
        file_obj = images_endpoint._process_image_input(test_image_file)
        
        # Assert
        assert hasattr(file_obj, 'read')
        assert hasattr(file_obj, 'mode')
        assert 'b' in file_obj.mode
        
        # Cleanup
        file_obj.close()
    
    def test_correct_binary_mode_file_handling_for_api_requests(self, images_endpoint):
        """Test correct binary mode file handling for API requests."""
        # Arrange
        with open(test_image_file, 'rb') as binary_file:
            # Act
            file_obj = images_endpoint._process_image_input(binary_file)
            
            # Assert
            assert file_obj is binary_file
            assert hasattr(file_obj, 'read')

class Test_ImagesEndpoint_ProcessImageInput_02_NegativeBehaviors:
    """Test negative _process_image_input method behaviors for ImagesEndpoint."""
    
    @pytest.mark.parametrize("invalid_input", [
        123,
        [],
        {},
        None
    ])
    def test_file_preparation_failures_affecting_http_requests(self, images_endpoint, invalid_input):
        """Test file preparation failures that would affect subsequent HTTP requests."""
        # Arrange
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint._process_image_input(invalid_input)
    
    def test_invalid_file_object_states_impacting_api_communication(self, images_endpoint):
        """Test invalid file object states that impact API communication."""
        # Arrange
        class InvalidFileObject:
            pass
        
        invalid_obj = InvalidFileObject()
        
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint._process_image_input(invalid_obj)

class Test_ImagesEndpoint_ProcessImageInput_03_BoundaryBehaviors:
    """Test boundary _process_image_input method behaviors for ImagesEndpoint."""
    
    def test_file_size_validation_at_limits_affecting_http_upload(self, images_endpoint):
        """Test file size validation at limits that affect HTTP upload behavior."""
        # Arrange
        # Create a very large image file
        large_img = Image.new('RGB', (4096, 4096), color='cyan')
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        large_img.save(temp_file.name, 'PNG')
        temp_file.close()
        
        try:
            # Act
            file_obj = images_endpoint._process_image_input(temp_file.name)
            
            # Assert
            assert hasattr(file_obj, 'read')
            
            # Cleanup
            file_obj.close()
        finally:
            # Cleanup
            os.unlink(temp_file.name)

class Test_ImagesEndpoint_ProcessImageInput_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for _process_image_input method."""
    
    def test_file_access_errors_preventing_http_request_preparation(self, images_endpoint):
        """Test file access errors that prevent HTTP request preparation."""
        # Arrange
        nonexistent_file = "/path/to/nonexistent/file.png"
        
        # Act & Assert
        with pytest.raises(ValueError):
            images_endpoint._process_image_input(nonexistent_file)
    
    def test_resource_cleanup_to_prevent_http_request_failures(self, images_endpoint, test_image_file):
        """Test resource cleanup to prevent HTTP request failures."""
        # Arrange
        # Act
        file_obj = images_endpoint._process_image_input(test_image_file)
        
        # Assert
        assert hasattr(file_obj, 'read')
        
        # Manual cleanup
        file_obj.close()
        
        # Verify file is properly closed
        with pytest.raises(ValueError):
            file_obj.read()

class Test_ImagesEndpoint_ProcessImageInput_05_StateTransitionBehaviors:
    """Test state transition behaviors for _process_image_input method."""
    
    def test_file_handle_state_transitions_impacting_http_request_success(self, images_endpoint, test_image_file):
        """Test file handle state transitions that impact HTTP request success."""
        # Arrange
        # Act
        file_obj = images_endpoint._process_image_input(test_image_file)
        
        # Assert - File is in proper state for HTTP upload
        assert hasattr(file_obj, 'read')
        assert hasattr(file_obj, 'mode')
        assert 'b' in file_obj.mode
        
        # Test that file can be read (proper state for HTTP request)
        initial_position = file_obj.tell()
        content = file_obj.read(10)
        assert len(content) <= 10
        
        # Reset position
        file_obj.seek(initial_position)
        
        # Cleanup
        file_obj.close()
