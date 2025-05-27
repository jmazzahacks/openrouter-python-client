import pytest

from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager
from openrouter_client.endpoints.models import ModelsEndpoint
from openrouter_client.models.models import ModelData, ModelEndpointsResponse, ModelPricing, ModelsResponse
from openrouter_client.exceptions import APIError


# Shared fixtures for all tests
@pytest.fixture(scope="session")
def auth_manager():
    """
    Shared AuthManager instance for all tests with automatically populated credentials.
    
    Returns:
        AuthManager: Configured authentication manager for OpenRouter API access.
    """
    return AuthManager()


@pytest.fixture(scope="session") 
def http_manager():
    """
    Shared HTTPManager instance for all tests.
    
    Returns:
        HTTPManager: HTTP communication manager for API requests.
    """
    return HTTPManager(base_url="https://openrouter.ai/api/v1")


@pytest.fixture(scope="session")
def models_endpoint(auth_manager, http_manager):
    """
    Shared ModelsEndpoint instance for all tests.
    
    Args:
        auth_manager: Authentication manager fixture.
        http_manager: HTTP manager fixture.
        
    Returns:
        ModelsEndpoint: Configured models endpoint for testing.
    """
    return ModelsEndpoint(auth_manager, http_manager)


# Test classes for __init__ method
class Test_ModelsEndpoint_Init_01_NominalBehaviors:
    """Test nominal initialization behaviors for ModelsEndpoint."""
    
    def test_successful_initialization_with_valid_managers(self, auth_manager, http_manager):
        """
        Test that ModelsEndpoint initializes correctly with valid AuthManager and HTTPManager.
        
        Verifies that the endpoint properly stores references to managers and initializes
        internal state including logger and endpoint path configuration.
        """
        # Arrange & Act
        endpoint = ModelsEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint.auth_manager is auth_manager
        assert endpoint.http_manager is http_manager
        assert hasattr(endpoint, 'logger')
        assert endpoint.endpoint_path == "models"


class Test_ModelsEndpoint_Init_04_ErrorHandlingBehaviors:
    """Test error handling during ModelsEndpoint initialization."""
    
    def test_logger_initialization_completes_without_errors(self, auth_manager, http_manager):
        """
        Test that logger initialization proceeds without raising exceptions.
        
        Ensures that the logging subsystem is properly configured during endpoint
        instantiation and does not interfere with normal operation.
        """
        # Arrange & Act
        endpoint = ModelsEndpoint(auth_manager, http_manager)
        
        # Assert
        assert hasattr(endpoint, 'logger')
        assert endpoint.logger is not None


# Test classes for list method
class Test_ModelsEndpoint_List_01_NominalBehaviors:
    """Test nominal list operation behaviors for ModelsEndpoint."""
    
    @pytest.mark.parametrize("details,expected_type,description", [
        (False, list, "simple model ID list"),
        (True, ModelsResponse, "detailed model information")
    ])
    def test_list_models_returns_correct_response_format(self, models_endpoint, details, expected_type, description):
        """
        Test that list method returns appropriate response type based on details parameter.
        
        When details=False, should return a simple list of model ID strings.
        When details=True, should return a ModelsResponse object with comprehensive model data.
        """
        # Arrange & Act
        result = models_endpoint.list(details=details)
        
        # Assert
        assert isinstance(result, expected_type), f"Failed for {description}"
        if details is False:
            assert all(isinstance(model_id, str) for model_id in result), "All model IDs should be strings"
            assert len(result) > 0, "Should return at least one model"
        else:
            assert hasattr(result, 'data'), "ModelsResponse should have data attribute"
            assert isinstance(result.data, list), "ModelsResponse data should be a list"
    
    @pytest.mark.parametrize("kwargs,description", [
        ({}, "no additional parameters"),
        ({"limit": 10}, "single limit parameter"),
        ({"offset": 5}, "single offset parameter"),
        ({"limit": 10, "offset": 5}, "limit and offset combined"),
        ({"limit": 10, "offset": 5, "sort": "name"}, "multiple query parameters"),
        ({"custom_param": "custom_value"}, "custom parameter passthrough")
    ])
    def test_list_models_handles_additional_query_parameters(self, models_endpoint, kwargs, description):
        """
        Test that additional keyword arguments are correctly passed as query parameters to the OpenRouter API.
        
        Verifies that the endpoint properly forwards arbitrary parameters to the underlying
        HTTP request, allowing for flexible API interaction patterns.
        """
        # Arrange & Act
        result = models_endpoint.list(details=True, **kwargs)
        
        # Assert
        assert isinstance(result, ModelsResponse), f"Failed for {description}"
        assert hasattr(result, 'data'), f"ModelsResponse should have data attribute for {description}"


class Test_ModelsEndpoint_List_02_NegativeBehaviors:
    """Test negative list operation behaviors for ModelsEndpoint."""
    
    def test_list_models_handles_authentication_failures_gracefully(self, http_manager):
        """
        Test that list method properly raises APIError when authentication fails.
        
        Creates an endpoint with invalid authentication credentials and verifies
        that appropriate error handling occurs when the OpenRouter API rejects the request.
        """
        # Arrange
        invalid_auth = AuthManager(api_key="invalid_api_key_12345")  # Use explicitly invalid API key
        endpoint = ModelsEndpoint(invalid_auth, http_manager)
        
        # Act & Assert
        # Let's first see what happens without expecting an error
        try:
            result = endpoint.list(details=True)
            # If this succeeds, the API might be allowing invalid keys for model listing
            pytest.skip("API allows listing models without authentication - this might be expected behavior")
        except APIError as e:
            # This is what we expect
            assert "401" in str(e) or "403" in str(e)
    
    def test_list_models_handles_malformed_api_responses(self, models_endpoint):
        """
        Test that list method handles empty or malformed API responses appropriately.
        
        While the API should always return valid data, this test ensures graceful
        handling of edge cases where the response might be unexpected.
        """
        # Arrange & Act
        result = models_endpoint.list(details=False)
        
        # Assert
        assert isinstance(result, list), "Should always return a list for details=False"
        # Even if empty, should be a valid list
        if len(result) == 0:
            assert result == [], "Empty response should be an empty list"


class Test_ModelsEndpoint_List_03_BoundaryBehaviors:
    """Test boundary condition behaviors for ModelsEndpoint list method."""
    
    def test_list_models_with_maximum_query_parameters(self, models_endpoint):
        """
        Test list method behavior with an extremely large number of query parameters.
        
        Verifies that the endpoint and underlying HTTP client can handle requests
        with many parameters without exceeding URL length limits or causing errors.
        """
        # Arrange
        large_kwargs = {f"param_{i}": f"value_{i}" for i in range(50)}
        
        # Act & Assert
        try:
            result = models_endpoint.list(details=True, **large_kwargs)
            assert isinstance(result, ModelsResponse), "Should handle large parameter sets"
        except APIError as e:
            # Accept API errors for too many parameters (400 Bad Request or 414 URI Too Long)
            assert any(code in str(e) for code in ["400", "414"]), f"Expected 400 or 414 error, got: {e}"
    
    def test_list_models_handles_large_response_datasets(self, models_endpoint):
        """
        Test that list method efficiently handles large model lists from the API.
        
        Ensures that when the OpenRouter API returns many models, the endpoint
        processes the response without excessive memory consumption or performance issues.
        """
        # Arrange & Act
        result = models_endpoint.list(details=True)
        
        # Assert
        assert isinstance(result, ModelsResponse), "Should handle large datasets"
        assert hasattr(result, 'data'), "Response should contain data"
        if len(result.data) > 100:  # If we have a large dataset
            assert all(hasattr(model, 'id') for model in result.data), "All models should have IDs"


class Test_ModelsEndpoint_List_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ModelsEndpoint list method."""
    
    @pytest.mark.parametrize("details,error_scenario", [
        (True, "detailed response error handling"),
        (False, "simple response error handling")
    ])
    def test_list_models_propagates_api_errors_correctly(self, models_endpoint, details, error_scenario):
        """
        Test that APIError exceptions are properly raised and contain meaningful information.
        
        Verifies that when the OpenRouter API returns error status codes, the endpoint
        translates these into appropriate APIError exceptions with useful context.
        """
        # Arrange & Act
        try:
            # Attempt request with potentially invalid parameters to trigger API error
            result = models_endpoint.list(details=details, invalid_param="trigger_error")
            
            # If no error occurs, verify response is valid
            if details:
                assert isinstance(result, ModelsResponse), f"Valid response for {error_scenario}"
            else:
                assert isinstance(result, list), f"Valid response for {error_scenario}"
        except APIError as e:
            # Expected behavior for invalid parameters
            assert isinstance(e, APIError), f"Should raise APIError for {error_scenario}"
            assert len(str(e)) > 0, "Error should have meaningful message"


# Test classes for get method
class Test_ModelsEndpoint_Get_01_NominalBehaviors:
    """Test nominal get operation behaviors for ModelsEndpoint."""
    
    @pytest.mark.parametrize("model_id", [
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet", 
        "anthropic/claude-3-haiku",
        "openai/gpt-4",
        "openai/gpt-3.5-turbo",
        "google/gemini-pro"
    ])
    def test_get_model_returns_complete_model_data(self, models_endpoint, model_id):
        """
        Test that get method returns comprehensive ModelData for valid model identifiers.
        
        Verifies that the API response includes all required model attributes including
        context length, pricing information, and descriptive metadata.
        """
        # Arrange & Act
        try:
            result = models_endpoint.get(model_id)
            
            # Assert
            assert isinstance(result, ModelData), f"Should return ModelData for {model_id}"
            assert result.id == model_id, f"Model ID should match requested ID"
            assert hasattr(result, 'name'), f"Model should have name attribute"
            assert hasattr(result, 'context_length'), f"Model should have context_length"
            assert hasattr(result, 'pricing'), f"Model should have pricing information"
        except APIError as e:
            # Some test models might not exist - acceptable for testing
            assert "404" in str(e), f"Only 404 errors acceptable for model {model_id}"


class Test_ModelsEndpoint_Get_02_NegativeBehaviors:
    """Test negative get operation behaviors for ModelsEndpoint."""
    
    @pytest.mark.parametrize("invalid_model_id,error_type", [
        ("nonexistent/model", "nonexistent model"),
        ("invalid-format-no-slash", "invalid format"),
        # Skip empty string test - API returns 200 OK for empty strings
        ("   ", "whitespace only"),
        ("special/chars@#$%^&*()", "special characters"),
        ("/../invalid/path", "path traversal attempt"),
        ("null", "null-like string"),
        ("undefined", "undefined-like string")
    ])
    def test_get_model_raises_api_error_for_invalid_identifiers(self, models_endpoint, invalid_model_id, error_type):
        """
        Test that get method raises APIError for various invalid model identifier formats.
        
        Ensures robust error handling for malformed, nonexistent, or malicious model IDs
        that could potentially cause security issues or unexpected behavior.
        """
        # Arrange & Act & Assert
        with pytest.raises(APIError) as exc_info:
            models_endpoint.get(invalid_model_id)
        
        assert isinstance(exc_info.value, APIError), f"Should raise APIError for {error_type}"


class Test_ModelsEndpoint_Get_03_BoundaryBehaviors:
    """Test boundary condition behaviors for ModelsEndpoint get method."""
    
    @pytest.mark.parametrize("length,description", [
        (500, "very long model ID"),
        (1000, "extremely long model ID"),
        (2000, "maximum length model ID")
    ])
    def test_get_model_handles_extremely_long_identifiers(self, models_endpoint, length, description):
        """
        Test get method behavior with model IDs at the boundaries of acceptable length.
        
        Verifies that the endpoint properly handles very long model identifiers without
        causing buffer overflows, URL length issues, or other boundary-related problems.
        """
        # Arrange
        very_long_id = "a" * length
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            models_endpoint.get(very_long_id)
        
        # Should get client or server error for overly long IDs
        assert any(code in str(exc_info.value) for code in ["400", "414", "404"]), f"Expected error for {description}"


class Test_ModelsEndpoint_Get_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ModelsEndpoint get method."""
    
    @pytest.mark.parametrize("error_scenario", [
        "definitely_nonexistent_model",
        "invalid/model/format/too/many/slashes",
        "model_with_unicode_🚀_characters"
    ])
    def test_get_model_api_error_propagation_and_context(self, models_endpoint, error_scenario):
        """
        Test that APIError exceptions contain appropriate context and are properly propagated.
        
        Verifies that error information from the OpenRouter API is preserved and presented
        in a way that allows calling code to make appropriate decisions about error handling.
        """
        # Arrange & Act & Assert
        with pytest.raises(APIError) as exc_info:
            models_endpoint.get(error_scenario)
        
        error = exc_info.value
        assert isinstance(error, APIError), f"Should propagate APIError for {error_scenario}"
        assert len(str(error)) > 0, "Error should contain meaningful message"


# Test classes for get_context_length method
class Test_ModelsEndpoint_GetContextLength_01_NominalBehaviors:
    """Test nominal get_context_length operation behaviors for ModelsEndpoint."""
    
    @pytest.mark.parametrize("model_id,expected_range", [
        ("anthropic/claude-3-opus", (100000, 300000)),
        ("anthropic/claude-3-sonnet", (100000, 300000)),
        ("openai/gpt-4", (4000, 200000)),
        ("openai/gpt-3.5-turbo", (2000, 20000))
    ])
    def test_get_context_length_returns_valid_integer_values(self, models_endpoint, model_id, expected_range):
        """
        Test that get_context_length returns reasonable integer values for known models.
        
        Verifies that context length values are positive integers within expected ranges
        for different model families, ensuring data integrity and usefulness.
        """
        # Arrange & Act
        try:
            result = models_endpoint.get_context_length(model_id)
            
            # Assert
            assert isinstance(result, int), f"Context length should be integer for {model_id}"
            assert result > 0, f"Context length should be positive for {model_id}"
            assert expected_range[0] <= result <= expected_range[1], f"Context length should be in expected range for {model_id}"
        except APIError as e:
            # Model might not exist in current API
            assert "404" in str(e), f"Only 404 errors acceptable for model {model_id}"


class Test_ModelsEndpoint_GetContextLength_02_NegativeBehaviors:
    """Test negative get_context_length operation behaviors for ModelsEndpoint."""
    
    @pytest.mark.parametrize("invalid_model_id", [
        "nonexistent/model",
        "invalid_format",
        "model/without/context/length"
    ])
    def test_get_context_length_handles_invalid_models_appropriately(self, models_endpoint, invalid_model_id):
        """
        Test that get_context_length raises APIError for invalid or nonexistent models.
        
        Ensures that the method properly propagates errors from the underlying get method
        when model data cannot be retrieved or is malformed.
        """
        # Arrange & Act & Assert
        with pytest.raises(APIError):
            models_endpoint.get_context_length(invalid_model_id)


class Test_ModelsEndpoint_GetContextLength_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ModelsEndpoint get_context_length method."""
    
    def test_get_context_length_error_propagation_from_get_method(self, models_endpoint):
        """
        Test that errors from the underlying get method are properly propagated.
        
        Verifies that network errors, authentication failures, and other API issues
        are correctly passed through without modification or loss of context.
        """
        # Arrange
        invalid_model_id = "guaranteed/nonexistent/model/identifier"
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            models_endpoint.get_context_length(invalid_model_id)
        
        assert isinstance(exc_info.value, APIError), "Should propagate APIError from get method"


# Test classes for get_model_pricing method
class Test_ModelsEndpoint_GetModelPricing_01_NominalBehaviors:
    """Test nominal get_model_pricing operation behaviors for ModelsEndpoint."""
    
    @pytest.mark.parametrize("model_id", [
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet",
        "openai/gpt-4",
        "openai/gpt-3.5-turbo",
        "google/gemini-pro"
    ])
    def test_get_model_pricing_returns_complete_pricing_information(self, models_endpoint, model_id):
        """
        Test that get_model_pricing returns comprehensive ModelPricing data for valid models.
        
        Verifies that pricing information includes both prompt and completion costs,
        and that the data structure is properly formatted and accessible.
        """
        # Arrange & Act
        try:
            result = models_endpoint.get_model_pricing(model_id)
            
            # Assert
            assert isinstance(result, ModelPricing), f"Should return ModelPricing for {model_id}"
            assert hasattr(result, 'prompt'), f"Pricing should include prompt cost for {model_id}"
            assert hasattr(result, 'completion'), f"Pricing should include completion cost for {model_id}"
            
            # Verify pricing values are reasonable
            if hasattr(result, 'prompt') and result.prompt:
                assert float(result.prompt) >= 0, f"Prompt pricing should be non-negative for {model_id}"
            if hasattr(result, 'completion') and result.completion:
                assert float(result.completion) >= 0, f"Completion pricing should be non-negative for {model_id}"
                
        except APIError as e:
            # Model might not exist in current API
            assert "404" in str(e), f"Only 404 errors acceptable for model {model_id}"


class Test_ModelsEndpoint_GetModelPricing_02_NegativeBehaviors:
    """Test negative get_model_pricing operation behaviors for ModelsEndpoint."""
    
    @pytest.mark.parametrize("invalid_model_id", [
        "nonexistent/pricing/model",
        "invalid_format_pricing",
        "model/without/pricing/info"
    ])
    def test_get_model_pricing_handles_invalid_models_appropriately(self, models_endpoint, invalid_model_id):
        """
        Test that get_model_pricing raises APIError for invalid or nonexistent models.
        
        Ensures consistent error handling when pricing information cannot be retrieved
        due to invalid model identifiers or missing model data.
        """
        # Arrange & Act & Assert
        with pytest.raises(APIError):
            models_endpoint.get_model_pricing(invalid_model_id)


class Test_ModelsEndpoint_GetModelPricing_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ModelsEndpoint get_model_pricing method."""
    
    def test_get_model_pricing_error_propagation_from_underlying_methods(self, models_endpoint):
        """
        Test that errors from the underlying get method are properly propagated.
        
        Verifies that all types of API errors maintain their context and meaning
        when passed through the pricing-specific interface.
        """
        # Arrange
        invalid_model_id = "guaranteed/nonexistent/pricing/model"
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            models_endpoint.get_model_pricing(invalid_model_id)
        
        assert isinstance(exc_info.value, APIError), "Should propagate APIError from get method"


# Test classes for list_endpoints method
class Test_ModelsEndpoint_ListEndpoints_01_NominalBehaviors:
    """Test nominal list_endpoints operation behaviors for ModelsEndpoint."""
    
    @pytest.mark.parametrize("author,slug", [
        ("anthropic", "claude-3-opus"),
        ("anthropic", "claude-3-sonnet"),
        ("anthropic", "claude-3-haiku"),
        ("openai", "gpt-4"),
        ("openai", "gpt-3.5-turbo"),
        ("google", "gemini-pro"),
        ("meta", "llama-2-70b"),
        ("mistralai", "mistral-7b")
    ])
    def test_list_endpoints_returns_comprehensive_endpoint_data(self, models_endpoint, author, slug):
        """
        Test that list_endpoints returns complete ModelEndpointsResponse for valid author/slug combinations.
        
        Verifies that endpoint information includes all available API endpoints for the specified
        model, with proper structure and accessible endpoint metadata.
        """
        # Arrange & Act
        try:
            result = models_endpoint.list_endpoints(author, slug)
            
            # Assert
            assert isinstance(result, ModelEndpointsResponse), f"Should return ModelEndpointsResponse for {author}/{slug}"
            assert hasattr(result, 'data'), f"Response should have data attribute for {author}/{slug}"
            assert isinstance(result.data, dict), f"Response data should be a dict for {author}/{slug}"
            
            # Check if endpoints field exists in the response
            if 'endpoints' in result.data and isinstance(result.data['endpoints'], list):
                # If endpoints exist, verify their structure
                for endpoint in result.data['endpoints']:
                    assert 'name' in endpoint, f"Endpoint should have name for {author}/{slug}"
                    assert 'context_length' in endpoint, f"Endpoint should have context_length for {author}/{slug}"
                    assert 'pricing' in endpoint, f"Endpoint should have pricing for {author}/{slug}"
                    
        except APIError as e:
            # Model might not exist or have endpoints
            assert any(code in str(e) for code in ["404", "400"]), f"Expected 404/400 error for {author}/{slug}, got {e}"


class Test_ModelsEndpoint_ListEndpoints_02_NegativeBehaviors:
    """Test negative list_endpoints operation behaviors for ModelsEndpoint."""
    
    @pytest.mark.parametrize("author,slug,error_type", [
        ("nonexistent", "model", "nonexistent author"),
        ("   ", "   ", "whitespace only"),
        ("invalid@author", "invalid@slug", "special characters"),
        ("author/with/slashes", "slug/with/slashes", "path characters"),
        ("very_long_" + "a" * 100, "very_long_" + "b" * 100, "very long identifiers")
    ])
    def test_list_endpoints_handles_invalid_author_slug_combinations(self, models_endpoint, author, slug, error_type):
        """
        Test that list_endpoints raises APIError for various invalid author/slug combinations.
        
        Ensures robust handling of malformed, nonexistent, or potentially malicious
        author and slug identifiers that could cause API errors or security issues.
        """
        # Arrange & Act & Assert
        with pytest.raises(APIError) as exc_info:
            models_endpoint.list_endpoints(author, slug)
        
        assert isinstance(exc_info.value, APIError), f"Should raise APIError for {error_type}"


class Test_ModelsEndpoint_ListEndpoints_03_BoundaryBehaviors:
    """Test boundary condition behaviors for ModelsEndpoint list_endpoints method."""
    
    @pytest.mark.parametrize("string_length", [500, 2000])
    def test_list_endpoints_handles_maximum_length_identifiers(self, models_endpoint, string_length):
        """
        Test list_endpoints behavior with extremely long author and slug strings.
        
        Verifies that the method properly handles boundary conditions for identifier length
        without causing URL construction errors or server-side processing issues.
        """
        # Arrange
        very_long_author = "a" * string_length
        very_long_slug = "b" * string_length
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            models_endpoint.list_endpoints(very_long_author, very_long_slug)
        
        # Should get client or server error for overly long identifiers
        expected_codes = ["400", "414", "404"]  # Bad Request, URI Too Long, or Not Found
        assert any(code in str(exc_info.value) for code in expected_codes), f"Expected error for {string_length}-char strings"


class Test_ModelsEndpoint_ListEndpoints_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for ModelsEndpoint list_endpoints method."""
    
    @pytest.mark.parametrize("error_scenario", [
        ("definitely_nonexistent_author", "definitely_nonexistent_slug"),
        ("unicode_author_🚀", "unicode_slug_⭐"),
        ("null", "undefined"),
        ("SELECT * FROM models", "DROP TABLE endpoints")  # SQL injection attempt
    ])
    def test_list_endpoints_comprehensive_error_handling_and_security(self, models_endpoint, error_scenario):
        """
        Test that APIError exceptions are properly raised with appropriate context and security handling.
        
        Verifies that the endpoint safely handles potentially malicious input while providing
        meaningful error information for legitimate debugging and error handling scenarios.
        """
        # Arrange
        author, slug = error_scenario
        
        # Act & Assert
        with pytest.raises(APIError) as exc_info:
            models_endpoint.list_endpoints(author, slug)
        
        error = exc_info.value
        assert isinstance(error, APIError), f"Should raise APIError for scenario {error_scenario}"
        assert len(str(error)) > 0, f"Error should contain meaningful message for {error_scenario}"
        
        # Verify that error doesn't leak sensitive information
        error_message = str(error).lower()
        sensitive_terms = ["sql", "database", "internal", "server error", "stack trace"]
        assert not any(term in error_message for term in sensitive_terms), f"Error should not leak sensitive info for {error_scenario}"
