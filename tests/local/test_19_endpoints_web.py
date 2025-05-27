import pytest
from unittest.mock import Mock, patch
from requests.exceptions import RequestException
import json

from openrouter_client.endpoints.web import WebEndpoint
from openrouter_client.auth import AuthManager  
from openrouter_client.http import HTTPManager
from openrouter_client.exceptions import APIError
from pydantic import ValidationError


class Test_WebEndpoint_Init_01_NominalBehaviors:
    """Test nominal initialization behaviors for WebEndpoint."""
    
    def test_initialize_with_valid_managers(self):
        """Test initialization with valid AuthManager and HTTPManager instances."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Act
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint.auth_manager == auth_manager
        assert endpoint.http_manager == http_manager
        assert hasattr(endpoint, 'logger')


class Test_WebEndpoint_Init_02_NegativeBehaviors:
    """Test negative initialization behaviors for WebEndpoint."""
    
    @pytest.mark.parametrize("auth_manager,http_manager", [
        (None, Mock(spec=HTTPManager)),
        (Mock(spec=AuthManager), None),
        (None, None),
        ("invalid_type", Mock(spec=HTTPManager)),
        (Mock(spec=AuthManager), "invalid_type"),
        (123, Mock(spec=HTTPManager)),
        (Mock(spec=AuthManager), 456),
        ([], Mock(spec=HTTPManager)),
        (Mock(spec=AuthManager), {})
    ])
    def test_initialize_with_invalid_manager_parameters(self, auth_manager, http_manager):
        """Test initialization with None or invalid types for manager parameters."""
        # Arrange/Act/Assert
        with pytest.raises(ValidationError):
            WebEndpoint(auth_manager, http_manager)


class Test_WebEndpoint_Init_04_ErrorHandlingBehaviors:
    """Test error handling during WebEndpoint initialization."""
    
    def test_handle_parent_class_initialization_exception(self):
        """Test handling of exceptions during parent class initialization."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        with patch('openrouter_client.endpoints.web.BaseEndpoint.__init__', side_effect=Exception("Parent init failed")):
            # Act/Assert
            with pytest.raises(Exception, match="Parent init failed"):
                WebEndpoint(auth_manager, http_manager)


class Test_WebEndpoint_Init_05_StateTransitionBehaviors:
    """Test state transition behaviors during WebEndpoint initialization."""
    
    def test_verify_authentication_state_establishment(self):
        """Test that authentication state is properly established after initialization."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Act
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint.auth_manager is not None
        assert hasattr(endpoint, '_get_headers')
        # Verify the endpoint can generate authentication headers without exceptions
        with patch.object(endpoint, '_get_headers', return_value={'Authorization': 'Bearer token'}):
            headers = endpoint._get_headers()
            assert headers is not None


class Test_WebEndpoint_Search_01_NominalBehaviors:
    """Test nominal search behaviors for WebEndpoint."""
    
    @pytest.mark.parametrize("query,max_results,recent,kwargs,expected_data", [
        ("test query", None, None, {}, {"query": "test query"}),
        ("python programming", 10, True, {}, {"query": "python programming", "max_results": 10, "recent": True}),
        ("machine learning", 5, False, {"domain": "academic"}, {"query": "machine learning", "max_results": 5, "recent": False, "domain": "academic"}),
        ("AI research", None, None, {"language": "en", "region": "US"}, {"query": "AI research", "language": "en", "region": "US"}),
        ("data science", 0, None, {}, {"query": "data science", "max_results": 0}),
        ("blockchain", None, True, {"safe_search": True}, {"query": "blockchain", "recent": True, "safe_search": True})
    ])
    def test_search_with_valid_parameters_returns_structured_response(self, query, max_results, recent, kwargs, expected_data):
        """Test search execution with valid query string and return structured JSON response."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        mock_response = Mock()
        expected_result = {"results": [{"title": "Result 1", "url": "https://example.com"}], "total": 1}
        mock_response.json.return_value = expected_result
        http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/search')
        
        # Act
        result = endpoint.search(query, max_results, recent, **kwargs)
        
        # Assert
        http_manager.post.assert_called_once()
        call_args = http_manager.post.call_args
        assert call_args.kwargs['json'] == expected_data
        assert call_args.kwargs['endpoint'] == 'https://api.example.com/web/search'
        assert 'headers' in call_args.kwargs
        assert result == expected_result


class Test_WebEndpoint_Search_02_NegativeBehaviors:
    """Test negative search behaviors for WebEndpoint."""
    
    @pytest.mark.parametrize("query", [
        "",
        "   ",
        "\t\n\r   \n",
        "   \t   "
    ])
    def test_handle_empty_or_whitespace_query_strings(self, query):
        """Test handling of empty or whitespace-only query strings."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Invalid query", "message": "Query cannot be empty"}
        http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/search')
        
        # Act
        result = endpoint.search(query)
        
        # Assert
        http_manager.post.assert_called_once()
        call_args = http_manager.post.call_args
        assert call_args.kwargs['json']['query'] == query
        assert result is not None


class Test_WebEndpoint_Search_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for WebEndpoint search."""
    
    @pytest.mark.parametrize("exception_type,exception_message", [
        (RequestException, "Network error"),
        (ConnectionError, "Connection failed"),
        (TimeoutError, "Request timeout")
    ])
    def test_handle_http_request_failures_during_post(self, exception_type, exception_message):
        """Test handling of HTTP request failures during POST operation."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        http_manager.post.side_effect = exception_type(exception_message)
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/search')
        
        # Act/Assert
        with pytest.raises(exception_type, match=exception_message):
            endpoint.search("test query")
    
    def test_handle_authentication_header_generation_failures(self):
        """Test handling of authentication header generation failures from _get_headers()."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        endpoint._get_headers = Mock(side_effect=Exception("Authentication failed"))
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/search')
        
        # Act/Assert
        with pytest.raises(Exception, match="Authentication failed"):
            endpoint.search("test query")
    
    def test_handle_malformed_json_responses_from_api(self):
        """Test handling of malformed JSON responses from the API."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/search')
        
        # Act/Assert
        with pytest.raises(json.JSONDecodeError):
            endpoint.search("test query")


class Test_WebEndpoint_Search_05_StateTransitionBehaviors:
    """Test state transition behaviors for WebEndpoint search."""
    
    def test_track_request_state_from_data_preparation_through_response_parsing(self):
        """Test request state tracking from data preparation through response parsing."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        mock_response = Mock()
        expected_result = {"results": [{"title": "Test Result"}], "status": "success", "query_time": 0.5}
        mock_response.json.return_value = expected_result
        http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/search')
        
        # Act
        result = endpoint.search("test query", max_results=10, recent=True)
        
        # Assert
        # Verify complete pipeline execution in correct order
        endpoint._get_headers.assert_called_once()
        endpoint._get_endpoint_url.assert_called_once_with('search')
        http_manager.post.assert_called_once()
        
        # Verify data preparation
        call_args = http_manager.post.call_args
        expected_data = {"query": "test query", "max_results": 10, "recent": True}
        assert call_args.kwargs['json'] == expected_data
        
        # Verify response parsing
        assert result == expected_result


class Test_WebEndpoint_Content_01_NominalBehaviors:
    """Test nominal content retrieval behaviors for WebEndpoint."""
    
    @pytest.mark.parametrize("url,kwargs,expected_data", [
        ("https://example.com", {}, {"url": "https://example.com"}),
        ("http://test.org/page", {"extract_links": True}, {"url": "http://test.org/page", "extract_links": True}),
        ("https://site.com/article", {"format": "markdown", "include_images": False}, {"url": "https://site.com/article", "format": "markdown", "include_images": False}),
        ("https://news.com/story", {"timeout": 30, "follow_redirects": True}, {"url": "https://news.com/story", "timeout": 30, "follow_redirects": True})
    ])
    def test_retrieve_and_process_content_with_structured_response(self, url, kwargs, expected_data):
        """Test content retrieval and processing from valid URLs with structured response."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        mock_response = Mock()
        expected_result = {"content": "extracted content", "title": "Page Title", "metadata": {"length": 1500}}
        mock_response.json.return_value = expected_result
        http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/content')
        
        # Act
        result = endpoint.content(url, **kwargs)
        
        # Assert
        http_manager.post.assert_called_once()
        call_args = http_manager.post.call_args
        assert call_args.kwargs['json'] == expected_data
        assert call_args.kwargs['endpoint'] == 'https://api.example.com/web/content'
        assert result == expected_result


class Test_WebEndpoint_Content_02_NegativeBehaviors:
    """Test negative content retrieval behaviors for WebEndpoint."""
    
    @pytest.mark.parametrize("url", [
        "not-a-url",
        "ftp://invalid-protocol.com",
        "https://",
        "http://",
        "",
        "javascript:alert('xss')",
        "data:text/html,<script>alert('xss')</script>",
        "file:///etc/passwd",
        "chrome://settings",
        "about:blank"
    ])
    def test_handle_malformed_urls_or_invalid_syntax(self, url):
        """Test handling of malformed URLs or invalid URL syntax."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Invalid URL", "message": "URL format not supported"}
        http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/content')
        
        # Act
        result = endpoint.content(url)
        
        # Assert
        http_manager.post.assert_called_once()
        call_args = http_manager.post.call_args
        assert call_args.kwargs['json']['url'] == url
        assert result is not None


class Test_WebEndpoint_Content_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for WebEndpoint content retrieval."""
    
    @pytest.mark.parametrize("exception_type,exception_message", [
        (RequestException, "Connection failed"),
        (ConnectionError, "Unable to reach endpoint"),
        (TimeoutError, "Content request timeout")
    ])
    def test_handle_http_request_failures_to_content_endpoint(self, exception_type, exception_message):
        """Test handling of HTTP request failures to the content endpoint."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        http_manager.post.side_effect = exception_type(exception_message)
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/content')
        
        # Act/Assert
        with pytest.raises(exception_type, match=exception_message):
            endpoint.content("https://example.com")
    
    def test_handle_authentication_failures_during_header_generation(self):
        """Test handling of authentication failures during header generation."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        endpoint._get_headers = Mock(side_effect=Exception("Authentication error"))
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/content')
        
        # Act/Assert
        with pytest.raises(Exception, match="Authentication error"):
            endpoint.content("https://example.com")
    
    def test_handle_malformed_json_responses(self):
        """Test handling of malformed JSON responses."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Bad JSON format", "", 0)
        http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/content')
        
        # Act/Assert
        with pytest.raises(json.JSONDecodeError):
            endpoint.content("https://example.com")


class Test_WebEndpoint_Content_05_StateTransitionBehaviors:
    """Test state transition behaviors for WebEndpoint content retrieval."""
    
    def test_monitor_complete_url_processing_pipeline_from_input_to_parsed_response(self):
        """Test complete URL processing pipeline from input to parsed response."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        mock_response = Mock()
        expected_result = {
            "content": "processed page content", 
            "metadata": {"title": "Test Page", "word_count": 500},
            "links": ["https://link1.com", "https://link2.com"]
        }
        mock_response.json.return_value = expected_result
        http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/content')
        
        # Act
        result = endpoint.content("https://example.com", extract_metadata=True, extract_links=True)
        
        # Assert
        # Verify complete pipeline execution in correct sequence
        endpoint._get_headers.assert_called_once()
        endpoint._get_endpoint_url.assert_called_once_with('content')
        http_manager.post.assert_called_once()
        
        # Verify data preparation and parameter passing
        call_args = http_manager.post.call_args
        expected_data = {"url": "https://example.com", "extract_metadata": True, "extract_links": True}
        assert call_args.kwargs['json'] == expected_data
        
        # Verify response parsing
        assert result == expected_result


class Test_WebEndpoint_Summarize_01_NominalBehaviors:
    """Test nominal summarization behaviors for WebEndpoint."""
    
    @pytest.mark.parametrize("url,length,format_param,kwargs,expected_data", [
        ("https://example.com", None, None, {}, {"url": "https://example.com"}),
        ("https://article.com", "short", "plain", {}, {"url": "https://article.com", "length": "short", "format": "plain"}),
        ("https://blog.org", "medium", "bullets", {"language": "en"}, {"url": "https://blog.org", "length": "medium", "format": "bullets", "language": "en"}),
        ("https://news.com", "long", "structured", {}, {"url": "https://news.com", "length": "long", "format": "structured"}),
        ("https://research.edu", "medium", "plain", {"include_citations": True, "academic_style": True}, {"url": "https://research.edu", "length": "medium", "format": "plain", "include_citations": True, "academic_style": True})
    ])
    def test_generate_summaries_for_valid_urls_with_optional_parameters(self, url, length, format_param, kwargs, expected_data):
        """Test summary generation for valid URLs with optional length and format parameters."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        mock_response = Mock()
        expected_result = {
            "summary": "Generated summary content", 
            "length": length or "medium",
            "format": format_param or "plain",
            "word_count": 150
        }
        mock_response.json.return_value = expected_result
        http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/summarize')
        
        # Act
        result = endpoint.summarize(url, length, format_param, **kwargs)
        
        # Assert
        http_manager.post.assert_called_once()
        call_args = http_manager.post.call_args
        assert call_args.kwargs['json'] == expected_data
        assert call_args.kwargs['endpoint'] == 'https://api.example.com/web/summarize'
        assert result == expected_result


class Test_WebEndpoint_Summarize_02_NegativeBehaviors:
    """Test negative summarization behaviors for WebEndpoint."""
    
    @pytest.mark.parametrize("url", [
        "invalid-url-format",
        "https://404-not-found-example.com",
        "",
        "ftp://unsupported-protocol.com",
        "javascript:void(0)",
        "data:text/plain,minimal"
    ])
    def test_handle_invalid_urls_that_cannot_be_summarized(self, url):
        """Test handling of invalid URLs that cannot be summarized."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Cannot summarize URL", "message": "URL is not accessible or has no content"}
        http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/summarize')
        
        # Act
        result = endpoint.summarize(url)
        
        # Assert
        http_manager.post.assert_called_once()
        call_args = http_manager.post.call_args
        assert call_args.kwargs['json']['url'] == url
        assert result is not None
    
    @pytest.mark.parametrize("length,format_param", [
        ("invalid_length", "plain"),
        ("short", "invalid_format"),
        ("extra_long", "bullets"),
        ("tiny", "structured"),
        ("medium", "xml"),
        ("huge", "yaml"),
        ("micro", "json")
    ])
    def test_handle_invalid_length_or_format_parameter_values(self, length, format_param):
        """Test handling of invalid length or format parameter values."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Invalid parameters", "message": "Unsupported length or format"}
        http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/summarize')
        
        # Act
        result = endpoint.summarize("https://example.com", length, format_param)
        
        # Assert
        http_manager.post.assert_called_once()
        call_args = http_manager.post.call_args
        assert call_args.kwargs['json']['length'] == length
        assert call_args.kwargs['json']['format'] == format_param
        assert result is not None


class Test_WebEndpoint_Summarize_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for WebEndpoint summarization."""
    
    @pytest.mark.parametrize("exception_type,exception_message", [
        (RequestException, "API unavailable"),
        (ConnectionError, "Summarization service down"),
        (TimeoutError, "Summarization timeout")
    ])
    def test_handle_summarization_api_endpoint_failures(self, exception_type, exception_message):
        """Test handling of summarization API endpoint failures."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        http_manager.post.side_effect = exception_type(exception_message)
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/summarize')
        
        # Act/Assert
        with pytest.raises(exception_type, match=exception_message):
            endpoint.summarize("https://example.com")
    
    def test_handle_authentication_or_authorization_failures(self):
        """Test handling of authentication or authorization failures."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        endpoint._get_headers = Mock(side_effect=Exception("Unauthorized access"))
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/summarize')
        
        # Act/Assert
        with pytest.raises(Exception, match="Unauthorized access"):
            endpoint.summarize("https://example.com")
    
    def test_handle_malformed_summary_responses_from_api(self):
        """Test handling of malformed summary responses from the API."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Malformed summary response", "", 0)
        http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/summarize')
        
        # Act/Assert
        with pytest.raises(json.JSONDecodeError):
            endpoint.summarize("https://example.com")


class Test_WebEndpoint_Summarize_05_StateTransitionBehaviors:
    """Test state transition behaviors for WebEndpoint summarization."""
    
    def test_verify_complete_parameter_validation_and_response_assembly_pipeline(self):
        """Test complete parameter validation and response assembly pipeline."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        endpoint = WebEndpoint(auth_manager, http_manager)
        
        mock_response = Mock()
        expected_result = {
            "summary": "Complete structured summary of the webpage content",
            "metadata": {"words": 150, "format": "structured", "processing_time": 2.3},
            "source_info": {"title": "Example Page", "word_count": 1200}
        }
        mock_response.json.return_value = expected_result
        http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={'Authorization': 'Bearer token'})
        endpoint._get_endpoint_url = Mock(return_value='https://api.example.com/web/summarize')
        
        # Act
        result = endpoint.summarize("https://example.com", "medium", "structured", include_metadata=True, include_source_info=True)
        
        # Assert
        # Verify complete pipeline execution in correct sequence
        endpoint._get_headers.assert_called_once()
        endpoint._get_endpoint_url.assert_called_once_with('summarize')
        http_manager.post.assert_called_once()
        
        # Verify parameter validation and data preparation
        call_args = http_manager.post.call_args
        expected_data = {
            "url": "https://example.com",
            "length": "medium", 
            "format": "structured",
            "include_metadata": True,
            "include_source_info": True
        }
        assert call_args.kwargs['json'] == expected_data
        
        # Verify response assembly
        assert result == expected_result
