import pytest
import logging
from logging import FileHandler as OriginalFileHandler
import os
import re
import sys
from unittest.mock import patch, MagicMock

from openrouter_client.logging import configure_logging, SensitiveFilter


class Test_ConfigureLogging_01_NominalBehaviors:
    """Test suite for nominal behaviors of configure_logging function."""

    def test_core_logger_configuration(self):
        """Verify the function returns a logger with correct name, level, and format."""
        logger = configure_logging()
        
        # Check core configuration
        assert logger.name == "openrouter_client"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
        
        # Check formatter
        for handler in logger.handlers:
            assert handler.formatter._fmt == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @pytest.mark.parametrize("to_console, to_file, expected_handlers", [
        (True, None, 1),     # Console only
        (False, "test.log", 1),  # File only
        (True, "test.log", 2),   # Both console and file
        (False, None, 0),    # No handlers
    ])
    def test_handler_configuration_based_on_parameters(self, to_console, to_file, expected_handlers, tmp_path):
        """
        Test that console and file handlers are correctly created based on input parameters.
        
        Args:
            to_console: Boolean indicating whether to log to console
            to_file: Path to log file or None
            expected_handlers: Expected number of handlers
            tmp_path: Pytest fixture providing a temporary directory
        """
        if to_file:
            full_path = os.path.join(tmp_path, to_file)
            logger = configure_logging(to_console=to_console, to_file=full_path)
        else:
            logger = configure_logging(to_console=to_console, to_file=to_file)
            
        # Verify handler count matches expectations
        assert len(logger.handlers) == expected_handlers
        
        # Verify handler types
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) 
                           and not isinstance(h, logging.FileHandler)]
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        
        if to_console:
            assert len(console_handlers) == 1
            assert console_handlers[0].stream == sys.stdout
            
        if to_file:
            assert len(file_handlers) == 1
            assert file_handlers[0].baseFilename == full_path
    
    def test_sensitive_filter_application(self):
        """Verify all handlers have the SensitiveFilter applied."""
        logger = configure_logging()
        
        # All handlers should have a SensitiveFilter
        for handler in logger.handlers:
            sensitive_filters = [f for f in handler.filters if isinstance(f, SensitiveFilter)]
            assert len(sensitive_filters) == 1


class Test_ConfigureLogging_02_NegativeBehaviors:
    """Test suite for negative behaviors of configure_logging function."""
    
    def test_invalid_parameter_handling(self):
        """Test function behavior with invalid logging level strings."""
        logger = configure_logging(level="INVALID_LEVEL")
        
        # Should default to INFO if level string is invalid
        assert logger.level == logging.INFO
        
    @pytest.mark.parametrize("invalid_path", [
        "/non/existent/directory/with/permissions/issues/log.txt",
        "",  # Empty string
    ])
    def test_invalid_file_path(self, invalid_path, monkeypatch):
        """
        Test function behavior with invalid file paths.
        
        Args:
            invalid_path: Invalid file path to test
            monkeypatch: Pytest fixture to mock os.makedirs
        """
        # Mock os.makedirs to raise an exception
        def mock_makedirs(*args, **kwargs):
            raise OSError("Cannot create directory")
            
        monkeypatch.setattr(os, "makedirs", mock_makedirs)
        
        # Function should not raise an exception but log to console only
        logger = configure_logging(to_file=invalid_path)
        
        # Should have only console handler
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert not isinstance(logger.handlers[0], logging.FileHandler)


class Test_ConfigureLogging_03_BoundaryBehaviors:
    """Test suite for boundary behaviors of configure_logging function."""
    
    @pytest.mark.parametrize("level_value, expected_level", [
        (logging.NOTSET, logging.NOTSET),      # Minimum level
        (logging.CRITICAL, logging.CRITICAL),  # Maximum level
        ("NOTSET", logging.NOTSET),            # Minimum level as string
        ("CRITICAL", logging.CRITICAL),        # Maximum level as string
    ])
    def test_extreme_logging_levels(self, level_value, expected_level):
        """
        Test with both minimum (NOTSET) and maximum (CRITICAL) logging levels.
        
        Args:
            level_value: The logging level to test
            expected_level: The expected level value to be set
        """
        logger = configure_logging(level=level_value)
        assert logger.level == expected_level
        
        # Verify handlers also have the correct level
        for handler in logger.handlers:
            assert handler.level == expected_level


class Test_ConfigureLogging_04_ErrorHandlingBehaviors:
    """Test suite for error handling behaviors of configure_logging function."""
    
    def test_file_system_error_handling(self, monkeypatch):
        """Test handling of file system errors during directory creation."""
        # Mock os.makedirs to raise an OSError
        def mock_makedirs(*args, **kwargs):
            raise OSError("Permission denied")
            
        monkeypatch.setattr(os, "makedirs", mock_makedirs)
        
        # Function should fall back to console logging
        logger = configure_logging(to_file="/some/path/log.txt")
        
        # Should have only console handler
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert not isinstance(logger.handlers[0], logging.FileHandler)
    
    @patch('logging.FileHandler')
    def test_handler_failure_recovery(self, mock_file_handler):
        """Test that the function continues operation when a handler fails to initialize."""
        # Mock FileHandler to raise an exception
        mock_file_handler.side_effect = ValueError("Failed to create handler")
        
        # Should not raise an exception and fall back to console logging only
        logger = configure_logging(to_file="test.log")
        
        # Should have only console handler
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert not isinstance(logger.handlers[0], OriginalFileHandler)


class Test_ConfigureLogging_05_StateTransitionBehaviors:
    """Test suite for state transition behaviors of configure_logging function."""
    
    def test_reconfiguration_behavior(self):
        """
        Verify that calling the function multiple times properly removes existing
        handlers before adding new ones.
        """
        # First configuration with console only
        logger = configure_logging(to_console=True, to_file=None)
        initial_handlers = len(logger.handlers)
        assert initial_handlers == 1
        
        # Second configuration with both console and file
        with patch('logging.FileHandler') as mock_file_handler:
            # Mock successful file handler creation
            mock_instance = MagicMock()
            mock_file_handler.return_value = mock_instance
            
            logger = configure_logging(to_console=True, to_file="test.log")
            
            # Verify old handlers were removed and new ones added
            assert len(logger.handlers) == 2
            assert mock_file_handler.called
        
        # Third configuration with neither
        logger = configure_logging(to_console=False, to_file=None)
        assert len(logger.handlers) == 0


class Test_SensitiveFilter_Init_01_NominalBehaviors:
    """Test suite for nominal behaviors of SensitiveFilter.__init__ method."""
    
    def test_pattern_initialization(self):
        """Verify regex patterns for sensitive data are properly compiled."""
        sensitive_filter = SensitiveFilter()
        
        # Check patterns are compiled and available
        assert hasattr(sensitive_filter, 'patterns')
        assert len(sensitive_filter.patterns) > 0
        
        # Check each pattern is a compiled regex
        for pattern in sensitive_filter.patterns:
            assert isinstance(pattern, re.Pattern)


class Test_SensitiveFilter_Init_03_BoundaryBehaviors:
    """Test suite for boundary behaviors of SensitiveFilter.__init__ method."""
    
    @pytest.mark.parametrize("replacement", [
        "",              # Empty string
        "*",             # Single character
        "*" * 100,       # Very long string
        "REDACTED",      # Text replacement
    ])
    def test_replacement_string_variations(self, replacement):
        """
        Test with various replacement string lengths to ensure the redaction 
        mechanism works correctly.
        
        Args:
            replacement: Replacement string to use for redaction
        """
        sensitive_filter = SensitiveFilter(replacement=replacement)
        assert sensitive_filter.replacement == replacement
        
        # Test the filter works with this replacement
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="api_key='secret12345'", args=(), exc_info=None
        )
        
        sensitive_filter.filter(record)
        assert replacement in record.msg
        assert "secret12345" not in record.msg


class Test_SensitiveFilter_Filter_01_NominalBehaviors:
    """Test suite for nominal behaviors of SensitiveFilter.filter method."""
    
    @pytest.mark.parametrize("input_msg, expected_contains, expected_not_contains", [
        # API keys
        ("api_key='secret12345'", ["api_key='********'"], ["secret12345"]),
        ("token: 'abcdef123456'", ["token: '********'"], ["abcdef123456"]),
        ("auth_token = xyz987abc", ["auth_token = ********"], ["xyz987abc"]),
        # OAuth patterns
        ("access_token: mysecrettoken", ["access_token: ********"], ["mysecrettoken"]),
        ("refresh_token='longrefreshtoken'", ["refresh_token='********'"], ["longrefreshtoken"]),
        # Authorization headers
        ("Authorization: Bearer abc123.def456.ghi789", ["Authorization: ********"], ["abc123.def456.ghi789"]),
        ("Authorization: 'Basic dXNlcjpwYXNz'", ["Authorization: '********'"], ["dXNlcjpwYXNz"]),
        # OpenRouter API keys
        ("OPENROUTER_API_KEY = 'sk-or-abc123'", ["OPENROUTER_API_KEY = '********'"], ["sk-or-abc123"]),
        # Passwords
        ("password='super_secret!'", ["password='********'"], ["super_secret!"]),
    ])
    def test_sensitive_data_redaction_in_messages(self, input_msg, expected_contains, expected_not_contains):
        """
        Verify all categories of sensitive data are properly redacted in log messages.
        
        Args:
            input_msg: Input message containing sensitive data
            expected_contains: Strings that should be in the redacted output
            expected_not_contains: Strings that should not be in the redacted output
        """
        # Create a filter and test record
        sensitive_filter = SensitiveFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=input_msg, args=(), exc_info=None
        )
        
        # Apply the filter
        result = sensitive_filter.filter(record)
        
        # Verify filter returns True (allow message through)
        assert result is True
        
        # Check that sensitive data is redacted
        for expected in expected_contains:
            assert expected in record.msg
        
        # Check that original sensitive data is removed
        for not_expected in expected_not_contains:
            assert not_expected not in record.msg
    
    def test_sensitive_data_redaction_in_record_args(self):
        """Test that sensitive information in record args is also redacted."""
        sensitive_filter = SensitiveFilter()
        
        # Create a record with sensitive data in args
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="User %s authenticated with token %s",
            args=("testuser", "secret-token-123"), exc_info=None
        )
        
        # Apply the filter
        sensitive_filter.filter(record)
        
        # Check that args are modified
        assert record.args[0] == "testuser"  # Non-sensitive arg unchanged
        assert "secret-token" not in record.args[1]
        assert "********" in record.args[1]
    
    def test_non_sensitive_data_preservation(self):
        """Verify regular, non-sensitive information remains unmodified."""
        sensitive_filter = SensitiveFilter()
        
        # Create a record with non-sensitive data
        original_msg = "User logged in from 192.168.1.1 with username john_doe"
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=original_msg, args=(), exc_info=None
        )
        
        # Apply the filter
        sensitive_filter.filter(record)
        
        # Verify message is unchanged
        assert record.msg == original_msg


class Test_SensitiveFilter_Filter_02_NegativeBehaviors:
    """Test suite for negative behaviors of SensitiveFilter.filter method."""
    
    @pytest.mark.parametrize("record_msg", [
        None,                   # None message
        123,                    # Integer message
        {"key": "value"},       # Dictionary message
        ["list", "of", "items"] # List message
    ])
    def test_non_string_record_handling(self, record_msg):
        """
        Test with records containing non-string messages.
        
        Args:
            record_msg: Non-string message to test
        """
        sensitive_filter = SensitiveFilter()
        
        # Create record with non-string message
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=record_msg, args=(), exc_info=None
        )
        
        # Filter should not raise exception and return True
        try:
            result = sensitive_filter.filter(record)
            assert result is True
            assert record.msg == record_msg  # Message should be unchanged
        except Exception as e:
            pytest.fail(f"Filter raised exception with non-string message: {e}")


class Test_SensitiveFilter_Filter_03_BoundaryBehaviors:
    """Test suite for boundary behaviors of SensitiveFilter.filter method."""
    
    @pytest.mark.parametrize("input_msg", [
        # Sensitive data at beginning of string
        "api_key=secret123 and some other text",
        # Sensitive data at end of string
        "Configuration completed with token=secret123",
        # Multiple occurrences in same message
        "First api_key=secret1 and second api_key=secret2",
        # Sensitive data with special characters
        "password='p@$$w0rd!#*'",
        # Very long sensitive data
        f"api_key={'a' * 1000}"
    ])
    def test_edge_case_pattern_matching(self, input_msg):
        """
        Test with sensitive data in various positions and multiple occurrences.
        
        Args:
            input_msg: Input message with edge case pattern
        """
        sensitive_filter = SensitiveFilter()
        
        # Create record with edge case message
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=input_msg, args=(), exc_info=None
        )
        
        # Apply filter
        sensitive_filter.filter(record)
        
        # Secret data should be redacted
        assert "secret1" not in record.msg
        assert "secret2" not in record.msg
        assert "secret123" not in record.msg
        assert "p@$$w0rd!#*" not in record.msg
        assert "a" * 100 not in record.msg
        
        # Replacement string should be present
        assert "********" in record.msg


class Test_SensitiveFilter_Filter_04_ErrorHandlingBehaviors:
    """Test suite for error handling behaviors of SensitiveFilter.filter method."""
    
    def test_exception_handling_during_filtering(self):
        """Verify the filter gracefully handles exceptions during pattern matching."""
        sensitive_filter = SensitiveFilter()
        
        # Create a filter with a problematic regex pattern
        with patch.object(sensitive_filter, 'patterns') as mock_patterns:
            # Mock a pattern that raises an exception when used
            bad_pattern = MagicMock()
            bad_pattern.search.side_effect = Exception("Regex error")
            bad_pattern.sub.side_effect = Exception("Substitution error")
            mock_patterns.__iter__.return_value = [bad_pattern]
            
            # Create a record
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg="This contains api_key=sensitive_value", args=(), exc_info=None
            )
            
            # Filter should not raise exception
            try:
                result = sensitive_filter.filter(record)
                # Should still let the record through
                assert result is True
                # Message should be unchanged when exception occurs
                assert record.msg == "This contains api_key=sensitive_value"
            except Exception as e:
                pytest.fail(f"Filter did not handle exception properly: {e}")
