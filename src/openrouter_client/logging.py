"""
Logging configuration for OpenRouter Client.

This module provides logging configuration and utilities for consistent
logging across the library.

Exported:
- configure_logging: Function to configure logging for the library
"""

import logging
import os
import sys
from typing import Optional, Union, Dict

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(
    level: Union[int, str] = logging.INFO,
    format: str = DEFAULT_LOG_FORMAT,
    handlers: Optional[Dict[str, logging.Handler]] = None,
    capture_warnings: bool = True,
    to_file: Optional[str] = None,
    to_console: bool = True,
) -> logging.Logger:
    """
    Configure logging for the OpenRouter Client.
    
    Args:
        level (Union[int, str]): Logging level, either as string or integer constant.
        format (str): Log message format string.
        handlers (Optional[Dict[str, logging.Handler]]): Additional handlers to add.
        capture_warnings (bool): Whether to capture Python warnings via logging.
        to_file (Optional[str]): Path to log file, if file logging is desired.
        to_console (bool): Whether to log to console.
        
    Returns:
        logging.Logger: Configured root logger for the package.
    """
    # If level is a string, convert it to the corresponding logging level constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Create or get the openrouter_client logger
    logger = logging.getLogger("openrouter_client")
    
    # Remove any existing handlers to avoid duplicates when reconfiguring
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    
    # Set the logger level
    logger.setLevel(level)
    
    # Create formatter using the provided format string
    formatter = logging.Formatter(format)
    
    # Add a sensitive information filter
    sensitive_filter = SensitiveFilter()
    
    # If console logging is enabled:
    if to_console:
        # Create a StreamHandler for sys.stdout
        console_handler = logging.StreamHandler(sys.stdout)
        # Set the formatter and level
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        # Add the sensitive filter
        console_handler.addFilter(sensitive_filter)
        # Add to the logger
        logger.addHandler(console_handler)
    
    # If file logging is enabled (to_file is provided):
    if to_file:
        try:
            # Ensure the directory exists by creating it if needed
            log_dir = os.path.dirname(to_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Create a FileHandler for the specified path
            file_handler = logging.FileHandler(to_file)
            # Set the formatter and level
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            # Add the sensitive filter
            file_handler.addFilter(sensitive_filter)
            # Add to the logger
            logger.addHandler(file_handler)
        except (OSError, IOError, ValueError) as e:
            # Log the error but continue with console logging only
            if to_console:
                logger.warning(f"Failed to set up file logging to '{to_file}': {str(e)}")
        except Exception as e:
            # Log the error but continue with console logging only
            if to_console:
                logger.warning(f"Unexpected error while setting up file logging to '{to_file}': {str(e)}")
    
    # If additional handlers are provided:
    if handlers:
        for name, handler in handlers.items():
            # Set the formatter
            handler.setFormatter(formatter)
            # Add the sensitive filter
            handler.addFilter(sensitive_filter)
            # Add to the logger
            logger.addHandler(handler)
    
    # If capture_warnings is True:
    if capture_warnings:
        logging.captureWarnings(True)
    
    # Return the configured logger
    return logger


class SensitiveFilter(logging.Filter):
    """
    Filter to redact sensitive information in logs.
    
    Attributes:
        patterns (List[str]): Regular expression patterns to match sensitive data.
        replacement (str): String to use as replacement for sensitive data.
    """

    def __init__(self, replacement: str = "********"):
        """
        Initialize the filter with patterns and replacement.
        
        Args:
            replacement (str): String to use as replacement for sensitive data.
        """
        # Call parent initializer
        super().__init__()
        
        # Set up default patterns for API keys, tokens, passwords
        import re
        self.patterns = [
            # Process patterns with quotes first (more specific)
            re.compile(r'(api[-_]?key|token|auth[-_]?token|bearer)(\s*[=:]\s*)(["\'])([a-zA-Z0-9_\-\.]+)(["\'])', re.IGNORECASE),
            re.compile(r'(access_token|refresh_token)(\s*[=:]\s*)(["\'])([a-zA-Z0-9_\-\.]+)(["\'])', re.IGNORECASE),
            re.compile(r'(Authorization:\s*)(["\'])?((?:Bearer|Basic|Digest)\s+)([a-zA-Z0-9_\-\.\+\/\=]+)(["\'])?', re.IGNORECASE),
            re.compile(r'(OPENROUTER[-_]?API[-_]?KEY)([^a-zA-Z0-9]\s*[=:]\s*)(["\'])([a-zA-Z0-9_\-\.]+)(["\'])', re.IGNORECASE),
            re.compile(r'(password|passwd|pwd)(\s*[=:]\s*)(["\'])([^"\']*)(["\'])', re.IGNORECASE),
            # Then process patterns without quotes (more general) with protection against matching already redacted content
            re.compile(r'(api[-_]?key|token|auth[-_]?token|bearer)(\s*[=:]\s*)(?!\*{8})([a-zA-Z0-9_\-\.]+)', re.IGNORECASE),
            re.compile(r'(access_token|refresh_token)(\s*[=:]\s*)(?!\*{8})([a-zA-Z0-9_\-\.]+)', re.IGNORECASE),
            re.compile(r'(OPENROUTER[-_]?API[-_]?KEY)([^a-zA-Z0-9]\s*[=:]\s*)(?!\*{8})([a-zA-Z0-9_\-\.]+)', re.IGNORECASE),
            re.compile(r'(password|passwd|pwd)(\s*[=:]\s*)(?!\*{8})([^\s,;}{"\'\\\]\)]+)', re.IGNORECASE),
            re.compile(r'[a-zA-Z0-9_\-\.]+-token-[a-zA-Z0-9_\-\.]+', re.IGNORECASE),
        ]
        
        # Store the replacement string
        self.replacement = replacement

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record by redacting sensitive information.
        
        Args:
            record (logging.LogRecord): Log record to filter.
            
        Returns:
            bool: Always True (the record is always processed, just modified).
        """
        try:
            # If the message is a string, redact sensitive information
            if isinstance(record.msg, str):
                for pattern in self.patterns:
                    try:
                        if "Authorization:" in str(pattern.pattern):
                            try:
                                # First, check if there's a match
                                match = pattern.search(record.msg)
                                if match:
                                    # Replace the original message with a constructed one
                                    prefix = match.group(1)  # "Authorization: "
                                    quote_open = match.group(2) or ""  # Opening quote (or empty string)
                                    quote_close = match.group(5) or ""  # Closing quote (or empty string)
                                    record.msg = prefix + quote_open + self.replacement + quote_close
                                    continue  # Skip the general case handling
                            except Exception:
                                pass  # Fall back to standard handling if special case fails
                        # Attempt to search for pattern
                        match = pattern.search(record.msg)
                        if match:
                            try:
                                # Get group information
                                groups_count = pattern.groups
                                if "Authorization:" in str(pattern.pattern): # Special handling for Authorization header
                                    try:
                                        record.msg = pattern.sub(r'\1' + self.replacement, record.msg)
                                    except Exception:
                                        pass
                                elif groups_count == 5:  # Complex pattern with quotes
                                    try:
                                        # Attempt substitution for complex pattern
                                        record.msg = pattern.sub(r'\1\2\3' + self.replacement + r'\5', record.msg)
                                    except Exception:
                                        # If substitution fails, continue with next pattern
                                        pass
                                elif groups_count == 3:  # Simple pattern
                                    try:
                                        # Attempt substitution for simple pattern
                                        record.msg = pattern.sub(r'\1\2' + self.replacement, record.msg)
                                    except Exception:
                                        # If substitution fails, continue with next pattern
                                        pass
                                else: # Other patterns
                                    try:
                                        # Attempt substitution for simple pattern
                                        record.msg = pattern.sub(self.replacement, record.msg)
                                    except Exception:
                                        # If substitution fails, continue with next pattern
                                        pass
                            except Exception:
                                # If accessing groups fails, continue with next pattern
                                pass
                    except Exception:
                        # If pattern search fails, continue with next pattern
                        pass
            
            # If record has args and they contain strings, redact sensitive information
            if hasattr(record, 'args') and record.args:
                try:
                    args_list = list(record.args)
                    for i, arg in enumerate(args_list):
                        if isinstance(arg, str):
                            for pattern in self.patterns:
                                if "Authorization:" in str(pattern.pattern):
                                    try:
                                        # First, check if there's a match
                                        match = pattern.search(record.msg)
                                        if match:
                                            # Replace the original message with a constructed one
                                            prefix = match.group(1)  # "Authorization: "
                                            quote_open = match.group(2) or ""  # Opening quote (or empty string)
                                            quote_close = match.group(5) or ""  # Closing quote (or empty string)
                                            record.msg = prefix + quote_open + self.replacement + quote_close
                                            continue  # Skip the general case handling
                                    except Exception:
                                        pass  # Fall back to standard handling if special case fails
                                try:
                                    # Attempt to search for pattern
                                    match = pattern.search(arg)
                                    if match:
                                        try:
                                            # Get group information
                                            groups_count = pattern.groups
                                            if "Authorization:" in str(pattern.pattern): # Special handling for Authorization header
                                                try:
                                                    record.msg = pattern.sub(r'\1' + self.replacement, record.msg)
                                                except Exception:
                                                    pass
                                            elif groups_count == 5:  # Complex pattern with quotes
                                                try:
                                                    # Attempt substitution for complex pattern
                                                    args_list[i] = pattern.sub(r'\1\2\3' + self.replacement + r'\5', arg)
                                                except Exception:
                                                    # If substitution fails, continue with next pattern
                                                    pass
                                            elif groups_count == 3:  # Simple pattern without quotes
                                                try:
                                                    # Attempt substitution for simple pattern
                                                    args_list[i] = pattern.sub(r'\1\2' + self.replacement, arg)
                                                except Exception:
                                                    # If substitution fails, continue with next pattern
                                                    pass
                                            else:  # Other patterns
                                                try:
                                                    # Attempt substitution for simple pattern
                                                    args_list[i] = pattern.sub(self.replacement, arg)
                                                except Exception:
                                                    # If substitution fails, continue with next pattern
                                                    pass
                                        except Exception:
                                            # If accessing groups fails, continue with next pattern
                                            pass
                                except Exception:
                                    # If pattern search fails, continue with next pattern
                                    pass
                    
                    # Update record.args with the modified args
                    record.args = tuple(args_list)
                except Exception:
                    # If processing args fails, continue
                    pass
        except Exception:
            # Catch any other exceptions in the filtering process
            pass
        
        # Always process the record
        return True
