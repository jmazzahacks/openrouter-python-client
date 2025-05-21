"""
Core types, enums and constants for OpenRouter Client.

This module defines shared type definitions used throughout the package,
providing consistent typing and enum values.

Exported:
- RequestMethod: Enum of HTTP request methods
- ModelRole: Enum of model roles (system, user, assistant)
- FinishReason: Enum of completion finish reasons
- TokenUsageType: Enum for token usage types
"""

from enum import Enum
from typing import TypeVar


T = TypeVar('T')


class ModelRole(str, Enum):
    """Roles in chat completions."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Reasons why a completion or generation finished.
    
    The API normalizes each model's finish_reason to one of the following values:
    tool_calls, stop, length, content_filter, error.
    """
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    ERROR = "error"
    NULL = None


class TokenUsageType(str, Enum):
    """Types of token usage accounting."""
    PROMPT = "prompt"
    COMPLETION = "completion"
    TOTAL = "total"


class RequestMethod(str, Enum):
    """HTTP request methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    TRACE = "TRACE"


# Common mime types for content
MIME_TYPES = {
    "json": "application/json",
    "text": "text/plain",
    "html": "text/html",
    "md": "text/markdown",
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "gif": "image/gif",
    "pdf": "application/pdf",
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "zip": "application/zip",
}


# Standard content type headers
JSON_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}
