"""
OpenRouter Client package initialization.

This module initializes the OpenRouter Client package, exposing the main client
class and configuring package-wide settings for interacting with OpenRouter API.

Exported:
- OpenRouterClient: Main client for interacting with OpenRouter API
- __version__: Package version
- configure_logging: Logging configuration function
- AuthManager: Authentication and API key manager
- HTTPManager: HTTP communication manager with rate limiting
- RequestMethod: Enum of HTTP methods

- TokenCounter: Token counting utility
- determine_context_length: Context length determination function

Tool utilities:
- tool: Decorator for creating typed tools from Python functions
- build_tool_definition: Create a ToolDefinition from a function
- build_chat_completion_tool: Create a ChatCompletionTool from a function
- build_function_definition: Create a FunctionDefinition from a function
- build_function_parameters: Build function parameters from type hints
- build_parameter_schema: Convert Python type annotations to parameter schema
- build_function_call: Create a function call from a function and arguments
- build_tool_call: Create a tool call from a function and arguments
- create_function_definition_from_dict: Create a function definition from a dictionary
- create_tool_definition_from_dict: Create a tool definition from a dictionary
- create_chat_completion_tool_from_dict: Create a chat completion tool from a dictionary
- create_function_call_from_dict: Create a function call from a dictionary
- create_tool_call_from_dict: Create a tool call from a dictionary
- create_parameter_schema_from_value: Create parameter schema from a value
"""
from .client import OpenRouterClient
from .version import __version__
from .logging import configure_logging
from .auth import AuthManager
from .http import HTTPManager
from .types import RequestMethod
from .tools import (
    tool,
    build_tool_definition,
    build_chat_completion_tool,
    build_function_definition,
    build_function_parameters,
    build_parameter_schema,
    build_function_call,
    build_tool_call,
    create_function_definition_from_dict,
    create_tool_definition_from_dict,
    create_chat_completion_tool_from_dict,
    create_function_call_from_dict,
    create_tool_call_from_dict,
    create_parameter_schema_from_value
)

__all__ = [
    # Core client and utilities
    'OpenRouterClient',
    '__version__',
    'configure_logging',
    'AuthManager',
    'HTTPManager',
    'RequestMethod',
    
    # Tool utilities
    'tool',
    'build_tool_definition',
    'build_chat_completion_tool',
    'build_function_definition',
    'build_function_parameters',
    'build_parameter_schema',
    'build_function_call',
    'build_tool_call',
    'create_function_definition_from_dict',
    'create_tool_definition_from_dict',
    'create_chat_completion_tool_from_dict',
    'create_function_call_from_dict',
    'create_tool_call_from_dict',
    'create_parameter_schema_from_value'
]
