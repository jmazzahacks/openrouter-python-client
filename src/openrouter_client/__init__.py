"""
OpenRouter Python Client - Unofficial Python client for the OpenRouter API.

This package provides a comprehensive, type-safe interface for interacting with 
the OpenRouter API, supporting all major endpoints including chat completions, 
text completions, model information, generations, credits, and API key management.

Features:
- Full API Support: Chat completions, text completions, models, generations, credits, keys
- Streaming Support: Real-time streaming for chat and text completions
- Automatic Rate Limiting: Smart rate limiting based on API key limits using SmartSurge
- Type Safety: Fully typed interfaces with Pydantic models
- Function Calling: Built-in support for OpenAI-style function calling with decorators
- Prompt Caching: Support for prompt caching on compatible models
- Safe Key Management: Secure API key handling with encryption and extensible secrets management
- Context Length Management: Automatic tracking and querying of model context lengths
- Comprehensive Testing: Extensive test suite with local unit tests and remote integration tests

Main Classes:
- OpenRouterClient: Main client for interacting with OpenRouter API
- AuthManager: Authentication and API key manager with encryption support
- HTTPManager: HTTP communication manager with smart rate limiting and retries
- SecretsManager: Base class for custom secrets management implementations

Tool Utilities:
- tool: Decorator for creating typed tools from Python functions
- build_tool_definition: Create a ToolDefinition from a function
- build_chat_completion_tool: Create a ChatCompletionTool from a function
- build_function_definition: Create a FunctionDefinition from a function
- build_function_parameters: Build function parameters from type hints
- build_parameter_schema: Convert Python type annotations to parameter schema
- build_function_call: Create a function call from a function and arguments
- build_tool_call: Create a tool call from a function and arguments

Factory Functions:
- create_function_definition_from_dict: Create a function definition from a dictionary
- create_tool_definition_from_dict: Create a tool definition from a dictionary
- create_chat_completion_tool_from_dict: Create a chat completion tool from a dictionary
- create_function_call_from_dict: Create a function call from a dictionary
- create_tool_call_from_dict: Create a tool call from a dictionary
- create_parameter_schema_from_value: Create parameter schema from a value

Example:
    >>> from openrouter_client import OpenRouterClient
    >>> 
    >>> client = OpenRouterClient(api_key="your-api-key")
    >>> response = client.chat.create(
    ...     model="anthropic/claude-3-opus",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
    >>> print(response.choices[0].message.content)
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
