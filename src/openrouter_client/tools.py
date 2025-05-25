"""Tool construction utilities for OpenRouter Client.

This module provides utilities for constructing type-checked tools and tool requests
from Python built-in types and functions, simplifying the creation of function and tool
definitions for chat completion requests. It also includes helper functions for prompt caching.

Exported:
- tool: Decorator to convert a Python function into a typed tool
- build_tool_definition: Create a ToolDefinition from a function or method
- build_chat_completion_tool: Create a ChatCompletionTool from a function or method
- build_parameter_schema: Convert Python type annotations to parameter schema
- build_function_call: Create a structured function call from a function and arguments
- build_tool_call: Create a ChatCompletionToolCall from a function and arguments
- infer_parameter_type: Infer OpenRouter parameter type from Python type annotation
- create_function_definition_from_dict: Create a function definition from a dictionary
- create_parameter_schema_from_value: Create parameter schema from Python values
- create_tool_definition_from_dict: Create a tool definition from a dictionary
- create_chat_completion_tool_from_dict: Create a chat completion tool from a dictionary
- create_function_call_from_dict: Create a function call from a dictionary
- create_tool_call_from_dict: Create a tool call from a dictionary
- cache_control: Create a cache control object for prompt caching
- create_cached_content: Create a cacheable text content part
- string_param_with_cache_control: Create a string parameter with cache control
"""

import inspect
import json
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints, get_origin, get_args

from pydantic import BaseModel

from .models.core import FunctionDefinition, ToolDefinition, TextContent, CacheControl
from .models.chat import (
    ParameterType, ParameterDefinition, StringParameter, NumberParameter,
    BooleanParameter, ArrayParameter, ObjectParameter, FunctionParameters,
    ChatCompletionTool, ChatCompletionToolCall, ToolCallFunction, FunctionCall
)


def infer_parameter_type(python_type: Type) -> Union[ParameterType, List[ParameterType]]:
    """
    Infer OpenRouter parameter type from Python type annotation.
    
    Args:
        python_type: The Python type annotation to convert.
            Supported types include: str, int, float, bool, list, dict, and their Union types.
            
    Returns:
        ParameterType or List[ParameterType]: The corresponding OpenRouter parameter type.
        
    Raises:
        ValueError: If the Python type cannot be mapped to an OpenRouter parameter type.
    """
    # Handle NoneType
    if python_type is type(None):
        return ParameterType.NULL
        
    # Handle primitive types
    if python_type is str:
        return ParameterType.STRING
    elif python_type is int:
        return ParameterType.INTEGER
    elif python_type is float:
        return ParameterType.NUMBER
    elif python_type is bool:
        return ParameterType.BOOLEAN
    elif python_type is list or python_type is List:
        return ParameterType.ARRAY
    elif python_type is dict or python_type is Dict:
        return ParameterType.OBJECT
        
    # Handle Union types
    origin = get_origin(python_type)
    if origin is Union:
        args = get_args(python_type)
        # Check if it's an Optional type first (Union with None)
        if type(None) in args:
            non_none_args = [arg for arg in args if arg is not type(None)]
            # For Optional[T], always return T's type rather than Union[T, None]
            if len(non_none_args) == 1:
                return infer_parameter_type(non_none_args[0])
        # Otherwise return list of all types
        return [infer_parameter_type(arg) for arg in args]  # type: ignore
        
    # Handle List and Dict with type arguments
    if origin is list:
        return ParameterType.ARRAY
    elif origin is dict:
        return ParameterType.OBJECT
            
    # Handle Enum types
    if inspect.isclass(python_type) and issubclass(python_type, Enum):
        return ParameterType.STRING
        
    # Handle Pydantic models
    if inspect.isclass(python_type) and issubclass(python_type, BaseModel):
        return ParameterType.OBJECT
    
    # Handle Any type - default to string
    if python_type is Any:
        return ParameterType.STRING
        
    raise ValueError(f"Cannot map Python type {python_type} to an OpenRouter parameter type")


def build_parameter_schema(
    param_name: str,
    param_type: Type,
    description: Optional[str] = None,
    default: Optional[Any] = None,
    required: bool = True
) -> ParameterDefinition:
    """
    Convert Python type annotation to a parameter definition.
    
    Args:
        param_name: Name of the parameter.
        param_type: Python type annotation.
        description: Optional description of the parameter.
        default: Optional default value for the parameter.
        required: Whether the parameter is required.
        
    Returns:
        ParameterDefinition: Parameter definition compatible with OpenRouter API.
    """
    # If parameter has a default value, it's not required unless explicitly specified
    if default is not None and required is True:
        required = False
        
    parameter_type = infer_parameter_type(param_type)
    
    # Handle Enum types
    enum_values = None
    if inspect.isclass(param_type) and issubclass(param_type, Enum):
        enum_values = [item.value for item in param_type]
    
    # Handle Union types
    origin = get_origin(param_type)
    if origin is Union:
        args = get_args(param_type)
        # If it's an Optional type (Union with None)
        if type(None) in args:
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                # Build parameter for the non-None type, and mark as not required
                # since it accepts None
                return build_parameter_schema(
                    param_name, non_none_args[0], description, default, False
                )
                
    # We'll handle the 'required' parameter differently based on the parameter type
    # We'll use it to set the appropriate properties in the build_function_parameters
    # function which adds the parameter name to the required array if needed
    
    # Create the appropriate parameter type based on the inferred type
    if parameter_type == ParameterType.STRING or (
        isinstance(parameter_type, list) and ParameterType.STRING in parameter_type
    ):
        return StringParameter(
            type=ParameterType.STRING,
            description=description,
            enum=enum_values,
            default=default,
            min_length=None,
            max_length=None,
            pattern=None,
            format=None
        )
    
    elif parameter_type == ParameterType.INTEGER or (
        isinstance(parameter_type, list) and ParameterType.INTEGER in parameter_type
    ):
        return NumberParameter(
            type=ParameterType.INTEGER,
            description=description,
            enum=enum_values,
            default=default,
            minimum=None,
            maximum=None,
            exclusive_minimum=None,
            exclusive_maximum=None,
            multiple_of=None
        )
    
    elif parameter_type == ParameterType.NUMBER or (
        isinstance(parameter_type, list) and ParameterType.NUMBER in parameter_type
    ):
        return NumberParameter(
            type=ParameterType.NUMBER,
            description=description,
            enum=enum_values,
            default=default,
            minimum=None,
            maximum=None,
            exclusive_minimum=None,
            exclusive_maximum=None,
            multiple_of=None
        )
    
    elif parameter_type == ParameterType.BOOLEAN or (
        isinstance(parameter_type, list) and ParameterType.BOOLEAN in parameter_type
    ):
        return BooleanParameter(
            type=ParameterType.BOOLEAN,
            description=description,
            enum=enum_values,
            default=default
        )
    
    elif parameter_type == ParameterType.ARRAY or (
        isinstance(parameter_type, list) and ParameterType.ARRAY in parameter_type
    ):
        # Handle array types
        items_schema = None
        args = get_args(param_type)
        if args:
            item_type = args[0]
            # Create a schema for the array items
            items_param = build_parameter_schema(
                f"{param_name}_item", item_type, None, None, True
            )
            # Convert parameter to dict for items
            items_schema = items_param.model_dump(exclude_none=True)
        
        return ArrayParameter(
            type=ParameterType.ARRAY,
            description=description,
            enum=enum_values,
            default=default,
            items=items_schema or {},
            min_items=None,
            max_items=None,
            unique_items=None
        )
    
    elif parameter_type == ParameterType.OBJECT or (
        isinstance(parameter_type, list) and ParameterType.OBJECT in parameter_type
    ):
        # Handle object types
        properties = {}
        required_props = []
        additional_properties = None
        
        # Handle Dict types
        args = get_args(param_type)
        if len(args) >= 2:
            # For Dict[str, ValueType], we need the ValueType
            value_type = args[1]
            # If the value type is a Pydantic model, get its schema
            if inspect.isclass(value_type) and issubclass(value_type, BaseModel):
                value_param = build_parameter_schema(
                    f"{param_name}_value", value_type, None, None, True
                )
                additional_properties = value_param.model_dump(exclude_none=True)
        
        # Handle Pydantic models
        if inspect.isclass(param_type) and issubclass(param_type, BaseModel):
            model_schema = param_type.model_json_schema()
            if "properties" in model_schema:
                properties = model_schema["properties"]
            if "required" in model_schema:
                required_props = model_schema["required"]
        
        return ObjectParameter(
            type=ParameterType.OBJECT,
            description=description,
            enum=enum_values,
            default=default,
            properties=properties,
            required=required_props or None,
            additional_properties=additional_properties
        )
    
    # Default fallback for any other type
    return ParameterDefinition(
        type=parameter_type,
        description=description,
        enum=enum_values,
        default=default
    )


def build_function_parameters(func: Callable) -> FunctionParameters:
    """
    Build function parameters schema from a Python function's type hints.
    
    Args:
        func: The Python function to analyze.
        
    Returns:
        FunctionParameters: Function parameters compatible with OpenRouter API.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    
    properties = {}
    required = []
    
    # Extract parameter descriptions from docstring if available
    param_descriptions = {}
    if func.__doc__:
        doc_lines = func.__doc__.split("\n")
        in_args_section = False
        current_param = None
        
        for line in doc_lines:
            line = line.strip()
            # Check if we're in the Args section
            if line.startswith("Args:"):
                in_args_section = True
                continue
                
            # If we're in the Args section, look for parameter descriptions
            if in_args_section:
                # If line is empty or starts a new section, end Args section
                if not line or line.endswith(":"):
                    if not line.startswith("    ") and not line.startswith("\t"):
                        in_args_section = False
                    continue
                    
                # Check if this line defines a parameter
                if not line.startswith("    ") and ":" in line:
                    param_name = line.split(":")[0].strip()
                    param_descriptions[param_name] = line.split(":", 1)[1].strip()
                    current_param = param_name
                # If this is a continuation of the previous parameter description
                elif current_param and (line.startswith("    ") or line.startswith("\t")):
                    param_descriptions[current_param] += " " + line.strip()
    
    for name, param in sig.parameters.items():
        # Skip self/cls for methods
        if name in ("self", "cls"):
            continue
            
        param_type = hints.get(name, Any)
        description = param_descriptions.get(name)
        
        # Determine if parameter is required based on:
        # 1. Whether it has a default value (not required if it has a default)
        # 2. Whether it's an Optional type (not required if it's Optional)
        has_default = param.default != inspect.Parameter.empty
        is_optional = False
        
        # Check if it's an Optional type (Union with None)
        origin = get_origin(param_type)
        if origin is Union and type(None) in get_args(param_type):
            is_optional = True
        
        # Parameter is required if it has no default and is not Optional
        is_required = not has_default and not is_optional
        
        # Add to required list if needed
        if is_required:
            required.append(name)
        
        default = None if param.default == inspect.Parameter.empty else param.default
        
        # Build the parameter definition as a typed parameter object
        # Pass the is_required flag to build_parameter_schema
        param_obj = build_parameter_schema(
            name, param_type, description, default, is_required
        )
        
        # Convert to dictionary for the properties field
        properties[name] = param_obj.model_dump(exclude_none=True)
    
    # Construct the full parameters schema
    return FunctionParameters(
        type="object",
        properties=properties,
        required=required if required else None
    )


def build_function_definition(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> FunctionDefinition:
    """
    Create a FunctionDefinition from a Python function.
    
    Args:
        func: The Python function to convert.
        name: Optional custom name for the function (defaults to function's name).
        description: Optional function description (defaults to function's docstring).
        
    Returns:
        FunctionDefinition: A function definition for use with chat completions.
    """
    func_name = name or func.__name__
    func_description = description or (func.__doc__ or "").strip()
    parameters = build_function_parameters(func)
    
    return FunctionDefinition(
        name=func_name,
        description=func_description,
        parameters=parameters.model_dump(exclude_none=True)
    )


def build_tool_definition(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> ToolDefinition:
    """
    Create a ToolDefinition from a Python function.
    
    Args:
        func: The Python function to convert.
        name: Optional custom name for the function (defaults to function's name).
        description: Optional function description (defaults to function's docstring).
        
    Returns:
        ToolDefinition: A tool definition for use with chat completions.
    """
    function_def = build_function_definition(func, name, description)
    return ToolDefinition(type="function", function=function_def)


def build_chat_completion_tool(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> ChatCompletionTool:
    """
    Create a ChatCompletionTool from a Python function.
    
    This function creates a chat completion tool that can be passed directly to 
    the ChatCompletionRequest's 'tools' parameter.
    
    Args:
        func: The Python function to convert.
        name: Optional custom name for the function (defaults to function's name).
        description: Optional function description (defaults to function's docstring).
        
    Returns:
        ChatCompletionTool: A chat completion tool for use with chat completions API.
    """
    function_def = build_function_definition(func, name, description)
    return ChatCompletionTool(type="function", function=function_def)


def build_function_call(
    func: Callable,
    args: Dict[str, Any],
    tool_call_id: Optional[str] = None
) -> FunctionCall:
    """
    Create a structured function call from a function and arguments.
    
    Args:
        func: The Python function to call.
        args: Dictionary of arguments to pass to the function.
        tool_call_id: Optional tool call ID.
        
    Returns:
        FunctionCall: A structured function call for use with chat completions.
    """
    func_name = func.__name__
    arguments_json = json.dumps(args)
    
    return FunctionCall(
        name=func_name,
        arguments=arguments_json,
        id=tool_call_id
    )


def build_tool_call(
    func: Callable,
    args: Dict[str, Any],
    tool_call_id: str
) -> ChatCompletionToolCall:
    """
    Create a ChatCompletionToolCall from a function and arguments.
    
    Args:
        func: The Python function to call.
        args: Dictionary of arguments to pass to the function.
        tool_call_id: Unique ID for the tool call.
        
    Returns:
        ChatCompletionToolCall: A tool call for use in chat completion responses.
    """
    func_name = func.__name__
    arguments_json = json.dumps(args)
    
    function = ToolCallFunction(
        name=func_name,
        arguments=arguments_json
    )
    
    return ChatCompletionToolCall(
        id=tool_call_id,
        type="function",
        function=function
    )


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Callable[[Callable], Callable]:
    """
    Decorator to convert a Python function into a typed tool.
    
    The decorated function will have three additional attributes:
    - as_function_definition: A FunctionDefinition object for the function
    - as_tool_definition: A ToolDefinition object for the function
    - as_chat_completion_tool: A ChatCompletionTool object for the function
    
    Args:
        name: Optional custom name for the tool (defaults to function's name).
        description: Optional tool description (defaults to function's docstring).
        
    Returns:
        Callable: Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        func_def = build_function_definition(func, name, description)
        tool_def = ToolDefinition(type="function", function=func_def)
        chat_tool = ChatCompletionTool(type="function", function=func_def)
        
        # Attach the definitions to the function
        func.as_function_definition = func_def  # type: ignore
        func.as_tool_definition = tool_def  # type: ignore
        func.as_chat_completion_tool = chat_tool  # type: ignore
        
        return func
    
    return decorator


# Functions for creating tool/function definitions and requests from Python built-in types

def create_parameter_schema_from_value(value: Any, name: str = "", description: Optional[str] = None) -> ParameterDefinition:
    """
    Create a parameter schema from a Python value, inferring the type.
    
    Args:
        value: The Python value to create a parameter schema from.
        name: Optional name for the parameter.
        description: Optional description for the parameter.
        
    Returns:
        ParameterDefinition: A parameter definition based on the value's type.
    """
    if value is None:
        return ParameterDefinition(
            type=ParameterType.NULL,
            description=description,
            enum=None,
            default=None
        )
    
    if isinstance(value, str):
        return StringParameter(
            type=ParameterType.STRING,
            description=description,
            default=value,
            enum=None,
            min_length=None,
            max_length=None,
            pattern=None,
            format=None
        )
    
    if isinstance(value, bool):
        return BooleanParameter(
            type=ParameterType.BOOLEAN,
            description=description,
            enum=None,
            default=value
        )
    
    if isinstance(value, int):
        return NumberParameter(
            type=ParameterType.INTEGER,
            description=description,
            default=value,
            enum=None,
            minimum=None,
            maximum=None,
            exclusive_minimum=None,
            exclusive_maximum=None,
            multiple_of=None
        )
    
    if isinstance(value, float):
        return NumberParameter(
            type=ParameterType.NUMBER,
            description=description,
            default=value,
            enum=None,
            minimum=None,
            maximum=None,
            exclusive_minimum=None,
            exclusive_maximum=None,
            multiple_of=None
        )
    
    if isinstance(value, list):
        # Handle empty lists
        if not value:
            return ArrayParameter(
                type=ParameterType.ARRAY,
                description=description,
                enum=None,
                default=[],
                items={},  # Empty schema for items
                min_items=None,
                max_items=None,
                unique_items=None
            )
        
        # Infer item type from the first element
        item_param = create_parameter_schema_from_value(
            value[0], f"{name}_item" if name else "item"
        )
        
        return ArrayParameter(
            type=ParameterType.ARRAY,
            description=description,
            enum=None,
            default=value,
            items=item_param.model_dump(exclude_none=True),
            min_items=None,
            max_items=None,
            unique_items=None
        )
    
    if isinstance(value, dict):
        properties = {}
        for k, v in value.items():
            param = create_parameter_schema_from_value(
                v, k, None  # No description for nested properties
            )
            properties[k] = param.model_dump(exclude_none=True)
        
        return ObjectParameter(
            type=ParameterType.OBJECT,
            description=description,
            enum=None,
            default=value,
            properties=properties,
            required=list(value.keys()) if value else None,
            additional_properties=None
        )
    
    # Default fallback for any other type
    return ParameterDefinition(
        type=ParameterType.STRING,  # Default to string for unknown types
        description=description,
        enum=None,
        default=str(value)
    )


def create_function_definition_from_dict(
    name: str,
    parameters: Dict[str, Any],
    description: Optional[str] = None,
    required_params: Optional[List[str]] = None
) -> FunctionDefinition:
    """
    Create a function definition from a dictionary of parameters.
    
    Args:
        name: Name of the function.
        parameters: Dictionary of parameter names to example values or parameter definitions.
        description: Optional description of the function.
        required_params: Optional list of required parameter names. If not provided,
            all parameters are considered required.
            
    Returns:
        FunctionDefinition: A function definition based on the provided parameters.
    """
    properties = {}
    
    for param_name, param_value in parameters.items():
        # If the value is already a parameter definition, use it directly
        if isinstance(param_value, ParameterDefinition):
            properties[param_name] = param_value.model_dump(exclude_none=True)
        else:
            # Otherwise, infer the parameter type from the value
            param_def = create_parameter_schema_from_value(param_value, param_name)
            properties[param_name] = param_def.model_dump(exclude_none=True)
    
    # If required_params is not provided, consider all parameters required
    if required_params is None:
        required_params = list(parameters.keys())
    
    # Create the parameters schema
    params_schema = {
        "type": "object",
        "properties": properties,
    }
    
    if required_params:
        params_schema["required"] = required_params
    
    return FunctionDefinition(
        name=name,
        description=description,
        parameters=params_schema
    )


def create_tool_definition_from_dict(
    name: str,
    parameters: Dict[str, Any],
    description: Optional[str] = None,
    required_params: Optional[List[str]] = None
) -> ToolDefinition:
    """
    Create a tool definition from a dictionary of parameters.
    
    Args:
        name: Name of the function.
        parameters: Dictionary of parameter names to example values or parameter definitions.
        description: Optional description of the function.
        required_params: Optional list of required parameter names. If not provided,
            all parameters are considered required.
            
    Returns:
        ToolDefinition: A tool definition based on the provided parameters.
    """
    function_def = create_function_definition_from_dict(
        name, parameters, description, required_params
    )
    
    return ToolDefinition(
        type="function",
        function=function_def
    )


def create_chat_completion_tool_from_dict(
    name: str,
    parameters: Dict[str, Any],
    description: Optional[str] = None,
    required_params: Optional[List[str]] = None
) -> ChatCompletionTool:
    """
    Create a chat completion tool from a dictionary of parameters.
    
    Args:
        name: Name of the function.
        parameters: Dictionary of parameter names to example values or parameter definitions.
        description: Optional description of the function.
        required_params: Optional list of required parameter names. If not provided,
            all parameters are considered required.
            
    Returns:
        ChatCompletionTool: A chat completion tool based on the provided parameters.
    """
    function_def = create_function_definition_from_dict(
        name, parameters, description, required_params
    )
    
    return ChatCompletionTool(
        type="function",
        function=function_def
    )


def create_function_call_from_dict(
    name: str,
    arguments: Dict[str, Any],
    call_id: Optional[str] = None
) -> FunctionCall:
    """
    Create a function call from a dictionary of arguments.
    
    Args:
        name: Name of the function to call.
        arguments: Dictionary of arguments to pass to the function.
        call_id: Optional ID for the function call.
        
    Returns:
        FunctionCall: A function call based on the provided arguments.
    """
    arguments_json = json.dumps(arguments)
    
    return FunctionCall(
        name=name,
        arguments=arguments_json,
        id=call_id
    )


def create_tool_call_from_dict(
    name: str,
    arguments: Dict[str, Any],
    tool_call_id: str
) -> ChatCompletionToolCall:
    """
    Create a tool call from a dictionary of arguments.
    
    Args:
        name: Name of the function to call.
        arguments: Dictionary of arguments to pass to the function.
        tool_call_id: Unique ID for the tool call.
        
    Returns:
        ChatCompletionToolCall: A tool call based on the provided arguments.
    """
    arguments_json = json.dumps(arguments)
    
    function = ToolCallFunction(
        name=name,
        arguments=arguments_json
    )
    
    return ChatCompletionToolCall(
        id=tool_call_id,
        type="function",
        function=function
    )


# Prompt caching helper functions

def cache_control(type: Optional[str] = "ephemeral") -> CacheControl:
    """
    Create a cache control object for prompt caching.
    
    OpenRouter provides prompt caching to reduce token costs. When the same content is sent
    multiple times, the cached content is not counted towards token usage costs after the
    first request. This is automatically applied for OpenAI models with prompts > 1024 tokens,
    and can be explicitly marked for other providers using cache_control.
    
    Args:
        type: Cache control type, currently only "ephemeral" is supported.
            
    Returns:
        CacheControl: A cache control object that can be used with TextContent.
        
    Example:
        >>> cached_content = create_cached_content("This is cached content", cache_control())
    """
    # Use default value if None is passed
    if type is None:
        type = "ephemeral"
    return CacheControl(type=type)


def create_cached_content(text: str, cache_control_obj: Optional[CacheControl] = None) -> TextContent:
    """
    Create a cacheable text content part for multimodal messages.
    
    This helper function creates a TextContent object with optional cache control,
    which can be used to mark specific parts of a message for caching. Cached content
    is only charged once, even if sent in multiple requests.
    
    Args:
        text: The text content to cache.
        cache_control_obj: Optional cache control settings. If None, creates a default one.
            
    Returns:
        TextContent: A text content object with cache control.
        
    Example:
        >>> # In a message with multiple content parts
        >>> message = {
        >>>     "role": "user",
        >>>     "content": [
        >>>         {"type": "text", "text": "Here is a document:"},
        >>>         create_cached_content("Very long document content..."),
        >>>         {"type": "text", "text": "Please summarize it."}
        >>>     ]
        >>> }
    """
    if cache_control_obj is None:
        cache_control_obj = cache_control()
        
    return TextContent(
        type="text",
        text=text,
        cache_control=cache_control_obj
    )


def string_param_with_cache_control(description: Optional[str] = None, required: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Create a string parameter that includes cache control information.
    
    This helper function creates documentation for string parameters that might use
    cache control, which is helpful for functions that process content with caching.
    
    Args:
        description: Description of the parameter.
        required: Whether the parameter is required.
        **kwargs: Additional keyword arguments for the string parameter.
            
    Returns:
        Dict[str, Any]: A dictionary with parameter definition including cache control info.
        
    Example:
        >>> # When creating tools that process potentially cached content
        >>> summarize_tool = create_tool(
        >>>     name="summarize_document",
        >>>     description="Summarize a document",
        >>>     parameters={
        >>>         "properties": {
        >>>             "document": string_param_with_cache_control(
        >>>                 description="The document to summarize. Can use cache_control.", 
        >>>                 required=True
        >>>             ),
        >>>         },
        >>>         "required": ["document"],
        >>>     }
        >>> )
    """
    if description and "cache control" not in description.lower() and "cache_control" not in description.lower():
        description += " This parameter supports cache_control for cost-efficient reuse of content."
        
    param = StringParameter(
        type=ParameterType.STRING,
        description=description,
        **kwargs
    )
    
    return param.model_dump(exclude_none=True)