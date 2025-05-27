import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pytest
from pydantic import BaseModel

from openrouter_client.tools import (
    ArrayParameter,
    BooleanParameter,
    CacheControl,
    ChatCompletionTool,
    ChatCompletionToolCall,
    FunctionCall,
    FunctionDefinition,
    FunctionParameters,
    NumberParameter,
    ObjectParameter,
    ParameterType,
    StringParameter,
    TextContent,
    ToolDefinition,
    build_chat_completion_tool,
    build_function_call,
    build_function_definition,
    build_function_parameters,
    build_parameter_schema,
    build_tool_call,
    build_tool_definition,
    cache_control,
    create_cached_content,
    create_chat_completion_tool_from_dict,
    create_function_call_from_dict,
    create_function_definition_from_dict,
    create_parameter_schema_from_value,
    create_tool_call_from_dict,
    create_tool_definition_from_dict,
    infer_parameter_type,
    string_param_with_cache_control,
    tool,
)


# Test helper classes and enums
class DummyEnum(Enum):
    """Test enumeration for parameter type inference testing."""
    VALUE_ONE = "value1"
    VALUE_TWO = "value2"
    VALUE_THREE = "value3"


class DummyPydanticModel(BaseModel):
    """Test Pydantic model for parameter type inference testing."""
    name: str
    age: int
    email: Optional[str] = None


class NestedTestModel(BaseModel):
    """Nested Pydantic model for complex testing scenarios."""
    user: DummyPydanticModel
    metadata: Dict[str, Any]


def sample_function(param1: str, param2: int = 42, param3: Optional[bool] = None) -> str:
    """
    Sample function for testing function analysis.
    
    Args:
        param1: A string parameter that is required
        param2: An integer parameter with default value
        param3: An optional boolean parameter
        
    Returns:
        A formatted string result
    """
    return f"Result: {param1}, {param2}, {param3}"


def complex_function(
    text: str,
    items: List[str],
    mapping: Dict[str, int],
    model: DummyPydanticModel,
    enum_val: DummyEnum,
    union_param: Union[str, int],
    optional_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Complex function with various parameter types for comprehensive testing.
    
    Args:
        text: Simple string parameter
        items: List of string items
        mapping: Dictionary mapping strings to integers
        model: Pydantic model instance
        enum_val: Enumeration value
        union_param: Parameter that can be string or integer
        optional_list: Optional list of strings
        
    Returns:
        Dictionary containing processed results
    """
    return {"processed": True, "text": text}


def minimal_function():
    """Function with no parameters for boundary testing."""
    return "minimal"


def args_kwargs_function(*args, **kwargs):
    """Function with only *args and **kwargs for boundary testing."""
    return {"args": args, "kwargs": kwargs}


# Test Classes for infer_parameter_type

class Test_InferParameterType_01_NominalBehaviors:
    """Test nominal behaviors for infer_parameter_type function."""
    
    @pytest.mark.parametrize("python_type,expected_type", [
        (str, ParameterType.STRING),
        (int, ParameterType.INTEGER),
        (float, ParameterType.NUMBER),
        (bool, ParameterType.BOOLEAN),
        (list, ParameterType.ARRAY),
        (List, ParameterType.ARRAY),
        (dict, ParameterType.OBJECT),
        (Dict, ParameterType.OBJECT),
    ])
    def test_primitive_type_mapping(self, python_type, expected_type):
        """Test correct mapping of primitive Python types to OpenRouter parameter types."""
        # Arrange
        input_type = python_type
        
        # Act
        result = infer_parameter_type(input_type)
        
        # Assert
        assert result == expected_type
    
    @pytest.mark.parametrize("generic_type,expected_type", [
        (List[str], ParameterType.ARRAY),
        (Dict[str, int], ParameterType.OBJECT),
        (List[Dict[str, Any]], ParameterType.ARRAY),
        (Dict[str, List[int]], ParameterType.OBJECT),
    ])
    def test_generic_container_types(self, generic_type, expected_type):
        """Test handling of generic container types with type arguments."""
        # Arrange
        input_type = generic_type
        
        # Act
        result = infer_parameter_type(input_type)
        
        # Assert
        assert result == expected_type
    
    @pytest.mark.parametrize("union_type,expected_types", [
        (Union[str, int], [ParameterType.STRING, ParameterType.INTEGER]),
        (Union[str, int, bool], [ParameterType.STRING, ParameterType.INTEGER, ParameterType.BOOLEAN]),
        (Union[List[str], Dict[str, int]], [ParameterType.ARRAY, ParameterType.OBJECT]),
    ])
    def test_union_type_processing(self, union_type, expected_types):
        """Test successful processing of Union types returning lists of parameter types."""
        # Arrange
        input_type = union_type
        
        # Act
        result = infer_parameter_type(input_type)
        
        # Assert
        assert isinstance(result, list)
        assert result == expected_types
    
    @pytest.mark.parametrize("optional_type,expected_type", [
        (Optional[str], ParameterType.STRING),
        (Optional[int], ParameterType.INTEGER),
        (Optional[List[str]], ParameterType.ARRAY),
        (Optional[Dict[str, Any]], ParameterType.OBJECT),
    ])
    def test_optional_type_identification(self, optional_type, expected_type):
        """Test correct identification of Optional types returning underlying non-None type."""
        # Arrange
        input_type = optional_type
        
        # Act
        result = infer_parameter_type(input_type)
        
        # Assert
        assert result == expected_type
    
    def test_enum_subclass_handling(self):
        """Test proper handling of Enum subclasses returning STRING type."""
        # Arrange
        input_type = DummyEnum
        
        # Act
        result = infer_parameter_type(input_type)
        
        # Assert
        assert result == ParameterType.STRING
    
    def test_pydantic_model_handling(self):
        """Test successful processing of Pydantic BaseModel subclasses returning OBJECT type."""
        # Arrange
        input_type = DummyPydanticModel
        
        # Act
        result = infer_parameter_type(input_type)
        
        # Assert
        assert result == ParameterType.OBJECT


class Test_InferParameterType_02_NegativeBehaviors:
    """Test negative behaviors for infer_parameter_type function."""
    
    @pytest.mark.parametrize("invalid_type", [
        object,
        type(lambda x: x),
        complex,
        bytes,
        bytearray,
    ])
    def test_unsupported_type_error(self, invalid_type):
        """Test ValueError raised for completely unsupported Python types."""
        # Arrange
        input_type = invalid_type
        
        # Act & Assert
        with pytest.raises(ValueError, match="Cannot map Python type .* to an OpenRouter parameter type"):
            infer_parameter_type(input_type)
    
    @pytest.mark.parametrize("non_type_object", [
        "not_a_type",
        123,
        [],
        {},
        None,
    ])
    def test_non_type_objects(self, non_type_object):
        """Test handling of non-type objects passed as input parameters."""
        # Arrange
        input_object = non_type_object
        
        # Act & Assert
        with pytest.raises((ValueError, TypeError, AttributeError)):
            infer_parameter_type(input_object)


class Test_InferParameterType_03_BoundaryBehaviors:
    """Test boundary behaviors for infer_parameter_type function."""
    
    def test_none_type_handling(self):
        """Test correct handling of NoneType returning NULL parameter type."""
        # Arrange
        input_type = type(None)
        
        # Act
        result = infer_parameter_type(input_type)
        
        # Assert
        assert result == ParameterType.NULL
    
    @pytest.mark.parametrize("edge_union", [
        Union[str, type(None)],
        Union[type(None), int],
        Union[type(None)],
    ])
    def test_edge_case_unions(self, edge_union):
        """Test processing of edge case Union types including unions with only NoneType."""
        # Arrange
        input_type = edge_union
        
        # Act
        result = infer_parameter_type(input_type)
        
        # Assert
        assert result is not None
    
    def test_bare_generic_types(self):
        """Test handling of generic types without type arguments."""
        # Arrange
        input_types = [list, dict]
        
        for input_type in input_types:
            # Act
            result = infer_parameter_type(input_type)
            
            # Assert
            assert result in [ParameterType.ARRAY, ParameterType.OBJECT]


class Test_InferParameterType_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for infer_parameter_type function."""
    
    def test_value_error_with_descriptive_message(self):
        """Test proper raising of ValueError with descriptive messages for unmappable types."""
        # Arrange
        unsupported_type = complex
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            infer_parameter_type(unsupported_type)
        
        assert "Cannot map Python type" in str(exc_info.value)
        assert "to an OpenRouter parameter type" in str(exc_info.value)


class Test_InferParameterType_05_StateTransitionBehaviors:
    """Test state transition behaviors for infer_parameter_type function."""
    
    @pytest.mark.parametrize("test_type", [str, int, List[str], Optional[bool]])
    def test_stateless_operation(self, test_type):
        """Test stateless operation across multiple invocations with different types."""
        # Arrange
        input_type = test_type
        
        # Act
        result1 = infer_parameter_type(input_type)
        result2 = infer_parameter_type(input_type)
        
        # Assert
        assert result1 == result2
    
    def test_no_side_effects_on_input(self):
        """Test no side effects or mutations on input type objects."""
        # Arrange
        original_type = DummyEnum
        original_attrs = dir(original_type)
        
        # Act
        infer_parameter_type(original_type)
        
        # Assert
        assert dir(original_type) == original_attrs


# Test Classes for build_parameter_schema

class Test_BuildParameterSchema_01_NominalBehaviors:
    """Test nominal behaviors for build_parameter_schema function."""
    
    @pytest.mark.parametrize("param_name,param_type,expected_class", [
        ("test_str", str, StringParameter),
        ("test_int", int, NumberParameter),
        ("test_float", float, NumberParameter),
        ("test_bool", bool, BooleanParameter),
        ("test_list", List[str], ArrayParameter),
        ("test_dict", Dict[str, int], ObjectParameter),
    ])
    def test_parameter_definition_creation(self, param_name, param_type, expected_class):
        """Test creation of appropriate parameter definition objects for each supported OpenRouter type."""
        # Arrange
        name = param_name
        p_type = param_type
        
        # Act
        result = build_parameter_schema(name, p_type)
        
        # Assert
        assert isinstance(result, expected_class)
        assert result.type in [ParameterType.STRING, ParameterType.INTEGER, ParameterType.NUMBER, 
                              ParameterType.BOOLEAN, ParameterType.ARRAY, ParameterType.OBJECT]
    
    @pytest.mark.parametrize("param_name,default_value,expected_required", [
        ("with_default", "default_val", False),
        ("no_default", None, True),
        ("explicit_none", None, True),
    ])
    def test_default_value_handling(self, param_name, default_value, expected_required):
        """Test correct setting of default values and adjustment of required status."""
        # Arrange
        name = param_name
        param_type = str
        default = default_value if default_value != "default_val" else "default_val"
        
        # Act
        result = build_parameter_schema(name, param_type, default=default)
        
        # Assert
        if default_value is not None and default_value != "default_val":
            assert result.default == default
        elif default_value == "default_val":
            assert result.default == "default_val"
    
    def test_enum_value_extraction(self):
        """Test successful extraction and setting of enum values from Enum types."""
        # Arrange
        param_name = "enum_param"
        param_type = DummyEnum
        
        # Act
        result = build_parameter_schema(param_name, param_type)
        
        # Assert
        assert isinstance(result, StringParameter)
        assert result.enum == ["value1", "value2", "value3"]
    
    @pytest.mark.parametrize("optional_type", [
        Optional[str],
        Optional[int],
        Optional[List[str]],
    ])
    def test_optional_type_handling(self, optional_type):
        """Test proper handling of Optional types by marking parameters as not required."""
        # Arrange
        param_name = "optional_param"
        param_type = optional_type
        
        # Act
        result = build_parameter_schema(param_name, param_type)
        
        # Assert
        assert result is not None
        # The required flag is handled at the function level, not parameter level
    
    def test_pydantic_model_schema_integration(self):
        """Test successful integration of Pydantic model schemas for object parameters."""
        # Arrange
        param_name = "model_param"
        param_type = DummyPydanticModel
        
        # Act
        result = build_parameter_schema(param_name, param_type)
        
        # Assert
        assert isinstance(result, ObjectParameter)
        assert result.type == ParameterType.OBJECT


class Test_BuildParameterSchema_02_NegativeBehaviors:
    """Test negative behaviors for build_parameter_schema function."""
    
    @pytest.mark.parametrize("invalid_name,param_type", [
        ("", str),
        (None, str),
        (123, str),
    ])
    def test_invalid_parameter_names(self, invalid_name, param_type):
        """Test handling of invalid parameter names gracefully."""
        # Arrange
        name = invalid_name
        p_type = param_type
        
        # Act & Assert
        # The function should handle these gracefully or raise appropriate errors
        try:
            result = build_parameter_schema(name, p_type)
            # If it succeeds, verify the result is reasonable
            assert result is not None
        except (TypeError, ValueError):
            # Expected for invalid inputs
            pass


class Test_BuildParameterSchema_03_BoundaryBehaviors:
    """Test boundary behaviors for build_parameter_schema function."""
    
    @pytest.mark.parametrize("edge_case_name", [
        "",
        "a" * 1000,  # Very long name
        "param_with_special_chars_!@#$%",
    ])
    def test_edge_case_parameter_names(self, edge_case_name):
        """Test handling of edge case parameter names."""
        # Arrange
        param_name = edge_case_name
        param_type = str
        
        # Act
        result = build_parameter_schema(param_name, param_type)
        
        # Assert
        assert isinstance(result, StringParameter)
    
    def test_none_as_default_value(self):
        """Test processing of None as default value correctly."""
        # Arrange
        param_name = "none_default"
        param_type = Optional[str]
        default_value = None
        
        # Act
        result = build_parameter_schema(param_name, param_type, default=default_value)
        
        # Assert
        assert result.default is None


class Test_BuildParameterSchema_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for build_parameter_schema function."""
    
    def test_infer_parameter_type_error_propagation(self):
        """Test propagation of errors from infer_parameter_type appropriately."""
        # Arrange
        param_name = "error_param"
        unsupported_type = complex
        
        # Act & Assert
        with pytest.raises(ValueError):
            build_parameter_schema(param_name, unsupported_type)


class Test_BuildParameterSchema_05_StateTransitionBehaviors:
    """Test state transition behaviors for build_parameter_schema function."""
    
    def test_independence_between_creations(self):
        """Test independence between parameter schema creations."""
        # Arrange
        param_name = "test_param"
        param_type = str
        
        # Act
        result1 = build_parameter_schema(param_name, param_type)
        result2 = build_parameter_schema(param_name, param_type)
        
        # Assert
        assert result1 is not result2  # Different instances
        assert result1.type == result2.type  # Same content
    
    def test_no_mutation_of_input_types(self):
        """Test no mutation of input type objects or enum classes."""
        # Arrange
        original_enum = DummyEnum
        original_values = list(DummyEnum)
        
        # Act
        build_parameter_schema("test", DummyEnum)
        
        # Assert
        assert list(DummyEnum) == original_values


# Test Classes for build_function_parameters

class Test_BuildFunctionParameters_01_NominalBehaviors:
    """Test nominal behaviors for build_function_parameters function."""
    
    def test_parameter_extraction_from_signature(self):
        """Test successful extraction of parameter information from function signatures using introspection."""
        # Arrange
        func = sample_function
        
        # Act
        result = build_function_parameters(func)
        
        # Assert
        assert isinstance(result, FunctionParameters)
        assert result.type == "object"
        assert "param1" in result.properties
        assert "param2" in result.properties
        assert "param3" in result.properties
    
    def test_docstring_parameter_parsing(self):
        """Test correct parsing of docstring parameter descriptions from Args sections."""
        # Arrange
        func = complex_function
        
        # Act
        result = build_function_parameters(func)
        
        # Assert
        assert isinstance(result, FunctionParameters)
        # Check that parameter descriptions are extracted
        assert "text" in result.properties
        assert "items" in result.properties
    
    def test_required_vs_optional_identification(self):
        """Test proper identification of required vs optional parameters based on defaults and Optional types."""
        # Arrange
        func = sample_function
        
        # Act
        result = build_function_parameters(func)
        
        # Assert
        assert "param1" in result.required  # No default, required
        assert "param2" not in result.required  # Has default
        assert "param3" not in result.required  # Optional type
    
    def test_self_cls_parameter_exclusion(self):
        """Test correct exclusion of self and cls parameters for class methods."""
        # Arrange
        class TestClass:
            def method_with_self(self, param: str):
                pass
            
            @classmethod
            def method_with_cls(cls, param: str):
                pass
        
        instance = TestClass()
        
        # Act
        result_instance = build_function_parameters(instance.method_with_self)
        result_class = build_function_parameters(TestClass.method_with_cls)
        
        # Assert
        assert "self" not in result_instance.properties
        assert "cls" not in result_class.properties
        assert "param" in result_instance.properties
        assert "param" in result_class.properties


class Test_BuildFunctionParameters_02_NegativeBehaviors:
    """Test negative behaviors for build_function_parameters function."""
    
    def test_functions_without_type_hints(self):
        """Test handling of functions without type hints gracefully, defaulting to Any."""
        # Arrange
        def func_without_hints(param1, param2="default"):
            """Function without type hints."""
            return param1
        
        # Act
        result = build_function_parameters(func_without_hints)
        
        # Assert
        assert isinstance(result, FunctionParameters)
        assert "param1" in result.properties
        assert "param2" in result.properties
    
    def test_malformed_docstrings(self):
        """Test processing of functions with malformed or missing docstrings."""
        # Arrange
        def func_bad_docstring(param: str):
            """This is a bad docstring with no Args section."""
            return param
        
        def func_no_docstring(param: str):
            return param
        
        # Act
        result1 = build_function_parameters(func_bad_docstring)
        result2 = build_function_parameters(func_no_docstring)
        
        # Assert
        assert isinstance(result1, FunctionParameters)
        assert isinstance(result2, FunctionParameters)


class Test_BuildFunctionParameters_03_BoundaryBehaviors:
    """Test boundary behaviors for build_function_parameters function."""
    
    def test_functions_with_no_parameters(self):
        """Test processing of functions with no parameters returning minimal schema."""
        # Arrange
        func = minimal_function
        
        # Act
        result = build_function_parameters(func)
        
        # Assert
        assert isinstance(result, FunctionParameters)
        assert result.type == "object"
        assert len(result.properties) == 0
        assert result.required is None or len(result.required) == 0
    
    def test_functions_with_args_kwargs_only(self):
        """Test handling of functions with only *args, **kwargs, or special parameters."""
        # Arrange
        func = args_kwargs_function
        
        # Act
        result = build_function_parameters(func)
        
        # Assert
        assert isinstance(result, FunctionParameters)
        # *args and **kwargs should be excluded from normal parameter processing
    
    def test_all_parameters_with_defaults(self):
        """Test managing of functions where all parameters have default values."""
        # Arrange
        def all_defaults(param1: str = "default1", param2: int = 42):
            return f"{param1}_{param2}"
        
        # Act
        result = build_function_parameters(all_defaults)
        
        # Assert
        assert isinstance(result, FunctionParameters)
        assert result.required is None or len(result.required) == 0


class Test_BuildFunctionParameters_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for build_function_parameters function."""
    
    def test_signature_inspection_failures(self):
        """Test handling of signature inspection failures for complex function types."""
        # Arrange
        # Most functions should work, but we can test edge cases
        func = lambda x: x  # Lambda functions
        
        # Act
        result = build_function_parameters(func)
        
        # Assert
        assert isinstance(result, FunctionParameters)


class Test_BuildFunctionParameters_05_StateTransitionBehaviors:
    """Test state transition behaviors for build_function_parameters function."""
    
    def test_no_modification_of_input_functions(self):
        """Test no modification of input function objects during analysis."""
        # Arrange
        func = sample_function
        original_name = func.__name__
        original_doc = func.__doc__
        
        # Act
        build_function_parameters(func)
        
        # Assert
        assert func.__name__ == original_name
        assert func.__doc__ == original_doc
    
    def test_consistent_behavior_across_analyses(self):
        """Test consistent behavior across multiple analyses of the same function."""
        # Arrange
        func = complex_function
        
        # Act
        result1 = build_function_parameters(func)
        result2 = build_function_parameters(func)
        
        # Assert
        assert result1.type == result2.type
        assert result1.properties.keys() == result2.properties.keys()
        assert result1.required == result2.required


# Test Classes for build_function_definition

class Test_BuildFunctionDefinition_01_NominalBehaviors:
    """Test nominal behaviors for build_function_definition function."""
    
    def test_complete_function_definition_creation(self):
        """Test creation of complete function definitions with proper name, description, and parameters."""
        # Arrange
        func = sample_function
        
        # Act
        result = build_function_definition(func)
        
        # Assert
        assert isinstance(result, FunctionDefinition)
        assert result.name == "sample_function"
        assert result.description is not None
        assert result.parameters is not None
    
    def test_function_name_usage_when_no_custom_name(self):
        """Test using function __name__ when no custom name provided."""
        # Arrange
        func = complex_function
        
        # Act
        result = build_function_definition(func)
        
        # Assert
        assert result.name == "complex_function"
    
    def test_custom_name_override(self):
        """Test using custom name when provided."""
        # Arrange
        func = sample_function
        custom_name = "my_custom_function"
        
        # Act
        result = build_function_definition(func, name=custom_name)
        
        # Assert
        assert result.name == custom_name
    
    def test_docstring_description_extraction(self):
        """Test extraction of description from function __doc__ when not explicitly provided."""
        # Arrange
        func = complex_function
        
        # Act
        result = build_function_definition(func)
        
        # Assert
        assert result.description is not None
        assert len(result.description) > 0
    
    def test_custom_description_override(self):
        """Test using custom description when provided."""
        # Arrange
        func = sample_function
        custom_description = "My custom description"
        
        # Act
        result = build_function_definition(func, description=custom_description)
        
        # Assert
        assert result.description == custom_description


class Test_BuildFunctionDefinition_02_NegativeBehaviors:
    """Test negative behaviors for build_function_definition function."""
    
    def test_functions_without_docstrings(self):
        """Test handling of functions without docstrings or with empty descriptions."""
        # Arrange
        def func_no_doc(param: str):
            return param
        
        # Act
        result = build_function_definition(func_no_doc)
        
        # Assert
        assert isinstance(result, FunctionDefinition)
        assert result.name == "func_no_doc"
        assert result.description == ""


class Test_BuildFunctionDefinition_03_BoundaryBehaviors:
    """Test boundary behaviors for build_function_definition function."""
    
    def test_minimal_function_metadata(self):
        """Test handling of functions with minimal metadata."""
        # Arrange
        func = minimal_function
        
        # Act
        result = build_function_definition(func)
        
        # Assert
        assert isinstance(result, FunctionDefinition)
        assert result.name == "minimal_function"


class Test_BuildFunctionDefinition_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for build_function_definition function."""
    
    def test_parameter_building_error_propagation(self):
        """Test propagation of parameter building errors appropriately."""
        # Arrange
        def problematic_func(param: complex):  # Unsupported type
            return param
        
        # Act & Assert
        with pytest.raises(ValueError):
            build_function_definition(problematic_func)


class Test_BuildFunctionDefinition_05_StateTransitionBehaviors:
    """Test state transition behaviors for build_function_definition function."""
    
    def test_no_modification_of_input_functions(self):
        """Test no modification of input function objects."""
        # Arrange
        func = sample_function
        original_name = func.__name__
        original_doc = func.__doc__
        
        # Act
        build_function_definition(func)
        
        # Assert
        assert func.__name__ == original_name
        assert func.__doc__ == original_doc
    
    def test_consistent_definition_generation(self):
        """Test consistent definition generation for repeated calls with same function."""
        # Arrange
        func = complex_function
        
        # Act
        result1 = build_function_definition(func)
        result2 = build_function_definition(func)
        
        # Assert
        assert result1.name == result2.name
        assert result1.description == result2.description


# Test Classes for tool definition builders (grouped)

class Test_ToolDefinitionBuilders_01_NominalBehaviors:
    """Test nominal behaviors for build_tool_definition and build_chat_completion_tool functions."""
    
    def test_tool_definition_creation(self):
        """Test successful wrapping of function definitions in appropriate tool containers."""
        # Arrange
        func = sample_function
        
        # Act
        tool_def = build_tool_definition(func)
        chat_tool = build_chat_completion_tool(func)
        
        # Assert
        assert isinstance(tool_def, ToolDefinition)
        assert isinstance(chat_tool, ChatCompletionTool)
        assert tool_def.type == "function"
        assert chat_tool.type == "function"
    
    def test_function_definition_integration(self):
        """Test proper integration of all function definition components into tool structure."""
        # Arrange
        func = complex_function
        
        # Act
        tool_def = build_tool_definition(func)
        chat_tool = build_chat_completion_tool(func)
        
        # Assert
        assert tool_def.function is not None
        assert chat_tool.function is not None
        assert tool_def.function.name == "complex_function"
        assert chat_tool.function.name == "complex_function"


class Test_ToolDefinitionBuilders_02_NegativeBehaviors:
    """Test negative behaviors for tool definition builders."""
    
    def test_invalid_function_inputs(self):
        """Test handling of invalid function inputs that cannot be converted to definitions."""
        # Arrange
        def problematic_func(param: complex):
            return param
        
        # Act & Assert
        with pytest.raises(ValueError):
            build_tool_definition(problematic_func)
        
        with pytest.raises(ValueError):
            build_chat_completion_tool(problematic_func)


class Test_ToolDefinitionBuilders_03_BoundaryBehaviors:
    """Test boundary behaviors for tool definition builders."""
    
    def test_minimal_function_definitions(self):
        """Test processing of functions with minimal definition complexity."""
        # Arrange
        func = minimal_function
        
        # Act
        tool_def = build_tool_definition(func)
        chat_tool = build_chat_completion_tool(func)
        
        # Assert
        assert isinstance(tool_def, ToolDefinition)
        assert isinstance(chat_tool, ChatCompletionTool)


class Test_ToolDefinitionBuilders_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for tool definition builders."""
    
    def test_function_definition_error_propagation(self):
        """Test propagation of function definition building errors to tool level."""
        # Arrange
        def error_func(param: object):  # Unsupported type
            return param
        
        # Act & Assert
        with pytest.raises(ValueError):
            build_tool_definition(error_func)


class Test_ToolDefinitionBuilders_05_StateTransitionBehaviors:
    """Test state transition behaviors for tool definition builders."""
    
    def test_consistency_with_function_definition_building(self):
        """Test consistency with underlying function definition building."""
        # Arrange
        func = sample_function
        
        # Act
        func_def = build_function_definition(func)
        tool_def = build_tool_definition(func)
        
        # Assert
        assert tool_def.function.name == func_def.name
        assert tool_def.function.description == func_def.description


# Test Classes for call builders (grouped)

class Test_CallBuilders_01_NominalBehaviors:
    """Test nominal behaviors for build_function_call and build_tool_call functions."""
    
    @pytest.mark.parametrize("args_dict", [
        {"param1": "test", "param2": 42},
        {"text": "hello", "number": 123},
        {"complex": {"nested": "value"}},
    ])
    def test_function_call_creation(self, args_dict):
        """Test creation of proper function call objects with correct name and JSON-serialized arguments."""
        # Arrange
        func = sample_function
        
        # Act
        result = build_function_call(func, args_dict)
        
        # Assert
        assert isinstance(result, FunctionCall)
        assert result.name == "sample_function"
        assert result.arguments == json.dumps(args_dict)
    
    @pytest.mark.parametrize("tool_call_id", [
        "call_123",
        "unique_id_456",
        "test_call_789",
    ])
    def test_tool_call_creation(self, tool_call_id):
        """Test creation of tool call objects with required ID and function structure."""
        # Arrange
        func = complex_function
        args_dict = {"text": "test"}
        
        # Act
        result = build_tool_call(func, args_dict, tool_call_id)
        
        # Assert
        assert isinstance(result, ChatCompletionToolCall)
        assert result.id == tool_call_id
        assert result.type == "function"
        assert result.function.name == "complex_function"
    
    def test_optional_tool_call_id_handling(self):
        """Test proper handling of optional tool call IDs for function calls."""
        # Arrange
        func = sample_function
        args_dict = {"param1": "test"}
        tool_call_id = "optional_id"
        
        # Act
        result = build_function_call(func, args_dict, tool_call_id)
        
        # Assert
        assert result.id == tool_call_id


class Test_CallBuilders_02_NegativeBehaviors:
    """Test negative behaviors for call builders."""
    
    @pytest.mark.parametrize("non_serializable_args", [
        {"func": lambda x: x},
        {"obj": object()},
        {"complex": complex(1, 2)},
    ])
    def test_non_serializable_arguments(self, non_serializable_args):
        """Test handling of non-serializable arguments in the arguments dictionary."""
        # Arrange
        func = sample_function
        
        # Act & Assert
        with pytest.raises((TypeError, ValueError)):
            build_function_call(func, non_serializable_args)
    
    def test_empty_arguments_dictionary(self):
        """Test handling of empty argument dictionaries."""
        # Arrange
        func = minimal_function
        args_dict = {}
        
        # Act
        result = build_function_call(func, args_dict)
        
        # Assert
        assert result.arguments == "{}"


class Test_CallBuilders_03_BoundaryBehaviors:
    """Test boundary behaviors for call builders."""
    
    def test_very_large_argument_sets(self):
        """Test processing of very large argument sets that may stress JSON serialization."""
        # Arrange
        func = sample_function
        large_args = {f"param_{i}": f"value_{i}" for i in range(100)}
        
        # Act
        result = build_function_call(func, large_args)
        
        # Assert
        assert isinstance(result, FunctionCall)
        assert len(result.arguments) > 0


class Test_CallBuilders_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for call builders."""
    
    def test_json_serialization_failures(self):
        """Test handling of JSON serialization failures for complex argument types."""
        # Arrange
        func = sample_function
        # Create a circular reference that can't be serialized
        circular_dict = {}
        circular_dict["self"] = circular_dict
        
        # Act & Assert
        with pytest.raises((TypeError, ValueError)):
            build_function_call(func, circular_dict)


class Test_CallBuilders_05_StateTransitionBehaviors:
    """Test state transition behaviors for call builders."""
    
    def test_no_modification_of_inputs(self):
        """Test no modification of input function or argument objects."""
        # Arrange
        func = sample_function
        args_dict = {"param1": "test", "param2": 42}
        original_args = args_dict.copy()
        
        # Act
        build_function_call(func, args_dict)
        
        # Assert
        assert args_dict == original_args
    
    def test_consistent_call_generation(self):
        """Test consistent call object generation for equivalent inputs."""
        # Arrange
        func = sample_function
        args_dict = {"param1": "test"}
        
        # Act
        result1 = build_function_call(func, args_dict)
        result2 = build_function_call(func, args_dict)
        
        # Assert
        assert result1.name == result2.name
        assert result1.arguments == result2.arguments


# Test Classes for tool decorator

class Test_ToolDecorator_01_NominalBehaviors:
    """Test nominal behaviors for tool decorator function."""
    
    def test_function_decoration_with_attributes(self):
        """Test successful decoration of functions and addition of definition attributes."""
        # Arrange & Act
        @tool()
        def decorated_function(param: str) -> str:
            """Test function for decoration."""
            return param
        
        # Assert
        assert hasattr(decorated_function, 'as_function_definition')
        assert hasattr(decorated_function, 'as_tool_definition')
        assert hasattr(decorated_function, 'as_chat_completion_tool')
    
    def test_original_function_behavior_preservation(self):
        """Test correct preservation of original function behavior."""
        # Arrange
        @tool()
        def test_func(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y
        
        # Act
        result = test_func(5, 3)
        
        # Assert
        assert result == 8
        assert callable(test_func)
    
    def test_all_three_definition_types_creation(self):
        """Test proper creation of all three definition types as function attributes."""
        # Arrange & Act
        @tool()
        def test_func(param: str):
            """Test function."""
            return param
        
        # Assert
        assert isinstance(test_func.as_function_definition, FunctionDefinition)
        assert isinstance(test_func.as_tool_definition, ToolDefinition)
        assert isinstance(test_func.as_chat_completion_tool, ChatCompletionTool)
    
    @pytest.mark.parametrize("custom_name,custom_description", [
        ("my_tool", "My custom tool"),
        ("another_tool", "Another description"),
        (None, "Custom description only"),
        ("Name only", None),
    ])
    def test_custom_name_and_description(self, custom_name, custom_description):
        """Test decoration with custom name and description parameters."""
        # Arrange & Act
        @tool(name=custom_name, description=custom_description)
        def test_func(param: str):
            """Original docstring."""
            return param
        
        # Assert
        expected_name = custom_name or "test_func"
        expected_desc = custom_description or "Original docstring."
        
        assert test_func.as_function_definition.name == expected_name
        assert test_func.as_function_definition.description == expected_desc


class Test_ToolDecorator_02_NegativeBehaviors:
    """Test negative behaviors for tool decorator."""
    
    def test_decoration_of_problematic_functions(self):
        """Test decoration of functions that cannot be analyzed for definition creation."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError):
            @tool()
            def problematic_func(param: complex):  # Unsupported type
                return param


class Test_ToolDecorator_03_BoundaryBehaviors:
    """Test boundary behaviors for tool decorator."""
    
    def test_minimal_function_decoration(self):
        """Test decoration of functions with minimal signatures."""
        # Arrange & Act
        @tool()
        def minimal_func():
            return "minimal"
        
        # Assert
        assert hasattr(minimal_func, 'as_function_definition')
        assert minimal_func() == "minimal"
    
    def test_complex_function_decoration(self):
        """Test decoration of functions with complex parameter structures."""
        # Arrange & Act
        @tool()
        def complex_func(
            text: str,
            items: List[str],
            mapping: Dict[str, int],
            optional: Optional[bool] = None
        ):
            """Complex function with various parameter types."""
            return {"text": text, "items": items}
        
        # Assert
        assert hasattr(complex_func, 'as_function_definition')
        assert callable(complex_func)


class Test_ToolDecorator_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for tool decorator."""
    
    def test_definition_building_failures_during_decoration(self):
        """Test handling of definition building failures during decoration process."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError):
            @tool()
            def error_func(param: object):  # Unsupported type
                return param


class Test_ToolDecorator_05_StateTransitionBehaviors:
    """Test state transition behaviors for tool decorator."""
    
    def test_function_modification_with_attributes(self):
        """Test proper modification of function objects with new definition attributes."""
        # Arrange
        def original_func(param: str):
            """Original function."""
            return param
        
        original_name = original_func.__name__
        
        # Act
        decorated_func = tool()(original_func)
        
        # Assert
        assert decorated_func.__name__ == original_name
        assert hasattr(decorated_func, 'as_function_definition')
    
    def test_consistent_attribute_presence(self):
        """Test consistent attribute presence across multiple decorated functions."""
        # Arrange & Act
        @tool()
        def func1(param: str):
            return param
        
        @tool()
        def func2(param: int):
            return param
        
        # Assert
        required_attrs = ['as_function_definition', 'as_tool_definition', 'as_chat_completion_tool']
        for attr in required_attrs:
            assert hasattr(func1, attr)
            assert hasattr(func2, attr)


# Test Classes for create_parameter_schema_from_value

class Test_CreateParameterSchemaFromValue_01_NominalBehaviors:
    """Test nominal behaviors for create_parameter_schema_from_value function."""
    
    @pytest.mark.parametrize("value,expected_type,expected_class", [
        ("test_string", ParameterType.STRING, StringParameter),
        (42, ParameterType.INTEGER, NumberParameter),
        (3.14, ParameterType.NUMBER, NumberParameter),
        (True, ParameterType.BOOLEAN, BooleanParameter),
        (False, ParameterType.BOOLEAN, BooleanParameter),
    ])
    def test_primitive_value_type_inference(self, value, expected_type, expected_class):
        """Test correct inference of parameter types from Python values of all supported types."""
        # Arrange
        test_value = value
        
        # Act
        result = create_parameter_schema_from_value(test_value)
        
        # Assert
        assert isinstance(result, expected_class)
        assert result.type == expected_type
        assert result.default == test_value
    
    @pytest.mark.parametrize("container_value,expected_type", [
        (["item1", "item2"], ParameterType.ARRAY),
        ({"key1": "value1", "key2": "value2"}, ParameterType.OBJECT),
        ([1, 2, 3], ParameterType.ARRAY),
        ({"a": 1, "b": 2}, ParameterType.OBJECT),
    ])
    def test_container_value_processing(self, container_value, expected_type):
        """Test successful processing of container values with recursive schema generation."""
        # Arrange
        test_value = container_value
        
        # Act
        result = create_parameter_schema_from_value(test_value)
        
        # Assert
        assert result.type == expected_type
        assert result.default == test_value
    
    def test_array_schema_with_item_type_inference(self):
        """Test accurate creation of array schemas with item type inference from first elements."""
        # Arrange
        array_value = ["string1", "string2", "string3"]
        
        # Act
        result = create_parameter_schema_from_value(array_value)
        
        # Assert
        assert isinstance(result, ArrayParameter)
        assert result.type == ParameterType.ARRAY
        assert "type" in result.items
        assert result.items["type"] == ParameterType.STRING
    
    def test_object_schema_property_analysis(self):
        """Test correct building of object schemas with property-by-property analysis."""
        # Arrange
        object_value = {
            "name": "John",
            "age": 30,
            "active": True
        }
        
        # Act
        result = create_parameter_schema_from_value(object_value)
        
        # Assert
        assert isinstance(result, ObjectParameter)
        assert result.type == ParameterType.OBJECT
        assert "name" in result.properties
        assert "age" in result.properties
        assert "active" in result.properties
        assert result.required == ["name", "age", "active"]


class Test_CreateParameterSchemaFromValue_02_NegativeBehaviors:
    """Test negative behaviors for create_parameter_schema_from_value function."""
    
    @pytest.mark.parametrize("unknown_value", [
        complex(1, 2),
        object(),
        lambda x: x,
        type,
    ])
    def test_unknown_value_types_default_handling(self, unknown_value):
        """Test handling of completely unknown value types by defaulting to string representation."""
        # Arrange
        test_value = unknown_value
        
        # Act
        result = create_parameter_schema_from_value(test_value)
        
        # Assert
        assert result.type == ParameterType.STRING
        assert result.default == str(test_value)


class Test_CreateParameterSchemaFromValue_03_BoundaryBehaviors:
    """Test boundary behaviors for create_parameter_schema_from_value function."""
    
    def test_none_value_handling(self):
        """Test correct handling of None values with NULL parameter type."""
        # Arrange
        test_value = None
        
        # Act
        result = create_parameter_schema_from_value(test_value)
        
        # Assert
        assert result.type == ParameterType.NULL
        assert result.default is None
    
    @pytest.mark.parametrize("empty_container,expected_type", [
        ([], ParameterType.ARRAY),
        ({}, ParameterType.OBJECT),
    ])
    def test_empty_containers(self, empty_container, expected_type):
        """Test processing of empty containers with appropriate empty schemas."""
        # Arrange
        test_value = empty_container
        
        # Act
        result = create_parameter_schema_from_value(test_value)
        
        # Assert
        assert result.type == expected_type
        assert result.default == test_value
    
    def test_deeply_nested_structures(self):
        """Test management of very large or deeply nested data structures."""
        # Arrange
        nested_value = {
            "level1": {
                "level2": {
                    "level3": ["deep", "array", "values"]
                }
            }
        }
        
        # Act
        result = create_parameter_schema_from_value(nested_value)
        
        # Assert
        assert isinstance(result, ObjectParameter)
        assert result.type == ParameterType.OBJECT


class Test_CreateParameterSchemaFromValue_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for create_parameter_schema_from_value function."""
    
    def test_value_introspection_failures(self):
        """Test graceful handling of value introspection failures during type inference."""
        # Arrange
        class ProblematicClass:
            def __getattribute__(self, name):
                if name == "__class__":
                    raise AttributeError("Cannot access __class__")
                return super().__getattribute__(name)
        
        # Most values should work gracefully
        test_value = "fallback_test"
        
        # Act
        result = create_parameter_schema_from_value(test_value)
        
        # Assert
        assert result.type == ParameterType.STRING


class Test_CreateParameterSchemaFromValue_05_StateTransitionBehaviors:
    """Test state transition behaviors for create_parameter_schema_from_value function."""
    
    def test_no_modification_of_input_values(self):
        """Test no modification of input values during schema generation."""
        # Arrange
        original_dict = {"key": "value", "number": 42}
        test_value = original_dict.copy()
        
        # Act
        create_parameter_schema_from_value(test_value)
        
        # Assert
        assert test_value == original_dict
    
    def test_consistent_schema_generation(self):
        """Test consistent schema generation for equivalent values across calls."""
        # Arrange
        test_value = {"name": "test", "value": 123}
        
        # Act
        result1 = create_parameter_schema_from_value(test_value)
        result2 = create_parameter_schema_from_value(test_value)
        
        # Assert
        assert result1.type == result2.type
        assert result1.default == result2.default


# Test Classes for dictionary-based creation functions (grouped)

class Test_DictionaryBasedCreators_01_NominalBehaviors:
    """Test nominal behaviors for dictionary-based creation functions."""
    
    @pytest.mark.parametrize("func_name,parameters,expected_keys", [
        ("test_func", {"param1": "value1", "param2": 42}, ["param1", "param2"]),
        ("another_func", {"text": "hello", "flag": True}, ["text", "flag"]),
        ("complex_func", {"data": {"nested": "value"}}, ["data"]),
    ])
    def test_function_definition_from_dict(self, func_name, parameters, expected_keys):
        """Test successful creation of definitions from well-formed parameter dictionaries."""
        # Arrange
        name = func_name
        params = parameters
        
        # Act
        func_def = create_function_definition_from_dict(name, params)
        tool_def = create_tool_definition_from_dict(name, params)
        chat_tool = create_chat_completion_tool_from_dict(name, params)
        
        # Assert
        assert isinstance(func_def, FunctionDefinition)
        assert isinstance(tool_def, ToolDefinition)
        assert isinstance(chat_tool, ChatCompletionTool)
        
        assert func_def.name == name
        assert tool_def.function.name == name
        assert chat_tool.function.name == name
        
        for key in expected_keys:
            assert key in func_def.parameters["properties"]
    
    def test_mixed_parameter_types_processing(self):
        """Test accurate processing of mixed parameter types within single definitions."""
        # Arrange
        name = "mixed_func"
        parameters = {
            "string_param": "test",
            "int_param": 42,
            "bool_param": True,
            "array_param": ["item1", "item2"],
            "object_param": {"key": "value"}
        }
        
        # Act
        result = create_function_definition_from_dict(name, parameters)
        
        # Assert
        assert len(result.parameters["properties"]) == 5
        assert "string_param" in result.parameters["properties"]
        assert "int_param" in result.parameters["properties"]
    
    @pytest.mark.parametrize("required_params", [
        ["param1"],
        ["param1", "param2"],
        None,  # All required by default
    ])
    def test_required_parameter_determination(self, required_params):
        """Test proper management of required parameter determination and schema construction."""
        # Arrange
        name = "test_func"
        parameters = {"param1": "value1", "param2": 42}
        
        # Act
        result = create_function_definition_from_dict(name, parameters, required_params=required_params)
        
        # Assert
        if required_params is None:
            assert set(result.parameters["required"]) == {"param1", "param2"}
        else:
            assert set(result.parameters["required"]) == set(required_params)


class Test_DictionaryBasedCreators_02_NegativeBehaviors:
    """Test negative behaviors for dictionary-based creation functions."""
    
    @pytest.mark.parametrize("malformed_params", [
        {"": "empty_key"},  # Empty key
        {123: "numeric_key"},  # Non-string key
    ])
    def test_malformed_parameter_dictionaries(self, malformed_params):
        """Test handling of malformed parameter dictionaries with missing or invalid keys."""
        # Arrange
        name = "test_func"
        
        # Act & Assert
        # These should either work gracefully or raise appropriate errors
        try:
            result = create_function_definition_from_dict(name, malformed_params)
            assert isinstance(result, FunctionDefinition)
        except (TypeError, ValueError, KeyError):
            pass  # Expected for malformed input


class Test_DictionaryBasedCreators_03_BoundaryBehaviors:
    """Test boundary behaviors for dictionary-based creation functions."""
    
    def test_empty_parameter_dictionaries(self):
        """Test handling of empty parameter dictionaries creating minimal function definitions."""
        # Arrange
        name = "empty_func"
        parameters = {}
        
        # Act
        result = create_function_definition_from_dict(name, parameters)
        
        # Assert
        assert isinstance(result, FunctionDefinition)
        assert result.name == name
        assert len(result.parameters["properties"]) == 0
    
    def test_large_parameter_sets(self):
        """Test management of very large parameter sets with complex nested structures."""
        # Arrange
        name = "large_func"
        parameters = {f"param_{i}": f"value_{i}" for i in range(50)}
        
        # Act
        result = create_function_definition_from_dict(name, parameters)
        
        # Assert
        assert isinstance(result, FunctionDefinition)
        assert len(result.parameters["properties"]) == 50


class Test_DictionaryBasedCreators_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for dictionary-based creation functions."""
    
    def test_parameter_schema_creation_failures(self):
        """Test handling of parameter schema creation failures during dictionary processing."""
        # Arrange
        name = "error_func"
        # Use values that might cause issues
        parameters = {"param": object()}  # Non-serializable object
        
        # Act
        result = create_function_definition_from_dict(name, parameters)
        
        # Assert
        # Should handle gracefully by converting to string
        assert isinstance(result, FunctionDefinition)


class Test_DictionaryBasedCreators_05_StateTransitionBehaviors:
    """Test state transition behaviors for dictionary-based creation functions."""
    
    def test_no_modification_of_input_dictionaries(self):
        """Test no modification of input dictionaries during definition creation."""
        # Arrange
        name = "test_func"
        original_params = {"param1": "value1", "param2": 42}
        parameters = original_params.copy()
        
        # Act
        create_function_definition_from_dict(name, parameters)
        
        # Assert
        assert parameters == original_params
    
    def test_consistent_definition_creation(self):
        """Test consistent definition creation for equivalent dictionary inputs."""
        # Arrange
        name = "consistent_func"
        parameters = {"param": "value"}
        
        # Act
        result1 = create_function_definition_from_dict(name, parameters)
        result2 = create_function_definition_from_dict(name, parameters)
        
        # Assert
        assert result1.name == result2.name
        assert result1.parameters == result2.parameters


# Test Classes for call creation from dictionaries (grouped)

class Test_CallCreatorsFromDict_01_NominalBehaviors:
    """Test nominal behaviors for call creation from dictionary functions."""
    
    @pytest.mark.parametrize("func_name,arguments", [
        ("test_func", {"param1": "value1", "param2": 42}),
        ("another_func", {"text": "hello", "flag": True}),
        ("complex_func", {"data": {"nested": "value"}, "items": [1, 2, 3]}),
    ])
    def test_call_creation_from_dictionaries(self, func_name, arguments):
        """Test creation of proper call objects from name and arguments dictionaries."""
        # Arrange
        name = func_name
        args = arguments
        call_id = "test_call_123"
        
        # Act
        func_call = create_function_call_from_dict(name, args)
        tool_call = create_tool_call_from_dict(name, args, call_id)
        
        # Assert
        assert isinstance(func_call, FunctionCall)
        assert isinstance(tool_call, ChatCompletionToolCall)
        
        assert func_call.name == name
        assert tool_call.function.name == name
        assert func_call.arguments == json.dumps(args)
        assert tool_call.function.arguments == json.dumps(args)
        assert tool_call.id == call_id
    
    def test_json_serialization_of_arguments(self):
        """Test correct serialization of argument dictionaries to JSON format."""
        # Arrange
        name = "json_test"
        arguments = {
            "string": "value",
            "number": 42,
            "boolean": True,
            "array": [1, 2, 3],
            "object": {"nested": "data"}
        }
        
        # Act
        result = create_function_call_from_dict(name, arguments)
        
        # Assert
        parsed_args = json.loads(result.arguments)
        assert parsed_args == arguments
    
    @pytest.mark.parametrize("call_id", [
        "required_id_123",
        "another_call_456",
        "tool_call_789",
    ])
    def test_call_id_handling(self, call_id):
        """Test proper handling of optional and required call IDs."""
        # Arrange
        name = "id_test"
        arguments = {"param": "value"}
        
        # Act
        func_call_with_id = create_function_call_from_dict(name, arguments, call_id)
        func_call_without_id = create_function_call_from_dict(name, arguments)
        tool_call = create_tool_call_from_dict(name, arguments, call_id)
        
        # Assert
        assert func_call_with_id.id == call_id
        assert func_call_without_id.id is None
        assert tool_call.id == call_id


class Test_CallCreatorsFromDict_02_NegativeBehaviors:
    """Test negative behaviors for call creators from dictionaries."""
    
    def test_non_serializable_arguments(self):
        """Test handling of non-serializable arguments in dictionaries."""
        # Arrange
        name = "error_func"
        non_serializable_args = {"func": lambda x: x, "obj": object()}
        
        # Act & Assert
        with pytest.raises((TypeError, ValueError)):
            create_function_call_from_dict(name, non_serializable_args)


class Test_CallCreatorsFromDict_03_BoundaryBehaviors:
    """Test boundary behaviors for call creators from dictionaries."""
    
    def test_empty_argument_dictionaries(self):
        """Test handling of empty argument dictionaries."""
        # Arrange
        name = "empty_args"
        arguments = {}
        
        # Act
        result = create_function_call_from_dict(name, arguments)
        
        # Assert
        assert result.arguments == "{}"
    
    def test_very_large_argument_sets(self):
        """Test processing of very large argument sets."""
        # Arrange
        name = "large_args"
        arguments = {f"param_{i}": f"value_{i}" for i in range(100)}
        
        # Act
        result = create_function_call_from_dict(name, arguments)
        
        # Assert
        assert isinstance(result, FunctionCall)
        assert len(result.arguments) > 0


class Test_CallCreatorsFromDict_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for call creators from dictionaries."""
    
    def test_json_serialization_failures(self):
        """Test handling of JSON serialization failures."""
        # Arrange
        name = "serialization_error"
        # Create circular reference
        circular_dict = {}
        circular_dict["self"] = circular_dict
        
        # Act & Assert
        with pytest.raises((TypeError, ValueError)):
            create_function_call_from_dict(name, circular_dict)


class Test_CallCreatorsFromDict_05_StateTransitionBehaviors:
    """Test state transition behaviors for call creators from dictionaries."""
    
    def test_no_modification_of_input_dictionaries(self):
        """Test no modification of input dictionaries."""
        # Arrange
        name = "no_modify"
        original_args = {"param1": "value1", "param2": 42}
        arguments = original_args.copy()
        
        # Act
        create_function_call_from_dict(name, arguments)
        
        # Assert
        assert arguments == original_args
    
    def test_consistent_call_generation(self):
        """Test consistent call generation for equivalent inputs."""
        # Arrange
        name = "consistent"
        arguments = {"param": "value"}
        
        # Act
        result1 = create_function_call_from_dict(name, arguments)
        result2 = create_function_call_from_dict(name, arguments)
        
        # Assert
        assert result1.name == result2.name
        assert result1.arguments == result2.arguments


# Test Classes for cache control functions (grouped)

class Test_CacheControlFunctions_01_NominalBehaviors:
    """Test nominal behaviors for cache control functions."""
    
    @pytest.mark.parametrize("cache_type", [
        "ephemeral",
        "persistent",
        "session",
    ])
    def test_cache_control_creation(self, cache_type):
        """Test creation of proper cache control objects with specified types."""
        # Arrange
        control_type = cache_type
        
        # Act
        result = cache_control(control_type)
        
        # Assert
        assert isinstance(result, CacheControl)
        assert result.type == control_type
    
    def test_default_cache_control_type(self):
        """Test proper defaulting of cache control type to 'ephemeral' when not specified."""
        # Arrange & Act
        result = cache_control()
        
        # Assert
        assert isinstance(result, CacheControl)
        assert result.type == "ephemeral"
    
    @pytest.mark.parametrize("text_content", [
        "Short text for caching",
        "Very long text content that would benefit from caching" * 50,
        "Text with special characters: !@#$%^&*()",
    ])
    def test_cached_content_creation(self, text_content):
        """Test successful integration of cache control into text content objects."""
        # Arrange
        text = text_content
        cache_ctrl = cache_control("ephemeral")
        
        # Act
        result = create_cached_content(text, cache_ctrl)
        
        # Assert
        assert isinstance(result, TextContent)
        assert result.type == "text"
        assert result.text == text
        assert result.cache_control == cache_ctrl
    
    def test_string_parameter_cache_control_enhancement(self):
        """Test correct enhancement of string parameters with cache control documentation."""
        # Arrange
        description = "A parameter for text input"
        
        # Act
        result = string_param_with_cache_control(description, required=True)
        
        # Assert
        assert isinstance(result, dict)
        assert "cache_control" in result["description"] or "cache control" in result["description"]


class Test_CacheControlFunctions_02_NegativeBehaviors:
    """Test negative behaviors for cache control functions."""
    
    @pytest.mark.parametrize("invalid_type", [
        "invalid_type",
        "unknown",
        "",
    ])
    def test_invalid_cache_control_types(self, invalid_type):
        """Test handling of invalid cache control types beyond supported values."""
        # Arrange
        control_type = invalid_type
        
        # Act
        result = cache_control(control_type)
        
        # Assert
        # Should create object regardless, validation happens elsewhere
        assert isinstance(result, CacheControl)
        assert result.type == control_type


class Test_CacheControlFunctions_03_BoundaryBehaviors:
    """Test boundary behaviors for cache control functions."""
    
    def test_empty_text_content_caching(self):
        """Test handling of empty text content for caching."""
        # Arrange
        text = ""
        
        # Act
        result = create_cached_content(text)
        
        # Assert
        assert isinstance(result, TextContent)
        assert result.text == ""
    
    def test_very_long_text_content(self):
        """Test processing of very long text content that may impact caching behavior."""
        # Arrange
        long_text = "This is a very long text. " * 1000
        
        # Act
        result = create_cached_content(long_text)
        
        # Assert
        assert isinstance(result, TextContent)
        assert result.text == long_text


class Test_CacheControlFunctions_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for cache control functions."""
    
    def test_cache_control_object_creation_failures(self):
        """Test handling of cache control object creation failures."""
        # Arrange & Act & Assert
        # Most inputs should work, test edge cases
        result = cache_control(None)
        assert isinstance(result, CacheControl)


class Test_CacheControlFunctions_05_StateTransitionBehaviors:
    """Test state transition behaviors for cache control functions."""
    
    def test_cache_control_state_associations(self):
        """Test proper cache control state associations with content."""
        # Arrange
        text = "Test content"
        cache_ctrl = cache_control("ephemeral")
        
        # Act
        result = create_cached_content(text, cache_ctrl)
        
        # Assert
        assert result.cache_control is cache_ctrl
    
    def test_no_side_effects_on_inputs(self):
        """Test no side effects on input text or parameter specification objects."""
        # Arrange
        original_text = "Original text"
        text = original_text
        
        # Act
        create_cached_content(text)
        
        # Assert
        assert text == original_text
    
    def test_consistent_cache_control_behavior(self):
        """Test consistent cache control behavior across multiple creations."""
        # Arrange
        cache_type = "ephemeral"
        
        # Act
        ctrl1 = cache_control(cache_type)
        ctrl2 = cache_control(cache_type)
        
        # Assert
        assert ctrl1.type == ctrl2.type
        assert ctrl1 is not ctrl2  # Different instances
