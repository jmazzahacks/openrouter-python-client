import pytest
from pydantic import ValidationError

from openrouter_client.models.core import (
    FunctionDefinition, ToolDefinition, ResponseFormat, CacheControl,
    TextContent, ImageUrl, ImageContentPart, FileData, FileContent,
    Prediction, Message
)
from openrouter_client.types import ModelRole


class Test_FunctionDefinition_01_NominalBehaviors:
    """Tests for normal, expected use cases of the FunctionDefinition class."""

    @pytest.mark.parametrize("name, description, parameters", [
        ("get_weather", "Get weather information", {"type": "object", "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        }}),
        ("search", "Search for information", {"type": "object", "properties": {
            "query": {"type": "string"}
        }}),
        ("minimal_function", None, {"type": "object"}),
    ])
    def test_create_valid_function_definition(self, name, description, parameters):
        """Test creating FunctionDefinition instances with valid parameters."""
        function = FunctionDefinition(
            name=name, 
            description=description, 
            parameters=parameters
        )
        
        assert function.name == name
        assert function.description == description
        assert function.parameters == parameters


class Test_FunctionDefinition_02_NegativeBehaviors:
    """Tests for invalid usage scenarios of the FunctionDefinition class."""

    @pytest.mark.parametrize("name, description, parameters, expected_error", [
        ("", "Some description", {"type": "object"}, "String should have at least 1 character"),
        (None, "Some description", {"type": "object"}, "none is not an allowed value"),
        ("name", "Some description", None, "none is not an allowed value"),
    ])
    def test_invalid_function_definition(self, name, description, parameters, expected_error):
        """Test that creating FunctionDefinition with invalid parameters raises appropriate errors."""
        with pytest.raises(ValidationError) as exc_info:
            FunctionDefinition(
                name=name,
                description=description,
                parameters=parameters
            )
        
        error_details = exc_info.value.errors()
        if name is None:
            assert any(e["loc"][0] == "name" for e in error_details) and "valid string" in str(exc_info.value)


class Test_FunctionDefinition_03_BoundaryBehaviors:
    """Tests for boundary conditions of the FunctionDefinition class."""

    def test_minimum_name_length(self):
        """Test that a name with exactly the minimum length (1 character) is accepted."""
        function = FunctionDefinition(
            name="a",
            parameters={"type": "object"}
        )
        
        assert function.name == "a"
        assert len(function.name) == 1


class Test_FunctionDefinition_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the FunctionDefinition class."""

    @pytest.mark.parametrize("invalid_data, expected_error_path", [
        ({"name": "", "parameters": {"type": "object"}}, "name"),
        ({"parameters": {"type": "object"}}, "name"),
        ({"name": "function"}, "parameters"),
    ])
    def test_error_message_clarity(self, invalid_data, expected_error_path):
        """Test that validation errors include clear information about the error location."""
        with pytest.raises(ValidationError) as exc_info:
            FunctionDefinition(**invalid_data)
        
        error_str = str(exc_info.value)
        assert expected_error_path in error_str


class Test_ToolDefinition_01_NominalBehaviors:
    """Tests for normal, expected use cases of the ToolDefinition class."""

    def test_create_with_valid_function(self):
        """Test creating a ToolDefinition with a valid function definition."""
        function = FunctionDefinition(
            name="test_function",
            description="A test function",
            parameters={"type": "object", "properties": {}}
        )
        
        tool = ToolDefinition(function=function)
        
        assert tool.type == "function"  # Default value
        assert tool.function == function
    
    def test_explicit_type_specification(self):
        """Test creating a ToolDefinition with explicit type specification."""
        function = FunctionDefinition(
            name="test_function",
            parameters={"type": "object"}
        )
        
        tool = ToolDefinition(type="function", function=function)
        
        assert tool.type == "function"
        assert tool.function == function


class Test_ToolDefinition_02_NegativeBehaviors:
    """Tests for invalid usage scenarios of the ToolDefinition class."""

    def test_missing_function(self):
        """Test that creating a ToolDefinition without a function raises an error."""
        with pytest.raises(ValidationError) as exc_info:
            ToolDefinition()
        
        assert "function" in str(exc_info.value)
    
    def test_invalid_function_type(self):
        """Test that providing an invalid type for function raises an error."""
        with pytest.raises(ValidationError):
            ToolDefinition(function="not a function definition")


class Test_ToolDefinition_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ToolDefinition class."""

    def test_error_message_clarity(self):
        """Test that validation errors include clear information about the error location."""
        with pytest.raises(ValidationError) as exc_info:
            ToolDefinition()
        
        error_str = str(exc_info.value)
        assert "function" in error_str
        assert "Field required" in error_str or "Required" in error_str


class Test_ResponseFormat_01_NominalBehaviors:
    """Tests for normal, expected use cases of the ResponseFormat class."""

    @pytest.mark.parametrize("type_val, json_schema", [
        ("json", None),
        ("text", None),
        ("json", {"type": "object", "properties": {"result": {"type": "string"}}}),
    ])
    def test_create_valid_response_format(self, type_val, json_schema):
        """Test creating ResponseFormat instances with valid parameters."""
        response_format = ResponseFormat(type=type_val, json_schema=json_schema)
        
        assert response_format.type == type_val
        assert response_format.json_schema == json_schema


class Test_ResponseFormat_02_NegativeBehaviors:
    """Tests for invalid usage scenarios of the ResponseFormat class."""

    def test_missing_required_type(self):
        """Test that creating a ResponseFormat without a type raises an error."""
        with pytest.raises(ValidationError) as exc_info:
            ResponseFormat()
        
        assert "type" in str(exc_info.value)


class Test_ResponseFormat_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ResponseFormat class."""

    def test_error_message_clarity(self):
        """Test that validation errors include clear information about the error location."""
        with pytest.raises(ValidationError) as exc_info:
            ResponseFormat()
        
        error_str = str(exc_info.value)
        assert "type" in error_str
        assert "Field required" in error_str or "Required" in error_str


class Test_CacheControl_01_NominalBehaviors:
    """Tests for normal, expected use cases of the CacheControl class."""

    def test_default_type(self):
        """Test that CacheControl uses 'ephemeral' as the default type."""
        cache_control = CacheControl()
        assert cache_control.type == "ephemeral"
    
    def test_explicit_ephemeral_type(self):
        """Test creating CacheControl with explicit 'ephemeral' type."""
        cache_control = CacheControl(type="ephemeral")
        assert cache_control.type == "ephemeral"


class Test_CacheControl_02_NegativeBehaviors:
    """Tests for invalid usage scenarios of the CacheControl class."""

    def test_non_default_type(self):
        """
        Test with a non-default type.
        Note: This test assumes non-default types might be valid,
        as the code doesn't explicitly restrict the type to 'ephemeral'.
        """
        cache_control = CacheControl(type="non-ephemeral")
        assert cache_control.type == "non-ephemeral"


class Test_TextContent_01_NominalBehaviors:
    """Tests for normal, expected use cases of the TextContent class."""

    @pytest.mark.parametrize("text, cache_control", [
        ("Hello world", None),
        ("Another text", CacheControl()),
        ("With custom cache", CacheControl(type="ephemeral")),
    ])
    def test_create_valid_text_content(self, text, cache_control):
        """Test creating TextContent with valid parameters."""
        text_content = TextContent(text=text, cache_control=cache_control)
        
        assert text_content.type == "text"  # Fixed value
        assert text_content.text == text
        assert text_content.cache_control == cache_control


class Test_TextContent_02_NegativeBehaviors:
    """Tests for invalid usage scenarios of the TextContent class."""

    def test_missing_text(self):
        """Test that creating TextContent without text raises an error."""
        with pytest.raises(ValidationError) as exc_info:
            TextContent()
        
        assert "text" in str(exc_info.value)


class Test_TextContent_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the TextContent class."""

    def test_error_message_clarity(self):
        """Test that validation errors include clear information about the error location."""
        with pytest.raises(ValidationError) as exc_info:
            TextContent()
        
        error_str = str(exc_info.value)
        assert "text" in error_str
        assert "Field required" in error_str or "Required" in error_str


class Test_ImageUrl_01_NominalBehaviors:
    """Tests for normal, expected use cases of the ImageUrl class."""

    @pytest.mark.parametrize("url, detail", [
        ("https://example.com/image.jpg", None),
        ("data:image/png;base64,iVBORw0KGgo=", None),
        ("https://example.com/image.jpg", "low"),
        ("https://example.com/image.jpg", "high"),
    ])
    def test_create_valid_image_url(self, url, detail):
        """Test creating ImageUrl with valid parameters."""
        image_url = ImageUrl(url=url, detail=detail)
        
        assert image_url.url == url
        assert image_url.detail == (detail or "auto")  # Default is "auto"


class Test_ImageUrl_02_NegativeBehaviors:
    """Tests for invalid usage scenarios of the ImageUrl class."""

    def test_missing_url(self):
        """Test that creating ImageUrl without URL raises an error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageUrl()
        
        assert "url" in str(exc_info.value)


class Test_ImageUrl_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the ImageUrl class."""

    def test_error_message_clarity(self):
        """Test that validation errors include clear information about the error location."""
        with pytest.raises(ValidationError) as exc_info:
            ImageUrl()
        
        error_str = str(exc_info.value)
        assert "url" in error_str
        assert "Field required" in error_str or "Required" in error_str


class Test_ImageContentPart_01_NominalBehaviors:
    """Tests for normal, expected use cases of the ImageContentPart class."""

    def test_create_with_valid_image_url(self):
        """Test creating ImageContentPart with a valid ImageUrl."""
        image_url = ImageUrl(url="https://example.com/image.jpg")
        
        image_part = ImageContentPart(image_url=image_url)
        
        assert image_part.type == "image_url"  # Fixed value
        assert image_part.image_url == image_url


class Test_ImageContentPart_02_NegativeBehaviors:
    """Tests for invalid usage scenarios of the ImageContentPart class."""

    def test_missing_image_url(self):
        """Test that creating ImageContentPart without an image_url raises an error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageContentPart()
        
        assert "image_url" in str(exc_info.value)


class Test_FileData_01_NominalBehaviors:
    """Tests for normal, expected use cases of the FileData class."""

    @pytest.mark.parametrize("filename, file_data", [
        ("test.txt", "data:text/plain;base64,SGVsbG8gV29ybGQ="),
        ("image.png", "data:image/png;base64,iVBORw0KGgo="),
        ("document.pdf", "data:application/pdf;base64,JVBERi0xLjMKJ..."),
    ])
    def test_create_valid_file_data(self, filename, file_data):
        """Test creating FileData with valid parameters."""
        file_data_obj = FileData(filename=filename, file_data=file_data)
        
        assert file_data_obj.filename == filename
        assert file_data_obj.file_data == file_data


class Test_FileData_02_NegativeBehaviors:
    """Tests for invalid usage scenarios of the FileData class."""

    @pytest.mark.parametrize("missing_field", [
        "filename", 
        "file_data"
    ])
    def test_missing_required_field(self, missing_field):
        """Test that creating FileData without required fields raises an error."""
        data = {
            "filename": "test.txt",
            "file_data": "data:text/plain;base64,SGVsbG8gV29ybGQ="
        }
        data.pop(missing_field)
        
        with pytest.raises(ValidationError) as exc_info:
            FileData(**data)
        
        assert missing_field in str(exc_info.value)


class Test_FileData_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the FileData class."""

    def test_error_message_clarity(self):
        """Test that validation errors include clear information about the error location."""
        with pytest.raises(ValidationError) as exc_info:
            FileData()
        
        error_str = str(exc_info.value)
        assert any(field in error_str for field in ["filename", "file_data"])
        assert "Field required" in error_str or "Required" in error_str


class Test_FileContent_01_NominalBehaviors:
    """Tests for normal, expected use cases of the FileContent class."""

    def test_create_with_valid_file_data(self):
        """Test creating FileContent with valid FileData."""
        file_data = FileData(
            filename="test.txt", 
            file_data="data:text/plain;base64,SGVsbG8gV29ybGQ="
        )
        
        file_content = FileContent(file=file_data)
        
        assert file_content.type == "file"  # Fixed value
        assert file_content.file == file_data


class Test_FileContent_02_NegativeBehaviors:
    """Tests for invalid usage scenarios of the FileContent class."""

    def test_missing_file(self):
        """Test that creating FileContent without a file raises an error."""
        with pytest.raises(ValidationError) as exc_info:
            FileContent()
        
        assert "file" in str(exc_info.value)


class Test_FileContent_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the FileContent class."""

    def test_error_message_clarity(self):
        """Test that validation errors include clear information about the error location."""
        with pytest.raises(ValidationError) as exc_info:
            FileContent()
        
        error_str = str(exc_info.value)
        assert "file" in error_str
        assert "Field required" in error_str or "Required" in error_str


class Test_Prediction_01_NominalBehaviors:
    """Tests for normal, expected use cases of the Prediction class."""

    def test_create_valid_prediction(self):
        """Test creating a Prediction with valid content."""
        prediction = Prediction(content="This is a predicted response")
        
        assert prediction.type == "content"  # Fixed value
        assert prediction.content == "This is a predicted response"


class Test_Prediction_02_NegativeBehaviors:
    """Tests for invalid usage scenarios of the Prediction class."""

    def test_missing_content(self):
        """Test that creating a Prediction without content raises an error."""
        with pytest.raises(ValidationError) as exc_info:
            Prediction()
        
        assert "content" in str(exc_info.value)


class Test_Prediction_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the Prediction class."""

    def test_error_message_clarity(self):
        """Test that validation errors include clear information about the error location."""
        with pytest.raises(ValidationError) as exc_info:
            Prediction()
        
        error_str = str(exc_info.value)
        assert "content" in error_str
        assert "Field required" in error_str or "Required" in error_str


class Test_Message_01_NominalBehaviors:
    """Tests for normal, expected use cases of the Message class."""

    @pytest.mark.parametrize("role, content", [
        (ModelRole.USER, "Hello, how can you help me?"),
        (ModelRole.ASSISTANT, "I can help answer your questions."),
        (ModelRole.SYSTEM, "You are an AI assistant."),
    ])
    def test_create_valid_message_with_string_content(self, role, content):
        """Test creating Message instances with valid string content."""
        message = Message(role=role, content=content)
        
        assert message.role == role
        assert message.content == content
    
    def test_create_user_message_with_multimodal_content(self):
        """Test creating a user message with multimodal content (text and image)."""
        text_content = TextContent(text="Look at this image")
        image_content = ImageContentPart(
            image_url=ImageUrl(url="https://example.com/image.jpg")
        )
        
        message = Message(
            role=ModelRole.USER,
            content=[text_content, image_content]
        )
        
        assert message.role == ModelRole.USER
        assert len(message.content) == 2
        assert message.content[0] == text_content
        assert message.content[1] == image_content
    
    def test_create_tool_message_with_tool_call_id(self):
        """Test creating a tool message with required tool_call_id."""
        message = Message(
            role=ModelRole.TOOL,
            content="The weather is sunny",
            tool_call_id="weather_tool_call_123"
        )
        
        assert message.role == ModelRole.TOOL
        assert message.content == "The weather is sunny"
        assert message.tool_call_id == "weather_tool_call_123"


class Test_Message_02_NegativeBehaviors:
    """Tests for invalid usage scenarios of the Message class."""

    @pytest.mark.parametrize("role, content, tool_call_id, expected_error", [
        (ModelRole.TOOL, "Result", None, "Messages with role 'tool' must have a tool_call_id"),
        (None, "Content", None, "should be 'system', 'user', 'assistant', 'function' or 'tool'"),
    ])
    def test_invalid_message_creation(self, role, content, tool_call_id, expected_error):
        """Test that creating Message with invalid parameters raises appropriate errors."""
        with pytest.raises(ValidationError) as exc_info:
            Message(
                role=role,
                content=content,
                tool_call_id=tool_call_id
            )
        
        assert expected_error in str(exc_info.value)


class Test_Message_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the Message class."""

    def test_error_message_clarity(self):
        """Test that validation errors include clear information about the error location."""
        with pytest.raises(ValidationError) as exc_info:
            Message()
        
        error_str = str(exc_info.value)
        assert "role" in error_str
        assert "Field required" in error_str or "Required" in error_str


class Test_Message_ValidateToolRole_01_NominalBehaviors:
    """Tests for normal, expected use cases of the Message.validate_tool_role validator."""

    def test_valid_tool_role_with_tool_call_id(self):
        """Test that a tool role message with tool_call_id passes validation."""
        message = Message(
            role=ModelRole.TOOL,
            content="Tool result",
            tool_call_id="tool_123"
        )
        
        # Validation happens during initialization, so reaching this point means validation passed
        assert message.role == ModelRole.TOOL
        assert message.tool_call_id == "tool_123"
    
    def test_non_tool_role_without_tool_call_id(self):
        """Test that a non-tool role message without tool_call_id passes validation."""
        message = Message(
            role=ModelRole.USER,
            content="User message"
        )
        
        # Validation happens during initialization, so reaching this point means validation passed
        assert message.role == ModelRole.USER
        assert message.tool_call_id is None


class Test_Message_ValidateToolRole_02_NegativeBehaviors:
    """Tests for invalid usage scenarios of the Message.validate_tool_role validator."""

    def test_tool_role_without_tool_call_id(self):
        """Test that creating a tool role message without tool_call_id raises an error."""
        with pytest.raises(ValueError) as exc_info:
            Message(
                role=ModelRole.TOOL,
                content="Tool result without ID"
            )
        
        assert "Messages with role 'tool' must have a tool_call_id" in str(exc_info.value)


class Test_Message_ValidateToolRole_04_ErrorHandlingBehaviors:
    """Tests for error handling behaviors of the Message.validate_tool_role validator."""

    def test_error_message_clarity(self):
        """Test that validation errors include clear information about the error."""
        with pytest.raises(ValueError) as exc_info:
            Message(
                role=ModelRole.TOOL,
                content="Tool result"
            )
        
        error_str = str(exc_info.value)
        assert "tool" in error_str
        assert "tool_call_id" in error_str
        assert "must have" in error_str
