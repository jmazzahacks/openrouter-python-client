"""Core data models for OpenRouter Client.

This module defines the core Pydantic models used across
the OpenRouter Client.

Exported:
- FunctionDefinition: Function definition model
- ToolDefinition: Tool definition model
- ResponseFormat: Response format specification model
- Message: Chat message model
- CompletionsParameters: Shared parameters for completion endpoints
"""

from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field, model_validator

from ..types import ModelRole


class FunctionDefinition(BaseModel):
    """
    Definition of a function that can be called by a model.
    
    Attributes:
        name (str): Name of the function.
        description (Optional[str]): Description of what the function does.
        parameters (Dict[str, Any]): JSON Schema object describing function parameters.
    """
    name: str = Field(..., min_length=1, description="Name of the function")
    description: Optional[str] = Field(None, description="Description of what the function does")
    parameters: Dict[str, Any] = Field(..., description="JSON Schema object describing function parameters")


class ToolDefinition(BaseModel):
    """
    Definition of a tool that can be used by a model.
    
    Attributes:
        type (str): Type of tool, typically "function".
        function (FunctionDefinition): Function definition for the tool.
    """
    type: str = Field("function", description="Type of tool, currently only 'function' is supported")
    function: FunctionDefinition = Field(..., description="Function definition for the tool")


class ResponseFormat(BaseModel):
    """
    Format specification for model responses.
    
    Attributes:
        type (str): Response format type (e.g., "json", "text").
    """
    type: str = Field(..., description="Response format type")
    json_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for the response format")


class CacheControl(BaseModel):
    """
    Cache control settings for content parts.
    
    Attributes:
        type (str): Cache control type, currently only "ephemeral" is supported.
    """
    type: str = Field("ephemeral", description="Cache control type, currently only 'ephemeral' is supported")


class TextContent(BaseModel):
    """
    Text content part for multimodal messages.
    
    Attributes:
        type (str): Content type, always "text".
        text (str): The text content.
        cache_control (Optional[CacheControl]): Cache control settings for this content.
    """
    type: str = Field("text", description="Content type, always 'text'")
    text: str = Field(..., description="The text content")
    cache_control: Optional[CacheControl] = Field(None, description="Cache control settings for this content")
    
    @model_validator(mode="after")
    def validate_type(self) -> "TextContent":
        if not isinstance(self.type, str) or self.type != "text":
            raise ValueError("Type must be 'text'")
        return self


class ImageUrl(BaseModel):
    """
    Image URL container with optional detail level.
    
    Attributes:
        url (str): URL or base64 encoded image data.
        detail (Optional[str]): Detail level, defaults to "auto".
    """
    url: str = Field(..., description="URL or base64 encoded image data")
    detail: Optional[str] = Field("auto", description="Detail level, defaults to 'auto'")
    
    @model_validator(mode="after")
    def validate_detail(self) -> "ImageUrl":
        if self.detail is None:
            self.detail = "auto"
        return self


class ImageContentPart(BaseModel):
    """
    Image content part for multimodal messages.
    
    Attributes:
        type (str): Content type, always "image_url".
        image_url (ImageUrl): Container for image URL and detail level.
    """
    type: str = Field("image_url", description="Content type, always 'image_url'")
    image_url: ImageUrl = Field(..., description="Container for image URL and detail level")


class FileData(BaseModel):
    """
    File data container for file content parts.
    
    Attributes:
        filename (str): Name of the file.
        file_data (str): Base64 encoded file data URL.
    """
    filename: str = Field(..., description="Name of the file")
    file_data: str = Field(..., description="Base64 encoded file data URL")


class FileContent(BaseModel):
    """
    File content part for multimodal messages.
    
    Attributes:
        type (str): Content type, always "file".
        file (FileData): Container for file data.
    """
    type: str = Field("file", description="Content type, always 'file'")
    file: FileData = Field(..., description="Container for file data")


# Union type for ContentPart, defined here for type hints
ContentPart = Union[TextContent, ImageContentPart, FileContent]


class Prediction(BaseModel):
    """
    Predicted output to reduce latency.
    
    Attributes:
        type (str): Type of prediction, currently only 'content' is supported.
        content (str): The predicted text content.
    """
    type: str = Field("content", description="Type of prediction, currently only 'content' is supported")
    content: str = Field(..., description="The predicted text content")


class Message(BaseModel):
    """
    Chat message model for conversations.
    
    Attributes:
        role (ModelRole): Role of the message sender.
        content (Optional[Union[str, List[ContentPart]]]): Message content.
            For user role, this can be a string or a list of content parts (text/images).
            For other roles, this should be a string.
        name (Optional[str]): Name of the sender.
        function_call (Optional[Dict[str, Any]]): Function call details if applicable.
        tool_calls (Optional[List[Dict[str, Any]]]): Tool calls details if applicable.
        tool_call_id (Optional[str]): Required when role is 'tool', the ID of the tool call being responded to.
    """
    role: ModelRole = Field(..., description="Role of the message sender")
    content: Optional[Union[str, List[ContentPart]]] = Field(None, description="Message content")
    name: Optional[str] = Field(None, description="Name of the sender")
    function_call: Optional[Dict[str, Any]] = Field(None, description="Function call details if applicable")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls details if applicable")
    tool_call_id: Optional[str] = Field(None, description="Required when role is 'tool', the ID of the tool call being responded to")
    
    @model_validator(mode="after")
    def validate_tool_role(self) -> "Message":
        """
        Validate that tool role messages have a tool_call_id.
        
        Returns:
            Message: Self after validation.
            
        Raises:
            ValueError: If role is 'tool' and tool_call_id is not provided.
        """
        if self.role == ModelRole.TOOL and not self.tool_call_id:
            raise ValueError("Messages with role 'tool' must have a tool_call_id")
        return self
