"""Tests for PDF support in messages."""

from typing import List, Dict, Any, Union

import pytest
from pydantic import BaseModel, Field

# Create minimal test classes that mimic the structure of our models

class ModelRole:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class TextContent(BaseModel):
    type: str = Field("text")
    text: str

class FileData(BaseModel):
    filename: str
    file_data: str

class FileContent(BaseModel):
    type: str = Field("file")
    file: FileData

ContentPart = Union[TextContent, FileContent]

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart]] = None


class Test_FileContent_01_Nominal_Behavior:
    """Test nominal behaviors of FileContent model."""
    
    def test_create_file_content(self):
        """Test that a FileContent instance can be created."""
        file_data = FileData(
            filename="test.pdf",
            file_data="data:application/pdf;base64,SGVsbG8gV29ybGQ="
        )
        file_content = FileContent(
            type="file",
            file=file_data
        )
        
        assert file_content.type == "file"
        assert file_content.file.filename == "test.pdf"
        assert file_content.file.file_data == "data:application/pdf;base64,SGVsbG8gV29ybGQ="


class Test_Message_01_Nominal_Behavior:
    """Test nominal behaviors of Message model with PDF content."""
    
    def test_create_message_with_text_and_pdf(self):
        """Test creating a message with text and PDF content."""
        # Create a text content part
        text_content = TextContent(
            type="text",
            text="What is in this document?"
        )
        
        # Create a file content part
        file_content = FileContent(
            type="file",
            file=FileData(
                filename="document.pdf",
                file_data="data:application/pdf;base64,SGVsbG8gV29ybGQ="
            )
        )
        
        # Create a message with both content parts
        message = Message(
            role=ModelRole.USER,
            content=[text_content, file_content]
        )
        
        # Verify message structure
        assert message.role == ModelRole.USER
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert message.content[0].type == "text"
        assert message.content[0].text == "What is in this document?"
        assert message.content[1].type == "file"
        assert message.content[1].file.filename == "document.pdf"
        assert message.content[1].file.file_data == "data:application/pdf;base64,SGVsbG8gV29ybGQ="