"""
Attachment model for simplified file and image handling.

This module provides an Attachment class that simplifies working with
images and files in OpenRouter API requests.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import base64
import mimetypes
from urllib.parse import urlparse
from pydantic import BaseModel, Field, model_validator


class Attachment(BaseModel):
    """
    Simple attachment class for images and files.
    
    Can be created with either a file path or a URL, but not both.
    Automatically handles base64 encoding for local files and MIME type detection.
    
    Attributes:
        path (Optional[Path]): Local file path
        url (Optional[str]): Remote URL
        mime_type (Optional[str]): MIME type (auto-detected if not provided)
    
    Examples:
        >>> # From local file
        >>> attachment = Attachment(path="image.jpg")
        >>> 
        >>> # From URL
        >>> attachment = Attachment(url="https://example.com/image.png")
        >>> 
        >>> # With explicit MIME type
        >>> attachment = Attachment(path="photo.jpg", mime_type="image/jpeg")
    """
    path: Optional[Path] = Field(None, description="Local file path")
    url: Optional[str] = Field(None, description="Remote URL")
    mime_type: Optional[str] = Field(None, description="MIME type")
    
    @model_validator(mode='after')
    def validate_source(self):
        """Ensure exactly one source (path or url) is provided."""
        if not (self.path or self.url):
            raise ValueError("Must provide either path or url")
        if self.path and self.url:
            raise ValueError("Cannot provide both path and url")
        
        # Convert string path to Path object if needed
        if self.path and isinstance(self.path, str):
            self.path = Path(self.path)
        
        # Validate path exists if provided
        if self.path and not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        
        return self
    
    def detect_mime_type(self) -> str:
        """
        Detect MIME type for the attachment.
        
        Returns:
            str: The MIME type
        """
        if self.mime_type:
            return self.mime_type
        
        if self.path:
            # Try to guess from file extension
            mime_type = mimetypes.guess_type(str(self.path))[0]
            
            # Handle common image types that might not be detected
            if not mime_type and self.path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                ext = self.path.suffix[1:].lower()
                if ext == 'jpg':
                    ext = 'jpeg'
                mime_type = f"image/{ext}"
            
            return mime_type or "application/octet-stream"
        
        # For URLs, try to detect from extension
        if self.url:
            parsed = urlparse(self.url)
            path = Path(parsed.path)
            
            if path.suffix:
                # Try standard mimetypes first
                mime_type = mimetypes.guess_type(str(path))[0]
                
                # Handle common cases that might not be in mimetypes
                if not mime_type:
                    ext = path.suffix.lower()
                    if ext in ['.jpg', '.jpeg']:
                        mime_type = "image/jpeg"
                    elif ext == '.png':
                        mime_type = "image/png"
                    elif ext == '.gif':
                        mime_type = "image/gif"
                    elif ext == '.webp':
                        mime_type = "image/webp"
                    elif ext == '.pdf':
                        mime_type = "application/pdf"
                    elif ext in ['.mp4', '.mpeg', '.mpg']:
                        mime_type = "video/mp4"
                    elif ext in ['.mp3', '.wav', '.m4a']:
                        mime_type = f"audio/{ext[1:]}"
                
                if mime_type:
                    return mime_type
            
            # Default for URLs without clear extension
            # Don't assume image/jpeg - use generic binary type
            return "application/octet-stream"
    
    def to_content_part(self) -> Dict[str, Any]:
        """
        Convert attachment to OpenRouter content part dictionary.
        
        Returns:
            Dict[str, Any]: Content part dictionary for the API
            
        Example:
            >>> attachment = Attachment(path="photo.jpg")
            >>> content_part = attachment.to_content_part()
            >>> # Returns: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        """
        if self.path:
            mime_type = self.detect_mime_type()
            
            # Read and encode file
            with open(self.path, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            
            # Determine content type based on MIME type
            if mime_type.startswith('image/'):
                content_type = "image_url"
            elif mime_type == 'application/pdf':
                # Some models support PDFs as documents
                content_type = "image_url"  # OpenAI format treats PDFs like images
            else:
                # For other file types, might need different handling
                content_type = "image_url"  # Default to image_url for now
            
            return {
                "type": content_type,
                "image_url": {"url": f"data:{mime_type};base64,{data}"}
            }
        else:
            # URL attachment - pass through directly
            # The API will handle fetching and determining type
            return {
                "type": "image_url", 
                "image_url": {"url": self.url}
            }
    
    def __repr__(self) -> str:
        """String representation of the attachment."""
        if self.path:
            return f"Attachment(path='{self.path}')"
        else:
            return f"Attachment(url='{self.url}')"