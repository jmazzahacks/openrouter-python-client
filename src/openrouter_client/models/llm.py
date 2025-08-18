"""
Simplified LLM-style API for OpenRouter Client.
"""

from typing import List, Optional, TYPE_CHECKING
from .attachment import Attachment

if TYPE_CHECKING:
    from ..client import OpenRouterClient


class LLMModel:
    """Model wrapper with simplified prompt API, inspired by Simon Willison's llm library."""
    
    def __init__(self, model_id: str, client: "OpenRouterClient"):
        self.model_id = model_id
        self.client = client
    
    def prompt(self, text: str, attachments: Optional[List[Attachment]] = None) -> str:
        """Send a prompt with optional attachments."""
        content = [{"type": "text", "text": text}]
        
        if attachments:
            for attachment in attachments:
                content.append(attachment.to_content_part())
        
        response = self.client.chat.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}]
        )
        
        return response.choices[0].message.content


def get_model(model_id: str, client: "OpenRouterClient") -> LLMModel:
    """Get a model instance."""
    return LLMModel(model_id, client)