"""
Simplified LLM-style API for OpenRouter Client.
"""

import json
from typing import List, Optional, Dict, Any, Union, TYPE_CHECKING
from .attachment import Attachment

if TYPE_CHECKING:
    from ..client import OpenRouterClient


class LLMModel:
    """Model wrapper with simplified prompt API, inspired by Simon Willison's llm library."""
    
    def __init__(self, model_id: str, client: "OpenRouterClient"):
        self.model_id = model_id
        self.client = client
    
    def prompt(
        self, 
        text: str, 
        system: Optional[str] = None,
        attachments: Optional[List[Attachment]] = None,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Send a prompt with optional system message, attachments, and structured output.
        
        Args:
            text: The user prompt text
            system: Optional system prompt to set context/behavior
            attachments: Optional list of file attachments
            schema: Optional JSON schema for structured output
            **kwargs: Additional parameters passed to chat.create()
            
        Returns:
            str: Response content if no schema provided
            Dict[str, Any]: Parsed JSON response if schema provided
        """
        # Build messages array
        messages = []
        
        # Add system message if provided
        if system:
            messages.append({"role": "system", "content": system})
        
        # Build user message content
        content = [{"type": "text", "text": text}]
        
        if attachments:
            for attachment in attachments:
                content.append(attachment.to_content_part())
        
        messages.append({"role": "user", "content": content})
        
        # Prepare chat.create() parameters
        chat_params = {
            "model": self.model_id,
            "messages": messages,
            **kwargs  # Include any additional parameters like temperature
        }
        
        # Add structured output if schema provided
        if schema:
            chat_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "schema": schema
                }
            }
        
        response = self.client.chat.create(**chat_params)
        
        content = response.choices[0].message.content
        
        # Parse JSON if schema was provided
        if schema:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback to raw content if JSON parsing fails
                return content
        
        return content


def get_model(model_id: str, client: "OpenRouterClient") -> LLMModel:
    """Get a model instance."""
    return LLMModel(model_id, client)