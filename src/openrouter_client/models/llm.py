"""
Simplified LLM-style API for OpenRouter Client.
"""

import json
from typing import List, Optional, Dict, Any, Union, TYPE_CHECKING
from .attachment import Attachment
from .chat import Usage
from ..exceptions import APIError

if TYPE_CHECKING:
    from ..client import OpenRouterClient


def _accumulate_usage(total: Optional[Usage], new: Optional[Usage]) -> Optional[Usage]:
    """
    Add a turn's ``Usage`` into a running total, summing tokens and cost.

    Only the aggregate token counts and cost are carried on the running total;
    the per-turn detail breakdowns (cost_details, *_tokens_details) are left unset
    on the aggregate since they don't sum meaningfully across turns.

    Args:
        total: The running total so far, or None before the first turn.
        new: This turn's usage, or None if the response carried no usage.

    Returns:
        Optional[Usage]: The updated running total (None only if both inputs are None).
    """
    if new is None:
        return total
    if total is None:
        return Usage(
            prompt_tokens=new.prompt_tokens,
            completion_tokens=new.completion_tokens,
            total_tokens=new.total_tokens,
            cost=new.cost,
        )

    if total.cost is None and new.cost is None:
        summed_cost: Optional[float] = None
    else:
        summed_cost = (total.cost or 0.0) + (new.cost or 0.0)

    return Usage(
        prompt_tokens=total.prompt_tokens + new.prompt_tokens,
        completion_tokens=total.completion_tokens + new.completion_tokens,
        total_tokens=total.total_tokens + new.total_tokens,
        cost=summed_cost,
    )


def build_json_schema_response_format(
    schema: Dict[str, Any],
    name: str = "response_schema",
) -> Dict[str, Any]:
    """
    Build the OpenRouter ``response_format`` block for structured JSON output.

    Args:
        schema: The JSON schema the model's output must conform to.
        name: Schema name reported to the API (default: "response_schema").

    Returns:
        Dict[str, Any]: A ``response_format`` dict suitable for chat.create().
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
        },
    }


def parse_schema_response(content: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and validate response content when a schema is provided.

    Guarantees that when a schema is provided, the return value is always
    a dict. Raises APIError with clear messages if validation fails.

    Args:
        content: Response content (str or dict)
        schema: The JSON schema that was provided

    Returns:
        Dict[str, Any]: Parsed and validated response as a dict

    Raises:
        APIError: If content is not valid JSON or doesn't match schema expectations
    """
    # If content is already a dict, return it
    if isinstance(content, dict):
        return content

    # If content is a string, try to parse as JSON
    if isinstance(content, str):
        # Check for empty or whitespace-only response
        if not content.strip():
            raise APIError(
                message="Model returned empty response when schema was provided. "
                        "Expected valid JSON matching the schema.",
                status_code=422,
                details={"schema": schema, "response": content}
            )

        try:
            parsed = json.loads(content)

            # Ensure parsed result is a dict (not array, string, etc.)
            if not isinstance(parsed, dict):
                raise APIError(
                    message=f"Model returned JSON of type '{type(parsed).__name__}' "
                            f"when schema requires an object (dict). Response: {content[:200]}",
                    status_code=422,
                    details={"schema": schema, "response": content, "parsed_type": type(parsed).__name__}
                )

            return parsed

        except json.JSONDecodeError as e:
            raise APIError(
                message=f"Model returned invalid JSON when schema was provided. "
                        f"JSON parse error: {str(e)}. Response: {content[:200]}",
                status_code=422,
                details={"schema": schema, "response": content, "parse_error": str(e)}
            )

    # Unexpected content type
    raise APIError(
        message=f"Model returned unexpected content type '{type(content).__name__}' "
                f"when schema was provided. Expected JSON string or dict.",
        status_code=422,
        details={"schema": schema, "content_type": type(content).__name__}
    )


class LLMModel:
    """Model wrapper with simplified prompt API, inspired by Simon Willison's llm library."""
    
    def __init__(self, model_id: str, client: "OpenRouterClient"):
        self.model_id = model_id
        self.client = client
        # Token/cost usage from the most recent prompt() call on this model, or
        # None before the first call (or if the response carried no usage block).
        # Updated only on success: after a failed prompt() it retains the prior
        # call's value. Not safe for concurrent prompt() calls on one instance.
        self.last_usage: Optional[Usage] = None

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
            Dict[str, Any]: Parsed and validated JSON response if schema provided.
                           GUARANTEED to be a dict, never a string.

        Raises:
            APIError: If schema is provided but model returns invalid JSON
                     or non-dict response
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
            chat_params["response_format"] = build_json_schema_response_format(schema)
        
        response = self.client.chat.create(**chat_params)

        # Capture this call's token/cost usage (None if the response carried no
        # usage block).
        self.last_usage = response.usage

        content = response.choices[0].message.content

        # Parse and validate JSON if schema was provided
        if schema:
            return parse_schema_response(content, schema)

        return content

    def conversation(self, system: Optional[str] = None) -> "Conversation":
        """
        Create a new conversation context for this model.
        
        Args:
            system: Optional system prompt to set context/behavior for the conversation
            
        Returns:
            Conversation: A conversation object that maintains message history
        """
        return Conversation(self.model_id, self.client, system)


class Conversation:
    """
    Conversation context that maintains message history for multi-turn interactions.
    
    Follows the llm library pattern where you can call conversation.prompt() multiple times
    and it automatically maintains the conversation context.
    """
    
    def __init__(self, model_id: str, client: "OpenRouterClient", system: Optional[str] = None):
        self.model_id = model_id
        self.client = client
        self.messages = []
        # Usage from the most recent turn, and the running total across all turns
        # in this conversation. Both are None before the first prompt() call.
        # Updated only on a successful turn, so a failed prompt() leaves them
        # unchanged and totals never double-count. A single Conversation is not
        # safe for concurrent prompt() calls.
        self.last_usage: Optional[Usage] = None
        self.total_usage: Optional[Usage] = None

        # Add system message if provided
        if system:
            self.messages.append({"role": "system", "content": system})

    @property
    def total_cost(self) -> float:
        """Cumulative cost in credits across all turns (0.0 if no cost reported)."""
        if self.total_usage is None or self.total_usage.cost is None:
            return 0.0
        return self.total_usage.cost

    def prompt(
        self,
        text: str,
        attachments: Optional[List[Attachment]] = None,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Send a prompt within this conversation context.

        Args:
            text: The user prompt text
            attachments: Optional list of file attachments
            schema: Optional JSON schema for structured output
            **kwargs: Additional parameters passed to chat.create()

        Returns:
            str: Response content if no schema provided
            Dict[str, Any]: Parsed and validated JSON response if schema provided.
                           GUARANTEED to be a dict, never a string.

        Raises:
            APIError: If schema is provided but model returns invalid JSON
                     or non-dict response
        """
        # Build user message content
        content = [{"type": "text", "text": text}]
        
        if attachments:
            for attachment in attachments:
                content.append(attachment.to_content_part())
        
        # Add user message to conversation history
        self.messages.append({"role": "user", "content": content})
        
        # Prepare chat.create() parameters
        chat_params = {
            "model": self.model_id,
            "messages": self.messages,
            **kwargs  # Include any additional parameters like temperature
        }
        
        # Add structured output if schema provided
        if schema:
            chat_params["response_format"] = build_json_schema_response_format(schema)
        
        response = self.client.chat.create(**chat_params)

        # Capture this turn's usage and fold it into the conversation running total.
        self.last_usage = response.usage
        self.total_usage = _accumulate_usage(self.total_usage, self.last_usage)

        response_content = response.choices[0].message.content

        # Add assistant response to conversation history
        self.messages.append({"role": "assistant", "content": response_content})

        # Parse and validate JSON if schema was provided
        if schema:
            return parse_schema_response(response_content, schema)

        return response_content
    
    def get_message_count(self) -> int:
        """Get the number of messages in this conversation."""
        return len(self.messages)
    
    def clear(self) -> None:
        """Clear the conversation history, keeping only the system prompt if any."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []


def get_model(model_id: str, client: "OpenRouterClient") -> LLMModel:
    """Get a model instance."""
    return LLMModel(model_id, client)