"""
Simplified LLM-style API for OpenRouter Client.
"""

import json
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, cast

from pydantic import BaseModel, Field, field_validator

from ..exceptions import APIError, ToolCallLimitExceeded, ToolExecutionError
from .attachment import Attachment
from .chat import ChatCompletionResponse, Usage

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
                details={"schema": schema, "response": content},
            )

        try:
            parsed = json.loads(content)

            # Ensure parsed result is a dict (not array, string, etc.)
            if not isinstance(parsed, dict):
                raise APIError(
                    message=f"Model returned JSON of type '{type(parsed).__name__}' "
                    f"when schema requires an object (dict). Response: {content[:200]}",
                    status_code=422,
                    details={
                        "schema": schema,
                        "response": content,
                        "parsed_type": type(parsed).__name__,
                    },
                )

            return parsed

        except json.JSONDecodeError as e:
            raise APIError(
                message=f"Model returned invalid JSON when schema was provided. "
                f"JSON parse error: {str(e)}. Response: {content[:200]}",
                status_code=422,
                details={"schema": schema, "response": content, "parse_error": str(e)},
            )

    # Unexpected content type
    raise APIError(
        message=f"Model returned unexpected content type '{type(content).__name__}' "
        f"when schema was provided. Expected JSON string or dict.",
        status_code=422,
        details={"schema": schema, "content_type": type(content).__name__},
    )


class ToolLoop(BaseModel):
    """
    Tool definitions plus the callables that satisfy them, for an automated tool loop.

    Pass one to ``LLMModel.prompt()`` or ``Conversation.prompt()`` to have the
    model's tool calls executed and fed back automatically until it produces a
    final answer. ``tools`` is what gets sent to the API (build these with the
    ``openrouter_client.tools`` helpers); ``handlers`` maps each tool's function
    name to the Python callable that runs it.

    A handler is invoked as ``handler(**arguments)`` with the arguments the model
    emitted, and may return any JSON-serializable value, a string, or a Pydantic
    model — the result is serialized into the ``role="tool"`` message.

    Attributes:
        tools (List[Any]): Tool definitions sent to the API each round. Pydantic
            tool models (what the ``tools`` helpers return) are converted to plain
            dicts on construction; see the validator below for why.
        handlers (Dict[str, Callable[..., Any]]): Function name -> callable.
        max_rounds (int): Maximum number of tool-executing rounds before
            ToolCallLimitExceeded is raised (default: 8).
    """

    tools: List[Any] = Field(
        ..., description="Tool definitions sent to the API each round"
    )
    handlers: Dict[str, Callable[..., Any]] = Field(
        ..., description="Mapping of tool function name to its executing callable"
    )
    max_rounds: int = Field(
        8, gt=0, description="Maximum number of tool-executing rounds before giving up"
    )

    @field_validator("tools")
    @classmethod
    def serialize_tool_models(cls, tools: List[Any]) -> List[Any]:
        """
        Convert Pydantic tool definitions to plain dicts.

        chat.create() only converts tool models when validate_request=True, which
        is not the default; otherwise it hands them to requests as ``json=``,
        where a ChatCompletionTool raises "Object of type ChatCompletionTool is
        not JSON serializable". Normalizing here means the helpers in
        ``openrouter_client.tools`` and hand-written dicts both just work.

        Args:
            tools: Tool definitions as supplied by the caller.

        Returns:
            List[Any]: The definitions with any Pydantic models dumped to dicts.
        """
        serialized: List[Any] = []
        for definition in tools:
            if isinstance(definition, BaseModel):
                serialized.append(definition.model_dump(exclude_none=True))
            else:
                serialized.append(definition)
        return serialized


class ToolCallRequest(BaseModel):
    """
    One tool call requested by the model, normalized from the response message.

    Deliberately more permissive than ``ChatCompletionToolCall``: models routinely
    emit an empty ``arguments`` string for a zero-argument tool, which the stricter
    model rejects. Argument parsing is handled by the loop so failures surface as
    ToolExecutionError rather than a validation error.

    Attributes:
        id (str): ID of the tool call, echoed back on the tool result message.
        name (str): Name of the function the model wants to call.
        arguments (str): Raw JSON string of arguments ("" when the model sent none).
    """

    id: str = Field(
        ..., description="ID of the tool call, echoed back on the tool result"
    )
    name: str = Field(..., description="Name of the function the model wants to call")
    arguments: str = Field("", description="Raw JSON string of arguments")


class TurnResult(BaseModel):
    """
    Outcome of one prompt() turn, which may have spanned several API calls.

    Attributes:
        content (Optional[Any]): Final assistant content for the turn, as the
            provider returned it — normally a string, but left untyped so an
            unexpected shape reaches parse_schema_response's clear errors rather
            than failing validation here.
        usage (Optional[Usage]): Usage summed across every API call in the turn.
    """

    content: Optional[Any] = Field(
        None, description="Final assistant content for the turn"
    )
    usage: Optional[Usage] = Field(
        None, description="Usage summed across the turn's API calls"
    )


def _read_tool_call(call: Any) -> ToolCallRequest:
    """
    Normalize a tool call from a response message into a ToolCallRequest.

    Accepts the plain dicts the API returns as well as objects exposing
    ``id``/``function.name``/``function.arguments``.

    Args:
        call: A single entry from ``message.tool_calls``.

    Returns:
        ToolCallRequest: The normalized tool call.

    Raises:
        ToolExecutionError: If the entry lacks the fields needed to execute it.
    """
    if isinstance(call, dict):
        function = call.get("function") or {}
        call_id = call.get("id")
        name = function.get("name") if isinstance(function, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
    else:
        function = getattr(call, "function", None)
        call_id = getattr(call, "id", None)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)

    if not call_id or not name:
        raise ToolExecutionError(
            message="Model returned a tool call without an id or function "
            f"name: {call!r}",
            tool_name=name,
            tool_call_id=call_id,
        )

    # Most providers send arguments as a JSON string, but some (Gemini-family
    # routes especially) send the object itself. Re-encode so the loop's own
    # parsing and its ToolExecutionError contract cover both shapes, rather than
    # leaking a pydantic ValidationError the caller has no reason to expect.
    if arguments is not None and not isinstance(arguments, str):
        arguments = json.dumps(arguments, default=str)

    return ToolCallRequest(id=call_id, name=name, arguments=arguments or "")


def _serialize_tool_result(result: Any) -> str:
    """
    Render a handler's return value as the content of a ``role="tool"`` message.

    Args:
        result: Whatever the handler returned.

    Returns:
        str: Strings pass through; Pydantic models and other values are JSON-encoded.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return json.dumps(result, default=str)


def _execute_tool_call(
    call: ToolCallRequest, handlers: Dict[str, Callable[..., Any]]
) -> Dict[str, Any]:
    """
    Run one tool call and build the ``role="tool"`` message carrying its result.

    Args:
        call: The normalized tool call requested by the model.
        handlers: Mapping of function name to callable.

    Returns:
        Dict[str, Any]: A tool-result message ready to append to the history.

    Raises:
        ToolExecutionError: If no handler is registered for the name, the arguments
            are not a JSON object, or the handler raises.
    """
    handler = handlers.get(call.name)
    if handler is None:
        raise ToolExecutionError(
            message=f"Model called tool '{call.name}', which has no registered "
            f"handler. Registered handlers: {sorted(handlers)}.",
            tool_name=call.name,
            tool_call_id=call.id,
        )

    raw_arguments = call.arguments.strip()
    if not raw_arguments:
        arguments: Dict[str, Any] = {}
    else:
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as e:
            raise ToolExecutionError(
                message=f"Model sent invalid JSON arguments for tool "
                f"'{call.name}': {str(e)}. Arguments: {call.arguments[:200]}",
                tool_name=call.name,
                tool_call_id=call.id,
                original_error=e,
            )

    if not isinstance(arguments, dict):
        raise ToolExecutionError(
            message=f"Model sent arguments of type "
            f"'{type(arguments).__name__}' for tool '{call.name}', but tool "
            f"arguments must be a JSON object.",
            tool_name=call.name,
            tool_call_id=call.id,
        )

    try:
        result = handler(**arguments)
    except Exception as e:
        raise ToolExecutionError(
            message=f"Handler for tool '{call.name}' raised "
            f"{type(e).__name__}: {str(e)}",
            tool_name=call.name,
            tool_call_id=call.id,
            original_error=e,
        ) from e

    return {
        "role": "tool",
        "tool_call_id": call.id,
        "content": _serialize_tool_result(result),
    }


def _plain_content(content: Any) -> Any:
    """
    Reduce assistant content to something re-sendable as JSON.

    Message.content may be a list of ContentPart models rather than a string.
    Those go back out on the next round's request, where requests' ``json=``
    cannot encode a pydantic model — the same failure ToolLoop's validator
    prevents for tool definitions.

    Args:
        content: The content as it came off the response message.

    Returns:
        Any: The content with any Pydantic parts dumped to dicts.
    """
    if isinstance(content, BaseModel):
        return content.model_dump(exclude_none=True)
    if isinstance(content, list):
        return [_plain_content(part) for part in content]
    return content


def _assistant_message(message: Any) -> Dict[str, Any]:
    """
    Build the history entry for an assistant turn, preserving any tool calls.

    Args:
        message: The message object from a chat completion choice.

    Returns:
        Dict[str, Any]: The assistant message to append to the history.
    """
    entry: Dict[str, Any] = {
        "role": "assistant",
        "content": _plain_content(message.content),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = _plain_content(tool_calls)
    return entry


# Asks for the structured answer on the tool loop's final call. Also keeps a
# user turn last, which is what stops Anthropic-family models from reading the
# request as a prefill of the preceding assistant message.
SCHEMA_TURN_INSTRUCTION = (
    "Now give your final answer as a single JSON object conforming to the "
    "required schema, using what you learned from the tools above."
)


def _combine_usage(usages: List[Usage]) -> Optional[Usage]:
    """
    Fold a turn's per-call usage into one figure, preserving a lone call's detail.

    A single call is returned unchanged rather than run through
    ``_accumulate_usage``, which drops the per-turn breakdowns (cost_details,
    is_byok, *_tokens_details). Those matter: BYOK callers read real spend from
    ``cost_details.upstream_inference_cost`` because ``cost`` is 0.0 for them, so
    aggregating a turn that never needed aggregating would silently zero their
    accounting. Genuine multi-call turns still lose the breakdowns, which don't
    sum meaningfully.

    Args:
        usages: Usage from each API call in the turn, in order.

    Returns:
        Optional[Usage]: The turn's usage, or None if no call reported any.
    """
    if not usages:
        return None
    if len(usages) == 1:
        return usages[0]

    total: Optional[Usage] = None
    for usage in usages:
        total = _accumulate_usage(total, usage)
    return total


def _create_completion(
    client: "OpenRouterClient",
    model_id: str,
    messages: List[Dict[str, Any]],
    extra: Dict[str, Any],
) -> ChatCompletionResponse:
    """
    Issue one non-streaming chat completion for the tool loop.

    chat.create() is typed as possibly returning a stream iterator; the loop
    never streams, so the result is narrowed here rather than at each use. It
    also falls back to the raw response dict when the response fails to validate,
    which is checked here so the failure is legible instead of surfacing as an
    AttributeError several frames deeper, possibly after handlers have run.

    Args:
        client: The OpenRouter client to issue the completion with.
        model_id: Model to call.
        messages: Conversation history to send.
        extra: Per-call parameters (tools or response_format) plus caller kwargs.

    Returns:
        ChatCompletionResponse: The parsed completion response.

    Raises:
        APIError: If the response could not be parsed into a completion.
    """
    response = client.chat.create(
        model=model_id, messages=cast(List[Any], messages), **extra
    )

    if not hasattr(response, "choices"):
        raise APIError(
            message="Chat completion response could not be parsed into a "
            "completion, so the tool loop cannot continue.",
            status_code=502,
            details={"response": response},
        )

    if not response.choices:
        raise APIError(
            message="Chat completion response carried no choices, so the tool "
            "loop has no message to act on.",
            status_code=502,
            details={"response": response},
        )

    return cast(ChatCompletionResponse, response)


def _run_tool_rounds(
    client: "OpenRouterClient",
    model_id: str,
    messages: List[Dict[str, Any]],
    tool_loop: ToolLoop,
    chat_kwargs: Dict[str, Any],
) -> TurnResult:
    """
    Call the model with tools attached until it answers without requesting more.

    No ``response_format`` is sent here, so tool calling is never constrained by
    a JSON grammar. ``messages`` is appended to in place with every assistant
    turn (tool_calls included) and every tool result.

    Args:
        client: The OpenRouter client to issue completions with.
        model_id: Model to call.
        messages: Conversation history, appended to in place.
        tool_loop: Tool definitions, handlers, and the round limit.
        chat_kwargs: Extra parameters forwarded to chat.create().

    Returns:
        TurnResult: The model's final content plus usage across the rounds.

    Raises:
        ToolCallLimitExceeded: If the model still wants tools after max_rounds.
        ToolExecutionError: If a tool call cannot be executed.
    """
    usages: List[Usage] = []

    for round_index in range(tool_loop.max_rounds + 1):
        response = _create_completion(
            client, model_id, messages, {"tools": tool_loop.tools, **chat_kwargs}
        )
        if response.usage is not None:
            usages.append(response.usage)

        message = response.choices[0].message
        messages.append(_assistant_message(message))

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return TurnResult(content=message.content, usage=_combine_usage(usages))

        # Check the budget BEFORE executing: handlers have side effects, and
        # running a batch whose results are about to be discarded would spend
        # one more round than max_rounds permits.
        if round_index == tool_loop.max_rounds:
            break

        for raw_call in tool_calls:
            messages.append(
                _execute_tool_call(_read_tool_call(raw_call), tool_loop.handlers)
            )

    raise ToolCallLimitExceeded(
        message=f"Model was still requesting tool calls after "
        f"{tool_loop.max_rounds} rounds. Raise ToolLoop.max_rounds if this "
        f"is expected.",
        max_rounds=tool_loop.max_rounds,
    )


def _run_schema_turn(
    client: "OpenRouterClient",
    model_id: str,
    messages: List[Dict[str, Any]],
    schema: Dict[str, Any],
    chat_kwargs: Dict[str, Any],
) -> TurnResult:
    """
    Make the final structured call, with the schema enforced and tools withheld.

    Withholding tools is what makes the model settle on an answer rather than
    opening another round. ``messages`` is appended to in place.

    A short user-role instruction is appended before the call. The tool rounds
    end on an assistant turn, and a request whose last message is from the
    assistant reads as a prefill to Anthropic-family models — they continue the
    previous prose instead of emitting a fresh object, and some providers reject
    prefill combined with response_format outright. The instruction restores a
    normal user-turn-last shape.

    Args:
        client: The OpenRouter client to issue completions with.
        model_id: Model to call.
        messages: Conversation history, appended to in place.
        schema: JSON schema the answer must conform to.
        chat_kwargs: Extra parameters forwarded to chat.create().

    Returns:
        TurnResult: The structured content plus this call's usage.
    """
    messages.append({"role": "user", "content": SCHEMA_TURN_INSTRUCTION})

    # tool_choice is meaningful during the tool rounds but not here, where tools
    # are withheld on purpose: providers reject tool_choice with no tools.
    extra = {key: value for key, value in chat_kwargs.items() if key != "tool_choice"}
    extra["response_format"] = build_json_schema_response_format(schema)

    response = _create_completion(client, model_id, messages, extra)

    content = _plain_content(response.choices[0].message.content)
    messages.append({"role": "assistant", "content": content})

    return TurnResult(content=content, usage=response.usage)


def _run_tool_loop(
    client: "OpenRouterClient",
    model_id: str,
    messages: List[Dict[str, Any]],
    tool_loop: ToolLoop,
    schema: Optional[Dict[str, Any]],
    chat_kwargs: Dict[str, Any],
) -> TurnResult:
    """
    Drive the model through tool calls until it answers, appending to ``messages``.

    Tool rounds run with ``tools`` attached and no ``response_format``. When a
    schema is given, the structured answer comes from one additional call made
    after the model stops requesting tools, with ``response_format`` attached and
    ``tools`` withheld — the two are kept apart because some providers cannot
    emit a tool call while a strict output schema is enforced.

    ``messages`` is mutated in place: every assistant turn (including its
    ``tool_calls``) and every tool result is appended, so the caller's history
    stays complete and reusable.

    On failure the history is rolled back to where this call found it. A partial
    loop otherwise leaves an assistant turn whose ``tool_calls`` have no matching
    ``role="tool"`` replies, which most providers reject outright — so a caller
    that catches ToolExecutionError and retries would be permanently stuck.

    Args:
        client: The OpenRouter client to issue completions with.
        model_id: Model to call.
        messages: Conversation history, appended to in place.
        tool_loop: Tool definitions, handlers, and the round limit.
        schema: Optional JSON schema for the final structured answer.
        chat_kwargs: Extra parameters forwarded to chat.create().

    Returns:
        TurnResult: The final content plus usage summed across every call made.

    Raises:
        ValueError: If chat_kwargs carries 'tools' or 'response_format', which
            the loop manages itself.
        ToolCallLimitExceeded: If the model still wants tools after max_rounds.
        ToolExecutionError: If a tool call cannot be executed.
    """
    for managed in ("tools", "response_format"):
        if managed in chat_kwargs:
            raise ValueError(
                f"'{managed}' is managed by the tool loop; pass tools via "
                f"ToolLoop(tools=...) and structured output via schema=."
            )

    # Caught here so it fails with a useful message: a stream returns a
    # generator, which would otherwise surface as an opaque "response could not
    # be parsed" APIError blaming the provider.
    if chat_kwargs.get("stream"):
        raise ValueError(
            "stream=True is not supported with tool_loop=; the loop needs "
            "complete responses to detect and execute tool calls."
        )

    history_depth = len(messages)
    try:
        rounds = _run_tool_rounds(client, model_id, messages, tool_loop, chat_kwargs)
        if schema is None:
            return rounds

        final = _run_schema_turn(client, model_id, messages, schema, chat_kwargs)
    except Exception:
        del messages[history_depth:]
        raise

    return TurnResult(
        content=final.content,
        usage=_combine_usage(
            [usage for usage in (rounds.usage, final.usage) if usage is not None]
        ),
    )


def _warn_if_tools_ignored(
    tool_loop: Optional[ToolLoop], kwargs: Dict[str, Any]
) -> None:
    """
    Warn when tools are passed through kwargs without a ToolLoop to execute them.

    Tools sent this way do reach the API, but nothing runs the resulting tool
    calls and their content is typically null — the silent no-op this warning
    exists to make audible.

    Args:
        tool_loop: The tool loop for this call, if any.
        kwargs: Extra parameters headed for chat.create().
    """
    if tool_loop is None and kwargs.get("tools"):
        warnings.warn(
            "tools= was passed without tool_loop=, so any tool calls the model makes "
            "will not be executed and the response content will likely be empty. "
            "Pass tool_loop=ToolLoop(tools=..., handlers=...) to run them "
            "automatically.",
            UserWarning,
            stacklevel=3,
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
        tool_loop: Optional[ToolLoop] = None,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Send a prompt with optional system message, attachments, and structured output.

        Args:
            text: The user prompt text
            system: Optional system prompt to set context/behavior
            attachments: Optional list of file attachments
            schema: Optional JSON schema for structured output
            tool_loop: Optional ToolLoop whose tools are offered to the model and
                whose handlers execute any tool calls it makes, automatically, until
                it produces a final answer
            **kwargs: Additional parameters passed to chat.create()

        Returns:
            str: Response content if no schema provided
            Dict[str, Any]: Parsed and validated JSON response if schema provided.
                           GUARANTEED to be a dict, never a string.

        Raises:
            APIError: If schema is provided but model returns invalid JSON
                     or non-dict response
            ToolExecutionError: If a tool call cannot be executed
            ToolCallLimitExceeded: If the model exceeds the loop's max_rounds
        """
        _warn_if_tools_ignored(tool_loop, kwargs)

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

        # Hand off to the tool loop, which manages its own tools/response_format
        # and may make several API calls before the model settles on an answer.
        if tool_loop is not None:
            result = _run_tool_loop(
                self.client, self.model_id, messages, tool_loop, schema, kwargs
            )
            self.last_usage = result.usage
            if schema:
                return parse_schema_response(result.content, schema)
            return cast(str, result.content)

        # Prepare chat.create() parameters
        chat_params = {
            "model": self.model_id,
            "messages": messages,
            **kwargs,  # Include any additional parameters like temperature
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

    def __init__(
        self, model_id: str, client: "OpenRouterClient", system: Optional[str] = None
    ):
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
        tool_loop: Optional[ToolLoop] = None,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Send a prompt within this conversation context.

        Args:
            text: The user prompt text
            attachments: Optional list of file attachments
            schema: Optional JSON schema for structured output
            tool_loop: Optional ToolLoop whose tools are offered to the model and
                whose handlers execute any tool calls it makes, automatically, until
                it produces a final answer. Every assistant turn (with its tool_calls)
                and every tool result is kept in this conversation's history.
            **kwargs: Additional parameters passed to chat.create()

        Returns:
            str: Response content if no schema provided
            Dict[str, Any]: Parsed and validated JSON response if schema provided.
                           GUARANTEED to be a dict, never a string.

        Raises:
            APIError: If schema is provided but model returns invalid JSON
                     or non-dict response
            ToolExecutionError: If a tool call cannot be executed
            ToolCallLimitExceeded: If the model exceeds the loop's max_rounds
        """
        _warn_if_tools_ignored(tool_loop, kwargs)

        # Depth before this turn's user message, so a failed tool loop can undo
        # the turn entirely rather than leaving the prompt stranded in history.
        history_depth = len(self.messages)

        # Build user message content
        content = [{"type": "text", "text": text}]

        if attachments:
            for attachment in attachments:
                content.append(attachment.to_content_part())

        # Add user message to conversation history
        self.messages.append({"role": "user", "content": content})

        # Hand off to the tool loop, which appends every assistant turn and tool
        # result to this conversation's history as it goes. The schema parse is
        # inside the guard because it, too, can fail on a completed turn.
        if tool_loop is not None:
            try:
                result = _run_tool_loop(
                    self.client, self.model_id, self.messages, tool_loop, schema, kwargs
                )
                self.last_usage = result.usage
                self.total_usage = _accumulate_usage(self.total_usage, self.last_usage)
                if schema:
                    return parse_schema_response(result.content, schema)
                return cast(str, result.content)
            except Exception:
                del self.messages[history_depth:]
                raise

        # Prepare chat.create() parameters
        chat_params = {
            "model": self.model_id,
            "messages": self.messages,
            **kwargs,  # Include any additional parameters like temperature
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
