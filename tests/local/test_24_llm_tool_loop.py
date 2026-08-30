"""
Unit tests for the automated tool-call loop on the high-level llm-style API.

Covers:
- LLMModel.prompt() / Conversation.prompt() executing model-requested tool calls
  via ToolLoop handlers and feeding results back until the model answers,
- conversation history keeping assistant tool_calls and role="tool" results,
- the tools + schema interaction (tool rounds unconstrained, one final
  response_format turn with tools withheld),
- usage summed across every API call a single turn made, and
- the error paths: unknown tool, bad arguments, raising handler, round limit,
  and the warning when tools are passed without a loop to run them.

The chat endpoint is mocked so no network or API key is needed; mocked responses
mirror ChatCompletionResponse's shape (choices[0].message + usage).
"""

import json
from unittest.mock import Mock

import pytest

from openrouter_client.exceptions import (
    APIError,
    ToolCallLimitExceeded,
    ToolExecutionError,
)
from openrouter_client.models.chat import Usage
from openrouter_client.models.llm import Conversation, LLMModel, ToolLoop
from openrouter_client.tools import build_chat_completion_tool

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

REPORT_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


def _usage(prompt=10, completion=5, cost=0.001) -> Usage:
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cost=cost,
    )


def _response(content=None, tool_calls=None, usage=None) -> Mock:
    """Build a stand-in for ChatCompletionResponse with an optional tool_calls block."""
    response = Mock()
    response.usage = usage
    message = Mock()
    message.content = content
    message.tool_calls = tool_calls
    choice = Mock()
    choice.message = message
    response.choices = [choice]
    return response


def _tool_call(
    name="get_weather", arguments='{"city": "Paris"}', call_id="call_1"
) -> dict:
    """A tool call in the raw dict shape the API returns."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _client_returning(*responses) -> Mock:
    client = Mock()
    client.chat.create.side_effect = list(responses)
    return client


def _weather_loop(handler=None, max_rounds=8) -> ToolLoop:
    if handler is None:

        def handler(city: str) -> str:
            return f"sunny in {city}"

    return ToolLoop(
        tools=[WEATHER_TOOL], handlers={"get_weather": handler}, max_rounds=max_rounds
    )


class Test_ToolLoop_01_NominalBehaviors:
    """A tool call is executed and its result fed back for a final answer."""

    def test_executes_tool_and_returns_final_answer(self):
        client = _client_returning(
            _response(tool_calls=[_tool_call()]),
            _response(content="It is sunny in Paris."),
        )
        model = LLMModel("test/model", client)

        result = model.prompt("What's the weather in Paris?", tool_loop=_weather_loop())

        assert result == "It is sunny in Paris."
        assert client.chat.create.call_count == 2

    def test_handler_receives_model_arguments(self):
        seen = {}

        def handler(city: str) -> str:
            seen["city"] = city
            return "sunny"

        client = _client_returning(
            _response(tool_calls=[_tool_call(arguments='{"city": "Berlin"}')]),
            _response(content="done"),
        )
        LLMModel("test/model", client).prompt(
            "weather?", tool_loop=_weather_loop(handler)
        )

        assert seen["city"] == "Berlin"

    def test_tools_are_sent_on_every_round(self):
        client = _client_returning(
            _response(tool_calls=[_tool_call()]),
            _response(content="done"),
        )
        LLMModel("test/model", client).prompt("weather?", tool_loop=_weather_loop())

        for call in client.chat.create.call_args_list:
            assert call.kwargs["tools"] == [WEATHER_TOOL]

    def test_no_tool_calls_means_single_round(self):
        client = _client_returning(_response(content="No tools needed."))
        result = LLMModel("test/model", client).prompt("hi", tool_loop=_weather_loop())

        assert result == "No tools needed."
        assert client.chat.create.call_count == 1

    def test_multiple_tool_calls_in_one_round_all_execute(self):
        calls = []

        def handler(city: str) -> str:
            calls.append(city)
            return "sunny"

        client = _client_returning(
            _response(
                tool_calls=[
                    _tool_call(arguments='{"city": "Paris"}', call_id="a"),
                    _tool_call(arguments='{"city": "Rome"}', call_id="b"),
                ]
            ),
            _response(content="done"),
        )
        LLMModel("test/model", client).prompt(
            "weather?", tool_loop=_weather_loop(handler)
        )

        assert calls == ["Paris", "Rome"]

    def test_zero_argument_tool_with_empty_arguments_string(self):
        def handler() -> str:
            return "pong"

        loop = ToolLoop(tools=[WEATHER_TOOL], handlers={"ping": handler})
        client = _client_returning(
            _response(tool_calls=[_tool_call(name="ping", arguments="")]),
            _response(content="done"),
        )

        assert LLMModel("test/model", client).prompt("ping", tool_loop=loop) == "done"


class Test_ToolLoop_02_History:
    """The conversation history keeps tool_calls and tool results intact."""

    def test_assistant_tool_calls_and_results_are_recorded(self):
        tool_calls = [_tool_call()]
        client = _client_returning(
            _response(tool_calls=tool_calls),
            _response(content="It is sunny."),
        )
        conversation = Conversation("test/model", client)
        conversation.prompt("weather?", tool_loop=_weather_loop())

        roles = [message["role"] for message in conversation.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]

        assistant_turn = conversation.messages[1]
        assert assistant_turn["tool_calls"] == tool_calls

        tool_result = conversation.messages[2]
        assert tool_result["tool_call_id"] == "call_1"
        assert tool_result["content"] == "sunny in Paris"

    def test_history_is_reused_on_the_next_turn(self):
        client = _client_returning(
            _response(tool_calls=[_tool_call()]),
            _response(content="sunny"),
            _response(content="still sunny"),
        )
        conversation = Conversation("test/model", client)
        conversation.prompt("weather?", tool_loop=_weather_loop())
        conversation.prompt("and tomorrow?")

        # The second turn builds on the first, tool round-trip included.
        assert [message["role"] for message in conversation.messages] == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "user",
            "assistant",
        ]
        assert (
            client.chat.create.call_args_list[-1].kwargs["messages"]
            is conversation.messages
        )

    def test_non_string_result_is_json_encoded(self):
        def handler(city: str) -> dict:
            return {"temp_c": 21, "city": city}

        client = _client_returning(
            _response(tool_calls=[_tool_call()]),
            _response(content="done"),
        )
        conversation = Conversation("test/model", client)
        conversation.prompt("weather?", tool_loop=_weather_loop(handler))

        assert json.loads(conversation.messages[2]["content"]) == {
            "temp_c": 21,
            "city": "Paris",
        }


class Test_ToolLoop_03_SchemaInteraction:
    """Tool rounds run unconstrained; the schema is enforced on a final tools-free turn."""

    def test_schema_applied_only_on_final_turn(self):
        client = _client_returning(
            _response(tool_calls=[_tool_call()]),
            _response(content="I have the weather now."),
            _response(content='{"summary": "sunny in Paris"}'),
        )
        model = LLMModel("test/model", client)

        result = model.prompt(
            "weather?", schema=REPORT_SCHEMA, tool_loop=_weather_loop()
        )

        assert result == {"summary": "sunny in Paris"}
        assert client.chat.create.call_count == 3

        tool_rounds = client.chat.create.call_args_list[:2]
        for call in tool_rounds:
            assert "response_format" not in call.kwargs
            assert call.kwargs["tools"] == [WEATHER_TOOL]

        final_call = client.chat.create.call_args_list[-1]
        assert final_call.kwargs["response_format"]["type"] == "json_schema"
        assert "tools" not in final_call.kwargs

    def test_tool_call_turn_no_longer_misparsed_as_invalid_json(self):
        # Regression: a null-content tool-call turn used to reach parse_schema_response
        # and surface as a confusing "invalid JSON" error.
        client = _client_returning(
            _response(tool_calls=[_tool_call()]),
            _response(content="ready"),
            _response(content='{"summary": "ok"}'),
        )
        result = LLMModel("test/model", client).prompt(
            "weather?", schema=REPORT_SCHEMA, tool_loop=_weather_loop()
        )
        assert result == {"summary": "ok"}

    def test_invalid_final_json_still_raises_api_error(self):
        client = _client_returning(
            _response(content="no tools needed"),
            _response(content="not json at all"),
        )
        with pytest.raises(APIError):
            LLMModel("test/model", client).prompt(
                "hi", schema=REPORT_SCHEMA, tool_loop=_weather_loop()
            )

    def test_schema_without_tool_loop_keeps_single_call_path(self):
        client = _client_returning(_response(content='{"summary": "ok"}'))
        result = LLMModel("test/model", client).prompt("hi", schema=REPORT_SCHEMA)

        assert result == {"summary": "ok"}
        assert client.chat.create.call_count == 1
        assert "response_format" in client.chat.create.call_args.kwargs


class Test_ToolLoop_04_Usage:
    """Usage is summed across every API call a single turn made."""

    def test_last_usage_sums_all_calls_in_the_turn(self):
        client = _client_returning(
            _response(tool_calls=[_tool_call()], usage=_usage(10, 5, 0.001)),
            _response(content="done", usage=_usage(20, 10, 0.002)),
        )
        model = LLMModel("test/model", client)
        model.prompt("weather?", tool_loop=_weather_loop())

        assert model.last_usage.prompt_tokens == 30
        assert model.last_usage.completion_tokens == 15
        assert model.last_usage.cost == pytest.approx(0.003)

    def test_conversation_total_usage_accumulates_across_turns(self):
        client = _client_returning(
            _response(tool_calls=[_tool_call()], usage=_usage(10, 5, 0.001)),
            _response(content="done", usage=_usage(10, 5, 0.001)),
            _response(content="again", usage=_usage(10, 5, 0.001)),
        )
        conversation = Conversation("test/model", client)
        conversation.prompt("weather?", tool_loop=_weather_loop())
        conversation.prompt("thanks")

        assert conversation.total_usage.total_tokens == 45
        assert conversation.total_cost == pytest.approx(0.003)


class Test_ToolLoop_05_ErrorConditions:
    """Tool-loop failures surface as specific, actionable exceptions."""

    def test_unknown_tool_name_raises(self):
        client = _client_returning(
            _response(tool_calls=[_tool_call(name="get_stock_price")])
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            LLMModel("test/model", client).prompt("hi", tool_loop=_weather_loop())

        assert exc_info.value.tool_name == "get_stock_price"

    def test_invalid_json_arguments_raise(self):
        client = _client_returning(
            _response(tool_calls=[_tool_call(arguments="{not json")])
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            LLMModel("test/model", client).prompt("hi", tool_loop=_weather_loop())

        assert "invalid JSON arguments" in str(exc_info.value)

    def test_non_object_arguments_raise(self):
        client = _client_returning(
            _response(tool_calls=[_tool_call(arguments="[1, 2]")])
        )

        with pytest.raises(ToolExecutionError):
            LLMModel("test/model", client).prompt("hi", tool_loop=_weather_loop())

    def test_raising_handler_is_wrapped_with_original_error(self):
        def handler(city: str) -> str:
            raise RuntimeError("upstream down")

        client = _client_returning(_response(tool_calls=[_tool_call()]))

        with pytest.raises(ToolExecutionError) as exc_info:
            LLMModel("test/model", client).prompt(
                "hi", tool_loop=_weather_loop(handler)
            )

        assert isinstance(exc_info.value.original_error, RuntimeError)
        assert exc_info.value.tool_call_id == "call_1"

    def test_round_limit_raises(self):
        client = Mock()
        client.chat.create.return_value = _response(tool_calls=[_tool_call()])

        with pytest.raises(ToolCallLimitExceeded) as exc_info:
            LLMModel("test/model", client).prompt(
                "hi", tool_loop=_weather_loop(max_rounds=3)
            )

        assert exc_info.value.max_rounds == 3
        # Three tool-executing rounds, plus the call that proved it was still
        # asking. That last round must NOT execute handlers — see
        # Test_ToolLoop_06.
        assert client.chat.create.call_count == 4

    def test_tools_in_kwargs_conflicts_with_tool_loop(self):
        client = _client_returning(_response(content="done"))

        with pytest.raises(ValueError, match="managed by the tool loop"):
            LLMModel("test/model", client).prompt(
                "hi", tool_loop=_weather_loop(), tools=[WEATHER_TOOL]
            )

    def test_response_format_in_kwargs_conflicts_with_tool_loop(self):
        client = _client_returning(_response(content="done"))

        with pytest.raises(ValueError, match="managed by the tool loop"):
            LLMModel("test/model", client).prompt(
                "hi", tool_loop=_weather_loop(), response_format={"type": "json_object"}
            )

    def test_tools_without_loop_warns_instead_of_silently_dropping(self):
        client = _client_returning(_response(content=None))

        with pytest.warns(UserWarning, match="tool_loop"):
            LLMModel("test/model", client).prompt("hi", tools=[WEATHER_TOOL])

    def test_conversation_tools_without_loop_warns(self):
        client = _client_returning(_response(content=None))

        with pytest.warns(UserWarning, match="tool_loop"):
            Conversation("test/model", client).prompt("hi", tools=[WEATHER_TOOL])


class Test_ToolLoop_06_ReviewRegressions:
    """
    Regressions found in code review of the original tool-loop implementation.

    Each of these passed silently before the fix, which is why they are pinned
    here rather than left to the behavioral tests above.
    """

    def test_pydantic_tool_definitions_are_serialized(self):
        # ToolLoop accepted ChatCompletionTool objects unchanged, and
        # chat.create() passes tools straight to requests as json= unless
        # validate_request=True, so the documented way of building tools raised
        # "Object of type ChatCompletionTool is not JSON serializable".
        def get_weather(city: str) -> dict:
            """Get the weather for a city."""
            return {"city": city}

        loop = ToolLoop(
            tools=[build_chat_completion_tool(get_weather)],
            handlers={"get_weather": get_weather},
        )

        assert isinstance(loop.tools[0], dict)
        json.dumps({"tools": loop.tools})  # would raise TypeError before the fix
        assert loop.tools[0]["function"]["name"] == "get_weather"

    def test_plain_dict_tools_are_left_alone(self):
        loop = _weather_loop()
        assert loop.tools == [WEATHER_TOOL]

    def test_round_limit_does_not_execute_an_extra_handler_batch(self):
        # range(max_rounds + 1) used to execute handlers on the round that blew
        # the budget, then discard the results — one real side effect past the
        # stated limit.
        runs = []

        def handler(city: str) -> str:
            runs.append(city)
            return "sunny"

        client = Mock()
        client.chat.create.return_value = _response(tool_calls=[_tool_call()])

        with pytest.raises(ToolCallLimitExceeded):
            LLMModel("test/model", client).prompt(
                "hi", tool_loop=_weather_loop(handler, max_rounds=3)
            )

        assert len(runs) == 3
        assert client.chat.create.call_count == 4

    def test_history_is_rolled_back_when_a_tool_fails(self):
        # A partial loop left an assistant turn whose tool_calls had no matching
        # role="tool" replies; most providers reject that history, so a caller
        # catching the error and retrying was permanently stuck.
        client = Mock()
        client.chat.create.return_value = _response(
            tool_calls=[_tool_call(name="get_stock_price")]
        )
        conversation = Conversation("test/model", client)

        with pytest.raises(ToolExecutionError):
            conversation.prompt("weather?", tool_loop=_weather_loop())

        assert [message["role"] for message in conversation.messages] == ["user"]
        assert not any("tool_calls" in message for message in conversation.messages)

    def test_history_is_rolled_back_when_the_round_limit_is_hit(self):
        client = Mock()
        client.chat.create.return_value = _response(tool_calls=[_tool_call()])
        conversation = Conversation("test/model", client)

        with pytest.raises(ToolCallLimitExceeded):
            conversation.prompt("weather?", tool_loop=_weather_loop(max_rounds=2))

        assert [message["role"] for message in conversation.messages] == ["user"]

    def test_successful_turn_still_keeps_its_history(self):
        # The rollback must not fire on the happy path.
        client = _client_returning(
            _response(tool_calls=[_tool_call()]),
            _response(content="sunny"),
        )
        conversation = Conversation("test/model", client)
        conversation.prompt("weather?", tool_loop=_weather_loop())

        assert [message["role"] for message in conversation.messages] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]

    def test_single_call_turn_preserves_usage_detail(self):
        # Routing every tool-loop turn through _accumulate_usage dropped
        # cost_details/is_byok even when only one API call was made. BYOK callers
        # read real spend from cost_details.upstream_inference_cost, so this
        # silently zeroed their accounting the moment they passed tool_loop=.
        usage = Usage.model_validate(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.0,
                "cost_details": {"upstream_inference_cost": 19},
                "is_byok": True,
            }
        )
        client = _client_returning(_response(content="no tools needed", usage=usage))
        model = LLMModel("test/model", client)
        model.prompt("hi", tool_loop=_weather_loop())

        assert model.last_usage.cost_details.upstream_inference_cost == 19
        assert model.last_usage.is_byok is True

    def test_multi_call_turn_still_aggregates(self):
        client = _client_returning(
            _response(tool_calls=[_tool_call()], usage=_usage(10, 5, 0.001)),
            _response(content="done", usage=_usage(20, 10, 0.002)),
        )
        model = LLMModel("test/model", client)
        model.prompt("weather?", tool_loop=_weather_loop())

        assert model.last_usage.total_tokens == 45
        assert model.last_usage.cost == pytest.approx(0.003)

    def test_schema_turn_is_not_sent_as_an_assistant_prefill(self):
        # Tool rounds end on an assistant turn; issuing the schema call on that
        # history reads as a prefill to Anthropic-family models, which continue
        # the prose instead of emitting a fresh object.
        seen_roles = []

        def record(**kwargs):
            seen_roles.append([m["role"] for m in kwargs["messages"]])
            index = len(seen_roles) - 1
            return [
                _response(tool_calls=[_tool_call()]),
                _response(content="I have the weather now."),
                _response(content='{"summary": "sunny"}'),
            ][index]

        client = Mock()
        client.chat.create.side_effect = record

        result = LLMModel("test/model", client).prompt(
            "weather?", schema=REPORT_SCHEMA, tool_loop=_weather_loop()
        )

        assert result == {"summary": "sunny"}
        # The final, schema-enforced call must end on a user turn.
        assert seen_roles[-1][-1] == "user"
        assert seen_roles[-1] == ["user", "assistant", "tool", "assistant", "user"]

    def test_unparsed_response_raises_api_error(self):
        # chat.create() falls back to the raw dict when validation fails; the
        # cast() is a runtime no-op, so this used to surface as AttributeError
        # deep in the loop, possibly after handlers had already run.
        client = Mock()
        client.chat.create.return_value = {"unexpected": "shape"}

        with pytest.raises(APIError, match="could not be parsed"):
            LLMModel("test/model", client).prompt("hi", tool_loop=_weather_loop())
