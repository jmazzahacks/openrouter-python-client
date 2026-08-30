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

from openrouter_client.auth import AuthManager
from openrouter_client.endpoints.chat import ChatEndpoint
from openrouter_client.exceptions import (
    APIError,
    ToolCallLimitExceeded,
    ToolExecutionError,
)
from openrouter_client.http import HTTPManager
from openrouter_client.models.chat import Usage
from openrouter_client.models.core import TextContent
from openrouter_client.models.llm import Conversation, LLMModel, ToolLoop
from openrouter_client.tools import (
    build_chat_completion_tool,
    build_function_definition,
)

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
        # The definitions still travel (the transcript references them and
        # Anthropic-family providers reject requests that omit them), but tool
        # calling is disabled so the model must settle on the answer.
        assert final_call.kwargs["tools"] == [WEATHER_TOOL]
        assert final_call.kwargs["tool_choice"] == "none"

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

        # The whole turn is undone, the prompt's own user message included, so a
        # retry does not stack duplicate user turns.
        assert conversation.messages == []

    def test_history_is_rolled_back_when_the_round_limit_is_hit(self):
        client = Mock()
        client.chat.create.return_value = _response(tool_calls=[_tool_call()])
        conversation = Conversation("test/model", client)

        with pytest.raises(ToolCallLimitExceeded):
            conversation.prompt("weather?", tool_loop=_weather_loop(max_rounds=2))

        assert conversation.messages == []

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


class Test_ToolLoop_07_SecondReviewRegressions:
    """Regressions from the second review pass over the tool loop."""

    def test_retrying_after_a_failure_does_not_stack_user_turns(self):
        # Rollback used to stop just after the turn's own user message, so each
        # caught-and-retried failure left another copy of the prompt behind.
        client = Mock()
        client.chat.create.return_value = _response(
            tool_calls=[_tool_call(name="get_stock_price")]
        )
        conversation = Conversation("test/model", client)

        for _ in range(3):
            with pytest.raises(ToolExecutionError):
                conversation.prompt("weather?", tool_loop=_weather_loop())

        assert conversation.messages == []

    def test_schema_parse_failure_rolls_the_turn_back(self):
        # parse_schema_response ran outside the guarded region, so an invalid
        # final answer left the schema instruction and the junk assistant turn
        # in history — and a retry appended another pair each time.
        client = _client_returning(
            _response(tool_calls=[_tool_call()]),
            _response(content="ready"),
            _response(content="not json at all"),
        )
        conversation = Conversation("test/model", client)

        with pytest.raises(APIError):
            conversation.prompt("go", schema=REPORT_SCHEMA, tool_loop=_weather_loop())

        assert conversation.messages == []

    def test_object_valued_arguments_are_accepted(self):
        # Some providers send function.arguments as an object rather than a JSON
        # string; the strict str field leaked a pydantic ValidationError, which
        # is outside the documented error contract.
        seen = {}

        def handler(city: str) -> str:
            seen["city"] = city
            return "sunny"

        object_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
        }
        client = _client_returning(
            _response(tool_calls=[object_call]),
            _response(content="done"),
        )
        LLMModel("test/model", client).prompt(
            "weather?", tool_loop=_weather_loop(handler)
        )

        assert seen["city"] == "Paris"

    def test_callers_tool_choice_does_not_reach_the_schema_turn(self):
        # The caller's tool_choice belongs to the tool rounds; the schema turn
        # forces "none" so the model must settle instead of opening a round.
        client = _client_returning(
            _response(content="ready"),
            _response(content='{"summary": "ok"}'),
        )
        LLMModel("test/model", client).prompt(
            "hi",
            schema=REPORT_SCHEMA,
            tool_loop=_weather_loop(),
            tool_choice="required",
        )

        rounds_call, schema_call = client.chat.create.call_args_list
        assert rounds_call.kwargs["tool_choice"] == "required"
        assert schema_call.kwargs["tool_choice"] == "none"
        assert schema_call.kwargs["tools"] == [WEATHER_TOOL]

    def test_structured_assistant_content_is_resendable(self):
        # Message.content may be a list of ContentPart models, which requests'
        # json= cannot encode when the history goes back out next round.
        content_parts = [TextContent(type="text", text="thinking")]
        sent = []

        def record(**kwargs):
            json.dumps(kwargs["messages"])  # raises before the fix
            sent.append(kwargs)
            return [
                _response(tool_calls=[_tool_call()], content=content_parts),
                _response(content="done"),
            ][len(sent) - 1]

        client = Mock()
        client.chat.create.side_effect = record
        conversation = Conversation("test/model", client)

        assert conversation.prompt("hi", tool_loop=_weather_loop()) == "done"
        assert conversation.messages[1]["content"] == [
            {"type": "text", "text": "thinking"}
        ]

    def test_empty_choices_raises_api_error(self):
        response = Mock()
        response.usage = None
        response.choices = []
        client = Mock()
        client.chat.create.return_value = response

        with pytest.raises(APIError, match="no choices"):
            LLMModel("test/model", client).prompt("hi", tool_loop=_weather_loop())

    def test_stream_with_tool_loop_is_rejected_clearly(self):
        # A stream returns a generator, which would otherwise surface as an
        # opaque "could not be parsed" APIError blaming the provider.
        client = Mock()

        with pytest.raises(ValueError, match="stream=True is not supported"):
            LLMModel("test/model", client).prompt(
                "hi", tool_loop=_weather_loop(), stream=True
            )

        client.chat.create.assert_not_called()


class Test_ToolLoop_08_ThirdReviewRegressions:
    """Regressions from the third review pass over the tool loop."""

    def test_forced_tool_choice_is_relaxed_after_the_first_round(self):
        # tool_choice="required" forwarded into every round makes the exit
        # condition (a response with no tool_calls) unsatisfiable: the loop
        # burned its whole budget and raised ToolCallLimitExceeded every time.
        client = _client_returning(
            _response(tool_calls=[_tool_call()]),
            _response(tool_calls=[_tool_call(call_id="c2")]),
            _response(content="done"),
        )
        result = LLMModel("test/model", client).prompt(
            "hi", tool_loop=_weather_loop(), tool_choice="required"
        )

        assert result == "done"
        choices = [
            "tool_choice" in call.kwargs for call in client.chat.create.call_args_list
        ]
        assert choices == [True, False, False]

    def test_tool_free_follow_up_resends_tool_definitions(self):
        # A successful tool turn leaves tool_calls / role="tool" messages in
        # history; a later tool-free prompt() used to send them with no tools
        # parameter, which Anthropic-family providers reject with a 400.
        client = _client_returning(
            _response(tool_calls=[_tool_call()]),
            _response(content="sunny"),
            _response(content="cloudy tomorrow"),
        )
        conversation = Conversation("test/model", client)
        conversation.prompt("weather?", tool_loop=_weather_loop())
        conversation.prompt("and tomorrow?")

        follow_up = client.chat.create.call_args_list[-1].kwargs
        assert follow_up["tools"] == [WEATHER_TOOL]
        # Definitions only — nothing here would execute a call the model made.
        assert follow_up["tool_choice"] == "none"

    def test_clear_forgets_the_retained_tool_definitions(self):
        client = _client_returning(
            _response(tool_calls=[_tool_call()]),
            _response(content="sunny"),
            _response(content="fresh start"),
        )
        conversation = Conversation("test/model", client)
        conversation.prompt("weather?", tool_loop=_weather_loop())
        conversation.clear()
        conversation.prompt("new topic")

        assert "tools" not in client.chat.create.call_args.kwargs

    def test_empty_schema_dict_behaves_as_no_schema(self):
        # _run_tool_loop tested `schema is None` while the call sites tested
        # `if schema:`, so schema={} paid for the schema turn and then returned
        # a raw string anyway.
        client = _client_returning(_response(content="plain answer"))
        result = LLMModel("test/model", client).prompt(
            "hi", schema={}, tool_loop=_weather_loop()
        )

        assert result == "plain answer"
        assert client.chat.create.call_count == 1  # no schema turn was paid for

    def test_keyboard_interrupt_in_handler_still_rolls_back(self):
        # `except Exception` skipped rollback for BaseException, leaving the
        # provider-rejected unanswered-tool_calls shape in history after Ctrl-C.
        def handler(city: str) -> str:
            raise KeyboardInterrupt

        client = Mock()
        client.chat.create.return_value = _response(tool_calls=[_tool_call()])
        conversation = Conversation("test/model", client)

        with pytest.raises(KeyboardInterrupt):
            conversation.prompt("weather?", tool_loop=_weather_loop(handler))

        assert conversation.messages == []

    def test_schema_parse_failure_does_not_record_usage(self):
        # Usage was recorded before parse_schema_response, so each failed parse
        # added a rolled-back turn's tokens to total_usage, violating the
        # documented "updated only on a successful turn" invariant.
        client = _client_returning(
            _response(content="ready", usage=_usage(10, 5, 0.001)),
            _response(content="not json", usage=_usage(10, 5, 0.001)),
        )
        conversation = Conversation("test/model", client)

        with pytest.raises(APIError):
            conversation.prompt("go", schema=REPORT_SCHEMA, tool_loop=_weather_loop())

        assert conversation.last_usage is None
        assert conversation.total_usage is None

    def test_parallel_tool_calls_is_stripped_from_the_schema_turn(self):
        client = _client_returning(
            _response(content="ready"),
            _response(content='{"summary": "ok"}'),
        )
        LLMModel("test/model", client).prompt(
            "hi",
            schema=REPORT_SCHEMA,
            tool_loop=_weather_loop(),
            parallel_tool_calls=False,
        )

        rounds_call, schema_call = client.chat.create.call_args_list
        assert rounds_call.kwargs["parallel_tool_calls"] is False
        assert "parallel_tool_calls" not in schema_call.kwargs

    def test_list_content_final_answer_is_returned_as_text(self):
        # The final answer escaped raw when a provider returned content parts,
        # handing a list of pydantic models to a caller promised a str.
        parts = [
            TextContent(type="text", text="hello "),
            TextContent(type="text", text="world"),
        ]
        client = _client_returning(_response(content=parts))
        result = LLMModel("test/model", client).prompt("hi", tool_loop=_weather_loop())

        assert result == "hello world"

    def test_nested_pydantic_model_in_tool_dict_is_serialized(self):
        # The validator only dumped top-level models, so a FunctionDefinition
        # nested in a hand-written dict still broke requests' json encoding.
        def my_func(city: str) -> str:
            """Do a thing."""
            return city

        loop = ToolLoop(
            tools=[
                {"type": "function", "function": build_function_definition(my_func)}
            ],
            handlers={"my_func": my_func},
        )

        json.dumps({"tools": loop.tools})  # raises TypeError before the fix
        assert loop.tools[0]["function"]["name"] == "my_func"

    def test_non_string_tool_call_id_is_coerced(self):
        # A provider sending id/name as non-strings leaked a pydantic
        # ValidationError outside the ToolExecutionError contract.
        seen = {}

        def handler(city: str) -> str:
            seen["city"] = city
            return "sunny"

        int_id_call = {
            "id": 123,
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
        client = _client_returning(
            _response(tool_calls=[int_id_call]),
            _response(content="done"),
        )
        conversation = Conversation("test/model", client)
        conversation.prompt("weather?", tool_loop=_weather_loop(handler))

        assert seen["city"] == "Paris"
        assert conversation.messages[2]["tool_call_id"] == "123"

    def test_chat_create_serializes_pydantic_tools_at_the_endpoint(self):
        # The gap lived in ChatEndpoint.create() itself: data["tools"] = tools
        # went to requests as json= untouched in BOTH validate modes, so the
        # low-level API broke for the tools parameter's own documented type.
        endpoint = ChatEndpoint(Mock(spec=AuthManager), Mock(spec=HTTPManager))
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")

        def my_tool(city: str) -> str:
            """Get weather."""
            return city

        post_response = Mock()
        post_response.json.return_value = {"unvalidatable": True}
        endpoint.http_manager.post = Mock(return_value=post_response)

        endpoint.create(
            messages=[{"role": "user", "content": "hi"}],
            model="test/model",
            tools=[build_chat_completion_tool(my_tool)],
        )

        sent = endpoint.http_manager.post.call_args.kwargs["json"]
        json.dumps(sent)  # raises TypeError before the fix
        assert sent["tools"][0]["function"]["name"] == "my_tool"


class Test_ToolLoop_09_FourthReviewRegressions:
    """Regressions from the fourth review pass over the tool loop."""

    def test_schema_turn_defines_tools_with_choice_none(self):
        # The schema turn withheld tools while sending a transcript full of
        # tool_calls / role="tool" messages — the exact shape this library's
        # own follow-up mechanism says Anthropic-family providers reject.
        client = _client_returning(
            _response(tool_calls=[_tool_call()]),
            _response(content="ready"),
            _response(content='{"summary": "ok"}'),
        )
        LLMModel("test/model", client).prompt(
            "hi", schema=REPORT_SCHEMA, tool_loop=_weather_loop()
        )

        schema_call = client.chat.create.call_args_list[-1].kwargs
        assert schema_call["tools"] == [WEATHER_TOOL]
        assert schema_call["tool_choice"] == "none"

    def test_null_final_content_returns_empty_string(self):
        # {content: null, tool_calls: null} (reasoning-only output, content
        # filter, length stop) escaped as None where the docs promise str.
        client = _client_returning(_response(content=None))
        result = LLMModel("test/model", client).prompt(
            "hi", tool_loop=_weather_loop()
        )

        assert result == ""

    def test_null_final_content_with_schema_raises_empty_response(self):
        client = _client_returning(
            _response(content="ready"),
            _response(content=None),
        )
        with pytest.raises(APIError, match="empty response"):
            LLMModel("test/model", client).prompt(
                "hi", schema=REPORT_SCHEMA, tool_loop=_weather_loop()
            )

    def test_non_loop_schema_failure_records_nothing(self):
        # The plain schema path recorded usage and kept the junk assistant
        # message before parse_schema_response raised — the same invariant
        # violation fixed for the loop path in an earlier round.
        client = _client_returning(
            _response(content="not json", usage=_usage(10, 5, 0.5)),
        )
        conversation = Conversation("test/model", client)

        with pytest.raises(APIError):
            conversation.prompt("x", schema=REPORT_SCHEMA)

        assert conversation.total_usage is None
        assert conversation.last_usage is None
        # The unparseable assistant message is not kept for a retry to re-send.
        assert [m["role"] for m in conversation.messages] == ["user"]

    def test_llm_model_non_loop_schema_failure_keeps_prior_usage(self):
        client = _client_returning(
            _response(content='{"summary": "ok"}', usage=_usage(10, 5, 0.001)),
            _response(content="not json", usage=_usage(99, 99, 9.9)),
        )
        model = LLMModel("test/model", client)
        model.prompt("first", schema=REPORT_SCHEMA)

        with pytest.raises(APIError):
            model.prompt("second", schema=REPORT_SCHEMA)

        # last_usage retains the prior successful call's value, per its docs.
        assert model.last_usage.cost == pytest.approx(0.001)

    def test_chat_create_serializes_functions_and_tool_choice(self):
        # create() dumped pydantic models for tools= but passed functions= and
        # tool_choice= (the signature's own documented types) into json= raw.
        endpoint = ChatEndpoint(Mock(spec=AuthManager), Mock(spec=HTTPManager))
        endpoint.logger = Mock()
        endpoint._get_headers = Mock(return_value={})
        endpoint._get_endpoint_url = Mock(return_value="chat/completions")
        post_response = Mock()
        post_response.json.return_value = {"unvalidatable": True}
        endpoint.http_manager.post = Mock(return_value=post_response)

        def my_func(city: str) -> str:
            """Get weather."""
            return city

        endpoint.create(
            messages=[{"role": "user", "content": "hi"}],
            model="test/model",
            functions=[build_function_definition(my_func)],
            tool_choice={
                "type": "function",
                "function": build_function_definition(my_func),
            },
        )

        sent = endpoint.http_manager.post.call_args.kwargs["json"]
        json.dumps(sent)  # raised TypeError before the fix
        assert sent["functions"][0]["name"] == "my_func"

    def test_second_tool_loop_merges_definitions_and_handlers(self):
        # A second ToolLoop overwrote the retained definitions, sending only
        # the new loop's tools over a transcript that still references the
        # first loop's tool calls.
        TIME_TOOL = {
            "type": "function",
            "function": {"name": "get_time", "parameters": {"type": "object"}},
        }

        def get_time() -> str:
            return "noon"

        time_loop = ToolLoop(tools=[TIME_TOOL], handlers={"get_time": get_time})
        client = _client_returning(
            _response(tool_calls=[_tool_call()]),      # turn 1 uses get_weather
            _response(content="sunny"),
            _response(tool_calls=[_tool_call(name="get_time", arguments="")]),
            _response(content="it is noon"),
        )
        conversation = Conversation("test/model", client)
        conversation.prompt("weather?", tool_loop=_weather_loop())
        result = conversation.prompt("time?", tool_loop=time_loop)

        assert result == "it is noon"
        # Turn 2's rounds must define BOTH loops' tools — the history still
        # references get_weather's calls.
        turn2_tools = client.chat.create.call_args_list[2].kwargs["tools"]
        names = {t["function"]["name"] for t in turn2_tools}
        assert names == {"get_weather", "get_time"}

    def test_zero_tool_call_turn_retains_nothing(self):
        # Retention fired even when the model never called a tool, billing tool
        # schemas into every later turn of the conversation for no reason.
        client = _client_returning(
            _response(content="no tools needed"),
            _response(content="follow-up answer"),
        )
        conversation = Conversation("test/model", client)
        conversation.prompt("hi", tool_loop=_weather_loop())
        conversation.prompt("more")

        assert "tools" not in client.chat.create.call_args.kwargs

    def test_retained_tools_are_not_aliased_to_the_callers_loop(self):
        # _history_tools stored a live reference to ToolLoop.tools, so caller
        # mutations (loop.tools.clear()) leaked into later requests.
        loop = _weather_loop()
        client = _client_returning(
            _response(tool_calls=[_tool_call()]),
            _response(content="sunny"),
            _response(content="later"),
        )
        conversation = Conversation("test/model", client)
        conversation.prompt("weather?", tool_loop=loop)
        loop.tools.clear()
        conversation.prompt("and tomorrow?")

        assert client.chat.create.call_args.kwargs["tools"] == [WEATHER_TOOL]

    def test_explicit_tools_none_does_not_suppress_reinjection(self):
        # tools=None means "not provided" throughout this API; passing it
        # explicitly used to skip the re-send and hit the undefined-tools 400.
        client = _client_returning(
            _response(tool_calls=[_tool_call()]),
            _response(content="sunny"),
            _response(content="later"),
        )
        conversation = Conversation("test/model", client)
        conversation.prompt("weather?", tool_loop=_weather_loop())
        conversation.prompt("more", tools=None)

        assert client.chat.create.call_args.kwargs["tools"] == [WEATHER_TOOL]

    def test_definitions_only_tools_with_choice_none_does_not_warn(self):
        # The docs sanction explicit tools= with tool_choice="none" (nothing
        # can be called), but the no-loop warning still fired on it — a hard
        # failure under warnings-as-errors configs.
        import warnings as warnings_module

        client = _client_returning(_response(content="ok"))
        with warnings_module.catch_warnings():
            warnings_module.simplefilter("error")
            Conversation("test/model", client).prompt(
                "hi", tools=[WEATHER_TOOL], tool_choice="none"
            )
