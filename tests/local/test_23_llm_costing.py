"""
Unit tests for cost/usage capture on the high-level llm-style API.

Covers:
- the extended Usage model parsing OpenRouter's inline cost block
  (cost, cost_details, prompt/completion token details), and
- LLMModel.prompt() / Conversation.prompt() exposing per-turn usage via
  last_usage, plus the conversation's cumulative total_usage / total_cost.

The chat endpoint is mocked so no network or API key is needed; the mocked
response object mirrors ChatCompletionResponse's shape (choices + usage).
"""

from unittest.mock import Mock

from openrouter_client.models.chat import Usage, CostDetails
from openrouter_client.models.llm import LLMModel, _accumulate_usage


def _usage(prompt=10, completion=5, cost=0.001) -> Usage:
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cost=cost,
    )


def _mock_response(content, usage) -> Mock:
    """Build a stand-in for ChatCompletionResponse: choices[0].message.content + usage."""
    response = Mock()
    response.usage = usage
    message = Mock()
    message.content = content
    choice = Mock()
    choice.message = message
    response.choices = [choice]
    return response


def _model_returning(*responses) -> LLMModel:
    """An LLMModel whose chat.create yields the given responses in order."""
    client = Mock()
    client.chat.create.side_effect = list(responses)
    return LLMModel("test/model", client)


class Test_Usage_01_ParsesCostBlock:
    """The Usage model captures OpenRouter's inline cost accounting."""

    def test_parses_full_cost_block(self):
        raw = {
            "prompt_tokens": 194,
            "completion_tokens": 2,
            "total_tokens": 196,
            "cost": 0.95,
            "cost_details": {"upstream_inference_cost": 19, "cache_discount": 0.01},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 100,
                "audio_tokens": 0,
            },
            "completion_tokens_details": {"reasoning_tokens": 0},
            "is_byok": False,
        }
        usage = Usage.model_validate(raw)
        assert usage.cost == 0.95
        assert usage.cost_details.upstream_inference_cost == 19
        assert usage.cost_details.cache_discount == 0.01
        assert usage.prompt_tokens_details.cache_write_tokens == 100
        assert usage.completion_tokens_details.reasoning_tokens == 0
        assert usage.is_byok is False

    def test_backward_compatible_tokens_only(self):
        # Older/mocked responses without a cost block still validate; cost is None.
        usage = Usage.model_validate(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        assert usage.cost is None
        assert usage.cost_details is None


class Test_LLMModel_02_LastUsage:
    """LLMModel.prompt() exposes the latest turn's usage via last_usage."""

    def test_last_usage_defaults_to_none(self):
        model = _model_returning()
        assert model.last_usage is None

    def test_prompt_sets_last_usage_and_returns_str(self):
        usage = _usage(cost=0.0006)
        model = _model_returning(_mock_response("hello", usage))

        result = model.prompt("hi")

        assert result == "hello"  # str return preserved
        assert model.last_usage is usage
        assert model.last_usage.cost == 0.0006

    def test_prompt_with_schema_still_returns_dict_and_sets_usage(self):
        usage = _usage(cost=0.002)
        model = _model_returning(_mock_response('{"answer": 42}', usage))

        result = model.prompt("q", schema={"type": "object"})

        assert result == {"answer": 42}  # dict guarantee preserved
        assert model.last_usage.cost == 0.002

    def test_last_usage_none_when_response_has_no_usage(self):
        model = _model_returning(_mock_response("no cost here", None))
        model.prompt("hi")
        assert model.last_usage is None


class Test_Conversation_03_Accumulation:
    """Conversation tracks per-turn and cumulative usage/cost."""

    def test_totals_default_to_zero(self):
        conv = _model_returning().conversation()
        assert conv.last_usage is None
        assert conv.total_usage is None
        assert conv.total_cost == 0.0

    def test_accumulates_across_turns(self):
        model = _model_returning(
            _mock_response("a", _usage(prompt=10, completion=5, cost=0.001)),
            _mock_response("b", _usage(prompt=20, completion=10, cost=0.002)),
        )
        conv = model.conversation()

        conv.prompt("turn 1")
        assert conv.last_usage.cost == 0.001
        assert conv.total_cost == 0.001

        conv.prompt("turn 2")
        # last_usage reflects only the latest turn...
        assert conv.last_usage.cost == 0.002
        # ...while totals sum across both turns.
        assert conv.total_cost == 0.003
        assert conv.total_usage.total_tokens == 45
        assert conv.total_usage.prompt_tokens == 30
        assert conv.total_usage.completion_tokens == 15

    def test_total_cost_zero_when_costs_absent(self):
        model = _model_returning(
            _mock_response("a", _usage(cost=None)),
            _mock_response("b", _usage(cost=None)),
        )
        conv = model.conversation()
        conv.prompt("t1")
        conv.prompt("t2")
        # Tokens still accumulate even when providers omit cost.
        assert conv.total_usage.total_tokens == 30
        assert conv.total_cost == 0.0

    def test_schema_turn_returns_dict_and_accumulates(self):
        model = _model_returning(
            _mock_response('{"ok": true}', _usage(cost=0.001)),
        )
        conv = model.conversation()

        result = conv.prompt("give json", schema={"type": "object"})

        assert result == {"ok": True}  # dict guarantee holds inside conversations
        assert conv.last_usage.cost == 0.001
        assert conv.total_cost == 0.001


class Test_AccumulateUsage_04_Helper:
    """Direct unit tests of the usage accumulation helper."""

    def test_none_new_returns_total_unchanged(self):
        total = _usage()
        assert _accumulate_usage(total, None) is total

    def test_first_turn_copies_tokens_and_cost(self):
        result = _accumulate_usage(None, _usage(prompt=10, completion=5, cost=0.001))
        assert result.prompt_tokens == 10
        assert result.total_tokens == 15
        assert result.cost == 0.001

    def test_mixed_cost_treats_missing_as_zero(self):
        total = _usage(cost=0.001)
        result = _accumulate_usage(total, _usage(cost=None))
        assert result.cost == 0.001

    def test_both_costs_none_stays_none(self):
        result = _accumulate_usage(_usage(cost=None), _usage(cost=None))
        assert result.cost is None

    def test_inputs_are_not_mutated(self):
        total = _usage(prompt=10, completion=5, cost=0.001)
        new = _usage(prompt=20, completion=10, cost=0.002)
        _accumulate_usage(total, new)
        # Neither operand is mutated; a fresh Usage is returned.
        assert (total.prompt_tokens, total.total_tokens, total.cost) == (10, 15, 0.001)
        assert (new.prompt_tokens, new.total_tokens, new.cost) == (20, 30, 0.002)

    def test_aggregate_drops_per_turn_detail_fields(self):
        turn = Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost=0.001,
            cost_details=CostDetails(upstream_inference_cost=0.0002),
        )
        result = _accumulate_usage(None, turn)
        # Running total carries tokens + cost only; details stay on last_usage.
        assert result.cost == 0.001
        assert result.cost_details is None
