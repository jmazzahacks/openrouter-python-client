"""Small billable smoke tests; skipped unless a regular API key is available."""

import os

import pytest

from openrouter_client import OpenRouterClient

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY is not set"
)


@pytest.mark.parametrize(
    "text", ["A short news article.", ["A news article.", "Another story."]]
)
def test_embeddings_text_and_batch(text):
    with OpenRouterClient() as client:
        result = client.embeddings.create(
            model="openai/text-embedding-3-small", input=text, encoding_format="float"
        )
    count = 1 if isinstance(text, str) else len(text)
    assert len(result.data) == count
    assert {item.index for item in result.data} == set(range(count))
    assert all(len(item.embedding) == 1536 for item in result.data)
    assert all(
        isinstance(value, float) for item in result.data for value in item.embedding
    )
    assert result.model
    assert result.usage is not None
    assert result.usage.prompt_tokens > 0
    assert result.usage.total_tokens >= result.usage.prompt_tokens
    # BYOK can return zero OpenRouter cost with separately billed provider charges.
    assert result.usage.cost is not None and result.usage.cost >= 0
    if result.usage.is_byok:
        assert result.usage.cost_details is not None
        assert result.usage.cost_details.upstream_inference_cost is not None
        assert result.usage.cost_details.upstream_inference_cost >= 0
