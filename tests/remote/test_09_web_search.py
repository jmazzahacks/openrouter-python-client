"""One billable Exa search smoke test; skipped without a regular API key."""

import os

import pytest

from openrouter_client import OpenRouterClient, UrlCitationAnnotation, WebSearchPlugin

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY is not set"
)


def test_exa_search_returns_citations_and_cost():
    with OpenRouterClient() as client:
        result = client.chat.create(
            model="openai/gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": "Search the web for recent NASA news. Briefly describe one story and cite its source.",
                }
            ],
            plugins=[WebSearchPlugin(engine="exa", mode="auto", max_results=5)],
            max_tokens=256,
        )
    citations = [
        item.url_citation
        for item in result.choices[0].message.annotations or []
        if isinstance(item, UrlCitationAnnotation)
    ]
    assert citations and any(item.url and item.title for item in citations)
    assert result.usage is not None
    # The documented Exa auto-mode fee is $0.007 even when model inference is BYOK.
    assert result.usage.cost is not None and result.usage.cost >= 0.007
