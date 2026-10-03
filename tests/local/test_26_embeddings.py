"""Embeddings wire contract and shared transport behavior, without API keys."""

import json
from unittest.mock import Mock, patch

import pytest
import requests
from smartsurge.client import SmartSurgeClient
from smartsurge.exceptions import RateLimitExceeded as SurgeRateLimitExceeded

from openrouter_client import (
    Embedding,
    EmbeddingCostDetails,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingUsage,
    OpenRouterClient,
    RetryConfig,
)
from openrouter_client.auth import AuthManager
from openrouter_client.endpoints import EmbeddingsEndpoint
from openrouter_client.endpoints.chat import ChatEndpoint
from openrouter_client.exceptions import (
    APIError,
    AuthenticationError,
    RateLimitExceeded,
)
from openrouter_client.http import HTTPManager
from openrouter_client.models import ProviderPreferences


def response(body, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(body).encode()
    result.headers.update(headers or {})
    return result


def payload():
    # Deliberately out of input order: consumers must use the index.
    return {
        "object": "list",
        "model": "provider/actual-model",
        "data": [
            {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
            {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
        ],
        "usage": {"prompt_tokens": 8, "total_tokens": 8, "cost": 0.000001},
    }


def endpoint(body=None, retry_config=None):
    transport = Mock(spec=SmartSurgeClient)
    transport.request.return_value = response(payload() if body is None else body)
    manager = HTTPManager(
        base_url="https://openrouter.ai/api/v1",
        client=transport,
        retry_config=retry_config,
    )
    return EmbeddingsEndpoint(AuthManager(api_key="test-key"), manager), transport


@pytest.mark.parametrize("text", ["one article", ["one article", "another article"]])
def test_text_and_batch_wire_format(text):
    embeddings, transport = endpoint()
    result = embeddings.create(model="openai/text-embedding-3-small", input=text)
    sent = transport.request.call_args.kwargs
    assert sent["method"] == "POST"
    assert sent["endpoint"] == "https://openrouter.ai/api/v1/embeddings"
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    assert sent["json"] == {"model": "openai/text-embedding-3-small", "input": text}
    assert isinstance(result, EmbeddingsResponse)
    assert isinstance(result.data[0], Embedding)
    assert isinstance(result.usage, EmbeddingUsage)
    assert result.model == "provider/actual-model"
    assert result.usage.prompt_tokens == result.usage.total_tokens == 8
    assert result.usage.cost == 0.000001


def test_batch_mapping_uses_index_without_reordering_response():
    embeddings, _ = endpoint()
    inputs = ["first", "second"]
    result = embeddings.create(model="m", input=inputs)
    assert [item.index for item in result.data] == [1, 0]
    by_input = {inputs[item.index]: item.embedding for item in result.data}
    assert by_input == {"first": [0.1, 0.2], "second": [0.3, 0.4]}


@pytest.mark.parametrize(
    "provider",
    [
        {"only": ["OpenAI"], "allow_fallbacks": False, "future_option": True},
        ProviderPreferences(only=["OpenAI"], allow_fallbacks=False),
    ],
)
def test_optional_parameters_and_provider_model(provider):
    embeddings, transport = endpoint()
    embeddings.create(
        model="m",
        input="article",
        dimensions=256,
        encoding_format="float",
        input_type="search_document",
        provider=provider,
        user="reader-1",
        session_id="ingestion-1",
        trace={"trace_id": "batch-1"},
        future_parameter={"enabled": True},
    )
    expected_provider = (
        provider.model_dump(exclude_none=True)
        if isinstance(provider, ProviderPreferences)
        else provider
    )
    assert transport.request.call_args.kwargs["json"] == {
        "model": "m",
        "input": "article",
        "dimensions": 256,
        "encoding_format": "float",
        "input_type": "search_document",
        "provider": expected_provider,
        "user": "reader-1",
        "session_id": "ingestion-1",
        "trace": {"trace_id": "batch-1"},
        "future_parameter": {"enabled": True},
    }


def test_base64_response_is_preserved():
    body = payload()
    body["data"] = [{"index": 0, "embedding": "zczMPc3MTD4="}]
    embeddings, transport = endpoint(body)
    result = embeddings.create(model="m", input="text", encoding_format="base64")
    assert result.data[0].embedding == "zczMPc3MTD4="
    assert transport.request.call_args.kwargs["json"]["encoding_format"] == "base64"


def test_byok_accounting_preserves_actual_live_response_fields():
    body = payload()
    body["usage"] = {
        "prompt_tokens": 5,
        "total_tokens": 5,
        "cost": 0.0,
        "is_byok": True,
        "cost_details": {
            "upstream_inference_cost": 1e-7,
            "upstream_inference_prompt_cost": 1e-7,
            "upstream_inference_completions_cost": 0,
        },
    }
    embeddings, _ = endpoint(body)
    usage = embeddings.create(model="m", input="text").usage
    assert usage.cost == 0.0
    assert usage.is_byok is True
    assert isinstance(usage.cost_details, EmbeddingCostDetails)
    assert usage.cost_details.upstream_inference_cost == 1e-7
    assert usage.cost_details.upstream_inference_prompt_cost == 1e-7
    assert usage.cost_details.upstream_inference_completions_cost == 0


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 8}, {"cost": 0.0}])
def test_missing_accounting_is_not_fabricated(usage):
    body = payload()
    if usage is None:
        del body["usage"]
    else:
        body["usage"] = usage
    embeddings, _ = endpoint(body)
    result = embeddings.create(model="m", input="text")
    if usage is None:
        assert result.usage is None
    else:
        assert result.usage.cost == usage.get("cost")
        assert result.usage.prompt_tokens == usage.get("prompt_tokens")
        assert result.usage.total_tokens is None


@pytest.mark.parametrize(
    "params",
    [
        {"input": ""},
        {"input": []},
        {"input": ["ok", ""]},
        {"input": [None]},
        {"model": ""},
        {"dimensions": 0},
        {"encoding_format": "invalid"},
    ],
)
def test_invalid_request_never_reaches_transport(params):
    embeddings, transport = endpoint()
    with pytest.raises(ValueError):
        embeddings.create(**{"model": "m", "input": "text", **params})
    transport.request.assert_not_called()


@pytest.mark.parametrize("body", [None, [], {}, {"data": [{}], "model": "m"}])
def test_malformed_response_raises_api_error(body):
    embeddings, transport = endpoint()
    transport.request.return_value = response(body)
    with pytest.raises(APIError, match="Invalid embeddings response"):
        embeddings.create(model="m", input="text")


def test_non_json_response_raises_api_error():
    embeddings, transport = endpoint()
    transport.request.return_value._content = b"not json"
    with pytest.raises(APIError, match="Invalid JSON"):
        embeddings.create(model="m", input="text")


@pytest.mark.parametrize(
    "error, expected",
    [
        ({"message": "limited", "code": 429}, RateLimitExceeded),
        ({"message": "bad key", "code": 401}, AuthenticationError),
        ({"message": "Rate limit exceeded"}, RateLimitExceeded),
        ({"message": "provider failed", "code": 500}, APIError),
        ("provider failed", APIError),
    ],
)
def test_embedded_errors_are_normalized(error, expected):
    embeddings, _ = endpoint({"error": error})
    with pytest.raises(expected):
        embeddings.create(model="m", input="text")


@pytest.mark.parametrize("code", [400, 402, 403, 404, 500, 502, 503, "503", "429"])
def test_embedded_error_preserves_valid_status_code(code):
    body = {"error": {"message": "Provider failed", "code": code}}
    embeddings, _ = endpoint(body)
    with pytest.raises(APIError) as caught:
        embeddings.create(model="m", input="text")
    assert caught.value.status_code == int(code)
    assert caught.value.response.status_code == 200
    assert caught.value.details["error"] == body["error"]


@pytest.mark.parametrize("code", [None, "unavailable", {}, [], True, 503.5, 200, 600])
def test_embedded_error_with_invalid_code_keeps_transport_status(code):
    embeddings, _ = endpoint({"error": {"message": "Provider failed", "code": code}})
    with pytest.raises(APIError) as caught:
        embeddings.create(model="m", input="text")
    assert caught.value.status_code == 200


@pytest.mark.parametrize("kind", ["embeddings", "chat"])
@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_http_errors_match_chat(kind, status):
    embeddings, transport = endpoint()
    transport.request.return_value = response(
        {"error": {"message": "failed"}}, status, {"Retry-After": "2"}
    )
    target = (
        embeddings
        if kind == "embeddings"
        else ChatEndpoint(embeddings.auth_manager, embeddings.http_manager)
    )
    params = {"input": "text"} if kind == "embeddings" else {"messages": []}
    expected = RateLimitExceeded if status == 429 else APIError
    with pytest.raises(expected) as caught:
        target.create(model="m", **params)
    assert caught.value.status_code == status
    assert transport.request.call_count == 1


@pytest.mark.parametrize(
    "enabled, succeeds", [(False, False), (True, True), (True, False)]
)
@pytest.mark.parametrize("returned_response", [False, True])
def test_shared_429_retry_policy(enabled, succeeds, returned_response):
    config = RetryConfig(enabled=enabled, max_retries=1, jitter=0)
    embeddings, transport = endpoint(retry_config=config)
    limited = (
        response({}, 429, {"Retry-After": "2"})
        if returned_response
        else SurgeRateLimitExceeded(
            message="limited", endpoint="embeddings", method="POST", retry_after=2
        )
    )
    transport.request.side_effect = [
        limited,
        response(payload()) if succeeds else limited,
    ]
    with patch("openrouter_client.http.time.sleep") as sleep:
        if succeeds:
            assert embeddings.create(model="m", input=["a", "b"]).usage.cost > 0
        else:
            with pytest.raises(RateLimitExceeded) as caught:
                embeddings.create(model="m", input=["a", "b"])
            assert int(caught.value.retry_after) == 2
        assert transport.request.call_count == (2 if enabled else 1)
        if enabled:
            sleep.assert_called_once_with(2)
            assert (
                transport.request.call_args_list[0]
                == transport.request.call_args_list[1]
            )
        else:
            sleep.assert_not_called()


def test_client_wiring_and_cleanup():
    # Mock every endpoint during client construction, as required by client fixtures.
    names = [
        "ChatEndpoint",
        "CompletionsEndpoint",
        "ModelsEndpoint",
        "ImagesEndpoint",
        "EmbeddingsEndpoint",
        "GenerationsEndpoint",
        "CreditsEndpoint",
        "KeysEndpoint",
    ]
    mocks = {name: Mock() for name in names}
    mocks["KeysEndpoint"].return_value.get_current.return_value = {
        "data": {"rate_limit": {"requests": 10, "interval": "10s"}}
    }
    with (
        patch.multiple("openrouter_client.client", **mocks),
        patch("openrouter_client.client.HTTPManager", spec=HTTPManager) as http,
    ):
        with OpenRouterClient(api_key="test-key") as client:
            mocks["EmbeddingsEndpoint"].assert_called_once_with(
                auth_manager=client.auth_manager, http_manager=client.http_manager
            )
            http.return_value.set_global_rate_limit.assert_called_once_with(
                max_requests=10, time_period=10.0, cooldown=None
            )
        assert client.embeddings is None
        http.return_value.close.assert_called_once()


def test_request_model_public_export():
    request = EmbeddingsRequest(model="m", input=["a", "b"])
    assert request.model_dump(exclude_none=True) == {"model": "m", "input": ["a", "b"]}
