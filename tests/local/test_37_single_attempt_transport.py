"""Exercise real client composition with only network I/O mocked."""

import json
from unittest.mock import Mock, patch

import pytest
import requests

from openrouter_client import OpenRouterClient
from openrouter_client.exceptions import APIError, RateLimitExceeded
from openrouter_client.http import HTTPManager, RetryConfig
from openrouter_client.transport import SingleAttemptClient


def response_for(status: int) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.headers["Retry-After"] = "3600"
    response.headers["Location"] = "https://example.invalid/redirect"
    response._content = json.dumps({"data": []}).encode()
    return response


@pytest.mark.parametrize("alias", ["retries", "max_retries"])
def test_no_network_on_construction_and_one_endpoint_attempt(alias: str) -> None:
    with patch(
        "requests.sessions.Session.send", return_value=response_for(200)
    ) as send:
        with OpenRouterClient(
            api_key="test-key",
            initialize_rate_limit=False,
            timeout=(2.0, 9.0),
            **{alias: 0},
        ) as client:
            send.assert_not_called()
            client.models.list()
            send.assert_called_once()
            assert send.call_args.kwargs["timeout"] == (2.0, 9.0)
            assert send.call_args.kwargs["allow_redirects"] is False
            assert (
                send.call_args.args[0].url
                == "https://openrouter.ai/api/v1/models?details=false"
            )
            for adapter in client.http_manager.client.session.adapters.values():
                assert adapter.max_retries.total == 0


@pytest.mark.parametrize("status", [301, 302, 307, 308, 429, 500, 502, 503, 504])
@pytest.mark.parametrize("method", ["get", "post"])
def test_errors_never_retry_or_sleep(status: int, method: str) -> None:
    response = response_for(status)
    manager = HTTPManager(base_url="https://example.invalid", retries=0)
    error = RateLimitExceeded if status == 429 else APIError
    with patch("requests.sessions.Session.send", return_value=response) as send:
        with patch("time.sleep") as sleep:
            with pytest.raises(error) as raised:
                getattr(manager, method)("/chat/completions", json={"model": "test"})
            assert raised.value.response is response
            if status == 429:
                assert str(raised.value.retry_after) == "3600"
            else:
                assert raised.value.status_code == status
            send.assert_called_once()
            sleep.assert_not_called()
    manager.close()


@pytest.mark.parametrize(
    "failure", [requests.ConnectTimeout, requests.ReadTimeout, requests.ConnectionError]
)
def test_network_errors_keep_cause_without_retry(failure: type) -> None:
    manager = HTTPManager(base_url="https://example.invalid", retries=0, timeout=7.0)
    original = failure("test failure")
    with patch("requests.sessions.Session.send", side_effect=original) as send:
        with patch("time.sleep") as sleep:
            with pytest.raises(APIError) as raised:
                manager.post("/chat/completions", json={})
            assert raised.value.__cause__ is original
            send.assert_called_once()
            assert send.call_args.kwargs["timeout"] == 7.0
            sleep.assert_not_called()
    manager.close()


def test_timeout_inheritance_override_and_positive_retry_mapping() -> None:
    with patch("openrouter_client.http.SmartSurgeClient") as surge:
        surge.return_value.request.return_value = response_for(200)
        manager = HTTPManager(base_url="https://example.invalid", retries=2, timeout=90)
        assert surge.call_args.kwargs["max_retries"] == 2
        assert "retries" not in surge.call_args.kwargs
        manager.get("/models")
        assert surge.return_value.request.call_args.kwargs["timeout"] == 90
        manager.get("/models", timeout=(1, 4))
        assert surge.return_value.request.call_args.kwargs["timeout"] == (1, 4)


def test_default_startup_still_fetches_key() -> None:
    with patch(
        "requests.sessions.Session.send", return_value=response_for(200)
    ) as send:
        with OpenRouterClient(api_key="test-key", retries=0):
            send.assert_called_once()
            assert send.call_args.args[0].url.endswith("/key")


@pytest.mark.parametrize(
    "timeout", [None, 0, -1, float("inf"), float("nan"), (1, None), (0, 2), (1,), True]
)
def test_unbounded_or_invalid_timeouts_rejected(timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout"):
        HTTPManager(base_url="https://example.invalid", retries=0, timeout=timeout)


def test_invalid_override_rejected_before_network() -> None:
    manager = HTTPManager(base_url="https://example.invalid", retries=0)
    with patch("requests.sessions.Session.send") as send:
        with pytest.raises(ValueError, match="timeout"):
            manager.get("/models", timeout=float("inf"))
        send.assert_not_called()
    manager.close()


@pytest.mark.parametrize(
    "options",
    [
        {"retry_config": RetryConfig(enabled=True)},
        {"client": Mock()},
        {"rate_limit": 10},
        {"max_retries": 2},
    ],
)
def test_contradictory_options_rejected(options: dict) -> None:
    with pytest.raises(ValueError):
        HTTPManager(base_url="https://example.invalid", retries=0, **options)


def test_high_level_conflicting_aliases_rejected() -> None:
    with pytest.raises(ValueError, match="must agree"):
        OpenRouterClient(
            api_key="test-key", retries=0, max_retries=3, initialize_rate_limit=False
        )


def test_streaming_transport_defaults_and_close() -> None:
    transport = SingleAttemptClient("https://example.invalid/api", (2, 8))
    with patch.object(
        transport.session, "send", return_value=response_for(200)
    ) as send:
        transport.post("chat/completions", stream=True, json={})
        send.assert_called_once()
        assert (
            send.call_args.args[0].url == "https://example.invalid/api/chat/completions"
        )
        assert send.call_args.kwargs["stream"] is True
        assert send.call_args.kwargs["timeout"] == (2, 8)
        assert send.call_args.kwargs["allow_redirects"] is False
    with patch.object(transport.session, "close") as close:
        transport.close()
        close.assert_called_once()


@pytest.mark.parametrize("stream", [False, True])
def test_chat_endpoint_uses_one_attempt_with_configured_timeout(stream: bool) -> None:
    response = response_for(200)
    payload = {
        "id": "test-chat",
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "delta" if stream else "message": {
                    "role": "assistant",
                    "content": "hello",
                },
            }
        ],
    }
    if stream:
        response._content = (
            b"data: " + json.dumps(payload).encode() + b"\n\ndata: [DONE]\n\n"
        )
    else:
        response._content = json.dumps(payload).encode()
    response._content_consumed = True
    with patch("requests.sessions.Session.send", return_value=response) as send:
        with OpenRouterClient(
            api_key="test-key", retries=0, timeout=13, initialize_rate_limit=False
        ) as client:
            result = client.chat.create(
                messages=[{"role": "user", "content": "hi"}],
                model="test-model",
                stream=stream,
            )
            if stream:
                chunks = list(result)
                assert chunks[0].choices[0].delta.content == "hello"
            else:
                assert result.choices[0].message.content == "hello"
            send.assert_called_once()
            assert send.call_args.kwargs["timeout"] == 13
            assert send.call_args.kwargs["allow_redirects"] is False
            request = send.call_args.args[0]
            assert request.method == "POST"
            assert request.url.endswith("/chat/completions")
            assert json.loads(request.body)["model"] == "test-model"
