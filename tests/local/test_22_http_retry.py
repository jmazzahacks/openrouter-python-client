"""
Unit tests for HTTPManager's opt-in 429 retry-with-backoff (RetryConfig).

These cover:
- the disabled-by-default behavior (a 429 propagates immediately, unchanged), and
- the enabled behavior (retry with backoff, honor Retry-After, surface attempts/elapsed
  on exhaustion), with time.sleep patched so tests don't actually wait.
"""

import pytest
from unittest.mock import Mock, patch
import requests

from openrouter_client import OpenRouterClient
from openrouter_client.exceptions import RateLimitExceeded
from openrouter_client.types import RequestMethod
from openrouter_client.http import HTTPManager, RetryConfig
from smartsurge.client import SmartSurgeClient
from smartsurge.exceptions import RateLimitExceeded as SmartSurgeRateLimitExceeded


def _ok_response() -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    return response


def _rate_limited_response(retry_after=None) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = 429
    response.headers = {}
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return response


def _smartsurge_rle(retry_after=None) -> SmartSurgeRateLimitExceeded:
    return SmartSurgeRateLimitExceeded(
        message="Rate limit exceeded for x POST",
        endpoint="x",
        method="POST",
        retry_after=retry_after,
    )


def _make_manager(client, retry_config=None) -> HTTPManager:
    return HTTPManager(
        base_url="https://api.example.com",
        client=client,
        retry_config=retry_config,
    )


class Test_RetryConfig_01_Defaults:
    """RetryConfig defaults keep the feature opt-in."""

    def test_default_config_is_disabled(self):
        cfg = RetryConfig()
        assert cfg.enabled is False
        assert cfg.max_retries == 5
        assert cfg.respect_retry_after is True

    def test_manager_defaults_to_disabled_config(self):
        mgr = _make_manager(Mock(spec=SmartSurgeClient))
        assert isinstance(mgr.retry_config, RetryConfig)
        assert mgr.retry_config.enabled is False


class Test_HTTPManager_Retry_02_Disabled:
    """With retries disabled, behavior is unchanged from before the feature."""

    def test_smartsurge_rle_propagates_without_retry(self):
        mock_client = Mock(spec=SmartSurgeClient)
        mock_client.request.side_effect = _smartsurge_rle()
        mgr = _make_manager(mock_client)  # disabled by default

        with pytest.raises(SmartSurgeRateLimitExceeded):
            mgr.request(method=RequestMethod.POST, endpoint="x")

        assert mock_client.request.call_count == 1

    def test_returned_429_raises_our_error_without_retry(self):
        mock_client = Mock(spec=SmartSurgeClient)
        mock_client.request.return_value = _rate_limited_response(retry_after="60")
        mgr = _make_manager(mock_client)

        with pytest.raises(RateLimitExceeded):
            mgr.request(method=RequestMethod.GET, endpoint="x")

        assert mock_client.request.call_count == 1


class Test_HTTPManager_Retry_03_Enabled:
    """With retries enabled, 429s are retried with backoff."""

    def test_retries_then_succeeds_on_smartsurge_rle(self):
        mock_client = Mock(spec=SmartSurgeClient)
        success = _ok_response()
        mock_client.request.side_effect = [_smartsurge_rle(), success]
        cfg = RetryConfig(enabled=True, max_retries=3, base_delay=0.01, jitter=0)
        mgr = _make_manager(mock_client, cfg)

        with patch("openrouter_client.http.time.sleep") as mock_sleep:
            response = mgr.request(method=RequestMethod.POST, endpoint="x")

        assert response is success
        assert mock_client.request.call_count == 2
        assert mock_sleep.call_count == 1

    def test_retries_then_succeeds_on_returned_429(self):
        mock_client = Mock(spec=SmartSurgeClient)
        success = _ok_response()
        mock_client.request.side_effect = [_rate_limited_response(), success]
        cfg = RetryConfig(enabled=True, max_retries=3, base_delay=0.01, jitter=0)
        mgr = _make_manager(mock_client, cfg)

        with patch("openrouter_client.http.time.sleep"):
            response = mgr.request(method=RequestMethod.GET, endpoint="x")

        assert response is success
        assert mock_client.request.call_count == 2

    def test_exhausts_retries_and_surfaces_attempts_and_elapsed(self):
        mock_client = Mock(spec=SmartSurgeClient)
        mock_client.request.side_effect = _smartsurge_rle()
        cfg = RetryConfig(enabled=True, max_retries=2, base_delay=0.01, jitter=0)
        mgr = _make_manager(mock_client, cfg)

        with patch("openrouter_client.http.time.sleep"):
            with pytest.raises(RateLimitExceeded) as exc_info:
                mgr.request(method=RequestMethod.POST, endpoint="x")

        # total attempts == max_retries + 1
        assert mock_client.request.call_count == 3
        assert exc_info.value.details.get("attempts") == 3
        assert "elapsed_seconds" in exc_info.value.details

    def test_honors_retry_after_for_sleep_duration(self):
        mock_client = Mock(spec=SmartSurgeClient)
        mock_client.request.side_effect = [_smartsurge_rle(retry_after=5), _ok_response()]
        cfg = RetryConfig(
            enabled=True, max_retries=3, base_delay=1.0, factor=2.0, jitter=0, max_delay=30.0
        )
        mgr = _make_manager(mock_client, cfg)

        with patch("openrouter_client.http.time.sleep") as mock_sleep:
            mgr.request(method=RequestMethod.POST, endpoint="x")

        mock_sleep.assert_called_once_with(5.0)


class Test_HTTPManager_BackoffDelay_04_Schedule:
    """Direct unit tests of the backoff computation."""

    def test_exponential_schedule_without_jitter(self):
        cfg = RetryConfig(enabled=True, base_delay=1.0, factor=2.0, jitter=0, max_delay=30.0)
        mgr = _make_manager(Mock(spec=SmartSurgeClient), cfg)
        assert mgr._backoff_delay(None, 0) == 1.0
        assert mgr._backoff_delay(None, 1) == 2.0
        assert mgr._backoff_delay(None, 2) == 4.0

    def test_max_delay_caps_exponential_growth(self):
        cfg = RetryConfig(enabled=True, base_delay=10.0, factor=10.0, jitter=0, max_delay=30.0)
        mgr = _make_manager(Mock(spec=SmartSurgeClient), cfg)
        assert mgr._backoff_delay(None, 5) == 30.0

    def test_retry_after_capped_at_max_delay(self):
        cfg = RetryConfig(enabled=True, jitter=0, max_delay=30.0)
        mgr = _make_manager(Mock(spec=SmartSurgeClient), cfg)
        assert mgr._backoff_delay(120, 0) == 30.0

    def test_retry_after_ignored_when_respect_disabled(self):
        cfg = RetryConfig(
            enabled=True, base_delay=1.0, factor=2.0, jitter=0, respect_retry_after=False
        )
        mgr = _make_manager(Mock(spec=SmartSurgeClient), cfg)
        # Retry-After present but ignored -> exponential schedule used
        assert mgr._backoff_delay(5, 1) == 2.0

    def test_jitter_adds_bounded_random_component(self):
        cfg = RetryConfig(enabled=True, base_delay=1.0, factor=2.0, jitter=0.25, max_delay=30.0)
        mgr = _make_manager(Mock(spec=SmartSurgeClient), cfg)
        delay = mgr._backoff_delay(None, 0)
        assert 1.0 <= delay <= 1.25


class Test_OpenRouterClient_RetryConfig_05_Wiring:
    """The retry_config kwarg threads from OpenRouterClient to its HTTPManager.

    This covers the exact usage documented in the README's
    'Retrying 429s with Backoff (opt-in)' section.
    """

    def test_retry_config_reaches_http_manager(self):
        cfg = RetryConfig(
            enabled=True,
            max_retries=5,
            base_delay=1.0,
            factor=2.0,
            max_delay=30.0,
            jitter=0.25,
            respect_retry_after=True,
        )
        with patch("openrouter_client.client.OpenRouterClient._initialize_rate_limit"):
            client = OpenRouterClient(api_key="test-key", retry_config=cfg)
        assert client.http_manager.retry_config is cfg

    def test_client_defaults_to_disabled_retry_config(self):
        with patch("openrouter_client.client.OpenRouterClient._initialize_rate_limit"):
            client = OpenRouterClient(api_key="test-key")
        assert client.http_manager.retry_config.enabled is False
