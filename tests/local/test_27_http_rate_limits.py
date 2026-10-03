"""Exercise real SmartSurge history keys; only network I/O and sleep are mocked."""

from unittest.mock import patch

import pytest
import requests

from openrouter_client.http import HTTPManager
from openrouter_client.types import RequestMethod


@pytest.mark.parametrize(
    "base_url", ["https://openrouter.ai/api/v1", "https://proxy.example/v2/"]
)
@pytest.mark.parametrize(
    "registration", ["global", "relative", "leading_slash", "absolute"]
)
def test_rate_limit_throttles_actual_embedding_requests(base_url, registration):
    manager = HTTPManager(base_url=base_url)
    full_url = base_url.rstrip("/") + "/embeddings"
    response = requests.Response()
    response.status_code = 200
    response._content = b"{}"
    try:
        if registration == "global":
            manager.set_global_rate_limit(max_requests=1, time_period=60)
        else:
            endpoint = {
                "relative": "embeddings",
                "leading_slash": "/embeddings",
                "absolute": full_url,
            }[registration]
            manager.set_rate_limit(endpoint, RequestMethod.POST, 1, 60)

        with (
            patch.object(
                manager.client.session, "request", return_value=response
            ) as network,
            patch("smartsurge.models.time.sleep") as sleep,
        ):
            manager.post("embeddings", json={"input": "first"})
            sleep.assert_not_called()
            manager.post("/embeddings", json={"input": "second"})
            sleep.assert_called_once()
            # SmartSurge adds a small buffer beyond the rate-limit window.
            assert 0 < sleep.call_args.args[0] <= 61
            assert network.call_count == 2
            assert all(
                call.kwargs["url"] == full_url for call in network.call_args_list
            )

        limit = manager.client.get_rate_limit(full_url, "POST")
        assert limit is not None and limit.max_requests == 1
        assert manager.client.get_rate_limit("/embeddings", "POST") is None
    finally:
        manager.close()


@pytest.mark.parametrize(
    "endpoint, method",
    [
        ("chat/completions", RequestMethod.POST),
        ("completions", RequestMethod.POST),
        ("models", RequestMethod.GET),
        ("credits", RequestMethod.GET),
        ("generation", RequestMethod.GET),
        ("auth/key", RequestMethod.GET),
        ("auth/keys", RequestMethod.POST),
        ("keys", RequestMethod.POST),
    ],
)
def test_global_limits_cover_other_endpoint_request_keys(endpoint, method):
    manager = HTTPManager(base_url="https://openrouter.ai/api/v1")
    response = requests.Response()
    response.status_code = 200
    response._content = b"{}"
    try:
        manager.set_global_rate_limit(max_requests=10, time_period=60)
        with patch.object(manager.client.session, "request", return_value=response):
            manager.request(method, endpoint)
        # The request must reuse the registered history rather than create another.
        assert len(manager.client.histories) == 9
        full_url = f"https://openrouter.ai/api/v1/{endpoint}"
        assert manager.client.get_rate_limit(full_url, method.value).max_requests == 10
    finally:
        manager.close()
