"""Single-attempt transport for callers that own scheduling and retries."""

import math
from typing import Any, Tuple, Union

import requests

Timeout = Union[float, Tuple[float, float]]


def validate_timeout(timeout: Timeout) -> Timeout:
    """Require finite, positive connect/read limits, never an unlimited wait."""
    values = timeout if isinstance(timeout, tuple) else (timeout,)
    if isinstance(timeout, tuple) and len(timeout) != 2:
        raise ValueError("timeout must be a number or a (connect, read) pair")
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("timeout values must be finite positive numbers")
    return timeout


class SingleAttemptClient:
    """A requests session with no retries, redirects, or rate-limit sleeps."""

    def __init__(self, base_url: str, timeout: Timeout) -> None:
        self.base_url = base_url
        self.timeout = validate_timeout(timeout)
        self.session = requests.Session()
        # Explicitly disable urllib3 retries on both schemes.
        for scheme in ("http://", "https://"):
            self.session.mount(scheme, requests.adapters.HTTPAdapter(max_retries=0))

    def request(self, method: str, endpoint: str, **kwargs: Any) -> requests.Response:
        kwargs["allow_redirects"] = False
        kwargs["timeout"] = validate_timeout(kwargs.get("timeout", self.timeout))
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        return self.session.request(method=method, url=endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs: Any) -> requests.Response:
        """Support the streaming layer, which calls the transport directly."""
        return self.request("POST", endpoint, **kwargs)

    def close(self) -> None:
        self.session.close()
