"""Text embeddings through the client's shared transport and retry policy."""

from typing import Any, Dict, List, Literal, Optional, Union

from ..auth import AuthManager
from ..exceptions import APIError, AuthenticationError, RateLimitExceeded
from ..http import HTTPManager
from ..models.embeddings import EmbeddingsRequest, EmbeddingsResponse
from ..models.providers import ProviderPreferences
from .base import BaseEndpoint


class EmbeddingsEndpoint(BaseEndpoint):
    """Handler for POST /api/v1/embeddings."""

    def __init__(self, auth_manager: AuthManager, http_manager: HTTPManager) -> None:
        super().__init__(auth_manager, http_manager, "embeddings")

    def create(
        self,
        model: str,
        input: Union[str, List[str]],
        *,
        dimensions: Optional[int] = None,
        encoding_format: Optional[Literal["float", "base64"]] = None,
        input_type: Optional[str] = None,
        provider: Optional[Union[Dict[str, Any], ProviderPreferences]] = None,
        user: Optional[str] = None,
        session_id: Optional[str] = None,
        trace: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> EmbeddingsResponse:
        """Embed a text or batch of texts.

        Optional parameters follow OpenRouter's embeddings API. Additional keyword
        arguments are passed through to the request body. Provider preferences may
        be a dictionary or a ProviderPreferences model. With encoding_format="base64",
        each embedding is returned as a string; otherwise it is a list of floats.

        Results retain server order: use each item's index to find its input.
        Missing usage or cost remains None, rather than implying a free request.
        HTTP errors and retries use the same HTTPManager as chat completions.

        Raises:
            ValueError: Invalid request parameters.
            APIError: Transport failure, API error, or malformed response.
            RateLimitExceeded: Rate limit reached after any configured retries.
        """
        request = EmbeddingsRequest(
            model=model,
            input=input,
            dimensions=dimensions,
            encoding_format=encoding_format,
            input_type=input_type,
            provider=provider,
            user=user,
            session_id=session_id,
            trace=trace,
            **kwargs,
        )
        response = self.http_manager.post(
            endpoint=self._get_endpoint_url(),
            headers=self._get_headers(),
            json=request.model_dump(mode="json", exclude_none=True),
        )
        try:
            body = response.json()
        except ValueError as exc:
            raise APIError(
                "Invalid JSON in embeddings response", status_code=response.status_code
            ) from exc

        # Like chat, handle providers returning an error in a successful HTTP body.
        if isinstance(body, dict) and "error" in body:
            error = body["error"]
            error = error if isinstance(error, dict) else {"message": str(error)}
            message = str(error.get("message") or "Unknown API error")
            error_type = error.get("type")
            code = error.get("code")
            if (
                str(code) == "429"
                or error_type == "rate_limit_exceeded"
                or "rate limit" in message.lower()
            ):
                raise RateLimitExceeded(message, response=response)
            if (
                str(code) == "401"
                or error_type == "authentication_error"
                or "authentication" in message.lower()
            ):
                raise AuthenticationError(message)
            raise APIError(message, status_code=response.status_code, response=response)

        try:
            return EmbeddingsResponse.model_validate(body)
        except ValueError as exc:
            raise APIError(
                "Invalid embeddings response", status_code=response.status_code
            ) from exc
