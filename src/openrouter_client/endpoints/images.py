"""Handler for OpenRouter's dedicated Image API (`/api/v1/images`).

A SEPARATE endpoint group from chat completions, with its own request shape and
its own capability-discovery routes:

    POST /api/v1/images                          generate
    GET  /api/v1/images/models                   discovery + capability descriptors
    GET  /api/v1/images/models/{id}/endpoints    per-provider capabilities and pricing

WHY `generate` TAKES ONE OBJECT
-------------------------------
`ImageGenerationRequest` carries about ten fields. Passed as one object it is also
the natural unit to log, diff and replay, which is what comparing models across a
matrix of prompts actually needs.

PRE-FLIGHT VALIDATION IS ADVISORY, NOT AUTHORITATIVE
----------------------------------------------------
A rejected request still costs a round trip, and a parameter the model does not
take is a self-inflicted, fixable mistake worth catching locally. But the catalog
is demonstrably incomplete -- `bytedance-seed/seedream-4.5` advertises a `1K` tier
the API then rejects -- so validation only refuses a request that CONTRADICTS a
published enum, never one the catalog merely fails to describe.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from pydantic import PrivateAttr

from ..auth import AuthManager
from ..exceptions import (APIError, ImageGenerationError, ImageGenerationFatal,
                          ImageRefused, ValidationError)
from ..http import HTTPManager
from ..models.images import (ImageGenerationRequest, ImageGenerationResponse, ImageModel,
                             ImageModelEndpoint, ImageModelEndpointsResponse,
                             ImageModelsResponse)
from .base import BaseEndpoint

logger = logging.getLogger(__name__)

#: Statuses where every remaining request would fail for the same reason.
FATAL_STATUSES = {401: "API key rejected", 402: "no credits remaining",
                  403: "forbidden", 404: "unknown or unusable model"}

#: Substrings identifying a provider refusal rather than a malfunction. Matched on
#: the message because a moderation block arrives as HTTP 400, not as a 200.
REFUSAL_MARKERS = ("moderation", "content polic", "safety", "blocked this request")

#: Read timeout for a generation, in seconds.
#: MEASURED 2026-09-18: generations took 10-91s across the models tried, against an
#: HTTPManager default of 60. A default that times out on a request the provider is
#: still working on would bill the caller for an image they never receive, so this
#: endpoint carries its own generous ceiling instead of inheriting the shared one.
GENERATION_TIMEOUT: tuple = (10.0, 300.0)

#: Parameters worth checking locally against a model's published enums.
VALIDATED_PARAMETERS = ("resolution", "aspect_ratio", "quality", "output_format")


class ImagesEndpoint(BaseEndpoint):
    """Handler for the image generation API endpoint."""

    #: Declared rather than merely assigned: `BaseEndpoint` is a Pydantic model, and an
    #: undeclared attribute only persists by way of the underscore bypass. A PrivateAttr
    #: states the intent and does not depend on that behaviour holding.
    _model_memo: Dict[str, Optional[ImageModel]] = PrivateAttr(default_factory=dict)

    def __init__(self, auth_manager: AuthManager, http_manager: HTTPManager):
        """Initialize the images endpoint handler.

        Args:
            auth_manager (AuthManager): Authentication manager.
            http_manager (HTTPManager): HTTP communication manager.
        """
        super().__init__(auth_manager, http_manager, "images")
        self.logger.debug("Initialized images endpoint handler")

    def refresh_models(self) -> None:
        """Drop the model records memoized by `validate`.

        `validate` runs once per generation and a matrix run is hundreds of generations
        against a handful of models, so refetching a 52-model catalogue each time would
        cost more round trips than the generating. Discovery through `list_models` is
        never cached and always reflects the live catalogue; only the validation path
        memoizes, and this clears it for a long-lived client.
        """
        self._model_memo = {}

    def list_models(self) -> List[ImageModel]:
        """List every image-capable model with its capability descriptors.

        Returns:
            List[ImageModel]: The available image models.
        """
        response = self.http_manager.get(self._get_endpoint_url("models"),
                                         headers=self._get_headers())
        return ImageModelsResponse.model_validate(response.json()).data

    def get_model(self, model_id: str) -> Optional[ImageModel]:
        """The discovery record for one model, or None when it is not listed.

        Args:
            model_id (str): Model slug, e.g. 'google/gemini-3-pro-image'.

        Returns:
            Optional[ImageModel]: The model's record, or None.
        """
        for model in self.list_models():
            if model.id == model_id:
                return model
        return None

    def model_endpoints(self, model_id: str) -> List[ImageModelEndpoint]:
        """Per-provider capabilities and pricing for one model.

        Definitive where `list_models` is only indicative, and the only place the
        billing unit appears -- which decides whether resolution affects price.

        Args:
            model_id (str): Model slug, e.g. 'bytedance-seed/seedream-4.5'.

        Returns:
            List[ImageModelEndpoint]: One record per serving provider.
        """
        url = self._get_endpoint_url("models/%s/endpoints" % model_id.strip("/"))
        response = self.http_manager.get(url, headers=self._get_headers())
        return ImageModelEndpointsResponse.model_validate(response.json()).endpoints

    def _memoized_model(self, model_id: str) -> Optional[ImageModel]:
        """`get_model`, fetched at most once per model for the lifetime of this handler."""
        if model_id not in self._model_memo:
            self._model_memo[model_id] = self.get_model(model_id)
        return self._model_memo[model_id]

    def validate(self, request: ImageGenerationRequest,
                 model: Optional[ImageModel] = None) -> None:
        """Check a request against a model's published capabilities.

        Only raises when a value CONTRADICTS a published enum. A parameter the
        catalog does not describe is left alone: the listing is incomplete, and
        refusing an unlisted-but-valid request would be worse than a round trip.

        Args:
            request (ImageGenerationRequest): The request to check.
            model (Optional[ImageModel]): The model's record. Fetched when omitted.

        Raises:
            ValidationError: If a parameter contradicts the model's published enum.
        """
        model = model or self._memoized_model(request.model)
        if model is None:
            return
        for parameter in VALIDATED_PARAMETERS:
            value = getattr(request, parameter, None)
            if value is None or model.allows(parameter, value) is not False:
                continue
            raise ValidationError(
                "%s does not support %s=%r; published values are %s"
                % (model.id, parameter, value, model.allowed_values(parameter)))
        if request.input_references and not model.accepts_references:
            raise ValidationError("%s does not accept input_references" % model.id)

    def generate(self, request: ImageGenerationRequest,
                 validate_request: bool = False,
                 timeout: Optional[tuple] = None) -> ImageGenerationResponse:
        """Generate one or more images.

        Args:
            request (ImageGenerationRequest): The generation request.
            validate_request (bool): Check parameters against the model's published
                capabilities first. Costs one discovery call. Defaults to False.
            timeout (Optional[tuple]): `(connect, read)` override. Defaults to
                GENERATION_TIMEOUT, which is far longer than the shared default
                because generation is slow (see that constant).

        Returns:
            ImageGenerationResponse: The generated image(s), with `usage.cost`.
                An answer that carried no image and stated no reason returns
                normally with `first_image is None` -- the one case that genuinely
                cannot be told apart from a provider hiccup.

        Raises:
            ImageRefused: The provider answered and declined to produce an image.
            ImageGenerationFatal: A failure that would repeat on every request.
            ImageGenerationError: A named, per-request failure.
            ValidationError: Only when `validate_request` is set and a parameter
                contradicts the model's published capabilities.
        """
        # Inside the try: `validate` performs a discovery GET of its own, and a rejected
        # key surfacing there as a raw APIError would skip the taxonomy entirely -- the
        # caller would never receive the "stop the run" signal the taxonomy exists for.
        try:
            if validate_request:
                self.validate(request)
            response = self.http_manager.post(self._get_endpoint_url(),
                                              headers=self._get_headers(),
                                              json=request.to_payload(),
                                              timeout=timeout or GENERATION_TIMEOUT)
        except APIError as problem:
            raise self._classify(problem) from problem

        # An unparseable body must still arrive as an image failure: a caller driving a
        # batch catches this taxonomy, and a bare ValueError escaping it would abort a
        # run that a single malformed answer should only have cost one row of.
        try:
            body = response.json()
        except ValueError as problem:
            raise ImageGenerationError(
                "could not parse the image response as JSON: %s" % problem,
                status_code=getattr(response, "status_code", None)) from problem

        embedded = self._error_message(body)
        result = ImageGenerationResponse.model_validate(body)
        if embedded:
            if result.first_image is None:
                # A 200 that names its own reason: a refusal, not a malfunction.
                raise ImageRefused(embedded, status_code=response.status_code)
            # A partial result (n>1 where some were refused) is NOT raised -- the images
            # that were produced are worth returning -- but the reason must not be lost.
            self.logger.warning("image response carried %d image(s) and an error: %s",
                                len(result.images), embedded)
        return result

    @staticmethod
    def _error_message(body: object) -> str:
        """The provider's stated reason, or '' when it gave none."""
        if not isinstance(body, dict):
            return ""
        error = body.get("error")
        if not error:
            return ""
        if isinstance(error, dict):
            # Fall back to the whole object: an error carrying only a code and metadata
            # is still the provider stating a reason, and returning "" here would file it
            # as the unexplained-empty case it demonstrably is not.
            return str(error.get("message") or error)
        return str(error)

    @classmethod
    def _classify(cls, problem: APIError) -> ImageGenerationError:
        """Map a transport-level APIError onto the image failure taxonomy.

        ORDER MATTERS. The fatal statuses are checked FIRST: `APIError.__str__` appends
        details drawn from the response body, so a dead key whose body happens to mention
        a safety or moderation field would otherwise be read as "refused -- the next
        request may well succeed", and a batch would keep hammering it. A moderation
        block arrives as HTTP 400, which is not a fatal status, so checking fatal first
        costs nothing on the case that matters.

        Markers are matched against `problem.message` rather than `str(problem)` for the
        same reason: the appended details are the provider's echo of the request, not its
        verdict on it.
        """
        status = getattr(problem, "status_code", None)
        message = getattr(problem, "message", None) or str(problem)
        # A rate limit's backoff hint is the one piece of a failure a caller can act on,
        # so it is carried onto the replacement rather than discarded with the type.
        extra = {}
        retry_after = getattr(problem, "retry_after", None)
        if retry_after is not None:
            extra["retry_after"] = retry_after

        if status in FATAL_STATUSES:
            return ImageGenerationFatal("%s: %s" % (FATAL_STATUSES[status], message),
                                        status_code=status, **extra)
        lowered = message.lower()
        for marker in REFUSAL_MARKERS:
            if marker in lowered:
                return ImageRefused(message, status_code=status, **extra)
        failure = ImageGenerationError(message, status_code=status, **extra)
        failure.retry_after = retry_after
        return failure
