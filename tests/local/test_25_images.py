"""Local tests for the Image API endpoint.

Fixtures are trimmed copies of REAL payloads captured from the live API on
2026-09-18, not invented shapes -- including the two that surprised us: a
capability listing that advertises a tier the API then rejects, and a moderation
block arriving as HTTP 400 rather than as a 200 with an empty body.
"""
import pytest
from unittest.mock import Mock

from openrouter_client.auth import AuthManager
from openrouter_client.endpoints.images import ImagesEndpoint
from openrouter_client.http import HTTPManager
from openrouter_client.exceptions import (APIError, ImageGenerationError,
                                          ImageGenerationFatal, ImageRefused,
                                          ValidationError)
from openrouter_client.models.images import (ImageGenerationRequest,
                                             ImageGenerationResponse, ImageModel,
                                             ImageReference)

# One-pixel PNG, base64 -- enough to prove decoding without shipping an image.
ONE_PIXEL = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
             "IQAAAABJRU5ErkJggg==")

MODELS_PAYLOAD = {"data": [
    {"id": "bytedance-seed/seedream-4.5", "name": "Seedream 4.5",
     "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["image"]},
     "supported_parameters": {
         "resolution": {"type": "enum", "values": ["1K", "2K", "4K"]},
         "aspect_ratio": {"type": "enum", "values": ["1:1", "2:3", "16:9"]},
         "seed": {"type": "boolean"}},
     "supports_streaming": False},
    {"id": "google/gemini-3-pro-image", "name": "Nano Banana Pro",
     "architecture": {"input_modalities": ["image", "text"], "output_modalities": ["image", "text"]},
     "supported_parameters": {"resolution": {"type": "enum", "values": ["1K", "2K", "4K"]}},
     "supports_streaming": False},
    {"id": "text-only/model", "name": "Text Only",
     "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
     "supported_parameters": {}},
]}

ENDPOINTS_PAYLOAD = {"id": "bytedance-seed/seedream-4.5", "endpoints": [
    {"provider_name": "Seed", "provider_slug": "bytedance",
     "supported_parameters": {"resolution": {"type": "enum", "values": ["1K", "2K", "4K"]}},
     "allowed_passthrough_parameters": [], "supports_streaming": False,
     "pricing": [{"billable": "output_image", "unit": "image", "cost_usd": 0.04},
                 {"billable": "input_image", "unit": "image", "cost_usd": 0.0}]}]}


def make_endpoint(response_json=None, raises=None, status_code=200):
    """An ImagesEndpoint whose HTTP manager returns or raises what a test needs."""
    auth = Mock(spec=AuthManager)
    auth.get_auth_headers = Mock(return_value={"Authorization": "Bearer test"})
    http = Mock(spec=HTTPManager)
    answer = Mock()
    answer.json = Mock(return_value=response_json or {})
    answer.status_code = status_code
    if raises is not None:
        http.post = Mock(side_effect=raises)
    else:
        http.post = Mock(return_value=answer)
    http.get = Mock(return_value=answer)
    return ImagesEndpoint(auth_manager=auth, http_manager=http)


class TestDiscovery:
    def test_list_models_parses_capabilities(self):
        models = make_endpoint(MODELS_PAYLOAD).list_models()
        assert len(models) == 3
        assert models[0].id == "bytedance-seed/seedream-4.5"
        assert models[0].allowed_values("resolution") == ["1K", "2K", "4K"]

    def test_get_model_returns_none_when_absent(self):
        endpoint = make_endpoint(MODELS_PAYLOAD)
        assert endpoint.get_model("google/gemini-3-pro-image") is not None
        assert endpoint.get_model("nope/not-a-model") is None

    def test_accepts_references_reads_input_modalities(self):
        models = {m.id: m for m in make_endpoint(MODELS_PAYLOAD).list_models()}
        assert models["bytedance-seed/seedream-4.5"].accepts_references is True
        assert models["text-only/model"].accepts_references is False

    def test_model_endpoints_exposes_billing_unit(self):
        endpoints = make_endpoint(ENDPOINTS_PAYLOAD).model_endpoints("bytedance-seed/seedream-4.5")
        assert endpoints[0].provider_name == "Seed"
        assert endpoints[0].pricing[0].unit == "image"
        assert endpoints[0].pricing[0].cost_usd == 0.04


class TestAllowsIsThreeValued:
    """The catalog is incomplete, so 'not listed' must not mean 'not allowed'."""

    def test_listed_value_is_true(self):
        model = ImageModel.model_validate(MODELS_PAYLOAD["data"][0])
        assert model.allows("resolution", "4K") is True

    def test_contradicting_value_is_false(self):
        model = ImageModel.model_validate(MODELS_PAYLOAD["data"][0])
        assert model.allows("resolution", "8K") is False

    def test_undescribed_parameter_is_none_not_false(self):
        model = ImageModel.model_validate(MODELS_PAYLOAD["data"][0])
        assert model.allows("quality", "high") is None


class TestRequest:
    def test_unset_fields_are_omitted(self):
        payload = ImageGenerationRequest(model="m", prompt="p").to_payload()
        assert payload == {"model": "m", "prompt": "p"}
        assert "seed" not in payload and "resolution" not in payload

    def test_references_serialise_to_wire_shape(self):
        request = ImageGenerationRequest(
            model="m", prompt="p",
            input_references=[ImageReference.from_bytes(b"abc", "image/webp")])
        reference = request.to_payload()["input_references"][0]
        assert reference["type"] == "image_url"
        assert reference["image_url"]["url"].startswith("data:image/webp;base64,")

    def test_reference_from_url_passes_through(self):
        reference = ImageReference.from_url("https://example.com/a.png")
        assert reference.image_url.url == "https://example.com/a.png"


class TestResponse:
    def test_decodes_image_and_cost(self):
        result = ImageGenerationResponse.model_validate({
            "created": 1, "data": [{"b64_json": ONE_PIXEL, "media_type": "image/png"}],
            "usage": {"completion_tokens": 4175, "cost": 0.035}})
        assert result.first_image.startswith(b"\x89PNG")
        assert result.cost == 0.035
        assert len(result.images) == 1

    def test_empty_answer_has_no_image(self):
        result = ImageGenerationResponse.model_validate({"created": 1, "data": []})
        assert result.first_image is None
        assert result.cost is None


class TestFailureTaxonomy:
    def test_moderation_at_http_400_is_a_refusal_not_an_error(self):
        # MEASURED: this is how a Gemini moderation block actually arrives.
        problem = APIError("Gemini blocked this request through content moderation",
                           status_code=400)
        endpoint = make_endpoint(raises=problem)
        with pytest.raises(ImageRefused):
            endpoint.generate(ImageGenerationRequest(model="m", prompt="p"))

    @pytest.mark.parametrize("status", [401, 402, 403, 404])
    def test_unrecoverable_statuses_are_fatal(self, status):
        endpoint = make_endpoint(raises=APIError("nope", status_code=status))
        with pytest.raises(ImageGenerationFatal):
            endpoint.generate(ImageGenerationRequest(model="m", prompt="p"))

    def test_transient_failure_is_a_plain_error(self):
        endpoint = make_endpoint(raises=APIError("upstream hiccup", status_code=500))
        with pytest.raises(ImageGenerationError) as caught:
            endpoint.generate(ImageGenerationRequest(model="m", prompt="p"))
        assert not isinstance(caught.value, (ImageRefused, ImageGenerationFatal))

    def test_200_stating_a_reason_is_a_refusal(self):
        endpoint = make_endpoint({"data": [], "error": {"message": "declined by provider"}})
        with pytest.raises(ImageRefused):
            endpoint.generate(ImageGenerationRequest(model="m", prompt="p"))

    def test_200_with_no_image_and_no_reason_returns_normally(self):
        # The one case that cannot be told from a hiccup: surfaced, not guessed at.
        result = make_endpoint({"data": []}).generate(
            ImageGenerationRequest(model="m", prompt="p"))
        assert result.first_image is None


class TestValidation:
    def test_contradicting_resolution_is_rejected_locally(self):
        endpoint = make_endpoint(MODELS_PAYLOAD)
        request = ImageGenerationRequest(model="bytedance-seed/seedream-4.5",
                                         prompt="p", resolution="8K")
        with pytest.raises(ValidationError):
            endpoint.validate(request)

    def test_undescribed_parameter_is_allowed_through(self):
        # seedream-4.5 publishes no `quality` enum; that is not grounds to refuse.
        endpoint = make_endpoint(MODELS_PAYLOAD)
        endpoint.validate(ImageGenerationRequest(model="bytedance-seed/seedream-4.5",
                                                 prompt="p", quality="high"))

    def test_references_to_a_text_only_model_are_rejected(self):
        endpoint = make_endpoint(MODELS_PAYLOAD)
        request = ImageGenerationRequest(
            model="text-only/model", prompt="p",
            input_references=[ImageReference.from_url("https://example.com/a.png")])
        with pytest.raises(ValidationError):
            endpoint.validate(request)

    def test_unknown_model_is_not_validated_locally(self):
        # Never refuse on the strength of a listing that does not mention the model.
        endpoint = make_endpoint(MODELS_PAYLOAD)
        endpoint.validate(ImageGenerationRequest(model="brand/new", prompt="p",
                                                 resolution="8K"))

    def test_generate_does_not_validate_unless_asked(self):
        endpoint = make_endpoint({"data": [{"b64_json": ONE_PIXEL}]})
        result = endpoint.generate(ImageGenerationRequest(model="m", prompt="p"))
        assert result.first_image is not None


class TestMalformedResponse:
    """An unparseable body is one failed row, not an escaped ValueError."""

    def test_non_json_body_raises_the_image_taxonomy(self):
        endpoint = make_endpoint()
        endpoint.http_manager.post.return_value.json = Mock(
            side_effect=ValueError("Expecting value: line 1 column 1"))
        with pytest.raises(ImageGenerationError):
            endpoint.generate(ImageGenerationRequest(model="m", prompt="p"))


class TestValidationDoesNotRefetchPerCall:
    """A matrix run validates hundreds of times against a handful of models."""

    def test_catalogue_is_fetched_once_per_model(self):
        endpoint = make_endpoint(MODELS_PAYLOAD)
        request = ImageGenerationRequest(model="bytedance-seed/seedream-4.5",
                                         prompt="p", resolution="2K")
        for _ in range(5):
            endpoint.validate(request)
        assert endpoint.http_manager.get.call_count == 1

    def test_refresh_models_forces_a_refetch(self):
        endpoint = make_endpoint(MODELS_PAYLOAD)
        request = ImageGenerationRequest(model="bytedance-seed/seedream-4.5", prompt="p")
        endpoint.validate(request)
        endpoint.refresh_models()
        endpoint.validate(request)
        assert endpoint.http_manager.get.call_count == 2


class TestDiscoveryToleratesOddPayloads:
    """One unusual model must not break the listing for every other model."""

    def test_list_of_names_shape_does_not_break_parsing(self):
        # The older /api/v1/models route returns supported_parameters as a plain list.
        payload = {"data": [dict(MODELS_PAYLOAD["data"][0]),
                            {"id": "odd/model", "supported_parameters": ["resolution", "seed"]}]}
        models = make_endpoint(payload).list_models()
        assert len(models) == 2
        assert models[1].allowed_values("resolution") is None   # accepted, values unknown

    def test_non_string_enum_values_do_not_break_parsing(self):
        payload = {"data": [{"id": "odd/model", "supported_parameters": {
            "resolution": {"type": "enum", "values": [1, 2]}}}]}
        models = make_endpoint(payload).list_models()
        assert models[0].allowed_values("resolution") == ["1", "2"]

    def test_unknown_descriptor_type_is_kept(self):
        payload = {"data": [{"id": "odd/model", "supported_parameters": {
            "resolution": {"type": "brand_new_shape", "ceiling": 9}}}]}
        assert make_endpoint(payload).list_models()[0].id == "odd/model"


class TestMalformedImageData:
    """Bad base64 must not escape the taxonomy as binascii.Error."""

    @pytest.mark.parametrize("bad", ["data:image/png;base64,AAA", "!!!not base64!!!", "A"])
    def test_undecodable_payload_never_raises_and_returns_bytes(self, bad):
        from openrouter_client.models.images import ImageData
        # The point is the absence of binascii.Error, and a usable bytes result either way.
        assert isinstance(ImageData(b64_json=bad).image_bytes, bytes)

    def test_generate_with_undecodable_image_does_not_raise_binascii(self):
        endpoint = make_endpoint({"data": [{"b64_json": "!!!"}]})
        result = endpoint.generate(ImageGenerationRequest(model="m", prompt="p"))
        assert result.first_image is None

    def test_image_bytes_decodes_once(self):
        from openrouter_client.models.images import ImageData
        entry = ImageData(b64_json=ONE_PIXEL)
        assert entry.image_bytes is entry.image_bytes   # cached, not re-decoded


class TestClassificationOrder:
    """A dead key whose body mentions safety is fatal, not a refusal."""

    def test_fatal_status_wins_over_a_refusal_marker_in_the_body(self):
        problem = APIError("key disabled pending safety review", status_code=403)
        endpoint = make_endpoint(raises=problem)
        with pytest.raises(ImageGenerationFatal):
            endpoint.generate(ImageGenerationRequest(model="m", prompt="p"))

    def test_moderation_at_400_is_still_a_refusal(self):
        problem = APIError("blocked this request through content moderation", status_code=400)
        endpoint = make_endpoint(raises=problem)
        with pytest.raises(ImageRefused):
            endpoint.generate(ImageGenerationRequest(model="m", prompt="p"))

    def test_rate_limit_backoff_hint_survives_classification(self):
        from openrouter_client.exceptions import RateLimitExceeded
        problem = RateLimitExceeded("slow down", retry_after=30)
        problem.status_code = 429
        endpoint = make_endpoint(raises=problem)
        with pytest.raises(ImageGenerationError) as caught:
            endpoint.generate(ImageGenerationRequest(model="m", prompt="p"))
        assert caught.value.details.get("retry_after") == 30 or caught.value.retry_after == 30


class TestValidationFailuresAreClassified:
    def test_rejected_key_during_validation_is_fatal_not_raw_apierror(self):
        endpoint = make_endpoint(MODELS_PAYLOAD)
        endpoint.http_manager.get = Mock(side_effect=APIError("bad key", status_code=401))
        with pytest.raises(ImageGenerationFatal):
            endpoint.generate(ImageGenerationRequest(model="m", prompt="p"),
                              validate_request=True)


class TestErrorExtraction:
    def test_error_without_a_message_is_still_a_refusal(self):
        endpoint = make_endpoint({"data": [], "error": {"code": 400, "metadata": {"r": "x"}}})
        with pytest.raises(ImageRefused):
            endpoint.generate(ImageGenerationRequest(model="m", prompt="p"))

    def test_partial_result_returns_images_and_does_not_raise(self):
        endpoint = make_endpoint({"data": [{"b64_json": ONE_PIXEL}],
                                  "error": {"message": "one of two refused"}})
        result = endpoint.generate(ImageGenerationRequest(model="m", prompt="p"))
        assert len(result.images) == 1


class TestMediaTypeSniffing:
    """from_bytes must not label PNG bytes as webp."""

    @pytest.mark.parametrize("raw,expected", [
        (b"\x89PNG\r\n\x1a\n rest", "image/png"),
        (b"\xff\xd8\xff\xe0 rest", "image/jpeg"),
        (b"RIFF____WEBPVP8 ", "image/webp"),
        (b"GIF89a rest", "image/gif"),
    ])
    def test_magic_bytes_decide_the_declared_type(self, raw, expected):
        assert ImageReference.from_bytes(raw).image_url.url.startswith("data:%s;base64," % expected)

    def test_explicit_media_type_still_wins(self):
        ref = ImageReference.from_bytes(b"\x89PNG\r\n\x1a\n", "image/png")
        assert ref.image_url.url.startswith("data:image/png;base64,")
