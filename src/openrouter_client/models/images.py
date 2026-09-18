"""Models for OpenRouter's dedicated Image API (`/api/v1/images`).

This is a SEPARATE endpoint group from chat completions, not a modality flag on
`chat.create`. It has its own request shape, its own response shape, its own
capability-discovery routes and its own per-endpoint pricing.

WHAT THE WIRE ACTUALLY LOOKS LIKE
---------------------------------
Request::

    {"model": ..., "prompt": ..., "input_references": [...],
     "aspect_ratio": "2:3", "resolution": "2K", "seed": 7, "n": 1}

Response::

    {"created": ..., "data": [{"b64_json": "...", "media_type": "image/png"}],
     "usage": {"completion_tokens": 4175, "cost": 0.04}}

THREE THINGS MEASURED AGAINST THE LIVE API (2026-09-18) THAT SHAPE THESE MODELS
------------------------------------------------------------------------------
1. `supported_parameters` from model discovery is NOT authoritative. The catalog
   advertises a `1K` tier for `bytedance-seed/seedream-4.5`; the API rejects it
   ("requires at least 3,686,400 output pixels"). So capability data is a hint
   that can rule a request out early, never a guarantee it will be accepted --
   `ImageModel.allows` is deliberately three-valued for that reason.
2. A resolution tier is a TOTAL PIXEL BUDGET, not an edge length, and the budget
   differs per provider. At 2:3, Seedream's `4K` is 2732x4096 while Gemini's is
   3392x5056.
3. The returned image may not be the resolution that was asked for, while still
   being billed at the requested tier's price. Callers that care must MEASURE the
   returned bytes; nothing in the response states the dimensions. `ImageData`
   therefore exposes the decoded bytes rather than pretending to know the size.
"""
from __future__ import annotations

import base64
import binascii
from functools import cached_property
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

#: Magic-byte prefixes, longest-first so RIFF/WEBP is not shadowed.
_MAGIC: tuple = ((b"\x89PNG\r\n\x1a\n", "image/png"),
                 (b"\xff\xd8\xff", "image/jpeg"),
                 (b"GIF87a", "image/gif"),
                 (b"GIF89a", "image/gif"))


def sniff_media_type(data: bytes, fallback: str = "application/octet-stream") -> str:
    """The media type of `data` from its magic bytes, or `fallback`."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    for prefix, media_type in _MAGIC:
        if data.startswith(prefix):
            return media_type
    return fallback


class CapabilityDescriptor(BaseModel):
    """One entry of a model's `supported_parameters`.

    Shapes seen in the wild: ``{"type": "enum", "values": [...]}`` and
    ``{"type": "boolean"}``. Unknown descriptor types are kept rather than
    rejected -- a new one must not break discovery.
    """
    model_config = ConfigDict(extra="allow")

    type: Optional[str] = Field(None, description="Descriptor kind, e.g. 'enum' or 'boolean'")
    # `Any`, not `str`: pydantic v2 does not coerce int -> str, so a numeric enum
    # ("values": [1, 2]) would fail the whole listing. Values are normalised to strings
    # by `ImageModel.allowed_values` at the point of comparison instead.
    values: Optional[List[Any]] = Field(None, description="Allowed values, for an enum")


class ImageArchitecture(BaseModel):
    """Input/output modalities an image model accepts and produces."""
    model_config = ConfigDict(extra="allow")

    input_modalities: List[str] = Field(default_factory=list)
    output_modalities: List[str] = Field(default_factory=list)


class ImageModel(BaseModel):
    """One model from `GET /api/v1/images/models`."""
    model_config = ConfigDict(extra="allow")

    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    created: Optional[int] = None
    architecture: Optional[ImageArchitecture] = None
    # A UNION, deliberately. The older `/api/v1/models` route returns this field as a
    # plain list of names, and a provider may add a descriptor shape nobody has seen. A
    # strict Dict here would make ONE odd model fail `model_validate` for the ENTIRE
    # listing -- exactly the breakage this module claims not to have.
    supported_parameters: Union[Dict[str, CapabilityDescriptor], List[str]] = Field(
        default_factory=dict)
    supports_streaming: bool = False
    endpoints: Optional[str] = Field(None, description="URL of the per-endpoint records")

    @property
    def accepts_references(self) -> bool:
        """Whether this model can take `input_references` at all."""
        if self.architecture is None:
            return False
        return "image" in self.architecture.input_modalities

    def allowed_values(self, parameter: str) -> Optional[List[str]]:
        """Enum values for `parameter`, or None when it is unconstrained/unknown.

        Returns None for the list-of-names shape too: that form says a parameter is
        accepted but not which values are legal, which is not something to validate on.
        """
        if not isinstance(self.supported_parameters, dict):
            return None
        descriptor = self.supported_parameters.get(parameter)
        if descriptor is None or descriptor.values is None:
            return None
        return [str(value) for value in descriptor.values]

    def allows(self, parameter: str, value: str) -> Optional[bool]:
        """Three-valued: True (listed), False (contradicts the listing), None (unknown).

        None is returned when the model does not describe the parameter at all --
        NOT a reason to refuse the request, because the catalog is incomplete (see
        the module docstring). Only an explicit contradiction returns False.
        """
        allowed = self.allowed_values(parameter)
        if allowed is None:
            return None
        return value in allowed


class ImageModelsResponse(BaseModel):
    """The payload of `GET /api/v1/images/models`."""
    model_config = ConfigDict(extra="allow")

    data: List[ImageModel] = Field(default_factory=list)


class ImageEndpointPricing(BaseModel):
    """One pricing line of a per-provider endpoint record.

    `unit` is the thing that matters and it is not consistent across providers:
    "image" means a flat charge per output image, "token" means the charge scales
    with resolution. A `billable` of "input_reference" is charged per reference
    attached, which makes a multi-reference workflow far dearer than the output
    price alone suggests.
    """
    model_config = ConfigDict(extra="allow")

    billable: Optional[str] = None
    unit: Optional[str] = None
    cost_usd: Optional[float] = None


class ImageModelEndpoint(BaseModel):
    """One provider's record for a model, from the `/endpoints` route.

    Definitive where the aggregated model listing is only indicative.
    """
    model_config = ConfigDict(extra="allow")

    provider_name: Optional[str] = None
    provider_slug: Optional[str] = None
    provider_tag: Optional[str] = None
    supported_parameters: Dict[str, CapabilityDescriptor] = Field(default_factory=dict)
    allowed_passthrough_parameters: List[str] = Field(default_factory=list)
    supports_streaming: bool = False
    pricing: List[ImageEndpointPricing] = Field(default_factory=list)


class ImageModelEndpointsResponse(BaseModel):
    """The payload of `GET /api/v1/images/models/{id}/endpoints`."""
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    endpoints: List[ImageModelEndpoint] = Field(default_factory=list)


class ImageReferenceUrl(BaseModel):
    """The `image_url` container of one reference."""
    url: str = Field(..., description="An https:// URL or a base64 data URL")


class ImageReference(BaseModel):
    """One entry of `input_references`.

    References are inlined into EVERY request, so a caller sending the same
    reference across many generations should encode it once and reuse the object.
    """
    type: str = "image_url"
    image_url: ImageReferenceUrl

    @classmethod
    def from_url(cls, url: str) -> "ImageReference":
        """A reference by https:// URL or by an already-built data URL."""
        return cls(image_url=ImageReferenceUrl(url=url))

    @classmethod
    def from_bytes(cls, data: bytes, media_type: Optional[str] = None) -> "ImageReference":
        """A reference from raw image bytes, encoded as a data URL.

        `media_type` is sniffed from the magic bytes when omitted. It used to default to
        `image/webp`, which silently mislabelled PNG or JPEG input -- a data URL whose
        declared type contradicts its bytes is something a provider may reject or
        misdecode, and the caller would have no hint why.
        """
        return cls(image_url=ImageReferenceUrl(
            url="data:%s;base64,%s" % (media_type or sniff_media_type(data),
                                       base64.b64encode(data).decode("ascii"))))


class ImageGenerationRequest(BaseModel):
    """A request to `POST /api/v1/images`.

    Passed as a single object rather than ten keyword arguments: the object is
    also the natural unit to log, diff and replay, which is what a model
    comparison run needs.

    Only fields that were actually set are sent -- an unset parameter must be
    omitted so the provider applies its own default rather than being handed a
    null.
    """
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str
    input_references: Optional[List[ImageReference]] = None
    aspect_ratio: Optional[str] = Field(None, description="e.g. '1:1', '2:3', '16:9'")
    resolution: Optional[str] = Field(None, description="e.g. '1K', '2K', '4K' -- a TOTAL "
                                                        "PIXEL BUDGET, not an edge length")
    seed: Optional[int] = None
    n: Optional[int] = None
    quality: Optional[str] = None
    background: Optional[str] = None
    output_format: Optional[str] = None
    provider: Optional[Dict[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        """The JSON body, with unset fields omitted."""
        return self.model_dump(exclude_none=True)


class ImageData(BaseModel):
    """One generated image.

    The response does NOT state the image's dimensions, and a provider may return
    a different resolution than the one requested while still billing the
    requested tier. A caller that depends on the size must measure `image_bytes`.
    """
    model_config = ConfigDict(extra="allow")

    b64_json: Optional[str] = None
    media_type: Optional[str] = None

    @cached_property
    def image_bytes(self) -> bytes:
        """The decoded image. Empty when the entry carried no usable image.

        Cached: `generate` decodes once to test for emptiness and a caller decodes again
        on every `.images` access, which for a 4K batch is megabytes re-decoded per read.

        Malformed base64 yields b"" rather than raising. A `binascii.Error` escaping from
        a property would bypass the image failure taxonomy entirely and abort a batch that
        one bad row should only have cost one image of; the caller sees "no image", which
        is what actually happened.
        """
        if not self.b64_json:
            return b""
        try:
            return base64.b64decode(self.b64_json, validate=False)
        except (binascii.Error, ValueError):
            return b""


class ImageUsage(BaseModel):
    """Token counts and cost for one generation. `cost` is in USD."""
    model_config = ConfigDict(extra="allow")

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None


class ImageGenerationResponse(BaseModel):
    """The payload of `POST /api/v1/images`."""
    model_config = ConfigDict(extra="allow")

    created: Optional[int] = None
    data: List[ImageData] = Field(default_factory=list)
    usage: Optional[ImageUsage] = None

    @property
    def images(self) -> List[bytes]:
        """Every generated image's bytes, in order, skipping empty entries."""
        found: List[bytes] = []
        for entry in self.data:
            raw = entry.image_bytes
            if raw:
                found.append(raw)
        return found

    @property
    def first_image(self) -> Optional[bytes]:
        """The first generated image, or None when the answer carried none."""
        images = self.images
        return images[0] if images else None

    @property
    def cost(self) -> Optional[float]:
        """USD cost of this generation, when the provider reported one."""
        return self.usage.cost if self.usage else None
