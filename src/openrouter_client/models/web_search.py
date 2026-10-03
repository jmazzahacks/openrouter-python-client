"""Web-search options and response annotations, including future annotation types."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class WebSearchPlugin(BaseModel):
    """Options for OpenRouter's web plugin; omitted options use API defaults."""

    model_config = ConfigDict(extra="allow")

    id: Literal["web"] = "web"
    enabled: Optional[bool] = None
    engine: Optional[
        Literal["native", "exa", "firecrawl", "parallel", "perplexity"]
    ] = None
    mode: Optional[str] = None
    max_results: Optional[int] = Field(None, ge=1)
    search_prompt: Optional[str] = None
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None


class WebSearchOptions(BaseModel):
    """Native search context settings; provider-specific fields are preserved."""

    model_config = ConfigDict(extra="allow")

    search_context_size: Optional[Literal["low", "medium", "high"]] = None
    user_location: Optional[Dict[str, Any]] = None


class UrlCitation(BaseModel):
    """A cited source. Excerpts and character offsets may be absent."""

    model_config = ConfigDict(extra="allow")

    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    start_index: Optional[int] = Field(None, ge=0)
    end_index: Optional[int] = Field(None, ge=0)


class Annotation(BaseModel):
    """Fallback for unfamiliar annotations, preserving every returned field."""

    model_config = ConfigDict(extra="allow")

    type: str


class UrlCitationAnnotation(Annotation):
    """The annotation envelope containing a typed URL citation."""

    type: Literal["url_citation"] = "url_citation"
    url_citation: UrlCitation


MessageAnnotation = Union[UrlCitationAnnotation, Annotation]
Plugin = Union[WebSearchPlugin, Dict[str, Any]]
