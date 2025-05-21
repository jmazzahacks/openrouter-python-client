"""Endpoint handlers for OpenRouter API.  

This module exports all endpoint handler classes for accessing
the various OpenRouter API endpoints, such as chat completions,
models, files, etc.

Exported:
- BaseEndpoint: Base class for all endpoint handlers
- ChatEndpoint: Handler for chat completions endpoints
- CompletionsEndpoint: Handler for completions endpoints
- ModelsEndpoint: Handler for models endpoint
- CreditsEndpoint: Handler for credits and payment endpoints
- GenerationsEndpoint: Handler for generations history endpoints
- ImagesEndpoint: Handler for image generation endpoints
- KeysEndpoint: Handler for API keys management
- PdfEndpoint: Handler for PDF processing endpoints
- PluginsEndpoint: Handler for plugins management
- WebEndpoint: Handler for web page processing endpoints
"""

from .base import BaseEndpoint
from .chat import ChatEndpoint
from .completions import CompletionsEndpoint
from .models import ModelsEndpoint
from .credits import CreditsEndpoint
from .generations import GenerationsEndpoint
from .images import ImagesEndpoint
from .keys import KeysEndpoint
from .plugins import PluginsEndpoint
from .web import WebEndpoint

__all__ = [
    'BaseEndpoint',
    'ChatEndpoint',
    'CompletionsEndpoint',
    'ModelsEndpoint',
    'CreditsEndpoint',
    'GenerationsEndpoint',
    'ImagesEndpoint',
    'KeysEndpoint',
    'PluginsEndpoint',
    'WebEndpoint',
]