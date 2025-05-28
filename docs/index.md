# OpenRouter Python Client Documentation

Welcome to the unofficial OpenRouter Python client documentation! This comprehensive library provides a type-safe, feature-rich interface for interacting with the OpenRouter API.

## Quick Start

```python
from openrouter_client import OpenRouterClient

# Initialize the client
client = OpenRouterClient(api_key="your-api-key")

# Create a chat completion
response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[{"role": "user", "content": "Hello, world!"}]
)

print(response.choices[0].message.content)
```

## Features

- **Full API Support**: Chat completions, text completions, models, generations, credits, keys
- **Streaming Support**: Real-time streaming for chat and text completions  
- **Automatic Rate Limiting**: Smart rate limiting based on API key limits using SmartSurge
- **Type Safety**: Fully typed interfaces with Pydantic models
- **Function Calling**: Built-in support for OpenAI-style function calling with decorators
- **Prompt Caching**: Support for prompt caching on compatible models
- **Safe Key Management**: Secure API key handling with encryption and extensible secrets management
- **Context Length Management**: Automatic tracking and querying of model context lengths
- **Comprehensive Testing**: Extensive test suite with local unit tests and remote integration tests

## Documentation Sections

- [Getting Started](getting-started.md) - Installation and basic usage
- [API Reference](api-reference.md) - Complete API documentation
- [Advanced Features](advanced-features.md) - Rate limiting, streaming, function calling
- [Examples](examples.md) - Practical usage examples
- [Configuration](configuration.md) - Client configuration and customization

## Installation

```bash
pip install openrouter-client-unofficial
```

For development:
```bash
pip install openrouter-client-unofficial[dev]
```

## Support

- [GitHub Repository](https://github.com/dingo-actual/openrouter-python-client)
- [Issue Tracker](https://github.com/dingo-actual/openrouter-python-client/issues)