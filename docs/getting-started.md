# Getting Started

This guide will help you get up and running with the OpenRouter Python client quickly.

## Installation

Install the package using pip:

```bash
pip install openrouter-client-unofficial
```

For development with additional tools:
```bash
pip install openrouter-client-unofficial[dev]
```

## Authentication

You'll need an OpenRouter API key to use this client. You can get one from [OpenRouter](https://openrouter.ai/keys).

### Basic Authentication

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(api_key="your-api-key-here")
```

### Environment Variables

You can also set your API key as an environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Then initialize the client without passing the key:

```python
client = OpenRouterClient()  # Will automatically use OPENROUTER_API_KEY
```

## Your First Request

Here's a simple example to get you started:

```python
from openrouter_client import OpenRouterClient

# Initialize the client
client = OpenRouterClient(api_key="your-api-key")

# Create a chat completion
response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

# Print the response
print(response.choices[0].message.content)
```

## Available Models

You can list available models and their details:

```python
# Get all available models
models = client.models.list()
for model in models.data:
    print(f"{model.id}: {model.name}")

# Get specific model information
model_info = client.models.get("anthropic/claude-3-opus")
print(f"Context length: {model_info.context_length}")
print(f"Price per token: {model_info.pricing}")
```

## Streaming Responses

For real-time responses, you can use streaming:

```python
stream = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Error Handling

The client provides comprehensive error handling:

```python
from openrouter_client import OpenRouterClient
from openrouter_client.exceptions import (
    OpenRouterError,
    AuthenticationError,
    RateLimitError,
    ValidationError
)

client = OpenRouterClient(api_key="your-api-key")

try:
    response = client.chat.create(
        model="anthropic/claude-3-opus",
        messages=[{"role": "user", "content": "Hello!"}]
    )
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}")
except ValidationError as e:
    print(f"Invalid request: {e}")
except OpenRouterError as e:
    print(f"API error: {e}")
```

## Context Management

Use the client as a context manager for automatic resource cleanup:

```python
with OpenRouterClient(api_key="your-api-key") as client:
    response = client.chat.create(
        model="anthropic/claude-3-opus",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)
# Client resources are automatically cleaned up here
```

## Next Steps

- Explore the [API Reference](api-reference.md) for detailed endpoint documentation
- Learn about [Advanced Features](advanced-features.md) like function calling and rate limiting
- Check out practical [Examples](examples.md)
- Customize your setup with [Configuration](configuration.md) options