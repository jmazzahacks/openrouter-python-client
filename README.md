# OpenRouter Python Client

An unofficial Python client for [OpenRouter](https://openrouter.ai/), providing a comprehensive interface for interacting with large language models through the OpenRouter API.

## Features

- **Full API Support**: Access all OpenRouter endpoints including chat completions, text completions, images, model information, and more
- **Streaming Support**: Stream responses from chat and completion endpoints
- **Resume Support**: Resume from a previous request if it fails or is interrupted (incurs additional costs for input tokens)
- **Rate Limiting & Retries**: Built-in rate limiting and retry logic for reliable API communication
- **Type Safety**: Fully typed interfaces with Pydantic models
- **Async Support**: Both synchronous and asynchronous API interfaces
- **Safe Key Management**: Safely manage API keys with in-memory encryption and adapter class for using your choice of external key management system
- **Tiered API**: High-level client class, mid-level helper functions, and low-level endpoint and request classes

## Installation

> **Note**: The PyPi package won't be published until initial testing for `openrouter-client-unofficial` is complete.

```bash
pip install openrouter-client-unofficial
```

## Quickstart

```python
from openrouter_client import OpenRouterClient

# Initialize the client
client = OpenRouterClient(
    api_key="your-api-key",  # Or set OPENROUTER_API_KEY environment variable
)

# Chat completion example
response = client.chat.completions.create(
    model="anthropic/claude-3-opus",  # Or any other model on OpenRouter
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about OpenRouter."}
    ]
)

print(response.choices[0].message.content)
```

## Client Configuration

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(
    api_key="your-api-key",  # API key for authentication
    base_url="https://openrouter.ai/api/v1",  # Base URL for API
    organization_id="your-org-id",  # Optional organization ID
    reference_id="your-ref-id",  # Optional reference ID
    log_level="INFO",  # Logging level
    timeout=60.0,  # Request timeout in seconds
    retries=3,  # Number of retries for failed requests
    backoff_factor=0.5,  # Exponential backoff factor
    rate_limit=None,  # Optional rate limit configuration
)
```

## Examples

### Streaming Responses

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(api_key="your-api-key")

# Stream the response
for chunk in client.chat.completions.create(
    model="openai/gpt-4",
    messages=[
        {"role": "user", "content": "Write a short poem about AI."}
    ],
    stream=True,
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Function Calling

```python
from openrouter_client import OpenRouterClient
from openrouter_client.models import create_tool, string_param, parse_tool_call_arguments

client = OpenRouterClient(api_key="your-api-key")

# Define a tool
weather_tool = create_tool(
    name="get_weather",
    description="Get the weather for a location",
    parameters={
        "properties": {
            "location": string_param(
                description="The city and state", 
                required=True
            ),
        },
        "required": ["location"],
    }
)

# Make a request with tool
response = client.chat.completions.create(
    model="anthropic/claude-3-opus",
    messages=[
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ],
    tools=[weather_tool],
)

# Process tool calls
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = parse_tool_call_arguments(tool_call)
    print(f"Tool called: {tool_call.function.name}")
    print(f"Arguments: {args}")
```

### Prompt Caching

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(api_key="your-api-key")

# OpenAI models: automatic caching for prompts > 1024 tokens
response = client.chat.completions.create(
    model="openai/gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": f"Here is a long document: {long_text}\n\nSummarize this document."}
    ],
    include={"usage": True}  # See caching metrics
)

# Anthropic models: explicit cache_control markers
response = client.chat.completions.create(
    model="anthropic/claude-3-opus",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is a long document:"},
                # Mark this part for caching
                {"type": "text", "text": long_text, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Summarize this document."}
            ]
        }
    ],
    include={"usage": True}
)
```

## Available Endpoints

- `client.chat`: Chat completions API
- `client.completions`: Text completions API
- `client.models`: Model information and selection
- `client.images`: Image generation capabilities
- `client.generations`: Generation statistics
- `client.credits`: Credit management
- `client.keys`: API key management
- `client.plugins`: Plugin operations
- `client.web`: Web search functionality

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
