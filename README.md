# OpenRouter Python Client (Unofficial)

<img src="images/openrouter_client_logo.png" alt="OpenRouter Client (Unofficial) Logo" width="830" height="415">
<br>

An unofficial Python client for [OpenRouter](https://openrouter.ai/), providing a comprehensive interface for interacting with large language models through the OpenRouter API.

## Features

- **Full API Support**: Access all* OpenRouter endpoints including chat completions, text completions, model information, and more
- **Streaming Support**: Stream responses from chat and completion endpoints
- **Resume Support**: Resume from a previous request if it fails or is interrupted (incurs additional costs for input tokens)
- **Automatic Rate Limiting**: Automatically configures rate limits based on your API key's limits
- **Smart Retries**: Built-in retry logic with exponential backoff for reliable API communication
- **Type Safety**: Fully typed interfaces with Pydantic models
- **Async Support**: Both synchronous and asynchronous API interfaces
- **Safe Key Management**: Safely manage API keys with in-memory encryption and adapter class for using your choice of external key management system
- **Tiered API**: High-level client class, mid-level helper functions, and low-level endpoint and request classes

* _The Coinbase endpoint is not currently supported._

## Disclaimer

This project is independently developed and is not affiliated with, endorsed, or sponsored by OpenRouter, Inc.

Your use of the OpenRouter API through this interface is subject to OpenRouter's Terms of Service, Privacy Policy, and any other relevant agreements provided by OpenRouter, Inc. You are responsible for reviewing and complying with these terms.

This project is an open-source interface designed to interact with the OpenRouter API. It is provided "as-is," without any warranty, express or implied, under the terms of the Apache 2.0 License.

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
    provisioning_api_key="your-prov-key",  # Optional: for API key management
    base_url="https://openrouter.ai/api/v1",  # Base URL for API
    organization_id="your-org-id",  # Optional organization ID
    reference_id="your-ref-id",  # Optional reference ID
    log_level="INFO",  # Logging level
    timeout=60.0,  # Request timeout in seconds
    retries=3,  # Number of retries for failed requests
    backoff_factor=0.5,  # Exponential backoff factor
    rate_limit=None,  # Optional custom rate limit (auto-configured by default)
)
```

### Automatic Rate Limiting

The client automatically configures rate limits based on your API key's limits during initialization. It fetches your current key information and sets appropriate rate limits to prevent hitting API limits. This happens transparently when you create a new client instance.

If you need custom rate limiting, you can still provide your own configuration via the `rate_limit` parameter.

You can also calculate rate limits based on your remaining credits:

```python
# Calculate rate limits based on available credits
rate_limits = client.calculate_rate_limits()
print(f"Recommended: {rate_limits['requests']} requests per {rate_limits['period']} seconds")
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

### Context Length Management

The client provides built-in context length management:

```python
# Refresh model context lengths from the API
context_lengths = client.refresh_context_lengths()

# Get context length for a specific model
max_tokens = client.get_context_length("anthropic/claude-3-opus")
print(f"Claude 3 Opus supports up to {max_tokens} tokens")
```

### API Key Management

Manage API keys programmatically (requires provisioning API key):

```python
client = OpenRouterClient(
    api_key="your-api-key",
    provisioning_api_key="your-provisioning-key"
)

# Get current key information
key_info = client.keys.get_current()
print(f"Current usage: {key_info['data']['usage']} credits")
print(f"Rate limit: {key_info['data']['rate_limit']['requests']} requests per {key_info['data']['rate_limit']['interval']}")

# List all keys
keys = client.keys.list()

# Create a new key
new_key = client.keys.create(
    name="My New Key",
    label="Production API Key",
    limit=1000.0  # Credit limit
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

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
