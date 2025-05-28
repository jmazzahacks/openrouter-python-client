# OpenRouter Python Client (Unofficial)

<img src="images/openrouter_client_logo.png" alt="OpenRouter Client (Unofficial) Logo" width="830" height="415">
<br>

An unofficial Python client for [OpenRouter](https://openrouter.ai/), providing a comprehensive interface for interacting with large language models through the OpenRouter API.

## Features

- **Full API Support (Almost)**: Access all major OpenRouter endpoints including chat completions, text completions, model information, generations, credits, and API key management
- **Streaming Support**: Stream responses from chat and completion endpoints
- **Automatic Rate Limiting**: Automatically configures rate limits based on your API key's limits using SmartSurge
- **Smart Retries**: Built-in retry logic with exponential backoff for reliable API communication
- **Type Safety**: Fully typed interfaces with Pydantic models for all request and response data
- **Tool Calling**: Built-in support for tool-calling with helper functions and decorators
- **Safe Key Management**: Secure API key management with in-memory encryption and extensible secrets management
- **Comprehensive Testing**: Extensive test suite with both local unit tests and remote integration tests

## Disclaimer

This project is independently developed and is not affiliated with, endorsed, or sponsored by OpenRouter, Inc.

Your use of the OpenRouter API through this interface is subject to OpenRouter's Terms of Service, Privacy Policy, and any other relevant agreements provided by OpenRouter, Inc. You are responsible for reviewing and complying with these terms.

This project is an open-source interface designed to interact with the OpenRouter API. It is provided "as-is," without any warranty, express or implied, under the terms of the Apache 2.0 License.

## Installation

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
response = client.chat.create(
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
for chunk in client.chat.create(
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
from openrouter_client import OpenRouterClient, tool
from openrouter_client.models import ChatCompletionTool, FunctionDefinition, StringParameter, FunctionParameters

client = OpenRouterClient(api_key="your-api-key")

# Method 1: Using the @tool decorator (recommended)
@tool
def get_weather(location: str) -> str:
    """Get the weather for a location.
    
    Args:
        location: The city and state
        
    Returns:
        Weather information for the location
    """
    # Your weather API logic here
    return f"The weather in {location} is sunny."

# Method 2: Manual tool definition
weather_tool = ChatCompletionTool(
    type="function",
    function=FunctionDefinition(
        name="get_weather",
        description="Get the weather for a location",
        parameters=FunctionParameters(
            type="object",
            properties={
                "location": StringParameter(
                    type="string",
                    description="The city and state"
                )
            },
            required=["location"]
        )
    )
)

# Make a request with tool
response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ],
    tools=[get_weather],  # Using the decorated function
)

# Process tool calls
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Tool called: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

### Prompt Caching

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(api_key="your-api-key")

# OpenAI models: automatic caching for prompts > 1024 tokens
response = client.chat.create(
    model="openai/gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": f"Here is a long document: {long_text}\n\nSummarize this document."}
    ]
)

# Anthropic models: explicit cache_control markers
response = client.chat.create(
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
    ]
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
- `client.generations`: Generation metadata and details
- `client.credits`: Credit management and usage tracking
- `client.keys`: API key management and provisioning

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
