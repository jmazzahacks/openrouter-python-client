# API Reference

Complete reference for all OpenRouter Python client endpoints and methods.

## Client Initialization

### OpenRouterClient

The main client class for interacting with the OpenRouter API.

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(
    api_key="your-api-key",
    base_url="https://openrouter.ai/api/v1",  # Optional
    http_referer="https://your-site.com",     # Optional
    x_title="Your App Name",                  # Optional
    timeout=30.0,                             # Optional
    max_retries=3                             # Optional
)
```

**Parameters:**
- `api_key` (str): Your OpenRouter API key
- `base_url` (str, optional): Base URL for the API (default: "https://openrouter.ai/api/v1")
- `http_referer` (str, optional): HTTP referer header for requests
- `x_title` (str, optional): X-Title header for requests
- `timeout` (float, optional): Request timeout in seconds (default: 30.0)
- `max_retries` (int, optional): Maximum number of retries (default: 3)

## Chat Completions

### client.chat.create()

Create a chat completion with message history.

```python
response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100,
    temperature=0.7,
    stream=False
)
```

**Parameters:**
- `model` (str): Model identifier (e.g., "anthropic/claude-3-opus")
- `messages` (List[Dict]): List of message objects with "role" and "content"
- `max_tokens` (int, optional): Maximum tokens to generate
- `temperature` (float, optional): Sampling temperature (0.0 to 2.0)
- `top_p` (float, optional): Nucleus sampling parameter
- `frequency_penalty` (float, optional): Frequency penalty (-2.0 to 2.0)
- `presence_penalty` (float, optional): Presence penalty (-2.0 to 2.0)
- `stop` (Union[str, List[str]], optional): Stop sequences
- `stream` (bool, optional): Enable streaming (default: False)
- `tools` (List[Dict], optional): Available tools for function calling
- `tool_choice` (Union[str, Dict], optional): Tool choice strategy

**Returns:** `ChatCompletionResponse` or `Iterator[ChatCompletionChunk]` if streaming

### client.chat.create() with Streaming

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

## Text Completions

### client.completions.create()

Create a text completion from a prompt.

```python
response = client.completions.create(
    model="openai/gpt-3.5-turbo-instruct",
    prompt="The capital of France is",
    max_tokens=50,
    temperature=0.7
)
```

**Parameters:**
- `model` (str): Model identifier
- `prompt` (Union[str, List[str]]): Input prompt(s)
- `max_tokens` (int, optional): Maximum tokens to generate
- `temperature` (float, optional): Sampling temperature
- `top_p` (float, optional): Nucleus sampling parameter
- `frequency_penalty` (float, optional): Frequency penalty
- `presence_penalty` (float, optional): Presence penalty
- `stop` (Union[str, List[str]], optional): Stop sequences
- `stream` (bool, optional): Enable streaming

**Returns:** `CompletionResponse` or `Iterator[CompletionChunk]` if streaming

## Models

### client.models.list()

List all available models.

```python
models = client.models.list()
for model in models.data:
    print(f"{model.id}: {model.name}")
```

**Returns:** `ModelsResponse` with list of available models

### client.models.get()

Get information about a specific model.

```python
model = client.models.get("anthropic/claude-3-opus")
print(f"Context length: {model.context_length}")
print(f"Pricing: {model.pricing}")
```

**Parameters:**
- `model_id` (str): Model identifier

**Returns:** `ModelInfo` with detailed model information

### client.models.list_endpoints()

Get model endpoint information.

```python
endpoints = client.models.list_endpoints()
print(endpoints.data)  # Dictionary of model endpoint information
```

**Returns:** `ModelEndpointsResponse` with endpoint data

## Generations

### client.generations.get()

Get information about a specific generation.

```python
generation = client.generations.get("gen_123456789")
print(f"Status: {generation.status}")
print(f"Created: {generation.created_at}")
```

**Parameters:**
- `generation_id` (str): Generation identifier

**Returns:** `GenerationResponse` with generation details

## Credits

### client.credits.get()

Get current credit balance and usage information.

```python
credits = client.credits.get()
print(f"Balance: ${credits.data.credits}")
print(f"Usage: ${credits.data.usage}")
```

**Returns:** `CreditsResponse` with balance and usage information

## API Keys

### client.keys.get()

Get information about API keys.

```python
keys_info = client.keys.get()
print(f"Label: {keys_info.data.label}")
print(f"Usage: {keys_info.data.usage}")
```

**Returns:** `KeysResponse` with API key information

## Client Utilities

### client.refresh_context_lengths()

Refresh cached context length information for models.

```python
client.refresh_context_lengths()
```

### client.get_context_length()

Get context length for a specific model.

```python
context_length = client.get_context_length("anthropic/claude-3-opus")
print(f"Context length: {context_length}")
```

**Parameters:**
- `model_id` (str): Model identifier

**Returns:** `int` - Context length in tokens

### client.calculate_rate_limits()

Calculate current rate limits based on credit balance.

```python
rate_limits = client.calculate_rate_limits()
print(f"Requests per minute: {rate_limits['requests_per_minute']}")
print(f"Tokens per minute: {rate_limits['tokens_per_minute']}")
```

**Returns:** `Dict` with rate limit information

## Response Models

### ChatCompletionResponse

```python
class ChatCompletionResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage]
```

### CompletionResponse

```python
class CompletionResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[Usage]
```

### ModelInfo

```python
class ModelInfo:
    id: str
    name: str
    description: str
    context_length: int
    pricing: ModelPricing
    top_provider: Optional[str]
```

## Error Handling

All methods can raise the following exceptions:

- `OpenRouterError`: Base exception for all API errors
- `AuthenticationError`: Invalid API key or authentication failure
- `RateLimitError`: Rate limit exceeded
- `ValidationError`: Invalid request parameters
- `NotFoundError`: Requested resource not found
- `ServerError`: Server-side error (5xx status codes)

```python
from openrouter_client.exceptions import *

try:
    response = client.chat.create(...)
except AuthenticationError:
    print("Check your API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}")
except ValidationError as e:
    print(f"Invalid request: {e}")
```