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
    timeout=60.0,                             # Optional (kwarg)
)
```

**Parameters:**
- `api_key` (str, optional): Your OpenRouter API key. Falls back to the `OPENROUTER_API_KEY` env var.
- `provisioning_api_key` (str, optional): Provisioning key for credit/key management endpoints. Falls back to `OPENROUTER_PROVISIONING_API_KEY`.
- `secrets_manager` (SecretsManager, optional): Custom secrets manager. Defaults to `EnvironmentSecretsManager`.
- `base_url` (str, optional): Base URL for the API (default: "https://openrouter.ai/api/v1")
- `organization_id` (str, optional): Organization identifier.
- `reference_id` (str, optional): Reference identifier.

Additional keyword arguments are consumed via `**kwargs`:
- `timeout` (float, optional): Request timeout in seconds (default: 60.0)
- `retries` (int, optional): Number of HTTP retries (default: 3)
- `backoff_factor` (float, optional): Backoff factor between retries (default: 0.5)
- `rate_limit` (optional): Manual rate-limit override (default: None)
- `retry_config` (RetryConfig, optional): Opt-in 429 retry-with-backoff config (default: None; disabled). See [RetryConfig](#retryconfig).

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

**Returns:** `ChatCompletionResponse` or `Iterator[ChatCompletionStreamResponse]` if streaming

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

**Returns:** `CompletionsResponse` or `Iterator[CompletionsStreamResponse]` if streaming

## Models

### client.models.list()

List all available models.

```python
# Default: a list of model-id strings
model_ids = client.models.list()
for model_id in model_ids:
    print(model_id)

# details=True: full ModelsResponse with model objects
models = client.models.list(details=True)
for model in models.data:
    print(f"{model.id}: {model.name}")
```

**Parameters:**
- `details` (bool, optional): When `False` (default) returns a `List[str]` of model IDs; when `True` returns a `ModelsResponse`.

**Returns:** `List[str]` by default, or `ModelsResponse` (with `.data`) when `details=True`

### client.models.get()

Get information about a specific model.

```python
model = client.models.get("anthropic/claude-3-opus")
print(f"Context length: {model.context_length}")
print(f"Pricing: {model.pricing}")
```

**Parameters:**
- `model_id` (str): Model identifier

**Returns:** `ModelData` with detailed model information

### client.models.list_endpoints()

Get model endpoint information. Both `author` and `slug` are required.

```python
endpoints = client.models.list_endpoints(author="anthropic", slug="claude-3-opus")
print(endpoints.data)  # dict containing an `endpoints` list
```

**Parameters:**
- `author` (str): Model author/namespace (e.g., "anthropic")
- `slug` (str): Model slug (e.g., "claude-3-opus")

**Returns:** `ModelEndpointsResponse` with endpoint data

## Generations

### client.generations.get()

Get information about a specific generation.

```python
generation = client.generations.get("gen_123456789")
print(generation)  # plain dict
```

**Parameters:**
- `generation_id` (str): Generation identifier

**Returns:** `dict` with generation details

## Credits

### client.credits.get()

Get current credit balance and usage information. Requires a provisioning API key.

```python
credits = client.credits.get()
print(f"Total credits: ${credits['data']['total_credits']}")
print(f"Total usage: ${credits['data']['total_usage']}")
```

**Returns:** `dict` of shape `{"data": {"total_credits": float, "total_usage": float}}`

## API Keys

### client.keys.get_current()

Get information about the currently authenticated API key.

```python
keys_info = client.keys.get_current()
print(f"Label: {keys_info['data']['label']}")
print(f"Usage: {keys_info['data']['usage']}")
```

**Returns:** `dict` of shape `{"data": {"label", "usage", "limit", "is_free_tier", "rate_limit": {...}}}`

> To fetch a specific key instead of the current one, use `client.keys.get(key_hash)`, which requires a `key_hash` argument.

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
print(f"Requests: {rate_limits['requests']}")
print(f"Period: {rate_limits['period']}")
print(f"Cooldown: {rate_limits['cooldown']}")
```

**Returns:** `Dict` with keys `requests`, `period`, and `cooldown`

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

### CompletionsResponse

```python
class CompletionsResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[Usage]
```

### ModelData

```python
class ModelData:
    id: str
    name: str
    description: Optional[str]
    context_length: int
    pricing: ModelPricing
    top_provider: Optional[TopProvider]
```

### Usage

Returned inline on every response as `response.usage` (`Optional[Usage]`). Cost is
always populated automatically — do **not** pass `include={"usage": True}` (a deprecated
no-op).

```python
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: Optional[float]                              # total credits spent
    cost_details: Optional[CostDetails]               # upstream_inference_cost, cache_discount
    prompt_tokens_details: Optional[PromptTokensDetails]      # cached_tokens, cache_write_tokens, audio_tokens
    completion_tokens_details: Optional[CompletionTokensDetails]  # reasoning_tokens
    is_byok: Optional[bool]
```

> **BYOK note:** when `is_byok` is `True`, `cost` is `0.0` and the real spend is in
> `cost_details.upstream_inference_cost`.

## Error Handling

All methods can raise the following exceptions:

- `OpenRouterError`: Base exception for all errors
- `AuthenticationError`: Invalid API key or authentication failure
- `APIError`: General API error; carries `.status_code` (branch on it for 404/5xx)
- `RateLimitExceeded` (subclass of `APIError`): Rate limit exceeded; has `.retry_after` and `.status_code`
- `ProviderError` (subclass of `APIError`): Upstream provider error
- `ValidationError`: Invalid request parameters
- `ContextLengthExceededError` (subclass of `ValidationError`): Prompt exceeds the model's context length
- `StreamingError`: Error during a streaming response
- `ResumeError` (subclass of `StreamingError`): Failure resuming an interrupted stream

```python
from openrouter_client.exceptions import (
    AuthenticationError,
    RateLimitExceeded,
    ValidationError,
    APIError,
)

try:
    response = client.chat.create(...)
except AuthenticationError:
    print("Check your API key")
except RateLimitExceeded as e:
    print(f"Rate limited. Retry after: {e.retry_after}")
except ValidationError as e:
    print(f"Invalid request: {e}")
except APIError as e:
    # 404, 5xx, etc. — branch on the status code
    if e.status_code == 404:
        print("Not found")
    else:
        print(f"API error {e.status_code}: {e}")
```

## High-Level llm-style API

A higher-level, `llm`-inspired wrapper lives in `models/llm.py`. Get a model handle with
`get_model` and call `.prompt()` for one-shot calls or `.conversation()` for multi-turn.

```python
from openrouter_client import OpenRouterClient, get_model

client = OpenRouterClient(api_key="your-api-key")
model = get_model("anthropic/claude-3-opus", client)

# Returns a str
answer = model.prompt("What is the capital of France?", system="Be concise.")

# Returns a validated dict when a schema is provided
data = model.prompt("Extract the city and country.", schema=my_json_schema)

# Usage from the most recent prompt
print(model.last_usage)  # Optional[Usage]
```

`model.prompt(text, system=None, attachments=None, schema=None, **kwargs)` returns a `str`,
or a parsed-and-validated `dict` when `schema` is given.

### Conversations

```python
conv = model.conversation(system="You are a helpful assistant.")
conv.prompt("Hello!")
conv.prompt("And what did I just say?")

print(conv.last_usage)   # usage for the most recent turn
print(conv.total_usage)  # Optional[Usage] — cumulative across the conversation
print(conv.total_cost)   # float — cumulative cost (0.0 if none reported)
```

`conv.prompt(text, attachments=None, schema=None, **kwargs)` returns a `str`/`dict` just like
`model.prompt`.

## RetryConfig

Opt-in retry-with-backoff for HTTP 429 responses. Disabled by default; the **only** way to
enable retries is by passing a `RetryConfig` as `retry_config=`.

```python
from openrouter_client import OpenRouterClient, RetryConfig

client = OpenRouterClient(
    api_key="your-api-key",
    retry_config=RetryConfig(enabled=True, max_retries=5),
)
```

**Fields (with defaults):**
- `enabled` (bool): Enable 429 retries (default: `False`)
- `max_retries` (int): Maximum retry attempts (default: `5`)
- `base_delay` (float): Initial backoff delay in seconds (default: `1.0`)
- `factor` (float): Exponential backoff multiplier (default: `2.0`)
- `max_delay` (float): Maximum delay between retries in seconds (default: `30.0`)
- `jitter` (float): Random jitter fraction applied to each delay (default: `0.25`)
- `respect_retry_after` (bool): Honor the server's `Retry-After` header (default: `True`)