# Advanced Features

This guide covers advanced features and configurations of the OpenRouter Python client.

## Function Calling

The client provides comprehensive support for OpenAI-style function calling with convenient decorators and utilities.

### Using the @tool Decorator

The `@tool` decorator automatically converts Python functions into OpenRouter-compatible tools:

```python
import json
from openrouter_client import OpenRouterClient, tool

@tool()
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city and state/country
        unit: Temperature unit (celsius or fahrenheit)
    """
    # Your weather API logic here
    return f"The weather in {location} is 22°{unit[0].upper()}"

client = OpenRouterClient(api_key="your-api-key")

response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[get_weather.as_chat_completion_tool],
    tool_choice="auto"
)

# Handle tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        if tool_call.function.name == "get_weather":
            # Arguments come back as a JSON string — parse and call the tool yourself
            args = json.loads(tool_call.function.arguments)
            result = get_weather(**args)
            print(result)
```

### Manual Tool Definition

You can also define tools manually using helper functions:

```python
from openrouter_client.tools import (
    build_tool_definition,
    build_parameter_schema
)
from openrouter_client.models import (
    StringParameter,
    NumberParameter,
    ArrayParameter,
    ChatCompletionTool
)
from typing import List

# Define the function with proper type hints
def search_database(query: str, limit: int = 10, categories: List[str] = None) -> dict:
    """Search for items in a database.
    
    Args:
        query: Search query
        limit: Maximum results (1-100)
        categories: Search categories
    """
    pass

# Create tool from function
search_tool = build_tool_definition(search_database)

response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[{"role": "user", "content": "Search for books about AI"}],
    tools=[search_tool],
    tool_choice="auto"
)
```

### Processing Tool Calls

Handle tool calls in your application:

```python
import json

def handle_tool_calls(response):
    """Process tool calls from a chat response."""
    if not response.choices[0].message.tool_calls:
        return response.choices[0].message.content
    
    messages = [{"role": "user", "content": "Original user message"}]
    messages.append(response.choices[0].message.dict())
    
    for tool_call in response.choices[0].message.tool_calls:
        # Parse arguments safely
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            print(f"Invalid tool arguments: {e}")
            continue
            
        # Execute tool (implement your logic here)
        result = execute_tool(tool_call.function.name, args)
        
        # Add tool response to conversation
        tool_response = {
            "role": "tool",
            "content": json.dumps(result) if isinstance(result, dict) else str(result),
            "tool_call_id": tool_call.id
        }
        messages.append(tool_response)
    
    # Continue conversation with tool results
    return client.chat.create(
        model="anthropic/claude-3-opus",
        messages=messages
    )
```

## Streaming Responses

### Basic Streaming

Stream responses for real-time output:

```python
stream = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[{"role": "user", "content": "Write a long story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Streaming with Function Calls

Handle function calls in streaming mode:

```python
import json

def handle_streaming_with_tools():
    stream = client.chat.create(
        model="anthropic/claude-3-opus",
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[get_weather.as_chat_completion_tool],
        stream=True
    )
    
    accumulated_content = ""
    tool_calls = []
    
    for chunk in stream:
        delta = chunk.choices[0].delta
        
        if delta.content:
            accumulated_content += delta.content
            print(delta.content, end="", flush=True)
        
        if delta.tool_calls:
            # Accumulate tool calls
            for i, tool_call in enumerate(delta.tool_calls):
                if i >= len(tool_calls):
                    tool_calls.append({
                        "id": tool_call.id,
                        "function": {"name": "", "arguments": ""}
                    })
                
                if tool_call.function.name:
                    tool_calls[i]["function"]["name"] = tool_call.function.name
                if tool_call.function.arguments:
                    tool_calls[i]["function"]["arguments"] += tool_call.function.arguments
    
    # Process completed tool calls
    for tool_call in tool_calls:
        if tool_call["function"]["name"] == "get_weather":
            args = json.loads(tool_call["function"]["arguments"])
            result = get_weather(**args)
            print(f"\nWeather result: {result}")
```

## Rate Limiting and Smart Retry

The client includes intelligent rate limiting powered by SmartSurge.

### Automatic Rate Limiting

Rate limiting is handled automatically based on your API key limits. On construction the
client fetches the key's rate limit and applies it to every endpoint.

### Opt-in 429 Retry with Backoff

Automatic retry of HTTP 429 (rate-limit) responses is **opt-in** and configured with a
`RetryConfig`. It is disabled by default. There is no `max_retries` constructor argument —
retries are driven only through `retry_config`:

```python
from openrouter_client import OpenRouterClient, RetryConfig

client = OpenRouterClient(
    api_key="your-api-key",
    retry_config=RetryConfig(
        enabled=True,
        max_retries=5,      # up to 5 retries on 429
        base_delay=1.0,     # first backoff delay in seconds
        factor=2.0,         # exponential growth factor
        max_delay=30.0,     # cap on any single backoff
        jitter=0.25,        # +/- randomization to avoid thundering herd
    ),
)

# With retry_config enabled, 429 responses are retried transparently
for i in range(100):
    response = client.chat.create(
        model="anthropic/claude-3-opus",
        messages=[{"role": "user", "content": f"Request {i}"}]
    )
    print(f"Completed request {i}")
```

### Manual Rate Limit Management

Check and manage rate limits manually:

```python
# Check current rate limits — calculate_rate_limits() returns a dict with the
# keys "requests", "period", and "cooldown".
rate_limits = client.calculate_rate_limits()
print(f"Allowed requests: {rate_limits['requests']}")
print(f"Per period (seconds): {rate_limits['period']}")
print(f"Cooldown (seconds): {rate_limits['cooldown']}")

# Check credit balance — credits.get() returns a plain dict (requires a provisioning key)
credits = client.credits.get()
print(f"Total credits: ${credits['data']['total_credits']}")
print(f"Total usage: ${credits['data']['total_usage']}")

# Wait for rate limit reset if needed
import time
if rate_limits['requests'] < 10:
    print("Low rate limit, waiting...")
    time.sleep(rate_limits['cooldown'])  # Wait for reset
```

## Context Length Management

Automatically track and manage model context lengths:

```python
# Get context length for a model
context_length = client.get_context_length("anthropic/claude-3-opus")
print(f"Claude 3 Opus context length: {context_length}")

# Refresh context length cache
client.refresh_context_lengths()

# Use context length for message truncation
def truncate_messages(messages, model, reserve_tokens=1000):
    """Truncate messages to fit within model context length."""
    max_tokens = client.get_context_length(model) - reserve_tokens
    
    # Simple truncation (implement token counting as needed)
    total_chars = sum(len(msg["content"]) for msg in messages)
    if total_chars > max_tokens * 4:  # Rough estimate: 4 chars per token
        # Keep system message and last few user messages
        system_msgs = [msg for msg in messages if msg["role"] == "system"]
        other_msgs = [msg for msg in messages if msg["role"] != "system"]
        
        # Take last N messages that fit
        truncated = system_msgs + other_msgs[-10:]
        return truncated
    
    return messages

# Use in chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    # ... many messages ...
    {"role": "user", "content": "Latest question"}
]

truncated_messages = truncate_messages(messages, "anthropic/claude-3-opus")
response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=truncated_messages
)
```

## Prompt Caching

For compatible models (Anthropic Claude, OpenAI, DeepSeek), use prompt caching to reduce
costs. For Anthropic Claude, `cache_control` goes on individual content **parts** (the
`content` is a list of typed parts), marking the part that should be cached:

```python
# Cache a long system prompt by placing cache_control on the text part
response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an expert programmer...",  # Long system prompt
                    "cache_control": {"type": "ephemeral"}  # Cache this part
                }
            ]
        },
        {"role": "user", "content": "Write a Python function"}
    ]
)

# Subsequent requests with the same cached content will be cheaper
response2 = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an expert programmer...",  # Same cached content
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        },
        {"role": "user", "content": "Write a JavaScript function"}
    ]
)
```

OpenAI caching is automatic for prompts over 1024 tokens.

## Cost & Usage Tracking

Cost is returned inline on every response — you do not need to pass any `include` flag.
`response.usage` is an `Optional[Usage]` with token counts plus a `cost` field (total credits
spent) and a `cost_details` breakdown:

```python
response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[{"role": "user", "content": "Hello!"}]
)

usage = response.usage
if usage:
    print(f"Prompt tokens:     {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Cost (credits):    {usage.cost}")

    # BYOK (bring-your-own-key): usage.cost is 0.0 and the real upstream spend
    # is reported separately.
    if usage.is_byok:
        print(f"Upstream cost: {usage.cost_details.upstream_inference_cost}")
```

The high-level llm-style API tracks usage for you:

```python
from openrouter_client import get_model

model = get_model("anthropic/claude-3-opus", client)
model.prompt("Summarize the theory of relativity.")
print(model.last_usage)          # Usage from the most recent prompt

conversation = model.conversation()
conversation.prompt("What is a black hole?")
conversation.prompt("And a neutron star?")
print(conversation.total_cost)   # cumulative cost (float, 0.0 if none reported)
print(conversation.total_usage)  # cumulative Usage across the conversation
```

## Secure Key Management

### Automatic Key Encryption

Keys are encrypted in memory automatically when PyNaCl is installed — there is no
`encrypt_key` flag to toggle, and you do not construct the auth manager yourself. Pass
your key to the client, which builds and manages an encrypted `AuthManager` internally:

```python
from openrouter_client import OpenRouterClient

# The client creates its own AuthManager; keys are encrypted at rest in memory.
client = OpenRouterClient(api_key="your-api-key")
```

### Custom Secrets Management

The `SecretsManager` protocol requires a single method, `get_key(self, name: str) ->
bytearray`, which returns the secret's raw bytes for the requested name (raise
`AuthenticationError` if it is missing). The client queries it for `OPENROUTER_API_KEY` and,
optionally, `OPENROUTER_PROVISIONING_API_KEY`. Wire a custom manager via the client's
`secrets_manager` argument:

```python
from openrouter_client import OpenRouterClient
from openrouter_client.auth import SecretsManager
from openrouter_client.exceptions import AuthenticationError

class CustomSecretsManager(SecretsManager):
    """Custom secrets manager using your preferred storage."""

    def get_key(self, name: str) -> bytearray:
        # Fetch the secret for `name` (e.g. "OPENROUTER_API_KEY") from your store —
        # AWS Secrets Manager, HashiCorp Vault, etc.
        value = my_secret_store.lookup(name)
        if value is None:
            raise AuthenticationError(f"Secret not found: {name}")
        return bytearray(value.encode("utf-8"))

# Use with client
client = OpenRouterClient(secrets_manager=CustomSecretsManager())
```

## Logging and Debugging

Configure detailed logging:

```python
from openrouter_client import configure_logging
import logging

# Enable debug logging
configure_logging(level=logging.DEBUG)

# Or configure specific loggers
logging.getLogger("openrouter_client.http").setLevel(logging.DEBUG)
logging.getLogger("openrouter_client.endpoints.chat").setLevel(logging.INFO)

client = OpenRouterClient(api_key="your-api-key")

# All HTTP requests and responses will be logged
response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Context Manager Usage

Use the client as a context manager for automatic cleanup:

```python
# Automatic resource cleanup
with OpenRouterClient(api_key="your-api-key") as client:
    response = client.chat.create(
        model="anthropic/claude-3-opus",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)
# Client resources are automatically cleaned up here

# Handle exceptions within context
try:
    with OpenRouterClient(api_key="invalid-key") as client:
        response = client.chat.create(
            model="anthropic/claude-3-opus",
            messages=[{"role": "user", "content": "Hello!"}]
        )
except Exception as e:
    print(f"Error: {e}")
# Resources still cleaned up even with exceptions
```