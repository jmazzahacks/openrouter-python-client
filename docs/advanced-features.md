# Advanced Features

This guide covers advanced features and configurations of the OpenRouter Python client.

## Function Calling

The client provides comprehensive support for OpenAI-style function calling with convenient decorators and utilities.

### Using the @tool Decorator

The `@tool` decorator automatically converts Python functions into OpenRouter-compatible tools:

```python
from openrouter_client import OpenRouterClient, tool

@tool
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
    tools=[get_weather.to_dict()],
    tool_choice="auto"
)

# Handle tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        if tool_call.function.name == "get_weather":
            # Execute the function
            result = get_weather.execute(tool_call.function.arguments)
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
def handle_streaming_with_tools():
    stream = client.chat.create(
        model="anthropic/claude-3-opus",
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[get_weather.to_dict()],
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
            result = get_weather.execute(tool_call["function"]["arguments"])
            print(f"\nWeather result: {result}")
```

## Rate Limiting and Smart Retry

The client includes intelligent rate limiting powered by SmartSurge.

### Automatic Rate Limiting

Rate limiting is handled automatically based on your API key limits:

```python
client = OpenRouterClient(
    api_key="your-api-key",
    max_retries=5,  # Automatic retries on rate limits
    timeout=60.0    # Extended timeout for retries
)

# The client automatically handles rate limits
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
# Check current rate limits
rate_limits = client.calculate_rate_limits()
print(f"Requests per minute: {rate_limits['requests_per_minute']}")
print(f"Tokens per minute: {rate_limits['tokens_per_minute']}")

# Check credit balance
credits = client.credits.get()
print(f"Current balance: ${credits.data.credits}")

# Wait for rate limit reset if needed
import time
if rate_limits['requests_per_minute'] < 10:
    print("Low rate limit, waiting...")
    time.sleep(60)  # Wait for reset
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

For compatible models, use prompt caching to reduce costs:

```python
# Cache a system prompt
response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[
        {
            "role": "system", 
            "content": "You are an expert programmer...",  # Long system prompt
            "cache_control": {"type": "ephemeral"}  # Cache this message
        },
        {"role": "user", "content": "Write a Python function"}
    ]
)

# Subsequent requests with the same cached system prompt will be cheaper
response2 = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[
        {
            "role": "system", 
            "content": "You are an expert programmer...",  # Same cached prompt
            "cache_control": {"type": "ephemeral"}
        },
        {"role": "user", "content": "Write a JavaScript function"}
    ]
)
```

## Secure Key Management

### Basic Authentication Manager

```python
from openrouter_client import AuthManager

# Create auth manager with encryption
auth = AuthManager(
    api_key="your-api-key",
    encrypt_key=True  # Encrypt the key in memory
)

client = OpenRouterClient(auth_manager=auth)
```

### Custom Secrets Management

Implement custom secrets management:

```python
from openrouter_client.auth import SecretsManager

class CustomSecretsManager(SecretsManager):
    """Custom secrets manager using your preferred storage."""
    
    def get_secret(self, key: str) -> str:
        # Implement your secret retrieval logic
        # e.g., from AWS Secrets Manager, HashiCorp Vault, etc.
        pass
    
    def set_secret(self, key: str, value: str) -> None:
        # Implement your secret storage logic
        pass

# Use with client
secrets_manager = CustomSecretsManager()
auth = AuthManager(secrets_manager=secrets_manager)
client = OpenRouterClient(auth_manager=auth)
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