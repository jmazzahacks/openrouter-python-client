# Examples

This page provides practical examples of using the OpenRouter Python client for various tasks.

## Basic Chat Completion

Simple chat completion with a model:

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(api_key="your-api-key")

response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
```

## Streaming Chat

Real-time streaming responses:

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(api_key="your-api-key")

stream = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[{"role": "user", "content": "Tell me a long story about space exploration"}],
    stream=True
)

print("Assistant: ", end="")
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # New line at the end
```

## Function Calling with @tool Decorator

Using the convenient `@tool` decorator:

```python
from openrouter_client import OpenRouterClient, tool
import requests

@tool
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get current weather for a location.
    
    Args:
        location: The city and state/country
        unit: Temperature unit (celsius or fahrenheit)
    """
    # Mock implementation - replace with real weather API
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "Sunny",
        "humidity": 65
    }

client = OpenRouterClient(api_key="your-api-key")

response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[{"role": "user", "content": "What's the weather like in Paris?"}],
    tools=[get_weather.to_dict()],
    tool_choice="auto"
)

# Process tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        if tool_call.function.name == "get_weather":
            result = get_weather.execute(tool_call.function.arguments)
            print(f"Weather result: {result}")
else:
    print(response.choices[0].message.content)
```

## Multi-turn Conversation with Function Calling

A complete conversation with function calling:

```python
from openrouter_client import OpenRouterClient
from openrouter_client.tools import (
    build_tool_definition,
    build_parameter_schema
)
from openrouter_client.models import (
    StringParameter,
    NumberParameter,
    FunctionCall,
    ChatCompletionTool
)
import json

def search_database(query: str, category: str = "all") -> dict:
    """Mock database search function."""
    return {
        "query": query,
        "category": category,
        "results": [
            {"title": f"Result 1 for {query}", "score": 0.95},
            {"title": f"Result 2 for {query}", "score": 0.87}
        ]
    }

# Create tool definition using helper function
def search_database_func(query: str, category: str = "all") -> dict:
    """Search for information in the database.
    
    Args:
        query: Search query
        category: Search category (all, books, or articles)
    """
    pass

search_tool = build_tool_definition(search_database_func)

client = OpenRouterClient(api_key="your-api-key")

def conversation():
    messages = [
        {"role": "system", "content": "You are a helpful search assistant."}
    ]
    
    # First user message
    user_input = "Find information about machine learning"
    print(f"User: {user_input}")
    messages.append({"role": "user", "content": user_input})
    
    # Get assistant response
    response = client.chat.create(
        model="anthropic/claude-3-opus",
        messages=messages,
        tools=[search_tool]
    )
    
    assistant_message = response.choices[0].message
    messages.append(assistant_message.dict())
    
    # Handle tool calls
    if assistant_message.tool_calls:
        print("Assistant is searching...")
        
        for tool_call in assistant_message.tool_calls:
            # Execute the tool (parse arguments manually)
            import json
            args = json.loads(tool_call.function.arguments)
            result = search_database(**args)
            
            # Add tool response to conversation
            tool_response = {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            }
            messages.append(tool_response)
            
            print(f"Search results: {json.dumps(result, indent=2)}")
        
        # Get final response with tool results
        final_response = client.chat.create(
            model="anthropic/claude-3-opus",
            messages=messages
        )
        print(f"Assistant: {final_response.choices[0].message.content}")
    else:
        print(f"Assistant: {assistant_message.content}")

conversation()
```

## Text Completions

Using the completions endpoint:

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(api_key="your-api-key")

response = client.completions.create(
    model="openai/gpt-3.5-turbo-instruct",
    prompt="The benefits of renewable energy include",
    max_tokens=150,
    temperature=0.7
)

print(response.choices[0].text)
```

## Model Information and Pricing

Get information about available models:

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(api_key="your-api-key")

# List all models
models = client.models.list()
print("Available models:")
for model in models.data[:5]:  # Show first 5 models
    print(f"- {model.id}: {model.name}")

# Get specific model info
model_info = client.models.get("anthropic/claude-3-opus")
print(f"\nClaude 3 Opus details:")
print(f"Context length: {model_info.context_length}")
print(f"Pricing: {model_info.pricing}")
print(f"Description: {model_info.description}")
```

## Prompt Caching

Reduce costs with prompt caching for compatible models:

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(api_key="your-api-key")

# Create a long system prompt to cache
long_system_prompt = """You are an expert software engineer with 20 years of experience.
You specialize in Python, JavaScript, and system architecture.
You always provide detailed explanations and consider edge cases.
You write clean, maintainable code with proper error handling.
""" * 10  # Make it long enough to cache

# First request - caches the system prompt
response1 = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[
        {
            "role": "system",
            "content": long_system_prompt,
            "cache_control": {"type": "ephemeral"}  # Cache this message
        },
        {"role": "user", "content": "Write a Python function to validate emails"}
    ]
)

print("First response:")
print(response1.choices[0].message.content)

# Second request - reuses cached system prompt (cheaper)
response2 = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[
        {
            "role": "system",
            "content": long_system_prompt,  # Same cached content
            "cache_control": {"type": "ephemeral"}
        },
        {"role": "user", "content": "Write a JavaScript function to parse URLs"}
    ]
)

print("\nSecond response (using cached prompt):")
print(response2.choices[0].message.content)
```

## Credit Management

Monitor your credit usage:

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(api_key="your-api-key")

# Check credit balance
credits = client.credits.get()
print(f"Current balance: ${credits.data.credits}")
print(f"Total usage: ${credits.data.usage}")

# Calculate rate limits based on credits
rate_limits = client.calculate_rate_limits()
print(f"Requests per minute: {rate_limits['requests_per_minute']}")
print(f"Tokens per minute: {rate_limits['tokens_per_minute']}")

# Make a request and monitor usage
response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Check updated balance
updated_credits = client.credits.get()
cost = credits.data.credits - updated_credits.data.credits
print(f"Request cost: ${cost:.6f}")
```

## Context Length Management

Manage long conversations within model context limits:

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(api_key="your-api-key")

def manage_conversation_length(messages, model, max_context_usage=0.8):
    """Keep conversation within context limits."""
    context_length = client.get_context_length(model)
    max_tokens = int(context_length * max_context_usage)
    
    # Rough token estimation (4 chars per token)
    total_chars = sum(len(msg["content"]) for msg in messages)
    estimated_tokens = total_chars // 4
    
    if estimated_tokens > max_tokens:
        # Keep system message and recent messages
        system_msgs = [msg for msg in messages if msg["role"] == "system"]
        other_msgs = [msg for msg in messages if msg["role"] != "system"]
        
        # Keep last N messages that fit
        char_budget = max_tokens * 4 - sum(len(msg["content"]) for msg in system_msgs)
        
        kept_messages = []
        current_chars = 0
        
        for msg in reversed(other_msgs):
            msg_chars = len(msg["content"])
            if current_chars + msg_chars <= char_budget:
                kept_messages.insert(0, msg)
                current_chars += msg_chars
            else:
                break
        
        return system_msgs + kept_messages
    
    return messages

# Example usage
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI stands for Artificial Intelligence..."},
    # ... many more messages ...
    {"role": "user", "content": "Tell me more about machine learning."}
]

# Manage conversation length
managed_messages = manage_conversation_length(conversation, "anthropic/claude-3-opus")

response = client.chat.create(
    model="anthropic/claude-3-opus",
    messages=managed_messages
)

print(response.choices[0].message.content)
```

## Error Handling

Comprehensive error handling:

```python
from openrouter_client import OpenRouterClient
from openrouter_client.exceptions import (
    OpenRouterError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NotFoundError,
    ServerError
)
import time

client = OpenRouterClient(api_key="your-api-key")

def robust_chat_request(messages, model, max_retries=3):
    """Make a chat request with robust error handling."""
    for attempt in range(max_retries):
        try:
            response = client.chat.create(
                model=model,
                messages=messages
            )
            return response
        
        except AuthenticationError:
            print("Authentication failed. Check your API key.")
            break
        
        except RateLimitError as e:
            print(f"Rate limited. Waiting {e.retry_after} seconds...")
            time.sleep(e.retry_after)
            continue
        
        except ValidationError as e:
            print(f"Invalid request parameters: {e}")
            break
        
        except NotFoundError:
            print(f"Model {model} not found. Try a different model.")
            break
        
        except ServerError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Server error. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Server error after {max_retries} attempts: {e}")
                break
        
        except OpenRouterError as e:
            print(f"Unexpected API error: {e}")
            break
    
    return None

# Use the robust function
response = robust_chat_request(
    messages=[{"role": "user", "content": "Hello!"}],
    model="anthropic/claude-3-opus"
)

if response:
    print(response.choices[0].message.content)
else:
    print("Failed to get response after retries.")
```

## Context Manager Usage

Using the client as a context manager:

```python
from openrouter_client import OpenRouterClient

# Automatic resource cleanup
with OpenRouterClient(api_key="your-api-key") as client:
    # Make multiple requests
    response1 = client.chat.create(
        model="anthropic/claude-3-opus",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    response2 = client.models.list()
    
    credits = client.credits.get()
    
    print(f"Response: {response1.choices[0].message.content}")
    print(f"Available models: {len(response2.data)}")
    print(f"Credits: ${credits.data.credits}")

# Client resources are automatically cleaned up here
print("Client cleanup completed automatically")
```

## Async-Style Usage with Streaming

Process streaming responses efficiently:

```python
from openrouter_client import OpenRouterClient
import sys

client = OpenRouterClient(api_key="your-api-key")

def stream_with_processing():
    """Stream response and process chunks as they arrive."""
    stream = client.chat.create(
        model="anthropic/claude-3-opus",
        messages=[{"role": "user", "content": "Write a detailed explanation of quantum computing"}],
        stream=True,
        max_tokens=1000
    )
    
    full_response = ""
    chunk_count = 0
    
    print("Assistant: ", end="")
    for chunk in stream:
        chunk_count += 1
        delta = chunk.choices[0].delta
        
        if delta.content:
            content = delta.content
            full_response += content
            print(content, end="", flush=True)
            
            # Process chunks in real-time (e.g., word counting)
            if chunk_count % 10 == 0:
                word_count = len(full_response.split())
                sys.stderr.write(f"\r[Words: {word_count}]")
                sys.stderr.flush()
    
    print()  # New line
    print(f"\nStream completed. Total words: {len(full_response.split())}")
    return full_response

# Run streaming example
result = stream_with_processing()
```