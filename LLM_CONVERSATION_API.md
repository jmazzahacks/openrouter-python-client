# OpenRouter Python Client - LLM-Style Conversation API

The OpenRouter Python client includes a simplified LLM-style API with conversation support, inspired by Simon Willison's llm library. This API provides an intuitive way to manage multi-turn conversations while automatically maintaining context.

## Installation & Setup

```python
from openrouter_client import OpenRouterClient
from openrouter_client.models.llm import get_model

# Initialize the client
client = OpenRouterClient(api_key="your-api-key")

# Get a model instance
model = get_model("openai/gpt-4o-mini", client)
```

## Basic Usage

### Single Prompt (No Conversation)
```python
# Simple one-off prompt
response = model.prompt("What's the capital of France?")
print(response)  # "The capital of France is Paris."

# With system prompt
response = model.prompt(
    "Translate this to Spanish: Hello world",
    system="You are a professional translator"
)
```

## Conversation API

### Creating a Conversation
```python
# Create a conversation with optional system prompt
conversation = model.conversation(
    system="You are a helpful assistant that remembers our entire conversation"
)
```

### Multi-turn Conversation
```python
# First turn
response1 = conversation.prompt("My name is Alice and I love Python programming")
print(response1)  # "Nice to meet you, Alice! Python is a great language..."

# Second turn - the model remembers context
response2 = conversation.prompt("What's my name?")
print(response2)  # "Your name is Alice."

# Third turn - continues building context
response3 = conversation.prompt("What did I say I love?")
print(response3)  # "You said you love Python programming."
```

## Advanced Features

### Structured Output with Conversations

**Important**: The return type depends on whether you provide a schema:
- **With schema**: Returns a Python `dict` (parsed JSON)
- **Without schema**: Returns a `string`

```python
# Define a JSON schema
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "occupation": {"type": "string"},
        "hobbies": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "occupation"]
}

# Get structured data in a conversation
conversation = model.conversation()

# WITH SCHEMA - Returns a dict
person_data = conversation.prompt(
    "Tell me about Albert Einstein",
    schema=person_schema
)
print(type(person_data))  # <class 'dict'>
print(person_data)  # {'name': 'Albert Einstein', 'age': 76, 'occupation': 'Theoretical Physicist', 'hobbies': ['violin', 'sailing', 'mathematics']}

# WITHOUT SCHEMA - Returns a string
follow_up = conversation.prompt("What was his most famous equation?")
print(type(follow_up))  # <class 'str'>
print(follow_up)  # "Einstein's most famous equation is E=mc²..."

# You can mix both types in the same conversation
# The return type is determined per prompt() call, not per conversation
```

### Additional Parameters
```python
# Pass any OpenRouter/OpenAI parameters through kwargs
conversation = model.conversation(system="You are a creative writer")

response = conversation.prompt(
    "Write a haiku about coding",
    temperature=0.9,  # Higher creativity
    max_tokens=100,
    top_p=0.95
)
```

### Conversation Management
```python
conversation = model.conversation()

# Add multiple exchanges
conversation.prompt("Let's talk about space")
conversation.prompt("What's the largest planet?")
conversation.prompt("How many moons does it have?")

# Check message count
count = conversation.get_message_count()
print(f"Conversation has {count} messages")  # Includes system, user, and assistant messages

# Access full conversation history
print("Full conversation history:")
for i, message in enumerate(conversation.messages):
    role = message["role"]
    content = message["content"]
    
    # Handle content that might be a list (for user messages with attachments)
    if isinstance(content, list):
        # Extract text from the first text part
        text_content = next((part["text"] for part in content if part.get("type") == "text"), str(content))
    else:
        text_content = content
    
    print(f"{i+1}. {role.upper()}: {text_content}")

# Clear conversation history (keeps system prompt if present)
conversation.clear()
print(f"After clear: {conversation.get_message_count()} messages")
```

## Key Points

1. **Automatic Context Management**: The conversation object automatically maintains the full message history, sending it with each request
2. **Stateful Conversations**: Each conversation object is independent - you can have multiple conversations running simultaneously
3. **Return Type Behavior**: 
   - When `schema` parameter is provided → Returns a Python `dict` (parsed JSON)
   - When no `schema` parameter → Returns a `string`
   - The return type is determined per `prompt()` call, not per conversation
4. **Conversation History Access**: You can access the full conversation history via `conversation.messages` - this contains the complete OpenAI chat format with system, user, and assistant messages
5. **Parameter Passthrough**: All standard OpenRouter/OpenAI parameters can be passed via kwargs
6. **System Prompt Persistence**: System prompts are maintained even when clearing conversation history

## Example Use Cases

```python
# Customer support bot
support_bot = model.conversation(
    system="You are a helpful customer support agent for an e-commerce platform"
)
support_bot.prompt("I need help with my order #12345")
support_bot.prompt("It hasn't arrived yet")
support_bot.prompt("Can you check the status?")

# Code review assistant
code_reviewer = model.conversation(
    system="You are an expert code reviewer focusing on Python best practices"
)
code_reviewer.prompt("Review this function: def add(a,b): return a+b")
code_reviewer.prompt("How can I make it more robust?")

# Data extraction with structured output
extractor = model.conversation()
schema = {"type": "object", "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}}}
result = extractor.prompt("I absolutely love this product!", schema=schema)
print(result["sentiment"])  # "positive"
```

The conversation API makes it easy to build applications that need to maintain context across multiple interactions, from chatbots to complex multi-step workflows.