"""
Basic Chat Completion Example

This example demonstrates the most basic usage of the OpenRouter client
for chat completions.
"""

import os
from openrouter_client import OpenRouterClient

def main():
    # Initialize the client
    # You can pass the API key directly or set OPENROUTER_API_KEY environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    
    client = OpenRouterClient(
        api_key=api_key,
        http_referer="https://your-site.com",  # Optional
        x_title="Basic Chat Example"           # Optional
    )
    
    # Simple chat completion
    print("=== Basic Chat Completion ===")
    response = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    
    print(f"Response: {response.choices[0].message.content}")
    
    # Chat with system message
    print("\n=== Chat with System Message ===")
    response = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that responds in a friendly, casual tone."},
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        max_tokens=150
    )
    
    print(f"Response: {response.choices[0].message.content}")
    
    # Multi-turn conversation
    print("\n=== Multi-turn Conversation ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's 2 + 2?"},
    ]
    
    # First exchange
    response = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=messages
    )
    
    # Add assistant's response to conversation history
    messages.append({
        "role": "assistant", 
        "content": response.choices[0].message.content
    })
    
    print(f"User: What's 2 + 2?")
    print(f"Assistant: {response.choices[0].message.content}")
    
    # Continue the conversation
    messages.append({
        "role": "user", 
        "content": "Now multiply that by 3."
    })
    
    response = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=messages
    )
    
    print(f"User: Now multiply that by 3.")
    print(f"Assistant: {response.choices[0].message.content}")
    
    # Print usage information if available
    if hasattr(response, 'usage') and response.usage:
        print(f"\nUsage:")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")

if __name__ == "__main__":
    main()