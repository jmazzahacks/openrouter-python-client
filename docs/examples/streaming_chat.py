"""
Streaming Chat Example

This example demonstrates how to use streaming responses for real-time
chat completions.
"""

import os
import sys
from openrouter_client import OpenRouterClient

def main():
    # Initialize the client
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    
    client = OpenRouterClient(
        api_key=api_key,
        http_referer="https://your-site.com",
        x_title="Streaming Chat Example"
    )
    
    # Basic streaming example
    print("=== Basic Streaming ===")
    print("Assistant: ", end="", flush=True)
    
    stream = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=[
            {"role": "user", "content": "Write a short poem about coding."}
        ],
        stream=True,
        max_tokens=200
    )
    
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end="", flush=True)
    
    print("\n")  # New line after streaming
    
    # Streaming with word counting
    print("\n=== Streaming with Real-time Processing ===")
    print("Assistant: ", end="", flush=True)
    
    stream = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=[
            {"role": "user", "content": "Explain the benefits of renewable energy in detail."}
        ],
        stream=True,
        max_tokens=300
    )
    
    full_response = ""
    word_count = 0
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            
            # Count words in real-time
            new_words = len(content.split())
            word_count += new_words
            
            # Print content
            print(content, end="", flush=True)
            
            # Show word count every 20 words
            if word_count > 0 and word_count % 20 == 0:
                sys.stderr.write(f"\r[Words: {word_count}]")
                sys.stderr.flush()
    
    print(f"\n\nFinal word count: {len(full_response.split())}")
    
    # Streaming conversation
    print("\n=== Streaming Conversation ===")
    
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "How do I create a list in Python?"}
    ]
    
    print("User: How do I create a list in Python?")
    print("Assistant: ", end="", flush=True)
    
    stream = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=messages,
        stream=True
    )
    
    assistant_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            assistant_response += content
            print(content, end="", flush=True)
    
    print("\n")
    
    # Add assistant response to conversation
    messages.append({"role": "assistant", "content": assistant_response})
    
    # Continue conversation
    follow_up = "Can you show me how to add items to it?"
    messages.append({"role": "user", "content": follow_up})
    
    print(f"User: {follow_up}")
    print("Assistant: ", end="", flush=True)
    
    stream = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=messages,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n")

def demonstrate_error_handling():
    """Demonstrate error handling with streaming."""
    print("\n=== Streaming Error Handling ===")
    
    client = OpenRouterClient(api_key="invalid-key")
    
    try:
        stream = client.chat.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        
        for chunk in stream:
            print(chunk.choices[0].delta.content, end="", flush=True)
            
    except Exception as e:
        print(f"Error during streaming: {e}")

if __name__ == "__main__":
    main()
    # Uncomment to test error handling
    # demonstrate_error_handling()