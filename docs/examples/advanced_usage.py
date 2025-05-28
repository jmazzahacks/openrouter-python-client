"""
Advanced Usage Example

This example demonstrates advanced patterns and techniques
for using the OpenRouter Python client effectively.
"""

import os
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Generator
from contextlib import contextmanager

from openrouter_client import OpenRouterClient, tool

def main():
    # Initialize the client
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    
    client = OpenRouterClient(
        api_key=api_key,
        http_referer="https://your-site.com",
        x_title="Advanced Usage Example"
    )
    
    # Example 1: Context manager usage
    print("=== Context Manager Usage ===")
    
    with OpenRouterClient(api_key=api_key) as client_ctx:
        response = client_ctx.chat.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": "Context manager test"}]
        )
        print(f"Response: {response.choices[0].message.content}")
    # Client resources automatically cleaned up here
    
    # Example 2: Conversation management
    print("\n=== Advanced Conversation Management ===")
    
    conversation_manager = ConversationManager(client)
    
    # Start a conversation
    conv_id = conversation_manager.start_conversation(
        system_prompt="You are a helpful coding assistant."
    )
    
    # Add messages to the conversation
    response1 = conversation_manager.add_message(
        conv_id, 
        "How do I create a list in Python?"
    )
    print(f"Response 1: {response1[:100]}...")
    
    response2 = conversation_manager.add_message(
        conv_id,
        "Can you show me how to add items to it?"
    )
    print(f"Response 2: {response2[:100]}...")
    
    # Get conversation history
    history = conversation_manager.get_history(conv_id)
    print(f"Conversation has {len(history)} messages")

def demonstrate_concurrent_requests():
    """Demonstrate handling multiple concurrent requests."""
    print("\n=== Concurrent Request Processing ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    
    # Create multiple clients for concurrent use
    def create_client():
        return OpenRouterClient(api_key=api_key)
    
    # Tasks to process concurrently
    tasks = [
        "Explain machine learning in simple terms.",
        "What are the benefits of cloud computing?",
        "How does blockchain technology work?",
        "Describe the principles of good software design.",
        "What is the difference between AI and ML?"
    ]
    
    def process_task(task_id: int, prompt: str) -> Dict[str, Any]:
        """Process a single task."""
        client = create_client()
        try:
            response = client.chat.create(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            return {
                "task_id": task_id,
                "prompt": prompt,
                "response": response.choices[0].message.content,
                "success": True
            }
        except Exception as e:
            return {
                "task_id": task_id,
                "prompt": prompt,
                "error": str(e),
                "success": False
            }
        finally:
            client.close()
    
    # Process tasks concurrently
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_task, i, task): i 
            for i, task in enumerate(tasks)
        }
        
        # Collect results as they complete
        results = []
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)
            
            if result["success"]:
                print(f"✅ Task {result['task_id']}: {result['response'][:80]}...")
            else:
                print(f"❌ Task {result['task_id']} failed: {result['error']}")
    
    print(f"\nProcessed {len(results)} tasks concurrently")

def demonstrate_streaming_with_processing():
    """Demonstrate advanced streaming with real-time processing."""
    print("\n=== Advanced Streaming with Processing ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    class StreamProcessor:
        def __init__(self):
            self.word_count = 0
            self.sentence_count = 0
            self.accumulated_text = ""
            self.keywords = ["AI", "machine learning", "artificial intelligence", "technology"]
            self.keyword_mentions = {kw: 0 for kw in self.keywords}
        
        def process_chunk(self, chunk_text: str) -> Dict[str, Any]:
            """Process a streaming chunk and return analysis."""
            self.accumulated_text += chunk_text
            
            # Count words and sentences
            new_words = len(chunk_text.split())
            self.word_count += new_words
            
            new_sentences = chunk_text.count('.') + chunk_text.count('!') + chunk_text.count('?')
            self.sentence_count += new_sentences
            
            # Check for keywords
            for keyword in self.keywords:
                if keyword.lower() in chunk_text.lower():
                    self.keyword_mentions[keyword] += chunk_text.lower().count(keyword.lower())
            
            return {
                "word_count": self.word_count,
                "sentence_count": self.sentence_count,
                "keyword_mentions": self.keyword_mentions.copy()
            }
        
        def get_final_analysis(self) -> Dict[str, Any]:
            """Get final analysis of the complete text."""
            avg_words_per_sentence = self.word_count / max(self.sentence_count, 1)
            
            return {
                "total_words": self.word_count,
                "total_sentences": self.sentence_count,
                "avg_words_per_sentence": round(avg_words_per_sentence, 2),
                "total_characters": len(self.accumulated_text),
                "keyword_mentions": self.keyword_mentions,
                "full_text": self.accumulated_text
            }
    
    # Start streaming with processing
    processor = StreamProcessor()
    
    print("Streaming response with real-time analysis...")
    print("Response: ", end="", flush=True)
    
    stream = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=[{
            "role": "user", 
            "content": "Write a detailed explanation of artificial intelligence and machine learning technologies."
        }],
        stream=True,
        max_tokens=400
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            chunk_text = chunk.choices[0].delta.content
            print(chunk_text, end="", flush=True)
            
            # Process chunk
            analysis = processor.process_chunk(chunk_text)
            
            # Print analysis every 50 words
            if analysis["word_count"] % 50 == 0 and analysis["word_count"] > 0:
                print(f"\n[Analysis: {analysis['word_count']} words, {analysis['sentence_count']} sentences]", end="")
    
    print("\n")
    
    # Final analysis
    final_analysis = processor.get_final_analysis()
    print("\nFinal Analysis:")
    for key, value in final_analysis.items():
        if key != "full_text":
            print(f"  {key.replace('_', ' ').title()}: {value}")

class ConversationManager:
    """Advanced conversation management with context tracking."""
    
    def __init__(self, client: OpenRouterClient):
        self.client = client
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self._conversation_counter = 0
    
    def start_conversation(self, system_prompt: str = None, model: str = "anthropic/claude-3-haiku") -> str:
        """Start a new conversation."""
        conv_id = f"conv_{self._conversation_counter}"
        self._conversation_counter += 1
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        self.conversations[conv_id] = {
            "messages": messages,
            "model": model,
            "created_at": time.time(),
            "message_count": 0
        }
        
        return conv_id
    
    def add_message(self, conv_id: str, content: str, role: str = "user") -> str:
        """Add a message to the conversation and get response."""
        if conv_id not in self.conversations:
            raise ValueError(f"Conversation {conv_id} not found")
        
        conv = self.conversations[conv_id]
        
        # Add user message
        conv["messages"].append({"role": role, "content": content})
        conv["message_count"] += 1
        
        # Get response if user message
        if role == "user":
            try:
                response = self.client.chat.create(
                    model=conv["model"],
                    messages=conv["messages"]
                )
                
                assistant_response = response.choices[0].message.content
                conv["messages"].append({"role": "assistant", "content": assistant_response})
                conv["message_count"] += 1
                
                return assistant_response
                
            except Exception as e:
                # Add error to conversation for debugging
                error_msg = f"Error: {str(e)}"
                conv["messages"].append({"role": "system", "content": error_msg})
                return error_msg
        
        return content
    
    def get_history(self, conv_id: str) -> List[Dict[str, str]]:
        """Get conversation history."""
        if conv_id not in self.conversations:
            return []
        return self.conversations[conv_id]["messages"].copy()
    
    def summarize_conversation(self, conv_id: str) -> str:
        """Generate a summary of the conversation."""
        if conv_id not in self.conversations:
            return "Conversation not found"
        
        conv = self.conversations[conv_id]
        
        # Create summary prompt
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conv["messages"] 
            if msg["role"] in ["user", "assistant"]
        ])
        
        summary_prompt = f"Please provide a brief summary of this conversation:\n\n{history_text}"
        
        try:
            response = self.client.chat.create(
                model=conv["model"],
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Could not generate summary: {e}"

def demonstrate_custom_client_configuration():
    """Demonstrate advanced client configuration."""
    print("\n=== Custom Client Configuration ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    
    # Custom configuration
    custom_client = OpenRouterClient(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=60.0,           # Extended timeout
        max_retries=5,          # More retries
        http_referer="https://advanced-example.com",
        x_title="Advanced OpenRouter Client"
    )
    
    print("Custom client configuration:")
    print(f"  Base URL: {custom_client.http_manager.base_url}")
    print(f"  Timeout: {custom_client.http_manager.timeout}")
    print(f"  Max Retries: {custom_client.http_manager.max_retries}")
    
    # Test custom client
    try:
        response = custom_client.chat.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": "Test custom configuration"}]
        )
        print(f"  Response: {response.choices[0].message.content[:100]}...")
    except Exception as e:
        print(f"  Error: {e}")

def demonstrate_model_fallback_strategy():
    """Demonstrate automatic model fallback strategy."""
    print("\n=== Model Fallback Strategy ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    class ModelFallbackClient:
        def __init__(self, client: OpenRouterClient):
            self.client = client
            self.model_preferences = [
                "anthropic/claude-3-opus",    # First choice (most capable)
                "anthropic/claude-3-sonnet",  # Second choice
                "anthropic/claude-3-haiku",   # Third choice (fastest/cheapest)
                "openai/gpt-4-turbo",         # Fourth choice
                "openai/gpt-3.5-turbo"        # Last resort
            ]
        
        def chat_with_fallback(self, messages: List[Dict], **kwargs) -> Any:
            """Try models in order of preference until one works."""
            last_error = None
            
            for model in self.model_preferences:
                try:
                    print(f"  Trying model: {model}")
                    response = self.client.chat.create(
                        model=model,
                        messages=messages,
                        **kwargs
                    )
                    print(f"  ✅ Success with {model}")
                    return response
                    
                except Exception as e:
                    print(f"  ❌ Failed with {model}: {e}")
                    last_error = e
                    continue
            
            raise Exception(f"All models failed. Last error: {last_error}")
    
    # Test fallback strategy
    fallback_client = ModelFallbackClient(client)
    
    try:
        response = fallback_client.chat_with_fallback(
            messages=[{"role": "user", "content": "Explain quantum computing"}],
            max_tokens=150
        )
        print(f"Final response: {response.choices[0].message.content[:100]}...")
    except Exception as e:
        print(f"All models failed: {e}")

import time

if __name__ == "__main__":
    main()
    # Uncomment to run specific advanced examples
    # demonstrate_concurrent_requests()
    # demonstrate_streaming_with_processing()  
    # demonstrate_custom_client_configuration()
    # demonstrate_model_fallback_strategy()