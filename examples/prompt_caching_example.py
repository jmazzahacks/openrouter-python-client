"""Example script demonstrating the prompt caching functionality.

This example shows how to use prompt caching with OpenRouter for both:
1. OpenAI models (automatic caching for prompts > 1024 tokens)
2. Anthropic models (using explicit cache_control markers)

Usage statistics are enabled to show the cache tokens and cache discount.
"""

from openrouter_client import OpenRouterClient
from openrouter_client.models.core import TextContent, CacheControl
from rich import print
import os

# Initialize the client
api_key = os.environ.get("OPENROUTER_API_KEY", "YOUR_API_KEY")
client = OpenRouterClient(api_key=api_key)

# Generate some long text to demonstrate caching
def generate_long_text(length=2000):
    """Generate a long text of approximately the requested length."""
    base_text = "This is a sample text to demonstrate caching. " * 40
    return base_text[:length]

# Example 1: Using OpenAI with automatic caching (prompts > 1024 tokens)
def openai_caching_example():
    """Demonstrate automatic caching with OpenAI models."""
    print("\n[bold green]Example 1: OpenAI Automatic Caching[/bold green]")
    long_text = generate_long_text()
    
    # First request - will cache the long prompt
    print("\n[yellow]First request - Cache will be initialized[/yellow]")
    response1 = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Here is a long document: {long_text}\n\nSummarize this document in one sentence."}
        ],
        include={"usage": True}  # Include usage statistics to see caching metrics
    )
    
    # Print usage information from the first request
    usage1 = response1.usage
    print(f"\nUsage for first request:")
    print(f"  - Prompt tokens: {usage1.prompt_tokens}")
    print(f"  - Completion tokens: {usage1.completion_tokens}")
    print(f"  - Total tokens: {usage1.total_tokens}")
    print(f"  - Cached tokens: {usage1.cached_tokens if hasattr(usage1, 'cached_tokens') else 'None'}")
    print(f"  - Cache discount: {usage1.cache_discount if hasattr(usage1, 'cache_discount') else 'None'}")
    
    # Second request with the same prompt - should use the cache
    print("\n[yellow]Second request - Should use cache[/yellow]")
    response2 = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Here is a long document: {long_text}\n\nSummarize this document in one sentence."}
        ],
        include={"usage": True}  # Include usage statistics to see caching metrics
    )
    
    # Print usage information from the second request
    usage2 = response2.usage
    print(f"\nUsage for second request:")
    print(f"  - Prompt tokens: {usage2.prompt_tokens}")
    print(f"  - Completion tokens: {usage2.completion_tokens}")
    print(f"  - Total tokens: {usage2.total_tokens}")
    print(f"  - Cached tokens: {usage2.cached_tokens if hasattr(usage2, 'cached_tokens') else 'None'}")
    print(f"  - Cache discount: {usage2.cache_discount if hasattr(usage2, 'cache_discount') else 'None'}")

# Example 2: Using Anthropic Claude with explicit cache_control markers
def anthropic_caching_example():
    """Demonstrate explicit caching with Anthropic Claude models."""
    print("\n[bold green]Example 2: Anthropic Claude Explicit Caching[/bold green]")
    long_text = generate_long_text()
    
    # First request - will cache the marked content
    print("\n[yellow]First request - Cache will be initialized[/yellow]")
    response1 = client.chat.completions.create(
        model="anthropic/claude-3-opus-20240229",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is a long document:"},
                    # Mark this part for caching
                    {"type": "text", "text": long_text, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "Summarize this document in one sentence."}
                ]
            }
        ],
        include={"usage": True}  # Include usage statistics to see caching metrics
    )
    
    # Print usage information from the first request
    usage1 = response1.usage
    print(f"\nUsage for first request:")
    print(f"  - Prompt tokens: {usage1.prompt_tokens}")
    print(f"  - Completion tokens: {usage1.completion_tokens}")
    print(f"  - Total tokens: {usage1.total_tokens}")
    print(f"  - Cached tokens: {usage1.cached_tokens if hasattr(usage1, 'cached_tokens') else 'None'}")
    print(f"  - Cache discount: {usage1.cache_discount if hasattr(usage1, 'cache_discount') else 'None'}")
    
    # Second request with the same prompt - should use the cache
    print("\n[yellow]Second request - Should use cache[/yellow]")
    response2 = client.chat.completions.create(
        model="anthropic/claude-3-opus-20240229",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is a long document:"},
                    # Mark this part for caching (same content as before)
                    {"type": "text", "text": long_text, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "Summarize this document in one sentence."}
                ]
            }
        ],
        include={"usage": True}  # Include usage statistics to see caching metrics
    )
    
    # Print usage information from the second request
    usage2 = response2.usage
    print(f"\nUsage for second request:")
    print(f"  - Prompt tokens: {usage2.prompt_tokens}")
    print(f"  - Completion tokens: {usage2.completion_tokens}")
    print(f"  - Total tokens: {usage2.total_tokens}")
    print(f"  - Cached tokens: {usage2.cached_tokens if hasattr(usage2, 'cached_tokens') else 'None'}")
    print(f"  - Cache discount: {usage2.cache_discount if hasattr(usage2, 'cache_discount') else 'None'}")

# Run the examples
if __name__ == "__main__":
    try:
        # Example 1: OpenAI
        openai_caching_example()
        
        # Example 2: Anthropic Claude
        anthropic_caching_example()
    except Exception as e:
        print(f"\n[bold red]Error: {e}[/bold red]")