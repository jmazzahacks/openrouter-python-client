"""
Text Completions Example

This example demonstrates how to use the text completions endpoint
for generating text from prompts.
"""

import os
from openrouter_client import OpenRouterClient

def main():
    # Initialize the client
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    
    client = OpenRouterClient(
        api_key=api_key,
        http_referer="https://your-site.com",
        x_title="Text Completions Example"
    )
    
    # Example 1: Basic text completion
    print("=== Basic Text Completion ===")
    
    response = client.completions.create(
        model="openai/gpt-3.5-turbo-instruct",
        prompt="The benefits of renewable energy include",
        max_tokens=100,
        temperature=0.7
    )
    
    print(f"Prompt: The benefits of renewable energy include")
    print(f"Completion: {response.choices[0].text}")
    
    # Example 2: Creative writing completion
    print("\n=== Creative Writing Completion ===")
    
    creative_prompt = """Once upon a time, in a small village nestled between rolling hills and a mysterious forest, there lived a young inventor named Elena. She had always been fascinated by"""
    
    response = client.completions.create(
        model="openai/gpt-3.5-turbo-instruct",
        prompt=creative_prompt,
        max_tokens=200,
        temperature=0.9,  # Higher temperature for more creativity
        stop=["\n\n"]    # Stop at paragraph breaks
    )
    
    print(f"Prompt: {creative_prompt}")
    print(f"Completion: {response.choices[0].text}")
    
    # Example 3: Code completion
    print("\n=== Code Completion ===")
    
    code_prompt = """def calculate_fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    else:"""
    
    response = client.completions.create(
        model="openai/gpt-3.5-turbo-instruct",
        prompt=code_prompt,
        max_tokens=150,
        temperature=0.1,  # Low temperature for more deterministic code
        stop=["def ", "class ", "\n\n"]  # Stop at new function/class definitions
    )
    
    print(f"Code prompt:\n{code_prompt}")
    print(f"Completion:\n{response.choices[0].text}")
    
    # Example 4: Multiple completions
    print("\n=== Multiple Completions ===")
    
    response = client.completions.create(
        model="openai/gpt-3.5-turbo-instruct",
        prompt="Three innovative ways to reduce plastic waste:",
        max_tokens=50,
        temperature=0.8,
        n=3  # Generate 3 different completions
    )
    
    print("Prompt: Three innovative ways to reduce plastic waste:")
    for i, choice in enumerate(response.choices, 1):
        print(f"\nCompletion {i}: {choice.text}")
    
    # Example 5: Completion with custom parameters
    print("\n=== Completion with Custom Parameters ===")
    
    response = client.completions.create(
        model="openai/gpt-3.5-turbo-instruct",
        prompt="Explain quantum computing in simple terms:",
        max_tokens=120,
        temperature=0.6,
        top_p=0.9,              # Nucleus sampling
        frequency_penalty=0.2,   # Reduce repetition
        presence_penalty=0.1,    # Encourage new topics
        best_of=2               # Generate 2 and return the best
    )
    
    print("Prompt: Explain quantum computing in simple terms:")
    print(f"Completion: {response.choices[0].text}")
    
    # Print usage information
    if hasattr(response, 'usage') and response.usage:
        print(f"\nUsage:")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")

def demonstrate_streaming_completions():
    """Demonstrate streaming text completions."""
    print("\n=== Streaming Text Completions ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    prompt = "Write a short essay about the future of artificial intelligence:"
    
    print(f"Prompt: {prompt}")
    print("Streaming completion: ", end="", flush=True)
    
    stream = client.completions.create(
        model="openai/gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
        stream=True
    )
    
    full_text = ""
    for chunk in stream:
        if chunk.choices[0].text:
            text = chunk.choices[0].text
            full_text += text
            print(text, end="", flush=True)
    
    print(f"\n\nFull completion length: {len(full_text)} characters")

def demonstrate_prompt_engineering():
    """Demonstrate different prompt engineering techniques."""
    print("\n=== Prompt Engineering Techniques ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    # Technique 1: Few-shot prompting
    print("1. Few-shot prompting:")
    few_shot_prompt = """Translate English to French:
English: Hello
French: Bonjour

English: Thank you
French: Merci

English: Good morning
French: """
    
    response = client.completions.create(
        model="openai/gpt-3.5-turbo-instruct",
        prompt=few_shot_prompt,
        max_tokens=10,
        temperature=0.1
    )
    
    print(f"Input: {few_shot_prompt}")
    print(f"Output: {response.choices[0].text.strip()}")
    
    # Technique 2: Step-by-step reasoning
    print("\n2. Step-by-step reasoning:")
    reasoning_prompt = """Solve this step by step:
What is 15% of 240?

Step 1:"""
    
    response = client.completions.create(
        model="openai/gpt-3.5-turbo-instruct",
        prompt=reasoning_prompt,
        max_tokens=150,
        temperature=0.1
    )
    
    print(f"Input: {reasoning_prompt}")
    print(f"Output: {response.choices[0].text}")
    
    # Technique 3: Format specification
    print("\n3. Format specification:")
    format_prompt = """Create a JSON object for a person with the following information:
Name: John Smith
Age: 30
City: New York
Occupation: Software Engineer

JSON:"""
    
    response = client.completions.create(
        model="openai/gpt-3.5-turbo-instruct",
        prompt=format_prompt,
        max_tokens=100,
        temperature=0.1
    )
    
    print(f"Input: {format_prompt}")
    print(f"Output: {response.choices[0].text}")

def demonstrate_error_handling():
    """Demonstrate error handling with completions."""
    print("\n=== Error Handling ===")
    
    from openrouter_client.exceptions import OpenRouterError, ValidationError
    
    client = OpenRouterClient(api_key="invalid-key")
    
    try:
        response = client.completions.create(
            model="openai/gpt-3.5-turbo-instruct",
            prompt="This will fail due to invalid API key",
            max_tokens=50
        )
    except OpenRouterError as e:
        print(f"API Error: {e}")
    
    # Test with valid client but invalid parameters
    valid_client = OpenRouterClient(api_key=os.environ.get("OPENROUTER_API_KEY", "your-api-key-here"))
    
    try:
        response = valid_client.completions.create(
            model="invalid-model-name",
            prompt="This will fail due to invalid model",
            max_tokens=50
        )
    except (OpenRouterError, ValidationError) as e:
        print(f"Validation Error: {e}")

if __name__ == "__main__":
    main()
    # Uncomment to run additional examples
    # demonstrate_streaming_completions()
    # demonstrate_prompt_engineering()
    # demonstrate_error_handling()