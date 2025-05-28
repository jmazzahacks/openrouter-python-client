"""
Prompt Caching Example

This example demonstrates how to use prompt caching with OpenRouter
to reduce costs when reusing long prompts or context.
"""

import os
from openrouter_client import OpenRouterClient

def generate_long_document(topic: str, length: int = 2000) -> str:
    """Generate a long document for caching demonstration."""
    base_content = f"""
    This is a comprehensive analysis of {topic}. The field has evolved significantly 
    over the past decade, with numerous breakthrough discoveries and technological 
    advancements that have transformed our understanding and capabilities.
    
    Historical Context:
    The origins of {topic} can be traced back to early theoretical work in the 
    mid-20th century. Pioneering researchers laid the groundwork for what would 
    become one of the most influential areas of study in modern science and technology.
    
    Current State:
    Today, {topic} encompasses a wide range of applications and methodologies. 
    From basic research to practical implementations, the field continues to 
    expand rapidly with new discoveries being made regularly.
    
    Technical Considerations:
    The implementation of {topic} requires careful consideration of multiple 
    factors including computational complexity, resource requirements, and 
    scalability challenges. Modern approaches have addressed many of these 
    concerns through innovative solutions.
    
    Future Directions:
    Looking ahead, {topic} promises to play an increasingly important role 
    in addressing complex global challenges. Emerging trends suggest significant 
    opportunities for breakthrough applications in the coming years.
    """
    
    # Repeat content to reach desired length
    repeated_content = base_content
    while len(repeated_content) < length:
        repeated_content += f"\n\nAdditional insights into {topic}: " + base_content
    
    return repeated_content[:length]

def main():
    # Initialize the client
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    
    client = OpenRouterClient(
        api_key=api_key,
        http_referer="https://your-site.com",
        x_title="Prompt Caching Example"
    )
    
    # Example 1: Basic prompt caching with Anthropic Claude
    print("=== Anthropic Claude Prompt Caching ===")
    
    # Generate a long document to cache
    long_document = generate_long_document("artificial intelligence", 3000)
    
    # First request - this will establish the cache
    print("First request (establishing cache)...")
    
    response1 = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyze the following document:"},
                    {
                        "type": "text", 
                        "text": long_document,
                        "cache_control": {"type": "ephemeral"}  # Mark this for caching
                    },
                    {"type": "text", "text": "Provide a brief summary of the key points."}
                ]
            }
        ]
    )
    
    print(f"Response 1: {response1.choices[0].message.content[:200]}...")
    
    if hasattr(response1, 'usage') and response1.usage:
        print(f"Usage - Prompt: {response1.usage.prompt_tokens}, "
              f"Completion: {response1.usage.completion_tokens}")
    
    # Second request - this should use the cached content
    print("\nSecond request (using cache)...")
    
    response2 = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyze the following document:"},
                    {
                        "type": "text", 
                        "text": long_document,  # Same content as before
                        "cache_control": {"type": "ephemeral"}
                    },
                    {"type": "text", "text": "What are the main challenges mentioned?"}
                ]
            }
        ]
    )
    
    print(f"Response 2: {response2.choices[0].message.content[:200]}...")
    
    if hasattr(response2, 'usage') and response2.usage:
        print(f"Usage - Prompt: {response2.usage.prompt_tokens}, "
              f"Completion: {response2.usage.completion_tokens}")
    
    # Example 2: Caching system prompts for consistent behavior
    print("\n=== Caching System Prompts ===")
    
    system_prompt = """
    You are an expert technical writer and analyst with deep knowledge across 
    multiple domains including technology, science, and engineering. Your responses 
    should be:
    
    1. Technically accurate and precise
    2. Well-structured with clear sections
    3. Include relevant examples when appropriate
    4. Consider multiple perspectives on complex topics
    5. Cite limitations or uncertainties when they exist
    
    When analyzing documents, focus on:
    - Key technical concepts and innovations
    - Practical applications and use cases
    - Potential challenges or limitations
    - Future implications and trends
    - Connections to related fields or technologies
    
    Your writing style should be professional yet accessible, avoiding unnecessary 
    jargon while maintaining technical accuracy. Always strive to provide value 
    through insightful analysis rather than mere summarization.
    """ * 3  # Make it longer to demonstrate caching
    
    def ask_question_with_cached_system(question: str, request_num: int):
        """Helper function to ask questions with cached system prompt."""
        print(f"\nRequest {request_num}: {question}")
        
        response = client.chat.create(
            model="anthropic/claude-3-haiku",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                },
                {"role": "user", "content": question}
            ]
        )
        
        print(f"Response: {response.choices[0].message.content[:150]}...")
        
        if hasattr(response, 'usage') and response.usage:
            print(f"Token usage - Prompt: {response.usage.prompt_tokens}, "
                  f"Completion: {response.usage.completion_tokens}")
    
    # Multiple requests with the same cached system prompt
    ask_question_with_cached_system(
        "Explain the benefits of using microservices architecture.", 1
    )
    
    ask_question_with_cached_system(
        "What are the security considerations for cloud computing?", 2
    )
    
    ask_question_with_cached_system(
        "How does machine learning impact software development?", 3
    )

def demonstrate_conversation_caching():
    """Demonstrate caching in multi-turn conversations."""
    print("\n=== Conversation Context Caching ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    # Long context that we want to cache
    conversation_context = generate_long_document("software engineering best practices", 2500)
    
    # Start conversation with cached context
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a software engineering consultant."},
                {
                    "type": "text",
                    "text": f"Here is the background context:\n\n{conversation_context}",
                    "cache_control": {"type": "ephemeral"}
                },
                {"type": "text", "text": "Use this context to answer questions about software engineering."}
            ]
        }
    ]
    
    # Series of questions that all reference the cached context
    questions = [
        "What are the key principles mentioned in the context?",
        "How do these principles apply to agile development?", 
        "What challenges are identified and how can they be addressed?",
        "What future trends are discussed?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        
        # Add user question to conversation
        conversation_messages = messages + [{"role": "user", "content": question}]
        
        response = client.chat.create(
            model="anthropic/claude-3-haiku",
            messages=conversation_messages
        )
        
        print(f"Answer: {response.choices[0].message.content[:200]}...")
        
        # Add response to conversation history (but keep system context cached)
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": response.choices[0].message.content})

def demonstrate_cost_comparison():
    """Demonstrate cost savings with caching."""
    print("\n=== Cost Comparison: With vs Without Caching ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    large_context = generate_long_document("data science methodologies", 4000)
    
    # Request without caching
    print("Request WITHOUT caching...")
    response_no_cache = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=[
            {
                "role": "user",
                "content": f"Context: {large_context}\n\nQuestion: Summarize the main points."
            }
        ]
    )
    
    if hasattr(response_no_cache, 'usage') and response_no_cache.usage:
        print(f"No cache - Prompt tokens: {response_no_cache.usage.prompt_tokens}")
    
    # Request with caching
    print("\nRequest WITH caching...")
    response_with_cache = client.chat.create(
        model="anthropic/claude-3-haiku", 
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Context:"},
                    {
                        "type": "text",
                        "text": large_context,
                        "cache_control": {"type": "ephemeral"}
                    },
                    {"type": "text", "text": "Question: Summarize the main points."}
                ]
            }
        ]
    )
    
    if hasattr(response_with_cache, 'usage') and response_with_cache.usage:
        print(f"With cache - Prompt tokens: {response_with_cache.usage.prompt_tokens}")
    
    print("\nNote: Subsequent requests with the same cached content will show cost savings.")

if __name__ == "__main__":
    main()
    # Uncomment to run additional examples
    # demonstrate_conversation_caching()
    # demonstrate_cost_comparison()