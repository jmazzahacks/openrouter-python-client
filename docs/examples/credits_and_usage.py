"""
Credits and Usage Management Example

This example demonstrates how to monitor credits, usage,
and manage API key information.
"""

import os
import time
from openrouter_client import OpenRouterClient

def main():
    # Initialize the client
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    
    client = OpenRouterClient(
        api_key=api_key,
        http_referer="https://your-site.com",
        x_title="Credits and Usage Example"
    )
    
    # Example 1: Check credit balance and usage
    print("=== Credit Balance and Usage ===")
    
    try:
        credits_response = client.credits.get()
        
        print("Credit Information:")
        if hasattr(credits_response, 'data') and credits_response.data:
            data = credits_response.data
            if hasattr(data, 'credits'):
                print(f"  Current Balance: ${data.credits:.6f}")
            if hasattr(data, 'usage'):
                print(f"  Total Usage: ${data.usage:.6f}")
            if hasattr(data, 'limit'):
                print(f"  Spending Limit: ${data.limit:.6f}")
            
            # Calculate remaining credits
            if hasattr(data, 'credits') and hasattr(data, 'usage'):
                remaining = data.credits - data.usage
                print(f"  Remaining: ${remaining:.6f}")
                
                # Estimate requests possible
                avg_cost_per_request = 0.001  # Rough estimate
                estimated_requests = remaining / avg_cost_per_request
                print(f"  Estimated requests remaining: ~{estimated_requests:,.0f}")
        else:
            print("  No credit data available")
            
    except Exception as e:
        print(f"Error getting credits: {e}")
    
    # Example 2: API Key information
    print("\n=== API Key Information ===")
    
    try:
        keys_response = client.keys.get()
        
        print("API Key Information:")
        if hasattr(keys_response, 'data') and keys_response.data:
            data = keys_response.data
            if hasattr(data, 'label'):
                print(f"  Label: {data.label}")
            if hasattr(data, 'usage'):
                print(f"  Usage: ${data.usage:.6f}")
            if hasattr(data, 'limit'):
                print(f"  Limit: ${data.limit:.6f}")
            if hasattr(data, 'is_free_tier'):
                print(f"  Free Tier: {data.is_free_tier}")
            if hasattr(data, 'rate_limit'):
                print(f"  Rate Limit: {data.rate_limit}")
        else:
            print("  No key data available")
            
    except Exception as e:
        print(f"Error getting key info: {e}")
    
    # Example 3: Rate limit calculation
    print("\n=== Rate Limit Management ===")
    
    try:
        rate_limits = client.calculate_rate_limits()
        
        print("Calculated Rate Limits:")
        for key, value in rate_limits.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
            
    except Exception as e:
        print(f"Error calculating rate limits: {e}")

def demonstrate_usage_monitoring():
    """Demonstrate monitoring usage during API calls."""
    print("\n=== Usage Monitoring During API Calls ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    # Get initial credit balance
    try:
        initial_credits = client.credits.get()
        initial_balance = initial_credits.data.credits if hasattr(initial_credits.data, 'credits') else 0
        print(f"Initial balance: ${initial_balance:.6f}")
    except Exception as e:
        print(f"Could not get initial balance: {e}")
        initial_balance = 0
    
    # Make a few API calls and monitor usage
    test_prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing briefly.",
        "What are the benefits of renewable energy?"
    ]
    
    total_cost = 0
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nRequest {i}: {prompt}")
        
        try:
            response = client.chat.create(
                model="anthropic/claude-3-haiku",  # Use a cheaper model for testing
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            
            print(f"Response: {response.choices[0].message.content[:100]}...")
            
            # Check usage information in response
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                print(f"Token usage - Prompt: {usage.prompt_tokens}, "
                      f"Completion: {usage.completion_tokens}, "
                      f"Total: {usage.total_tokens}")
                
                # Estimate cost (rough calculation)
                estimated_cost = (usage.prompt_tokens + usage.completion_tokens) * 0.00001
                total_cost += estimated_cost
                print(f"Estimated cost: ${estimated_cost:.6f}")
            
            # Small delay between requests
            time.sleep(1)
            
        except Exception as e:
            print(f"Error with request {i}: {e}")
    
    print(f"\nTotal estimated cost for {len(test_prompts)} requests: ${total_cost:.6f}")
    
    # Get final credit balance
    try:
        final_credits = client.credits.get()
        final_balance = final_credits.data.credits if hasattr(final_credits.data, 'credits') else 0
        actual_cost = initial_balance - final_balance
        print(f"Final balance: ${final_balance:.6f}")
        print(f"Actual cost: ${actual_cost:.6f}")
    except Exception as e:
        print(f"Could not get final balance: {e}")

def demonstrate_budget_management():
    """Demonstrate budget management and alerts."""
    print("\n=== Budget Management ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    def check_budget_status(warning_threshold=0.80, critical_threshold=0.90):
        """Check budget status and return alerts."""
        try:
            credits_response = client.credits.get()
            data = credits_response.data
            
            if not (hasattr(data, 'credits') and hasattr(data, 'usage') and hasattr(data, 'limit')):
                return "Budget information not available"
            
            balance = data.credits
            usage = data.usage
            limit = data.limit
            
            if limit > 0:
                usage_percentage = usage / limit
                
                print(f"Budget Status:")
                print(f"  Limit: ${limit:.2f}")
                print(f"  Used: ${usage:.6f} ({usage_percentage:.1%})")
                print(f"  Remaining: ${limit - usage:.6f}")
                
                if usage_percentage >= critical_threshold:
                    return f"🚨 CRITICAL: {usage_percentage:.1%} of budget used!"
                elif usage_percentage >= warning_threshold:
                    return f"⚠️  WARNING: {usage_percentage:.1%} of budget used"
                else:
                    return f"✅ Budget OK: {usage_percentage:.1%} used"
            else:
                return "No spending limit set"
                
        except Exception as e:
            return f"Error checking budget: {e}"
    
    # Check current budget status
    status = check_budget_status()
    print(status)
    
    # Demonstrate usage projection
    def project_usage(daily_requests=100, avg_tokens_per_request=500):
        """Project future usage based on current patterns."""
        try:
            # Rough cost estimation
            cost_per_1k_tokens = 0.0005  # Approximate average
            tokens_per_day = daily_requests * avg_tokens_per_request
            cost_per_day = (tokens_per_day / 1000) * cost_per_1k_tokens
            
            print(f"\nUsage Projection:")
            print(f"  Daily requests: {daily_requests}")
            print(f"  Avg tokens per request: {avg_tokens_per_request}")
            print(f"  Estimated daily cost: ${cost_per_day:.4f}")
            print(f"  Estimated weekly cost: ${cost_per_day * 7:.4f}")
            print(f"  Estimated monthly cost: ${cost_per_day * 30:.4f}")
            
            # Check against current balance
            credits_response = client.credits.get()
            if hasattr(credits_response.data, 'credits'):
                balance = credits_response.data.credits
                days_remaining = balance / cost_per_day if cost_per_day > 0 else float('inf')
                print(f"  Days remaining at current rate: {days_remaining:.1f}")
                
        except Exception as e:
            print(f"Error projecting usage: {e}")
    
    project_usage()

def demonstrate_cost_optimization():
    """Demonstrate cost optimization strategies."""
    print("\n=== Cost Optimization Strategies ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    # Strategy 1: Model selection for cost efficiency
    print("1. Cost-Efficient Model Selection:")
    
    try:
        models_response = client.models.list()
        
        # Find cheapest models that are still capable
        cheap_capable_models = []
        
        for model in models_response.data:
            if hasattr(model, 'pricing') and model.pricing and hasattr(model.pricing, 'prompt'):
                try:
                    price = float(model.pricing.prompt)
                    if price < 0.001 and any(keyword in model.id.lower() for keyword in ['claude', 'gpt', 'llama']):
                        cheap_capable_models.append((model.id, price))
                except (ValueError, TypeError):
                    continue
        
        cheap_capable_models.sort(key=lambda x: x[1])
        
        print("  Cheapest capable models:")
        for model_id, price in cheap_capable_models[:5]:
            print(f"    {model_id}: ${price}/1K tokens")
    
    except Exception as e:
        print(f"  Error analyzing models: {e}")
    
    # Strategy 2: Token optimization
    print("\n2. Token Optimization Tips:")
    print("  - Use shorter, more specific prompts")
    print("  - Set appropriate max_tokens limits") 
    print("  - Use stop sequences to avoid over-generation")
    print("  - Consider prompt caching for repeated content")
    print("  - Use cheaper models for simple tasks")
    
    # Strategy 3: Batch processing
    print("\n3. Batch Processing Example:")
    
    # Example of processing multiple items efficiently
    items_to_process = [
        "Summarize: AI is transforming healthcare...",
        "Summarize: Renewable energy adoption is growing...",
        "Summarize: Remote work trends are changing..."
    ]
    
    # Inefficient: Multiple separate requests
    print("  Inefficient approach (separate requests):")
    separate_cost = len(items_to_process) * 0.001  # Rough estimate
    print(f"    Estimated cost: ${separate_cost:.4f}")
    
    # Efficient: Batch processing
    print("  Efficient approach (batch processing):")
    batch_prompt = "Please provide brief summaries for each of the following:\n\n"
    for i, item in enumerate(items_to_process, 1):
        batch_prompt += f"{i}. {item}\n"
    
    try:
        batch_response = client.chat.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": batch_prompt}],
            max_tokens=300
        )
        
        if hasattr(batch_response, 'usage') and batch_response.usage:
            usage = batch_response.usage
            estimated_cost = (usage.total_tokens / 1000) * 0.0005
            print(f"    Actual tokens used: {usage.total_tokens}")
            print(f"    Estimated cost: ${estimated_cost:.6f}")
            print(f"    Savings: ~{((separate_cost - estimated_cost) / separate_cost * 100):.1f}%")
        
        print(f"    Batch result: {batch_response.choices[0].message.content[:200]}...")
        
    except Exception as e:
        print(f"    Error with batch processing: {e}")

if __name__ == "__main__":
    main()
    # Uncomment to run additional examples
    # demonstrate_usage_monitoring()
    # demonstrate_budget_management()
    # demonstrate_cost_optimization()