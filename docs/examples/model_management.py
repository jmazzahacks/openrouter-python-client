"""
Model Management Example

This example demonstrates how to work with model information,
pricing, and context length management.
"""

import os
from openrouter_client import OpenRouterClient

def main():
    # Initialize the client
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    
    client = OpenRouterClient(
        api_key=api_key,
        http_referer="https://your-site.com",
        x_title="Model Management Example"
    )
    
    # Example 1: List all available models
    print("=== Available Models ===")
    
    models_response = client.models.list()
    
    print(f"Total models available: {len(models_response.data)}")
    print("\nFirst 10 models:")
    
    for i, model in enumerate(models_response.data[:10]):
        print(f"{i+1:2d}. {model.id}")
        print(f"    Name: {model.name}")
        if model.description:
            print(f"    Description: {model.description[:100]}...")
        if hasattr(model, 'context_length') and model.context_length:
            print(f"    Context Length: {model.context_length:,} tokens")
        print()
    
    # Example 2: Get specific model information
    print("=== Specific Model Information ===")
    
    model_id = "anthropic/claude-3-haiku"
    
    try:
        model_info = client.models.get(model_id)
        
        print(f"Model: {model_info.id}")
        print(f"Name: {model_info.name}")
        print(f"Description: {model_info.description}")
        print(f"Context Length: {model_info.context_length:,} tokens")
        
        if hasattr(model_info, 'pricing') and model_info.pricing:
            pricing = model_info.pricing
            if hasattr(pricing, 'prompt') and pricing.prompt:
                print(f"Prompt Price: ${pricing.prompt} per 1K tokens")
            if hasattr(pricing, 'completion') and pricing.completion:
                print(f"Completion Price: ${pricing.completion} per 1K tokens")
        
        if hasattr(model_info, 'top_provider') and model_info.top_provider:
            print(f"Top Provider: {model_info.top_provider}")
            
    except Exception as e:
        print(f"Error getting model info: {e}")
    
    # Example 3: Filter models by criteria
    print("\n=== Filter Models by Criteria ===")
    
    # Find models with large context windows
    large_context_models = [
        model for model in models_response.data 
        if hasattr(model, 'context_length') and model.context_length and model.context_length >= 100000
    ]
    
    print(f"Models with 100K+ context length ({len(large_context_models)} found):")
    for model in large_context_models[:5]:
        print(f"  - {model.id}: {model.context_length:,} tokens")
    
    # Find Anthropic models
    anthropic_models = [
        model for model in models_response.data 
        if 'anthropic' in model.id.lower()
    ]
    
    print(f"\nAnthropic models ({len(anthropic_models)} found):")
    for model in anthropic_models[:5]:
        print(f"  - {model.id}")
    
    # Find OpenAI models
    openai_models = [
        model for model in models_response.data 
        if 'openai' in model.id.lower()
    ]
    
    print(f"\nOpenAI models ({len(openai_models)} found):")
    for model in openai_models[:5]:
        print(f"  - {model.id}")

def demonstrate_context_length_management():
    """Demonstrate context length management features."""
    print("\n=== Context Length Management ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    # Get context length for specific models
    models_to_check = [
        "anthropic/claude-3-haiku",
        "anthropic/claude-3-sonnet", 
        "anthropic/claude-3-opus",
        "openai/gpt-3.5-turbo",
        "openai/gpt-4-turbo"
    ]
    
    print("Context lengths for popular models:")
    for model_id in models_to_check:
        try:
            context_length = client.get_context_length(model_id)
            print(f"  {model_id}: {context_length:,} tokens")
        except Exception as e:
            print(f"  {model_id}: Error - {e}")
    
    # Refresh context length cache
    print("\nRefreshing context length cache...")
    client.refresh_context_lengths()
    print("Context length cache refreshed successfully.")

def demonstrate_model_pricing():
    """Demonstrate model pricing analysis."""
    print("\n=== Model Pricing Analysis ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    models_response = client.models.list()
    
    # Find cheapest models
    priced_models = []
    
    for model in models_response.data:
        if hasattr(model, 'pricing') and model.pricing:
            pricing = model.pricing
            if hasattr(pricing, 'prompt') and pricing.prompt:
                try:
                    prompt_price = float(pricing.prompt)
                    priced_models.append((model.id, prompt_price, model.name))
                except (ValueError, TypeError):
                    continue
    
    # Sort by price
    priced_models.sort(key=lambda x: x[1])
    
    print("10 Cheapest Models (by prompt price):")
    for i, (model_id, price, name) in enumerate(priced_models[:10]):
        print(f"{i+1:2d}. {model_id}")
        print(f"    Price: ${price} per 1K tokens")
        print(f"    Name: {name}")
        print()
    
    print("10 Most Expensive Models (by prompt price):")
    for i, (model_id, price, name) in enumerate(priced_models[-10:]):
        print(f"{i+1:2d}. {model_id}")
        print(f"    Price: ${price} per 1K tokens") 
        print(f"    Name: {name}")
        print()

def demonstrate_model_selection():
    """Demonstrate intelligent model selection based on requirements."""
    print("\n=== Intelligent Model Selection ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    def select_model_for_task(task_type: str, max_budget: float = None, min_context: int = None):
        """Select the best model for a given task type and constraints."""
        models_response = client.models.list()
        suitable_models = []
        
        for model in models_response.data:
            # Check budget constraint
            if max_budget:
                if not (hasattr(model, 'pricing') and model.pricing and 
                       hasattr(model.pricing, 'prompt') and model.pricing.prompt):
                    continue
                try:
                    price = float(model.pricing.prompt)
                    if price > max_budget:
                        continue
                except (ValueError, TypeError):
                    continue
            
            # Check context length constraint
            if min_context:
                if not (hasattr(model, 'context_length') and model.context_length):
                    continue
                if model.context_length < min_context:
                    continue
            
            # Task-specific filtering
            if task_type == "coding":
                if any(keyword in model.id.lower() for keyword in ['code', 'gpt-4', 'claude-3']):
                    suitable_models.append(model)
            elif task_type == "creative":
                if any(keyword in model.id.lower() for keyword in ['gpt-4', 'claude-3', 'opus']):
                    suitable_models.append(model)
            elif task_type == "analysis":
                if any(keyword in model.id.lower() for keyword in ['claude-3', 'gpt-4']):
                    suitable_models.append(model)
            else:
                suitable_models.append(model)
        
        return suitable_models[:5]  # Return top 5 matches
    
    # Example selections
    print("Models for coding tasks (budget: $0.01/1K tokens):")
    coding_models = select_model_for_task("coding", max_budget=0.01)
    for model in coding_models:
        print(f"  - {model.id}")
    
    print("\nModels for creative writing (min context: 50K tokens):")
    creative_models = select_model_for_task("creative", min_context=50000)
    for model in creative_models:
        context = getattr(model, 'context_length', 'Unknown')
        print(f"  - {model.id} (Context: {context})")
    
    print("\nModels for data analysis (budget: $0.005/1K tokens, min context: 100K):")
    analysis_models = select_model_for_task("analysis", max_budget=0.005, min_context=100000)
    for model in analysis_models:
        context = getattr(model, 'context_length', 'Unknown')
        price = getattr(model.pricing, 'prompt', 'Unknown') if hasattr(model, 'pricing') and model.pricing else 'Unknown'
        print(f"  - {model.id} (Context: {context}, Price: ${price})")

def demonstrate_model_endpoints():
    """Demonstrate model endpoints functionality."""
    print("\n=== Model Endpoints ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    try:
        # Get model endpoints information
        endpoints_response = client.models.list_endpoints()
        
        print("Model endpoints information:")
        if hasattr(endpoints_response, 'data') and isinstance(endpoints_response.data, dict):
            # Show first few entries
            items = list(endpoints_response.data.items())[:5]
            for model_id, endpoint_info in items:
                print(f"\nModel: {model_id}")
                if isinstance(endpoint_info, dict):
                    for key, value in endpoint_info.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  Info: {endpoint_info}")
        else:
            print(f"Endpoints data: {endpoints_response.data}")
            
    except Exception as e:
        print(f"Error getting model endpoints: {e}")

if __name__ == "__main__":
    main()
    # Uncomment to run additional examples
    # demonstrate_context_length_management()
    # demonstrate_model_pricing()
    # demonstrate_model_selection()
    # demonstrate_model_endpoints()