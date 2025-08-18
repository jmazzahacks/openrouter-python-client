#!/usr/bin/env python
"""
Filter models that support image/multimodal input.

This script queries the OpenRouter API to find models that support
image input via their architecture.input_modalities, filtering out
deprecated models, and sorts them by price.
"""

import os
from openrouter_client import OpenRouterClient

def get_pricing_per_1m_tokens(model):
    """Extract prompt pricing per 1M tokens from model."""
    if hasattr(model, 'pricing') and model.pricing and hasattr(model.pricing, 'prompt'):
        try:
            # Pricing is typically per token, convert to per 1M tokens
            prompt_price = float(model.pricing.prompt)
            return prompt_price * 1_000_000
        except (ValueError, TypeError):
            return float('inf')  # Put unparseable prices at the end
    return float('inf')  # Put models without pricing at the end

def get_models_with_image_support(include_deprecated=False):
    """Get all models that support image input, sorted by price."""
    client = OpenRouterClient(api_key=os.getenv("OPENROUTER_API_KEY"))
    
    print("Fetching models with details...")
    response = client.models.list(details=True)
    
    image_models = []
    deprecation_keywords = ['deprecated', 'removed', 'discontinued', 'being deprecated']
    
    for model in response.data:
        # Check if model supports image input
        if hasattr(model, 'architecture') and model.architecture:
            if hasattr(model.architecture, 'input_modalities') and model.architecture.input_modalities:
                if 'image' in model.architecture.input_modalities:
                    # Skip deprecated models unless explicitly requested
                    if not include_deprecated and model.description:
                        desc_lower = model.description.lower()
                        if any(keyword in desc_lower for keyword in deprecation_keywords):
                            continue
                    
                    image_models.append(model)
    
    # Sort by price per 1M tokens (cheapest first)
    image_models.sort(key=get_pricing_per_1m_tokens)
    
    return image_models

if __name__ == "__main__":
    print("Finding models that support image input (sorted by price)...\n")
    
    models = get_models_with_image_support()
    
    print(f"Found {len(models)} models that support image input (cheapest first):\n")
    
    for model in models:
        print(f"- {model.id}")
        
        if hasattr(model, 'name') and model.name:
            print(f"  Name: {model.name}")
        
        # Show pricing per 1M tokens
        price_per_1m = get_pricing_per_1m_tokens(model)
        if price_per_1m == float('inf'):
            print(f"  Price: No pricing available")
        elif price_per_1m == 0:
            print(f"  Price: FREE")
        else:
            print(f"  Price: ${price_per_1m:.2f} per 1M prompt tokens")
        
        # Show image pricing if available
        if hasattr(model, 'pricing') and model.pricing and hasattr(model.pricing, 'image'):
            try:
                image_price = float(model.pricing.image)
                if image_price > 0:
                    print(f"  Image price: ${image_price:.6f} per image")
            except (ValueError, TypeError):
                pass
        
        # Show supported modalities
        if hasattr(model, 'architecture') and model.architecture and hasattr(model.architecture, 'input_modalities'):
            modalities = model.architecture.input_modalities
            print(f"  Input modalities: {modalities}")
        
        print()
    
    print(f"\nTotal: {len(models)} models support image input")
    
    # Show pricing summary
    free_models = [m for m in models if get_pricing_per_1m_tokens(m) == 0]
    paid_models = [m for m in models if 0 < get_pricing_per_1m_tokens(m) < float('inf')]
    
    if free_models:
        print(f"- {len(free_models)} are FREE")
    if paid_models:
        cheapest = min(paid_models, key=get_pricing_per_1m_tokens)
        cheapest_price = get_pricing_per_1m_tokens(cheapest)
        print(f"- Cheapest paid model: {cheapest.id} at ${cheapest_price:.2f}/1M tokens")
    
    # Also show count with deprecated models
    all_models = get_models_with_image_support(include_deprecated=True)
    deprecated_count = len(all_models) - len(models)
    if deprecated_count > 0:
        print(f"- {deprecated_count} additional deprecated models were filtered out")