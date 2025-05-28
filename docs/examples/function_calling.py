"""
Function Calling Example

This example demonstrates how to use function calling with the OpenRouter client
using the @tool decorator for easy function definition.
"""

import os
import json
from typing import List, Dict, Any
from openrouter_client import OpenRouterClient, tool

# Define functions using the @tool decorator
@tool
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get the current weather for a location.
    
    Args:
        location: The city and state/country to get weather for
        unit: Temperature unit (celsius or fahrenheit)
        
    Returns:
        Weather information dictionary
    """
    # Mock weather data - in a real app, you'd call a weather API
    weather_data = {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "Sunny",
        "humidity": 65,
        "wind_speed": 10
    }
    return weather_data

@tool  
def search_restaurants(location: str, cuisine: str = "any", price_range: str = "medium") -> List[dict]:
    """Search for restaurants in a location.
    
    Args:
        location: City or area to search in
        cuisine: Type of cuisine (italian, chinese, mexican, etc.)
        price_range: Price range (budget, medium, expensive)
        
    Returns:
        List of restaurant information
    """
    # Mock restaurant data
    restaurants = [
        {
            "name": f"Best {cuisine.title()} Place" if cuisine != "any" else "Great Restaurant",
            "location": location,
            "cuisine": cuisine,
            "price_range": price_range,
            "rating": 4.5,
            "address": "123 Main St"
        },
        {
            "name": f"Amazing {cuisine.title()} Spot" if cuisine != "any" else "Wonderful Eatery", 
            "location": location,
            "cuisine": cuisine,
            "price_range": price_range,
            "rating": 4.2,
            "address": "456 Oak Ave"
        }
    ]
    return restaurants

@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 18.0) -> dict:
    """Calculate tip amount and total bill.
    
    Args:
        bill_amount: The bill amount in dollars
        tip_percentage: Tip percentage (default 18%)
        
    Returns:
        Calculation results
    """
    tip_amount = bill_amount * (tip_percentage / 100)
    total_amount = bill_amount + tip_amount
    
    return {
        "bill_amount": bill_amount,
        "tip_percentage": tip_percentage,
        "tip_amount": round(tip_amount, 2),
        "total_amount": round(total_amount, 2)
    }

def main():
    # Initialize the client
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    
    client = OpenRouterClient(
        api_key=api_key,
        http_referer="https://your-site.com",
        x_title="Function Calling Example"
    )
    
    # Available tools
    available_tools = [
        get_weather.to_dict(),
        search_restaurants.to_dict(), 
        calculate_tip.to_dict()
    ]
    
    # Example 1: Single function call
    print("=== Single Function Call ===")
    
    response = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=[
            {"role": "system", "content": "You are a helpful assistant with access to various tools."},
            {"role": "user", "content": "What's the weather like in Paris?"}
        ],
        tools=available_tools,
        tool_choice="auto"
    )
    
    # Process tool calls
    if response.choices[0].message.tool_calls:
        print("Assistant is calling a tool...")
        
        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            print(f"Calling function: {function_name}")
            
            # Execute the appropriate function
            if function_name == "get_weather":
                result = get_weather.execute(tool_call.function.arguments)
            elif function_name == "search_restaurants":
                result = search_restaurants.execute(tool_call.function.arguments)
            elif function_name == "calculate_tip":
                result = calculate_tip.execute(tool_call.function.arguments)
            else:
                result = {"error": f"Unknown function: {function_name}"}
            
            print(f"Function result: {json.dumps(result, indent=2)}")
    else:
        print(f"Assistant: {response.choices[0].message.content}")
    
    # Example 2: Multi-turn conversation with function calling
    print("\n=== Multi-turn Conversation with Function Calling ===")
    
    messages = [
        {"role": "system", "content": "You are a helpful travel assistant."},
        {"role": "user", "content": "I'm planning a trip to Tokyo. Can you help me with the weather and restaurant recommendations?"}
    ]
    
    # First request
    response = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=messages,
        tools=available_tools
    )
    
    # Add assistant message to conversation
    messages.append(response.choices[0].message.dict())
    
    # Process any tool calls
    if response.choices[0].message.tool_calls:
        print("Assistant is gathering information...")
        
        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            
            # Execute function
            if function_name == "get_weather":
                result = get_weather.execute(tool_call.function.arguments)
            elif function_name == "search_restaurants":
                result = search_restaurants.execute(tool_call.function.arguments)
            else:
                result = {"error": f"Unknown function: {function_name}"}
            
            # Add tool result to conversation
            tool_response = {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            }
            messages.append(tool_response)
            
            print(f"Called {function_name}: {json.dumps(result, indent=2)}")
        
        # Get final response with tool results
        final_response = client.chat.create(
            model="anthropic/claude-3-haiku",
            messages=messages
        )
        
        print(f"\nAssistant: {final_response.choices[0].message.content}")
    else:
        print(f"Assistant: {response.choices[0].message.content}")
    
    # Example 3: Forced function calling
    print("\n=== Forced Function Calling ===")
    
    response = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=[
            {"role": "user", "content": "I had dinner that cost $85.50"}
        ],
        tools=[calculate_tip.to_dict()],
        tool_choice={"type": "function", "function": {"name": "calculate_tip"}}
    )
    
    # The model should be forced to call calculate_tip
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = calculate_tip.execute(tool_call.function.arguments)
        print(f"Tip calculation: {json.dumps(result, indent=2)}")

def demonstrate_complex_workflow():
    """Demonstrate a complex workflow with multiple function calls."""
    print("\n=== Complex Workflow Example ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    tools = [get_weather.to_dict(), search_restaurants.to_dict(), calculate_tip.to_dict()]
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can help with weather, restaurants, and calculations."},
        {"role": "user", "content": "I'm going to San Francisco for dinner tonight. Can you check the weather, find me a good Italian restaurant, and help me calculate a 20% tip on a $120 bill?"}
    ]
    
    print("User request: Weather + Restaurant + Tip calculation")
    
    # Make initial request
    response = client.chat.create(
        model="anthropic/claude-3-haiku",
        messages=messages,
        tools=tools
    )
    
    conversation_messages = messages + [response.choices[0].message.dict()]
    
    # Process all tool calls
    if response.choices[0].message.tool_calls:
        print("\nExecuting multiple functions...")
        
        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            
            if function_name == "get_weather":
                result = get_weather.execute(tool_call.function.arguments)
            elif function_name == "search_restaurants":
                result = search_restaurants.execute(tool_call.function.arguments)
            elif function_name == "calculate_tip":
                result = calculate_tip.execute(tool_call.function.arguments)
            else:
                continue
            
            # Add tool response
            tool_response = {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            }
            conversation_messages.append(tool_response)
            
            print(f"✓ {function_name}: {json.dumps(result, indent=2)}")
        
        # Get final comprehensive response
        final_response = client.chat.create(
            model="anthropic/claude-3-haiku", 
            messages=conversation_messages
        )
        
        print(f"\nFinal Assistant Response:\n{final_response.choices[0].message.content}")

if __name__ == "__main__":
    main()
    # Uncomment to run complex workflow
    # demonstrate_complex_workflow()