import json
import requests
from typing import List, Dict, Any, Optional

from openrouter_client import OpenRouterClient
from openrouter_client.models import (
    # Helper functions for creating tools
    create_tool,
    string_param,
    number_param,
    # Helper functions for processing tool responses
    parse_tool_call_arguments,
    create_tool_response,
)

# Initialize the OpenRouter client
client = OpenRouterClient(
    api_key="your-api-key",
    http_referer="your-site-url",
    x_title="Your Site Name",
)

# Define a function that we'll expose to the LLM
def search_weather(location: str, days: Optional[int] = 1) -> Dict[str, Any]:
    """Search for weather information for a location."""
    # This would normally call a real weather API
    # For demo purposes, we'll return mock data
    print(f"Searching weather for {location} for {days} days")
    return {
        "location": location,
        "forecast": [
            {"day": 1, "condition": "Sunny", "temperature": 25, "precipitation": 0},
            {"day": 2, "condition": "Partly Cloudy", "temperature": 22, "precipitation": 20},
            {"day": 3, "condition": "Rainy", "temperature": 18, "precipitation": 80},
        ][:days]
    }

# Create a tool definition for our function
weather_tool = create_tool(
    name="search_weather",
    description="Get the weather forecast for a location",
    parameters={
        "properties": {
            "location": string_param(
                description="The location to get weather for (city name)", 
                required=True
            ),
            "days": number_param(
                description="Number of days to forecast",
                minimum=1,
                maximum=7,
                integer=True,
            ),
        },
        "required": ["location"],
    }
)

# Function to map tool calls to functions
def execute_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Execute the appropriate function based on the tool name."""
    if name == "search_weather":
        return search_weather(**arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

# Have a conversation with the model
def conversation():
    """Demonstrate a multi-turn conversation with function calling."""
    messages = [
        {"role": "system", "content": "You are a helpful weather assistant that can provide forecasts."}
    ]
    
    # First user message
    user_message = "What's the weather like in New York?"
    print(f"\nUser: {user_message}")
    messages.append({"role": "user", "content": user_message})
    
    # Get the model's response
    response = client.chat.completions.create(
        model="anthropic/claude-3-opus",  # Or any model that supports function calling
        messages=messages,
        tools=[weather_tool],
    )
    
    assistant_message = response.choices[0].message
    messages.append(assistant_message.model_dump())  # Add to history
    
    # Check if the model wants to call a function
    if assistant_message.tool_calls:
        print(f"\nAssistant is calling a tool...")
        
        # Process each tool call
        for tool_call in assistant_message.tool_calls:
            # Parse and execute the function
            args = parse_tool_call_arguments(tool_call)
            result = execute_tool(tool_call.function.name, args)
            
            # Create a response from the tool
            tool_response = create_tool_response(
                tool_call_id=tool_call.id,
                function_name=tool_call.function.name,
                result=result
            )
            
            # Add the tool response to the conversation
            messages.append(tool_response)
            print(f"Tool result: {json.dumps(result, indent=2)}")
        
        # Get the final response from the assistant
        final_response = client.chat.completions.create(
            model="anthropic/claude-3-opus",
            messages=messages,
        )
        
        print(f"\nAssistant: {final_response.choices[0].message.content}")
        messages.append(final_response.choices[0].message.model_dump())
    else:
        # Model responded directly without calling a function
        print(f"\nAssistant: {assistant_message.content}")
    
    # Continue the conversation with a follow-up question
    user_message = "And what about the weather in San Francisco for the next 3 days?"
    print(f"\nUser: {user_message}")
    messages.append({"role": "user", "content": user_message})
    
    # Get the response to the follow-up
    response = client.chat.completions.create(
        model="anthropic/claude-3-opus",
        messages=messages,
        tools=[weather_tool],
    )
    
    # Process the response similar to before...
    # (continuing the same pattern as above)

# Run the conversation
if __name__ == "__main__":
    conversation()