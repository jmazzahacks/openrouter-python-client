from openrouter_client import OpenRouterClient
from openrouter_client.models import (
    # Helper functions for creating tools
    create_tool,
    string_param,
    number_param,
    array_param,
    # Helper functions for tool choice
    auto_tool_choice,
    no_tool_choice,
    function_tool_choice,
    # Helper functions for processing tool responses
    parse_tool_call_arguments,
    create_tool_response,
)

# Create the OpenRouter client
client = OpenRouterClient(
    api_key="your-api-key",
    http_referer="your-site-url",
    x_title="Your Site Name",
)

# Example 1: Creating a tool definition using the helper functions
search_books_tool = create_tool(
    name="search_books",
    description="Search for books in a database",
    parameters={
        "properties": {
            "query": string_param(
                description="The search query", 
                required=True,
                min_length=1,
            ),
            "max_results": number_param(
                description="Maximum number of results to return",
                integer=True,
                minimum=1,
                maximum=50,
            ),
            "categories": array_param(
                description="Categories to filter by",
                items={"type": "string"},
                min_items=1,
            ),
        },
        "required": ["query"],
    }
)

# Example 2: Making a request with tools
response = client.chat.completions.create(
    model="anthropic/claude-3-opus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that can search for books."},
        {"role": "user", "content": "Find me some books about artificial intelligence."}
    ],
    tools=[search_books_tool],
    tool_choice=auto_tool_choice(),  # Let the model decide whether to use tools
)

# Example 3: Processing tool calls
if response.choices and response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        # Parse the arguments from the tool call
        args = parse_tool_call_arguments(tool_call)
        
        # In a real application, you would process the arguments and call the actual function
        print(f"Tool called: {tool_call.function.name}")
        print(f"Arguments: {args}")
        
        # Create a fake result to demonstrate the response creation
        fake_result = [
            {"title": "Artificial Intelligence: A Modern Approach", "author": "Stuart Russell and Peter Norvig"},
            {"title": "Deep Learning", "author": "Ian Goodfellow, Yoshua Bengio, and Aaron Courville"},
        ]
        
        # Create a tool response message
        tool_response = create_tool_response(
            tool_call_id=tool_call.id,
            function_name=tool_call.function.name,
            result=fake_result,
        )
        
        # In a real application, you would add this response to your messages
        # and make another API call to complete the conversation
        print(f"Tool response: {tool_response}")

# Example 4: Forcing a specific function to be called
forced_response = client.chat.completions.create(
    model="anthropic/claude-3-opus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that can search for books."},
        {"role": "user", "content": "I need to find information about space exploration."}
    ],
    tools=[search_books_tool],
    tool_choice=function_tool_choice("search_books"),  # Force the model to call this function
)

# Example 5: Disabling tool calling
disabled_response = client.chat.completions.create(
    model="anthropic/claude-3-opus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that can search for books."},
        {"role": "user", "content": "Tell me about artificial intelligence."}
    ],
    tools=[search_books_tool],
    tool_choice=no_tool_choice(),  # Disable tool calling
)