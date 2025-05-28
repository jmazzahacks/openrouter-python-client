# OpenRouter Python Client Examples

This directory contains comprehensive examples demonstrating how to use the OpenRouter Python client effectively. Each example focuses on specific features and use cases.

## Example Files

### 1. [basic_chat.py](basic_chat.py)
**Basic chat completions and conversation management**

- Simple chat completions
- System messages and conversation context
- Multi-turn conversations
- Usage information and token counting

**Run with:**
```bash
python basic_chat.py
```

### 2. [streaming_chat.py](streaming_chat.py)
**Real-time streaming responses**

- Basic streaming for real-time output
- Streaming with real-time processing (word counting)
- Multi-turn streaming conversations
- Error handling in streaming mode

**Run with:**
```bash
python streaming_chat.py
```

### 3. [function_calling.py](function_calling.py)
**Function calling and tool integration**

- Using the `@tool` decorator for easy function definition
- Single and multiple function calls
- Multi-turn conversations with function calling
- Forced function calling and complex workflows

**Run with:**
```bash
python function_calling.py
```

### 4. [prompt_caching.py](prompt_caching.py)
**Prompt caching for cost optimization**

- Basic prompt caching with Anthropic Claude
- Caching system prompts for consistent behavior
- Conversation context caching
- Cost comparison with and without caching

**Run with:**
```bash
python prompt_caching.py
```

### 5. [text_completions.py](text_completions.py)
**Text completion endpoint usage**

- Basic text completions
- Creative writing and code completion
- Multiple completions and custom parameters
- Streaming text completions
- Prompt engineering techniques

**Run with:**
```bash
python text_completions.py
```

### 6. [model_management.py](model_management.py)
**Model information and management**

- Listing and filtering available models
- Getting specific model information
- Context length management
- Model pricing analysis
- Intelligent model selection based on requirements

**Run with:**
```bash
python model_management.py
```

### 7. [credits_and_usage.py](credits_and_usage.py)
**Credits, usage monitoring, and budget management**

- Checking credit balance and usage
- API key information
- Usage monitoring during API calls
- Budget management and alerts
- Cost optimization strategies

**Run with:**
```bash
python credits_and_usage.py
```

### 8. [error_handling.py](error_handling.py)
**Comprehensive error handling patterns**

- Basic error handling for all exception types
- Authentication and validation errors
- Rate limit handling with retry logic
- Network and server error handling
- Comprehensive error handling wrapper
- Logging for debugging

**Run with:**
```bash
python error_handling.py
```

### 9. [advanced_usage.py](advanced_usage.py)
**Advanced patterns and techniques**

- Context manager usage
- Concurrent request processing
- Advanced streaming with real-time analysis
- Conversation management system
- Custom client configuration
- Model fallback strategies

**Run with:**
```bash
python advanced_usage.py
```

## Setup Requirements

Before running any examples, make sure you have:

1. **Installed the OpenRouter client:**
   ```bash
   pip install openrouter-client-unofficial
   ```

2. **Set your API key** (choose one method):
   
   **Environment variable (recommended):**
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```
   
   **Or edit the examples directly** and replace `"your-api-key-here"` with your actual API key.

3. **Get an API key** from [OpenRouter](https://openrouter.ai/keys) if you don't have one.

## Running Examples

### Basic Usage
```bash
# Run a specific example
python basic_chat.py

# Run with debug output
OPENROUTER_LOG_LEVEL=DEBUG python function_calling.py
```

### Running All Examples
```bash
# Run all basic examples
for file in basic_chat.py streaming_chat.py function_calling.py; do
    echo "Running $file..."
    python "$file"
    echo "---"
done
```

### Customizing Examples

Most examples include additional demonstration functions that are commented out. Uncomment them in the `if __name__ == "__main__":` section to run specific features:

```python
if __name__ == "__main__":
    main()
    # Uncomment to run additional examples
    # demonstrate_advanced_feature()
    # demonstrate_error_cases()
```

## Example Categories

### 🚀 **Getting Started**
- `basic_chat.py` - Start here for basic usage
- `streaming_chat.py` - Learn real-time streaming

### 🛠️ **Core Features**
- `function_calling.py` - Tool integration and function calling
- `text_completions.py` - Text completion endpoint
- `model_management.py` - Working with models

### 💰 **Cost Management**
- `prompt_caching.py` - Reduce costs with caching
- `credits_and_usage.py` - Monitor spending and usage

### 🔧 **Production Ready**
- `error_handling.py` - Robust error handling
- `advanced_usage.py` - Advanced patterns and techniques

## Common Use Cases

### Chat Application
```python
# Combine basic_chat.py + streaming_chat.py + error_handling.py
# For a robust chat application with real-time responses
```

### Tool-Enabled Assistant
```python
# Combine function_calling.py + error_handling.py + advanced_usage.py
# For an assistant that can call external functions and APIs
```

### Cost-Optimized Processing
```python
# Combine prompt_caching.py + credits_and_usage.py + model_management.py
# For cost-effective batch processing with monitoring
```

### Production Service
```python
# Combine all examples focusing on:
# - error_handling.py for robustness
# - advanced_usage.py for scalability
# - credits_and_usage.py for monitoring
```

## Tips for Development

1. **Start with basic examples** before moving to advanced ones
2. **Set up proper error handling** early in your development
3. **Monitor your usage** to avoid unexpected costs
4. **Use streaming** for better user experience in interactive applications
5. **Implement function calling** to extend your assistant's capabilities
6. **Use prompt caching** for repeated content to reduce costs

## Support

If you encounter issues running these examples:

1. Check that your API key is valid and has sufficient credits
2. Verify you're using a supported Python version (3.9+)
3. Ensure you have the latest version of the client
4. Review the error handling examples for common issues
5. Check the [main documentation](../README.md) for additional help

For more information, see the [full documentation](../index.md).