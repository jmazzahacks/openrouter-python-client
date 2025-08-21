"""
Error Handling Example

This example demonstrates comprehensive error handling patterns
for the OpenRouter Python client.
"""

import os
import time
from openrouter_client import OpenRouterClient
from openrouter_client.exceptions import (
    OpenRouterError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NotFoundError,
    ServerError
)

def main():
    # Initialize the client
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    
    client = OpenRouterClient(
        api_key=api_key,
        http_referer="https://your-site.com",
        x_title="Error Handling Example"
    )
    
    # Example 1: Basic error handling
    print("=== Basic Error Handling ===")
    
    try:
        response = client.chat.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(f"Success: {response.choices[0].message.content}")
        
    except AuthenticationError as e:
        print(f"Authentication failed: {e}")
        print("Check your API key and ensure it's valid")
        
    except RateLimitError as e:
        print(f"Rate limit exceeded: {e}")
        print(f"Retry after: {e.retry_after} seconds")
        
    except ValidationError as e:
        print(f"Invalid request parameters: {e}")
        print("Check your request parameters and model availability")
        
    except NotFoundError as e:
        print(f"Resource not found: {e}")
        print("The specified model or endpoint may not exist")
        
    except ServerError as e:
        print(f"Server error: {e}")
        print("This is likely a temporary issue, try again later")
        
    except OpenRouterError as e:
        print(f"General API error: {e}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")

def demonstrate_authentication_errors():
    """Demonstrate handling authentication errors."""
    print("\n=== Authentication Error Handling ===")
    
    # Test with invalid API key
    invalid_client = OpenRouterClient(api_key="invalid-key-12345")
    
    try:
        response = invalid_client.chat.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": "Test message"}]
        )
    except AuthenticationError as e:
        print(f"Expected authentication error: {e}")
        print("Solution: Verify your API key is correct and active")
    
    # Test with missing API key
    try:
        no_key_client = OpenRouterClient(api_key="")
        response = no_key_client.chat.create(
            model="anthropic/claude-3-haiku", 
            messages=[{"role": "user", "content": "Test"}]
        )
    except AuthenticationError as e:
        print(f"Missing API key error: {e}")
        print("Solution: Provide a valid API key")
    except Exception as e:
        print(f"Other error (missing key): {e}")

def demonstrate_rate_limit_handling():
    """Demonstrate handling rate limit errors with retry logic."""
    print("\n=== Rate Limit Error Handling ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    def robust_request_with_retry(max_retries=3, backoff_multiplier=2):
        """Make a request with exponential backoff retry on rate limits."""
        
        for attempt in range(max_retries):
            try:
                response = client.chat.create(
                    model="anthropic/claude-3-haiku",
                    messages=[{"role": "user", "content": f"Request attempt {attempt + 1}"}]
                )
                return response
                
            except RateLimitError as e:
                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    wait_time = e.retry_after if hasattr(e, 'retry_after') else (backoff_multiplier ** attempt)
                    print(f"Rate limited on attempt {attempt + 1}. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries exceeded. Final error: {e}")
                    raise
                    
            except Exception as e:
                print(f"Non-rate-limit error on attempt {attempt + 1}: {e}")
                raise
        
        return None
    
    try:
        response = robust_request_with_retry()
        if response:
            print(f"Success after retries: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Final failure: {e}")

def demonstrate_validation_errors():
    """Demonstrate handling validation errors."""
    print("\n=== Validation Error Handling ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    # Test various validation errors
    validation_tests = [
        {
            "name": "Invalid model",
            "params": {
                "model": "invalid-model-name",
                "messages": [{"role": "user", "content": "Test"}]
            }
        },
        {
            "name": "Invalid message format",
            "params": {
                "model": "anthropic/claude-3-haiku",
                "messages": [{"role": "invalid_role", "content": "Test"}]
            }
        },
        {
            "name": "Negative max_tokens",
            "params": {
                "model": "anthropic/claude-3-haiku",
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": -100
            }
        },
        {
            "name": "Invalid temperature range",
            "params": {
                "model": "anthropic/claude-3-haiku",
                "messages": [{"role": "user", "content": "Test"}],
                "temperature": 5.0  # Should be 0-2
            }
        }
    ]
    
    for test in validation_tests:
        print(f"\nTesting: {test['name']}")
        try:
            response = client.chat.create(**test['params'])
            print(f"  Unexpected success: {response.choices[0].message.content}")
        except ValidationError as e:
            print(f"  Expected validation error: {e}")
        except Exception as e:
            print(f"  Other error: {e}")

def demonstrate_network_errors():
    """Demonstrate handling network and server errors."""
    print("\n=== Network and Server Error Handling ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    
    # Test with invalid base URL to simulate network error
    client_bad_url = OpenRouterClient(
        api_key=api_key,
        base_url="https://invalid-url-that-does-not-exist.com/api/v1"
    )
    
    try:
        response = client_bad_url.chat.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": "Test"}]
        )
    except Exception as e:
        print(f"Network error (expected): {e}")
        print("Solution: Check your internet connection and API endpoint URL")

def demonstrate_comprehensive_error_handler():
    """Demonstrate a comprehensive error handling wrapper."""
    print("\n=== Comprehensive Error Handler ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    def safe_chat_request(client, model, messages, max_retries=3, **kwargs):
        """
        A robust wrapper for chat requests with comprehensive error handling.
        
        Returns:
            tuple: (success: bool, result: response or error message)
        """
        
        for attempt in range(max_retries):
            try:
                response = client.chat.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                return True, response
                
            except AuthenticationError as e:
                return False, f"Authentication failed: {e}. Check your API key."
                
            except ValidationError as e:
                return False, f"Invalid request: {e}. Check your parameters."
                
            except NotFoundError as e:
                return False, f"Resource not found: {e}. Check model availability."
                
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = getattr(e, 'retry_after', 2 ** attempt)
                    print(f"Rate limited (attempt {attempt + 1}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return False, f"Rate limit exceeded after {max_retries} attempts: {e}"
                    
            except ServerError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Server error (attempt {attempt + 1}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return False, f"Server error after {max_retries} attempts: {e}"
                    
            except OpenRouterError as e:
                return False, f"API error: {e}"
                
            except Exception as e:
                return False, f"Unexpected error: {e}"
        
        return False, "Max retries exceeded"
    
    # Test the comprehensive handler
    test_cases = [
        {
            "name": "Valid request",
            "model": "anthropic/claude-3-haiku",
            "messages": [{"role": "user", "content": "Hello!"}]
        },
        {
            "name": "Invalid model",
            "model": "invalid-model",
            "messages": [{"role": "user", "content": "Hello!"}]
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        success, result = safe_chat_request(
            client, 
            test_case['model'], 
            test_case['messages']
        )
        
        if success:
            print(f"  ✅ Success: {result.choices[0].message.content[:100]}...")
        else:
            print(f"  ❌ Failed: {result}")

def demonstrate_logging_for_debugging():
    """Demonstrate using logging for error debugging."""
    print("\n=== Logging for Error Debugging ===")
    
    import logging
    from openrouter_client import configure_logging
    
    # Enable debug logging
    configure_logging(level=logging.DEBUG)
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    print("Making request with debug logging enabled...")
    
    try:
        response = client.chat.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": "This is a test with logging"}]
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error with detailed logs: {e}")
    
    # Reset logging level
    configure_logging(level=logging.WARNING)
    print("Logging level reset to WARNING")

def demonstrate_enhanced_error_details():
    """Demonstrate the enhanced error handling with detailed information."""
    print("\n=== Enhanced Error Details ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    try:
        # Try to use a model that might have specific requirements or restrictions
        response = client.chat.create(
            model="meta-llama/llama-2-70b-chat",  # This might have specific requirements
            messages=[
                {"role": "user", "content": "Hello!"}
            ],
            max_tokens=100
        )
        print(f"Success: {response.choices[0].message.content}")
        
    except Exception as e:
        print("=" * 60)
        print("ENHANCED ERROR HANDLING DEMONSTRATION")
        print("=" * 60)
        
        # Check if this is an APIError with enhanced details
        if hasattr(e, 'get_detailed_error_info'):
            print("Enhanced Error Information Available!")
            print("\n1. Basic Error Details:")
            print(f"   Status Code: {getattr(e, 'status_code', 'N/A')}")
            print(f"   Error Message: {getattr(e, 'message', str(e))}")
            
            print("\n2. Comprehensive Error Summary:")
            print(e.get_detailed_error_info())
            
            print("\n3. Structured Error Details:")
            if hasattr(e, 'details') and e.details:
                for key, value in e.details.items():
                    if value is not None:
                        print(f"   {key}: {value}")
            
            print("\n4. Raw Response Information:")
            if hasattr(e, 'response') and e.response:
                if hasattr(e.response, 'headers'):
                    print(f"   Response Headers: {dict(e.response.headers)}")
                if hasattr(e.response, 'text') and e.response.text:
                    print(f"   Response Text: {e.response.text}")
        else:
            print(f"Standard error: {e}")
            print("This error doesn't have enhanced details available")

def demonstrate_provider_error_debugging():
    """Demonstrate debugging provider-specific errors with enhanced information."""
    print("\n=== Provider Error Debugging ===")
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    client = OpenRouterClient(api_key=api_key)
    
    # Test cases that might trigger different types of provider errors
    test_cases = [
        {
            "name": "Model with specific requirements",
            "model": "meta-llama/llama-2-70b-chat",
            "description": "Testing model that might have specific requirements"
        },
        {
            "name": "Invalid model name",
            "model": "invalid-provider/invalid-model",
            "description": "Testing completely invalid model name"
        },
        {
            "name": "Model with specific parameters",
            "model": "anthropic/claude-3-opus-20240229",
            "description": "Testing model that might have specific parameter requirements"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        print(f"Description: {test_case['description']}")
        
        try:
            response = client.chat.create(
                model=test_case['model'],
                messages=[
                    {"role": "user", "content": "Hello!"}
                ],
                max_tokens=100
            )
            print(f"✅ Success: {response.choices[0].message.content[:50]}...")
            
        except Exception as e:
            print(f"❌ Error: {type(e).__name__}")
            
            # Use enhanced error handling if available
            if hasattr(e, 'get_detailed_error_info'):
                print("\nDetailed Error Information:")
                print(e.get_detailed_error_info())
                
                # Check for provider-specific error patterns
                if hasattr(e, 'details') and e.details:
                    if any(key in str(e.details).lower() for key in ['provider', 'model', 'requirement', 'restriction']):
                        print("\n🔍 This appears to be a provider-specific error.")
                        print("   Check the model requirements or try a different model.")
                    
                    if 'type' in e.details:
                        print(f"   Error Type: {e.details['type']}")
                    if 'code' in e.details:
                        print(f"   Error Code: {e.details['code']}")
            else:
                print(f"   Standard error: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
    # Uncomment to run specific error handling demonstrations
    # demonstrate_authentication_errors()
    # demonstrate_rate_limit_handling()
    # demonstrate_validation_errors()
    # demonstrate_network_errors()
    # demonstrate_comprehensive_error_handler()
    # demonstrate_logging_for_debugging()
    # demonstrate_enhanced_error_details()
    # demonstrate_provider_error_debugging()
    
    # Run the enhanced error handling demonstrations by default
    print("\n" + "="*60)
    print("RUNNING ENHANCED ERROR HANDLING DEMONSTRATIONS")
    print("="*60)
    demonstrate_enhanced_error_details()
    demonstrate_provider_error_debugging()