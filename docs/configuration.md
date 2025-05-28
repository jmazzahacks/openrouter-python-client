# Configuration

This guide covers configuration options and customization for the OpenRouter Python client.

## Client Configuration

### Basic Configuration

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(
    api_key="your-api-key",                    # Required
    base_url="https://openrouter.ai/api/v1",   # Default base URL
    http_referer="https://your-site.com",      # Optional referer header
    x_title="Your App Name",                   # Optional app name header
    timeout=30.0,                              # Request timeout in seconds
    max_retries=3                              # Maximum retry attempts
)
```

### Advanced Configuration

```python
from openrouter_client import OpenRouterClient, AuthManager, HTTPManager

# Custom authentication manager
auth_manager = AuthManager(
    api_key="your-api-key",
    encrypt_key=True  # Encrypt API key in memory
)

# Custom HTTP manager with advanced settings
http_manager = HTTPManager(
    base_url="https://openrouter.ai/api/v1",
    timeout=60.0,
    max_retries=5,
    retry_delay=1.0,                           # Base retry delay
    retry_backoff=2.0,                         # Backoff multiplier
    rate_limit_enabled=True,                   # Enable automatic rate limiting
    rate_limit_buffer=0.1                      # Rate limit safety buffer (10%)
)

client = OpenRouterClient(
    auth_manager=auth_manager,
    http_manager=http_manager
)
```

## Environment Variables

The client supports several environment variables for configuration:

```bash
# API key (alternative to passing in code)
export OPENROUTER_API_KEY="your-api-key"

# Base URL override
export OPENROUTER_BASE_URL="https://custom-endpoint.com/api/v1"

# Default headers
export OPENROUTER_HTTP_REFERER="https://your-site.com"
export OPENROUTER_X_TITLE="Your App Name"

# Timeout and retry settings
export OPENROUTER_TIMEOUT="60.0"
export OPENROUTER_MAX_RETRIES="5"

# Logging level
export OPENROUTER_LOG_LEVEL="INFO"
```

Using environment variables:

```python
from openrouter_client import OpenRouterClient

# Client will automatically use environment variables
client = OpenRouterClient()  # Uses OPENROUTER_API_KEY automatically
```

## Logging Configuration

### Basic Logging Setup

```python
from openrouter_client import configure_logging
import logging

# Configure all OpenRouter client logging
configure_logging(level=logging.INFO)

# Or configure specific components
configure_logging(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

### Advanced Logging Configuration

```python
import logging
from openrouter_client import OpenRouterClient

# Create custom logger configuration
logger = logging.getLogger("openrouter_client")
logger.setLevel(logging.DEBUG)

# Create custom handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Create custom formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

logger.addHandler(handler)

# Configure component-specific logging levels
logging.getLogger("openrouter_client.http").setLevel(logging.DEBUG)      # HTTP requests
logging.getLogger("openrouter_client.auth").setLevel(logging.INFO)       # Authentication
logging.getLogger("openrouter_client.endpoints").setLevel(logging.WARN)  # Endpoint calls
logging.getLogger("openrouter_client.streaming").setLevel(logging.ERROR) # Streaming

client = OpenRouterClient(api_key="your-api-key")
```

### File Logging

```python
import logging
from openrouter_client import configure_logging

# Log to file
configure_logging(
    level=logging.INFO,
    filename="openrouter_client.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

## Custom Authentication

### Basic Custom Authentication

```python
from openrouter_client import AuthManager, OpenRouterClient

class CustomAuthManager(AuthManager):
    """Custom authentication with additional headers."""
    
    def get_headers(self) -> dict:
        """Get authentication headers."""
        headers = super().get_headers()
        headers.update({
            "X-Custom-Header": "custom-value",
            "User-Agent": "MyApp/1.0"
        })
        return headers

auth_manager = CustomAuthManager(api_key="your-api-key")
client = OpenRouterClient(auth_manager=auth_manager)
```

### Secrets Management Integration

```python
from openrouter_client.auth import SecretsManager, AuthManager
from openrouter_client import OpenRouterClient

class AWSSecretsManager(SecretsManager):
    """Example AWS Secrets Manager integration."""
    
    def __init__(self, secret_name: str, region: str):
        self.secret_name = secret_name
        self.region = region
        # Initialize AWS client here
    
    def get_secret(self, key: str) -> str:
        """Retrieve secret from AWS Secrets Manager."""
        # Implement AWS Secrets Manager retrieval
        # This is a simplified example
        import boto3
        
        client = boto3.client('secretsmanager', region_name=self.region)
        response = client.get_secret_value(SecretId=self.secret_name)
        secrets = json.loads(response['SecretString'])
        return secrets.get(key)
    
    def set_secret(self, key: str, value: str) -> None:
        """Store secret in AWS Secrets Manager."""
        # Implement AWS Secrets Manager storage
        pass

# Use with OpenRouter client
secrets_manager = AWSSecretsManager("openrouter-secrets", "us-east-1")
auth_manager = AuthManager(secrets_manager=secrets_manager)
client = OpenRouterClient(auth_manager=auth_manager)
```

## Rate Limiting Configuration

### Built-in Rate Limiting

The client includes intelligent rate limiting via SmartSurge:

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(
    api_key="your-api-key",
    max_retries=5,                    # Retry on rate limits
    rate_limit_buffer=0.2             # 20% safety buffer
)

# Rate limiting is automatic
for i in range(100):
    response = client.chat.create(
        model="anthropic/claude-3-opus",
        messages=[{"role": "user", "content": f"Message {i}"}]
    )
    print(f"Completed request {i}")
```

### Custom Rate Limiting

```python
from openrouter_client import HTTPManager, OpenRouterClient
import time

class CustomRateLimitHTTPManager(HTTPManager):
    """Custom HTTP manager with additional rate limiting."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
    
    def make_request(self, *args, **kwargs):
        """Make request with custom rate limiting."""
        # Enforce minimum interval between requests
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
        return super().make_request(*args, **kwargs)

http_manager = CustomRateLimitHTTPManager()
client = OpenRouterClient(
    api_key="your-api-key",
    http_manager=http_manager
)
```

## Model Selection and Fallbacks

### Automatic Model Fallbacks

```python
from openrouter_client import OpenRouterClient
from openrouter_client.exceptions import NotFoundError, ServerError

class FallbackClient:
    """Client with automatic model fallbacks."""
    
    def __init__(self, api_key: str):
        self.client = OpenRouterClient(api_key=api_key)
        self.preferred_models = [
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet", 
            "openai/gpt-4-turbo",
            "openai/gpt-3.5-turbo"
        ]
    
    def chat_with_fallback(self, messages, **kwargs):
        """Try models in order of preference."""
        for model in self.preferred_models:
            try:
                return self.client.chat.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
            except (NotFoundError, ServerError) as e:
                print(f"Model {model} failed: {e}")
                continue
        
        raise Exception("All fallback models failed")

fallback_client = FallbackClient(api_key="your-api-key")
response = fallback_client.chat_with_fallback(
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Cost-Based Model Selection

```python
from openrouter_client import OpenRouterClient

class CostOptimizedClient:
    """Client that selects models based on cost and context needs."""
    
    def __init__(self, api_key: str):
        self.client = OpenRouterClient(api_key=api_key)
        self.model_tiers = {
            "cheap": ["openai/gpt-3.5-turbo", "anthropic/claude-3-haiku"],
            "medium": ["openai/gpt-4", "anthropic/claude-3-sonnet"],
            "premium": ["openai/gpt-4-turbo", "anthropic/claude-3-opus"]
        }
    
    def estimate_tokens(self, messages):
        """Rough token estimation."""
        total_chars = sum(len(msg["content"]) for msg in messages)
        return total_chars // 4  # Rough estimate
    
    def select_model(self, messages, max_cost_per_1k_tokens=0.01):
        """Select most capable model within cost budget."""
        token_count = self.estimate_tokens(messages)
        
        # Get model pricing
        models = self.client.models.list()
        suitable_models = []
        
        for model in models.data:
            if model.pricing and model.pricing.prompt:
                cost_per_1k = float(model.pricing.prompt)
                if cost_per_1k <= max_cost_per_1k_tokens:
                    suitable_models.append((model.id, cost_per_1k))
        
        # Sort by cost (descending) to get best model within budget
        suitable_models.sort(key=lambda x: x[1], reverse=True)
        
        return suitable_models[0][0] if suitable_models else "openai/gpt-3.5-turbo"
    
    def smart_chat(self, messages, max_cost_per_1k_tokens=0.01, **kwargs):
        """Chat with cost-optimized model selection."""
        model = self.select_model(messages, max_cost_per_1k_tokens)
        print(f"Selected model: {model}")
        
        return self.client.chat.create(
            model=model,
            messages=messages,
            **kwargs
        )

cost_client = CostOptimizedClient(api_key="your-api-key")
response = cost_client.smart_chat(
    messages=[{"role": "user", "content": "Simple question"}],
    max_cost_per_1k_tokens=0.005  # Prefer cheaper models
)
```

## Context Manager Patterns

### Resource Pool Management

```python
from openrouter_client import OpenRouterClient
from contextlib import contextmanager
import threading

class ClientPool:
    """Pool of OpenRouter clients for concurrent usage."""
    
    def __init__(self, api_key: str, pool_size: int = 5):
        self.pool = []
        self.lock = threading.Lock()
        
        for _ in range(pool_size):
            client = OpenRouterClient(api_key=api_key)
            self.pool.append(client)
    
    @contextmanager
    def get_client(self):
        """Get a client from the pool."""
        with self.lock:
            if not self.pool:
                raise Exception("No clients available in pool")
            client = self.pool.pop()
        
        try:
            yield client
        finally:
            with self.lock:
                self.pool.append(client)
    
    def close_all(self):
        """Close all clients in the pool."""
        with self.lock:
            for client in self.pool:
                client.close()
            self.pool.clear()

# Usage
pool = ClientPool(api_key="your-api-key", pool_size=3)

try:
    with pool.get_client() as client:
        response = client.chat.create(
            model="anthropic/claude-3-opus",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)
finally:
    pool.close_all()
```

## Configuration Files

### YAML Configuration

```yaml
# openrouter_config.yaml
openrouter:
  api_key: "your-api-key"
  base_url: "https://openrouter.ai/api/v1"
  timeout: 30.0
  max_retries: 3
  
  headers:
    http_referer: "https://your-site.com"
    x_title: "Your App Name"
  
  rate_limiting:
    enabled: true
    buffer: 0.1
  
  logging:
    level: "INFO"
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  models:
    preferred: ["anthropic/claude-3-opus", "openai/gpt-4-turbo"]
    fallback: ["openai/gpt-3.5-turbo"]
```

```python
import yaml
from openrouter_client import OpenRouterClient

def load_config(config_file: str):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config['openrouter']

def create_client_from_config(config_file: str):
    """Create client from YAML configuration."""
    config = load_config(config_file)
    
    return OpenRouterClient(
        api_key=config['api_key'],
        base_url=config.get('base_url'),
        timeout=config.get('timeout'),
        max_retries=config.get('max_retries'),
        http_referer=config.get('headers', {}).get('http_referer'),
        x_title=config.get('headers', {}).get('x_title')
    )

# Usage
client = create_client_from_config("openrouter_config.yaml")
```

### JSON Configuration

```json
{
  "openrouter": {
    "api_key": "your-api-key",
    "base_url": "https://openrouter.ai/api/v1",
    "timeout": 30.0,
    "max_retries": 3,
    "headers": {
      "http_referer": "https://your-site.com",
      "x_title": "Your App Name"
    },
    "rate_limiting": {
      "enabled": true,
      "buffer": 0.1
    }
  }
}
```

```python
import json
from openrouter_client import OpenRouterClient

def create_client_from_json(config_file: str):
    """Create client from JSON configuration."""
    with open(config_file, 'r') as f:
        config = json.load(f)['openrouter']
    
    return OpenRouterClient(
        api_key=config['api_key'],
        base_url=config.get('base_url'),
        timeout=config.get('timeout'),
        max_retries=config.get('max_retries'),
        http_referer=config.get('headers', {}).get('http_referer'),
        x_title=config.get('headers', {}).get('x_title')
    )

client = create_client_from_json("openrouter_config.json")
```