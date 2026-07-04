# Configuration

This guide covers configuration options and customization for the OpenRouter Python client.

## Client Configuration

### Basic Configuration

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(
    api_key="your-api-key",                    # Required (falls back to OPENROUTER_API_KEY)
    base_url="https://openrouter.ai/api/v1",   # Default base URL
    timeout=30.0,                              # Request timeout in seconds
)
```

The API key is encrypted in memory automatically whenever PyNaCl is installed;
no flag is required. `timeout` (default `60.0`) is one of several tuning kwargs
forwarded to the underlying HTTP layer — see the retry and rate-limiting sections
below for `retry_config`, `retries`, and `backoff_factor`.

### Advanced Configuration

```python
from openrouter_client import OpenRouterClient, RetryConfig

# Opt in to 429 retry-with-backoff and tune the HTTP layer
client = OpenRouterClient(
    api_key="your-api-key",
    base_url="https://openrouter.ai/api/v1",
    timeout=60.0,
    retries=3,                                       # transport-level retries (default 3)
    backoff_factor=0.5,                              # backoff multiplier (default 0.5)
    retry_config=RetryConfig(enabled=True, max_retries=5),
)
```

## Environment Variables

The client reads exactly two environment variables:

```bash
# API key (alternative to passing api_key= in code)
export OPENROUTER_API_KEY="your-api-key"

# Provisioning key (needed for credits and key-management endpoints)
export OPENROUTER_PROVISIONING_API_KEY="your-provisioning-key"
```

These are the only environment variables the client looks up. Everything else
(base URL, timeout, retries, logging) is configured through constructor
arguments or `configure_logging()`.

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

# Log to file (the file argument is `to_file`, a path)
configure_logging(
    level=logging.INFO,
    to_file="openrouter_client.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

## Custom Authentication

### Supplying keys through a custom SecretsManager

Authentication is customized by plugging in a `SecretsManager`, not by
subclassing `AuthManager`. The `SecretsManager` protocol requires a single
method:

```python
def get_key(self, name: str) -> bytearray:
    ...
```

`AuthManager` calls `get_key("OPENROUTER_API_KEY")` (and, when needed,
`get_key("OPENROUTER_PROVISIONING_API_KEY")`) to resolve credentials, and
raises `AuthenticationError` if the key is missing. The default is
`EnvironmentSecretsManager`, which reads those names from the environment. Wire
your own manager through the client constructor:

```python
from openrouter_client.auth import SecretsManager
from openrouter_client.exceptions import AuthenticationError
from openrouter_client import OpenRouterClient

class DictSecretsManager(SecretsManager):
    """Resolve keys from an in-memory mapping."""

    def __init__(self, secrets: dict[str, str]):
        self._secrets = secrets

    def get_key(self, name: str) -> bytearray:
        value = self._secrets.get(name)
        if value is None:
            raise AuthenticationError(f"Secret not found: {name}")
        return bytearray(value, "utf-8")

secrets_manager = DictSecretsManager({"OPENROUTER_API_KEY": "your-api-key"})
client = OpenRouterClient(secrets_manager=secrets_manager)
```

### Secrets Management Integration

```python
import json
from openrouter_client.auth import SecretsManager
from openrouter_client.exceptions import AuthenticationError
from openrouter_client import OpenRouterClient

class AWSSecretsManager(SecretsManager):
    """Example AWS Secrets Manager integration."""

    def __init__(self, secret_name: str, region: str):
        self.secret_name = secret_name
        self.region = region
        # Initialize the AWS client here

    def get_key(self, name: str) -> bytearray:
        """Retrieve a key (e.g. "OPENROUTER_API_KEY") from AWS Secrets Manager."""
        import boto3

        client = boto3.client("secretsmanager", region_name=self.region)
        response = client.get_secret_value(SecretId=self.secret_name)
        secrets = json.loads(response["SecretString"])
        value = secrets.get(name)
        if value is None:
            raise AuthenticationError(f"Secret not found: {name}")
        return bytearray(value, "utf-8")

# Use with OpenRouter client
secrets_manager = AWSSecretsManager("openrouter-secrets", "us-east-1")
client = OpenRouterClient(secrets_manager=secrets_manager)
```

## Rate Limiting Configuration

### Built-in Rate Limiting

The client applies rate limiting automatically. On construction it fetches the
key's `rate_limit` and configures every endpoint accordingly (via SmartSurge),
so no extra configuration is needed:

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(api_key="your-api-key")

# Rate limiting is automatic
for i in range(100):
    response = client.chat.create(
        model="anthropic/claude-3-opus",
        messages=[{"role": "user", "content": f"Message {i}"}]
    )
    print(f"Completed request {i}")
```

### Deriving and setting rate limits from credits

`calculate_rate_limits()` derives limits from remaining credits and returns a
dict with the keys `requests`, `period`, and `cooldown`. Feed that into
`set_rate_limit()` to apply a custom limit:

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(api_key="your-api-key")

limits = client.calculate_rate_limits()   # {"requests": ..., "period": ..., "cooldown": ...}
client.set_rate_limit(
    requests=limits["requests"],
    period=limits["period"],
    cooldown=limits["cooldown"],
)
```

### Retry-with-backoff on 429s

Retries are opt-in and configured only through `RetryConfig` (there is no
`max_retries` constructor argument):

```python
from openrouter_client import OpenRouterClient, RetryConfig

client = OpenRouterClient(
    api_key="your-api-key",
    retry_config=RetryConfig(
        enabled=True,          # off by default
        max_retries=5,
        base_delay=1.0,
        factor=2.0,
        max_delay=30.0,
        jitter=0.25,
        respect_retry_after=True,
    ),
)
```

## Model Selection and Fallbacks

### Automatic Model Fallbacks

```python
from openrouter_client import OpenRouterClient
from openrouter_client.exceptions import APIError, OpenRouterError

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
            except APIError as e:
                # 404 (unknown model) or 5xx (server error) -> try the next model
                if e.status_code == 404 or (e.status_code or 0) >= 500:
                    print(f"Model {model} failed ({e.status_code}): {e}")
                    continue
                raise
            except OpenRouterError as e:
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
        
        # Get model pricing (details=True returns ModelData objects, not id strings)
        models = self.client.models.list(details=True)
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

  retry:
    enabled: true
    max_retries: 5

  logging:
    level: "INFO"
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

  models:
    preferred: ["anthropic/claude-3-opus", "openai/gpt-4-turbo"]
    fallback: ["openai/gpt-3.5-turbo"]
```

```python
import yaml
from openrouter_client import OpenRouterClient, RetryConfig

def load_config(config_file: str):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config['openrouter']

def create_client_from_config(config_file: str):
    """Create client from YAML configuration."""
    config = load_config(config_file)

    retry = config.get('retry', {})
    return OpenRouterClient(
        api_key=config['api_key'],
        base_url=config.get('base_url'),
        timeout=config.get('timeout'),
        retry_config=RetryConfig(
            enabled=retry.get('enabled', False),
            max_retries=retry.get('max_retries', 5),
        ),
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
    "retry": {
      "enabled": true,
      "max_retries": 5
    }
  }
}
```

```python
import json
from openrouter_client import OpenRouterClient, RetryConfig

def create_client_from_json(config_file: str):
    """Create client from JSON configuration."""
    with open(config_file, 'r') as f:
        config = json.load(f)['openrouter']

    retry = config.get('retry', {})
    return OpenRouterClient(
        api_key=config['api_key'],
        base_url=config.get('base_url'),
        timeout=config.get('timeout'),
        retry_config=RetryConfig(
            enabled=retry.get('enabled', False),
            max_retries=retry.get('max_retries', 5),
        ),
    )

client = create_client_from_json("openrouter_config.json")
```