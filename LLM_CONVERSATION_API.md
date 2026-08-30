# OpenRouter Python Client - LLM-Style Conversation API

The OpenRouter Python client includes a simplified LLM-style API with conversation support, inspired by Simon Willison's llm library. This API provides an intuitive way to manage multi-turn conversations while automatically maintaining context.

## Installation & Setup

```python
from openrouter_client import OpenRouterClient
from openrouter_client.models.llm import get_model

# Initialize the client
client = OpenRouterClient(api_key="your-api-key")

# Get a model instance
model = get_model("openai/gpt-4o-mini", client)
```

## Basic Usage

### Single Prompt (No Conversation)
```python
# Simple one-off prompt
response = model.prompt("What's the capital of France?")
print(response)  # "The capital of France is Paris."

# With system prompt
response = model.prompt(
    "Translate this to Spanish: Hello world",
    system="You are a professional translator"
)
```

## Conversation API

### Creating a Conversation
```python
# Create a conversation with optional system prompt
conversation = model.conversation(
    system="You are a helpful assistant that remembers our entire conversation"
)
```

### Multi-turn Conversation
```python
# First turn
response1 = conversation.prompt("My name is Alice and I love Python programming")
print(response1)  # "Nice to meet you, Alice! Python is a great language..."

# Second turn - the model remembers context
response2 = conversation.prompt("What's my name?")
print(response2)  # "Your name is Alice."

# Third turn - continues building context
response3 = conversation.prompt("What did I say I love?")
print(response3)  # "You said you love Python programming."
```

## Advanced Features

### Structured Output with Conversations

**Important**: The return type depends on whether you provide a schema:
- **With schema**: Returns a Python `dict` (parsed JSON)
- **Without schema**: Returns a `string`

```python
# Define a JSON schema
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "occupation": {"type": "string"},
        "hobbies": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "occupation"]
}

# Get structured data in a conversation
conversation = model.conversation()

# WITH SCHEMA - Returns a dict
person_data = conversation.prompt(
    "Tell me about Albert Einstein",
    schema=person_schema
)
print(type(person_data))  # <class 'dict'>
print(person_data)  # {'name': 'Albert Einstein', 'age': 76, 'occupation': 'Theoretical Physicist', 'hobbies': ['violin', 'sailing', 'mathematics']}

# WITHOUT SCHEMA - Returns a string
follow_up = conversation.prompt("What was his most famous equation?")
print(type(follow_up))  # <class 'str'>
print(follow_up)  # "Einstein's most famous equation is E=mc²..."

# You can mix both types in the same conversation
# The return type is determined per prompt() call, not per conversation
```

### Additional Parameters
```python
# Pass any OpenRouter/OpenAI parameters through kwargs
conversation = model.conversation(system="You are a creative writer")

response = conversation.prompt(
    "Write a haiku about coding",
    temperature=0.9,  # Higher creativity
    max_tokens=100,
    top_p=0.95
)
```

### Cost & Usage Tracking

Every OpenRouter response carries token counts and the authoritative per-request
cost (in credits) inline — no request flag is needed. After each `prompt()` call,
that data is available on the model / conversation object. The return value of
`prompt()` is unchanged (still `str`, or `dict` when a schema is given); usage is
exposed as a side attribute.

```python
# Single prompt — usage from the most recent call lives on the model
model = get_model("openai/gpt-4o-mini", client)
reply = model.prompt("What's the capital of France?")   # -> str (unchanged)

print(model.last_usage.cost)          # e.g. 0.0000123  (credits charged)
print(model.last_usage.prompt_tokens, model.last_usage.completion_tokens)
print(model.last_usage.total_tokens)

# Richer breakdowns when the provider supplies them:
if model.last_usage.cost_details:
    print(model.last_usage.cost_details.upstream_inference_cost)  # BYOK only
if model.last_usage.prompt_tokens_details:
    print(model.last_usage.prompt_tokens_details.cached_tokens)
```

```python
# Conversations track each turn AND a running cumulative total
conversation = model.conversation()

conversation.prompt("Tell me a joke")
print(conversation.last_usage.cost)     # cost of just that turn

conversation.prompt("Explain why it's funny")
print(conversation.last_usage.cost)     # cost of the second turn only
print(conversation.total_cost)          # summed cost across all turns
print(conversation.total_usage.total_tokens)   # summed tokens across all turns
```

Notes:
- `last_usage` is a `Usage` model (`from openrouter_client import Usage`), or `None`
  before the first call / if a response carried no usage block.
- `cost` (and `cost_details`) may be `None` for some providers; `total_cost`
  returns `0.0` when no cost was reported, and token totals still accumulate.
- The running total carries summed token counts and cost only; per-turn detail
  breakdowns live on each turn's `last_usage`, not on `total_usage`.
- **BYOK accounts:** when the request runs on a bring-your-own-key provider
  (`usage.is_byok == True`), `usage.cost` is `0.0` — OpenRouter charges nothing —
  and the real per-request spend is in `usage.cost_details.upstream_inference_cost`.
  If you track spend, read `upstream_inference_cost` for BYOK requests.

### Tool Calling (automatic tool loop)

Pass a `ToolLoop` and the model's tool calls are executed for you, round after
round, until it produces a final answer. `tools` is what gets sent to the API;
`handlers` maps each tool's function name to the Python callable that runs it.

```python
from openrouter_client import ToolLoop
from openrouter_client.tools import build_chat_completion_tool

def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    return {"city": city, "temp_c": 21, "conditions": "sunny"}

loop = ToolLoop(
    tools=[build_chat_completion_tool(get_weather)],
    handlers={"get_weather": get_weather},
    max_rounds=8,   # optional; default 8
)

conversation = model.conversation()
answer = conversation.prompt("Should I bring a jacket in Paris?", tool_loop=loop)
```

Each handler is called as `handler(**arguments)` with the arguments the model
emitted. It may return a string, any JSON-serializable value, or a Pydantic
model — the result is serialized into the `role="tool"` message automatically.

`tool_loop` works the same way on a one-off `model.prompt(...)`.

**Conversation history stays complete.** Every assistant turn is recorded *with*
its `tool_calls`, and every tool result is appended as a `role="tool"` message,
so `conversation.messages` remains a valid, reusable transcript:

```python
[m["role"] for m in conversation.messages]
# ['user', 'assistant', 'tool', 'assistant']
```

**A failed turn rolls the history back.** If a tool loop raises — including when
the final structured answer fails to parse — `messages` is restored to exactly
where it stood before `prompt()` was called, *this turn's own user message
included*. You never keep an assistant turn whose `tool_calls` have no matching
results (a shape most providers reject), and retrying on the same conversation
never stacks duplicate user turns:

```python
try:
    answer = conversation.prompt("...", tool_loop=loop)
except ToolExecutionError:
    answer = conversation.prompt("...", tool_loop=loop)   # clean history
```

The non-tool-loop path keeps its user message on failure, but likewise records
no usage and keeps no unparseable assistant message when a schema parse fails —
`last_usage` / `total_usage` only ever reflect successful turns.

**Usage covers the whole turn.** A single `prompt()` with a tool loop makes
several API calls; `last_usage` covers all of them, and a conversation's
`total_usage` / `total_cost` accumulate normally. When a turn made exactly one
call, that call's `Usage` is passed through untouched, so the per-request
breakdowns (`cost_details`, `is_byok`) survive — which matters for BYOK spend
tracking. Genuine multi-call turns report summed tokens and cost only, since the
breakdowns don't sum meaningfully.

**`max_rounds` bounds tool execution.** With `max_rounds=3` the handlers run at
most three times; the loop may make one further API call to discover the model
is *still* asking for tools, but it raises `ToolCallLimitExceeded` without
executing that batch. Handlers with side effects never run past the budget.

#### Tools together with `schema`

The two are kept deliberately apart, in this order:

1. **Tool rounds** are sent with `tools` attached and **no** `response_format`,
   so tool calling is never subject to a provider constraining output to a JSON
   grammar. Note that OpenAI supports the combined form (see *Verified
   behavior* below) — the separation is portability insurance, not a workaround
   for a limitation observed there.
2. Once the model stops requesting tools, **one final call** is made with
   `response_format` attached, the tool definitions still included, and
   `tool_choice="none"`. The definitions travel because the transcript is full
   of tool calls and tool results, and Anthropic's API requires a request
   carrying those blocks to define the tools; `"none"` stops the model opening
   another round. That response is parsed and validated, and is what `prompt()`
   returns.

That final call appends a short `role="user"` instruction asking for the
structured answer, so you will see one extra user turn in `conversation.messages`
for a schema'd tool turn. It is there for a reason: the tool rounds end on an
assistant message, and a request whose last message is from the assistant reads
as an *assistant prefill* to Anthropic-family models — they continue the previous
prose instead of emitting a fresh object, and some providers reject prefill
combined with `response_format` outright.

```python
report = conversation.prompt(
    "Design a strategy for this pair.",
    schema=strategy_schema,
    tool_loop=loop,
)
# -> dict, guaranteed, after the model has used tools freely
```

The cost of this design is one extra completion per `prompt()` compared to
`schema` alone. In exchange the behavior is deterministic and does not depend on
whether a given provider supports tool calls and structured output in the same
request. Using `schema` without `tool_loop` is unchanged — still exactly one call.

#### Verified behavior

The provider claims above are load-bearing, so they were checked against a live
API on **2026-08-30** (OpenAI `gpt-4o-mini`, driving this library at
`base_url="https://api.openai.com/v1"`):

| Claim | Result |
|---|---|
| Tool loop executes handlers and returns an answer | works |
| Schema turn (`tools` + `tool_choice="none"` + `response_format`) is accepted | works |
| `tool_choice="required"` forces a call even when told not to use tools | **confirmed** — this is why it is relaxed after round one |
| `tool_choice` with no `tools` | rejected, HTTP 400 |
| `parallel_tool_calls` with no `tools` | rejected, HTTP 400 |
| A transcript with tool calls, sent with no `tools` defined | **accepted by OpenAI** |
| `tools` + strict `json_schema` in one request | **accepted, and tool calls are still emitted** |

Two things follow, stated plainly:

- **OpenAI does not need the two-phase split.** It happily combines `tools` with
  a strict schema and still calls tools, so the extra completion buys
  portability, not correctness, on that provider.
- **The Anthropic-specific claims here are not verified.** Anthropic's documented
  requirement that tool-carrying transcripts define their tools is the reason
  the definitions are re-sent, but no Anthropic-routed model was exercised.
  Treat those two rows as reasoned-but-untested.

#### Errors

| Situation | Raised |
|---|---|
| Model calls a tool with no registered handler | `ToolExecutionError` |
| Model sends arguments that aren't a JSON object | `ToolExecutionError` |
| A handler raises | `ToolExecutionError` (original on `.original_error`) |
| Model still calling tools after `max_rounds` | `ToolCallLimitExceeded` |
| Response the client could not parse, or carrying no choices | `APIError` (502) |
| `tools`, `response_format`, or `stream=True` passed alongside `tool_loop` | `ValueError` |

`tool_choice` *is* accepted, with two adjustments. It is honored on the **first
round only**: a `"required"` (or named-function) choice forced on every round
would make the loop's exit condition — a response with no tool calls —
unsatisfiable, so after round one the provider default (`"auto"`) applies and
the model can settle. On the final schema call your `tool_choice` is replaced
with `"none"` and `parallel_tool_calls` is dropped as moot.

**Follow-up turns keep working.** After a tool-loop turn that actually recorded
tool calls in the transcript, the conversation remembers the loop's tool
definitions (a copy, not your list) and re-sends them with `tool_choice="none"`
on later tool-free `prompt()` calls. Anthropic's API requires a request carrying
tool-use or tool-result blocks to define the tools; OpenAI accepts it either way
(see *Verified behavior*), so this is portability insurance. A turn where the model never
called a tool retains nothing — no schemas are re-billed for it. `clear()`
forgets the retained definitions along with the history; an explicit non-None
`tools=` or `tool_choice=` from you wins. Passing your own `tools=` with
`tool_choice="none"` this way does not trigger the missing-`tool_loop` warning.

**A different `ToolLoop` on a later turn is merged, not swapped.** The history
still references the earlier loop's tools, so its definitions (and handlers)
are combined with the new loop's for that turn; a same-named tool takes the
newer definition.

**Return values are text.** If a provider returns the final answer as a list of
content parts rather than a string, `prompt()` joins the text parts; if it
returns no content at all (reasoning-only output, content filter), you get `""`
rather than `None`. The documented `str` (or schema `dict`) contract always
holds; the raw form is kept in `conversation.messages`.

These live in `openrouter_client.exceptions`. Tool failures are raised rather
than fed back to the model — if you want the model to see an error and recover,
catch it inside your handler and return the message as the tool's result:

```python
def get_weather(city: str) -> dict:
    try:
        return fetch(city)
    except UpstreamError as e:
        return {"error": str(e)}   # the model reads this and can try again
```

> **Note:** passing `tools=` without `tool_loop=` sends the tools to the API but
> nothing executes the resulting calls, and the response content is typically
> empty. That case now emits a `UserWarning`; use `tool_loop` to run them.

### Conversation Management
```python
conversation = model.conversation()

# Add multiple exchanges
conversation.prompt("Let's talk about space")
conversation.prompt("What's the largest planet?")
conversation.prompt("How many moons does it have?")

# Check message count
count = conversation.get_message_count()
print(f"Conversation has {count} messages")  # Includes system, user, and assistant messages

# Access full conversation history
print("Full conversation history:")
for i, message in enumerate(conversation.messages):
    role = message["role"]
    content = message["content"]
    
    # Handle content that might be a list (for user messages with attachments)
    if isinstance(content, list):
        # Extract text from the first text part
        text_content = next((part["text"] for part in content if part.get("type") == "text"), str(content))
    else:
        text_content = content
    
    print(f"{i+1}. {role.upper()}: {text_content}")

# Clear conversation history (keeps system prompt if present)
conversation.clear()
print(f"After clear: {conversation.get_message_count()} messages")
```

## Key Points

1. **Automatic Context Management**: The conversation object automatically maintains the full message history, sending it with each request
2. **Stateful Conversations**: Each conversation object is independent - you can have multiple conversations running simultaneously
3. **Return Type Behavior**: 
   - When `schema` parameter is provided → Returns a Python `dict` (parsed JSON)
   - When no `schema` parameter → Returns a `string`
   - The return type is determined per `prompt()` call, not per conversation
4. **Conversation History Access**: You can access the full conversation history via `conversation.messages` - this contains the complete OpenAI chat format with system, user, and assistant messages
5. **Parameter Passthrough**: All standard OpenRouter/OpenAI parameters can be passed via kwargs
6. **System Prompt Persistence**: System prompts are maintained even when clearing conversation history
7. **Cost & Usage Tracking**: After each `prompt()`, token counts and the per-request cost are on `.last_usage`; conversations also expose cumulative `.total_cost` and `.total_usage`
8. **Tool Calling**: Pass `tool_loop=ToolLoop(tools=..., handlers=...)` to have tool calls executed automatically. With `schema`, tool rounds run unconstrained and the schema is enforced on one final tools-free call

## Example Use Cases

```python
# Customer support bot
support_bot = model.conversation(
    system="You are a helpful customer support agent for an e-commerce platform"
)
support_bot.prompt("I need help with my order #12345")
support_bot.prompt("It hasn't arrived yet")
support_bot.prompt("Can you check the status?")

# Code review assistant
code_reviewer = model.conversation(
    system="You are an expert code reviewer focusing on Python best practices"
)
code_reviewer.prompt("Review this function: def add(a,b): return a+b")
code_reviewer.prompt("How can I make it more robust?")

# Data extraction with structured output
extractor = model.conversation()
schema = {"type": "object", "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}}}
result = extractor.prompt("I absolutely love this product!", schema=schema)
print(result["sentiment"])  # "positive"
```

The conversation API makes it easy to build applications that need to maintain context across multiple interactions, from chatbots to complex multi-step workflows.