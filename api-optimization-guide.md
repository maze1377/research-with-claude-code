# API Optimization Guide
## Practical Techniques for OpenAI and Anthropic Models (2024-2025)

**Purpose:** Actionable best practices for maximizing performance, minimizing costs, and ensuring reliability when using OpenAI and Anthropic APIs for multi-agent systems.

**Last Updated:** 2025-11-08

---

## Table of Contents

1. [Model Selection Strategy](#model-selection-strategy)
2. [OpenAI API Best Practices](#openai-api-best-practices)
3. [Anthropic API Best Practices](#anthropic-api-best-practices)
4. [Prompt Engineering Techniques](#prompt-engineering-techniques)
5. [Function Calling Optimization](#function-calling-optimization)
6. [Cost Optimization Strategies](#cost-optimization-strategies)
7. [Latency Optimization](#latency-optimization)
8. [Error Handling and Reliability](#error-handling-and-reliability)
9. [Production Monitoring](#production-monitoring)

---

## Model Selection Strategy

### OpenAI Model Lineup (2024-2025)

| Model | Best For | Input Cost | Output Cost | Context Window | Key Features |
|-------|----------|------------|-------------|----------------|--------------|
| gpt-4o | General tasks, balanced | $2.50/1M | $10/1M | 128K | Fast, multimodal, function calling |
| gpt-4o-2024-08-06 | Structured outputs | $2.50/1M | $10/1M | 128K | 100% schema adherence |
| gpt-4o-mini | High-volume, simple | $0.15/1M | $0.60/1M | 128K | 80% cheaper, fast |
| o1-preview | Complex reasoning | $15/1M | $60/1M | 128K | Extended thinking |
| o1-mini | STEM reasoning | $3/1M | $12/1M | 128K | Faster reasoning |

### Anthropic Model Lineup (2024-2025)

| Model | Best For | Input Cost | Output Cost | Context Window | Key Features |
|-------|----------|------------|-------------|----------------|--------------|
| Claude Sonnet 4.5 | Balanced performance | $3/1M | $15/1M | 200K | Fast, capable |
| Claude 3.7 Sonnet | Extended thinking | $3/1M | $15/1M | 200K | Serial test-time compute |
| Claude Opus 4 | Complex tasks | $15/1M | $75/1M | 200K | Highest capability |
| Claude Haiku 4 | Speed/volume | $0.25/1M | $1.25/1M | 200K | Fast, economical |

### Selection Decision Tree

```
Need complex reasoning?
├─ Yes → Use reasoning models (o1, Claude Opus 4, Claude 3.7 Extended)
└─ No
   ├─ Need strict schema adherence?
   │  └─ Yes → gpt-4o-2024-08-06
   └─ High-volume, simple tasks?
      ├─ Yes → gpt-4o-mini or Claude Haiku
      └─ No → gpt-4o or Claude Sonnet 4.5
```

### Multi-Agent Model Assignment Strategy

**Heterogeneous Approach (Recommended):**
```python
agents = {
    "router": "gpt-4o-mini",           # Fast, cheap routing
    "researcher": "claude-sonnet-4.5",  # Deep research
    "analyst": "gpt-4o",                # Balanced analysis
    "critic": "claude-opus-4",          # High-quality evaluation
    "writer": "claude-sonnet-4.5",      # Long-form writing
    "validator": "gpt-4o-2024-08-06"    # Structured validation
}
```

**Cost Optimization:**
- Use cheaper models for routing/coordination
- Use expensive models only for critical steps
- Cache results from expensive models when possible

---

## OpenAI API Best Practices

### 1. Structured Outputs (2024)

**Always use `strict: true` for guaranteed schema adherence:**

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

# Define schema with Pydantic
class ResearchOutput(BaseModel):
    summary: str
    key_findings: list[str]
    confidence_score: float
    sources: list[str]

# Use structured outputs
response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "You are a research analyst."},
        {"role": "user", "content": "Analyze this topic: {topic}"}
    ],
    response_format=ResearchOutput
)

# Guaranteed to match schema
output = response.choices[0].message.parsed
```

**Benefits:**
- 100% schema adherence (vs ~40% with JSON mode on older models)
- No retry logic needed
- Type-safe outputs
- Eliminates validation overhead

**Limitations:**
- max_tokens cutoff can cause incomplete responses
- Model can still hallucinate values (validate content separately)
- Not all JSON schemas supported (check documentation)

### 2. Function Calling Optimization

**Problem: Too many tools reduce accuracy**

```python
# BAD: Passing all 50 tools
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=all_tools  # 50+ tools → poor selection accuracy
)

# GOOD: Use RAG to select relevant tools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def select_relevant_tools(query: str, all_tools: list, top_k: int = 5):
    """Select most relevant tools using embeddings"""
    # Embed query
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Embed tool descriptions
    tool_embeddings = []
    for tool in all_tools:
        desc = f"{tool['function']['name']}: {tool['function']['description']}"
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=desc
        ).data[0].embedding
        tool_embeddings.append(emb)

    # Compute similarities
    similarities = cosine_similarity(
        [query_embedding],
        tool_embeddings
    )[0]

    # Select top-k
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [all_tools[i] for i in top_indices]

# Use only relevant tools
relevant_tools = select_relevant_tools(user_query, all_tools, top_k=5)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=relevant_tools  # Only 5 tools → better accuracy
)
```

**Tool Description Best Practices:**

```python
# BAD: Vague description
{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Searches for information"
    }
}

# GOOD: Specific, detailed description
{
    "type": "function",
    "function": {
        "name": "search_scientific_papers",
        "description": """Search for peer-reviewed scientific papers in academic databases.
        Use this when the user asks for research papers, academic studies, or scientific evidence.
        Do NOT use for general web searches or news articles.
        Returns: List of papers with titles, authors, abstracts, and DOIs.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. Use academic keywords and field-specific terminology."
                },
                "field": {
                    "type": "string",
                    "enum": ["computer_science", "biology", "physics", "chemistry"],
                    "description": "Academic field to search within"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of papers to return (1-20)",
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["query", "field"],
            "additionalProperties": False
        }
    }
}
```

### 3. Parallel Function Calling

```python
# Call multiple independent tools in one request
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What's the weather in NYC and stock price of AAPL?"}
    ],
    tools=[weather_tool, stock_tool],
    parallel_tool_calls=True  # Enable parallel calling
)

# Process multiple tool calls
tool_calls = response.choices[0].message.tool_calls
for tool_call in tool_calls:
    if tool_call.function.name == "get_weather":
        result = get_weather(**json.loads(tool_call.function.arguments))
    elif tool_call.function.name == "get_stock_price":
        result = get_stock_price(**json.loads(tool_call.function.arguments))
```

**Benefits:**
- Faster execution (parallel instead of sequential)
- Fewer API calls
- Lower latency

### 4. Response Streaming for Better UX

```python
# Stream responses for long outputs
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 5. Prompt Caching (Not yet available for GPT-4o)

**Note:** OpenAI doesn't currently support prompt caching like Anthropic. Use workarounds:

```python
# Cache embeddings for repeated content
from functools import lru_cache

@lru_cache(maxsize=100)
def get_embedding(text: str):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# Cache common system prompts in your application layer
```

---

## Anthropic API Best Practices

### 1. Prompt Caching (Major Cost Saver)

**Anthropic offers automatic prompt caching for repeated content:**

```python
import anthropic

client = anthropic.Anthropic()

# Large context that will be reused
codebase_context = """
[100K tokens of codebase documentation]
"""

# First call: Pay full price
response1 = client.messages.create(
    model="claude-sonnet-4.5",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a code assistant.",
        },
        {
            "type": "text",
            "text": codebase_context,
            "cache_control": {"type": "ephemeral"}  # Cache this
        }
    ],
    messages=[
        {"role": "user", "content": "Explain this function: foo()"}
    ]
)

# Subsequent calls within 5 minutes: ~90% cheaper for cached content
response2 = client.messages.create(
    model="claude-sonnet-4.5",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a code assistant.",
        },
        {
            "type": "text",
            "text": codebase_context,
            "cache_control": {"type": "ephemeral"}  # Hits cache
        }
    ],
    messages=[
        {"role": "user", "content": "Explain this function: bar()"}
    ]
)
```

**Pricing Impact:**
- Regular input: $3/1M tokens
- Cached input: $0.30/1M tokens (90% discount)
- Cache write: $3.75/1M tokens (one-time)
- Cache lifetime: 5 minutes

**Best Practices:**
- Cache large system prompts
- Cache codebase/document context
- Cache few-shot examples
- Place cache_control on last eligible block
- Minimum 1024 tokens to cache (smaller not cached)

### 2. Extended Thinking (Claude 3.7+)

```python
# Enable extended thinking for complex reasoning
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # Allocate tokens for thinking
    },
    messages=[
        {
            "role": "user",
            "content": "Solve this complex math problem: ..."
        }
    ]
)

# Access thinking process (for debugging/transparency)
for block in response.content:
    if block.type == "thinking":
        print(f"Thinking: {block.thinking}")
    elif block.type == "text":
        print(f"Answer: {block.text}")
```

**When to Use:**
- Complex mathematical problems
- Multi-step reasoning tasks
- Code debugging and optimization
- Strategic planning

**Cost Consideration:**
- Thinking tokens count toward input pricing
- Can significantly increase cost (2-5x)
- Use only when complexity justifies cost

### 3. Tool Use Best Practices

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location. Use this when user asks about current weather, temperature, or conditions. Returns temperature, conditions, and humidity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or zip code"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
]

response = client.messages.create(
    model="claude-sonnet-4.5",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}]
)

# Handle tool use
if response.stop_reason == "tool_use":
    tool_use_block = next(
        block for block in response.content if block.type == "tool_use"
    )

    # Execute tool
    tool_result = execute_tool(
        tool_use_block.name,
        tool_use_block.input
    )

    # Continue conversation with result
    response = client.messages.create(
        model="claude-sonnet-4.5",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"},
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": str(tool_result)
                    }
                ]
            }
        ]
    )
```

### 4. System Prompt Best Practices

**Anthropic's recommendations based on Claude's own system prompt evolution:**

```python
# Structure: Role → Task → Context → Examples → Constraints

system_prompt = """You are an expert Python developer with 10 years of experience.

Your task is to review code for bugs, security vulnerabilities, and performance issues.

Context:
- This is a production codebase for a financial application
- Security and correctness are paramount
- Performance should be optimized for high-volume transactions

Examples:
<example>
Bad code:
```python
password = input("Enter password: ")
if password == "admin123":
    grant_access()
```

Issues:
- Hardcoded password (security vulnerability)
- Plain text comparison (no hashing)
- No rate limiting

Good code:
```python
import bcrypt
from ratelimit import limits

@limits(calls=3, period=60)
def check_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)
```
</example>

Constraints:
- Always explain WHY code is problematic, not just WHAT is wrong
- Provide specific code fixes, not just general advice
- Prioritize security > correctness > performance
- Use Python 3.11+ features when beneficial
"""

response = client.messages.create(
    model="claude-sonnet-4.5",
    max_tokens=2048,
    system=system_prompt,
    messages=[{"role": "user", "content": "Review this code: ..."}]
)
```

### 5. Message Batching for Cost Efficiency

```python
# Bad: Multiple separate calls
for question in questions:
    response = client.messages.create(
        model="claude-sonnet-4.5",
        max_tokens=1024,
        messages=[{"role": "user", "content": question}]
    )
    # Cost: Full API call × len(questions)

# Good: Batch in single call
batched_questions = "\n\n".join([
    f"Question {i+1}: {q}" for i, q in enumerate(questions)
])

response = client.messages.create(
    model="claude-sonnet-4.5",
    max_tokens=4096,
    messages=[
        {
            "role": "user",
            "content": f"Answer each question separately:\n\n{batched_questions}"
        }
    ]
)
# Cost: 1 API call (lower overhead)
```

---

## Prompt Engineering Techniques

### 1. Few-Shot Examples

```python
# Zero-shot (no examples)
prompt = "Classify sentiment: 'This movie was terrible.'"

# Few-shot (with examples) - Much better performance
prompt = """Classify the sentiment of the given text as positive, negative, or neutral.

Examples:
Text: "I loved this restaurant! Best meal ever."
Sentiment: positive

Text: "The service was okay, nothing special."
Sentiment: neutral

Text: "Worst experience of my life. Never going back."
Sentiment: negative

Now classify:
Text: "This movie was terrible."
Sentiment:"""
```

**Best Practices:**
- 3-5 examples optimal (more doesn't always help)
- Examples should cover edge cases
- Match example format to desired output format
- Use diverse examples

### 2. Chain-of-Thought Prompting

```python
# Without CoT
prompt = "If a train travels 120 miles in 2 hours, how far will it travel in 5 hours?"

# With CoT (better for complex reasoning)
prompt = """If a train travels 120 miles in 2 hours, how far will it travel in 5 hours?

Let's solve this step by step:
1. First, calculate the train's speed
2. Then, use that speed to find the distance for 5 hours
3. Finally, provide the answer

Please show your work:"""
```

**Variants:**
```python
# Zero-shot CoT
prompt = f"{question}\n\nLet's think step by step:"

# Self-consistency CoT (sample multiple times, majority vote)
responses = []
for _ in range(5):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7  # Higher temperature for diversity
    )
    responses.append(extract_answer(response))

final_answer = most_common(responses)
```

### 3. Role Prompting

```python
# Generic (weaker)
prompt = "Explain quantum computing."

# Role-specific (stronger)
prompt = """You are a quantum computing researcher with a PhD from MIT and 15 years of experience.
You specialize in explaining complex topics to technical audiences.

Explain quantum computing to a software engineer who understands classical computing well
but has no background in quantum mechanics. Focus on practical implications for cryptography."""
```

### 4. Constraint Specification

```python
# Vague (unreliable output)
prompt = "Write a product description."

# Constrained (reliable output)
prompt = """Write a product description with these exact constraints:
- Length: Exactly 3 paragraphs
- Tone: Professional but friendly
- Include: 2-3 key features, 1 benefit, 1 call-to-action
- Avoid: Technical jargon, superlatives ("best", "perfect")
- Format: HTML with <p> tags
- Target audience: Small business owners

Product: Cloud backup service"""
```

### 5. Output Format Specification

```python
# For Anthropic (with structured thinking)
prompt = """Analyze this data and provide output in this exact XML format:

<analysis>
  <summary>One paragraph summary</summary>
  <key_findings>
    <finding>First finding</finding>
    <finding>Second finding</finding>
  </key_findings>
  <recommendations>
    <recommendation priority="high">High priority recommendation</recommendation>
    <recommendation priority="medium">Medium priority recommendation</recommendation>
  </recommendations>
</analysis>

Data: {data}"""

# For OpenAI (use structured outputs)
class Analysis(BaseModel):
    summary: str
    key_findings: list[str]
    recommendations: list[dict[str, str]]  # [{"priority": "high", "text": "..."}]

response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": f"Analyze: {data}"}],
    response_format=Analysis
)
```

---

## Function Calling Optimization

### 1. Validation Before Execution

```python
def safe_execute_tool(tool_name: str, arguments: dict) -> any:
    """Validate and execute tools safely"""

    # 1. Validate tool exists
    if tool_name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool = AVAILABLE_TOOLS[tool_name]

    # 2. Validate required parameters
    required = tool.get("required_params", [])
    missing = set(required) - set(arguments.keys())
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")

    # 3. Validate parameter types
    for param, value in arguments.items():
        expected_type = tool["param_types"].get(param)
        if expected_type and not isinstance(value, expected_type):
            raise TypeError(f"Parameter {param} should be {expected_type}, got {type(value)}")

    # 4. Validate parameter values
    for param, value in arguments.items():
        validators = tool.get("validators", {}).get(param, [])
        for validator in validators:
            if not validator(value):
                raise ValueError(f"Invalid value for {param}: {value}")

    # 5. Execute with timeout
    try:
        with timeout(tool.get("timeout", 30)):
            return tool["function"](**arguments)
    except TimeoutError:
        raise TimeoutError(f"Tool {tool_name} exceeded timeout")
```

### 2. Retry Logic for Failed Tool Calls

```python
def execute_with_retry(client, messages, tools, max_retries=3):
    """Retry tool calls with error feedback"""

    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools
        )

        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]

            try:
                # Try to execute
                result = safe_execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                return result

            except Exception as e:
                # Add error to conversation
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error: {str(e)}. Please try again with corrected parameters."
                })
                # Will retry in next iteration
        else:
            # No tool call, return response
            return response.choices[0].message.content

    raise Exception(f"Failed after {max_retries} attempts")
```

### 3. Tool Result Formatting

```python
def format_tool_result(result: any, tool_name: str) -> str:
    """Format tool results for optimal LLM consumption"""

    # Bad: Raw JSON dump
    # return json.dumps(result)

    # Good: Structured, human-readable format
    if tool_name == "search_papers":
        formatted = "Found papers:\n\n"
        for i, paper in enumerate(result["papers"], 1):
            formatted += f"{i}. {paper['title']}\n"
            formatted += f"   Authors: {', '.join(paper['authors'])}\n"
            formatted += f"   Year: {paper['year']}\n"
            formatted += f"   Abstract: {paper['abstract'][:200]}...\n"
            formatted += f"   DOI: {paper['doi']}\n\n"
        return formatted

    elif tool_name == "get_weather":
        return f"Weather in {result['location']}:\n" \
               f"Temperature: {result['temp']}°{result['unit']}\n" \
               f"Conditions: {result['conditions']}\n" \
               f"Humidity: {result['humidity']}%"

    # Default: Pretty-printed JSON
    return json.dumps(result, indent=2)
```

---

## Cost Optimization Strategies

### 1. Token Usage Monitoring

```python
class TokenTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0

    def track_openai_call(self, response, model="gpt-4o"):
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # Pricing (update with current rates)
        pricing = {
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10 / 1_000_000},
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        }

        cost = (input_tokens * pricing[model]["input"] +
                output_tokens * pricing[model]["output"])

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "cumulative_cost": self.total_cost
        }

tracker = TokenTracker()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
)

usage = tracker.track_openai_call(response, "gpt-4o")
print(f"This call cost: ${usage['cost']:.4f}")
print(f"Total cost so far: ${usage['cumulative_cost']:.4f}")
```

### 2. Model Cascading

```python
def cascading_inference(query: str, complexity_threshold: float = 0.7):
    """Try cheap model first, escalate to expensive if needed"""

    # Step 1: Assess complexity with cheap model
    complexity_check = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Rate the complexity of this query on a scale of 0-1:
            0 = very simple, factual question
            1 = very complex, requires deep reasoning

            Query: {query}

            Respond with only a number between 0 and 1."""
        }],
        max_tokens=10
    )

    complexity = float(complexity_check.choices[0].message.content.strip())

    # Step 2: Route to appropriate model
    if complexity < complexity_threshold:
        # Use cheap model
        model = "gpt-4o-mini"
        print(f"Using gpt-4o-mini (complexity: {complexity})")
    else:
        # Use expensive model
        model = "gpt-4o"
        print(f"Using gpt-4o (complexity: {complexity})")

    # Step 3: Execute with selected model
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}]
    )

    return response.choices[0].message.content
```

### 3. Response Caching Strategy

```python
import hashlib
import redis

class ResponseCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 3600  # 1 hour cache

    def get_cache_key(self, model: str, messages: list, temperature: float) -> str:
        """Generate cache key from request parameters"""
        cache_input = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        content = json.dumps(cache_input, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get_cached_response(self, model, messages, temperature):
        """Check cache before API call"""
        # Only cache deterministic responses (temperature=0)
        if temperature > 0:
            return None

        key = self.get_cache_key(model, messages, temperature)
        cached = self.redis.get(key)

        if cached:
            print("Cache hit! Saving API call.")
            return json.loads(cached)

        return None

    def cache_response(self, model, messages, temperature, response):
        """Cache response for future use"""
        if temperature > 0:
            return  # Don't cache non-deterministic responses

        key = self.get_cache_key(model, messages, temperature)
        self.redis.setex(
            key,
            self.ttl,
            json.dumps(response)
        )

# Usage
cache = ResponseCache(redis.Redis())

cached = cache.get_cached_response("gpt-4o", messages, temperature=0)
if cached:
    response = cached
else:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0
    )
    cache.cache_response("gpt-4o", messages, 0, response)
```

### 4. Prompt Compression

```python
def compress_prompt(long_context: str, query: str, max_tokens: int = 2000):
    """Use embedding-based retrieval to compress context"""

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Split long context into chunks
    chunks = [sent for sent in long_context.split('.') if sent.strip()]

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    query_vector = vectorizer.transform([query])

    # Calculate relevance scores
    scores = cosine_similarity(query_vector, chunk_vectors)[0]

    # Select top chunks
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    compressed = []
    token_count = 0
    for idx in top_indices:
        chunk_tokens = len(chunks[idx].split()) * 1.3  # Rough estimate
        if token_count + chunk_tokens > max_tokens:
            break
        compressed.append(chunks[idx])
        token_count += chunk_tokens

    return '. '.join(compressed) + '.'

# Use compressed context
full_context = """[50,000 tokens of documentation]"""
compressed_context = compress_prompt(full_context, user_query, max_tokens=2000)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": compressed_context},
        {"role": "user", "content": user_query}
    ]
)
```

---

## Latency Optimization

### 1. Streaming Responses

```python
# Reduce perceived latency with streaming
def stream_response(messages):
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    return full_response
```

### 2. Parallel API Calls

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def process_multiple_queries(queries: list[str]):
    """Process multiple queries in parallel"""

    tasks = []
    for query in queries:
        task = async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}]
        )
        tasks.append(task)

    # Execute all in parallel
    responses = await asyncio.gather(*tasks)

    return [r.choices[0].message.content for r in responses]

# Usage
queries = ["What is 2+2?", "What is the capital of France?", "Explain Python"]
results = asyncio.run(process_multiple_queries(queries))
```

### 3. Request Batching

```python
# For independent queries, batch them
def batch_queries(queries: list[str], batch_size: int = 10):
    """Batch multiple queries into fewer API calls"""

    batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
    results = []

    for batch in batches:
        # Combine queries
        combined_prompt = "Answer each question briefly:\n\n"
        for i, q in enumerate(batch, 1):
            combined_prompt += f"Q{i}: {q}\n"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": combined_prompt}]
        )

        # Parse responses (assumes numbered format)
        batch_results = parse_batch_response(response.choices[0].message.content, len(batch))
        results.extend(batch_results)

    return results
```

### 4. Reduce max_tokens

```python
# Don't request more tokens than you need
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    max_tokens=500  # Instead of default 4096 if you only need short response
)
# Faster response, lower cost
```

---

## Error Handling and Reliability

### 1. Exponential Backoff Retry

```python
import time
import random
from openai import RateLimitError, APIError

def api_call_with_retry(client, max_retries=5, **kwargs):
    """Retry API calls with exponential backoff"""

    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise

            # Exponential backoff: 1s, 2s, 4s, 8s, 16s
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limited. Retrying in {wait_time:.2f}s...")
            time.sleep(wait_time)

        except APIError as e:
            if attempt == max_retries - 1:
                raise

            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"API error: {e}. Retrying in {wait_time:.2f}s...")
            time.sleep(wait_time)
```

### 2. Fallback Models

```python
def resilient_inference(messages, preferred_model="gpt-4o"):
    """Try multiple models with fallback"""

    models = [
        preferred_model,
        "gpt-4o-mini",  # Cheaper fallback
        "gpt-3.5-turbo"  # Last resort
    ]

    for model in models:
        try:
            response = api_call_with_retry(
                client,
                model=model,
                messages=messages,
                max_retries=3
            )
            return response
        except Exception as e:
            print(f"{model} failed: {e}")
            if model == models[-1]:
                raise
            print(f"Falling back to {models[models.index(model) + 1]}")

    raise Exception("All models failed")
```

### 3. Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                print(f"Circuit breaker opened after {self.failure_count} failures")

            raise

# Usage
breaker = CircuitBreaker(failure_threshold=5, timeout=60)

def make_api_call():
    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

try:
    response = breaker.call(make_api_call)
except Exception as e:
    print(f"Call failed or circuit open: {e}")
```

### 4. Timeout Management

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def api_call_with_timeout(client, timeout_seconds=30, **kwargs):
    """Enforce timeout on API calls"""

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            client.chat.completions.create,
            **kwargs
        )

        try:
            return future.result(timeout=timeout_seconds)
        except TimeoutError:
            future.cancel()
            raise TimeoutError(f"API call exceeded {timeout_seconds}s timeout")
```

---

## Production Monitoring

### 1. Metrics Collection

```python
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class APICallMetrics:
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: float
    success: bool
    error: str = None

class MetricsCollector:
    def __init__(self):
        self.metrics = []
        self.logger = logging.getLogger(__name__)

    def log_call(self, metric: APICallMetrics):
        self.metrics.append(metric)

        # Log to structured logging system
        self.logger.info(
            "API call completed",
            extra={
                "model": metric.model,
                "tokens": metric.input_tokens + metric.output_tokens,
                "latency_ms": metric.latency_ms,
                "cost": metric.cost,
                "success": metric.success
            }
        )

        # Alert on anomalies
        if metric.latency_ms > 10000:  # > 10s
            self.logger.warning(f"High latency detected: {metric.latency_ms}ms")

        if metric.cost > 1.0:  # > $1 per call
            self.logger.warning(f"High cost detected: ${metric.cost:.4f}")

    def get_summary(self, last_n_minutes: int = 60):
        """Get summary of recent calls"""
        cutoff = datetime.now() - timedelta(minutes=last_n_minutes)
        recent = [m for m in self.metrics if m.timestamp > cutoff]

        return {
            "total_calls": len(recent),
            "successful": sum(1 for m in recent if m.success),
            "failed": sum(1 for m in recent if not m.success),
            "total_cost": sum(m.cost for m in recent),
            "avg_latency": sum(m.latency_ms for m in recent) / len(recent) if recent else 0,
            "total_tokens": sum(m.input_tokens + m.output_tokens for m in recent)
        }

# Usage
metrics = MetricsCollector()

start = time.time()
try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    latency = (time.time() - start) * 1000

    metrics.log_call(APICallMetrics(
        timestamp=datetime.now(),
        model="gpt-4o",
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        latency_ms=latency,
        cost=calculate_cost(response.usage),
        success=True
    ))
except Exception as e:
    latency = (time.time() - start) * 1000
    metrics.log_call(APICallMetrics(
        timestamp=datetime.now(),
        model="gpt-4o",
        input_tokens=0,
        output_tokens=0,
        latency_ms=latency,
        cost=0,
        success=False,
        error=str(e)
    ))
```

### 2. Rate Limit Management

```python
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = deque()

    def acquire(self):
        """Wait if necessary to respect rate limit"""
        now = time.time()

        # Remove requests older than 1 minute
        while self.requests and self.requests[0] < now - 60:
            self.requests.popleft()

        # If at limit, wait
        if len(self.requests) >= self.max_requests:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                print(f"Rate limit reached. Waiting {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                return self.acquire()  # Recursively try again

        # Record this request
        self.requests.append(now)

# Usage
limiter = RateLimiter(max_requests_per_minute=60)

for query in large_query_list:
    limiter.acquire()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}]
    )
```

### 3. Cost Budgeting

```python
class CostBudget:
    def __init__(self, daily_budget: float):
        self.daily_budget = daily_budget
        self.today = datetime.now().date()
        self.today_cost = 0

    def check_budget(self, estimated_cost: float) -> bool:
        """Check if request would exceed budget"""
        # Reset if new day
        if datetime.now().date() != self.today:
            self.today = datetime.now().date()
            self.today_cost = 0

        if self.today_cost + estimated_cost > self.daily_budget:
            print(f"Budget exceeded! Today: ${self.today_cost:.2f}, Budget: ${self.daily_budget:.2f}")
            return False

        return True

    def record_cost(self, actual_cost: float):
        self.today_cost += actual_cost

# Usage
budget = CostBudget(daily_budget=100.0)

# Estimate cost before calling
estimated_tokens = len(prompt.split()) * 1.3
estimated_cost = estimate_cost("gpt-4o", estimated_tokens)

if budget.check_budget(estimated_cost):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    actual_cost = calculate_cost(response.usage)
    budget.record_cost(actual_cost)
else:
    # Use cheaper model or queue for later
    pass
```

---

## Quick Reference: Model Selection

### When to Use Each Model

**OpenAI:**
- `gpt-4o-mini`: High-volume, simple tasks, routing, classification
- `gpt-4o`: General-purpose, balanced performance
- `gpt-4o-2024-08-06`: When you need guaranteed JSON schema adherence
- `o1-preview`: Complex reasoning, STEM problems, strategic planning
- `o1-mini`: Coding, math, science (cheaper than o1-preview)

**Anthropic:**
- `claude-haiku-4`: Speed-critical, high-volume, simple tasks
- `claude-sonnet-4.5`: General-purpose, long-form writing, analysis
- `claude-3-7-sonnet`: When you need extended thinking for complex problems
- `claude-opus-4`: Highest quality, critical tasks, complex reasoning

### Cost-Performance Trade-off

```
High Performance ↑
│
│  o1-preview ($$$$$)
│  claude-opus-4 ($$$$)
│  claude-3-7-sonnet ($$$)
│  gpt-4o ($$$)
│  claude-sonnet-4.5 ($$)
│  gpt-4o-mini ($)
│  claude-haiku-4 ($)
│
└─────────────────────→ Low Cost
```

---

## Conclusion

Effective API usage for multi-agent systems requires:

1. **Smart Model Selection** - Use the right model for each task
2. **Prompt Engineering** - Clear, structured, well-constrained prompts
3. **Cost Management** - Caching, batching, cascading, compression
4. **Reliability** - Retries, fallbacks, circuit breakers, timeouts
5. **Monitoring** - Track costs, latency, errors, and usage patterns

By following these best practices, you can build production-grade multi-agent systems that are:
- **Cost-effective** (50-80% cost reduction possible)
- **Reliable** (99.9%+ success rate)
- **Fast** (sub-second latency for most tasks)
- **Scalable** (handle thousands of requests/minute)

Remember: The cheapest API call is the one you don't make. Cache aggressively, compress context, and route intelligently.
