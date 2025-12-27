# LLM Fundamentals for Agentic AI Development

## Bridging Prerequisites to Agent Development

**Purpose:** This document provides the foundational knowledge developers need to effectively work with LLMs before building agentic systems. It covers how LLMs work internally, prompting best practices, API patterns, and a decision framework for when agents add value.

**Target Audience:** Developers with programming experience transitioning to agentic AI development.

**Last Updated:** 2025-12-27

---

## Table of Contents

1. [How LLMs Actually Work](#1-how-llms-actually-work)
2. [Prompting Fundamentals](#2-prompting-fundamentals)
3. [API Patterns](#3-api-patterns)
4. [When Agents vs When Not](#4-when-agents-vs-when-not)
5. [Hands-On Exercises](#5-hands-on-exercises)
6. [Quick Reference](#6-quick-reference)

---

## 1. How LLMs Actually Work

### 1.1 Tokenization Deep Dive

Tokenization converts text into numerical representations that neural networks can process. This is one of the most consequential design decisions in the LLM pipeline, directly impacting vocabulary size, inference speed, and context window efficiency.

#### Byte Pair Encoding (BPE)

BPE is the dominant tokenization algorithm used by GPT models (GPT-2 through GPT-5.2):

1. **Start** with all individual bytes/characters as tokens
2. **Identify** the most frequently occurring adjacent pair
3. **Merge** that pair into a single new token
4. **Repeat** until vocabulary reaches target size (typically 50K-100K tokens)

```python
# Example: How "encoding" might be tokenized
"encoding" -> ["encod", "ing"]  # "ing" is a common morphological pattern
"unprecedented" -> ["un", "prec", "ed", "ent", "ed"]  # rarer word, more pieces
```

**Key Insight:** Common patterns like "-ing", "-tion", "the" become single tokens, while rare technical terms split into character-level pieces.

#### Major Tokenizer Implementations

| Tokenizer | Used By | Key Characteristics |
|-----------|---------|---------------------|
| **tiktoken** | OpenAI (GPT-4, GPT-5.2) | BPE-based, cl100k_base encoding |
| **SentencePiece** | T5, XLNet, LLaMA | Language-agnostic, no pre-tokenization |
| **WordPiece** | BERT, DistilBERT | Likelihood-based merging, preserves words |

#### Token Counting in Practice

```python
import tiktoken

# Load the encoding for GPT-4/GPT-5
encoding = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text: str) -> int:
    """Count tokens for a given text string."""
    return len(encoding.encode(text))

def estimate_cost(prompt: str, completion: str,
                  input_price: float = 1.25,  # per 1M tokens
                  output_price: float = 10.0) -> float:
    """Estimate API cost for GPT-5.2."""
    input_tokens = count_tokens(prompt)
    output_tokens = count_tokens(completion)

    cost = (input_tokens / 1_000_000 * input_price +
            output_tokens / 1_000_000 * output_price)
    return cost

# Critical: Message framing adds overhead
# GPT models add ~3 tokens per message + 1 if name field present
```

**Common Pitfall:** Naive word counting (`len(text.split())`) underestimates actual tokens by 2-3x.

### 1.2 Context Window Management

The context window is the maximum tokens a model can process in a single call. This limit stems from the transformer's self-attention mechanism, which computes O(n^2) attention weights.

#### Current Model Context Limits (December 2025)

| Model | Context Window | Notes |
|-------|----------------|-------|
| GPT-5.2 | 400K tokens | 256K optimized for long-context reasoning |
| Claude Opus 4.5 | 200K tokens | <5% accuracy degradation across full context |
| Gemini 3 Pro | 1M tokens | Full codebase analysis capable |
| Llama 4 Scout | 10M tokens | Industry-leading, open-source |
| DeepSeek V3.2 | 128K tokens | Cost-efficient option |

#### The "Lost in the Middle" Problem

Research shows models perform worse on information placed in the middle of long contexts. This has practical implications:

```python
# GOOD: Place critical information at beginning and end
system_prompt = """
<critical_instructions>
{Most important rules here - models attend strongly to start}
</critical_instructions>

<context>
{Background information - less critical positioning}
</context>

<final_reminder>
{Key constraints restated - models attend strongly to end}
</final_reminder>
"""

# BAD: Burying critical info in the middle of a long context
```

#### Context Management Strategies

**1. Selective Context Injection**
```python
def build_relevant_context(query: str, history: list, max_tokens: int = 8000) -> list:
    """Include only relevant conversation history."""
    relevant = []
    token_count = 0

    # Prioritize recent + semantically similar turns
    for turn in prioritize_by_relevance(history, query):
        turn_tokens = count_tokens(str(turn))
        if token_count + turn_tokens > max_tokens:
            break
        relevant.append(turn)
        token_count += turn_tokens

    return relevant
```

**2. Hierarchical Summarization**
```python
def hierarchical_context(conversation: list) -> str:
    """Recent turns verbatim, older turns summarized."""
    recent = conversation[-5:]  # Last 5 turns verbatim
    older = conversation[:-5]

    if older:
        summary = summarize(older)  # Compress older context
        return f"<summary>{summary}</summary>\n{format_turns(recent)}"
    return format_turns(recent)
```

**3. Sliding Window with Memory**
```python
class ConversationMemory:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.buffer = []
        self.key_facts = []  # Extracted important information

    def add(self, turn: dict):
        self.buffer.append(turn)
        if len(self.buffer) > self.window_size:
            # Extract key facts before removing
            old_turn = self.buffer.pop(0)
            self.key_facts.extend(extract_facts(old_turn))

    def get_context(self) -> str:
        facts_section = "\n".join(self.key_facts[-20:])
        turns_section = format_turns(self.buffer)
        return f"<key_facts>\n{facts_section}\n</key_facts>\n\n{turns_section}"
```

### 1.3 Attention Mechanisms Simplified

The attention mechanism is the core innovation enabling transformers to capture relationships between distant elements in a sequence.

#### Self-Attention: The Core Operation

For each token position, the model computes:
1. **Query (Q):** "What am I looking for?"
2. **Key (K):** "What do I contain?"
3. **Value (V):** "What information do I provide?"

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

The scaling factor `sqrt(d_k)` prevents attention scores from becoming too large in high-dimensional spaces.

#### Multi-Head Attention

Rather than single attention, models run multiple parallel attention operations ("heads"):

- **Different heads capture different relationships:**
  - Head 1: Subject-verb agreement
  - Head 2: Coreference resolution
  - Head 3: Semantic similarity
  - Head 4: Positional relationships

```python
# Conceptual: Each head attends to different patterns
def multi_head_attention(x, num_heads=12):
    heads = []
    for i in range(num_heads):
        # Each head has its own Q, K, V projections
        q = x @ W_q[i]
        k = x @ W_k[i]
        v = x @ W_v[i]
        heads.append(attention(q, k, v))

    # Concatenate and project
    return concat(heads) @ W_o
```

#### Why This Matters for Agents

1. **Tool Selection:** Attention helps models identify which tools match user intent
2. **Context Prioritization:** Recent observations vs. long-term goals
3. **Ambiguity Resolution:** Determining referents for pronouns/references
4. **Instruction Following:** Attending to constraints in system prompts

**Practical Insight:** Place critical information where attention naturally focuses (beginning/end of context, close to relevant queries).

### 1.4 Model Capabilities and Limitations

#### What LLMs Can Do Well

| Capability | Example | Reliability |
|------------|---------|-------------|
| Pattern recognition | Code completion, text classification | High |
| Knowledge synthesis | Summarization, Q&A on provided docs | High |
| Format transformation | JSON extraction, translation | High |
| Step-by-step reasoning | Math with CoT, logic puzzles | Medium-High |
| Code generation | Function implementation, debugging | Medium-High |
| Creative writing | Stories, marketing copy | Medium |

#### Fundamental Limitations

**1. Hallucination (Theoretically Inevitable)**
- Proven: No computable model can avoid hallucination on unbounded queries
- Training data compression ensures some information loss
- **Mitigation:** RAG, verification steps, permission to say "I don't know"

**2. Context Compression**
- "Lost in the middle" phenomenon
- Information at context boundaries receives more attention
- **Mitigation:** Strategic information placement, chunking

**3. Reasoning Degradation**
- Performance drops on novel multi-step reasoning
- Longer chains = more error accumulation
- **Mitigation:** Chain-of-thought, problem decomposition

**4. Retrieval Fragility**
- Models may miss information even when present in context
- Paraphrasing can break retrieval
- **Mitigation:** Multiple query formulations, explicit citation requests

**5. Confidence Miscalibration**
- False information stated with same confidence as true
- No internal uncertainty signal exposed to users
- **Mitigation:** Explicit uncertainty prompting, multi-sample verification

### 1.5 December 2025 Model Landscape

#### Frontier Model Comparison

| Model | Best For | Context | Speed | Price (per 1M tokens) |
|-------|----------|---------|-------|----------------------|
| **GPT-5.2** | Speed, math reasoning | 400K | 187 tok/s | $1.25 in / $10 out |
| **Claude Opus 4.5** | Coding, agents | 200K | 49 tok/s | $5 in / $25 out |
| **Gemini 3 Pro** | Multimodal, long context | 1M | ~95 tok/s | $2-4 in / $12-18 out |
| **DeepSeek V3.2** | Cost efficiency | 128K | Varies | $0.28 in / $0.56 out |
| **Llama 4 Scout** | Self-hosting, privacy | 10M | Self-host | Free (self-host) |

#### Key Model Differentiators

**GPT-5.2:**
- 100% accuracy on AIME 2025 (math competition)
- 52.9% on ARC-AGI-2 (abstract reasoning benchmark)
- 3.8x faster than Claude Opus 4.5
- Best for: Real-time applications, mathematical reasoning

**Claude Opus 4.5:**
- 80.9% on SWE-bench Verified (coding benchmark leader)
- 59.3% on Terminal-bench 2.0 (command-line proficiency)
- Strongest prompt injection resistance
- Memory Tool for persistent agent state
- Best for: Coding, agentic workflows, security-sensitive applications

**Gemini 3 Pro:**
- 81% on MMMU-Pro (multimodal understanding)
- Native multimodal architecture (text, image, video, audio)
- 1M token context for full codebase analysis
- Best for: Vision tasks, document analysis, video understanding

**Extended Thinking Modes:**
All frontier models now support inference-time compute allocation:
- OpenAI: `reasoning` parameter (instant/thinking/pro)
- Anthropic: Extended thinking with `effort` parameter
- Google: Deep Think mode
- Benefit: 10-30% accuracy improvement on reasoning tasks

---

## 2. Prompting Fundamentals

### 2.1 Zero-Shot vs Few-Shot Prompting

#### Zero-Shot Prompting

No examples provided; the model relies entirely on pre-trained knowledge:

```python
# Zero-shot classification
prompt = """
Classify the following customer feedback as: positive, negative, or neutral.

Feedback: "The product works great but shipping took forever."

Classification:
"""
# Model must infer the task from instructions alone
```

**When to use:**
- Simple, well-defined tasks
- Tasks common in training data
- When example curation is expensive
- Rapid prototyping

#### Few-Shot Prompting

Provide examples that demonstrate the expected input-output pattern:

```python
# Few-shot classification with examples
prompt = """
Classify customer feedback as: positive, negative, or neutral.

Examples:
Feedback: "Absolutely love this product! Best purchase ever."
Classification: positive

Feedback: "Terrible quality. Broke after one week."
Classification: negative

Feedback: "It works as described. Nothing special."
Classification: neutral

Now classify:
Feedback: "The product works great but shipping took forever."
Classification:
"""
```

**When to use:**
- Complex tasks requiring specific format
- Domain-specific terminology or conventions
- Edge cases that need explicit handling
- Tasks where zero-shot consistently fails

#### Chain-of-Thought (CoT) Prompting

Adding reasoning steps dramatically improves complex reasoning:

```python
# Zero-shot CoT (remarkably effective)
prompt = """
A store had 20 books. They sold 7 on Monday and 5 on Tuesday.
How many books remain?

Let's think step by step:
"""

# Few-shot CoT (more controlled)
prompt = """
Q: Sarah had 15 apples. She gave 6 to her friend and ate 2. How many left?

Let's think step by step:
- Started with: 15 apples
- Gave away: 6 apples, leaving 15 - 6 = 9 apples
- Ate: 2 apples, leaving 9 - 2 = 7 apples
Answer: 7

Q: A store had 20 books. They sold 7 on Monday and 5 on Tuesday. How many remain?

Let's think step by step:
"""
```

**Key Finding:** For reasoning-heavy tasks, zero-shot CoT often outperforms few-shot because examples can bias the model toward surface patterns rather than genuine reasoning.

### 2.2 System Prompts vs User Prompts

#### System Prompt Purpose

System prompts define persistent behavior, role, and constraints:

```python
system_prompt = """
You are a senior software engineer conducting code reviews.

Behavioral guidelines:
- Be constructive and specific in feedback
- Prioritize security issues, then bugs, then style
- Reference specific line numbers when discussing code
- Suggest concrete improvements, not just problems

Constraints:
- Never suggest changes that would break existing tests
- Maintain the project's existing code style
- If uncertain about intent, ask clarifying questions
"""
```

#### User Prompt Purpose

User prompts contain the specific request and dynamic content:

```python
user_prompt = """
Please review this pull request:

<code>
def process_payment(user_id: str, amount: float):
    user = db.get_user(user_id)
    if user.balance >= amount:
        user.balance -= amount
        db.save(user)
        return True
    return False
</code>

Focus areas: concurrency safety, error handling
"""
```

#### Role Separation Best Practices

```python
# GOOD: Clear separation of concerns
messages = [
    {
        "role": "system",
        "content": """You are a helpful assistant for a SaaS company.

        Rules:
        - Never share internal documentation
        - Always verify user identity before account changes
        - Escalate billing disputes to human support"""
    },
    {
        "role": "user",
        "content": "I need to update my billing address to 123 Main St"
    }
]

# BAD: Mixing role and task in user prompt
messages = [
    {
        "role": "user",
        "content": """You are a helpful assistant. You should never share
        internal docs. Now, I need to update my billing address..."""
    }
]
```

### 2.3 Structured Outputs

Modern LLMs support guaranteed schema-conformant outputs, eliminating parsing failures.

#### JSON Mode vs Structured Outputs

| Feature | JSON Mode | Structured Outputs |
|---------|-----------|-------------------|
| Valid JSON | Guaranteed | Guaranteed |
| Schema adherence | Not guaranteed | Guaranteed |
| Type safety | No | Yes |
| Required fields | May be missing | Always present |

#### Implementation with Pydantic

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI

# Define your schema
class ExtractedEvent(BaseModel):
    """Event information extracted from text."""
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Date in YYYY-MM-DD format")
    location: Optional[str] = Field(description="Event location if mentioned")
    attendees: List[str] = Field(description="Names of attendees mentioned")

# Use structured outputs
client = OpenAI()

response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Extract event information from text."},
        {"role": "user", "content": "Alice and Bob are attending the tech conference in SF on 2025-03-15."}
    ],
    response_format=ExtractedEvent
)

event = response.choices[0].message.parsed
print(f"Event: {event.event_name}")
print(f"Date: {event.date}")
print(f"Attendees: {', '.join(event.attendees)}")
```

#### Function Calling for Structured Outputs

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": "Extract named entities from text",
            "parameters": {
                "type": "object",
                "properties": {
                    "people": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of people mentioned"
                    },
                    "organizations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Organization names mentioned"
                    },
                    "locations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Location names mentioned"
                    }
                },
                "required": ["people", "organizations", "locations"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "John from Microsoft met Sarah in Seattle."}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_entities"}}
)
```

### 2.4 Common Prompting Mistakes

#### Mistake 1: Vague Instructions

```python
# BAD: Ambiguous
prompt = "Analyze customer feedback"

# GOOD: Specific
prompt = """
Analyze the following customer feedback and provide:
1. Sentiment (positive/negative/neutral)
2. Top 3 specific issues mentioned
3. Suggested action items for product team

Format as JSON with keys: sentiment, issues, actions
"""
```

#### Mistake 2: Missing Role Assignment

```python
# BAD: No context
prompt = "Write a response to this complaint"

# GOOD: Clear role
prompt = """
You are a senior customer success manager at a B2B SaaS company.
You are empathetic but maintain professional boundaries.
You can offer refunds up to $500 without escalation.

Write a response to this complaint:
{complaint}
"""
```

#### Mistake 3: Overloaded Prompts

```python
# BAD: Too many tasks
prompt = """
Analyze market position, project revenue, develop hiring plan,
create marketing strategy, and assess competitive threats.
"""

# GOOD: Focused single task
prompt = """
Analyze the competitive threats in our market segment.

Focus on:
- Direct competitors (companies solving the same problem)
- Substitute solutions (different approaches to the problem)
- New entrants (startups and big tech pivots)

For each threat, assess: likelihood, impact, and mitigation options.
"""
```

#### Mistake 4: No Output Format Specification

```python
# BAD: Format left to model
prompt = "Compare Python and JavaScript"

# GOOD: Explicit format
prompt = """
Compare Python and JavaScript for backend web development.

Structure your response as a markdown table with columns:
| Aspect | Python | JavaScript | Winner |

Cover: performance, ecosystem, learning curve, job market, tooling
"""
```

#### Mistake 5: Not Giving Permission to Fail

```python
# BAD: Forces fabrication
prompt = "What is the revenue of CompanyXYZ in 2024?"

# GOOD: Allows uncertainty
prompt = """
What is the revenue of CompanyXYZ in 2024?

If this information is not in the provided context or you're uncertain,
say "I don't have reliable information about this" rather than guessing.
"""
```

### 2.5 XML Prompting Patterns (Claude-Specific)

Claude models respond particularly well to XML-structured prompts due to training on XML-formatted data.

#### Basic XML Structure

```xml
<task>
You are an expert code reviewer analyzing Python code for security issues.
</task>

<instructions>
1. Identify potential security vulnerabilities
2. Rate severity as: critical, high, medium, low
3. Provide specific remediation steps
4. Reference relevant security standards (OWASP, CWE)
</instructions>

<code_to_review>
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    return db.execute(query)
</code_to_review>

<output_format>
Respond with JSON:
{
  "vulnerabilities": [
    {
      "type": "string",
      "severity": "critical|high|medium|low",
      "line_number": number,
      "description": "string",
      "remediation": "string",
      "reference": "string"
    }
  ]
}
</output_format>
```

#### Nested XML for Complex Tasks

```xml
<task_definition>
Extract and structure data from unstructured customer reviews.
</task_definition>

<background>
Our product is a SaaS analytics platform. Customers include both
technical data analysts and business stakeholders.
</background>

<instructions>
  <step_1>Identify the customer's primary use case</step_1>
  <step_2>Extract features mentioned with sentiment for each</step_2>
  <step_3>Generate a 2-3 sentence summary of needs and satisfaction</step_3>
</instructions>

<example>
<input>
"Love the dashboards but the API is too slow for real-time needs."
</input>
<output>
{
  "use_case": "business reporting and API integration",
  "features": [
    {"name": "dashboards", "sentiment": "positive"},
    {"name": "API performance", "sentiment": "negative"}
  ],
  "summary": "Customer values visualization capabilities but blocked on API latency for real-time applications. Priority: optimize API performance."
}
</output>
</example>

<review_to_analyze>
{customer_review}
</review_to_analyze>
```

#### Claude 4.5 Extended Thinking Pattern

```xml
<task>
Solve this complex optimization problem.
</task>

<thinking_instructions>
Take your time to think through this carefully. Consider:
- Multiple approaches before committing
- Edge cases and failure modes
- Trade-offs between solutions
- Verification of your reasoning
</thinking_instructions>

<problem>
{complex_problem_description}
</problem>
```

**Note:** Claude 4.5 improved instruction following means you need less emphasis (avoid "CRITICAL:", "MUST") and can use more natural phrasing.

---

## 3. API Patterns

### 3.1 Streaming vs Non-Streaming

#### When to Use Streaming

| Scenario | Recommendation |
|----------|----------------|
| Chat interfaces | Stream (better UX) |
| Long-form generation | Stream (reduces perceived latency) |
| Batch processing | Non-stream (simpler error handling) |
| Function calling | Non-stream (need complete call) |
| Programmatic integration | Non-stream (easier parsing) |
| Accessibility requirements | Offer both options |

#### Streaming Implementation

```python
from openai import OpenAI

client = OpenAI()

def stream_response(messages: list) -> str:
    """Stream response with real-time display."""
    full_response = ""

    stream = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    print()  # Newline at end
    return full_response
```

#### Non-Streaming for Reliability

```python
def get_response(messages: list, max_retries: int = 3) -> str:
    """Non-streaming with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 3.2 Error Handling and Retries

#### Exponential Backoff with Jitter

```python
import time
import random
from typing import TypeVar, Callable

T = TypeVar('T')

def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 60.0,
    jitter: bool = True
) -> T:
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        jitter: Add randomness to prevent thundering herd
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (2 ** attempt), max_delay)

            # Add jitter to prevent synchronized retries
            if jitter:
                delay = delay * (0.5 + random.random())

            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
            time.sleep(delay)
```

#### Handling Specific Error Types

```python
from openai import (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError
)

def safe_api_call(messages: list) -> str:
    """Handle different error types appropriately."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return response.choices[0].message.content

    except AuthenticationError:
        # Don't retry - fix API key
        raise ValueError("Invalid API key. Check OPENAI_API_KEY.")

    except RateLimitError as e:
        # Retry with longer delay
        retry_after = getattr(e, 'retry_after', 60)
        print(f"Rate limited. Waiting {retry_after}s")
        time.sleep(retry_after)
        return safe_api_call(messages)  # Retry

    except APIConnectionError:
        # Network issue - retry with backoff
        raise  # Let retry_with_backoff handle

    except APITimeoutError:
        # Consider reducing prompt size or increasing timeout
        raise
```

### 3.3 Rate Limiting Strategies

#### Understanding Rate Limits

LLM APIs enforce limits on multiple dimensions:
- **RPM:** Requests per minute
- **TPM:** Tokens per minute
- **RPD:** Requests per day
- **TPD:** Tokens per day

You hit whichever limit you reach first.

#### Token Bucket Implementation

```python
import time
import threading
from collections import deque

class TokenBucket:
    """Rate limiter using token bucket algorithm."""

    def __init__(self, tokens_per_second: float, max_tokens: int):
        self.tokens_per_second = tokens_per_second
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire tokens. Returns True if successful."""
        with self.lock:
            now = time.time()
            # Add tokens based on time elapsed
            elapsed = now - self.last_update
            self.tokens = min(
                self.max_tokens,
                self.tokens + elapsed * self.tokens_per_second
            )
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_for_tokens(self, tokens: int = 1):
        """Block until tokens are available."""
        while not self.acquire(tokens):
            time.sleep(0.1)

# Usage
limiter = TokenBucket(tokens_per_second=10, max_tokens=100)

def rate_limited_call(messages: list) -> str:
    limiter.wait_for_tokens(1)
    return client.chat.completions.create(
        model="gpt-4",
        messages=messages
    ).choices[0].message.content
```

#### Request Queue with Batching

```python
import asyncio
from typing import List, Dict, Any

class BatchProcessor:
    """Batch requests to maximize throughput within rate limits."""

    def __init__(self, batch_size: int = 10, wait_time: float = 1.0):
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.queue: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}

    async def add_request(self, request_id: str, messages: list) -> str:
        """Add request to queue and wait for result."""
        self.queue.append({
            "id": request_id,
            "messages": messages
        })

        # Wait for batch processing
        while request_id not in self.results:
            await asyncio.sleep(0.1)

        return self.results.pop(request_id)

    async def process_batches(self):
        """Continuously process batches."""
        while True:
            if len(self.queue) >= self.batch_size:
                batch = self.queue[:self.batch_size]
                self.queue = self.queue[self.batch_size:]

                # Process batch (could use OpenAI Batch API)
                for request in batch:
                    result = await self._process_single(request["messages"])
                    self.results[request["id"]] = result

            await asyncio.sleep(self.wait_time)

    async def _process_single(self, messages: list) -> str:
        # Actual API call
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4",
            messages=messages
        )
        return response.choices[0].message.content
```

### 3.4 Cost Estimation

#### Token Counting Utility

```python
import tiktoken

class CostEstimator:
    """Estimate and track API costs."""

    # December 2025 pricing (per 1M tokens)
    PRICING = {
        "gpt-5.2": {"input": 1.25, "output": 10.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "claude-opus-4.5": {"input": 5.0, "output": 25.0},
        "claude-sonnet-4.5": {"input": 3.0, "output": 15.0},
        "gemini-3-pro": {"input": 3.0, "output": 15.0},
        "gemini-3-flash": {"input": 0.5, "output": 3.0},
        "deepseek-v3.2": {"input": 0.28, "output": 0.56},
    }

    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.encoding = tiktoken.encoding_for_model("gpt-4")  # Approximation
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def estimate_cost(self, input_text: str, estimated_output: int = 500) -> float:
        """Estimate cost before making API call."""
        input_tokens = self.count_tokens(input_text)
        prices = self.PRICING.get(self.model, self.PRICING["gpt-4"])

        cost = (
            input_tokens / 1_000_000 * prices["input"] +
            estimated_output / 1_000_000 * prices["output"]
        )
        return cost

    def track_usage(self, input_tokens: int, output_tokens: int):
        """Track actual usage."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def get_total_cost(self) -> float:
        """Get total cost so far."""
        prices = self.PRICING.get(self.model, self.PRICING["gpt-4"])
        return (
            self.total_input_tokens / 1_000_000 * prices["input"] +
            self.total_output_tokens / 1_000_000 * prices["output"]
        )

    def get_summary(self) -> str:
        """Get usage summary."""
        return f"""
Usage Summary:
- Model: {self.model}
- Input tokens: {self.total_input_tokens:,}
- Output tokens: {self.total_output_tokens:,}
- Total cost: ${self.get_total_cost():.4f}
"""
```

### 3.5 Multi-Provider Patterns

#### Provider Abstraction Layer

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, messages: List[Dict], **kwargs) -> str:
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4"):
        self.client = OpenAI()
        self.model = model

    def complete(self, messages: List[Dict], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

    def is_healthy(self) -> bool:
        try:
            self.complete([{"role": "user", "content": "test"}], max_tokens=5)
            return True
        except:
            return False

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-opus-4.5"):
        self.client = Anthropic()
        self.model = model

    def complete(self, messages: List[Dict], **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 4096)
        )
        return response.content[0].text

    def is_healthy(self) -> bool:
        try:
            self.complete([{"role": "user", "content": "test"}], max_tokens=5)
            return True
        except:
            return False
```

#### Fallback with Load Balancing

```python
import random
from typing import Optional

class MultiProviderClient:
    """Client with automatic fallback and load balancing."""

    def __init__(self, providers: List[LLMProvider]):
        self.providers = providers
        self.unhealthy_until: Dict[int, float] = {}

    def _get_available_providers(self) -> List[LLMProvider]:
        """Get providers not currently marked unhealthy."""
        now = time.time()
        available = []
        for i, provider in enumerate(self.providers):
            if i not in self.unhealthy_until or now > self.unhealthy_until[i]:
                available.append((i, provider))
        return available

    def _mark_unhealthy(self, index: int, duration: float = 60.0):
        """Mark provider as unhealthy for duration seconds."""
        self.unhealthy_until[index] = time.time() + duration

    def complete(self, messages: List[Dict], **kwargs) -> str:
        """Try providers in order with fallback."""
        available = self._get_available_providers()

        if not available:
            # All providers unhealthy, try primary anyway
            available = [(0, self.providers[0])]

        # Optional: shuffle for load balancing
        # random.shuffle(available)

        last_error = None
        for idx, provider in available:
            try:
                return provider.complete(messages, **kwargs)
            except Exception as e:
                last_error = e
                self._mark_unhealthy(idx)
                continue

        raise Exception(f"All providers failed. Last error: {last_error}")

# Usage
client = MultiProviderClient([
    OpenAIProvider("gpt-4"),           # Primary
    AnthropicProvider("claude-opus-4.5"),  # First fallback
    OpenAIProvider("gpt-3.5-turbo"),   # Cost-effective fallback
])

response = client.complete([
    {"role": "user", "content": "Hello, world!"}
])
```

#### Cost-Optimized Routing

```python
class CostOptimizedRouter:
    """Route to cheapest provider that can handle the task."""

    def __init__(self):
        self.providers = {
            "simple": DeepSeekProvider(),      # Cheapest
            "standard": OpenAIProvider("gpt-4"),
            "complex": AnthropicProvider("claude-opus-4.5"),
            "multimodal": GeminiProvider("gemini-3-pro"),
        }

    def classify_complexity(self, messages: List[Dict]) -> str:
        """Classify task complexity."""
        text = " ".join(m.get("content", "") for m in messages)

        # Simple heuristics (could use classifier model)
        if len(text) < 100 and "?" in text:
            return "simple"
        if "image" in text or "analyze this screenshot" in text.lower():
            return "multimodal"
        if any(kw in text.lower() for kw in ["code", "debug", "refactor", "implement"]):
            return "complex"
        return "standard"

    def complete(self, messages: List[Dict], **kwargs) -> str:
        """Route to appropriate provider based on task."""
        complexity = self.classify_complexity(messages)
        provider = self.providers[complexity]

        try:
            return provider.complete(messages, **kwargs)
        except Exception:
            # Fallback to more capable provider
            if complexity == "simple":
                return self.providers["standard"].complete(messages, **kwargs)
            return self.providers["complex"].complete(messages, **kwargs)
```

---

## 4. When Agents vs When Not

### 4.1 The Five-Question Decision Framework

Before implementing an agent, answer these questions:

```
1. Does this task require AUTONOMOUS DECISION-MAKING?
   - If decisions are predetermined: Use workflow automation
   - If decisions depend on context: Consider agent

2. Does this task require MULTI-STEP REASONING?
   - Single step: Simple LLM call
   - 2-3 steps: Chain/pipeline
   - Dynamic steps: Agent

3. Does this task require TOOL INTERACTION?
   - No tools: Chatbot
   - Fixed tool sequence: Workflow
   - Dynamic tool selection: Agent

4. Is there meaningful UNCERTAINTY to handle?
   - Predictable outcomes: Automation
   - Need to adapt to surprises: Agent

5. Is the COST-COMPLEXITY justified?
   - High volume, simple tasks: Automation
   - Complex, valuable tasks: Agent
```

### 4.2 Chatbot vs Agent Distinction

| Characteristic | Chatbot | Agent |
|---------------|---------|-------|
| **Initiative** | Reactive (waits for input) | Proactive (pursues goals) |
| **Memory** | Session-only or none | Persistent, goal-oriented |
| **Actions** | Responds with text | Executes tools, modifies systems |
| **Planning** | None | Multi-step planning |
| **Learning** | Static or minimal | Adapts based on outcomes |
| **Autonomy** | Human-driven | Goal-driven |

#### When Chatbot is Better

```python
# Chatbot use cases:
# - FAQ answering
# - Simple information retrieval
# - Guided form filling
# - Basic customer support
# - Content generation requests

# Example: FAQ Chatbot (NOT an agent)
def faq_chatbot(question: str, faq_db: dict) -> str:
    """Simple retrieval-based chatbot."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Answer using this FAQ: {faq_db}"},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content
```

#### When Agent is Better

```python
# Agent use cases:
# - Multi-step research tasks
# - Code generation with testing
# - Complex scheduling with constraints
# - Incident response automation
# - Autonomous data pipeline management

# Example: Research Agent (IS an agent)
class ResearchAgent:
    def __init__(self):
        self.tools = [web_search, document_reader, note_taker]
        self.memory = PersistentMemory()

    def research(self, topic: str, depth: str = "comprehensive") -> Report:
        """Autonomously research topic, adapting strategy based on findings."""
        plan = self.create_research_plan(topic, depth)

        while not plan.is_complete():
            # Autonomous decision: which step next?
            next_step = plan.get_next_step()

            # Autonomous decision: which tool to use?
            tool = self.select_tool(next_step)
            result = tool.execute(next_step.query)

            # Adapt based on results
            if result.reveals_new_direction():
                plan.adjust(result.new_directions)

            self.memory.store(result)

        return self.synthesize_report()
```

### 4.3 Workflow Automation vs Agent

| Characteristic | Workflow Automation | Agent |
|----------------|---------------------|-------|
| **Path** | Predetermined | Dynamic |
| **Decisions** | Rule-based | Reasoning-based |
| **Error handling** | Predefined branches | Adaptive recovery |
| **Changes** | Requires reprogramming | Adapts to instructions |
| **Cost** | Low operational | Higher operational |

#### When Workflow Automation is Better

```python
# Workflow automation use cases:
# - Invoice processing (fixed steps)
# - Employee onboarding (checklist)
# - Report generation (templated)
# - Data ETL (extract, transform, load)
# - Notification triggers (if-then rules)

# Example: Invoice Workflow (NOT an agent)
def process_invoice(invoice: dict) -> ProcessingResult:
    """Fixed workflow - no autonomous decisions."""
    # Step 1: Validate format
    if not validate_format(invoice):
        return ProcessingResult(status="rejected", reason="invalid_format")

    # Step 2: Extract data
    data = extract_invoice_data(invoice)

    # Step 3: Match to PO
    po_match = find_matching_po(data.po_number)

    # Step 4: Route based on amount (predetermined rule)
    if data.amount > 10000:
        return route_to_approval(data, po_match)
    else:
        return auto_approve(data, po_match)
```

#### When Agent is Better

```python
# Agent use cases:
# - Vendor negotiation (dynamic strategy)
# - Anomaly investigation (adaptive exploration)
# - Customer issue resolution (context-dependent)
# - Complex scheduling (constraint satisfaction)

# Example: Anomaly Investigation Agent
class AnomalyInvestigator:
    """Agent that adaptively investigates anomalies."""

    def investigate(self, anomaly: Anomaly) -> Investigation:
        """Investigate with adaptive strategy."""
        hypotheses = self.generate_hypotheses(anomaly)

        for hypothesis in self.prioritize(hypotheses):
            # Autonomous: choose investigation approach
            evidence = self.gather_evidence(hypothesis)

            if evidence.supports(hypothesis):
                # Autonomous: decide if more evidence needed
                confidence = self.assess_confidence(evidence)
                if confidence > 0.9:
                    return self.conclude(hypothesis, evidence)
                else:
                    # Gather more evidence (adaptive)
                    additional = self.targeted_investigation(hypothesis)
                    evidence.extend(additional)
            else:
                # Autonomous: might generate new hypotheses
                if evidence.suggests_alternative():
                    hypotheses.append(evidence.alternative_hypothesis())

        return self.conclude_inconclusive(hypotheses)
```

### 4.4 Complexity Thresholds

**Agent complexity levels (from simplest to most complex):**

| Level | Type | Characteristics | When to Use |
|-------|------|-----------------|-------------|
| 1 | Functional Agent | API bridging, no memory/learning | Simple integrations |
| 2 | Reactive Agent | Responds to inputs, no planning | Real-time routing |
| 3 | Model-Based | Internal world model, prediction | Supply chain |
| 4 | Goal-Based | Plans to achieve objectives | Project management |
| 5 | Learning Agent | Improves from experience | Recommendations |
| 6 | Utility-Based | Optimizes across trade-offs | Trading, pricing |
| 7 | Hierarchical | Delegates to sub-agents | Large automation |
| 8 | Multi-Agent | Coordinated agent teams | Enterprise workflows |

**Key Insight:** Start at Level 1-3. Only escalate when simpler levels demonstrably fail.

### 4.5 Cost-Benefit Analysis

#### Total Cost of Ownership

```
Implementation Costs:
- Planning & assessment: $25K-75K
- Development: $75K-500K (depending on complexity)
- Integration: $20K-100K
- Testing & validation: $15K-50K

Ongoing Costs (annual):
- Operations & monitoring: 15-30% of development cost
- Infrastructure: $20K-60K
- Model API costs: Variable (track TPM)
- Human oversight: 1-3 FTE equivalent

Break-even Considerations:
- Per-interaction savings: $2-5 vs human ($0.25-0.50 for agent)
- Volume threshold: ~50K interactions/year minimum
- Payback period: 6-18 months typical
```

#### Decision Matrix

```python
def should_implement_agent(workflow: dict) -> dict:
    """Evaluate whether agent implementation is justified."""

    scores = {
        "complexity_fit": 0,  # Does task need agent capabilities?
        "volume_fit": 0,      # Enough volume to justify cost?
        "risk_fit": 0,        # Acceptable risk profile?
        "readiness_fit": 0,   # Organization ready?
    }

    # Complexity scoring
    if workflow["requires_multi_step_reasoning"]:
        scores["complexity_fit"] += 3
    if workflow["requires_dynamic_tool_selection"]:
        scores["complexity_fit"] += 3
    if workflow["requires_adaptation"]:
        scores["complexity_fit"] += 2
    if workflow["is_rule_based"]:
        scores["complexity_fit"] -= 4

    # Volume scoring
    annual_interactions = workflow["annual_interactions"]
    if annual_interactions > 100000:
        scores["volume_fit"] += 4
    elif annual_interactions > 50000:
        scores["volume_fit"] += 2
    elif annual_interactions < 10000:
        scores["volume_fit"] -= 2

    # Risk scoring
    if workflow["cost_of_error"] == "low":
        scores["risk_fit"] += 2
    elif workflow["cost_of_error"] == "high":
        scores["risk_fit"] -= 2
    if workflow["human_oversight_feasible"]:
        scores["risk_fit"] += 1

    # Readiness scoring
    if workflow["data_quality"] == "high":
        scores["readiness_fit"] += 2
    if workflow["has_clear_success_metrics"]:
        scores["readiness_fit"] += 2
    if workflow["team_has_ml_experience"]:
        scores["readiness_fit"] += 1

    total = sum(scores.values())

    return {
        "scores": scores,
        "total": total,
        "recommendation": (
            "Implement agent" if total > 8 else
            "Consider hybrid" if total > 4 else
            "Use simpler automation"
        ),
        "confidence": "high" if abs(total) > 6 else "medium" if abs(total) > 3 else "low"
    }
```

---

## 5. Hands-On Exercises

### Exercise 1: Build a Simple Chat Interface

**Objective:** Create a basic chat interface with conversation memory.

```python
"""
Exercise 1: Simple Chat Interface
Build a command-line chat that maintains conversation history.
"""

from openai import OpenAI
from typing import List, Dict

client = OpenAI()

def create_chat():
    """Create a simple chat interface with memory."""

    system_prompt = """You are a helpful assistant. You remember the entire
    conversation and can reference previous messages."""

    messages: List[Dict] = [
        {"role": "system", "content": system_prompt}
    ]

    print("Chat started. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'quit':
            break

        if not user_input:
            continue

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Get response
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        assistant_message = response.choices[0].message.content

        # Add assistant response to history
        messages.append({"role": "assistant", "content": assistant_message})

        print(f"Assistant: {assistant_message}\n")

    print("Chat ended.")
    return messages

if __name__ == "__main__":
    create_chat()

# Expected behavior:
# - Maintains full conversation history
# - Can reference previous messages
# - Handles empty input gracefully
# - Clean exit with 'quit'
```

### Exercise 2: Add Structured Output

**Objective:** Extract structured data from natural language with validation.

```python
"""
Exercise 2: Structured Output Extraction
Extract meeting information from natural language and validate the output.
"""

from openai import OpenAI
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

client = OpenAI()

class Attendee(BaseModel):
    name: str = Field(description="Full name of attendee")
    email: Optional[str] = Field(description="Email if mentioned")
    role: Optional[str] = Field(description="Role in meeting if mentioned")

class Meeting(BaseModel):
    title: str = Field(description="Meeting title or subject")
    date: str = Field(description="Date in YYYY-MM-DD format")
    time: str = Field(description="Time in HH:MM format (24-hour)")
    duration_minutes: int = Field(description="Duration in minutes")
    attendees: List[Attendee] = Field(description="List of attendees")
    location: Optional[str] = Field(description="Physical or virtual location")
    agenda: Optional[List[str]] = Field(description="Agenda items if mentioned")

    @validator('date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

    @validator('time')
    def validate_time(cls, v):
        try:
            datetime.strptime(v, '%H:%M')
            return v
        except ValueError:
            raise ValueError('Time must be in HH:MM format')

def extract_meeting(text: str) -> Meeting:
    """Extract meeting information from natural language."""

    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": """Extract meeting information from the text.
                If information is not mentioned, use reasonable defaults or null.
                For dates, use the format YYYY-MM-DD.
                For times, use 24-hour format HH:MM."""
            },
            {
                "role": "user",
                "content": text
            }
        ],
        response_format=Meeting
    )

    return response.choices[0].message.parsed

# Test cases
test_inputs = [
    "Schedule a team standup with John, Sarah, and Mike for tomorrow at 9am for 30 minutes",
    "Let's have a project review meeting on 2025-02-15 at 2:30 PM in Conference Room B. We need to discuss the Q1 roadmap and budget allocation. Attendees: alice@company.com (PM), bob@company.com (Engineering Lead)",
    "Quick sync at 10am"
]

if __name__ == "__main__":
    for text in test_inputs:
        print(f"Input: {text}")
        try:
            meeting = extract_meeting(text)
            print(f"Extracted: {meeting.model_dump_json(indent=2)}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

# Expected output structure:
# {
#   "title": "Team Standup",
#   "date": "2025-01-02",
#   "time": "09:00",
#   "duration_minutes": 30,
#   "attendees": [
#     {"name": "John", "email": null, "role": null},
#     {"name": "Sarah", "email": null, "role": null},
#     {"name": "Mike", "email": null, "role": null}
#   ],
#   "location": null,
#   "agenda": null
# }
```

### Exercise 3: Implement Basic Tool Use

**Objective:** Build a simple agent that can use tools to answer questions.

```python
"""
Exercise 3: Basic Tool Use
Implement an agent that can use tools to look up information and perform calculations.
"""

from openai import OpenAI
import json
from typing import Callable, Dict, Any
import math

client = OpenAI()

# Define tools
def get_weather(location: str) -> dict:
    """Simulated weather API."""
    # In production, this would call a real API
    weather_data = {
        "San Francisco": {"temp": 62, "condition": "Foggy", "humidity": 78},
        "New York": {"temp": 45, "condition": "Cloudy", "humidity": 65},
        "Miami": {"temp": 78, "condition": "Sunny", "humidity": 70},
    }
    return weather_data.get(location, {"error": f"No data for {location}"})

def calculate(expression: str) -> dict:
    """Safe calculator for mathematical expressions."""
    try:
        # Allow only safe operations
        allowed = {"__builtins__": {}, "math": math}
        result = eval(expression, allowed, {})
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e)}

def search_knowledge(query: str) -> dict:
    """Simulated knowledge base search."""
    knowledge_base = {
        "company founded": "The company was founded in 2020 by Jane Doe.",
        "product pricing": "Basic: $10/mo, Pro: $25/mo, Enterprise: Custom",
        "office hours": "Monday to Friday, 9 AM to 5 PM PST",
    }
    for key, value in knowledge_base.items():
        if key in query.lower():
            return {"answer": value, "source": "internal_kb"}
    return {"answer": "No relevant information found", "source": None}

# Tool registry
TOOLS: Dict[str, Callable] = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_knowledge": search_knowledge,
}

# Tool definitions for the API
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations. Supports basic math and math module functions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g., '2 + 2' or 'math.sqrt(16)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Search internal knowledge base for company information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def run_agent(user_message: str, max_iterations: int = 5) -> str:
    """Run an agent that can use tools to answer questions."""

    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant with access to tools.
            Use tools when needed to provide accurate information.
            If you don't need a tool, respond directly."""
        },
        {"role": "user", "content": user_message}
    ]

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=TOOL_DEFINITIONS
        )

        choice = response.choices[0]

        # If no tool calls, return the response
        if not choice.message.tool_calls:
            return choice.message.content

        # Process tool calls
        messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Execute the tool
            if function_name in TOOLS:
                result = TOOLS[function_name](**function_args)
            else:
                result = {"error": f"Unknown tool: {function_name}"}

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

        print(f"Iteration {iteration + 1}: Called {[tc.function.name for tc in choice.message.tool_calls]}")

    return "Maximum iterations reached without final answer."

# Test cases
if __name__ == "__main__":
    test_queries = [
        "What's the weather like in San Francisco?",
        "What is the square root of 144 plus 15% tip on $85.50?",
        "When was the company founded and what are the office hours?",
        "What's 2+2? Just answer directly."
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = run_agent(query)
        print(f"Result: {result}")
        print("-" * 60)

# Expected behavior:
# - Uses tools when needed (weather, calculation)
# - Chains multiple tool calls when necessary
# - Responds directly for simple questions
# - Handles errors gracefully
```

### Exercise 4: Practice Exercises

Complete these challenges to reinforce your learning:

#### Challenge 1: Token Budget Manager
```python
"""
Challenge: Implement a token budget manager that:
1. Tracks token usage across a conversation
2. Warns when approaching budget limit
3. Automatically summarizes old messages when limit is near
4. Never exceeds the maximum budget
"""

# Your implementation here
class TokenBudgetManager:
    def __init__(self, max_tokens: int = 8000, warning_threshold: float = 0.8):
        pass

    def add_message(self, message: dict) -> bool:
        """Add message if within budget. Returns False if rejected."""
        pass

    def get_messages(self) -> list:
        """Get current messages, summarizing if needed."""
        pass
```

#### Challenge 2: Multi-Model Router
```python
"""
Challenge: Build a router that:
1. Classifies queries by complexity (simple/medium/complex)
2. Routes simple queries to a fast, cheap model
3. Routes complex queries to a capable model
4. Tracks cost savings from routing decisions
"""

# Your implementation here
class SmartRouter:
    def __init__(self):
        pass

    def route(self, query: str) -> str:
        """Route query and return response."""
        pass

    def get_savings_report(self) -> dict:
        """Report on cost savings from smart routing."""
        pass
```

#### Challenge 3: Retry with Circuit Breaker
```python
"""
Challenge: Implement a resilient API client that:
1. Retries with exponential backoff
2. Opens circuit after N failures
3. Periodically tests if service recovered
4. Logs all failures and recovery events
"""

# Your implementation here
class ResilientClient:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        pass

    def call(self, messages: list) -> str:
        """Make API call with resilience."""
        pass

    def get_health_status(self) -> dict:
        """Return circuit breaker status and statistics."""
        pass
```

---

## 6. Quick Reference

### Token Count Estimations

```
Rule of thumb for English text:
- 1 token ~ 4 characters
- 1 token ~ 0.75 words
- 100 tokens ~ 75 words
- 1000 tokens ~ 750 words

For code:
- 1 token ~ 3-4 characters (more variable)
- Symbols often get their own tokens
```

### API Response Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad request | Fix request format |
| 401 | Unauthorized | Check API key |
| 403 | Forbidden | Check permissions |
| 429 | Rate limited | Retry with backoff |
| 500 | Server error | Retry with backoff |
| 503 | Overloaded | Retry with backoff |

### Model Selection Quick Guide

```
Need speed? -> GPT-5.2
Need coding? -> Claude Opus 4.5
Need vision? -> Gemini 3 Pro
Need cost efficiency? -> DeepSeek V3.2
Need privacy/self-host? -> Llama 4
Need long context? -> Gemini 3 Pro (1M) or Llama 4 Scout (10M)
```

### Prompting Quick Tips

1. **Be specific:** State exactly what you want
2. **Provide examples:** Show, don't just tell
3. **Structure clearly:** Use headers, lists, XML tags
4. **Set constraints:** Specify format, length, restrictions
5. **Allow uncertainty:** Let the model say "I don't know"
6. **Iterate:** Test and refine prompts systematically

### When to Use What

```
Simple Q&A           -> Chatbot
Fixed workflow       -> Automation
Dynamic decisions    -> Agent
Multi-step reasoning -> Agent with tools
High volume, simple  -> Batch processing
Real-time required   -> Streaming
High reliability     -> Non-streaming + retries
```

---

## Further Reading

### Official Documentation
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Anthropic Claude Documentation](https://docs.anthropic.com)
- [Google Gemini API](https://ai.google.dev/docs)

### Key Papers
- Wei et al., 2022: "Chain-of-Thought Prompting" (arXiv:2201.11903)
- Yao et al., 2022: "ReAct: Synergizing Reasoning and Acting" (arXiv:2210.03629)

### Recommended Guides
- [Anthropic's Guide to Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [OpenAI Cookbook](https://cookbook.openai.com)
- [LangChain Documentation](https://docs.langchain.com)

---

**Next Step:** After mastering these fundamentals, proceed to `theoretical-foundations.md` for advanced multi-agent architecture concepts.
