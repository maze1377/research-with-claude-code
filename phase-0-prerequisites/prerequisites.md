# Phase 0: Prerequisites for Agentic AI Development

> **For Complete Beginners with No ML Background**

**Estimated Time:** 2-4 weeks | **Self-Paced** | **Hands-On Focus**

---

## How to Use This Document

This document prepares you for agentic AI development from scratch. If you have ML experience, skip to the [Skill Check](#skill-check-ready-for-phase-1) at the end---if you can answer those questions, proceed directly to Phase 1.

**Learning Path:**
1. Python Essentials (1 week) - Master the language features agents require
2. Machine Learning Foundations (3-5 days) - Conceptual understanding, no math
3. LLM Fundamentals (3-5 days) - How language models work
4. Development Environment (1 day) - Tools and setup
5. Skill Check - Verify readiness for Phase 1

---

## Table of Contents

1. [Python Essentials for Agent Development](#1-python-essentials-for-agent-development)
   - [Python 3.10+ Required Features](#python-310-required-features)
   - [Essential Libraries](#essential-libraries)
   - [Virtual Environments](#virtual-environments-and-dependency-management)
   - [Dependency Management Best Practices](#dependency-management-best-practices)
2. [Machine Learning Foundations](#2-machine-learning-foundations-conceptual)
   - [What is Machine Learning?](#what-is-machine-learning)
   - [Supervised vs Unsupervised Learning](#supervised-vs-unsupervised-learning)
   - [Neural Networks at a High Level](#neural-networks-at-a-high-level)
   - [When NOT to Use Agents](#when-not-to-use-agents-decision-framework)
3. [LLM Fundamentals](#3-llm-fundamentals)
   - [What Are LLMs?](#what-are-llms)
   - [Tokens and Context Windows](#tokens-and-context-windows)
   - [Temperature and Sampling Parameters](#temperature-and-sampling-parameters)
   - [Getting Started with APIs](#getting-started-with-llm-apis)
4. [Development Environment](#4-development-environment)
   - [IDE Comparison](#ide-comparison-vs-code-cursor-claude-code)
   - [API Key Management](#api-key-management)
   - [Local vs Cloud Trade-offs](#local-vs-cloud-development-trade-offs)
   - [Essential Extensions](#essential-vs-code-extensions)
5. [Skill Check](#skill-check-ready-for-phase-1)

---

## 1. Python Essentials for Agent Development

Agent development requires modern Python features that you may not have encountered in basic tutorials. This section covers the specific Python skills you need.

### Python 3.10+ Required Features

**Why Python 3.10+?**
Agent frameworks like LangGraph, CrewAI, and Pydantic AI require Python 3.10 or later. These versions introduced features that make agent code cleaner, safer, and more maintainable.

#### Type Hints and Union Syntax

Type hints document what types your functions expect and return. They catch errors before runtime and enable better IDE support.

```python
# Python 3.9 style (old)
from typing import Union, Optional, List

def process_message(content: Union[str, List[str]]) -> Optional[str]:
    pass

# Python 3.10+ style (modern)
def process_message(content: str | list[str]) -> str | None:
    """Process a message or list of messages."""
    if isinstance(content, list):
        return " ".join(content)
    return content
```

**Why this matters for agents:**
- Agent code handles many data types (strings, lists, dicts, custom objects)
- Type hints prevent bugs where wrong data flows through your agent pipeline
- IDE autocomplete works better with type hints
- Tools like Pyright catch type errors before you run your code

**Key patterns you will use:**

```python
# Function with typed parameters and return
def call_llm(prompt: str, temperature: float = 0.7) -> str:
    """Call an LLM and return the response."""
    ...

# Optional values (may be None)
def get_user_context(user_id: str) -> dict | None:
    """Return user context if available, None otherwise."""
    ...

# Lists and dicts with typed contents
def process_tools(tools: list[dict[str, str]]) -> list[str]:
    """Process tool definitions and return tool names."""
    return [tool["name"] for tool in tools]

# Callable types for callbacks
from typing import Callable

def register_handler(callback: Callable[[str], None]) -> None:
    """Register a message handler callback."""
    ...
```

#### Async/Await for Concurrent Operations

Agents frequently call external APIs (LLMs, databases, web services). Async programming lets you handle multiple operations efficiently without blocking.

```python
import asyncio
import httpx

# Synchronous approach (SLOW - waits for each call)
def fetch_all_sync(urls: list[str]) -> list[str]:
    results = []
    for url in urls:
        response = httpx.get(url)  # Blocks until complete
        results.append(response.text)
    return results

# Async approach (FAST - concurrent calls)
async def fetch_all_async(urls: list[str]) -> list[str]:
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)  # All at once
        return [r.text for r in responses]

# Running async code
async def main():
    urls = ["https://api.example.com/1", "https://api.example.com/2"]
    results = await fetch_all_async(urls)
    print(results)

# Entry point
asyncio.run(main())
```

**Key async patterns for agents:**

```python
# Async function definition
async def query_llm(prompt: str) -> str:
    """Query an LLM asynchronously."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": prompt}]},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        return response.json()["choices"][0]["message"]["content"]

# Running multiple async operations
async def parallel_tool_calls(prompts: list[str]) -> list[str]:
    """Call multiple tools in parallel."""
    tasks = [query_llm(p) for p in prompts]
    return await asyncio.gather(*tasks)

# Async context manager for resources
async def with_database():
    async with get_db_connection() as db:
        result = await db.query("SELECT * FROM users")
        return result
```

**Why async matters for agents:**
- LLM API calls take 1-30 seconds each
- An agent making 5 tool calls sequentially takes 5x longer than in parallel
- Production agents handle multiple user requests simultaneously
- Most agent frameworks (LangGraph, Pydantic AI) are async-first

#### Dataclasses for Structured Data

Dataclasses create classes that primarily hold data, with automatic `__init__`, `__repr__`, and comparison methods.

```python
from dataclasses import dataclass, field
from typing import Any

# Basic dataclass
@dataclass
class ToolCall:
    """Represents a tool call from an LLM."""
    name: str
    arguments: dict[str, Any]
    call_id: str

# Creating instances
tool = ToolCall(name="search", arguments={"query": "weather"}, call_id="tc_123")
print(tool)  # ToolCall(name='search', arguments={'query': 'weather'}, call_id='tc_123')

# Dataclass with defaults and slots (Python 3.10+)
@dataclass(slots=True)
class AgentMessage:
    """A message in an agent conversation."""
    role: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

# Keyword-only fields (Python 3.10+)
@dataclass(kw_only=True)
class AgentConfig:
    """Configuration for an agent."""
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: list[str] = field(default_factory=list)

# Must use keyword arguments
config = AgentConfig(model="gpt-4", temperature=0.5)
```

**Why dataclasses matter for agents:**
- Agent state often involves many structured fields
- `slots=True` reduces memory usage (important for many agent instances)
- `kw_only=True` prevents parameter order bugs in complex configurations
- Works seamlessly with Pydantic for validation

#### Pattern Matching for Control Flow

Pattern matching (Python 3.10+) handles complex branching logic cleanly.

```python
# Processing different tool results
def handle_tool_result(result: dict) -> str:
    match result:
        case {"status": "success", "data": data}:
            return f"Success: {data}"
        case {"status": "error", "code": 429}:
            return "Rate limited - retrying..."
        case {"status": "error", "code": code, "message": msg}:
            return f"Error {code}: {msg}"
        case {"status": "pending", "job_id": job_id}:
            return f"Job {job_id} pending..."
        case _:
            return "Unknown result format"

# Pattern matching with types
def route_message(message: dict) -> str:
    match message:
        case {"type": "user", "content": str(content)}:
            return f"User said: {content}"
        case {"type": "assistant", "tool_calls": list(tools)} if tools:
            return f"Assistant calling {len(tools)} tools"
        case {"type": "assistant", "content": str(content)}:
            return f"Assistant said: {content}"
        case {"type": "tool", "name": name, "result": result}:
            return f"Tool {name} returned: {result}"
        case _:
            return "Unknown message type"
```

**Why pattern matching matters for agents:**
- Agents process many different response types
- Tool results come in various formats
- State machines need clean case handling
- More readable than nested if/elif chains

---

### Essential Libraries

#### Pydantic: Data Validation and Schemas

Pydantic validates data using Python type hints. It is the foundation of most agent frameworks.

```python
from pydantic import BaseModel, Field, validator
from typing import Literal

# Define a validated model
class ToolDefinition(BaseModel):
    """A tool that an agent can use."""
    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., max_length=1024)
    parameters: dict = Field(default_factory=dict)

    @validator('name')
    def name_must_be_snake_case(cls, v):
        if not v.replace('_', '').isalnum():
            raise ValueError('name must be snake_case')
        return v.lower()

# Valid data passes
tool = ToolDefinition(
    name="search_web",
    description="Search the web for information"
)
print(tool.model_dump())  # {'name': 'search_web', 'description': '...', 'parameters': {}}

# Invalid data raises ValidationError
try:
    bad_tool = ToolDefinition(name="", description="x")  # name too short
except Exception as e:
    print(f"Validation error: {e}")
```

**LLM structured output with Pydantic:**

```python
from pydantic import BaseModel
from openai import OpenAI

class SearchQuery(BaseModel):
    """A structured search query."""
    query: str
    max_results: int = 10
    filters: list[str] = []

client = OpenAI()

# Force LLM to return structured data
response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "Search for Python tutorials"}],
    response_format=SearchQuery  # Pydantic model as schema
)

search_query = response.choices[0].message.parsed
print(search_query.query)  # "Python tutorials"
print(search_query.max_results)  # 10
```

#### HTTPX: Async HTTP Client

HTTPX provides both sync and async HTTP clients with a clean API.

```python
import httpx

# Synchronous usage
def get_weather_sync(city: str) -> dict:
    response = httpx.get(f"https://api.weather.com/{city}")
    response.raise_for_status()  # Raise on 4xx/5xx
    return response.json()

# Async usage (preferred for agents)
async def get_weather_async(city: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        response.raise_for_status()
        return response.json()

# With timeout and retries
async def robust_api_call(url: str) -> dict:
    timeout = httpx.Timeout(30.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(3):
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

#### Asyncio Patterns for Agent Development

```python
import asyncio
from typing import Any

# Gather with error handling
async def safe_gather(tasks: list) -> list[Any]:
    """Run tasks concurrently, capturing errors."""
    results = await asyncio.gather(*tasks, return_exceptions=True)
    successes = [r for r in results if not isinstance(r, Exception)]
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"Warning: {len(errors)} tasks failed")
    return successes

# Semaphore for rate limiting
async def rate_limited_calls(urls: list[str], max_concurrent: int = 5) -> list:
    """Call URLs with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(url: str) -> str:
        async with semaphore:  # Only N concurrent
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                return response.text

    return await asyncio.gather(*[fetch_one(url) for url in urls])

# Timeout wrapper
async def with_timeout(coro, seconds: float) -> Any:
    """Run coroutine with timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {seconds}s")
```

---

### Virtual Environments and Dependency Management

Never install packages globally. Virtual environments isolate each project's dependencies.

#### Option 1: venv (Built-in, Simple)

```bash
# Create environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install packages
pip install pydantic httpx openai

# Save dependencies
pip freeze > requirements.txt

# Reproduce environment
pip install -r requirements.txt

# Deactivate
deactivate
```

**Best for:** Learning, simple projects, quick scripts

#### Option 2: Poetry (Recommended for Projects)

Poetry manages dependencies, virtual environments, and packaging in one tool.

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Create new project
poetry new my-agent
cd my-agent

# Or initialize in existing directory
poetry init

# Add dependencies
poetry add pydantic httpx openai
poetry add --group dev pytest black mypy  # Dev-only dependencies

# Install all dependencies
poetry install

# Run commands in virtual environment
poetry run python my_script.py
poetry run pytest

# Activate shell in virtual environment
poetry shell
```

**pyproject.toml created by Poetry:**

```toml
[tool.poetry]
name = "my-agent"
version = "0.1.0"
description = "An AI agent project"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
pydantic = "^2.0"
httpx = "^0.27"
openai = "^1.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
black = "^24.0"
mypy = "^1.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

**Best for:** Production projects, team collaboration, reproducibility

#### Option 3: uv (Fastest, Newest)

uv is a Rust-based package manager that is dramatically faster than pip or Poetry.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create new project
uv init my-agent
cd my-agent

# Add dependencies
uv add pydantic httpx openai
uv add --group dev pytest black

# Run commands
uv run python my_script.py
uv run pytest

# Sync dependencies from lock file
uv sync
```

**Key advantages of uv:**
- 10-100x faster than pip
- Automatic Python version management (like pyenv)
- Global package cache reduces disk usage
- Drop-in replacement for pip commands

**.python-version file:**
```
3.12
```

When `uv run` sees this file, it automatically downloads and uses Python 3.12.

**Best for:** Large projects, CI/CD pipelines, developers who value speed

#### Recommendation

| Situation | Recommendation |
|-----------|----------------|
| Learning/tutorials | `venv` (built-in, no installation) |
| Production projects | Poetry (mature, widely supported) |
| Speed is critical | uv (fastest, Python version management) |
| Team projects | Poetry or uv with lock files committed |

---

### Dependency Management Best Practices

1. **Always use lock files**
   - `requirements.txt` with pinned versions (`pydantic==2.5.3`)
   - `poetry.lock` (Poetry)
   - `uv.lock` (uv)

2. **Separate dev and production dependencies**
   ```bash
   # Poetry
   poetry add --group dev pytest mypy

   # uv
   uv add --group dev pytest mypy
   ```

3. **Pin Python version**
   ```toml
   # pyproject.toml
   [tool.poetry.dependencies]
   python = "^3.10"  # 3.10 or higher, but < 4.0
   ```

4. **Update regularly but carefully**
   ```bash
   # Show outdated packages
   poetry show --outdated

   # Update with caution
   poetry update --dry-run  # Preview changes
   poetry update pydantic   # Update specific package
   ```

5. **Document Python version requirements**
   ```python
   # setup.py or pyproject.toml
   python_requires = ">=3.10"
   ```

---

## 2. Machine Learning Foundations (Conceptual)

You do not need to understand the math behind ML to build agents. You need to understand what ML can and cannot do, and when to use it.

### What is Machine Learning?

**Simple Definition:** Machine learning is software that learns patterns from data instead of following explicit rules.

**Traditional Programming:**
```
Rules + Data → Output
```
You write: "If email contains 'win money', mark as spam."

**Machine Learning:**
```
Data + Expected Outputs → Learned Rules
```
You provide: 10,000 emails labeled "spam" or "not spam". The system learns what makes spam.

#### When to Use ML vs. Rule-Based Systems

| Use Machine Learning When... | Use Rule-Based Systems When... |
|------------------------------|-------------------------------|
| Patterns are too complex to write as rules | Rules are simple and clear |
| You have lots of labeled examples | You have few examples or domain expertise |
| Patterns change over time | Rules are stable and well-defined |
| You need to find patterns you cannot articulate | You can explicitly state all decision criteria |
| Approximate answers are acceptable | Exact, auditable decisions are required |

**Examples:**

| Problem | Approach | Why |
|---------|----------|-----|
| "Is this image a cat?" | ML | Millions of pixels, impossible to write rules |
| "Is this user over 18?" | Rule | Simple: `if age >= 18` |
| "Will this customer churn?" | ML | Complex patterns in behavior data |
| "Is this transaction over $10,000?" | Rule | Single threshold check |
| "Is this email spam?" | ML | Spam tactics constantly evolve |
| "Is this date in the future?" | Rule | Simple date comparison |

### Supervised vs Unsupervised Learning

#### Supervised Learning: Learning with Answers

You provide input-output pairs. The model learns to predict outputs for new inputs.

```
Training Data:
  Input: "I love this product!" → Output: Positive
  Input: "Terrible experience"   → Output: Negative
  Input: "It's okay I guess"     → Output: Neutral

Model learns patterns, then predicts:
  Input: "Best purchase ever!" → Predicted: Positive
```

**Two types:**
- **Classification:** Predict categories (spam/not-spam, positive/negative/neutral)
- **Regression:** Predict numbers (house price, temperature, stock price)

**Examples in agent development:**
- Intent classification: "Is this a question, command, or complaint?"
- Sentiment analysis: "Is the user frustrated?"
- Entity extraction: "What product is being discussed?"

#### Unsupervised Learning: Finding Hidden Structure

No labels provided. The model discovers patterns on its own.

```
Input Data:
  Purchase histories of 100,000 customers

Model discovers:
  Group 1: "Budget shoppers" - buy sale items, low frequency
  Group 2: "Premium buyers" - buy expensive items, high frequency
  Group 3: "Tech enthusiasts" - focus on electronics
```

**Common applications:**
- **Clustering:** Group similar items (customer segments, document categories)
- **Anomaly detection:** Find unusual patterns (fraud, system errors)
- **Dimensionality reduction:** Simplify complex data for visualization

**Examples in agent development:**
- Grouping similar user queries to identify common needs
- Detecting unusual agent behavior patterns
- Organizing knowledge base documents by topic

### Neural Networks at a High Level

Neural networks are ML models loosely inspired by the brain. You do not need to understand their math, but understanding what they do helps you use LLMs effectively.

**What Neural Networks Do:**

```
Input (raw data) → [Neural Network] → Output (prediction)

Image pixels → [Neural Network] → "This is a cat"
Text tokens  → [Neural Network] → "Next word is 'the'"
Audio waves  → [Neural Network] → "Speaker said 'hello'"
```

**Key Concepts (No Math Required):**

1. **Layers:** Neural networks stack simple computations. Early layers detect basic patterns (edges, sounds). Later layers combine patterns into complex concepts (faces, words, meaning).

2. **Training:** The network sees millions of examples and gradually adjusts to make better predictions. Like learning to ride a bike through practice.

3. **Generalization:** A trained network can handle inputs it has never seen before, as long as they are similar to training data.

**Types You Will Encounter:**

| Type | Used For | Example |
|------|----------|---------|
| Transformer | Text, code, general | GPT-4, Claude, Gemini |
| CNN (Convolutional) | Images, video | Image classification |
| RNN/LSTM | Sequences | Older language models |

**What makes transformers special:**
- Process entire sequences at once (not word-by-word)
- "Attention" mechanism lets them understand long-range relationships
- Scale extremely well with more data and compute
- Foundation of all modern LLMs

---

### When NOT to Use Agents (Decision Framework)

AI agents are powerful but not always the right choice. Use this framework before building an agent.

#### The Three-Question Decision Framework

**Question 1: How complex is the task?**

| Complexity | Example | Recommended Approach |
|------------|---------|---------------------|
| Simple, clear rules | "Convert temperature F to C" | Regular code |
| Some ambiguity | "Categorize support tickets" | LLM API call, no agent |
| Multi-step, variable | "Research topic and write report" | Agent |
| Requires exploration | "Debug this codebase" | Agent with tools |

**Question 2: Is accuracy critical?**

| Accuracy Need | Example | Recommended Approach |
|---------------|---------|---------------------|
| Must be 100% correct | Financial calculations | Traditional code |
| High but some tolerance | Customer support routing | LLM with validation |
| Approximate is fine | Brainstorming, drafts | Agent |
| Human reviews anyway | Document summarization | Agent with HITL |

**Question 3: Do you have the infrastructure?**

| Capability | Required For | Workaround |
|------------|--------------|------------|
| API access and budget | Any LLM usage | Use smaller/local models |
| Error handling expertise | Production agents | Start with simple LLM calls |
| Monitoring/observability | Debugging agents | Add logging, use evaluation tools |
| Security knowledge | User-facing agents | Sandbox, rate limit, review |

#### When NOT to Use Agents

**Do NOT use agents when:**

1. **A deterministic algorithm exists**
   - Sorting, searching, mathematical calculations
   - Date/time operations, data transformations
   - Rule: If you can write a for-loop or if-statement, do that

2. **The task has a single, clear answer**
   - Looking up a specific fact in a database
   - Retrieving a user's account balance
   - Rule: Agents add latency and cost for no benefit

3. **Errors are unacceptable**
   - Medical diagnosis (without physician review)
   - Legal advice (without lawyer review)
   - Financial transactions (without validation)
   - Rule: Agents hallucinate; critical decisions need human oversight

4. **You cannot monitor or debug the system**
   - No logging infrastructure
   - No way to review agent decisions
   - Rule: Invisible agents become uncontrollable liabilities

5. **Cost per query matters**
   - High-volume, low-value queries
   - Operations that will scale to millions of calls
   - Rule: Calculate cost at scale before building

#### Decision Flowchart

```
Start: "Should I use an agent?"
        │
        ▼
Can you write deterministic code?
        │
    Yes ─┴─ No
    │       │
    ▼       ▼
Use code  Does it require multiple steps
          with dynamic decisions?
                │
            Yes ─┴─ No
            │       │
            ▼       ▼
    Is some error   Use simple LLM call
    tolerance       (no agent loop)
    acceptable?
        │
    Yes ─┴─ No
    │       │
    ▼       ▼
Use Agent  Add human-in-the-loop
           or use hybrid approach
```

#### Agent Alternatives

| Instead of Agent | Consider | When |
|------------------|----------|------|
| Full reasoning agent | Single LLM call with prompt engineering | Simple Q&A, classification |
| Multi-step agent | Workflow engine with LLM nodes | Predictable process flows |
| Autonomous agent | Human-in-the-loop at key points | High-stakes decisions |
| Expensive agent | Smaller model with retrieval (RAG) | Large document Q&A |
| Complex agent | Breaking into simpler components | Easier testing, debugging |

---

## 3. LLM Fundamentals

Understanding how LLMs work helps you use them effectively. You do not need to know the math, but these concepts are essential.

### What Are LLMs?

**Large Language Models (LLMs)** are neural networks trained to predict the next word (token) in a sequence. Through massive-scale training, they develop capabilities far beyond simple prediction.

**How They Are Built:**

```
Step 1: Pre-training
  - Feed the model trillions of words from the internet
  - Model learns to predict next words
  - Takes weeks on thousands of GPUs
  - Creates "foundation model"

Step 2: Fine-tuning
  - Show model examples of desired behavior
  - "Human: X" → "Assistant: Y" pairs
  - Model learns conversational patterns
  - Creates "instruction-tuned model"

Step 3: RLHF (Reinforcement Learning from Human Feedback)
  - Humans rank model outputs
  - Model learns human preferences
  - Reduces harmful outputs
  - Creates "aligned model"
```

**What They Actually Do:**

At inference time, the model:
1. Takes your input text
2. Converts it to numbers (tokens)
3. Predicts probability of each possible next token
4. Samples from those probabilities (controlled by temperature)
5. Repeats until done

```
Input: "The capital of France is"

Model predicts:
  "Paris" → 95% probability
  "Lyon" → 2%
  "the" → 1%
  ... other tokens → 2%

Output: "Paris"
```

### Tokens and Context Windows

#### What Are Tokens?

Tokens are the units LLMs process. They are not words---they are pieces of text that commonly appear together.

```
"Hello, world!" → ["Hello", ",", " world", "!"]  # 4 tokens
"Tokenization" → ["Token", "ization"]            # 2 tokens
"GPT-4" → ["G", "PT", "-", "4"]                  # 4 tokens
"AI" → ["AI"]                                    # 1 token
```

**Rules of Thumb:**
- 1 token ~= 4 characters in English
- 1 token ~= 0.75 words
- 100 tokens ~= 75 words
- 1,000 tokens ~= 1-2 pages of text

**Checking Token Count:**

```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4")

text = "Hello, world! This is a test."
tokens = encoding.encode(text)

print(f"Text: {text}")
print(f"Token count: {len(tokens)}")
print(f"Tokens: {tokens}")
# Tokens: [9906, 11, 1917, 0, 1115, 374, 264, 1296, 13]

# Decode back to text
decoded = encoding.decode(tokens)
print(f"Decoded: {decoded}")
```

**Why Tokens Matter:**
- API pricing is per token (input + output)
- Context window is measured in tokens
- Some languages use more tokens per word
- Code often uses many tokens due to symbols

#### Context Windows

The context window is the maximum number of tokens the model can process at once.

| Model | Context Window | Approximate Pages |
|-------|----------------|-------------------|
| GPT-3.5 | 16,384 tokens | ~25 pages |
| GPT-4 | 8,192 or 128,000 | ~12 or ~200 pages |
| GPT-4o | 128,000 | ~200 pages |
| Claude 3.5 | 200,000 | ~300 pages |
| Claude Opus 4.5 | 200,000 | ~300 pages |
| Gemini 1.5 Pro | 2,000,000 | ~3,000 pages |

**What Goes in the Context Window:**

```
[System Prompt]          # Instructions to the model
[Conversation History]   # Previous messages
[Retrieved Context]      # RAG documents
[Current User Message]   # What user just said
─────────────────────────
Total must fit in context window
```

**Context Window Management:**

```python
def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Rough token estimate without tiktoken."""
    return int(len(text) / chars_per_token)

def truncate_to_fit(
    messages: list[dict],
    system_prompt: str,
    max_tokens: int = 8000,
    reserve_output: int = 1000
) -> list[dict]:
    """Truncate conversation history to fit context window."""
    available = max_tokens - reserve_output - estimate_tokens(system_prompt)

    result = []
    used = 0

    # Keep most recent messages
    for msg in reversed(messages):
        msg_tokens = estimate_tokens(msg["content"])
        if used + msg_tokens > available:
            break
        result.insert(0, msg)
        used += msg_tokens

    return result
```

#### Attention: How Models Understand Context

The "attention" mechanism is why transformers understand context so well. Each token can "attend to" every other token, learning which words are relevant to which.

**Example:**
```
"The cat sat on the mat because it was warm."
                              ^
When predicting meaning of "it":
  - Model attends strongly to "mat" (the thing that's warm)
  - Model attends weakly to "cat" (possible but less likely)
  - Context resolves ambiguity
```

**What This Means for You:**
- LLMs understand long-range dependencies
- Related information does not need to be adjacent
- However, attention still degrades over very long contexts
- Place important information at the beginning or end of prompts

---

### Temperature and Sampling Parameters

LLMs do not always pick the highest-probability token. You can control randomness with parameters.

#### Temperature

Temperature controls randomness in output. Lower = more focused, higher = more creative.

```
Temperature = 0.0 (Deterministic)
  Prompt: "The color of the sky is"
  Output: "blue" (always)

Temperature = 0.7 (Balanced)
  Prompt: "The color of the sky is"
  Output: "blue" (usually), "gray" (sometimes), "orange" (rarely)

Temperature = 1.5 (Creative)
  Prompt: "The color of the sky is"
  Output: "cerulean" (sometimes), "infinite" (occasionally), "blue" (sometimes)
```

**Guidelines:**

| Temperature | Use For |
|-------------|---------|
| 0.0 - 0.2 | Code generation, factual Q&A, data extraction |
| 0.3 - 0.5 | Business writing, structured responses |
| 0.6 - 0.8 | Creative writing, brainstorming, conversation |
| 0.9 - 1.2 | Poetry, experimental content |

#### Top-P (Nucleus Sampling)

Top-P limits sampling to tokens that cumulatively reach probability P.

```
Top-P = 0.9:
  Consider only the smallest set of tokens whose probabilities sum to 0.9
  Excludes very unlikely tokens even if temperature would include them
```

**Example:**
```
Probabilities for next word:
  "blue" → 60%
  "gray" → 20%
  "orange" → 10%
  "green" → 5%
  "purple" → 3%
  "infinite" → 2%

Top-P = 0.9:
  Only sample from: blue, gray, orange (sum to 90%)
  Excludes: green, purple, infinite
```

**Guidelines:**

| Top-P | Effect |
|-------|--------|
| 0.1 | Very focused, only top choices |
| 0.5 | Moderate diversity |
| 0.9 | Wide diversity, exclude only rare tokens |
| 1.0 | No filtering (all tokens possible) |

#### Common Configurations

| Task | Temperature | Top-P | Why |
|------|-------------|-------|-----|
| Code generation | 0.0 - 0.2 | 0.1 | Accuracy critical |
| Data extraction | 0.0 | 1.0 | Deterministic output |
| Customer support | 0.3 - 0.5 | 0.9 | Consistent but natural |
| Creative writing | 0.7 - 0.9 | 0.95 | Varied, interesting |
| Brainstorming | 0.9 - 1.2 | 1.0 | Maximum diversity |

**Code Example:**

```python
from openai import OpenAI

client = OpenAI()

# Deterministic (same output every time)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    temperature=0.0
)

# Creative (varied outputs)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a poem about coding"}],
    temperature=0.8,
    top_p=0.95
)
```

---

### Getting Started with LLM APIs

#### OpenAI API

**Setup:**

```bash
pip install openai
```

```python
import os
from openai import OpenAI

# API key from environment variable (recommended)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Or set directly (not recommended for production)
# client = OpenAI(api_key="sk-...")
```

**Basic Chat Completion:**

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)
# "The capital of France is Paris."
```

**Streaming Responses:**

```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### Anthropic (Claude) API

**Setup:**

```bash
pip install anthropic
```

```python
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.content[0].text)
# "The capital of France is Paris."
```

**Streaming:**

```python
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

#### Google Gemini API

**Setup:**

```bash
pip install google-generativeai
```

```python
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-pro")

response = model.generate_content("What is the capital of France?")

print(response.text)
# "The capital of France is Paris."
```

**Streaming:**

```python
response = model.generate_content(
    "Tell me a story",
    stream=True
)

for chunk in response:
    print(chunk.text, end="", flush=True)
```

#### API Pricing Comparison (December 2025)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-4o | $5.00 | $20.00 |
| GPT-4o-mini | $0.60 | $2.40 |
| Claude Sonnet 4 | $3.00 | $15.00 |
| Claude Opus 4.5 | $15.00 | $75.00 |
| Claude Haiku 3.5 | $0.80 | $4.00 |
| Gemini 1.5 Pro | $1.25 | $5.00 |
| Gemini 1.5 Flash | $0.075 | $0.30 |

**Cost Estimation:**

```python
def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float
) -> float:
    """Estimate API cost."""
    input_cost = (input_tokens / 1_000_000) * input_price_per_million
    output_cost = (output_tokens / 1_000_000) * output_price_per_million
    return input_cost + output_cost

# Example: 1000 input tokens, 500 output tokens with GPT-4o
cost = estimate_cost(1000, 500, 5.00, 20.00)
print(f"Estimated cost: ${cost:.4f}")  # $0.015
```

---

## 4. Development Environment

### IDE Comparison: VS Code, Cursor, Claude Code

| Feature | VS Code | Cursor | Claude Code |
|---------|---------|--------|-------------|
| **Base** | Standalone | VS Code fork | CLI + Extensions |
| **AI Integration** | Via extensions | Built-in | External partner |
| **Code Completion** | Copilot extension | Native agent | Via IDE integration |
| **Context Understanding** | Extension-dependent | Deep codebase embedding | Extended context window |
| **Price** | Free + Copilot ($10-19/mo) | $20-200/mo | Usage-based |
| **Best For** | Flexibility, customization | Streamlined AI workflow | Deep analysis, complex refactors |

#### VS Code (Recommended for Beginners)

**Why:**
- Free, extensible, massive community
- Copilot extension provides AI assistance
- Works with all agent frameworks
- Familiar to most developers

**Setup:**
1. Download from [code.visualstudio.com](https://code.visualstudio.com)
2. Install Python extension
3. Install GitHub Copilot (optional, requires subscription)

#### Cursor (Recommended for AI-Heavy Workflows)

**Why:**
- AI assistance built into every feature
- Agent mode handles multi-file changes
- Better for rapid prototyping with AI

**Setup:**
1. Download from [cursor.com](https://cursor.com)
2. Sign in with GitHub
3. Configure model preferences

#### Claude Code (Recommended for Complex Analysis)

**Why:**
- Extended context for large codebases
- Superior for documentation and refactoring
- CLI-first workflow for automation

**Setup:**
```bash
# Install via npm
npm install -g @anthropic-ai/claude-code

# Or use in VS Code via extension
```

#### Hybrid Approach (Used by Professionals)

Many developers use:
- Cursor for daily coding (inline completions, quick edits)
- Claude Code for complex analysis (architecture decisions, large refactors)
- VS Code for specific workflows (debugging, testing)

---

### API Key Management

Never commit API keys to version control. Use environment variables.

#### Local Development with .env Files

```bash
# .env (NEVER commit this file)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
```

```bash
# .gitignore (ALWAYS commit this)
.env
.env.*
*.pem
*.key
```

**Loading in Python:**

```python
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

# Validate keys exist
if not openai_key:
    raise ValueError("OPENAI_API_KEY not set in environment")
```

#### Creating a .env.example Template

```bash
# .env.example (COMMIT this file)
# Copy to .env and fill in your values

# LLM API Keys
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
GEMINI_API_KEY=your-gemini-key-here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Optional
DEBUG=false
LOG_LEVEL=INFO
```

#### Environment Variables Best Practices

| Practice | Why |
|----------|-----|
| Use `.env` files in development | Easy to manage locally |
| Use platform secrets in production | AWS Secrets Manager, Vault, etc. |
| Never commit secrets | Git history is permanent |
| Use `.env.example` for documentation | Team members know what to configure |
| Validate at startup | Fail fast if misconfigured |
| Rotate keys periodically | Limit exposure from leaks |

#### Pre-commit Hook for Secret Detection

```bash
# Install pre-commit
pip install pre-commit

# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']

# Initialize
pre-commit install
detect-secrets scan > .secrets.baseline
```

---

### Local vs Cloud Development Trade-offs

| Factor | Local Development | Cloud Development |
|--------|-------------------|-------------------|
| **Initial Cost** | $0 (existing machine) or $3-10K (GPU) | $0 upfront, pay-as-you-go |
| **Latency** | None for local models | Network round-trip |
| **Privacy** | Complete (data stays local) | Data sent to provider |
| **Scaling** | Limited by hardware | Unlimited (pay more) |
| **Maintenance** | You manage everything | Provider manages infrastructure |
| **Model Access** | Open-source only (locally) | All models via API |
| **Cost at Scale** | Fixed after hardware purchase | Grows linearly with usage |

#### When to Develop Locally

- Prototyping with open-source models (Llama, Mistral)
- Privacy-sensitive data that cannot leave your machine
- High-volume testing (avoid API costs)
- Offline development needs

**Local LLM Setup with Ollama:**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download a model
ollama pull llama3.2:8b

# Run interactively
ollama run llama3.2:8b

# Or via API
curl http://localhost:11434/api/generate \
  -d '{"model": "llama3.2:8b", "prompt": "Hello!"}'
```

```python
# Python client
import ollama

response = ollama.chat(
    model='llama3.2:8b',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response['message']['content'])
```

#### When to Use Cloud APIs

- Production applications serving users
- Need for latest, most capable models (GPT-4, Claude Opus 4.5)
- Team collaboration on shared infrastructure
- Scaling beyond local hardware limits

---

### Essential VS Code Extensions

#### Required Extensions

| Extension | Purpose |
|-----------|---------|
| **Python** (Microsoft) | Python language support, debugging, IntelliSense |
| **Pylance** | Fast, feature-rich Python language server |
| **Python Debugger** | Debugging with breakpoints, step-through |

#### Highly Recommended

| Extension | Purpose |
|-----------|---------|
| **GitHub Copilot** | AI code completion and chat |
| **GitLens** | Enhanced Git integration, blame, history |
| **Error Lens** | Inline error highlighting |
| **autoDocstring** | Generate docstrings from function signatures |

#### For Agent Development

| Extension | Purpose |
|-----------|---------|
| **Thunder Client** | API testing (like Postman, in VS Code) |
| **REST Client** | Send HTTP requests from `.http` files |
| **Docker** | Container management |
| **Remote - SSH** | Develop on remote machines |
| **Dev Containers** | Develop inside containers |

#### For Code Quality

| Extension | Purpose |
|-----------|---------|
| **Black Formatter** | Python code formatting |
| **Ruff** | Fast Python linting |
| **Prettier** | Format JSON, YAML, Markdown |
| **Code Spell Checker** | Catch typos |

#### Recommended settings.json

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/*.egg-info": true
  },
  "editor.rulers": [88, 120],
  "python.testing.pytestEnabled": true
}
```

---

## Skill Check: Ready for Phase 1?

Answer these questions to assess your readiness. If you can answer 8+ correctly, proceed to Phase 1.

### Python Questions

1. **What does `async def` indicate about a function?**
   <details>
   <summary>Answer</summary>
   It defines an asynchronous coroutine function that can use `await` to pause execution while waiting for I/O operations, allowing other code to run in the meantime.
   </details>

2. **What is the difference between `list[str]` and `List[str]`?**
   <details>
   <summary>Answer</summary>
   `list[str]` (lowercase) is the modern Python 3.9+ syntax using built-in types. `List[str]` (uppercase) requires importing from `typing` and is the older syntax. Both mean the same thing: a list containing strings.
   </details>

3. **Why would you use `@dataclass(slots=True)`?**
   <details>
   <summary>Answer</summary>
   To reduce memory usage and improve attribute access speed. Slots prevent the creation of `__dict__` and `__weakref__` for each instance.
   </details>

### Machine Learning Questions

4. **When would you use a rule-based system instead of ML?**
   <details>
   <summary>Answer</summary>
   When rules are simple, clear, and stable; when you have few examples; when decisions must be fully explainable and auditable; when 100% accuracy is required for specific cases.
   </details>

5. **What is the main difference between supervised and unsupervised learning?**
   <details>
   <summary>Answer</summary>
   Supervised learning trains on labeled data (input-output pairs) to predict outputs for new inputs. Unsupervised learning discovers patterns in unlabeled data without predefined answers.
   </details>

### LLM Questions

6. **Approximately how many tokens is 1,000 words of English text?**
   <details>
   <summary>Answer</summary>
   Approximately 1,300-1,500 tokens. The rule of thumb is 1 token ~= 0.75 words, so 1,000 words / 0.75 ~= 1,333 tokens.
   </details>

7. **When would you set temperature to 0.0?**
   <details>
   <summary>Answer</summary>
   When you need deterministic, reproducible output---same input always gives same output. Common for code generation, data extraction, and factual Q&A.
   </details>

8. **What is a context window?**
   <details>
   <summary>Answer</summary>
   The maximum number of tokens the model can process in a single request. This includes system prompt, conversation history, retrieved context, and current message. Ranges from 4K to 2M tokens depending on model.
   </details>

### Development Questions

9. **Why should you never commit `.env` files to git?**
   <details>
   <summary>Answer</summary>
   `.env` files contain secrets like API keys. Git history is permanent---once committed, secrets can be recovered even after deletion. Leaked keys can be exploited for unauthorized access and costs.
   </details>

10. **What is the advantage of using Poetry over pip + requirements.txt?**
    <details>
    <summary>Answer</summary>
    Poetry provides lock files for reproducibility, automatic virtual environment management, separation of dev/production dependencies, and unified configuration in pyproject.toml.
    </details>

### Scoring

| Score | Recommendation |
|-------|----------------|
| 0-3 | Review this document thoroughly, practice the code examples |
| 4-6 | Review weak areas, then try Phase 1 |
| 7-8 | Ready for Phase 1, refer back as needed |
| 9-10 | Proceed to Phase 1 with confidence |

---

## Next Steps

**Proceed to:** [Phase 1: Theoretical Foundations](../phase-1-foundations/theoretical-foundations.md)

Phase 1 covers:
- What makes agents different from chatbots
- Core reasoning patterns (ReAct, Chain-of-Thought, Tree-of-Thought)
- Framework landscape (LangGraph, CrewAI, OpenAI SDK)
- When to use which architecture

---

## Quick Reference

### Python Type Hints Cheatsheet

```python
# Basic types
name: str = "Alice"
age: int = 30
price: float = 19.99
is_active: bool = True

# Optional (may be None)
result: str | None = get_result()

# Collections
names: list[str] = ["Alice", "Bob"]
scores: dict[str, int] = {"Alice": 100}
point: tuple[int, int] = (10, 20)

# Callable
handler: Callable[[str, int], bool]  # Takes str and int, returns bool

# Any type
data: Any = get_data()
```

### Async/Await Cheatsheet

```python
import asyncio

# Define async function
async def fetch_data() -> str:
    await asyncio.sleep(1)
    return "data"

# Run multiple tasks
async def main():
    # Concurrent execution
    results = await asyncio.gather(
        fetch_data(),
        fetch_data(),
        fetch_data()
    )

# Entry point
asyncio.run(main())
```

### API Call Template

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Your question here"}
    ],
    temperature=0.7,
    max_tokens=1000
)

answer = response.choices[0].message.content
```

### Environment Setup Checklist

```bash
# 1. Create project directory
mkdir my-agent-project && cd my-agent-project

# 2. Initialize with uv or poetry
uv init  # or: poetry init

# 3. Add core dependencies
uv add pydantic httpx openai anthropic python-dotenv

# 4. Add dev dependencies
uv add --group dev pytest black mypy

# 5. Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# 6. Create .gitignore
echo ".env\n.venv\n__pycache__" > .gitignore

# 7. Verify setup
uv run python -c "import pydantic; print('Ready!')"
```

---

## Additional Resources

### Official Documentation
- [Python 3.12 Docs](https://docs.python.org/3.12/)
- [Pydantic Docs](https://docs.pydantic.dev/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [Anthropic API Reference](https://docs.anthropic.com/)
- [Google Gemini API](https://ai.google.dev/docs)

### Learning Resources
- [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners)
- [Real Python Tutorials](https://realpython.com/)
- [FastAPI + Pydantic Tutorial](https://fastapi.tiangolo.com/tutorial/)

### Community
- [LangChain Discord](https://discord.gg/langchain)
- [OpenAI Developer Forum](https://community.openai.com/)
- [Anthropic Discord](https://discord.gg/anthropic)

---

**Last Updated:** 2025-12-27

**Proceed to:** [Phase 1: Theoretical Foundations](../phase-1-foundations/theoretical-foundations.md)
