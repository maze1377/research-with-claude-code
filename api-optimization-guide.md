# API Optimization Guide
## Practical Techniques for OpenAI, Anthropic, Google and DeepSeek Models (December 2025)

**Purpose:** Actionable best practices for maximizing performance, minimizing costs, and ensuring reliability when using LLM APIs for multi-agent systems.

**Last Updated:** 2025-12-25

**Note:** Detailed code examples available in `agentic-systems-cookbook.md`

---

## December 2025 Key Stats

| Metric | Value | Source |
|--------|-------|--------|
| API price drop (year over year) | 75% | $10/1M → $2.50/1M |
| DeepSeek cost advantage | 94% cheaper | vs Claude Opus 4.5 |
| Prompt caching savings | 90% | All major providers |
| Tool Search + defer_loading | 85% token reduction | Anthropic |
| Best cascade routing improvement | 14% | Cost-performance |

---

## Table of Contents

1. [Model Selection Strategy](#model-selection-strategy)
2. [OpenAI API Best Practices](#openai-api-best-practices)
3. [Anthropic API Best Practices](#anthropic-api-best-practices)
4. [Google Gemini Best Practices](#google-gemini-best-practices)
5. [DeepSeek Cost Optimization](#deepseek-cost-optimization)
6. [Cost Optimization Strategies](#cost-optimization-strategies)
7. [Quick Reference](#quick-reference)

---

## Model Selection Strategy

### OpenAI Model Lineup (December 2025)

| Model | Best For | Input Cost | Output Cost | Context Window | Key Features |
|-------|----------|------------|-------------|----------------|--------------|
| GPT-5 | Coding, agentic tasks | $1.25/1M | $10/1M | 400K | Frontier capability |
| GPT-5 mini | Fast, defined tasks | $0.25/1M | $2/1M | 400K | 5x cheaper than GPT-5 |
| GPT-5 nano | Summarization, classification | $0.05/1M | $0.40/1M | 128K | Cheapest, fastest |
| GPT-5 Pro | Complex reasoning | $15/1M | $120/1M | 400K | Extended thinking |
| o3 | Advanced reasoning | Premium | Premium | 200K | 96.7% AIME, 87.5% ARC-AGI |
| o4-mini | Efficient reasoning | $3/1M | $12/1M | 128K | 99.5% AIME with Python |
| gpt-4o-2024-08-06 | Structured outputs | $2.50/1M | $10/1M | 128K | 100% schema adherence |

**Note:** Cache pricing is 90% cheaper ($0.125/1M for GPT-5 cached input)

### Anthropic Model Lineup (December 2025)

| Model | Best For | Input Cost | Output Cost | Context Window | Key Features |
|-------|----------|------------|-------------|----------------|--------------|
| Claude Opus 4.5 | Complex tasks, coding | **$5/1M** | **$25/1M** | 200K | 80.9% SWE-bench, 49 t/s |
| Claude Sonnet 4.5 | Balanced performance | $3/1M | $15/1M | 200K | Fast, capable |
| Claude Haiku 4.5 | Speed/volume | $1/1M | $5/1M | 200K | 90% of Sonnet quality, 4-5x faster |

**Note:** Claude Opus 4.5 price **reduced** from $15/$75 to $5/$25 (67% cheaper)

### Google Gemini Lineup (December 2025)

| Model | Best For | Input Cost | Output Cost | Context Window | Key Features |
|-------|----------|------------|-------------|----------------|--------------|
| Gemini 3 Pro | Reasoning, multimodal | $2/1M | $8/1M | **1M** | 91.9% GPQA Diamond |
| Gemini 3 Flash | Fast, efficient | $0.50/1M | $3/1M | 1M | 218 t/s, 3x faster than 2.5 Pro |
| Gemini 2.5 Flash | Budget multimodal | $0.30/1M | $2.50/1M | 1M | Thinking levels |

### DeepSeek Lineup (December 2025)

| Model | Best For | Input Cost | Output Cost | Context Window | Key Features |
|-------|----------|------------|-------------|----------------|--------------|
| DeepSeek V3.2 | High-volume, cost-sensitive | **$0.28/1M** | **$0.42/1M** | 128K | 96% AIME, 73% SWE-bench |

**DeepSeek Cost Comparison:**
- 100K input + 10K output: $0.006 (DeepSeek) vs $1.30 (GPT-5) = **216x cheaper**
- 1M cached input + 200K output: $0.106 vs $3.25 = **31x cheaper**

### Selection Decision Tree (December 2025)

```
Need complex reasoning?
├─ Yes → o3, GPT-5 Pro, or Claude Opus 4.5 (extended thinking)
└─ No
   ├─ Need strict schema adherence?
   │  └─ Yes → gpt-4o-2024-08-06 or Gemini 3 (enhanced JSON schema)
   ├─ Need 1M+ context?
   │  └─ Yes → Gemini 3 Pro/Flash (1M context)
   ├─ High-volume, cost-sensitive?
   │  └─ Yes → DeepSeek V3.2 ($0.28/1M) or GPT-5 nano
   └─ Balanced performance?
      └─ Yes → GPT-5 mini, Claude Sonnet 4.5, or Gemini 3 Flash
```

### Speed vs Quality Trade-offs (December 2025)

| Task | Recommended Model | Speed | Cost |
|------|------------------|-------|------|
| Interactive chat | GPT-5.2 | 187 t/s | $$ |
| Real-time analysis | Gemini 3 Flash | 218 t/s | $ |
| Complex reasoning | Claude Opus 4.5 | 49 t/s | $$$ |
| Bulk processing | DeepSeek V3.2 | 120 t/s | ¢ |
| Code generation | Claude Opus 4.5 | 49 t/s | $$$ |
| Classification | GPT-5 nano | 300+ t/s | ¢ |

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

**Benefits:**
- 100% schema adherence (vs ~40% with JSON mode on older models)
- No retry logic needed for parsing
- Type-safe outputs
- Eliminates validation overhead

**Implementation:**
- Use `client.beta.chat.completions.parse()` with Pydantic models
- Model: `gpt-4o-2024-08-06` required
- Define clear schemas with proper types and descriptions

**Limitations:**
- max_tokens cutoff can cause incomplete responses
- Model can still hallucinate values (validate content separately)
- Not all JSON schemas supported (check documentation)

**Code Example:** See `agentic-systems-cookbook.md` → Structured Outputs section

### 2. Function Calling Optimization

**Problem: Too many tools reduce accuracy**

**Best Practices:**
- Limit to 5-10 tools per call (use RAG to select relevant tools from larger sets)
- Provide detailed, specific tool descriptions
- Include use case examples in descriptions
- Specify exact parameter types and constraints
- Use `parallel_tool_calls=True` for independent operations

**Tool Description Structure:**
```
- Name: Descriptive, action-oriented
- Description: What it does, when to use, what NOT to use for, what it returns
- Parameters: Detailed descriptions with types, enums, ranges
- Examples: Edge cases and typical usage
```

**Code Example:** See `agentic-systems-cookbook.md` → Function Calling section

### 3. Parallel Function Calling

**Enable for independent operations:**
- Weather + stock price
- Multiple database queries
- Concurrent API calls

**Benefits:**
- Faster execution (parallel vs sequential)
- Fewer API calls
- Lower latency

### 4. Response Streaming

**Use streaming for:**
- Long-form content generation
- Real-time user feedback
- Progressive rendering
- Improved perceived performance

**Code Example:** See `agentic-systems-cookbook.md` → Streaming section

### 5. Prompt Caching

**Note:** OpenAI doesn't currently support prompt caching like Anthropic.

**Workarounds:**
- Cache embeddings for repeated content with `@lru_cache`
- Application-layer caching of responses (Redis, memcached)
- Cache only deterministic responses (temperature=0)

---

## Anthropic API Best Practices

### 1. Prompt Caching (Major Cost Saver)

**Automatic caching for repeated content:**

**Pricing Impact:**
- Regular input: $3/1M tokens
- Cached input: $0.30/1M tokens (90% discount)
- Cache write: $3.75/1M tokens (one-time)
- Cache lifetime: 5 minutes

**Best Practices:**
- Cache large system prompts (place `cache_control` on blocks)
- Cache codebase/document context
- Cache few-shot examples
- Place `cache_control` on last eligible block
- Minimum 1024 tokens to cache (smaller content not cached)

**Use Cases:**
- Repeated queries against same codebase
- Multiple questions about same document
- Consistent system prompts across conversations
- Few-shot examples that don't change

**Code Example:** See `agentic-systems-cookbook.md` → Prompt Caching section

### 2. Extended Thinking (Claude 3.7+)

**Enable for complex reasoning tasks:**

**When to Use:**
- Complex mathematical problems
- Multi-step reasoning tasks
- Code debugging and optimization
- Strategic planning

**Cost Consideration:**
- Thinking tokens count toward input pricing
- Can significantly increase cost (2-5x)
- Use only when complexity justifies cost
- Set appropriate `budget_tokens` limits

**Implementation:**
- Model: `claude-3-7-sonnet-20250219`
- Set `thinking.type = "enabled"`
- Allocate `budget_tokens` (e.g., 10000)
- Access thinking blocks for transparency/debugging

### 3. Tool Use Best Practices

**Key Principles:**
- Detailed descriptions (what, when, what returns)
- Proper input schemas with types
- Handle `stop_reason == "tool_use"`
- Return results in structured format
- Continue conversation with tool results

**Tool Response Handling:**
- Extract tool_use blocks from response.content
- Execute tool with provided inputs
- Format results for LLM consumption (human-readable > raw JSON)
- Continue with tool_result message

**Code Example:** See `agentic-systems-cookbook.md` → Tool Use section

### 4. Advanced Tool Use (November 2025)

**Problem:** Large tool sets (50+ tools) consume 75%+ of context window

**Solution 1: Tool Search with defer_loading**
```python
# Before: 150,000 tokens (50 tools × 3,000 tokens each)
# After: 17,000 tokens (search infrastructure + 5 discovered tools)
# Reduction: 85%

tools = load_tools_with_defer(
    tool_directory="/tools",
    defer_loading=True,  # Don't load all tool definitions upfront
    search_tool=True     # Enable Tool Search Tool
)
```

**Solution 2: MCP Code Execution**
```python
# Agents discover tools by exploring filesystem
# Load tools on demand, not all at once

# Before: 150,000 tokens for all tool definitions
# After: 2,000 tokens (filesystem exploration)
# Reduction: 98.7%
```

**Solution 3: Structured Skill Patterns**
- Organize tools into skill folders with SKILL.md files
- Agents reference higher-level skills, not individual tools
- Build toolbox of reusable capabilities over time

**Impact:** 85-98% token reduction for tool-heavy applications

### 4. System Prompt Best Practices

**Anthropic's recommended structure:**

1. **Role** - Define expertise and background
2. **Task** - Specific objective
3. **Context** - Environment, constraints, priorities
4. **Examples** - Show desired behavior (use `<example>` tags)
5. **Constraints** - Guardrails and requirements

**Best Practices:**
- Use XML tags for structure (`<example>`, `<context>`, etc.)
- Prioritize constraints clearly (security > correctness > performance)
- Show both good and bad examples
- Explain WHY, not just WHAT
- Update prompts based on failure patterns

### 5. Message Batching

**Combine independent queries:**
- Batch 5-10 similar questions in one call
- Format as numbered list
- Parse structured responses
- Reduces API overhead significantly

**When NOT to batch:**
- Dependent/sequential operations
- Need individual error handling
- Responses > max_tokens limit

---

## Google Gemini Best Practices (2025)

### 1. Dynamic Thinking Modulation

**Four granular thinking levels:**

| Level | Response Time | Use Case |
|-------|---------------|----------|
| Minimal | 3-5 seconds | Quick answers |
| Low | 10-15 seconds | Standard tasks |
| Medium | 20-30 seconds | Complex analysis |
| High | 45+ seconds | Deep reasoning |

**Impact:** 30% token reduction vs binary thinking implementations

### 2. Million-Token Context

**Best practices for 1M context:**
- Use Gemini 3 Pro/Flash for repository-scale analysis
- Long-context pricing: $0.075/1M input (Flash, >200K context)
- Cheaper than filling multiple smaller contexts

### 3. Enhanced Structured Outputs (December 2025)

**New JSON Schema features:**
- `anyOf` for conditional structures
- `$ref` for recursive schemas
- `minimum`/`maximum` for numeric constraints
- Key ordering preserved from schema
- Works with Pydantic and Zod

---

## DeepSeek Cost Optimization

### When to Use DeepSeek

**Ideal for:**
- High-volume processing (94% cost savings)
- Non-critical classification tasks
- Bulk content generation
- Development/testing (before production models)

**Accuracy considerations:**
- 73.1% SWE-bench (vs 80.9% Claude Opus 4.5)
- 7.8% accuracy gap may require additional review
- Calculate: API savings vs error correction costs

### Cost-Sensitive Architecture

```python
def route_by_criticality(task):
    if task.critical or task.requires_accuracy > 0.9:
        return "claude-opus-4.5"  # High accuracy
    elif task.interactive:
        return "gpt-5.2"          # Fast response
    elif task.high_volume:
        return "deepseek-v3.2"    # 216x cheaper
    else:
        return "gpt-5-mini"       # Balanced
```

---

## Cost Optimization Strategies

### 1. Token Usage Monitoring

**Track metrics:**
- Input/output tokens per call
- Cumulative costs by model
- Cost per agent/task type
- Daily/weekly spending trends

**Set alerts for:**
- Individual calls > $0.10
- Daily budget thresholds
- Unusual token spikes
- High error rates

### 2. Model Cascading & Cascade Routing (2025)

**Cascade Routing** (research-proven 14% improvement):
1. Use quality estimators to assess response quality
2. Route simple tasks to cheap models (DeepSeek, GPT-5 nano)
3. Escalate if quality score below threshold
4. Combine routing + cascading for optimal cost-performance

**December 2025 routing:**
- Complexity < 0.3 → DeepSeek V3.2/GPT-5 nano
- Complexity 0.3-0.7 → GPT-5 mini/Claude Sonnet 4.5
- Complexity > 0.7 → Claude Opus 4.5/GPT-5 Pro
- Complex reasoning → o3/o4-mini

**Key Insight:** Quality estimator accuracy is the primary determinant of routing success

### 3. Response Caching

**Cache when:**
- Temperature = 0 (deterministic)
- Same query repeated
- Similar context patterns
- Frequently accessed data

**Caching strategies:**
- Redis/Memcached for response storage
- Hash key: model + messages + temperature
- TTL: 1-24 hours depending on freshness requirements
- Invalidate on data updates

**Expected savings:** 40-70% reduction on repeated queries

### 4. Prompt Compression

**Techniques:**
- Use embedding-based retrieval for context selection
- TF-IDF scoring to select relevant chunks
- Keep only top-k most relevant sentences
- Target 20-40% of original context size

**When to compress:**
- Context > 10K tokens
- Only portions relevant to query
- Multiple documents/sources
- Historical conversation context

### 5. Batching Strategies

**Batch similar requests:**
- Multiple classification tasks
- Bulk content generation
- Repeated analysis tasks
- Independent evaluations

**Benefits:**
- Fewer API calls (lower overhead)
- Better token efficiency
- Reduced latency (overall)
- Lower rate limit impact

### 6. Selective Streaming

**Stream only when:**
- User needs progressive output
- Long-form generation (> 500 tokens)
- Real-time interaction required

**Don't stream for:**
- Batch processing
- API integrations
- Short responses
- Background tasks

### 7. Smart Token Limits

**Set appropriate max_tokens:**
- Short answers: 100-300 tokens
- Medium responses: 500-1000 tokens
- Long-form: 2000-4000 tokens
- Don't use defaults blindly

**Benefits:**
- Faster responses
- Lower costs
- Prevents rambling
- Better latency

### 8. Context Window Management (2025 Research)

**JetBrains Research Findings:**
- Unmanaged context consumes 2-3x more tokens
- Observation masking: 52% cost reduction + 2.6% accuracy boost
- LLM summarization: Similar savings but adds API overhead

**Two key techniques:**

1. **Observation Masking:**
```python
# Replace old observations with placeholders
# Maintains semantic structure without reprocessing
for i, obs in enumerate(old_observations):
    if i < len(observations) - 5:  # Keep last 5
        observations[i] = "[Observation masked]"
```

2. **Distraction Ceiling (Databricks Research):**
- Effective limit: ~32K tokens (even for 1M context models)
- Performance degrades beyond ceiling
- Solution: Compress to stay under 30K active tokens

**For long conversations:**
- Rolling window (last 10-20 messages)
- Periodic summarization with dedicated summarizer
- Extract entities/facts to persistent memory
- Prune redundant information aggressively

### 9. Tool Selection Optimization

**For large tool sets (> 20 tools):**
- Use RAG to select top-k relevant tools
- Embed tool descriptions offline
- Match user query to tool embeddings
- Pass only 5-10 most relevant tools

**Savings:** Reduces context size by 60-80%

### 10. Heterogeneous Model Assignment

**Assign models by task complexity:**
- Routing/coordination: cheapest models
- Content generation: mid-tier models
- Critical evaluation: expensive models
- Validation: structured output models

**Example cost reduction:** 50-70% vs using same expensive model for all tasks

---

## Quick Reference

### Model Selection Quick Guide (December 2025)

**OpenAI:**
- `GPT-5 nano`: Classification, summarization, cheapest ($0.05/1M)
- `GPT-5 mini`: Balanced, defined tasks ($0.25/1M)
- `GPT-5`: Coding, agentic tasks ($1.25/1M)
- `GPT-5 Pro`: Extended thinking, complex ($15/1M)
- `o3`: Frontier reasoning (96.7% AIME)
- `o4-mini`: Efficient reasoning ($3/1M)

**Anthropic:**
- `Claude Haiku 4.5`: Speed-critical, 90% Sonnet quality, 4-5x faster ($1/1M)
- `Claude Sonnet 4.5`: General-purpose, balanced ($3/1M)
- `Claude Opus 4.5`: Highest quality, 80.9% SWE-bench ($5/1M)

**Google:**
- `Gemini 3 Flash`: Fastest (218 t/s), multimodal ($0.50/1M)
- `Gemini 3 Pro`: 1M context, 91.9% GPQA Diamond ($2/1M)

**DeepSeek:**
- `DeepSeek V3.2`: Cost-optimal, 94% cheaper ($0.28/1M)

### Cost-Performance Trade-off (December 2025)

```
High Performance ↑
│
│  o3 ($$$$$)
│  GPT-5 Pro ($$$$)
│  Claude Opus 4.5 ($$$)  ← Now 67% cheaper!
│  Gemini 3 Pro ($$$)
│  GPT-5 ($$)
│  Claude Sonnet 4.5 ($$)
│  Gemini 3 Flash ($)
│  GPT-5 mini ($)
│  Claude Haiku 4.5 ($)
│  GPT-5 nano (¢)
│  DeepSeek V3.2 (¢)  ← 216x cheaper than GPT-5
│
└─────────────────────→ Low Cost
```

### Key Optimization Principles

**Cost Reduction (50-80% savings possible):**
1. Cache aggressively (Anthropic prompt caching, application-layer caching)
2. Route intelligently (cheap models for simple tasks)
3. Compress context (keep only relevant information)
4. Batch requests (reduce API overhead)
5. Monitor spending (set budgets and alerts)

**Reliability (99.9%+ success rate):**
1. Exponential backoff retry (handle rate limits)
2. Fallback models (degrade gracefully)
3. Circuit breakers (prevent cascading failures)
4. Timeout management (fail fast)
5. Validation (check outputs before use)

**Performance (sub-second latency):**
1. Stream responses (progressive output)
2. Parallel calls (concurrent operations)
3. Reduce max_tokens (only request what you need)
4. Use faster models (mini/haiku for simple tasks)
5. Connection pooling (reuse connections)

### Error Handling Principles

**Retry Strategy:**
- Exponential backoff: 1s, 2s, 4s, 8s, 16s
- Max 3-5 retries
- Add jitter (±1s random)
- Different handling for rate limits vs API errors

**Fallback Strategy:**
- Primary → Secondary → Tertiary model
- Expensive → Cheap degradation
- Fail gracefully with partial results
- Alert on fallback usage

**Circuit Breaker:**
- Track consecutive failures
- Open circuit after 5 failures
- Half-open after timeout (60s)
- Close on successful call

**Timeout Management:**
- Set reasonable limits (30-60s)
- Cancel on timeout
- Log slow calls
- Alert on patterns

### Monitoring Essentials

**Track these metrics:**
- Request count (by model, endpoint, agent)
- Success/failure rates
- Latency (p50, p95, p99)
- Token usage (input/output)
- Cost (per call, cumulative)
- Error types and frequencies

**Set alerts for:**
- Error rate > 5%
- Latency > 10s
- Daily cost > budget
- Individual call > $1
- Unusual traffic patterns

**Review regularly:**
- Top cost contributors
- Slow queries
- Failed requests
- Model performance comparison
- Optimization opportunities

---

## Hallucination Reduction (Advanced)

### Strategy 1: Retrieval-Augmented Generation (RAG)
Ground responses in retrieved facts rather than parametric knowledge.

**Implementation**:
```
1. Chunk documents (500-1000 tokens)
2. Embed and index in vector store
3. Retrieve top-k relevant chunks
4. Inject as context: "Based on: {chunks}"
5. Instruct: "Only use provided information"
```

**Impact**: 40-60% hallucination reduction

### Strategy 2: Multi-Agent Validation
Use separate critic agent to verify claims.

**Pattern**:
```
Generator → Output → Critic → [Valid/Invalid + Feedback]
                         ↓
              If Invalid → Generator (with feedback)
```

**Key**: Critic has access to ground truth sources

### Strategy 3: Prompt Engineering for Accuracy
```
# Effective patterns:
- "If unsure, say 'I don't know'"
- "Cite sources for each claim"
- "Distinguish facts from opinions"
- "Rate confidence 1-10 for each statement"
```

### Strategy 4: Fine-Tuning on Curated Data
When prompting isn't enough:
- Fine-tune on verified Q&A pairs
- Use constitutional AI training
- RLHF on accuracy metrics

### Strategy 5: Multi-Agent Collaborative Filtering (MCF)
Agents collaboratively filter and verify each other's outputs.

**Pattern**:
```
Agent A → Output → Agent B (Filter) → Agent C (Verify) → Final Output
                        ↓                    ↓
                   Feedback to A        Feedback to B
```

**Impact**: 4-8% accuracy improvement over single-agent

### Strategy 6: Adversarial Debate
Two agents argue opposing positions to find truth.

**Pattern**:
```
Generator A (Pro) ←→ Generator B (Con)
         ↓                  ↓
         └──→ Judge Agent ←─┘
                   ↓
            Final Verdict
```

**Impact**: 4-6% higher accuracy, 30% fewer factual errors

### Strategy 7: ICE (Iterative Consensus Ensemble)
Multiple agents iterate until reaching consensus.

**Pattern**:
```
Round 1: Agent1, Agent2, Agent3 → [Output1, Output2, Output3]
         ↓
Compare outputs → Disagreement?
         ↓ Yes
Round 2: Share outputs, re-generate with context
         ↓
Repeat until consensus or max rounds
```

**Impact**: Up to 27% accuracy improvement
**Note**: Ensemble size saturates at 3-5 agents (diminishing returns beyond)

### Strategy 8: Cross-Validation Voting
Simple but effective voting across multiple agents.

**Pattern**:
```
Query → [Agent1, Agent2, Agent3] → [Response1, Response2, Response3]
                    ↓
          Majority voting / Weighted average
                    ↓
              Final Response
```

**Impact**: 40% accuracy boost with 3-5 agents
**Best for**: Classification, factual questions, structured outputs

---

## Performance & Speed Optimization

### 1. Streaming Responses
```
# Start showing output immediately
for chunk in client.chat.completions.create(stream=True):
    yield chunk.choices[0].delta.content
```
**Benefit**: Perceived latency drops 80%+

### 2. Parallel API Calls
```
# Run independent calls concurrently
import asyncio
results = await asyncio.gather(
    call_model(prompt1),
    call_model(prompt2),
    call_model(prompt3)
)
```
**Benefit**: 3x faster for independent tasks

### 3. Model Cascading for Speed
```
# Fast model first, escalate if needed
response = fast_model(query)          # gpt-4o-mini: 50ms
if needs_more_depth(response):
    response = slow_model(query)      # claude-opus: 500ms
```

### 4. Response Caching
```
# Cache deterministic queries
cache_key = hash(system_prompt + user_query)
if cache_key in cache:
    return cache[cache_key]           # 0ms
```

### 5. Context Pruning
- Keep only relevant history (last 5-10 messages)
- Summarize old context instead of including verbatim
- Remove redundant system instructions

### 6. M1-Parallel Orchestration (Multi-Agent)
Optimized parallel agent execution with dependency tracking.

**Pattern**:
```
Task Graph Analysis → Identify Independent Tasks
         ↓
Parallel Dispatch → [Agent1, Agent2, Agent3] (concurrent)
         ↓
Dependency Resolution → Sequential for dependent tasks
         ↓
Result Aggregation
```

**Impact**: 1.8-2.2x speedup over sequential execution
**Best for**: Multi-agent workflows with independent subtasks

### 7. Agentic Plan Caching
Cache execution plans for repeated patterns.

**Implementation**:
```
1. Hash task pattern + context signature
2. Check plan cache before planning
3. If hit: Execute cached plan directly
4. If miss: Generate plan, cache for future
5. Invalidate on context/tool changes
```

**Impact**:
- 50.31% cost reduction
- 27.28% latency reduction
- Works best with stable task patterns

### 8. Speculative Actions
Pre-compute likely next actions while waiting for user/system.

**Pattern**:
```
Current State → Predict likely next actions (top 3)
         ↓
Pre-execute in background (low priority)
         ↓
User action arrives → Match prediction?
         ↓ Yes               ↓ No
Use cached result    Discard, execute fresh
```

**Impact**: Real-time performance for common workflows
**Caution**: Increases compute cost; use for high-value paths only

### 9. KV Cache Routing
Route requests to maximize cache hits across distributed LLM instances.

**Pattern**:
```
Request → Hash prefix (system prompt + context)
         ↓
Route to instance with matching KV cache
         ↓
Higher hit rate → Lower latency + cost
```

**Impact**: 87% cache hit rate achievable
**Requires**: Load balancer with cache-aware routing

### 10. DAG-based Task Scheduling (LLMCompiler)
Use directed acyclic graph for optimal task ordering.

**Pattern**:
```
       Task DAG
         ┌─────┐
         │  A  │
         └──┬──┘
       ┌────┴────┐
       ↓         ↓
    ┌─────┐   ┌─────┐
    │  B  │   │  C  │  ← Execute in parallel
    └──┬──┘   └──┬──┘
       └────┬────┘
            ↓
         ┌─────┐
         │  D  │  ← Wait for B, C
         └─────┘
```

**Impact**: Optimal parallelization with dependency respect
**Best for**: Complex multi-step agent workflows

---

## Combined Optimization Strategies

### Production-Ready Configuration (December 2025)

| Optimization | Expected Improvement | Source |
|--------------|---------------------|--------|
| Prompt caching | 90% cost reduction on repeated | All providers |
| Model cascading | 40-60% cost reduction | Production data |
| Cascade routing | 14% better cost-performance | Research |
| Tool Search + defer_loading | 85% token reduction | Anthropic |
| MCP code execution | 98.7% token reduction | Anthropic |
| Parallel calls | 2-5x speed improvement | Benchmarks |
| Streaming | 80% perceived latency reduction | UX studies |
| RAG grounding | 40-60% hallucination reduction | Production |
| Multi-agent validation | 85→99% accuracy | Research |
| Observation masking | 52% cost + 2.6% accuracy | JetBrains |
| Context distraction ceiling | 30K effective tokens | Databricks |
| DeepSeek routing | 216x cost reduction | API comparison |
| Dynamic thinking modulation | 30% token reduction | Google |

### Recommended Stack
```
Layer 1: Cache (Exact + Semantic)
    ↓
Layer 2: Router (Classify complexity)
    ↓
Layer 3: Model Selection (Cheap → Expensive cascade)
    ↓
Layer 4: Execution (Parallel where possible)
    ↓
Layer 5: Validation (Critic agent for high-stakes)
```

---

## Conclusion

**The Four Pillars of Production API Usage (December 2025):**

1. **Smart Selection** - Right model for the right task
   - Use December 2025 decision tree and cost tables
   - Multi-provider strategy: DeepSeek for bulk, GPT-5.2 for speed, Claude Opus 4.5 for quality
   - Cascade routing with quality estimators (14% improvement)

2. **Cost Management** - The cheapest call is the one you don't make
   - Cache aggressively (90% savings on repeated content)
   - Compress context (observation masking: 52% reduction)
   - Use DeepSeek for high-volume (216x cheaper than GPT-5)
   - Tool Search + defer_loading (85% token reduction)

3. **Context Engineering** - Respect the 32K distraction ceiling
   - Even 1M context models degrade beyond 32K active tokens
   - Use observation masking, not raw context growth
   - Dynamic thinking modulation (30% savings)
   - MCP code execution (98.7% token reduction)

4. **Reliability Engineering** - Build for failure
   - Retry with exponential backoff
   - Multi-provider fallback (Claude → GPT → DeepSeek)
   - Circuit breakers prevent cascades
   - Validate all inputs and outputs

**Expected Outcomes (December 2025):**
- **Cost**: 80-95% reduction with DeepSeek routing + caching
- **Reliability**: 99.9%+ success rate
- **Latency**: Sub-second with Gemini 3 Flash (218 t/s)
- **Scalability**: Thousands of requests/minute
- **Quality**: 80.9% SWE-bench with Claude Opus 4.5

**Next Steps:**
1. Implement cascade routing with quality estimators
2. Add prompt caching (90% savings on repeated content)
3. Set up DeepSeek for bulk/non-critical workloads
4. Implement Tool Search + defer_loading for large tool sets
5. Monitor context size, stay under 32K active tokens
6. Track costs across providers, optimize continuously

**Key 2025 Insight:** "Teams getting the best results aren't just those with the cleverest prompts—they're those who figured out how to measure whether output was actually good."

**See Also:**
- `agentic-systems-cookbook.md` - Complete code examples
- `patterns-and-antipatterns.md` - Common pitfalls and solutions
- `multi-agent-patterns.md` - Multi-agent orchestration patterns
- `2025-updates.md` - Latest model releases and features
