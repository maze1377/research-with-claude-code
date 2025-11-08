# API Optimization Guide
## Practical Techniques for OpenAI and Anthropic Models (2024-2025)

**Purpose:** Actionable best practices for maximizing performance, minimizing costs, and ensuring reliability when using OpenAI and Anthropic APIs for multi-agent systems.

**Last Updated:** 2025-11-08

**Note:** Detailed code examples available in `production-cookbook.md`

---

## Table of Contents

1. [Model Selection Strategy](#model-selection-strategy)
2. [OpenAI API Best Practices](#openai-api-best-practices)
3. [Anthropic API Best Practices](#anthropic-api-best-practices)
4. [Cost Optimization Strategies](#cost-optimization-strategies)
5. [Quick Reference](#quick-reference)

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

**Code Example:** See `production-cookbook.md` → Structured Outputs section

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

**Code Example:** See `production-cookbook.md` → Function Calling section

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

**Code Example:** See `production-cookbook.md` → Streaming section

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

**Code Example:** See `production-cookbook.md` → Prompt Caching section

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

**Code Example:** See `production-cookbook.md` → Tool Use section

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

### 2. Model Cascading

**Strategy:**
1. Assess complexity with cheap model (gpt-4o-mini)
2. Route simple tasks to cheap models
3. Escalate complex tasks to expensive models
4. Track accuracy vs cost trade-offs

**Typical routing:**
- Complexity < 0.3 → gpt-4o-mini/Claude Haiku
- Complexity 0.3-0.7 → gpt-4o/Claude Sonnet
- Complexity > 0.7 → o1/Claude Opus

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

### 8. Context Window Management

**Strategies:**
- Summarize old messages
- Keep only recent N messages
- Extract key facts from history
- Use semantic compression

**For long conversations:**
- Rolling window (last 10-20 messages)
- Periodic summarization
- Extract entities/facts to system prompt
- Prune redundant information

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

### Model Selection Quick Guide

**OpenAI:**
- `gpt-4o-mini`: High-volume, simple tasks, routing, classification
- `gpt-4o`: General-purpose, balanced performance
- `gpt-4o-2024-08-06`: Guaranteed JSON schema adherence
- `o1-preview`: Complex reasoning, STEM problems, strategic planning
- `o1-mini`: Coding, math, science (cheaper than o1-preview)

**Anthropic:**
- `claude-haiku-4`: Speed-critical, high-volume, simple tasks
- `claude-sonnet-4.5`: General-purpose, long-form writing, analysis
- `claude-3-7-sonnet`: Extended thinking for complex problems
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

## Conclusion

**The Three Pillars of Production API Usage:**

1. **Smart Selection** - Right model for the right task
   - Use decision tree and cost tables
   - Heterogeneous agent design
   - Cascade from cheap to expensive

2. **Cost Management** - The cheapest call is the one you don't make
   - Cache aggressively (90% savings on repeated content)
   - Compress context (40-60% token reduction)
   - Batch intelligently (lower overhead)
   - Monitor continuously (prevent budget overruns)

3. **Reliability Engineering** - Build for failure
   - Retry with exponential backoff
   - Fallback to alternative models
   - Circuit breakers prevent cascades
   - Validate all inputs and outputs

**Expected Outcomes:**
- **Cost**: 50-80% reduction through optimization
- **Reliability**: 99.9%+ success rate
- **Latency**: Sub-second for most operations
- **Scalability**: Thousands of requests/minute

**Next Steps:**
- Implement token tracking (day 1)
- Add prompt caching (Anthropic) or response caching (OpenAI)
- Set up model cascading for your use cases
- Monitor and iterate based on production metrics

**See Also:**
- `production-cookbook.md` - Complete code examples
- `patterns-and-antipatterns.md` - Common pitfalls and solutions
- `langgraph-research.md` - Multi-agent orchestration patterns
