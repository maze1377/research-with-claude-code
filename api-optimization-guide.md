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
7. [Context Engineering](#context-engineering)
8. [Quick Reference](#quick-reference)

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

## Context Engineering

**Context engineering** is the discipline of designing, curating, and optimizing the contextual information provided to LLM agents—distinct from memory systems and more advanced than prompt engineering. While prompt engineering optimizes single prompts and memory systems focus on persistent storage, context engineering dynamically assembles and prioritizes ephemeral context for each inference step in agent loops.

**Key Insight:** "The primary job of AI engineers is context engineering—providing the right information and tools in the optimal format to maintain coherence, reduce hallucinations, and handle multi-step tasks."

### Why Context Engineering ≠ Memory Systems

| Aspect | Memory Systems | Context Engineering |
|--------|----------------|---------------------|
| **Focus** | Persistent storage, recall across sessions | Dynamic assembly for current inference |
| **Scope** | Long-term continuity | Per-step optimization |
| **Goal** | Remember everything important | Select only what's relevant now |
| **Token Strategy** | Store everything, retrieve when needed | Curate tokens for each call |
| **Failure Mode** | Forgetting important info | Context rot from overload |

### The Context Rot Problem

**Context rot** is the phenomenon where LLM accuracy degrades as context length increases—even with 200K+ context windows.

**Research Findings:**
- Studies across 18 models show non-uniform context use
- Performance drops sharply beyond ~32K tokens (even with 1M context windows)
- One distractor (irrelevant similar info) can tank accuracy
- Complex reasoning (multi-hop questions) degrades faster than simple tasks
- "Lost in the middle" effect: models favor early and recent tokens

**Symptoms of Context Rot:**
1. Forgetting earlier facts while fixating on recent details
2. Increased hallucinations as context grows
3. Confusion between similar concepts
4. Failing simple tasks like counting or recall
5. Attention drifting from key information

**Causes:**
- Limited attention budgets distributed unevenly
- Noise dilution from irrelevant information
- Interference from distractors (similar but irrelevant info)
- Sequential logic weakening over long spans

### Key Components of Agent Context

Agent context comprises layered elements assembled for each LLM call:

```
┌─────────────────────────────────────────────────────────┐
│ SYSTEM PROMPT (10-15% of budget)                        │
│ - Agent identity, role, expertise                       │
│ - Core capabilities and constraints                     │
│ - Error-handling rules                                  │
├─────────────────────────────────────────────────────────┤
│ TOOL DEFINITIONS (10-20% of budget)                     │
│ - 5-10 most relevant tools (not all 50+)                │
│ - Detailed descriptions of when/how to use              │
│ - Parameter constraints and examples                    │
├─────────────────────────────────────────────────────────┤
│ RETRIEVED DOCUMENTS (20-30% of budget)                  │
│ - Just-in-time retrieval (not pre-loaded)               │
│ - Top-k most relevant chunks only                       │
│ - Freshness-weighted selection                          │
├─────────────────────────────────────────────────────────┤
│ CONVERSATION HISTORY (20-30% of budget)                 │
│ - Recent messages (rolling window)                      │
│ - Summarized older context                              │
│ - Key decisions preserved                               │
├─────────────────────────────────────────────────────────┤
│ TOOL RESULTS (variable, clear after processing)         │
│ - Current step's tool outputs                           │
│ - Previous outputs cleared after extracting insights    │
│ - Avoid accumulation of raw results                     │
└─────────────────────────────────────────────────────────┘
```

### Tiered Context Architecture

**Production Pattern:** Separate persistent data from working memory with three tiers.

```
┌─────────────────────────────────────────────────────────┐
│ TIER 1: CORE CONTEXT (Stable, cached)                   │
│ - System prompt                                         │
│ - Agent identity                                        │
│ - Shared knowledge base                                 │
│ - Rarely changes → Cache-friendly                       │
├─────────────────────────────────────────────────────────┤
│ TIER 2: SESSION CONTEXT (Semi-stable)                   │
│ - Current task objectives                               │
│ - Session state                                         │
│ - User preferences                                      │
│ - Changes per session → Summarize periodically          │
├─────────────────────────────────────────────────────────┤
│ TIER 3: TASK CONTEXT (Dynamic, short-lived)             │
│ - Current step's tool results                           │
│ - Recent conversation turns                             │
│ - Working memory for current operation                  │
│ - Changes per step → Clear aggressively                 │
└─────────────────────────────────────────────────────────┘
```

**Ordering for Cache Optimization:**
1. Place stable Core context at the **front** (benefits from KV cache)
2. Session context in the **middle**
3. Dynamic Task context at the **end**

### Context Engineering Techniques

#### 1. Just-in-Time Retrieval vs Pre-Loading

**Pre-loading (Anti-pattern):**
```
❌ Load entire document → Fill context → Trigger rot
   Result: Overwhelmed attention, increased hallucinations
```

**Just-in-Time Retrieval (Best Practice):**
```
✓ Query arrives → Retrieve relevant chunks → Inject only top-k
  Result: Focused attention, maintained accuracy
```

**Impact:** Just-in-time retrieval reduces token waste and maintains coherence

#### 2. Tool Result Clearing

**Problem:** Tool results accumulate, bloating context across steps.

**Solution:**
```python
def process_tool_result(result):
    # Extract key insights
    insights = extract_key_points(result)

    # Store insights in session memory
    session.add_insight(insights)

    # Clear raw tool output from context
    # Don't carry forward: raw JSON, full API responses
    return insights  # Return only distilled information
```

**Pattern for Multi-Agent Handoffs:**
```python
# When transferring context between agents
translated_context = f"[For context]: Agent A determined: {key_findings}"
# NOT: raw tool outputs from Agent A
```

#### 3. Structured Note-Taking (NOTES Files)

**Externalize memory to bypass context limits:**

```
┌─────────────────────────────────────────────────────────┐
│ CONTEXT WINDOW                │ EXTERNAL NOTES          │
│ (Limited: 32K effective)      │ (Unlimited)             │
├───────────────────────────────┼─────────────────────────┤
│ Current step's working memory │ notes/task_123.md       │
│ Recent decisions              │ notes/insights.json     │
│ Active tool results           │ notes/decisions.log     │
│                               │                         │
│ ← Retrieve on-demand ←────────│                         │
└───────────────────────────────┴─────────────────────────┘
```

**Anthropic's Recommendation:**
- Agents should maintain external notes for long-horizon tasks
- Summarize insights into external stores
- Retrieve only relevant snippets via tools
- Don't stuff everything into context

#### 4. Tool Set Bloat Management

**The 5-10 Tool Limit:**
- Performance degrades beyond 5-10 tools per agent
- Each tool adds ~3,000 tokens to context
- 50 tools = 150,000 tokens consumed before any actual work

**Solutions:**

| Technique | Token Reduction | Implementation |
|-----------|-----------------|----------------|
| Tool Search + defer_loading | 85% | Load tools on-demand |
| MCP code execution | 98.7% | Agents explore filesystem |
| Skill-based organization | 70% | Higher-level abstractions |
| RAG for tool selection | 60-80% | Embed tool descriptions, retrieve relevant |

**Example:**
```python
# Before: 150,000 tokens (50 tools × 3,000 tokens)
all_tools = load_all_tools()

# After: 17,000 tokens (search + 5 discovered tools)
tools = load_tools_with_search(
    tool_directory="/tools",
    defer_loading=True,
    search_tool=True
)
```

#### 5. Context Compaction and Summarization

**When context grows too large:**

```python
def compact_context(messages, max_tokens=30000):
    if count_tokens(messages) < max_tokens:
        return messages

    # Strategy 1: Rolling window (keep last N messages)
    recent = messages[-10:]

    # Strategy 2: Summarize old messages
    old_messages = messages[:-10]
    summary = summarize(old_messages)  # Dedicated summarizer call

    # Strategy 3: Extract entities/facts to external memory
    entities = extract_entities(old_messages)
    store_in_memory(entities)

    return [{"role": "system", "content": summary}] + recent
```

**Key Insight:** Compaction preserves semantic structure without reprocessing all tokens.

### Context Window Budgeting

**The 32K Distraction Ceiling (Databricks Research):**
- Even models with 1M context windows degrade beyond ~32K active tokens
- Effective limit is attention capacity, not window size
- Solution: Compress to stay under 30K active tokens

**Budget Allocation:**

| Component | Allocation | Notes |
|-----------|------------|-------|
| System Prompt | 10-15% | Stable, cache-friendly |
| Tools | 10-20% | Use tool search to minimize |
| Session/Core Knowledge | 20-30% | Summarize periodically |
| Task Context | 40-50% | Dynamic, clear aggressively |
| Buffer | 5-10% | Room for tool results |

**Monitoring:**
```python
def check_context_budget(messages, tools, max_budget=30000):
    current = count_tokens(messages) + count_tool_tokens(tools)
    if current > max_budget * 0.8:  # 80% threshold
        trigger_compaction(messages)
    if current > max_budget:
        raise ContextBudgetExceeded(f"{current}/{max_budget}")
```

### Provider-Specific Approaches

**Anthropic:**
- Autonomous navigation with progressive disclosure
- Agent-driven note-taking for persistence
- Minimize window size via layered retrieval
- Use `cache_control` on stable system prompts

**Google ADK:**
- Session as immutable ground truth storage
- Working Context as computed projection
- Cache-friendly ordering (stable first, dynamic last)
- Explicit cleaning during agent handoffs

**LangChain:**
- Memory buffering for multi-session recall
- RAG-Token alternatives with encoder-decoder separation
- Linear compute for passage processing
- Joint decoding for final output

**Factory.ai:**
- External scaling beyond 1M tokens for enterprise
- Specialized retrieval for monorepo-scale codebases
- Decomposition strategies for billion-token contexts

### Anti-Patterns to Avoid

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| **Context Stuffing** | Dump everything into context | Use just-in-time retrieval |
| **Tool Hoarding** | Load all 50+ tools upfront | Use Tool Search + defer_loading |
| **History Accumulation** | Keep all conversation history | Rolling window + summarization |
| **Raw Result Retention** | Keep all tool outputs | Clear after extracting insights |
| **Flat Context** | No hierarchy in context | Use tiered architecture |
| **Static Context** | Same context for all steps | Dynamic per-step curation |

### Implementation Checklist

```markdown
□ Implement tiered context architecture (Core/Session/Task)
□ Set up just-in-time retrieval instead of pre-loading
□ Add tool result clearing after processing
□ Create external notes system for long-horizon tasks
□ Limit tools to 5-10 per call (use Tool Search if more needed)
□ Monitor token usage against 30K effective ceiling
□ Order context: stable content first, dynamic last
□ Set up periodic summarization for long conversations
□ Implement context budget alerts (80% warning, 100% error)
□ Test for context rot symptoms regularly
```

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

## Cost Attribution & Monitoring Dashboards

### Multi-Dimensional Cost Tracking

**Track costs across multiple dimensions for accurate attribution:**

```
┌─────────────────────────────────────────────────────────────────┐
│                 COST ATTRIBUTION DIMENSIONS                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   BY AGENT           BY USER            BY USE CASE              │
│   ┌─────────┐       ┌─────────┐       ┌─────────────────┐       │
│   │Research │$45    │User_123 │$120   │Customer Support │$500   │
│   │Writer   │$32    │User_456 │$85    │Code Generation  │$350   │
│   │Analyst  │$28    │User_789 │$42    │Data Analysis    │$200   │
│   └─────────┘       └─────────┘       └─────────────────┘       │
│                                                                  │
│   BY CONVERSATION    BY MODEL           BY TIME WINDOW           │
│   ┌─────────┐       ┌─────────┐       ┌─────────────────┐       │
│   │Conv_A   │$12.50 │GPT-4o   │$450   │Last 24h  │$1,200       │
│   │Conv_B   │$8.20  │Claude   │$380   │Last 7d   │$8,500       │
│   │Conv_C   │$15.30 │DeepSeek │$45    │MTD       │$32,000      │
│   └─────────┘       └─────────┘       └─────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
class CostAttributionTracker:
    """Multi-dimensional cost tracking for AI agents."""

    def __init__(self):
        self.metrics_store = MetricsStore()

    def record_usage(self, event: dict):
        """Record usage event with full attribution metadata."""
        cost_record = {
            "timestamp": datetime.now(),
            "request_id": event["request_id"],

            # Token usage
            "input_tokens": event["usage"]["input_tokens"],
            "output_tokens": event["usage"]["output_tokens"],
            "cached_tokens": event["usage"].get("cached_tokens", 0),

            # Cost calculation
            "cost_usd": self.calculate_cost(event),

            # Attribution dimensions
            "dimensions": {
                "agent_id": event["agent_id"],
                "agent_type": event["agent_type"],
                "user_id": event["user_id"],
                "tenant_id": event.get("tenant_id"),
                "conversation_id": event["conversation_id"],
                "use_case": event["use_case"],
                "model": event["model"],
                "feature": event.get("feature", "general"),
                "environment": event.get("environment", "production")
            },

            # Quality correlation
            "quality_metrics": {
                "task_success": event.get("task_success"),
                "latency_ms": event.get("latency_ms"),
                "error": event.get("error")
            }
        }

        self.metrics_store.insert(cost_record)
        self.update_real_time_aggregates(cost_record)

    def calculate_cost(self, event: dict):
        """Calculate cost based on model and token usage."""
        pricing = self.get_model_pricing(event["model"])

        input_cost = (event["usage"]["input_tokens"] / 1_000_000) * pricing["input"]
        output_cost = (event["usage"]["output_tokens"] / 1_000_000) * pricing["output"]

        # Apply cache discount
        if event["usage"].get("cached_tokens"):
            cache_discount = (event["usage"]["cached_tokens"] / 1_000_000) * pricing["input"] * 0.9
            input_cost -= cache_discount

        return input_cost + output_cost

    def get_cost_breakdown(self, filters: dict, group_by: str):
        """Get cost breakdown by dimension."""
        return self.metrics_store.aggregate(
            filters=filters,
            group_by=group_by,
            metrics=["sum(cost_usd)", "count(*)", "avg(latency_ms)"]
        )
```

---

### Cost-Quality Correlation

**Track ROI and cost-per-successful-task:**

| Metric | Formula | Target |
|--------|---------|--------|
| **Cost per Task** | Total cost / Tasks attempted | < $0.50 |
| **Cost per Success** | Total cost / Successful tasks | < $0.75 |
| **Quality-Adjusted Cost** | Cost / (Success rate × Quality score) | Minimize |
| **Model Cascade ROI** | (Baseline cost - Cascade cost) / Baseline | > 40% |
| **Cache Hit Savings** | Cached tokens × Unit price × 0.9 | Maximize |

```python
class CostQualityAnalyzer:
    """Analyze cost in relation to quality metrics."""

    def calculate_cost_efficiency_metrics(self, timeframe: str):
        """Calculate key cost-quality metrics."""
        data = self.metrics_store.query(timeframe=timeframe)

        total_cost = sum(r["cost_usd"] for r in data)
        total_tasks = len(data)
        successful_tasks = sum(1 for r in data if r["quality_metrics"]["task_success"])

        success_rate = successful_tasks / total_tasks if total_tasks else 0

        return {
            "total_cost": total_cost,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "cost_per_task": total_cost / total_tasks if total_tasks else 0,
            "cost_per_success": total_cost / successful_tasks if successful_tasks else 0,
            "quality_adjusted_cost": total_cost / (success_rate * 0.9) if success_rate else float('inf')
        }

    def analyze_model_cascade_roi(self, timeframe: str):
        """Analyze ROI from model cascading."""
        data = self.metrics_store.query(timeframe=timeframe)

        # Calculate baseline (if all used expensive model)
        baseline_cost = sum(
            self.calculate_baseline_cost(r) for r in data
        )

        # Actual cost with cascading
        actual_cost = sum(r["cost_usd"] for r in data)

        roi = (baseline_cost - actual_cost) / baseline_cost if baseline_cost else 0

        return {
            "baseline_cost": baseline_cost,
            "actual_cost": actual_cost,
            "savings": baseline_cost - actual_cost,
            "roi_percentage": roi * 100,
            "cascade_effectiveness": self.analyze_cascade_patterns(data)
        }

    def analyze_cache_savings(self, timeframe: str):
        """Analyze savings from prompt caching."""
        data = self.metrics_store.query(timeframe=timeframe)

        total_cached_tokens = sum(
            r.get("cached_tokens", 0) for r in data
        )

        # Calculate what those tokens would have cost
        savings = 0
        for r in data:
            if r.get("cached_tokens"):
                pricing = self.get_model_pricing(r["model"])
                savings += (r["cached_tokens"] / 1_000_000) * pricing["input"] * 0.9

        return {
            "total_cached_tokens": total_cached_tokens,
            "estimated_savings": savings,
            "cache_hit_rate": self.calculate_cache_hit_rate(data)
        }
```

---

### Real-Time Spend Dashboards

**Dashboard Components:**

| Panel | Metrics | Update Frequency |
|-------|---------|------------------|
| **Spend Velocity** | Current $/minute, trend | 1 minute |
| **Budget Burn** | % of daily/weekly/monthly budget used | 5 minutes |
| **Top Cost Drivers** | Agents/users/models consuming most | 15 minutes |
| **Anomaly Detection** | Unusual spend patterns | Real-time |
| **Cost Forecast** | Projected end-of-period spend | Hourly |

```python
class RealTimeSpendDashboard:
    """Real-time spend monitoring and alerting."""

    def __init__(self):
        self.alert_thresholds = {
            "hourly_budget": 50,
            "daily_budget": 500,
            "single_request_max": 5,
            "spike_multiplier": 3  # 3x normal rate
        }

    def get_dashboard_data(self):
        """Get current dashboard state."""
        now = datetime.now()

        return {
            "current_velocity": self.get_spend_velocity(),
            "budget_status": self.get_budget_status(),
            "top_cost_drivers": self.get_top_drivers(),
            "anomalies": self.detect_anomalies(),
            "forecast": self.forecast_spend(),
            "last_updated": now
        }

    def get_spend_velocity(self):
        """Calculate current spending rate."""
        last_hour = self.metrics_store.query(timeframe="1h")
        last_10_min = self.metrics_store.query(timeframe="10m")

        hourly_rate = sum(r["cost_usd"] for r in last_hour)
        recent_rate = sum(r["cost_usd"] for r in last_10_min) * 6  # Annualized

        return {
            "hourly_rate": hourly_rate,
            "projected_hourly": recent_rate,
            "trend": "increasing" if recent_rate > hourly_rate else "decreasing"
        }

    def detect_anomalies(self):
        """Detect unusual spending patterns."""
        anomalies = []

        # Check for single expensive requests
        recent = self.metrics_store.query(timeframe="1h")
        for r in recent:
            if r["cost_usd"] > self.alert_thresholds["single_request_max"]:
                anomalies.append({
                    "type": "expensive_request",
                    "request_id": r["request_id"],
                    "cost": r["cost_usd"],
                    "agent": r["dimensions"]["agent_id"]
                })

        # Check for spending spikes
        baseline = self.get_baseline_velocity()
        current = self.get_spend_velocity()["projected_hourly"]

        if current > baseline * self.alert_thresholds["spike_multiplier"]:
            anomalies.append({
                "type": "spending_spike",
                "baseline": baseline,
                "current": current,
                "multiplier": current / baseline if baseline else float('inf')
            })

        return anomalies

    def trigger_alerts(self, anomalies: list):
        """Send alerts for detected anomalies."""
        for anomaly in anomalies:
            if anomaly["type"] == "spending_spike":
                self.send_alert(
                    severity="critical",
                    message=f"Spending spike detected: {anomaly['multiplier']:.1f}x baseline",
                    details=anomaly
                )
            elif anomaly["type"] == "expensive_request":
                self.send_alert(
                    severity="warning",
                    message=f"Expensive request: ${anomaly['cost']:.2f}",
                    details=anomaly
                )
```

---

### LLM Monitoring Tools Integration

**Platform Comparison:**

| Platform | Strengths | Best For |
|----------|-----------|----------|
| **LangSmith** | Deep LangChain integration, tracing, prompt versioning | LangGraph users |
| **Helicone** | 100% accurate cost tracking, model routing, sessions | Multi-provider setups |
| **Arize AI** | ML observability, drift detection, performance | Enterprise ML teams |
| **OpenLLMetry** | Open standard (OTEL), vendor-agnostic | Custom implementations |
| **Braintrust** | Evaluation-focused, auto cost estimation | Quality-first teams |
| **Langfuse** | Open-source, self-hostable | Privacy-conscious |

**OpenTelemetry Integration:**
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

class OTELAgentCostTracker:
    """OpenTelemetry-based cost tracking for AI agents."""

    def __init__(self):
        self.tracer = trace.get_tracer("agent-cost-tracker")

    def track_llm_call(self, func):
        """Decorator to track LLM calls with cost attributes."""
        def wrapper(*args, **kwargs):
            with self.tracer.start_as_current_span("llm_call") as span:
                # Set pre-call attributes
                span.set_attribute("llm.model", kwargs.get("model", "unknown"))
                span.set_attribute("llm.agent_id", kwargs.get("agent_id"))
                span.set_attribute("llm.user_id", kwargs.get("user_id"))

                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Set post-call attributes
                if hasattr(result, "usage"):
                    span.set_attribute("llm.input_tokens", result.usage.input_tokens)
                    span.set_attribute("llm.output_tokens", result.usage.output_tokens)
                    span.set_attribute("llm.cost_usd", self.calculate_cost(result))

                span.set_attribute("llm.duration_ms", duration * 1000)
                span.set_attribute("llm.success", True)

                return result
        return wrapper

    def export_to_dashboard(self, exporter):
        """Configure export to monitoring platform."""
        provider = TracerProvider()
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
```

---

### Cost Optimization Recommendations

**Automated Analysis:**
```python
class CostOptimizationAdvisor:
    """Generate cost optimization recommendations."""

    def analyze_and_recommend(self):
        """Analyze usage patterns and generate recommendations."""
        recommendations = []

        # Check for model over-use
        model_usage = self.analyze_model_patterns()
        for model, stats in model_usage.items():
            if stats["simple_task_ratio"] > 0.5 and "gpt-4" in model:
                recommendations.append({
                    "priority": "high",
                    "type": "model_downgrade",
                    "message": f"{stats['simple_task_ratio']*100:.0f}% of {model} calls are simple tasks",
                    "action": "Route simple tasks to GPT-4o-mini",
                    "estimated_savings": stats["potential_savings"]
                })

        # Check cache utilization
        cache_stats = self.analyze_cache_patterns()
        if cache_stats["hit_rate"] < 0.3:
            recommendations.append({
                "priority": "medium",
                "type": "caching",
                "message": f"Cache hit rate is only {cache_stats['hit_rate']*100:.0f}%",
                "action": "Enable prompt caching for repeated system prompts",
                "estimated_savings": cache_stats["potential_savings"]
            })

        # Check for chatty agents
        agent_stats = self.analyze_agent_patterns()
        for agent, stats in agent_stats.items():
            if stats["avg_turns"] > 10:
                recommendations.append({
                    "priority": "medium",
                    "type": "agent_optimization",
                    "message": f"Agent {agent} averages {stats['avg_turns']:.0f} turns per task",
                    "action": "Review agent logic for unnecessary iterations",
                    "estimated_savings": stats["potential_savings"]
                })

        return sorted(recommendations, key=lambda x: x["estimated_savings"], reverse=True)
```

---

### Cost Attribution Checklist

**Setup:**
- [ ] Implement cost tracking on all LLM calls
- [ ] Add attribution dimensions (agent, user, use case)
- [ ] Configure real-time aggregation
- [ ] Set up monitoring dashboards

**Alerts:**
- [ ] Single request cost threshold ($5)
- [ ] Hourly budget threshold (50% of daily)
- [ ] Daily budget threshold (80%)
- [ ] Spending spike detection (3x baseline)
- [ ] New agent cost monitoring

**Optimization:**
- [ ] Weekly cost review meetings
- [ ] Monthly model usage analysis
- [ ] Quarterly ROI assessment
- [ ] Cache hit rate monitoring
- [ ] Cascade effectiveness tracking

---

## Agent Inference Optimization (AgentInfer Framework)

**End-to-end acceleration for LLM agents: 1.8×-2.5× speedup with 50%+ token reduction**

### Overview: The AgentInfer Architecture

AgentInfer is a unified framework for accelerating LLM agent inference, co-designing optimization with architectural enhancements. Unlike per-token optimizations, it targets agentic task completion.

```
AgentInfer Progressive Pipeline:

┌─────────────────────────────────────────────────────────────────────┐
│                    Self-Evolution Engine                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Stage 1: Reasoning Optimization                               │  │
│  │  ┌──────────────┐                                             │  │
│  │  │ AgentCollab  │  Dual-model collaboration                   │  │
│  │  │ Large + Small│  Dynamic role assignment                    │  │
│  │  └──────────────┘  85% accuracy recovery                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ↓                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Stage 2: Context Optimization                                 │  │
│  │  ┌───────────────┐                                            │  │
│  │  │ AgentCompress │  Semantic compression                      │  │
│  │  │ Memory Prune  │  Remove redundant context                  │  │
│  │  └───────────────┘  50%+ token reduction                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ↓                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Stage 3: Execution Acceleration                               │  │
│  │  ┌──────────────┐  ┌──────────────┐                          │  │
│  │  │ AgentSAM     │  │ AgentSched   │                          │  │
│  │  │ Speculative  │  │ Cache-aware  │                          │  │
│  │  │ Decoding     │  │ Scheduling   │                          │  │
│  │  └──────────────┘  └──────────────┘                          │  │
│  │        1.8×-2.5× latency reduction                            │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Components

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import asyncio

class ModelTier(Enum):
    LARGE = "large"   # GPT-4, Claude Opus, etc.
    SMALL = "small"   # GPT-4o-mini, Claude Haiku, etc.

@dataclass
class AgentCollabConfig:
    """Configuration for dual-model collaboration."""
    large_model: str
    small_model: str
    difficulty_threshold: float = 0.7  # Switch to large above this
    self_eval_enabled: bool = True

class AgentCollab:
    """
    Hierarchical dual-model framework.
    Achieves ~85% accuracy of large-model-only at fraction of cost.
    """

    def __init__(self, config: AgentCollabConfig):
        self.config = config
        self.performance_history: List[Dict] = []

    async def execute_with_collaboration(
        self,
        task: str,
        context: Dict
    ) -> Tuple[str, ModelTier]:
        """
        Route task to appropriate model based on difficulty.
        """
        # Step 1: Estimate task difficulty with small model
        difficulty = await self._estimate_difficulty(task, context)

        # Step 2: Self-evaluation check
        if self.config.self_eval_enabled:
            confidence = await self._self_evaluate(task, context)
            difficulty = max(difficulty, 1.0 - confidence)

        # Step 3: Route based on difficulty
        if difficulty > self.config.difficulty_threshold:
            result = await self._call_large_model(task, context)
            model_used = ModelTier.LARGE
        else:
            result = await self._call_small_model(task, context)
            model_used = ModelTier.SMALL

        # Step 4: Verify and potentially escalate
        if model_used == ModelTier.SMALL:
            if not await self._verify_result(result, task):
                result = await self._call_large_model(task, context)
                model_used = ModelTier.LARGE

        self._record_performance(task, difficulty, model_used)
        return result, model_used

    async def _estimate_difficulty(
        self,
        task: str,
        context: Dict
    ) -> float:
        """
        Quick difficulty estimation using small model.
        Factors: complexity, tool requirements, reasoning depth.
        """
        difficulty_prompt = f"""
        Rate task difficulty from 0.0 to 1.0:
        - 0.0-0.3: Simple lookup, single-step
        - 0.3-0.6: Multi-step but straightforward
        - 0.6-0.8: Complex reasoning or tool chains
        - 0.8-1.0: Novel, requires deep understanding

        Task: {task[:200]}
        Context keys: {list(context.keys())}

        Return only a number.
        """
        # Use small model for estimation
        score = await self._call_small_model(difficulty_prompt, {})
        return float(score.strip())


class AgentCompress:
    """
    Asynchronous semantic compression for agent memory.
    Reduces context by 50%+ without disrupting reasoning.
    """

    def __init__(
        self,
        compression_model: str,
        target_reduction: float = 0.5
    ):
        self.compression_model = compression_model
        self.target_reduction = target_reduction

    def compress_trajectory(
        self,
        trajectory: List[Dict],
        current_task: str
    ) -> List[Dict]:
        """
        Compress agent trajectory by:
        1. Removing redundant observations
        2. Merging similar steps
        3. Extracting only task-relevant context
        """
        # Identify redundant entries
        redundant_indices = self._find_redundant(trajectory)

        # Calculate relevance to current task
        relevance_scores = self._score_relevance(trajectory, current_task)

        # Compress while preserving essential information
        compressed = []
        for i, step in enumerate(trajectory):
            if i in redundant_indices:
                continue

            if relevance_scores[i] < 0.3:
                # Low relevance - summarize instead of include
                summary = self._summarize_step(step)
                compressed.append({"type": "summary", "content": summary})
            else:
                # High relevance - keep full content
                compressed.append(step)

        return compressed

    def _find_redundant(self, trajectory: List[Dict]) -> set:
        """
        AgentDiet technique: identify 39.9-59.7% redundant data.
        - Repeated observations
        - Expired context (no longer relevant)
        - Duplicate tool calls
        """
        redundant = set()
        seen_observations = {}

        for i, step in enumerate(trajectory):
            obs_hash = self._hash_observation(step.get("observation", ""))

            if obs_hash in seen_observations:
                redundant.add(seen_observations[obs_hash])

            seen_observations[obs_hash] = i

        return redundant


class AgentSAM:
    """
    Suffix-Automaton-based Speculative Decoding.
    Reuses multi-session semantic memory for acceleration.
    """

    def __init__(self):
        self.suffix_automaton = SuffixAutomaton()
        self.session_cache: Dict[str, str] = {}

    def speculative_decode(
        self,
        prefix: str,
        target_model: callable,
        draft_model: callable,
        num_speculative_tokens: int = 5
    ) -> str:
        """
        Speculative decoding with semantic memory.

        1. Check suffix automaton for common continuations
        2. Use draft model for speculation
        3. Verify with target model in parallel
        """
        # Check for cached continuations
        cached_continuation = self.suffix_automaton.find_continuation(prefix)
        if cached_continuation:
            # Verify cached result still valid
            if self._verify_continuation(prefix, cached_continuation, target_model):
                return cached_continuation

        # Standard speculative decoding
        generated = ""
        while True:
            # Draft multiple tokens
            draft_tokens = draft_model.generate(
                prefix + generated,
                num_tokens=num_speculative_tokens
            )

            # Verify with target model
            verified = target_model.verify(
                prefix + generated,
                draft_tokens
            )

            if verified["accepted_tokens"] == 0:
                # All rejected, use target model directly
                generated += target_model.generate(
                    prefix + generated,
                    num_tokens=1
                )
            else:
                generated += draft_tokens[:verified["accepted_tokens"]]

            if self._is_complete(generated):
                break

        # Cache for future reuse
        self.suffix_automaton.add(prefix, generated)

        return generated


class AgentSched:
    """
    Cache-aware hybrid scheduler for agent workloads.
    Minimizes latency while maximizing KV cache reuse.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        cache_size_mb: int = 4096
    ):
        self.max_batch_size = max_batch_size
        self.cache_size_mb = cache_size_mb
        self.request_queue: List[Dict] = []
        self.kv_cache = KVCacheManager(cache_size_mb)

    async def schedule_requests(
        self,
        requests: List[Dict]
    ) -> List[Dict]:
        """
        Schedule requests to maximize cache hits and minimize latency.

        Strategy:
        1. Group requests by prefix similarity
        2. Prioritize requests with cached prefixes
        3. Balance batch size vs. cache eviction
        """
        # Sort by prefix for cache locality
        sorted_requests = self._sort_by_prefix(requests)

        # Group into batches
        batches = self._create_cache_aware_batches(sorted_requests)

        results = []
        for batch in batches:
            # Check cache hits
            cache_status = self.kv_cache.check_batch(batch)

            # Process with cache-aware execution
            batch_results = await self._process_batch(batch, cache_status)
            results.extend(batch_results)

        return results

    def _create_cache_aware_batches(
        self,
        requests: List[Dict]
    ) -> List[List[Dict]]:
        """
        Create batches that maximize cache reuse.
        Uses prefix-sharing to reduce memory pressure.
        """
        batches = []
        current_batch = []
        current_prefix = None

        for request in requests:
            prefix = self._extract_prefix(request)

            if current_prefix is None:
                current_prefix = prefix

            # Check if adding request would evict useful cache entries
            if len(current_batch) >= self.max_batch_size:
                batches.append(current_batch)
                current_batch = [request]
                current_prefix = prefix
            elif self._prefix_similarity(current_prefix, prefix) > 0.8:
                current_batch.append(request)
            else:
                batches.append(current_batch)
                current_batch = [request]
                current_prefix = prefix

        if current_batch:
            batches.append(current_batch)

        return batches
```

### Trajectory Compression (AgentDiet)

**Research finding**: 39.9-59.7% of agent trajectory tokens are redundant or expired.

```python
class TrajectoryOptimizer:
    """
    AgentDiet implementation for trajectory compression.
    Achieves 21.1-35.9% cost reduction while maintaining accuracy.
    """

    def __init__(self, reflection_model: str = "gpt-4o-mini"):
        self.reflection_model = reflection_model
        self.compression_stats = {
            "tokens_before": 0,
            "tokens_after": 0,
            "redundant_removed": 0
        }

    def optimize_trajectory(
        self,
        trajectory: List[Dict],
        task_context: str
    ) -> Tuple[List[Dict], Dict]:
        """
        Apply AgentDiet compression to trajectory.

        Returns:
            compressed_trajectory: Optimized trajectory
            stats: Compression statistics
        """
        self.compression_stats["tokens_before"] = self._count_tokens(trajectory)

        # Step 1: Identify redundant observations
        trajectory = self._remove_redundant_observations(trajectory)

        # Step 2: Compress expired context
        trajectory = self._compress_expired_context(trajectory, task_context)

        # Step 3: Merge duplicate tool calls
        trajectory = self._merge_duplicate_tools(trajectory)

        # Step 4: LLM reflection for semantic compression
        trajectory = self._reflect_and_compress(trajectory, task_context)

        self.compression_stats["tokens_after"] = self._count_tokens(trajectory)

        return trajectory, self.compression_stats

    def _remove_redundant_observations(
        self,
        trajectory: List[Dict]
    ) -> List[Dict]:
        """
        Remove observations that appear multiple times.
        Keeps only the most recent occurrence.
        """
        seen = {}
        result = []

        for i, step in enumerate(reversed(trajectory)):
            if step.get("type") == "observation":
                obs_key = self._observation_key(step)
                if obs_key not in seen:
                    seen[obs_key] = i
                    result.append(step)
            else:
                result.append(step)

        return list(reversed(result))

    def _compress_expired_context(
        self,
        trajectory: List[Dict],
        current_task: str
    ) -> List[Dict]:
        """
        Remove context that's no longer relevant to current task.
        """
        result = []

        for step in trajectory:
            relevance = self._compute_relevance(step, current_task)

            if relevance > 0.3:
                result.append(step)
            elif relevance > 0.1:
                # Partially relevant - keep summary only
                summary = self._summarize(step)
                result.append({"type": "summary", "content": summary})
            # else: fully expired, drop

        return result

    def _reflect_and_compress(
        self,
        trajectory: List[Dict],
        task_context: str
    ) -> List[Dict]:
        """
        Use LLM reflection for semantic compression.
        GPT-4o-mini provides cost-effective reflection.
        """
        reflection_prompt = f"""
        Compress this agent trajectory while preserving essential information:

        Current Task: {task_context}

        Trajectory:
        {self._format_trajectory(trajectory)}

        Output a compressed version with:
        1. Key decisions and their reasoning
        2. Important observations (no duplicates)
        3. Final tool call results only

        Keep only information needed to continue the task.
        """

        compressed = self._call_reflection_model(reflection_prompt)
        return self._parse_compressed(compressed)
```

### KV-Cache Optimization for Multi-Turn Agents

```python
class AgentKVCacheManager:
    """
    Optimized KV cache management for multi-turn agent interactions.
    Uses PagedAttention + GQA for memory efficiency.
    """

    def __init__(
        self,
        max_cache_size_gb: float = 8.0,
        page_size: int = 256,  # tokens per page
        eviction_policy: str = "lru_with_priority"
    ):
        self.max_cache_size_gb = max_cache_size_gb
        self.page_size = page_size
        self.eviction_policy = eviction_policy
        self.cache: Dict[str, List[torch.Tensor]] = {}
        self.access_times: Dict[str, float] = {}
        self.priorities: Dict[str, float] = {}

    def get_or_compute(
        self,
        prefix_key: str,
        prefix_tokens: List[int],
        compute_fn: callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached KV pairs or compute and cache.
        """
        if prefix_key in self.cache:
            self.access_times[prefix_key] = time.time()
            return self.cache[prefix_key]

        # Compute new KV pairs
        kv_pairs = compute_fn(prefix_tokens)

        # Check if eviction needed
        if self._cache_size_gb() + self._size_gb(kv_pairs) > self.max_cache_size_gb:
            self._evict_until_fits(self._size_gb(kv_pairs))

        # Cache with paging
        self.cache[prefix_key] = self._page_kv_pairs(kv_pairs)
        self.access_times[prefix_key] = time.time()

        return kv_pairs

    def set_priority(self, prefix_key: str, priority: float):
        """
        Set priority for cache entries.
        Higher priority = less likely to evict.

        Use cases:
        - System prompts: priority=1.0
        - Recent conversations: priority=0.8
        - Old context: priority=0.2
        """
        self.priorities[prefix_key] = priority

    def _evict_until_fits(self, required_gb: float):
        """
        Evict entries until space available.
        Uses LRU weighted by priority.
        """
        entries = [
            (key, self.access_times[key], self.priorities.get(key, 0.5))
            for key in self.cache
        ]

        # Sort by priority-weighted access time
        entries.sort(key=lambda x: x[1] * x[2])

        freed = 0.0
        while freed < required_gb and entries:
            key, _, _ = entries.pop(0)
            freed += self._size_gb(self.cache[key])
            del self.cache[key]
            del self.access_times[key]
            if key in self.priorities:
                del self.priorities[key]
```

### Inference Optimization Benchmarks

| Technique | Benchmark | Improvement | Source |
|-----------|-----------|-------------|--------|
| **AgentInfer (full)** | BrowseComp-zh | 1.8×-2.5× speedup | arXiv:2512.18337 |
| **AgentDiet** | Multi-SWE-bench | 35.9% cost reduction | arXiv:2509.23586 |
| **AgentDiet** | Gemini 2.5 Pro | 50% trajectory reduction | arXiv:2509.23586 |
| **Speculative Decoding** | General | 2-4× latency reduction | Industry standard |
| **KV Cache (PagedAttention)** | Multi-turn | 2-5× throughput | vLLM benchmark |
| **Quantization (AWQ)** | Inference | 3-4× speed | Model compression |
| **AgentCollab** | Cost routing | 85% accuracy at lower cost | arXiv:2512.18337 |

### Production Agent Inference Stack

```python
class ProductionAgentInferenceStack:
    """
    Complete inference optimization stack for production agents.
    Integrates all AgentInfer components.
    """

    def __init__(
        self,
        large_model: str = "claude-opus-4-5",
        small_model: str = "claude-haiku",
        cache_size_gb: float = 8.0,
        enable_speculation: bool = True
    ):
        # AgentCollab for model routing
        self.collab = AgentCollab(AgentCollabConfig(
            large_model=large_model,
            small_model=small_model,
            difficulty_threshold=0.7
        ))

        # AgentCompress for trajectory optimization
        self.compress = AgentCompress(
            compression_model=small_model,
            target_reduction=0.5
        )

        # AgentSched for request scheduling
        self.sched = AgentSched(
            max_batch_size=32,
            cache_size_mb=int(cache_size_gb * 1024)
        )

        # AgentSAM for speculative decoding
        self.sam = AgentSAM() if enable_speculation else None

        # KV cache management
        self.kv_cache = AgentKVCacheManager(max_cache_size_gb=cache_size_gb)

        # Trajectory optimizer
        self.trajectory_opt = TrajectoryOptimizer(reflection_model=small_model)

    async def execute_agent_step(
        self,
        task: str,
        trajectory: List[Dict],
        context: Dict
    ) -> Tuple[str, Dict]:
        """
        Execute single agent step with full optimization stack.
        """
        metrics = {"original_tokens": 0, "optimized_tokens": 0}

        # Step 1: Compress trajectory
        compressed_trajectory, compress_stats = self.trajectory_opt.optimize_trajectory(
            trajectory, task
        )
        metrics["original_tokens"] = compress_stats["tokens_before"]
        metrics["optimized_tokens"] = compress_stats["tokens_after"]

        # Step 2: Build optimized context
        optimized_context = self._build_context(compressed_trajectory, context)

        # Step 3: Route to appropriate model
        result, model_used = await self.collab.execute_with_collaboration(
            task, optimized_context
        )
        metrics["model_used"] = model_used.value

        # Step 4: Update trajectory for next step
        new_trajectory = compressed_trajectory + [
            {"type": "action", "content": result}
        ]

        return result, metrics

    def get_optimization_report(self) -> Dict:
        """
        Return comprehensive optimization metrics.
        """
        return {
            "token_reduction": self.trajectory_opt.compression_stats,
            "cache_hit_rate": self.kv_cache.hit_rate(),
            "model_routing": self.collab.get_routing_stats(),
            "speculation_acceptance": self.sam.acceptance_rate() if self.sam else None
        }
```

### Agent Inference Optimization Checklist

**Model Routing:**
- [ ] AgentCollab configured with appropriate thresholds
- [ ] Self-evaluation enabled for quality routing
- [ ] Model tier costs tracked per request

**Trajectory Optimization:**
- [ ] AgentDiet compression enabled
- [ ] Redundancy detection tuned (39-60% typical)
- [ ] Relevance scoring calibrated for task domain

**Caching:**
- [ ] KV cache sized for workload (8GB typical)
- [ ] Priority levels set for system prompts
- [ ] Eviction policy matches access patterns

**Speculation:**
- [ ] Draft model aligned with target
- [ ] Suffix automaton populated with common patterns
- [ ] Acceptance rate monitored (target >70%)

**Monitoring:**
- [ ] Token reduction tracked per step
- [ ] Latency percentiles measured
- [ ] Cost attribution per optimization layer

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
