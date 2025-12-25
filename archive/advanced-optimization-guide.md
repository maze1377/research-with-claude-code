# Advanced Optimization Guide for Agentic Systems

**Last Updated:** 2025-11-08

A comprehensive guide to advanced optimization techniques for reducing costs, improving performance, eliminating hallucinations, and accelerating workflows in production agentic systems.

## Table of Contents

1. [Cache Hit Optimization](#cache-hit-optimization)
2. [Prompt Optimization](#prompt-optimization)
3. [Hallucination Reduction](#hallucination-reduction)
4. [Performance & Speed Optimization](#performance--speed-optimization)
5. [Combined Optimization Strategies](#combined-optimization-strategies)
6. [Production Monitoring](#production-monitoring)

---

## Cache Hit Optimization

Multi-level caching strategies can reduce API costs by 60-90% while improving response times.

### 1. Exact Match Caching

**Cost Savings:** 100% on cache hits
**Implementation:** Redis/Memcached with query hashing

```python
import hashlib
import redis

cache = redis.Redis(host='localhost', port=6379)

def get_cached_response(prompt: str, model: str):
    cache_key = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()
    cached = cache.get(cache_key)

    if cached:
        return cached.decode()

    response = call_llm_api(prompt, model)
    cache.setex(cache_key, 3600, response)  # 1-hour TTL
    return response
```

**Best Practices:**
- Use short TTL (1-6 hours) for dynamic content
- Hash full prompt + model + temperature for key
- Monitor hit rates (target: 20-40% for typical workloads)

### 2. Semantic Caching

**Cost Savings:** 60-90% on semantically similar queries
**Hit Rate:** 31% average, up to 68.8% for focused domains

**Architecture:**
```
Query → Embedding → Similarity Search (>0.95) → Cache Hit
                                       (<0.95) → LLM Call → Store
```

**Implementation with GPTCache:**

```python
from gptcache import Cache
from gptcache.embedding import OpenAI as EmbeddingOpenAI
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

cache = Cache()
cache.init(
    embedding_func=EmbeddingOpenAI(),
    similarity_evaluation=SearchDistanceEvaluation(threshold=0.95)
)

def semantic_cached_call(prompt: str):
    return cache.get_or_set(
        prompt,
        lambda: expensive_llm_call(prompt)
    )
```

**Key Findings (2025 Research):**
- Domain-specific embeddings: 78% → 87% precision
- Ensemble embedding approach: 92% hit ratio
- 68.8% reduction in API calls for customer support
- Best threshold: 0.93-0.97 cosine similarity

### 3. Prompt Caching (Provider-Specific)

**Anthropic Prompt Caching:**
- 90% cost savings on cached content
- 85% cost reduction (real customer case)
- Cache up to 200K tokens of context

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "Long system context...",
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[{"role": "user", "content": "Query"}]
)
```

**Cost Breakdown:**
- Write: $3.75/MTok (standard $3/MTok)
- Read: $0.30/MTok (90% savings vs $3/MTok)
- 5-minute TTL (automatic extension on use)

**OpenAI Context Caching:**
- 50% discount on cached tokens
- Automatic for prompts >1024 tokens
- No code changes required

### 4. KV Cache-Aware Routing

**Cost Savings:** 70% compute reduction
**Method:** Route requests to instances with warm KV cache

**Benefits:**
- Reduced first-token latency
- Lower memory bandwidth
- Better GPU utilization

### Multi-Level Caching Strategy

**Recommended Architecture:**

```
Level 1: Exact Match (Redis)     → 100% savings, 20-30% hit rate
Level 2: Semantic Cache          → 90% savings, 40-60% hit rate
Level 3: Prompt Cache            → 50-90% savings, 70-80% hit rate
Level 4: LLM Call                → Full cost
```

**Expected Combined Savings:** 60-85% overall cost reduction

---

## Prompt Optimization

Reduce token usage by 30-75% through strategic prompt engineering.

### 1. Concise Prompting

**Token Reduction:** 30-50%
**Performance Impact:** Minimal (sometimes improved)

**Before (85 tokens):**
```
I need you to carefully analyze the following customer feedback and provide
a detailed summary of the main themes, sentiment, and any actionable insights
that our product team should be aware of. Please be thorough in your analysis.

Customer feedback: [text]
```

**After (42 tokens):**
```
Analyze customer feedback. Return: themes, sentiment, actionable insights.

Feedback: [text]
```

**Techniques:**
- Remove filler words ("please", "carefully", "detailed")
- Use structured formats (bullet points, JSON)
- Eliminate redundant instructions
- Use abbreviations where clear

### 2. Compression Tools

**LLMLingua:**
- 30-50% compression with minimal quality loss
- Preserves key information using perplexity filtering

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor()
compressed = compressor.compress_prompt(
    long_prompt,
    instruction="Compress while preserving key facts",
    target_token=512
)
```

**500xCompressor:**
- Extreme compression for reference documents
- Best for retrieval contexts

**PCToolkit (Prompt Compression Toolkit):**
- Iterative compression with quality validation
- Automatic rewrite and testing

### 3. BatchPrompt Technique

**Token Reduction:** 40-60% for bulk operations
**Use Case:** Processing multiple similar inputs

**Individual Calls (300 tokens × 5 = 1,500 tokens):**
```
Classify sentiment: "Product is great!" → Positive
Classify sentiment: "Terrible service" → Negative
[3 more individual calls]
```

**BatchPrompt (600 tokens total):**
```
Classify sentiment for each:
1. "Product is great!"
2. "Terrible service"
3. "Works as expected"
4. "Disappointed with quality"
5. "Exceeded expectations"

Format: JSON array of {id, sentiment}
```

**Savings:** 60% reduction (1,500 → 600 tokens)

### 4. Fine-Tuning for Compression

**Token Reduction:** 50-75%
**ROI:** Positive after 1M+ tokens processed

**Strategy:**
- Train on task-specific data (500-2,000 examples)
- Replace long prompts with short instructions
- Suitable for repeated, well-defined tasks

**Example:**
- Before: 500-token prompt for code review
- After: "Review code" (2 tokens) with fine-tuned model

**Cost Analysis:**
- Training: $300-500 one-time
- Inference savings: $2-4 per 1M tokens
- Break-even: 150K-250K tokens

### 5. Progress Tracking Optimization

**Token Reduction:** 40% for iterative tasks
**Method:** Track completed work instead of repeating context

```python
state = {
    "completed_steps": [],
    "current_focus": "step_3",
    "remaining": ["step_4", "step_5"]
}

# Instead of passing full conversation history,
# pass only: state + current step context
```

### Monitoring Prompt Efficiency

**Key Metrics:**
- Average tokens per request
- Cost per successful task completion
- Token waste ratio (retry tokens / total tokens)

**Tools:**
- OpenAI Usage Dashboard
- Anthropic Console
- Custom tracking with Langfuse/LangSmith

**Target Improvements:**
- 3-10% from concise prompting (quick win)
- 20-40% from batching strategies
- 50-75% from fine-tuning (long-term)

---

## Hallucination Reduction

Techniques to improve factual accuracy from ~85% to 95-99%.

### 1. Retrieval-Augmented Generation (RAG)

**Accuracy Improvement:** 85% → 92-95%
**Best For:** Knowledge-intensive tasks

**Architecture:**
```
Query → Retrieve Relevant Docs → Ground LLM Response → Validate
```

**Implementation:**

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Build knowledge base
vectorstore = Chroma.from_documents(
    documents=knowledge_docs,
    embedding=OpenAIEmbeddings()
)

# Retrieve + generate
def rag_query(question: str):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information."

Context: {context}

Question: {question}"""

    return llm.invoke(prompt)
```

**Key Findings:**
- RAG dramatically reduces hallucinations
- Cite sources in responses for transparency
- Use hybrid search (keyword + semantic) for best results

### 2. Fact Grounded Attention (FGA)

**Accuracy Improvement:** 87.1% → 99.7%
**Method:** Constrain attention to grounding documents

**Research Results (2025):**
- 99.7% accuracy on TriviaQA with FGA
- 99.4% on Natural Questions
- Reduces "plausible-sounding" errors

**Implementation Concept:**
```python
# Modify attention mechanism to prefer grounding context
response = model.generate(
    prompt=question,
    grounding_docs=retrieved_docs,
    attention_constraint="prioritize_grounding"
)
```

*(Currently research-level, not widely available in production APIs)*

### 3. Knowledge Graph Integration

**Accuracy Improvement:** Varies by domain (10-30% reduction in errors)
**Best For:** Structured knowledge domains

**Integration Points:**

**Pre-retrieval (Query Enhancement):**
```
Query: "When did Einstein win Nobel Prize?"
→ KG expansion: "Albert Einstein, Nobel Prize, Physics, 1921"
→ Enhanced retrieval accuracy
```

**Post-generation (Fact Verification):**
```
LLM Output: "Einstein won Nobel Prize in 1921 for Theory of Relativity"
→ KG check: Prize was for photoelectric effect, not relativity
→ Correction triggered
```

**In-context (Structured Grounding):**
```
Provide KG triples as structured context:
(Albert Einstein, awarded, Nobel Prize in Physics)
(Nobel Prize in Physics, year, 1921)
(Nobel Prize in Physics, for, Photoelectric Effect)
```

### 4. Multi-Agent Validation

**Accuracy Improvement:** 15-25% reduction in errors
**Cost:** 2-3× LLM calls

**Patterns:**

**Critic-Revise Loop:**
```python
def validated_response(query: str):
    # Generator
    response = generator_agent.invoke(query)

    # Critic
    critique = critic_agent.invoke(f"""
    Query: {query}
    Response: {response}

    Check for: factual errors, logical inconsistencies, unsupported claims.
    """)

    # Revise if needed
    if critique.has_issues:
        response = generator_agent.invoke(f"""
        Original query: {query}
        Previous response: {response}
        Issues found: {critique.issues}

        Provide corrected response.
        """)

    return response
```

**Consensus Voting (3+ agents):**
```python
def consensus_answer(query: str, n_agents: int = 3):
    responses = [agent.invoke(query) for _ in range(n_agents)]

    # Vote on answer
    votes = Counter(responses)
    consensus = votes.most_common(1)[0]

    # Require 2/3 agreement
    if consensus[1] >= n_agents * 0.67:
        return consensus[0]
    else:
        return "No consensus - requires human review"
```

### 5. Fine-Tuning on Curated Data

**Accuracy Improvement:** 10-20% for domain-specific tasks
**Best For:** Repeated workflows with known ground truth

**Strategy:**
- Collect 1,000+ high-quality examples
- Include both correct and incorrect attempts
- Train model to recognize reliable patterns

**Limitations:**
- Doesn't guarantee factual accuracy
- Best combined with RAG or validation

### 6. Prompt Engineering for Accuracy

**Techniques:**

**Uncertainty Expression:**
```
If you're not certain about a fact, say "Based on my knowledge..." or
"I'm not completely certain, but..."
```

**Citation Requirements:**
```
For each factual claim, cite the source from the provided context.
Format: [Claim] (Source: document X, paragraph Y)
```

**Explicit Guardrails:**
```
DO NOT:
- Make up statistics or numbers
- Invent quotes or citations
- Extrapolate beyond provided information

If information is not available, say "I don't have that information."
```

### Combined Anti-Hallucination Strategy

**Recommended Production Stack:**

```
1. RAG for knowledge grounding          → 85% → 93% accuracy
2. Structured output validation         → 93% → 96% accuracy
3. Critic agent for high-stakes tasks   → 96% → 98% accuracy
4. Human-in-the-loop for critical cases → 98% → 99%+ accuracy
```

**Cost-Performance Tradeoff:**
- Basic RAG: 1.5× cost, 8% accuracy gain
- + Validation: 2× cost, 11% accuracy gain
- + Critic: 3× cost, 13% accuracy gain
- + HITL: Variable cost, 14%+ accuracy gain

---

## Performance & Speed Optimization

Techniques to achieve 2-10× speedup in agentic workflows.

### 1. Model Quantization

**Speedup:** 2× inference
**Memory Reduction:** 2-4×
**Quality Impact:** 1-3% loss (acceptable for most tasks)

**Quantization Levels:**

| Format | Precision | Memory | Speed | Quality |
|--------|-----------|--------|-------|---------|
| FP32 | Full | 4× | 1× | 100% |
| FP16 | Half | 2× | 1.5× | 99.5% |
| INT8 | 8-bit | 4× | 2× | 97-99% |
| INT4 | 4-bit | 8× | 3× | 92-96% |

**Implementation (AWS Neuron):**
```python
import torch
import torch_neuronx

model = torch.load("model.pt")
quantized = torch_neuronx.quantize(model, dtype=torch.int8)
quantized.save("model_int8.pt")

# Inference
output = quantized(input_tensor)  # 2× faster
```

**Best Practices:**
- Use INT8 for production (best speed/quality tradeoff)
- Validate outputs on test set before deploying
- Combine with batching for maximum throughput

### 2. Hardware Acceleration

**Speedup:** 2-10× depending on hardware

**Options:**

**AWS Inferentia2 (Inf2):**
- 2-4× speedup vs GPU
- 50-70% cost reduction
- Best for high-throughput inference

**NVIDIA GPUs (A100, H100):**
- 5-10× speedup vs CPU
- Best for low-latency requirements
- TensorRT optimization available

**Google TPUs:**
- 3-6× speedup for compatible models
- Optimized for large batch sizes

**Example (GPU acceleration):**
```python
import torch

model = model.to("cuda")  # Move to GPU
input_tensor = input_tensor.to("cuda")

with torch.cuda.amp.autocast():  # Mixed precision
    output = model(input_tensor)  # 3-5× faster
```

### 3. Continuous Batching

**Throughput Improvement:** 3-5× for high request volumes
**Best For:** Production services with varying request rates

**Traditional Batching:**
```
Wait for batch_size requests → Process batch → Return all
Latency: avg_wait_time + processing_time
```

**Continuous Batching:**
```
Process requests as they arrive in dynamic batches
Latency: minimal_wait + processing_time
```

**Tools:**
- vLLM (continuous batching for LLMs)
- TensorRT-LLM
- Text Generation Inference (TGI)

### 4. Streaming Responses

**Perceived Latency:** ~0s (immediate first token)
**Actual Speedup:** None (but better UX)

**Implementation (OpenAI):**
```python
from openai import OpenAI

client = OpenAI()
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**Benefits:**
- User sees progress immediately
- Can cancel if response goes off-track
- Better for long-form generation

### 5. Parallel API Calls

**Speedup:** N× for N independent calls
**Best For:** Multi-agent systems, tool calling

**Sequential (slow):**
```python
result1 = agent1.invoke(query)  # 2s
result2 = agent2.invoke(query)  # 2s
result3 = agent3.invoke(query)  # 2s
# Total: 6s
```

**Parallel (fast):**
```python
import asyncio

async def parallel_agents(query):
    tasks = [
        agent1.ainvoke(query),
        agent2.ainvoke(query),
        agent3.ainvoke(query)
    ]
    results = await asyncio.gather(*tasks)
    return results

# Total: 2s (3× speedup)
```

### 6. Hybrid Routing & Model Cascading

**LLM Usage Reduction:** 37-46%
**Method:** Route simple queries to smaller models

**Strategy:**
```
Classifier → Simple query → GPT-4o-mini ($0.15/1M)
          → Complex query → GPT-4o ($2.50/1M)
```

**Implementation:**
```python
def route_query(query: str):
    # Fast classifier (fine-tuned or embedding-based)
    complexity = classifier.predict(query)

    if complexity < 0.3:
        return cheap_model.invoke(query)  # 90% of queries
    else:
        return expensive_model.invoke(query)  # 10% of queries
```

**Expected Savings:**
- 90% queries to mini: 90% × $0.15 = $0.135
- 10% queries to full: 10% × $2.50 = $0.25
- Total: $0.385 vs $2.50 (85% savings)

### 7. Response Caching (Workflow-Level)

**Speedup:** Infinite (instant response)
**Best For:** Frequently repeated workflows

**Example:**
```python
workflow_cache = {}

def cached_workflow(input_hash):
    if input_hash in workflow_cache:
        return workflow_cache[input_hash]  # Instant

    result = expensive_multi_step_workflow(input_hash)
    workflow_cache[input_hash] = result
    return result
```

**Use Cases:**
- Same analysis on unchanged data
- Repeated code reviews on same PR
- Standard report generation

---

## Combined Optimization Strategies

Real-world production optimizations typically combine multiple techniques.

### Strategy 1: Cost-Optimized RAG System

**Techniques:**
- Semantic caching (60% cost reduction)
- Prompt caching for context (50% reduction on remaining)
- Model cascading (40% reduction on remaining)

**Combined Savings:**
```
Base cost: $100
After semantic cache (60% hit rate): $40
After prompt cache (50% on remaining): $20
After cascading (40% on remaining): $12

Total savings: 88% ($100 → $12)
```

### Strategy 2: Speed-Optimized Multi-Agent

**Techniques:**
- Parallel agent calls (3× speedup)
- INT8 quantization (2× speedup)
- Streaming (perceived 0s latency)

**Combined Speedup:**
```
Base latency: 10s sequential
After parallelization: 3.3s
After quantization: 1.7s
Streaming: First token in 0.2s

Total speedup: 6× (10s → 1.7s)
```

### Strategy 3: Production-Grade Accuracy

**Techniques:**
- RAG (85% → 93% accuracy)
- Output validation (93% → 96%)
- Critic agent for flagged cases (96% → 98%)

**Quality Improvement:**
- 13% absolute accuracy gain
- 87% reduction in error rate (15% → 2%)

---

## Production Monitoring

Essential metrics and dashboards for optimized systems.

### Key Metrics

**Cost Metrics:**
- Cost per request
- Cost per successful task completion
- Cache hit rate (target: 40-70%)
- Token efficiency (tokens per task)

**Performance Metrics:**
- P50, P95, P99 latency
- First token latency (streaming)
- Throughput (requests/second)
- Batch size utilization

**Quality Metrics:**
- Task success rate
- Hallucination rate (human eval)
- User satisfaction score
- Retry rate

### Monitoring Dashboard

**Real-Time Alerts:**
```python
# Cost spike detection
if hourly_cost > baseline * 1.5:
    alert("Cost spike detected")

# Quality degradation
if success_rate < 0.90:
    alert("Success rate below threshold")

# Latency issues
if p95_latency > 5000:  # 5s
    alert("High latency detected")
```

**Weekly Reviews:**
- Cache hit rate trends
- Cost per task trends
- Model performance comparisons
- Optimization opportunities

### Continuous Improvement

**Monthly Optimization Cycle:**
1. Analyze metrics for bottlenecks
2. Test optimization hypotheses
3. A/B test improvements
4. Roll out successful changes
5. Update baselines

**Example ROI:**
- Month 1: Implement semantic caching → 60% cost reduction
- Month 2: Optimize prompts → Additional 20% reduction
- Month 3: Add model cascading → Additional 15% reduction
- Cumulative: 88% cost reduction

---

## Quick Reference

### Cost Optimization Priority

1. **Semantic caching** (60-90% savings, medium effort)
2. **Prompt caching** (50-90% savings, low effort)
3. **Concise prompting** (30-50% savings, low effort)
4. **Model cascading** (40-60% savings, medium effort)
5. **Fine-tuning** (50-75% savings, high effort, long-term)

### Speed Optimization Priority

1. **Parallel calls** (N× speedup, low effort)
2. **Streaming** (perceived 0s latency, low effort)
3. **INT8 quantization** (2× speedup, medium effort)
4. **Continuous batching** (3-5× throughput, medium effort)
5. **Hardware upgrade** (2-10× speedup, high cost)

### Accuracy Optimization Priority

1. **RAG** (8% accuracy gain, medium effort)
2. **Output validation** (3% gain, low effort)
3. **Prompt guardrails** (2-5% gain, low effort)
4. **Critic agent** (2-3% gain, medium effort)
5. **Fine-tuning** (10-20% gain, high effort)

### Expected Outcomes (Combined)

**Aggressive Optimization:**
- Cost: 80-90% reduction
- Speed: 3-5× improvement
- Accuracy: 85% → 96%+
- Effort: 4-6 weeks implementation

**Balanced Optimization:**
- Cost: 60-70% reduction
- Speed: 2-3× improvement
- Accuracy: 85% → 93%+
- Effort: 2-3 weeks implementation

**Quick Wins:**
- Cost: 40-50% reduction
- Speed: 1.5-2× improvement
- Accuracy: 85% → 90%+
- Effort: 3-5 days implementation

---

## References

**Academic Research:**
- Semantic Caching: "Efficient LLM Inference via Semantic Caching" (arXiv:2502.xxxxx)
- Hallucination Reduction: "Fact Grounded Attention for LLMs" (arXiv:2501.xxxxx)
- Multi-Agent Systems: "Foundations and Emerging Paradigms of Multi-Agent LLM Systems" (arXiv:2503.xxxxx)

**Industry Sources:**
- Anthropic Prompt Caching Documentation (2025)
- OpenAI Optimization Best Practices (2025)
- AWS Inferentia2 Performance Benchmarks
- GPTCache Framework Documentation

**Production Case Studies:**
- 85% cost reduction with Anthropic caching (customer case study)
- 68.8% API reduction with semantic caching (research paper)
- 37-46% LLM usage reduction with hybrid routing (industry report)

---

**Document Version:** 1.0
**Related Documents:**
- api-optimization-guide.md - Strategic model selection
- patterns-and-antipatterns.md - Failure modes and fixes
- agentic-systems-cookbook.md - Implementation recipes
- theoretical-foundations.md - Research foundations
