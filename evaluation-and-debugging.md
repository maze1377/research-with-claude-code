# Agent Evaluation, Debugging & Improvement

**Comprehensive guide for evaluating, debugging, and improving LLM agents**

---

## December 2025 Statistics

| Metric | Value | Source |
|--------|-------|--------|
| Organizations with agents deployed | 57% | Industry survey |
| Production-ready infrastructure | **Only 6%** | Industry survey |
| Multi-agent system failure rate | 60% | MAST Research |
| RAFFLES fault attribution accuracy | 43.6% (vs 16.6% baseline) | arXiv 2025 |
| DoVer failure recovery | 18-49% | ICLR 2026 submission |
| Agentic plan caching cost reduction | 50.31% | NeurIPS 2025 |
| Predicted project cancellation by 2027 | 40% | Gartner |

---

## Executive Summary

| Challenge | Solution | Impact |
|-----------|----------|--------|
| Measuring agent performance | Multi-dimensional evaluation (CLASSic framework) | Reliable quality metrics |
| Debugging failures | RAFFLES + DoVer intervention testing | 18-49% failure recovery |
| Understanding failure modes | MAST taxonomy (14 modes, 3 categories) | Targeted fixes |
| Production monitoring | Observability platforms + OpenTelemetry | Real-time issue detection |
| Cost optimization | Agentic plan caching | 50-76% cost reduction |
| Continuous improvement | Feedback loops + A/B testing | Iterative refinement |

---

## 1. Agent Evaluation Framework

### 1.1 Evaluation Dimensions (KDD 2025 Taxonomy)

**What to Evaluate:**
- Agent behavior and decision-making
- Task completion capabilities
- Reliability and consistency
- Safety and alignment

**How to Evaluate:**
- Interaction modes (single-turn, multi-turn)
- Datasets and benchmarks
- Metric computation methods
- Evaluation tooling

### 1.2 Single-Agent Evaluation

| Metric | Description | Measurement |
|--------|-------------|-------------|
| Task Success Rate | Did agent complete the task? | Binary or graded |
| Reasoning Depth | Quality of reasoning steps | LLM-as-Judge |
| Tool Usage Correctness | Right tool, right parameters | Precision/Recall |
| Latency | Time to complete | P50, P95, P99 |
| Factual Accuracy | Correctness of outputs | Ground truth comparison |
| Token Efficiency | Tokens used per task | Cost tracking |

**Fudan NLP 4-Dimensional Framework:**
```
1. Task Completion → Did it succeed?
2. Reasoning Quality → Was the logic sound?
3. Tool Usage → Were tools used correctly?
4. Efficiency → Was it cost/time effective?
```

### 1.3 Multi-Agent Evaluation

| Metric | Description | Challenge |
|--------|-------------|-----------|
| Coordination Efficiency | How well agents work together | Hard to measure |
| Communication Overhead | Messages between agents | Token/latency cost |
| Plan Quality | Effectiveness of shared plans | Subjective |
| Group Alignment | Do agents share goals? | Emergent behavior |
| Milestone Progress | Incremental task completion | MultiAgentBench |

**Key Insight**: Traditional single-agent metrics fall short for multi-agent systems. You need specialized benchmarks like MultiAgentBench or MARBLE.

### 1.4 Evaluation Methods

**Offline Evaluation (Development)**
```
Purpose: Test before deployment
When: CI/CD pipeline, local development
Tools: Langfuse, DeepEval, custom harnesses
Data: Curated test datasets

Workflow:
1. Define test cases with expected outputs
2. Run agent on test set
3. Compare outputs to ground truth
4. Calculate metrics (accuracy, F1, etc.)
5. Track regressions across versions
```

**Online Evaluation (Production)**
```
Purpose: Measure real-world performance
When: After deployment
Tools: A/B testing, user feedback, monitoring
Data: Live user interactions

Workflow:
1. Deploy variants (control vs treatment)
2. Route traffic to variants
3. Collect user feedback and metrics
4. Analyze statistical significance
5. Roll out winning variant
```

### 1.5 LLM-as-a-Judge

Use a separate LLM to evaluate agent outputs:

```
Evaluation Types:
├── Pairwise Comparison → "Which response is better?"
├── Direct Scoring → "Rate this response 1-5"
├── Criteria-Based → "Does this meet criteria X, Y, Z?"
└── Reference-Based → "How close to the reference?"

Best Practices:
- Use stronger model as judge (GPT-4, Claude Opus)
- Provide clear rubrics
- Use multiple judges for consensus
- Track judge reliability over time
```

### 1.6 Key Benchmarks (December 2025)

| Benchmark | Focus | Instances | Top Score |
|-----------|-------|-----------|-----------|
| **SWE-bench Full** | Real GitHub issues | 2,294 | Gemini 3 Pro: 74.2% |
| **SWE-bench Verified** | Engineer-confirmed | 500 | GPT-5.2: 71.8% |
| **SWE-bench Multimodal** | Visual elements | 517 | NEW in 2025 |
| **AgentBench** | 8 environments | 29 LLMs tested | See benchmark |
| **MARBLE/MultiAgentBench** | Multi-agent teamwork | Milestone KPIs | +3% with cognitive planning |
| **MMAU** | 5 domains, 5 capabilities | 3,000+ prompts | Gemini 2.0: 59.9% |
| **GAIA** | General assistant | Multi-level | Broad capability |
| **WebArena** | Web navigation | Browser tasks | Operator 38.1% |
| **BFCL** | Function calling | Tool use | Accuracy ranking |

**SWE-bench Variants:**
```
Full (2,294)        → Complete evaluation, high cost
Lite (300)          → Cost-effective subset
Verified (500)      → Engineer-confirmed solvable
Multimodal (517)    → Issues with visual elements
Bash Only           → Shell-focused subset
```

**MMAU 5 Capabilities:**
1. Understanding
2. Reasoning
3. Planning
4. Problem-solving
5. Self-correction

---

## 2. Debugging & Observability

### 2.1 The Observability Stack

```
Level 1: Logging
    ↓ What happened?
Level 2: Metrics
    ↓ How is it performing?
Level 3: Traces
    ↓ Why did it happen?
Level 4: Profiling
    ↓ Where are bottlenecks?
```

### 2.2 Trace Analysis

**What to Capture:**
- Input prompts and context
- LLM responses at each step
- Tool calls and results
- Memory reads/writes
- Agent handoffs (multi-agent)
- Latency per step
- Token usage per step

**Trace Structure:**
```
Trace
├── Span: User Input
├── Span: Context Retrieval
│   ├── Span: Vector Search
│   └── Span: Reranking
├── Span: LLM Reasoning
│   ├── Span: Tool Selection
│   └── Span: Response Generation
├── Span: Tool Execution
│   └── Span: API Call
└── Span: Final Response
```

### 2.3 Observability Tools Comparison

| Tool | Type | Best For | Key Features |
|------|------|----------|--------------|
| **LangSmith** | Commercial | LangChain users | Deep integration, low overhead |
| **Langfuse** | Open Source | General use | Self-hosted, MIT license |
| **DeepEval** | Open Source | Testing focus | Spans, component evals |
| **Traceloop** | Commercial | Production | Continuous feedback |
| **AgentOps** | Commercial | Agent-specific | Multi-agent workflows |

### 2.4 MAST: Multi-Agent System Failure Taxonomy (NEW)

**Source**: arXiv:2503.13657 - Analysis of 1,600+ annotated execution traces

| Category | % of Failures | Key Failure Modes |
|----------|---------------|-------------------|
| **System Design Issues** | 44.2% | Disobeying spec (15.7%), Step repetition (13.2%), Context loss (6.8%), Completion recognition (12.4%) |
| **Inter-Agent Misalignment** | 32.3% | Assumption drift (12.4%), Task derailment (11.8%), Info withholding (8.2%), Reasoning-action mismatch (13.2%) |
| **Task Verification** | 23.5% | Premature termination (9.1%), Incomplete verification (8.2%), Incorrect verification (2.8%) |

**Critical Insight**: Most failures are fixable through system design improvements WITHOUT better models.

```
System Design Interventions:
    Better role specs → +9.4% success (ChatDev study)
    Checkpointing    → Time-travel debugging enabled
    Reflexion pattern → 50% → 90% success (2-3x latency)
```

### 2.5 DoVer Auto-Debugging Framework (ICLR 2026)

**Key Innovation**: Intervention-driven debugging ("do-then-verify") that actively tests failure hypotheses.

```
How It Works:
1. Segment execution logs into trials
2. Apply targeted interventions (modify messages, alter plans)
3. Re-execute from intervention point
4. Empirically test if changes resolve failures
5. Validate or refute hypotheses

Results (December 2025):
- Recovers 18% failures (AssistantBench)
- Recovers 28% failures (GAIA)
- Recovers **49%** failures (GSMPlus with AutoGen2)
- Validates 30-60% of hypotheses
- Self-correction baselines: 0% recovery (same dataset)
```

**DoVer vs Self-Correction:**
| Method | Recovery Rate (Who&When) |
|--------|--------------------------|
| CRITIC-style self-correction | 0% |
| DoVer intervention | **17.6%** |

### 2.6 RAFFLES Framework (NeurIPS 2025)

**Key Innovation**: Structured reasoning for fault attribution using Judge + Evaluators.

```
Architecture:
    ┌─────────────────────────────────────────┐
    │             RAFFLES Pipeline            │
    │  ┌─────────┐    ┌─────────────────┐    │
    │  │  Judge  │←──→│   Evaluators    │    │
    │  │ Agent   │    │ (Component +    │    │
    │  └────┬────┘    │  Reasoning)     │    │
    │       │         └─────────────────┘    │
    │       ↓                                 │
    │  Hypothesis History                     │
    └─────────────────────────────────────────┘

Results:
- Step-level accuracy: 43.6% (vs 16.6% baseline) - 162% improvement
- Hand-crafted dataset: 20%+ (vs 8.8% baseline)
- Outperforms Tool-Caller baseline by 31-53%
```

### 2.7 Who&When Benchmark

First benchmark for automated failure attribution:
- **127 LLM multi-agent systems** with fine-grained annotations
- **184 annotated failure tasks** from CaptainAgent and Magnetic-One
- Best method: 53.5% agent identification, only 14.2% step identification
- Even o1 and DeepSeek R1 fail at practical usability

### 2.8 TRAIL Framework

Turn-level traces with fine-grained taxonomy:

| Category | Examples |
|----------|----------|
| Reasoning Errors | Wrong inference, missed context |
| Planning Errors | Bad decomposition, wrong order |
| Execution Errors | Tool failure, wrong parameters |
| Memory Errors | Lost context, wrong retrieval |

**Finding**: Even strong long-context models struggle at trace debugging.

### 2.9 Debugging Workflow

```
1. DETECT
   └── Monitor alerts, user reports, metric anomalies

2. ISOLATE
   └── Find the failing trace(s)
   └── Identify the failing span(s)

3. REPRODUCE
   └── Create minimal test case
   └── Verify failure is consistent

4. ANALYZE
   └── Examine inputs, prompts, context
   └── Check tool responses
   └── Review agent reasoning

5. HYPOTHESIZE
   └── Form theory about root cause
   └── Test with interventions (DoVer)

6. FIX
   └── Update prompts, tools, or logic
   └── Add guardrails if needed

7. VERIFY
   └── Run regression tests
   └── Monitor in production
```

---

## 3. Improvement Loops

### 3.1 Continuous Improvement Cycle

```
       ┌──────────────────────────────────────┐
       │                                      │
       ↓                                      │
   COLLECT ──→ ANALYZE ──→ IMPROVE ──→ DEPLOY
       ↑                                      │
       └──────────────────────────────────────┘

Collect: User feedback, metrics, traces
Analyze: Find patterns, identify issues
Improve: Update prompts, logic, tools
Deploy: Release changes, monitor
```

### 3.2 Feedback Loop Types

**Explicit Feedback**
- Thumbs up/down from users
- Star ratings
- Written comments
- Correction submissions

**Implicit Feedback**
- Task completion rate
- Retry behavior
- Session length
- Abandonment rate

**LLM Feedback**
- LLM-as-Judge scores
- Self-critique
- Reflection outputs

### 3.3 A/B Testing for Agents

**AgentA/B Framework (2025)**
```
Innovation: Use LLM agents to simulate user behavior

Setup:
1. Generate 100K virtual customer personas
2. Sample 1K agents for simulation
3. Run both variants against simulated users
4. Measure behavior differences

Finding:
- LLM agents take fewer actions than humans
- Behavioral direction matches human A/B tests
- Valid complement to traditional testing
```

**Traditional A/B Testing:**
```
1. Define variants (prompts, models, logic)
2. Split traffic randomly
3. Run for statistical significance
4. Measure key metrics
5. Roll out winner
```

### 3.4 Error Pattern Mining

**Braintrust Loop Approach:**
```
1. Collect millions of traces
2. Analyze logs at scale
3. Surface common patterns
4. Generate improvement reports
5. Suggest specific fixes
```

### 3.5 Agentic Plan Caching (NeurIPS 2025)

**Innovation**: Reduce LLM serving costs by reusing prior execution plans.

```
How It Works:
1. Extract plan templates from completed agent executions
2. Use keyword extraction to match new requests to cached plans
3. Lightweight models adapt templates to task-specific contexts
4. Reuse adapted plans instead of full re-planning

Results:
- Average cost reduction: **50.31%**
- GAIA benchmark: 76.42% cost reduction ($69.02 → $16.27)
- Accuracy maintained: 96.61% of optimal
- Overhead: Only 1.04% of total serving cost
- Works alongside existing caching techniques
```

---

## 4. Production Agent Characteristics (December 2025)

### 4.1 Real-World Deployment Statistics

| Characteristic | Value | Implication |
|----------------|-------|-------------|
| Max steps before human intervention | 68% ≤ 10 steps, 47% ≤ 5 | Deliberate autonomy constraints |
| Reliance on prompting vs fine-tuning | 70% prompting only | Lower barrier, less specialization |
| Human vs automated evaluation | 74% human | Trust gaps in auto-metrics |
| Manual prompt construction | 79% manual | 10,000+ token production prompts |
| Use of third-party frameworks | 85% avoid | Custom builds preferred |

**Key Insight**: Production agents are deliberately constrained for reliability, not capability.

### 4.2 CLASSic Framework (Enterprise Readiness)

| Dimension | What It Measures | Key Metrics |
|-----------|------------------|-------------|
| **C**ost | Token usage, API costs, inference | Semantic cache hit rate (38% typical) |
| **L**atency | Time to response | Affects 15% of deployments |
| **A**ccuracy | Task correctness | Trust and hallucination prevention |
| **S**ecurity | Access control, action limits | Preventing unauthorized actions |
| **S**tability | Consistent performance at scale | Variation under load |

### 4.3 Production Readiness Gap

```
The 51-Point Gap:
    57% have agents in production
    -  6% have production-ready infrastructure
    ─────────────────────────────────────────
    51% operating without proper support

Common Pitfalls:
    ├── Over-engineering (too many agents)
    ├── Poor error handling (no circuit breakers)
    ├── High costs (excessive LLM calls)
    └── Latency (sequential API calls)
```

---

## 5. Error Recovery & Resilience

### 5.1 Retry Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| Immediate Retry | Retry right away | Transient network errors |
| Exponential Backoff | Increase delay each retry | Rate limits |
| Jitter | Add randomness to delay | Avoid thundering herd |
| Limited Retries | Cap max attempts | Prevent infinite loops |

**Exponential Backoff Formula:**
```
delay = min(cap, base * 2^attempt + random_jitter)

Example:
- Base: 1 second
- Cap: 60 seconds
- Attempts: 1=1s, 2=2s, 3=4s, 4=8s, 5=16s...
```

### 5.2 Circuit Breakers

```
States:
┌────────┐     fails > threshold     ┌────────┐
│ CLOSED │ ─────────────────────────→ │  OPEN  │
└────────┘                            └────────┘
     ↑                                     │
     │         timeout expires             │
     │                                     ↓
     │                              ┌────────────┐
     └──────── success ────────────│ HALF-OPEN  │
                                   └────────────┘

Configuration:
- Failure threshold: 5 failures
- Reset timeout: 30 seconds
- Half-open successes: 3 to close
```

### 5.3 Fallback Chains

```
Primary Model → Fallback 1 → Fallback 2 → Default Response
     │              │             │              │
   GPT-4o       Claude-3.5    GPT-4o-mini    Static response

Error Classification:
├── Non-retryable (4xx except 429) → Throw immediately
├── Retryable (5xx, 429, timeout) → Try fallbacks
└── Unknown → Log and retry
```

### 5.4 Multi-Tier Retry Architecture

```
Tier 1: User-Level
    └── Frontend retry button
    └── Auto-retry with notification

Tier 2: Application-Level
    └── Exception catching
    └── Status tracking
    └── Graceful degradation

Tier 3: Database-Level
    └── Connection pooling
    └── Automatic reconnection
    └── Query retry
```

### 5.5 Schema Validation Recovery

```
Attempt 1: Generate response
     ↓
Validate against schema (Pydantic)
     ↓
Failed? → Retry with error context
     ↓
Attempt 2: Generate with feedback
     ↓
Still failed? → Escalate or fallback
```

---

## 6. Observability Metrics

### 6.1 Key Metrics to Track

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Task Success Rate | < 90% | < 80% | Review failures |
| P95 Latency | > 5s | > 15s | Optimize or scale |
| Error Rate | > 2% | > 5% | Debug immediately |
| Token Cost/Task | > budget | 2x budget | Optimize prompts |
| Retry Rate | > 10% | > 25% | Fix root cause |

### 6.2 Dashboard Components

```
Real-Time Panel:
├── Request rate (RPM)
├── Error rate (%)
├── Latency distribution
└── Active sessions

Historical Panel:
├── Success rate trend
├── Cost trend
├── Latency trend
└── Error breakdown

Agent-Specific Panel:
├── Tool usage distribution
├── Reasoning step count
├── Memory operations
└── Handoff frequency (multi-agent)
```

### 6.3 Alerting Rules

```
Critical Alerts:
- Error rate > 5% for 5 minutes
- Latency P99 > 30s
- Circuit breaker opened
- Cost spike > 3x normal

Warning Alerts:
- Error rate > 2% for 10 minutes
- Latency P95 > 10s
- Retry rate > 15%
- Unusual tool failure rate
```

---

## 7. Tool Comparison

### 7.1 By Team Size

| Size | Recommended | Reasoning |
|------|-------------|-----------|
| Solo/Pairs | Helicone, Langfuse | Quick setup, free tier |
| Small (3-10) | Langfuse, LangSmith | All-in-one, less maintenance |
| Medium (11-50) | LangSmith, Arize | Scalability, support |
| Enterprise | Datadog, New Relic | Integration with existing stack |

### 7.2 By Use Case

| Use Case | Best Tool |
|----------|-----------|
| LangChain apps | LangSmith |
| Self-hosted requirement | Langfuse |
| Testing focus | DeepEval |
| Multi-agent | AgentOps |
| Cost tracking | Helicone |

---

## 8. Best Practices Checklist

### Pre-Deployment
- [ ] Define evaluation metrics
- [ ] Create test dataset
- [ ] Set up offline evaluation pipeline
- [ ] Configure observability
- [ ] Set alert thresholds
- [ ] Document fallback behavior

### Launch
- [ ] Enable tracing
- [ ] Start with limited traffic
- [ ] Monitor all dashboards
- [ ] Set up on-call rotation

### Post-Launch
- [ ] Review failure traces daily
- [ ] Run A/B tests for improvements
- [ ] Update test cases from production failures
- [ ] Track cost trends
- [ ] Conduct weekly improvement reviews

---

## Quick Reference

```
EVALUATE → TRACE → DEBUG → IMPROVE → MONITOR

Evaluation (December 2025):
  CLASSic: Cost, Latency, Accuracy, Security, Stability
  Benchmarks: SWE-bench (74.2%), MMAU (59.9%), MARBLE

Tracing:
  Tools: LangSmith, Langfuse, Maxim AI, AgentOps
  Standards: OpenTelemetry semantic conventions

Debugging (NEW):
  MAST: 14 failure modes, 3 categories (44%, 32%, 24%)
  RAFFLES: 43.6% fault attribution (162% improvement)
  DoVer: 18-49% failure recovery (intervention-driven)

Production Reality:
  57% deployed, 6% ready → 51-point gap
  68% ≤ 10 steps, 74% human eval, 79% manual prompts

Cost Optimization:
  Plan caching: 50-76% cost reduction
  Semantic caching: 38% hit rate

Resilience:
  Retry: exponential backoff with jitter
  Fallback: model chains with classification
  Circuit: break on threshold, reset on timeout
```

---

## Related Documents

- [patterns-and-antipatterns.md](patterns-and-antipatterns.md) - Failure modes and fixes
- [api-optimization-guide.md](api-optimization-guide.md) - Cost and latency optimization
- [topics.md](topics.md) - Quick reference for all topics
- [task.md](task.md) - Research tracking and resources

---

**Document Version**: 2.0 (Updated with December 2025 research)
**Last Updated**: 2025-12-25
**Status**: Concepts and references (no production implementations)

**Sources**:
- MAST: arXiv:2503.13657 (1,600+ execution traces, 14 failure modes)
- RAFFLES: NeurIPS 2025 (43.6% fault attribution accuracy)
- DoVer: ICLR 2026 submission (18-49% failure recovery)
- Who&When: arXiv 2025 (127 multi-agent systems benchmark)
- Agentic Plan Caching: NeurIPS 2025 (50.31% cost reduction)
- [KDD 2025 Tutorial](https://sap-samples.github.io/llm-agents-eval-tutorial/)
- [Langfuse Docs](https://langfuse.com/docs)
- [LangSmith Docs](https://www.langchain.com/langsmith)
- [OpenTelemetry AI Agent Observability](https://opentelemetry.io/blog/2025/ai-agent-observability/)
