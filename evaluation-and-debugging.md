# Agent Evaluation, Debugging & Improvement

**Comprehensive guide for evaluating, debugging, and improving LLM agents**

---

## Executive Summary

| Challenge | Solution | Impact |
|-----------|----------|--------|
| Measuring agent performance | Multi-dimensional evaluation (task, tool, coordination) | Reliable quality metrics |
| Debugging failures | Trace analysis + intervention testing | 18-28% failure recovery |
| Production monitoring | Observability platforms | Real-time issue detection |
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

### 1.6 Key Benchmarks

| Benchmark | Focus | Agents Tested |
|-----------|-------|---------------|
| **AgentBench** | 8 environments (OS, DB, web, games) | Single agents |
| **MARBLE** | Multi-agent teamwork | Coordination |
| **SWE-bench** | Code generation/editing | Coding agents |
| **GAIA** | General assistant tasks | Broad capability |
| **WebArena** | Web navigation | Browser agents |
| **BFCL** | Function calling accuracy | Tool use |

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

### 2.4 DoVer Auto-Debugging Framework

**Key Innovation**: Intervention-driven debugging that automatically tests failure hypotheses.

```
How It Works:
1. Decompose failure trace into separate trials
2. Intervene at each step
3. Test if intervention fixes failure
4. Validate or refute hypotheses
5. Apply successful interventions

Results:
- Recovers 18% failures (AssistantBench)
- Recovers 28% failures (GAIA)
- Validates 30-60% of hypotheses
```

### 2.5 TRAIL Framework

Turn-level traces with fine-grained taxonomy:

| Category | Examples |
|----------|----------|
| Reasoning Errors | Wrong inference, missed context |
| Planning Errors | Bad decomposition, wrong order |
| Execution Errors | Tool failure, wrong parameters |
| Memory Errors | Lost context, wrong retrieval |

**Finding**: Even strong long-context models struggle at trace debugging.

### 2.6 Debugging Workflow

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

---

## 4. Error Recovery & Resilience

### 4.1 Retry Strategies

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

### 4.2 Circuit Breakers

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

### 4.3 Fallback Chains

```
Primary Model → Fallback 1 → Fallback 2 → Default Response
     │              │             │              │
   GPT-4o       Claude-3.5    GPT-4o-mini    Static response

Error Classification:
├── Non-retryable (4xx except 429) → Throw immediately
├── Retryable (5xx, 429, timeout) → Try fallbacks
└── Unknown → Log and retry
```

### 4.4 Multi-Tier Retry Architecture

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

### 4.5 Schema Validation Recovery

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

## 5. Observability Metrics

### 5.1 Key Metrics to Track

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Task Success Rate | < 90% | < 80% | Review failures |
| P95 Latency | > 5s | > 15s | Optimize or scale |
| Error Rate | > 2% | > 5% | Debug immediately |
| Token Cost/Task | > budget | 2x budget | Optimize prompts |
| Retry Rate | > 10% | > 25% | Fix root cause |

### 5.2 Dashboard Components

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

### 5.3 Alerting Rules

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

## 6. Tool Comparison

### 6.1 By Team Size

| Size | Recommended | Reasoning |
|------|-------------|-----------|
| Solo/Pairs | Helicone, Langfuse | Quick setup, free tier |
| Small (3-10) | Langfuse, LangSmith | All-in-one, less maintenance |
| Medium (11-50) | LangSmith, Arize | Scalability, support |
| Enterprise | Datadog, New Relic | Integration with existing stack |

### 6.2 By Use Case

| Use Case | Best Tool |
|----------|-----------|
| LangChain apps | LangSmith |
| Self-hosted requirement | Langfuse |
| Testing focus | DeepEval |
| Multi-agent | AgentOps |
| Cost tracking | Helicone |

---

## 7. Best Practices Checklist

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

Evaluation:
  Single-Agent: success rate, latency, accuracy
  Multi-Agent: coordination, communication, alignment

Tracing:
  Tools: LangSmith, Langfuse, DeepEval
  Capture: prompts, responses, tools, latency

Debugging:
  DoVer: intervention-driven (18-28% recovery)
  TRAIL: turn-level taxonomy

Improvement:
  A/B testing: AgentA/B with simulated users
  Feedback: explicit + implicit + LLM

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

**Document Version**: 1.0
**Last Updated**: December 2025
**Status**: Concepts and references (no production implementations)

**Sources**:
- [KDD 2025 Tutorial](https://sap-samples.github.io/llm-agents-eval-tutorial/)
- [DoVer Framework](https://arxiv.org/abs/2512.06749)
- [Langfuse Docs](https://langfuse.com/docs)
- [LangSmith Docs](https://www.langchain.com/langsmith)
- [AgentA/B Paper](https://arxiv.org/abs/2504.09723)
