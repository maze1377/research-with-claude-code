# Advanced Agent Paradigms

**Self-improvement, advanced planning, and learning techniques for LLM agents**

---

## Executive Summary

| Paradigm | Key Innovation | Performance Gain |
|----------|----------------|------------------|
| Reflexion | Verbal reinforcement learning | Significant improvement (p < 0.001) |
| Plan-and-Execute | Separate planner/executor | 57.58% WebArena success |
| ReWOO | Pre-planned tool chains | 80% token reduction |
| Memento | Memory-based RL | 87.88% GAIA (top-1) |
| Deep Agent | Hierarchical Task DAG | Complex scenario handling |
| GoalAct | Continuous global planning | 12.22% improvement |

---

## 1. Self-Improvement Paradigms

### 1.1 Reflexion

**Core Idea**: Agents learn from verbal self-reflection without fine-tuning.

```
Architecture:
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Actor ──→ Environment ──→ Evaluator              │
│     ↑                           │                  │
│     │         Feedback          │                  │
│     │                           ↓                  │
│     └────── Self-Reflection ◄───┘                  │
│                   │                                │
│                   ↓                                │
│            Episodic Memory                         │
│                                                    │
└─────────────────────────────────────────────────────┘
```

**Components:**
| Component | Role | Implementation |
|-----------|------|----------------|
| Actor | Generates actions | CoT or ReAct agent |
| Evaluator | Scores outcomes | LLM or heuristic |
| Self-Reflection | Critiques attempts | LLM-generated feedback |
| Memory | Stores reflections | Episodic buffer |

**How It Works:**
1. Actor attempts task
2. Evaluator scores the attempt
3. Self-Reflection generates critique
4. Critique stored in memory
5. Next attempt includes past reflections as context

**Key Finding**: All types of self-reflection improve performance (p < 0.001).

**Advantages:**
- No fine-tuning required
- Lightweight and efficient
- Works with any base LLM
- Interpretable improvements

### 1.2 Gödel Agent

**Core Idea**: Self-modifying agents inspired by the Gödel machine.

```
Traditional Agent:
  Fixed logic ──→ Execute ──→ Output

Gödel Agent:
  Logic ──→ Self-Evaluation ──→ Modify Logic ──→ Execute
    ↑                                              │
    └──────────────────────────────────────────────┘
```

**Key Properties:**
- Recursively improves own logic
- No predefined optimization routines
- Guided by high-level objectives
- Dynamic behavior modification

**Results:**
- Surpasses manually-crafted agents
- Better efficiency and generalizability
- Continuous self-improvement on math/reasoning

**Caution**: Raises alignment concerns as agents modify themselves.

### 1.3 LADDER Framework

**Core Idea**: Recursive generation of progressively simpler problem variants.

```
Complex Problem
      ↓
Generate Simpler Variant
      ↓
Solve Simpler Variant
      ↓
Learn from Solution
      ↓
Apply to Original Problem
      ↓
Repeat Until Solved
```

**Results:**
- 3B parameter Llama 3.2: 1% → 82% on integration problems
- No human intervention required
- Creates natural difficulty gradient

### 1.4 AlphaEvolve

**Core Idea**: Evolutionary optimization of algorithms by LLM.

```
Process:
1. Start with initial algorithm + metrics
2. LLM proposes mutations/combinations
3. Evaluate new candidates
4. Select best performers
5. Repeat evolution
```

**Key Innovation**: Can optimize components of itself.

**Limitation**: Requires automated evaluation functions.

### 1.5 Memento/AgentFly

**Core Idea**: Memory-based RL without fine-tuning.

```
Traditional RL: Update model weights
Memento: Update memory contents

Memory Structure:
├── Experience Store → Past attempts and outcomes
├── Strategy Cache → Successful approaches
└── Reflection Log → Self-critiques
```

**Results:**
- GAIA validation: 87.88% Pass@3 (top-1)
- GAIA test: 79.40%
- Outperforms training-based methods

**Advantages:**
- No gradient updates needed
- Continual adaptation
- Low computational cost

---

## 2. Advanced Planning Patterns

### 2.1 Plan-and-Execute

**Core Idea**: Separate planning from execution for better control.

```
┌─────────────┐      Plan       ┌─────────────┐
│   Planner   │ ─────────────→  │  Executor   │
│   (LLM)     │                 │  (Tools)    │
└─────────────┘                 └─────────────┘
       ↑                              │
       │         Results              │
       └──────────────────────────────┘
```

**Components:**
| Component | Responsibility |
|-----------|----------------|
| Planner | Generate multi-step plan |
| Executor | Execute individual steps |
| Joiner | Decide if replanning needed |

**Benefits:**
- Cost savings (smaller models for execution)
- Better reasoning (explicit planning)
- Easier debugging (clear separation)

**Results:**
- Plan-and-Act: 57.58% WebArena-Lite success
- Plan-and-Act: 81.36% WebVoyager success

### 2.2 LLMCompiler

**Core Idea**: DAG-based task scheduling with parallel execution.

```
       Task DAG
         ┌─────┐
         │  A  │
         └──┬──┘
       ┌────┴────┐
       ↓         ↓
    ┌─────┐   ┌─────┐
    │  B  │   │  C  │  ← Can run in parallel
    └──┬──┘   └──┬──┘
       └────┬────┘
            ↓
         ┌─────┐
         │  D  │
         └─────┘
```

**Components:**
1. **Planner**: Streams DAG of tasks
2. **Task Fetching Unit**: Schedules by dependencies
3. **Joiner**: Replans or finishes based on history

### 2.3 Hierarchical Task DAG (Deep Agent)

**Core Idea**: Multi-layer decomposition for complex tasks.

```
Level 0: High-Level Goal
         ↓
Level 1: Sub-Goals
         ↓
Level 2: Tasks
         ↓
Level 3: Actions
```

**Key Innovation**: Dynamic decomposition prevents premature over-planning.

**Benefits:**
- Handles complex scenarios
- Maintains LLM focus on relevant tasks
- Prevents getting lost in details

### 2.4 GoalAct Framework

**Core Idea**: Continuous global planning with hierarchical execution.

```
┌─────────────────────────────────────────┐
│           Global Planner                │
│  (Continuously updated world model)     │
└───────────────────┬─────────────────────┘
                    │
        ┌───────────┴───────────┐
        ↓           ↓           ↓
    ┌───────┐  ┌───────┐  ┌───────┐
    │Search │  │ Code  │  │ Write │  High-level skills
    └───────┘  └───────┘  └───────┘
```

**Results:**
- State-of-the-art performance
- 12.22% average improvement in success rate
- Reduced planning complexity

### 2.5 ReWOO (Reasoning Without Observation)

**Core Idea**: Generate full plan before any execution.

```
ReAct (Traditional):
  Think → Act → Observe → Think → Act → Observe...
  (Many LLM calls, redundant context)

ReWOO:
  Plan (all steps) → Execute (all steps) → Solve
  (Fewer LLM calls, no redundant context)
```

**Components:**
| Component | Role |
|-----------|------|
| Planner | Full multi-step plan upfront |
| Worker | Executes all steps (can parallelize) |
| Solver | Synthesizes final answer |

**Results:**
- HotpotQA: 42.4% (ReWOO) vs 40.8% (ReAct)
- Token usage: 2,000 vs 9,795 (80% reduction)

**Best For:**
- Tasks where plan is stable
- Cost-sensitive applications
- Parallelizable subtasks

### 2.6 BOLAA (Benchmarking and Orchestrating LLM-Augmented Autonomous Agents)

**Core Idea**: Specialized agents orchestrated together.

```
┌─────────────────────────────────────────┐
│            Orchestrator                 │
└───────────────────┬─────────────────────┘
        ┌───────────┼───────────┐
        ↓           ↓           ↓
    ┌────────┐  ┌────────┐  ┌────────┐
    │Reasoning│  │Searching│  │ Expert │
    │ Agent   │  │ Agent   │  │ Agent  │
    └────────┘  └────────┘  └────────┘
```

**Key Finding**: Separating reasoning and searching into different agents improves performance.

**Limitation**: Improvements marginal with small LLMs (poor communication generation).

---

## 3. Learning & Adaptation

### 3.1 RLHF/RLAIF

**RLHF (Reinforcement Learning from Human Feedback)**
```
1. Collect human preferences on outputs
2. Train reward model from preferences
3. Fine-tune LLM with reward model
4. Iterate until alignment achieved
```

**RLAIF (RL from AI Feedback)**
```
1. Use LLM to generate preferences
2. Train reward model from AI preferences
3. Fine-tune with reward model
4. Potentially iterate with human verification
```

**Trade-offs:**
| Aspect | RLHF | RLAIF |
|--------|------|-------|
| Cost | High (human labor) | Lower |
| Quality | Gold standard | Good but may miss nuance |
| Scale | Limited | Highly scalable |

### 3.2 Agentic RL

**Core Idea**: RL for long-horizon, multi-turn agent interactions.

```
Traditional RL: Single-turn, short horizon
Agentic RL: Multi-turn, dynamic environment

Training Loop:
1. Agent interacts with environment
2. Collect trajectory (states, actions, rewards)
3. Update policy based on trajectory
4. Repeat with new scenarios
```

**Platforms:**
- NVIDIA NeMo Gym: Create realistic environments
- NeMo RL: Scale training efficiently

### 3.3 Lifelong Learning

**Core Idea**: Continual adaptation without forgetting.

```
Challenges:
├── Catastrophic Forgetting → Lose old knowledge
├── Task Interference → New learning hurts old
└── Knowledge Transfer → Apply learning broadly

Solutions:
├── Elastic Weight Consolidation
├── Progressive Neural Networks
├── Memory Replay
└── Regularization Techniques
```

### 3.4 CoDA Framework

**Core Idea**: Context-Decoupled Hierarchical Agent.

```
Problem: "Context Explosion" in agentic tasks

Solution:
┌──────────────────────────────────────────┐
│            Single LLM Backbone           │
├─────────────────┬────────────────────────┤
│    Planner      │       Executor         │
│ (High-level)    │    (Low-level)         │
└─────────────────┴────────────────────────┘

PECO: Planner-Executor Co-Optimization
- Joint end-to-end RL optimization
- Decoupled context handling
```

---

## 4. Verification & Cross-Validation

### 4.1 Critic Agent Pattern

**Core Idea**: Dedicated agent to assess and improve outputs.

```
Generator ──→ Output ──→ Critic ──→ Feedback
     ↑                                  │
     └──────────────────────────────────┘
```

**Implementation:**
```
Critic Prompt Template:
"Review this output for:
- Factual accuracy
- Logical consistency
- Task completion
- Potential issues

Provide specific feedback for improvement."
```

### 4.2 Cross-Validation

**Core Idea**: Multiple agents verify each other.

```
        ┌─────────┐
        │ Agent A │──→ Output A ──┐
        └─────────┘               │
        ┌─────────┐               ↓    ┌──────────┐
        │ Agent B │──→ Output B ──→────│ Consensus│──→ Final
        └─────────┘               ↑    │  Logic   │
        ┌─────────┐               │    └──────────┘
        │ Agent C │──→ Output C ──┘
        └─────────┘
```

**Results:**
- Up to 40% accuracy improvement
- Significant hallucination reduction
- More robust decisions

### 4.3 Adversarial Debate

**Core Idea**: Agents argue opposing positions to find truth.

```
Round 1: Agent A presents argument
         Agent B critiques, presents counter
Round 2: Agent A responds to critique
         Agent B responds
Round N: Continue until convergence or limit
Final:   Judge selects or synthesizes winner
```

**Key Components:**
| Component | Role |
|-----------|------|
| Pro Agent | Argues in favor of initial claim |
| Con Agent | Challenges claims, finds weaknesses |
| Judge Agent | Evaluates arguments, decides winner |

**Results:**
- 4-6% higher accuracy than single-agent
- 30% fewer factual errors
- More robust reasoning through adversarial pressure

### 4.4 Multi-Agent Collaborative Filtering (MCF)

**Core Idea**: Agents collaboratively filter and verify each other's outputs.

```
Agent A → Output → Agent B (Filter) → Agent C (Verify) → Final Output
                        ↓                    ↓
                   Feedback to A        Feedback to B
```

**Process:**
1. Generator produces initial output
2. Filter agent removes low-confidence claims
3. Verifier agent checks remaining claims against sources
4. Feedback loops improve subsequent generations

**Results:**
- 4-8% accuracy improvement over single-agent
- Iterative refinement catches subtle errors
- Works well with specialized filter/verify agents

### 4.5 ICE (Iterative Consensus Ensemble)

**Core Idea**: Multiple agents iterate until reaching consensus.

```
Round 1: Agent1, Agent2, Agent3 → [Output1, Output2, Output3]
         ↓
Compare outputs → Disagreement?
         ↓ Yes
Round 2: Share outputs, re-generate with context
         ↓
Repeat until consensus or max rounds (typically 3-5)
```

**Key Insight**: Ensemble size saturates at 3-5 agents (diminishing returns beyond).

**Results:**
- Up to 27% accuracy improvement
- Reduces overconfident errors
- Best for factual/structured tasks

### 4.6 Voting & Ensemble Methods

**Core Idea**: Simple but effective voting across multiple agents.

```
Query → [Agent1, Agent2, Agent3] → [Response1, Response2, Response3]
                    ↓
          Majority voting / Weighted average
                    ↓
              Final Response
```

**Voting Strategies:**
| Strategy | Description | Best For |
|----------|-------------|----------|
| Majority Vote | Most common answer wins | Classification |
| Weighted Vote | Weight by agent confidence | Mixed quality agents |
| Unanimous | All must agree | High-stakes decisions |
| Best-of-N | Select highest-scored | Quality ranking |

**Results:**
- 40% accuracy boost with 3-5 agents
- Simple to implement, robust
- Best for: Classification, factual questions, structured outputs

### 4.7 Layered Validation

```
Level 1: Unit Testing
    └── Individual agent components

Level 2: Integration Testing
    └── Agent interactions, tool chains

Level 3: System Testing
    └── Full workflow, end-to-end

Level 4: Production Monitoring
    └── Real-world performance
```

---

## 5. Pattern Selection Guide

### By Task Complexity

| Complexity | Recommended Patterns |
|------------|---------------------|
| Simple | Single agent, basic ReAct |
| Medium | Reflexion, Plan-and-Execute |
| Complex | Hierarchical DAG, GoalAct |
| Very Complex | Multi-agent with cross-validation |

### By Resource Constraints

| Constraint | Recommended Patterns |
|------------|---------------------|
| Token-limited | ReWOO (80% reduction) |
| Latency-critical | Pre-planned execution |
| Compute-limited | Memento (no fine-tuning) |
| Quality-critical | Multi-agent verification |

### Decision Tree

```
Is task long-horizon?
├── Yes → Plan-and-Execute or Hierarchical DAG
└── No → Is self-improvement needed?
    ├── Yes → Reflexion or Memento
    └── No → Is verification critical?
        ├── Yes → Cross-Validation or Debate
        └── No → Is cost a concern?
            ├── Yes → ReWOO
            └── No → Standard ReAct
```

---

## 6. Implementation Considerations

### Starting Simple

```
1. Start with basic ReAct
2. Add Reflexion if quality insufficient
3. Add Plan-and-Execute for complex tasks
4. Add verification for critical applications
5. Consider hierarchical only if needed
```

### Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Over-planning | Start with simpler patterns |
| Reflection loops | Set max iterations |
| Context explosion | Use CoDA or ReWOO |
| Verification overhead | Apply selectively |

### Hybrid Approaches

```
Production Pattern:
1. ReWOO for planning efficiency
2. Reflexion for quality improvement
3. Critic for final verification
4. Fallback to simpler pattern on failure
```

---

## Quick Reference

```
SELF-IMPROVEMENT:
  Reflexion    → Verbal RL, episodic memory
  Gödel Agent  → Self-modifying logic
  LADDER       → Progressive difficulty
  Memento      → Memory-based RL (87.88% GAIA)

PLANNING:
  Plan-and-Execute → Separate planning/execution
  LLMCompiler      → DAG scheduling, parallel
  Hierarchical DAG → Multi-layer decomposition
  GoalAct          → Continuous global planning
  ReWOO            → 80% token reduction
  BOLAA            → Specialized agent orchestration

LEARNING:
  RLHF/RLAIF       → Reward-based fine-tuning
  Agentic RL       → Long-horizon adaptation
  Lifelong         → Continual learning
  CoDA             → Context-decoupled RL

VERIFICATION & ACCURACY:
  Critic Agent     → Dedicated assessor
  Cross-Validation → Multi-agent checking (40% boost)
  Adversarial Debate → Pro/Con agents + Judge (30% fewer errors)
  MCF              → Collaborative filtering (4-8% improvement)
  ICE Consensus    → Iterative ensemble (27% improvement)
  Voting/Ensemble  → 3-5 agents optimal (40% accuracy boost)
  Layered          → Multi-level testing
```

---

## Related Documents

- [theoretical-foundations.md](theoretical-foundations.md) - Academic citations
- [framework-comparison.md](framework-comparison.md) - Framework analysis
- [evaluation-and-debugging.md](evaluation-and-debugging.md) - Testing and debugging
- [topics.md](topics.md) - Quick reference

---

**Document Version**: 1.0
**Last Updated**: December 2025
**Status**: Concepts and references (no production implementations)

**Sources**:
- [Reflexion Paper](https://arxiv.org/abs/2303.11366)
- [Gödel Agent](https://arxiv.org/abs/2410.04444)
- [Plan-and-Act](https://arxiv.org/abs/2503.09572)
- [Deep Agent](https://arxiv.org/abs/2502.07056)
- [Memento](https://arxiv.org/abs/2508.16153)
- [ReWOO](https://langchain-ai.github.io/langgraph/tutorials/rewoo/rewoo/)
- [BOLAA](https://arxiv.org/abs/2308.05960)
