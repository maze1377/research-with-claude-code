# Advanced Agent Paradigms

**Self-improvement, advanced planning, and learning techniques for LLM agents**

**Last Updated:** 2025-12-25

---

## December 2025 Key Advances

| Innovation | Key Finding | Source |
|------------|-------------|--------|
| **Multi-Agent Reflexion** | 82.6% HumanEval (vs 76.4% single-agent) | arXiv:2512.20845 |
| **Darwin Gödel Machine** | 20%→50% SWE-bench through self-evolution | arXiv:2505.22954 |
| **RLVR Dominance** | Foundation of o3, DeepSeek-R1, replaced RLHF | Karpathy Review 2025 |
| **AlphaEvolve Production** | 0.7% global compute saved, 23% Gemini speedup | DeepMind |
| **LADDER Scaling** | 1%→82% on integration (3B model) | arXiv:2503.00735 |
| **Test-Time Limits** | Extended reasoning increases hallucinations | arXiv:2509.06861 |
| **Confucius Code Agent** | 54.3% SWE-Bench-Pro (scaffolding > model size) | arXiv:2512.10398 |

---

## Executive Summary

| Paradigm | Key Innovation | Performance Gain |
|----------|----------------|------------------|
| Multi-Agent Reflexion | Persona-based debate critics | 82.6% HumanEval (+6.2% over single) |
| Plan-and-Execute | Separate planner/executor | 57.58% WebArena success |
| ReWOO | Pre-planned tool chains | 80% token reduction |
| Memento | Memory-based RL | 87.88% GAIA (top-1) |
| Deep Agent | Hierarchical Task DAG | Complex scenario handling |
| GoalAct | Continuous global planning | 12.22% improvement |
| Darwin Gödel Machine | Self-evolving code | 20%→50% SWE-bench |
| RLVR | Verifiable reward training | o3, DeepSeek-R1 foundation |
| CodeTree | Agent-guided tree search | 95.1% HumanEval SOTA |

---

## 1. Self-Improvement Paradigms

### 1.1 Reflexion and Multi-Agent Reflexion (MAR)

**Core Idea**: Agents learn from verbal self-reflection without fine-tuning.

```
Single-Agent Reflexion:
┌─────────────────────────────────────────────────────┐
│   Actor ──→ Environment ──→ Evaluator              │
│     ↑                           │                  │
│     └────── Self-Reflection ◄───┘                  │
│                   ↓                                │
│            Episodic Memory                         │
└─────────────────────────────────────────────────────┘

Multi-Agent Reflexion (MAR) - December 2025:
┌─────────────────────────────────────────────────────┐
│                     Actor                           │
│                       ↓                             │
│              Failed Attempt                         │
│    ┌──────────────┼──────────────┐                 │
│    ↓              ↓              ↓                 │
│ ┌────────┐  ┌────────┐  ┌────────┐                 │
│ │Critic A│  │Critic B│  │Critic C│  ← Persona-based│
│ │(Logic) │  │(Domain)│  │(Code)  │                 │
│ └────────┘  └────────┘  └────────┘                 │
│    └──────────────┼──────────────┘                 │
│                   ↓                                │
│         Debate Coordinator                          │
│                   ↓                                │
│         Consensus Reflection                        │
│                   ↓                                │
│         Actor (with enriched memory)                │
└─────────────────────────────────────────────────────┘
```

**Components:**
| Component | Role | Implementation |
|-----------|------|----------------|
| Actor | Generates actions | CoT or ReAct agent |
| Evaluator | Scores outcomes | LLM or heuristic |
| Self-Reflection | Critiques attempts | LLM-generated feedback |
| Memory | Stores reflections | Episodic buffer |
| **Debate Critics** (MAR) | Multiple perspectives | Persona-based agents |
| **Coordinator** (MAR) | Aggregates debate | Consensus synthesis |

**Key Finding**: All types of self-reflection improve performance (p < 0.001).

**MAR vs Single-Agent Results (December 2025):**
| Benchmark | ReAct | Single Reflexion | MAR |
|-----------|-------|------------------|-----|
| HotPotQA | 32% | 44% | **47%** |
| HumanEval | 67.1% | 76.4% | **82.6%** |

**Why MAR Works:**
- Single-agent reflexion suffers from "degeneration of thought"
- Same model repeats flawed reasoning even after explicit failures
- Multi-agent debate surfaces alternative interpretations
- Persona diversity helps escape mental-set failures

**Advantages:**
- No fine-tuning required
- Lightweight and efficient
- Works with any base LLM
- Interpretable improvements
- MAR reduces confirmation bias by 20%+

### 1.2 Gödel Agent and Darwin Gödel Machine

**Core Idea**: Self-modifying agents inspired by the Gödel machine.

```
Traditional Agent:
  Fixed logic ──→ Execute ──→ Output

Gödel Agent:
  Logic ──→ Self-Evaluation ──→ Modify Logic ──→ Execute
    ↑                                              │
    └──────────────────────────────────────────────┘

Darwin Gödel Machine (DGM) - December 2025:
  Archive of Agents
       ↓
  Sample Agent → Foundation Model → New Variants
       ↓                                ↓
  Tree of Diverse, High-Quality Agents ←┘
       ↓
  Parallel Exploration of Search Space
       ↓
  Self-Discovered: Better tools, context management, peer review
```

**Key Properties:**
- Recursively improves own logic
- No predefined optimization routines
- Guided by high-level objectives
- Dynamic behavior modification

**Darwin Gödel Machine Results (December 2025):**
| Benchmark | Before DGM | After DGM |
|-----------|------------|-----------|
| SWE-bench | 20.0% | **50.0%** |
| Polyglot | 14.2% | **30.7%** |

**Self-Discovered Improvements:**
- Better code editing tools
- Long-context window management strategies
- Peer-review mechanisms for code quality

**Caution**: Raises alignment concerns as agents modify themselves.

### 1.3 LADDER Framework (December 2025 Updates)

**Core Idea**: Recursive generation of progressively simpler problem variants with RLVR.

```
Complex Problem
      ↓
Generate Simpler Variant Trees ← Multiple variants per level
      ↓
Solve Simpler Variants
      ↓
Verify with Numerical Integration ← Automatic verification
      ↓
Reinforce Learning (RLVR)
      ↓
Apply to Original Problem
      ↓
Repeat Until Solved
```

**Key Components:**
| Component | Function |
|-----------|----------|
| Variant Generation | Structured tree of progressively simpler problems |
| Solution Verification | Numerical integration for mathematical correctness |
| RL Protocol | Train on variant trees with verified rewards |

**Results (December 2025):**
| Model | Task | Before | After |
|-------|------|--------|-------|
| Llama 3.2 3B | Integration | 1% | **82%** |
| Llama 3.2 7B | MIT Integration Bee | - | **73%** |
| GPT-4o | MIT Integration Bee | - | 42% |
| Human average | MIT Integration Bee | - | 15-30% |

**Key Insight:** 7B model beats GPT-4o and most humans on math competition.

**Why It Works:**
- No human intervention required
- Creates natural difficulty gradient
- Uses verifiable rewards (not human judgment)
- Bootstraps from model's existing capabilities

### 1.4 AlphaEvolve (December 2025 Production)

**Core Idea**: Evolutionary optimization of algorithms by LLM.

```
Process:
1. Start with initial algorithm + metrics
2. LLM proposes mutations/combinations
3. Evaluate new candidates with automated evaluators
4. Select best performers
5. Repeat evolution indefinitely
6. Deploy human-readable solutions
```

**Production Deployments at Google (December 2025):**
| Application | Impact |
|-------------|--------|
| **Borg Scheduling** | 0.7% worldwide compute recovered (in production 1+ year) |
| **Gemini Matrix Mult** | 23% kernel acceleration, 1% training time reduction |
| **TPU Circuit Design** | Discovered novel heuristics |

**Key Innovation**:
- Can optimize components of itself
- Produces human-readable, auditable code
- Continuous improvement pipeline

**Limitation**: Requires automated evaluation functions.

**Critical Insight:** AlphaEvolve finds solutions that offer both strong performance AND significant operational advantages—human engineers can understand and modify the code.

### 1.5 Memento/AgentFly (December 2025)

**Core Idea**: Memory-based RL without fine-tuning—continual learning as memory-based online RL.

```
Traditional RL: Update model weights (expensive)
Memento: Update memory contents (efficient)

Architecture:
┌─────────────────────────────────────────────────────┐
│                 Single LLM Backbone                  │
├───────────────────┬─────────────────────────────────┤
│   Meta-Planner    │         Executor                │
│  (Break queries   │   (Execute subtasks via         │
│   into subtasks)  │    MCP tools)                   │
├───────────────────┴─────────────────────────────────┤
│                   Case Memory                        │
│  Store: (state, action, reward) final-step tuples   │
│  Retrieve: By value similarity to guide planning    │
├─────────────────────────────────────────────────────┤
│               MCP Tool Layer                         │
│  (Unified interface for external tools/services)    │
└─────────────────────────────────────────────────────┘
```

**Results (December 2025):**
| Benchmark | Metric | Score |
|-----------|--------|-------|
| GAIA Validation | Pass@3 | **87.88%** (Top-1) |
| GAIA Test | - | **79.40%** |
| DeepResearcher | F1 | 66.6% |
| DeepResearcher | Performance | 80.4% |
| OOD Tasks | Improvement | +4.7% to +9.6% |

**Key Mechanism:**
- Neural case-selection policy retrieves relevant past experiences
- Stores final-step tuples (when solution became valid)
- Online RL updates selection policy based on success/failure
- No labeled training data needed for case relevance

**Advantages:**
- No gradient updates needed
- Continual adaptation in real-time
- Low computational cost
- Superior OOD generalization vs parameter-based learning

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

### 3.1 RLHF/RLAIF/RLVR Paradigms

**Evolution of Training Paradigms:**
```
2022-2024: RLHF Dominance
  Pretraining → SFT → RLHF (human preferences)

2025: RLVR Revolution
  Pretraining → SFT → RLVR (verifiable rewards)

Key Shift: Most capability progress now from extended RL, not pretraining
```

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

**RLVR (RL from Verifiable Rewards) - December 2025 DOMINANT:**
```
1. Define objective, automatically verifiable rewards
   - Math: Correct/incorrect (symbolic or numerical)
   - Code: Tests pass/fail
2. Train directly against verifiable signal
3. No reward model needed → no reward hacking
4. Models spontaneously develop reasoning behaviors

Key Models Using RLVR:
- OpenAI o3, o4-mini
- DeepSeek-R1
- QwQ, Llama-Reasoning
```

**Trade-offs:**
| Aspect | RLHF | RLAIF | RLVR |
|--------|------|-------|------|
| Cost | High (human labor) | Lower | Lowest |
| Quality | Gold standard | May miss nuance | Domain-limited |
| Scale | Limited | Scalable | Highly scalable |
| Domains | General | General | Verifiable only |
| Reward Hacking | Possible | Possible | Minimal |

### 3.1a GRPO and DAPO Training Optimizations

**Group Relative Policy Optimization (GRPO):**
- Simpler variant of PPO for RLVR training
- Used by DeepSeek-R1 for pure RL reasoning training
- No need for SFT intermediate step

**GRPO Limitations Discovered (December 2025):**
| Issue | Problem |
|-------|---------|
| Length Bias | Longer incorrect answers overweighted |
| Question Weighting | Very easy/hard questions overweighted |
| Training Instability | Normalizations introduce bias |

**Dynamic Asymmetric Policy Optimization (DAPO):**
```
Fixes to GRPO:
- Asymmetric clipping for stable learning
- Dynamic sampling adjustment
- Token-level losses (not per-sample)
- Truncated importance sampling
```

**Dr. GRPO Alternative:**
- Removes length and std dev normalizations
- Returns to unbiased PPO-like objective
- Different organizations use different approaches

### 3.2 Agentic RL and Test-Time Scaling

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

**Platforms (December 2025):**
| Platform | Purpose |
|----------|---------|
| **NVIDIA NeMo Gym** | Build RL training environments for LLMs |
| **NeMo RL** | Scale training efficiently |

**NeMo Gym Architecture:**
```
┌─────────────────────────────────────────────────────┐
│                    Agents                            │
│  (Orchestrate rollout lifecycle, call models,       │
│   execute tools through Resources, coordinate       │
│   verification)                                      │
├─────────────────────────────────────────────────────┤
│                    Models                            │
│  (Stateless text generation via LLM endpoints)      │
├─────────────────────────────────────────────────────┤
│                   Resources                          │
│  (Task definitions, tool implementations,           │
│   verification logic)                               │
└─────────────────────────────────────────────────────┘
```

**Test-Time Scaling Discovery:**
- Performance scales with reasoning compute at inference
- Similar to pretraining scaling laws but for thinking time
- o3 used 10× reasoning compute vs o1 → substantial gains

**CRITICAL: Test-Time Scaling Limitations (December 2025):**
| Domain | Effect of Extended Reasoning |
|--------|------------------------------|
| Math/Code | ✅ Performance improves |
| Knowledge-Intensive | ❌ Increases hallucinations |
| Factual Recall | ❌ Confirmation bias increases |

**Why It Fails for Knowledge Tasks:**
1. Extended reasoning encourages attempts on previously unanswered questions
2. Many attempts result in hallucinations
3. Extended reasoning induces confirmation bias
4. Leads to overconfident hallucinations

**Implication:** Different approaches needed for domains where factual accuracy is essential.

### 3.2a OREO: Offline Reasoning Optimization

**Core Idea**: Multi-step reasoning improvement without online environment interaction.

**Problem with DPO for Reasoning:**
- DPO relies on paired preference data
- Treats all tokens uniformly
- Ineffective for credit assignment in sparse reward tasks

**OREO Solution:**
```
Based on Maximum Entropy RL:
1. Jointly learn policy model AND value function
2. Optimize soft Bellman equation
3. Better credit assignment across reasoning steps
4. Value function guides tree search at inference
```

**Results:**
| Benchmark | Domain | OREO Performance |
|-----------|--------|------------------|
| GSM8K | Math Reasoning | Outperforms offline methods |
| MATH | Advanced Math | Outperforms offline methods |
| ALFWorld | Embodied Agent | Outperforms offline methods |

**Key Advantage:** Learned value function enables test-time tree search without additional training.

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

**Ensemble Results (December 2025):**
- 30%+ F1 improvement with 2-3 models
- Error counterbalancing: different models' errors partially cancel
- Best for: Classification, factual questions, structured outputs

### 4.7 Multi-Agent Collusion Risks (NEW - December 2025)

**Critical Finding**: Multi-agent systems can be exploited through adversarial collusion.

```
Normal Multi-Agent:
Agent1 → Opinion1 ─┐
Agent2 → Opinion2 ─┼→ Consensus → Decision
Agent3 → Opinion3 ─┘

Adversarial Collusion Attack:
Adversary1 → Wrong1 ─┐
Adversary2 → Wrong2 ─┼→ FALSE Consensus → Unsafe Decision
Adversary3 → Wrong3 ─┘
```

**Research Finding (Healthcare AI Study):**
| Condition | Attack Success Rate |
|-----------|---------------------|
| Isolated wrong suggestions | Low |
| Coordinated wrong consensus | **Significantly higher** |

**The Paradox:**
- Multi-agent systems were designed for safety through diversity
- But consensus pressure makes them vulnerable to collusion
- More agents ≠ more safety if agents can be compromised

**Defense: Verifier Agent Pattern:**
```
Multi-Agent Opinions
        ↓
Consensus-Biased Decision
        ↓
Verifier Agent ← Trusted Guideline Knowledge
        ↓
If verified: Execute
If not: Block + Alert
```

**Result:** Verifier agent completely blocked collusion attacks while preserving multi-agent benefits.

### 4.8 Layered Validation

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

## 5. Emerging Paradigms (December 2025)

### 5.1 CodeTree: Agent-Guided Tree Search

**Core Idea**: Explicit tree structures for exploring diverse coding paths.

```
Task Input
    ↓
┌─────────────────────────────────────────────────────┐
│              Unified Tree Structure                  │
│  ┌───────┐                                          │
│  │Strategy│─┬─→ Strategy A ─→ Impl A1, A2, A3      │
│  │  Root  │ │                                       │
│  └───────┘ ├─→ Strategy B ─→ Impl B1, B2           │
│            │                                        │
│            └─→ Strategy C ─→ Impl C1               │
│                                                     │
│  Each node: Execution feedback + LLM feedback       │
│  Ranking, termination, expansion guided by both     │
└─────────────────────────────────────────────────────┘
    ↓
Best Solution Selected
```

**Results (December 2025, GPT-4o):**
| Benchmark | CodeTree Score |
|-----------|----------------|
| HumanEval | **95.1%** (SOTA) |
| MBPP | **98.7%** |
| CodeContests | **43.0%** |
| SWE-Bench | **31.9%** |

**Key Innovation:**
- Parallel exploration of diverse coding strategies
- Both environmental (test) and LLM-generated feedback
- Explicit ranking guides search termination

### 5.2 Confucius Code Agent: Scaffolding > Model Size

**Critical Finding**: Agent architecture can outweigh base model capability.

```
Experiment Setup:
┌────────────────────────────────────────────────────┐
│                                                     │
│  Weaker Model + Strong Scaffold  vs  Stronger Model│
│  (Claude 4.5 Sonnet + CCA)      vs  (Claude 4.5 Opus)│
│           52.7%                 vs       52.0%      │
│                                                     │
│  Result: Scaffold wins despite weaker base model!   │
└────────────────────────────────────────────────────┘
```

**SWE-Bench-Pro Results:**
| Configuration | Resolution Rate |
|---------------|-----------------|
| CCA Framework | **54.3%** (exceeds prior baselines) |
| Claude 4.5 Sonnet + CCA | 52.7% |
| Claude 4.5 Opus (other scaffold) | 52.0% |

**What CCA Provides:**
- Enhanced orchestration of subtasks
- Sophisticated context management
- Hierarchical working memory
- Tool-use extensions

**Implication:** Invest in scaffolding design, not just model scaling.

### 5.3 EnCompass: Programmable Inference-Time Search

**Core Idea**: Separate workflow logic from inference-time search strategies.

```
Traditional Agent:
  Fixed workflow → Fixed inference strategy

EnCompass:
  Workflow Program (Python decorator) → Compiled to Search Space
                                              ↓
                          ┌─────────────────────────────────┐
                          │ Choose Inference Strategy:       │
                          │ - Monte Carlo Tree Search        │
                          │ - Beam Search                    │
                          │ - Custom strategies              │
                          │ - Change by input parameter      │
                          └─────────────────────────────────┘
```

**Key Concept: Probabilistic Angelic Nondeterminism**
- Programmers describe agent workflows
- Independently experiment with inference strategies
- Switch strategies by changing parameters, not code

**Results (December 2025):**
| Strategy | Accuracy Improvement | Search Budget |
|----------|---------------------|---------------|
| Two-level beam search | **15-40%** | 16× standard LLM calls |

**Use Case:** Rapid experimentation to find optimal inference-time strategies without reimplementing agents.

### 5.4 OR-LLM-Agent: Operations Research Specialists

**Core Idea**: Decomposed agents for complex optimization problems.

```
Operations Research Task
        ↓
┌─────────────────────────────────────────────────────┐
│  Sub-Agent 1: Mathematical Modeling                 │
│  (Formalize constraints, objective function)        │
├─────────────────────────────────────────────────────┤
│  Sub-Agent 2: Code Generation                       │
│  (Generate solver code)                             │
├─────────────────────────────────────────────────────┤
│  Sub-Agent 3: Debugging                             │
│  (Fix errors, validate solutions)                   │
└─────────────────────────────────────────────────────┘
        ↓
Optimized Solution
```

**Results:**
- Outperforms GPT-o3, Gemini 2.5 Pro, specialized OR solvers by 7%+
- Found that NL4OPT dataset contains significant incorrect answers

---

## 6. Pattern Selection Guide

### By Task Complexity

| Complexity | Recommended Patterns |
|------------|---------------------|
| Simple | Single agent, basic ReAct |
| Medium | Reflexion, Plan-and-Execute |
| Complex | Hierarchical DAG, GoalAct |
| Very Complex | Multi-agent with cross-validation |
| **Code Generation** | CodeTree (95.1% HumanEval) |
| **Self-Evolution** | Darwin Gödel Machine |

### By Resource Constraints

| Constraint | Recommended Patterns |
|------------|---------------------|
| Token-limited | ReWOO (80% reduction) |
| Latency-critical | Pre-planned execution |
| Compute-limited | Memento (no fine-tuning) |
| Quality-critical | Multi-agent verification + Verifier |
| **Offline Training** | OREO (no environment needed) |

### Decision Tree (December 2025)

```
Is task long-horizon?
├── Yes → Plan-and-Execute or Hierarchical DAG
└── No → Is self-improvement needed?
    ├── Yes → Is it code/algorithm?
    │   ├── Yes → Darwin Gödel Machine or AlphaEvolve
    │   └── No → Multi-Agent Reflexion or Memento
    └── No → Is verification critical?
        ├── Yes → Cross-Validation + Verifier Agent (block collusion)
        └── No → Is cost a concern?
            ├── Yes → ReWOO or OREO
            └── No → Is scaffolding available?
                ├── Yes → Use strong scaffold (CCA pattern)
                └── No → Standard ReAct + CodeTree for code
```

---

## 7. Implementation Considerations

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
SELF-IMPROVEMENT (December 2025):
  Multi-Agent Reflexion → Persona debate (82.6% HumanEval, +6.2%)
  Darwin Gödel Machine  → Self-evolving code (20%→50% SWE-bench)
  LADDER               → Progressive difficulty (1%→82% with 3B model)
  AlphaEvolve          → Production evolution (0.7% global compute saved)
  Memento              → Memory-based RL (87.88% GAIA top-1)

PLANNING:
  Plan-and-Execute → Separate planning/execution
  LLMCompiler      → DAG scheduling, parallel
  Hierarchical DAG → Multi-layer decomposition
  GoalAct          → Continuous global planning (12.22% improvement)
  ReWOO            → 80% token reduction
  BOLAA            → Specialized agent orchestration

LEARNING (December 2025):
  RLVR             → Dominant paradigm (o3, DeepSeek-R1)
  GRPO/DAPO        → Training optimizations for RLVR
  OREO             → Offline multi-step reasoning
  Agentic RL       → Long-horizon adaptation (NeMo Gym)
  Test-Time Limits → Extended reasoning hurts knowledge tasks

VERIFICATION & ACCURACY:
  Critic Agent       → Dedicated assessor
  Cross-Validation   → Multi-agent checking (30%+ improvement)
  Adversarial Debate → Pro/Con + Judge (30% fewer errors)
  Ensemble Methods   → 2-3 models (30%+ F1 improvement)
  Collusion Defense  → Verifier agent pattern (blocks attacks)

EMERGING PARADIGMS (December 2025):
  CodeTree         → Agent-guided tree search (95.1% HumanEval SOTA)
  CCA Framework    → Scaffolding > model size (54.3% SWE-Bench-Pro)
  EnCompass        → Programmable inference search (15-40% improvement)
  OR-LLM-Agent     → Operations research specialists (+7% vs o3)
```

---

## Related Documents

- [theoretical-foundations.md](theoretical-foundations.md) - Academic citations
- [framework-comparison.md](framework-comparison.md) - Framework analysis
- [evaluation-and-debugging.md](evaluation-and-debugging.md) - Testing and debugging
- [topics.md](topics.md) - Quick reference

---

**Document Version**: 2.0
**Last Updated**: December 2025
**Status**: Validated research with production examples

**Sources**:
- [Multi-Agent Reflexion (MAR)](https://arxiv.org/abs/2512.20845) - December 2025
- [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) - 50% SWE-bench
- [LADDER Framework](https://arxiv.org/abs/2503.00735) - Self-learning
- [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) - Production at Google
- [Gödel Agent](https://aclanthology.org/2025.acl-long.1354/) - ACL 2025
- [Memento](https://arxiv.org/abs/2508.16153) - 87.88% GAIA
- [CodeTree](https://aclanthology.org/2025.naacl-long.189/) - 95.1% HumanEval
- [Confucius Code Agent](https://arxiv.org/abs/2512.10398) - Scaffolding research
- [EnCompass](https://neurips.cc/virtual/2025/poster/118817) - NeurIPS 2025
- [OREO](https://aclanthology.org/2025.findings-acl.464/) - Offline reasoning
- [RLVR Analysis](https://karpathy.bearblog.dev/year-in-review-2025/) - Karpathy 2025
- [Test-Time Scaling Limits](https://arxiv.org/abs/2509.06861) - Hallucination research
- [Multi-Agent Collusion](https://arxiv.org/abs/2512.03097) - Safety research
- [NeMo Gym](https://docs.nvidia.com/nemo/gym/latest/index.html) - NVIDIA RL infrastructure
- [ReWOO](https://www.ibm.com/think/topics/rewoo) - IBM documentation
- [GoalAct](https://arxiv.org/abs/2504.16563) - Hierarchical planning
