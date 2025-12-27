# DSPy Guide: Programmatic Prompt Optimization

**From Manual Prompting to Systematic LLM Programming**

**Last Updated:** 2025-12-27

---

## Executive Summary

| Concept | Description |
|---------|-------------|
| **Philosophy** | Programming > Prompting: Define behavior declaratively, optimize automatically |
| **Core Abstractions** | Signatures (contracts), Modules (behavior), Optimizers (tuning) |
| **Key Benefit** | 10-40% improvement through systematic optimization vs manual prompts |
| **Best For** | RAG systems, multi-hop reasoning, agentic workflows, structured extraction |
| **DSPy 3.0 Features** | GEPA optimizer, RL training (Arbor), multi-modal support, MLflow integration |

---

## Table of Contents

1. [DSPy Philosophy](#1-dspy-philosophy)
2. [Core Concepts](#2-core-concepts)
3. [Basic Patterns](#3-basic-patterns)
4. [Optimization Strategies](#4-optimization-strategies)
5. [Integration with Agent Frameworks](#5-integration-with-agent-frameworks)
6. [Production Deployment](#6-production-deployment)
7. [Complete Code Examples](#7-complete-code-examples)
8. [Quick Reference](#8-quick-reference)

---

## 1. DSPy Philosophy

### 1.1 Programming vs Prompting Paradigm Shift

Traditional prompt engineering treats LLM interaction as an art form requiring endless manual iteration. DSPy fundamentally reimagines this approach by treating prompt generation as a **machine learning problem**.

```
Traditional Prompting:
  Write prompt → Test → Evaluate → Tweak prompt → Test again → Repeat...
  (Brittle, model-specific, hard to maintain)

DSPy Programming:
  Define signature → Compose modules → Provide metric → Compile → Deploy
  (Systematic, portable, measurable)
```

**The Core Insight**: Rather than asking "how should I phrase this instruction," DSPy asks "what behavior do I want" and uses optimization algorithms to discover effective prompts automatically.

### 1.2 Declarative Approach to LLM Programming

DSPy separates **interface specification** (what the LLM should do) from **implementation details** (how to ask it). This separation creates several advantages:

| Benefit | Description |
|---------|-------------|
| **Portability** | Same program works across different LLMs |
| **Maintainability** | Application logic is explicit Python code |
| **Measurability** | Metrics define success, not intuition |
| **Scalability** | Optimize once, deploy everywhere |

```python
# Traditional approach: Prompt string is both interface AND implementation
prompt = """You are an expert QA system. Given a question and context,
provide a detailed, accurate answer. Make sure to cite relevant passages.
Think step by step before answering..."""

# DSPy approach: Interface is declarative, implementation is compiled
class QASignature(dspy.Signature):
    """Answer questions based on provided context."""
    context: str = dspy.InputField(desc="Relevant passages")
    question: str = dspy.InputField(desc="User question")
    answer: str = dspy.OutputField(desc="Factual answer")
```

### 1.3 Why DSPy? Key Advantages

**Reproducibility**: Every optimization run is tracked with exact prompts, examples, and metrics.

**Automatic Optimization**: Algorithms discover effective prompts instead of manual guessing.

**Composability**: Build complex systems from simple, well-tested modules.

**Model Independence**: Switch LLMs by recompiling, not rewriting.

### 1.4 Comparison to Traditional Prompting

| Aspect | Traditional Prompting | DSPy |
|--------|----------------------|------|
| **Development** | Trial and error | Systematic optimization |
| **Testing** | Manual spot-checks | Metric-driven evaluation |
| **Debugging** | Read prompts | Inspect module traces |
| **Model changes** | Rewrite prompts | Recompile program |
| **Team scaling** | Knowledge siloed | Reproducible pipelines |
| **Maintenance** | Prompt drift | Version-controlled modules |

### 1.5 DSPy 3.0+ Features (December 2025)

| Feature | Description |
|---------|-------------|
| **GEPA Optimizer** | Genetic-Pareto prompt evolution, 35x more efficient than MIPROv2 |
| **Arbor RL** | Reinforcement learning optimization via GRPO |
| **Multi-Modal** | `dspy.Image` and `dspy.Audio` types for vision/audio models |
| **MLflow Integration** | Native experiment tracking and deployment |
| **Async Support** | High-throughput production via `acall()` |
| **dspy.Refine** | Runtime constraint enforcement with retry logic |
| **SIMBA Optimizer** | Self-reflective mini-batch improvement rules |

---

## 2. Core Concepts

### 2.1 Signatures: Input/Output Specifications

A **signature** declares the input/output contract for an LLM interaction. It specifies *what* should happen, not *how* to ask for it.

**Inline Signatures** (Simple tasks):
```python
import dspy

# Simple string format
qa = dspy.Predict("question -> answer")
sentiment = dspy.Predict("sentence -> sentiment: bool")
summarize = dspy.Predict("document -> summary")
```

**Class-Based Signatures** (Complex tasks):
```python
class RAGSignature(dspy.Signature):
    """Answer questions using retrieved context. Be factual and cite sources."""

    context: str = dspy.InputField(
        desc="Retrieved passages from knowledge base"
    )
    question: str = dspy.InputField(
        desc="User's question to answer"
    )
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning before answering"
    )
    answer: str = dspy.OutputField(
        desc="Final answer, 1-3 sentences, cite context"
    )
```

**Key Principle**: Signatures abstract away prompt engineering. DSPy's adapters automatically expand signatures into effective prompts for your chosen LLM.

### 2.2 Modules: Composable Units

A **module** encapsulates a prompting technique and works with any compatible signature. Modules are analogous to PyTorch's `nn.Module`.

**Built-in Modules**:

| Module | Purpose | When to Use |
|--------|---------|-------------|
| `dspy.Predict` | Basic prediction | Simple tasks |
| `dspy.ChainOfThought` | Step-by-step reasoning | Complex reasoning |
| `dspy.ProgramOfThought` | Generate + execute code | Math, logic |
| `dspy.ReAct` | Reason + Act with tools | Agentic tasks |
| `dspy.Retrieve` | Semantic search | RAG systems |
| `dspy.Refine` | Retry with feedback | Constraint enforcement |

**Module Composition**:
```python
class RAGModule(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        answer = self.generate(context=context, question=question)
        return answer
```

### 2.3 Optimizers (Teleprompters)

An **optimizer** tunes DSPy program parameters (prompts, examples, weights) to maximize a metric.

**Optimizer Categories**:

| Category | Optimizers | What They Tune |
|----------|-----------|----------------|
| **Few-Shot** | BootstrapFewShot, KNNFewShot | Demonstrations |
| **Instruction** | COPRO, MIPROv2 | Natural language instructions |
| **Joint** | MIPROv2 (full), GEPA | Both instructions + demos |
| **Finetuning** | BootstrapFinetune | Model weights |
| **Reflective** | SIMBA, GEPA | Self-improvement rules |

**The Optimization Process**:
```
1. Define metric: metric(example, prediction) -> score
2. Collect training data: List of (input, expected_output) pairs
3. Select optimizer: Based on data size and compute budget
4. Compile: optimizer.compile(program, trainset=data, metric=metric)
5. Evaluate: Test on held-out data
```

### 2.4 Assertions and Constraints

**Assertions** enforce runtime constraints on LLM outputs:

```python
class ConstrainedQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("context, question -> answer")

    def forward(self, context, question):
        answer = self.qa(context=context, question=question)

        # Hard constraint: Answer must be present in context
        dspy.Assert(
            answer.answer.lower() in context.lower(),
            "Answer must be grounded in context"
        )

        # Soft constraint: Prefer concise answers
        dspy.Suggest(
            len(answer.answer.split()) < 50,
            "Keep answers concise"
        )

        return answer
```

**Assert vs Suggest**:
- `dspy.Assert`: Hard constraint, fails if violated
- `dspy.Suggest`: Soft constraint, logged but continues

### 2.5 Metrics for Evaluation

Metrics define what "success" means for your task:

```python
# Binary metric: exact match
def exact_match(example, prediction):
    return example.answer.lower() == prediction.answer.lower()

# Continuous metric: semantic similarity
def semantic_f1(example, prediction):
    return dspy.evaluate.SemanticF1()(example.answer, prediction.answer)

# Multi-dimensional metric
def comprehensive_metric(example, prediction, trace=None):
    correct = exact_match(example, prediction)
    grounded = prediction.answer.lower() in example.context.lower()
    concise = len(prediction.answer.split()) < 50

    # Combine dimensions
    return (correct * 0.6 + grounded * 0.3 + concise * 0.1)
```

**Metric Best Practices**:
- Return `bool`, `float`, or `dict` with `score` key
- Accept optional `trace` parameter for intermediate validation
- Capture multiple quality dimensions
- Test metric on known good/bad examples first

---

## 3. Basic Patterns

### 3.1 ChainOfThought Module

**Purpose**: Encourage step-by-step reasoning before final answer.

```python
import dspy

# Configure LLM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Simple CoT
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="What is 15% of 80?")
print(result.reasoning)  # Shows step-by-step work
print(result.answer)     # "12"

# CoT with custom signature
class MathSignature(dspy.Signature):
    """Solve math word problems step by step."""
    problem: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Show all work")
    answer: float = dspy.OutputField(desc="Numeric answer only")

math_solver = dspy.ChainOfThought(MathSignature)
```

### 3.2 ReAct in DSPy

**Purpose**: Iterative reasoning with tool calls for agentic tasks.

```python
import dspy

# Define tools
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation here
    return f"Search results for: {query}"

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# Create tool objects
tools = [
    dspy.Tool(search_web),
    dspy.Tool(calculate),
]

# Configure ReAct agent
class ResearchSignature(dspy.Signature):
    """Research a topic and provide comprehensive answer."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

react_agent = dspy.ReAct(
    signature=ResearchSignature,
    tools=tools,
    max_iters=5
)

# Run agent
result = react_agent(question="What is the population of Tokyo times 2?")
```

**ReAct Loop**:
```
1. Observe: Receive question + current state
2. Think: Reason about what information is needed
3. Act: Call tool OR provide final answer
4. Repeat until done or max_iters reached
```

### 3.3 Multi-Hop Reasoning

**Purpose**: Complex queries requiring multiple retrieval/reasoning steps.

```python
class MultiHopRAG(dspy.Module):
    """Multi-hop QA with iterative retrieval."""

    def __init__(self, num_hops=3):
        super().__init__()
        self.num_hops = num_hops
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_query = dspy.ChainOfThought(
            "context, question -> search_query"
        )
        self.generate_answer = dspy.ChainOfThought(
            "context, question -> answer"
        )

    def forward(self, question):
        context = ""

        for hop in range(self.num_hops):
            # Generate search query based on current context
            if hop == 0:
                query = question
            else:
                query = self.generate_query(
                    context=context,
                    question=question
                ).search_query

            # Retrieve new passages
            passages = self.retrieve(query).passages
            context += "\n".join(passages) + "\n"

        # Generate final answer
        answer = self.generate_answer(
            context=context,
            question=question
        )
        return answer
```

### 3.4 Few-Shot Learning

**Purpose**: Learn from examples to improve performance.

```python
import dspy
from dspy.datasets import HotPotQA

# Load dataset
dataset = HotPotQA(train_seed=1, eval_seed=2023)
trainset = [x.with_inputs('question') for x in dataset.train[:50]]
devset = [x.with_inputs('question') for x in dataset.dev[:50]]

# Define program
qa = dspy.ChainOfThought("question -> answer")

# Define metric
def validate_answer(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# Optimize with few-shot learning
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=validate_answer,
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)

compiled_qa = optimizer.compile(qa, trainset=trainset)

# Evaluate
from dspy.evaluate import Evaluate
evaluator = Evaluate(devset=devset, metric=validate_answer, num_threads=4)
score = evaluator(compiled_qa)
print(f"Accuracy: {score}%")
```

### 3.5 Retrieval-Augmented Generation

**Purpose**: Ground LLM responses in retrieved knowledge.

```python
import dspy

# Configure retriever (example with ColBERT)
colbertv2 = dspy.ColBERTv2(url='http://localhost:8893/api/search')
dspy.configure(rm=colbertv2)

class RAGPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.ChainOfThought(
            "context, question -> answer"
        )

    def forward(self, question):
        # Retrieve relevant passages
        retrieved = self.retrieve(question)
        context = "\n\n".join(retrieved.passages)

        # Generate grounded answer
        response = self.generate(
            context=context,
            question=question
        )
        return dspy.Prediction(
            context=context,
            answer=response.answer
        )

# Usage
rag = RAGPipeline()
result = rag(question="What causes climate change?")
```

---

## 4. Optimization Strategies

### 4.1 BootstrapFewShot

**Best for**: Small datasets (10-50 examples), quick optimization.

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=my_metric,
    max_bootstrapped_demos=4,  # Generated demonstrations
    max_labeled_demos=4,       # From training data
    max_rounds=3              # Bootstrap iterations
)

compiled = optimizer.compile(
    student=my_program,
    trainset=trainset
)
```

**How It Works**:
1. Run program on training examples
2. Collect successful traces as demonstrations
3. Include best demonstrations in prompts

### 4.2 MIPROv2 (Multi-Stage Optimization)

**Best for**: Larger datasets (200+ examples), comprehensive optimization.

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=my_metric,
    auto="medium",  # "light", "medium", or "heavy"
    num_candidates=10,
    init_temperature=1.0
)

compiled = optimizer.compile(
    student=my_program,
    trainset=trainset,
    valset=valset,  # Separate validation set
    max_bootstrapped_demos=3,
    max_labeled_demos=3
)
```

**MIPROv2 Stages**:
1. **Bootstrap**: Generate candidate demonstrations
2. **Propose**: LLM-generated instruction candidates
3. **Bayesian Search**: TPE-guided combination search

**Auto Settings**:
| Setting | Trials | Best For |
|---------|--------|----------|
| `light` | ~10 | Quick experiments |
| `medium` | ~25 | Production systems |
| `heavy` | ~50+ | Critical applications |

### 4.3 GEPA (Genetic-Pareto Optimizer)

**Best for**: Complex tasks, multi-objective optimization, December 2025+.

```python
from dspy.teleprompt import GEPA

optimizer = GEPA(
    metric=my_metric,
    # GEPA accepts textual feedback for reflective optimization
    feedback_fn=lambda pred, gold: explain_errors(pred, gold)
)

compiled = optimizer.compile(
    student=my_program,
    trainset=trainset
)
```

**GEPA Advantages**:
- 35x more efficient than MIPROv2
- Generates 9x shorter prompts
- 10% better performance
- Uses genetic algorithms + Pareto optimization
- Accepts qualitative feedback, not just scores

### 4.4 Evaluation-Driven Optimization

**The Core Loop**:
```python
import dspy
from dspy.evaluate import Evaluate

# 1. Define comprehensive metric
def multi_objective_metric(example, pred, trace=None):
    scores = {
        "accuracy": exact_match(example, pred),
        "groundedness": is_grounded(pred, example.context),
        "conciseness": 1.0 if len(pred.answer.split()) < 50 else 0.5
    }
    # Weight and combine
    return 0.6 * scores["accuracy"] + 0.3 * scores["groundedness"] + 0.1 * scores["conciseness"]

# 2. Create evaluator
evaluator = Evaluate(
    devset=devset,
    metric=multi_objective_metric,
    num_threads=8,
    display_progress=True,
    display_table=5  # Show 5 examples
)

# 3. Baseline evaluation
baseline_score = evaluator(uncompiled_program)
print(f"Baseline: {baseline_score:.1f}%")

# 4. Optimize
compiled_program = optimizer.compile(program, trainset=trainset)

# 5. Post-optimization evaluation
optimized_score = evaluator(compiled_program)
print(f"Optimized: {optimized_score:.1f}%")
print(f"Improvement: {optimized_score - baseline_score:.1f}%")
```

### 4.5 Cost-Quality Trade-offs

**Optimization Budget Recommendations**:

| Data Size | Compute Budget | Recommended Optimizer |
|-----------|---------------|----------------------|
| <10 examples | Low | LabeledFewShot |
| 10-50 examples | Low | BootstrapFewShot |
| 50-200 examples | Medium | BootstrapFewShotWithRandomSearch |
| 200+ examples | Medium | MIPROv2 (auto="light") |
| 200+ examples | High | MIPROv2 (auto="heavy") or GEPA |
| Finetuning desired | Very High | BootstrapFinetune |

**Data Split Strategy** (Different from deep learning):
```python
# DSPy recommendation: More validation than training
train_size = int(0.2 * len(data))  # 20% for training
val_size = int(0.6 * len(data))    # 60% for validation
test_size = int(0.2 * len(data))   # 20% for test
```

---

## 5. Integration with Agent Frameworks

### 5.1 DSPy + LangGraph

**Architecture**: LangGraph handles orchestration, DSPy optimizes reasoning.

```python
import dspy
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

# DSPy modules for each node
class AnalyzeIntent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(
            "query -> intent: str, entities: list[str]"
        )

    def forward(self, query):
        return self.classify(query=query)

class GenerateResponse(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.ChainOfThought(
            "intent, entities, context -> response"
        )

    def forward(self, intent, entities, context):
        return self.respond(intent=intent, entities=entities, context=context)

# LangGraph state
class AgentState(TypedDict):
    query: str
    intent: str
    entities: list[str]
    context: str
    response: str

# Create optimized DSPy modules
analyze_intent = AnalyzeIntent()
generate_response = GenerateResponse()

# LangGraph nodes using DSPy
def analyze_node(state: AgentState) -> AgentState:
    result = analyze_intent(state["query"])
    return {
        "intent": result.intent,
        "entities": result.entities
    }

def respond_node(state: AgentState) -> AgentState:
    result = generate_response(
        state["intent"],
        state["entities"],
        state["context"]
    )
    return {"response": result.response}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("analyze", analyze_node)
workflow.add_node("respond", respond_node)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "respond")
workflow.add_edge("respond", END)

app = workflow.compile()
```

### 5.2 DSPy + CrewAI

**Architecture**: CrewAI handles team coordination, DSPy optimizes agent prompts.

```python
import dspy
from crewai import Agent, Task, Crew

# DSPy-optimized reasoning modules
class ResearchModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.research = dspy.ChainOfThought(
            "topic -> findings: str, sources: list[str]"
        )

    def forward(self, topic):
        return self.research(topic=topic)

class WritingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.write = dspy.ChainOfThought(
            "findings, style, audience -> content"
        )

    def forward(self, findings, style, audience):
        return self.write(findings=findings, style=style, audience=audience)

# Compile optimized modules
research_module = optimizer.compile(ResearchModule(), trainset=research_data)
writing_module = optimizer.compile(WritingModule(), trainset=writing_data)

# CrewAI agents using optimized modules
def create_optimized_crew():
    researcher = Agent(
        role="Research Specialist",
        goal="Find comprehensive information",
        backstory="Expert researcher with deep domain knowledge",
        # Use DSPy module in agent's tools
        tools=[research_module]
    )

    writer = Agent(
        role="Content Writer",
        goal="Create engaging content",
        backstory="Experienced writer for technical audiences",
        tools=[writing_module]
    )

    return Crew(agents=[researcher, writer], tasks=[...])
```

### 5.3 Custom Agent Patterns

**Intent-Based Orchestration with DSPy**:

```python
import dspy
from enum import Enum

class Intent(Enum):
    SEARCH = "search"
    RETRIEVE = "retrieve"
    ANALYZE = "analyze"
    UNKNOWN = "unknown"

class IntentRouter(dspy.Module):
    """Route queries to specialized handlers."""

    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(
            "query -> intent: str, confidence: float"
        )
        self.handlers = {
            Intent.SEARCH: SearchHandler(),
            Intent.RETRIEVE: RetrieveHandler(),
            Intent.ANALYZE: AnalyzeHandler(),
        }

    def forward(self, query):
        # Classify intent
        classification = self.classify(query=query)
        intent = Intent(classification.intent)

        # Route to appropriate handler
        if intent in self.handlers:
            return self.handlers[intent](query)
        else:
            return dspy.Prediction(
                response="I'm not sure how to handle that request."
            )

class SearchHandler(dspy.Module):
    def __init__(self):
        super().__init__()
        self.search = dspy.Retrieve(k=5)
        self.synthesize = dspy.ChainOfThought(
            "query, results -> response"
        )

    def forward(self, query):
        results = self.search(query)
        return self.synthesize(query=query, results=results.passages)
```

### 5.4 Hybrid Approaches

**Best Practice**: Combine framework strengths strategically.

```
Layer 1: LangGraph
  - Complex workflow orchestration
  - State management
  - Checkpointing

Layer 2: CrewAI
  - Team-based reasoning
  - Role specialization
  - Collaborative filtering

Layer 3: DSPy
  - Prompt optimization
  - Few-shot learning
  - Metric-driven improvement
```

**When to Use Each**:

| Requirement | Use |
|-------------|-----|
| Complex state machines | LangGraph |
| Multi-agent collaboration | CrewAI |
| Prompt optimization | DSPy |
| GUI/browser automation | LAM patterns |
| Simple tool calling | OpenAI SDK |

---

## 6. Production Deployment

### 6.1 Saving and Loading Programs

**State-Only Saving** (Recommended):
```python
# Save compiled program
compiled_program.save("./models/qa_v1.json")

# Load into new program instance
new_program = QAModule()
new_program.load("./models/qa_v1.json")
```

**Full Program Saving**:
```python
# Save entire program with architecture
compiled_program.save(
    "./models/qa_v1/",
    save_program=True
)

# Load complete program
loaded = dspy.load("./models/qa_v1/")
```

### 6.2 Version Management

```python
import dspy
from datetime import datetime
import json

class PromptVersionManager:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.manifest_path = f"{base_path}/manifest.json"

    def save_version(self, program, version: str, metrics: dict):
        """Save program version with metadata."""
        version_path = f"{self.base_path}/{version}"
        program.save(version_path)

        # Update manifest
        manifest = self._load_manifest()
        manifest["versions"][version] = {
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "path": version_path
        }
        manifest["current"] = version
        self._save_manifest(manifest)

    def load_version(self, version: str = None):
        """Load specific version or current."""
        manifest = self._load_manifest()
        version = version or manifest["current"]
        path = manifest["versions"][version]["path"]
        return dspy.load(path)

    def rollback(self, version: str):
        """Rollback to previous version."""
        manifest = self._load_manifest()
        manifest["current"] = version
        self._save_manifest(manifest)
```

### 6.3 A/B Testing Optimized Prompts

```python
import random
from dataclasses import dataclass

@dataclass
class ExperimentResult:
    variant: str
    score: float
    latency_ms: float

class PromptExperiment:
    def __init__(self, variants: dict):
        """
        variants: {"control": program_a, "treatment": program_b}
        """
        self.variants = variants
        self.results = []

    def run(self, query: str, allocation: dict = None):
        """Run experiment with traffic allocation."""
        allocation = allocation or {"control": 0.5, "treatment": 0.5}

        # Select variant
        r = random.random()
        cumulative = 0
        selected = None
        for variant, weight in allocation.items():
            cumulative += weight
            if r < cumulative:
                selected = variant
                break

        # Execute and measure
        import time
        start = time.time()
        result = self.variants[selected](query)
        latency = (time.time() - start) * 1000

        return result, selected, latency

    def analyze(self):
        """Statistical analysis of results."""
        from scipy import stats

        control = [r for r in self.results if r.variant == "control"]
        treatment = [r for r in self.results if r.variant == "treatment"]

        control_scores = [r.score for r in control]
        treatment_scores = [r.score for r in treatment]

        t_stat, p_value = stats.ttest_ind(control_scores, treatment_scores)

        return {
            "control_mean": sum(control_scores) / len(control_scores),
            "treatment_mean": sum(treatment_scores) / len(treatment_scores),
            "p_value": p_value,
            "significant": p_value < 0.05
        }
```

### 6.4 Monitoring Prompt Performance

```python
import dspy
import mlflow

# Enable MLflow autologging
mlflow.dspy.autolog(
    log_traces=True,
    log_traces_from_compile=True,
    log_traces_from_eval=True
)

class MonitoredProgram:
    def __init__(self, program, metrics_collector):
        self.program = program
        self.metrics = metrics_collector

    def __call__(self, **kwargs):
        import time

        start = time.time()
        try:
            result = self.program(**kwargs)
            latency = time.time() - start

            self.metrics.record({
                "latency_ms": latency * 1000,
                "success": True,
                "tokens": self._estimate_tokens(result)
            })

            return result
        except Exception as e:
            self.metrics.record({
                "success": False,
                "error": str(e)
            })
            raise
```

### 6.5 Caching Strategies

```python
import dspy
from dspy.clients import Cache

# Default: In-memory + on-disk caching enabled
# Automatic caching of LLM responses

# Custom cache for cross-model sharing
class ContentBasedCache(Cache):
    """Cache based on message content, ignoring model."""

    def _key(self, request):
        # Only use message content for key
        messages = request.get("messages", [])
        content = "".join(m.get("content", "") for m in messages)
        return hash(content)

# Configure custom cache
dspy.configure(
    lm=dspy.LM("openai/gpt-4o"),
    cache=ContentBasedCache()
)

# Disable caching for specific calls
with dspy.settings.context(cache=False):
    result = program(query="...")
```

### 6.6 FastAPI Deployment

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import dspy

app = FastAPI()

# Load compiled program
program = dspy.load("./models/production_v1/")

class QueryRequest(BaseModel):
    question: str
    context: str = ""

class QueryResponse(BaseModel):
    answer: str
    reasoning: str = None

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        # Use async for high throughput
        async_program = dspy.asyncify(program)
        result = await async_program(
            question=request.question,
            context=request.context
        )
        return QueryResponse(
            answer=result.answer,
            reasoning=getattr(result, 'reasoning', None)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

---

## 7. Complete Code Examples

### 7.1 Basic DSPy Signature and Module

```python
"""
Basic DSPy example: Question answering with chain-of-thought.
"""
import dspy

# Step 1: Configure LLM
lm = dspy.LM("openai/gpt-4o-mini", temperature=0.7)
dspy.configure(lm=lm)

# Step 2: Define signature
class QASignature(dspy.Signature):
    """Answer questions accurately and concisely."""
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="A clear, accurate answer")

# Step 3: Create module
qa_module = dspy.Predict(QASignature)

# Step 4: Use module
result = qa_module(question="What is the capital of France?")
print(f"Answer: {result.answer}")

# Step 5: Chain of thought variant
cot_module = dspy.ChainOfThought(QASignature)
result = cot_module(question="What is 15% of 80?")
print(f"Reasoning: {result.reasoning}")
print(f"Answer: {result.answer}")
```

### 7.2 ChainOfThought Implementation

```python
"""
Chain-of-thought reasoning for complex problems.
"""
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o"))

class MathProblemSignature(dspy.Signature):
    """Solve math word problems step by step."""
    problem: str = dspy.InputField(desc="Math word problem")
    reasoning: str = dspy.OutputField(desc="Step-by-step solution")
    answer: str = dspy.OutputField(desc="Final numeric answer")

class MathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought(MathProblemSignature)

    def forward(self, problem):
        result = self.solve(problem=problem)

        # Validate answer is numeric
        try:
            float(result.answer.replace(",", ""))
        except ValueError:
            # Retry with stricter instruction
            result = self.solve(
                problem=problem + " (Answer must be a number only)"
            )

        return result

# Usage
solver = MathSolver()
result = solver(
    problem="A store sells apples for $2 each. If someone buys 5 apples "
            "and pays with a $20 bill, how much change do they receive?"
)
print(f"Reasoning:\n{result.reasoning}")
print(f"Answer: ${result.answer}")
```

### 7.3 RAG with DSPy

```python
"""
Complete RAG pipeline with DSPy optimization.
"""
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM

# Configure
lm = dspy.LM("openai/gpt-4o")
retriever = ChromadbRM(
    collection_name="documents",
    persist_directory="./chroma_db",
    k=5
)
dspy.configure(lm=lm, rm=retriever)

class RAGSignature(dspy.Signature):
    """Answer questions using provided context. Cite sources."""
    context: str = dspy.InputField(desc="Retrieved passages")
    question: str = dspy.InputField(desc="User question")
    reasoning: str = dspy.OutputField(desc="How context supports answer")
    answer: str = dspy.OutputField(desc="Factual answer with citations")

class RAGPipeline(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought(RAGSignature)

    def forward(self, question):
        # Retrieve relevant passages
        retrieved = self.retrieve(question)
        context = "\n\n".join([
            f"[{i+1}] {p}" for i, p in enumerate(retrieved.passages)
        ])

        # Generate grounded answer
        response = self.generate(
            context=context,
            question=question
        )

        return dspy.Prediction(
            context=context,
            reasoning=response.reasoning,
            answer=response.answer,
            passages=retrieved.passages
        )

# Create and test
rag = RAGPipeline()

# Define metric for optimization
def rag_metric(example, pred, trace=None):
    # Check answer correctness
    answer_correct = example.answer.lower() in pred.answer.lower()

    # Check if answer is grounded in context
    grounded = any(
        phrase in pred.context.lower()
        for phrase in pred.answer.lower().split(".")[:2]
    )

    return 0.7 * answer_correct + 0.3 * grounded

# Optimize
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=rag_metric, max_bootstrapped_demos=4)
compiled_rag = optimizer.compile(rag, trainset=train_examples)

# Save for production
compiled_rag.save("./models/rag_v1.json")
```

### 7.4 Optimization Pipeline

```python
"""
Complete optimization pipeline with evaluation.
"""
import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate
from datasets import load_dataset

# Step 1: Load and prepare data
dataset = load_dataset("hotpot_qa", "fullwiki")

def prepare_example(item):
    return dspy.Example(
        question=item["question"],
        answer=item["answer"]
    ).with_inputs("question")

# DSPy recommends: 20% train, 60% val, 20% test
all_data = [prepare_example(x) for x in dataset["train"][:500]]
trainset = all_data[:100]      # 20%
valset = all_data[100:400]     # 60%
testset = all_data[400:]       # 20%

# Step 2: Define program
class MultiHopQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(
            "question, context -> search_query"
        )
        self.retrieve = dspy.Retrieve(k=3)
        self.answer = dspy.ChainOfThought(
            "question, context -> answer"
        )

    def forward(self, question):
        # First hop
        context = ""
        passages = self.retrieve(question).passages
        context += "\n".join(passages)

        # Second hop (refined query)
        refined = self.generate_query(question=question, context=context)
        more_passages = self.retrieve(refined.search_query).passages
        context += "\n" + "\n".join(more_passages)

        # Answer
        result = self.answer(question=question, context=context)
        return result

# Step 3: Define metrics
def exact_match(example, pred, trace=None):
    return example.answer.lower().strip() == pred.answer.lower().strip()

def fuzzy_match(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# Step 4: Baseline evaluation
program = MultiHopQA()
evaluator = Evaluate(devset=valset, metric=fuzzy_match, num_threads=8)
baseline = evaluator(program)
print(f"Baseline accuracy: {baseline:.1f}%")

# Step 5: Optimize with MIPROv2
optimizer = MIPROv2(
    metric=fuzzy_match,
    auto="medium",
    num_candidates=7,
    init_temperature=1.0
)

compiled = optimizer.compile(
    student=program,
    trainset=trainset,
    valset=valset,
    max_bootstrapped_demos=3,
    max_labeled_demos=3
)

# Step 6: Post-optimization evaluation
optimized = evaluator(compiled)
print(f"Optimized accuracy: {optimized:.1f}%")
print(f"Improvement: {optimized - baseline:.1f}%")

# Step 7: Final test evaluation
test_evaluator = Evaluate(devset=testset, metric=fuzzy_match, num_threads=8)
final_score = test_evaluator(compiled)
print(f"Test set accuracy: {final_score:.1f}%")

# Step 8: Save production model
compiled.save("./models/multihop_qa_v1.json")
```

### 7.5 Agent Integration Example

```python
"""
DSPy-powered agent with tool use and optimization.
"""
import dspy
from typing import List

# Configure
dspy.configure(lm=dspy.LM("openai/gpt-4o"))

# Define tools
def web_search(query: str) -> str:
    """Search the web for current information."""
    # Implementation would call actual search API
    return f"Search results for '{query}': [simulated results]"

def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: 72F, sunny"

# Create tool objects
tools = [
    dspy.Tool(web_search, name="search", desc="Search the web"),
    dspy.Tool(calculator, name="calculate", desc="Do math"),
    dspy.Tool(get_weather, name="weather", desc="Get weather"),
]

# Define agent signature
class AgentSignature(dspy.Signature):
    """You are a helpful assistant that can search, calculate, and check weather."""
    user_request: str = dspy.InputField(desc="User's request")
    response: str = dspy.OutputField(desc="Helpful response to user")

# Create ReAct agent
class OptimizableAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.agent = dspy.ReAct(
            signature=AgentSignature,
            tools=tools,
            max_iters=5
        )

    def forward(self, user_request):
        return self.agent(user_request=user_request)

# Create training data
train_examples = [
    dspy.Example(
        user_request="What is 15% of 200?",
        response="15% of 200 is 30."
    ).with_inputs("user_request"),
    dspy.Example(
        user_request="What's the weather in Seattle?",
        response="The weather in Seattle is 72F and sunny."
    ).with_inputs("user_request"),
    # Add more examples...
]

# Define metric
def agent_metric(example, pred, trace=None):
    # Check if core information is present
    expected_keywords = example.response.lower().split()
    pred_lower = pred.response.lower()

    matches = sum(1 for kw in expected_keywords if kw in pred_lower)
    return matches / len(expected_keywords)

# Optimize agent
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=agent_metric,
    max_bootstrapped_demos=3
)

agent = OptimizableAgent()
compiled_agent = optimizer.compile(agent, trainset=train_examples)

# Test optimized agent
result = compiled_agent(user_request="Calculate 25% tip on $80 dinner")
print(f"Response: {result.response}")

# Save for production
compiled_agent.save("./models/agent_v1.json")
```

---

## 8. Quick Reference

### Module Cheatsheet

```python
import dspy

# Basic prediction
predict = dspy.Predict("question -> answer")

# Chain of thought
cot = dspy.ChainOfThought("question -> answer")

# Program of thought (code generation)
pot = dspy.ProgramOfThought("question -> answer")

# ReAct agent
react = dspy.ReAct(signature, tools=tools, max_iters=5)

# Retrieval
retrieve = dspy.Retrieve(k=5)

# Refinement with retries
refine = dspy.Refine(module, max_retries=3)
```

### Optimizer Selection

```
Data < 10:     LabeledFewShot
Data 10-50:    BootstrapFewShot
Data 50-200:   BootstrapFewShotWithRandomSearch
Data 200+:     MIPROv2 (auto="light" to "heavy")
Multi-obj:     GEPA (reflective feedback)
Fine-tune:     BootstrapFinetune
Zero-shot:     COPRO or MIPROv2 (demos=0)
```

### Common Patterns

```python
# Pattern 1: RAG
class RAG(dspy.Module):
    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# Pattern 2: Multi-hop
class MultiHop(dspy.Module):
    def forward(self, question):
        for hop in range(self.num_hops):
            query = self.refine_query(question, context)
            context += self.retrieve(query).passages
        return self.answer(question=question, context=context)

# Pattern 3: Ensemble
class Ensemble(dspy.Module):
    def forward(self, question):
        answers = [m(question).answer for m in self.models]
        return self.aggregate(answers)
```

### Data Split Recommendations

```python
# DSPy recommends (different from deep learning!)
train = 20%   # For generating demonstrations
val = 60%     # For optimization evaluation
test = 20%    # For final evaluation

# Minimum viable:
# 30 train, 200+ val, 50+ test
```

### Production Checklist

```
[ ] Compiled program saved (JSON preferred)
[ ] Version control for prompts
[ ] Metrics defined and logged
[ ] A/B testing infrastructure
[ ] Rollback procedure documented
[ ] Monitoring for latency/cost/quality
[ ] Caching strategy implemented
[ ] Error handling for LLM failures
```

---

## Related Documents

- [theoretical-foundations.md](../phase-1-foundations/theoretical-foundations.md) - ReAct, CoT academic foundations
- [advanced-agent-paradigms.md](advanced-agent-paradigms.md) - Self-improvement patterns
- [evaluation-and-debugging.md](../phase-4-production/evaluation-and-debugging.md) - Testing strategies
- [framework-comparison.md](../phase-1-foundations/framework-comparison.md) - When to use DSPy vs alternatives

---

## Sources and Further Reading

**Official Resources**:
- [DSPy Documentation](https://dspy.ai) - Official docs and tutorials
- [DSPy GitHub](https://github.com/stanfordnlp/dspy) - Source code and issues
- [DSPy Cheatsheet](https://dspy.ai/cheatsheet/) - Quick reference

**Research Papers**:
- [DSPy: Compiling Declarative Language Model Calls](https://arxiv.org/abs/2310.03714) - Original paper
- [MIPRO Optimizer](https://dspy.ai/api/optimizers/MIPROv2/) - Multi-stage optimization
- [GEPA: Genetic-Pareto Prompt Optimization](https://dspy.ai/api/optimizers/GEPA/) - December 2025

**Community Resources**:
- [DSPy Discord](https://discord.gg/dspy) - Community support
- [Context7 DSPy Docs](https://context7.com/stanfordnlp/dspy) - Additional examples

---

**Document Version**: 1.0
**Last Updated**: December 2025
**Status**: Production-ready guide with December 2025 features
