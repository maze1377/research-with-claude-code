# LangGraph Multi-Agent Patterns: Complete Guide (2025)

**Based on**: Production case studies (LinkedIn, Uber, Replit, Elastic, Anthropic, Zapier), LangChain benchmarks, academic research (arXiv 2025), official documentation, and real-world lessons learned

**Last Updated**: 2025-12-25

---

## Key 2025 Research Findings

| Finding | Source | Impact |
|---------|--------|--------|
| Architecture-task alignment > team size | arXiv:2512.08296 | Don't scale agents; align architecture |
| 79% of failures: specification + coordination | getmaxim.ai | Focus on prompts, not infrastructure |
| Swarm outperforms Supervisor (token efficiency) | LangChain | Use Swarm when no central control needed |
| Context management bottleneck | Google ADK | Separate Session from Working Context |
| 90.2% improvement with multi-agent (breadth) | Anthropic | Multi-agent excels at parallel search |
| Theory-of-mind prompts enable emergence | arXiv:2510.05174 | Add "think about what others might do" |

---

## Table of Contents

1. [Overview](#overview)
2. [When to Use Multi-Agent vs Single Agent](#when-to-use-multi-agent-vs-single-agent)
3. [Three Core Multi-Agent Architectures](#three-core-multi-agent-architectures)
4. [Agent Communication Patterns](#agent-communication-patterns)
5. [Talkshow & Debate Patterns](#talkshow--debate-patterns)
6. [Production Case Studies](#production-case-studies)
7. [Best Practices](#best-practices)
8. [Implementation Patterns](#implementation-patterns)
9. [Common Pitfalls](#common-pitfalls)

---

## Overview

### What is a Multi-Agent System?

> "Multiple independent actors powered by language models connected in a specific way"

Each agent in a LangGraph multi-agent system maintains:
- **Own prompt**: Specialized instructions and context
- **Own LLM instance**: Can use different models per agent
- **Own tools**: Domain-specific capabilities
- **Own state/scratchpad**: Isolated or shared workspace

### Graph Representation

- **Agents = Nodes**: Each agent is a computation node
- **Connections = Edges**: Define communication paths
- **Control Flow**: Managed by edges (who goes next)
- **Communication**: Via shared state updates

---

## Workflows vs Agents: Fundamental Distinction

**Reference:** arXiv:2404.11584 "Landscape of Emerging AI Agent Architectures"

Before diving into multi-agent patterns, it's crucial to understand the fundamental distinction between **workflows** and **agents**—two paradigms that exist on a spectrum of determinism vs autonomy.

### Core Definitions

| Aspect | Workflows | Agents |
|--------|-----------|--------|
| **Definition** | Deterministic systems with predefined, fixed sequences | Autonomous systems with dynamic planning and adaptation |
| **Control** | Developer sets the sequence | LLM decides the sequence |
| **Path** | Fixed, unidirectional | Dynamic, looping, adaptive |
| **Predictability** | High (same input → same path) | Low (same input → varying paths) |
| **Flexibility** | Low (can't handle edge cases) | High (adapts to novel scenarios) |
| **Analogy** | Assembly line | Autonomous employee |

### The Autonomy Spectrum

```
DETERMINISTIC ◄─────────────────────────────────────► AUTONOMOUS

┌─────────────────────────────────────────────────────────────────┐
│ WORKFLOWS              AGENT LOOPS           TRUE AGENTS       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│ │ Fixed Steps │     │ Fixed Loop  │     │ Dynamic     │        │
│ │ A → B → C   │     │ but LLM     │     │ Planning &  │        │
│ │             │     │ decides HOW │     │ Adaptation  │        │
│ └─────────────┘     └─────────────┘     └─────────────┘        │
│                                                                 │
│ Examples:            Examples:            Examples:             │
│ - RAG pipelines      - ReAct loops        - Coding agents       │
│ - ETL processes      - Tool-calling       - Research agents     │
│ - Approval chains    - Chat + tools       - Auto-debugging      │
│                                                                 │
│ Predictable ●        Balanced ●           Flexible ●            │
│ Cheap ●              Moderate ●           Expensive ●           │
│ Debuggable ●         Traceable ●          Complex ●             │
└─────────────────────────────────────────────────────────────────┘
```

### When to Use Each

#### Use Workflows When:
- ✅ Task has well-defined, repeatable steps
- ✅ Predictability and auditability required (compliance, finance)
- ✅ High volume with low variance
- ✅ Cost sensitivity is high
- ✅ Long-running processes (days/weeks) with external events
- ✅ Debugging and traceability are critical

#### Use Agents When:
- ✅ Task requires adaptation to unknown conditions
- ✅ Edge cases are common and unpredictable
- ✅ Complex reasoning or multi-step problem solving needed
- ✅ Goals are open-ended or exploratory
- ✅ Self-correction capabilities needed
- ✅ Flexibility more important than cost

#### Use Hybrid (Recommended for Production):
- ✅ Workflow orchestrates overall process
- ✅ Agent nodes handle specific complex steps
- ✅ Deterministic routing with agentic reasoning
- ✅ Best of both: predictability + flexibility

### Five Planning Approaches for Agents

From arXiv:2404.11584, agents use these planning strategies:

| Approach | Description | Best For |
|----------|-------------|----------|
| **Classical Planning** | Symbolic planners (PDDL) for deterministic action sequences | Structured domains, known state spaces |
| **LLM-as-a-Judge** | LLM evaluates intermediate outputs for quality | Self-critique loops, iterative improvement |
| **Monte Carlo Tree Search** | Simulates action branches probabilistically | Complex search spaces, game-like decisions |
| **ReAct** | Interleaves reasoning traces with acting (observe-think-act) | Tool use, grounded planning |
| **Multi-Agent Debate** | Agents propose, critique, refine plans via discussion | Reducing hallucinations, consensus building |

### Practical Implementation: LangGraph Hybrid

**Workflow with ReAct Agent Node:**
```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from typing import TypedDict

class State(TypedDict):
    messages: list
    risk_level: str
    analysis_result: str

# Deterministic workflow node
def parse_input(state: State) -> State:
    # Fixed logic: classify risk level
    content = state["messages"][-1]["content"]
    risk = "high" if "urgent" in content.lower() else "low"
    return {"risk_level": risk}

# Agent node for complex reasoning
def risk_analysis_agent(state: State) -> State:
    # Agent handles dynamic reasoning
    agent = create_react_agent(
        model=llm,
        tools=[search_tool, calculator_tool, policy_tool]
    )
    result = agent.invoke(state)
    return {"analysis_result": result["messages"][-1]["content"]}

# Deterministic routing
def route_by_risk(state: State) -> str:
    if state["risk_level"] == "high":
        return "agent_analysis"  # Complex: use agent
    return "simple_response"     # Simple: skip agent

# Build hybrid graph
graph = StateGraph(State)
graph.add_node("parse", parse_input)
graph.add_node("agent_analysis", risk_analysis_agent)
graph.add_node("simple_response", lambda s: {"analysis_result": "Standard processing complete"})

graph.set_entry_point("parse")
graph.add_conditional_edges("parse", route_by_risk)
graph.add_edge("agent_analysis", END)
graph.add_edge("simple_response", END)

app = graph.compile()
```

### Trade-off Analysis

| Factor | Workflows | Agents | Winner |
|--------|-----------|--------|--------|
| **Development Cost** | Lower (fixed paths) | Higher (complex testing) | Workflows |
| **Runtime Cost** | Lower (fewer LLM calls) | Higher (iterations) | Workflows |
| **Predictability** | High | Low | Workflows |
| **Flexibility** | Low | High | Agents |
| **Debugging** | Easy (traceable) | Hard (dynamic paths) | Workflows |
| **Edge Case Handling** | Poor | Good | Agents |
| **Scalability** | Excellent | Good | Workflows |
| **Long-running Tasks** | Excellent (state persistence) | Poor (drift risk) | Workflows |

### Failure Mode Comparison

| Failure Type | Workflow Behavior | Agent Behavior |
|--------------|-------------------|----------------|
| **Unexpected Input** | Fails predictably | May adapt or spiral |
| **Infinite Loops** | Not possible (fixed) | Risk without limits |
| **Hallucination** | Contained to step | Can compound |
| **Recovery** | Manual reprogramming | Self-correction possible |
| **Debugging** | Trace specific step | Trace entire reasoning chain |

### Decision Criteria: Escalate from Workflow to Agent

```
START with Workflow (Crawl Phase)
├── Task has fixed, known steps? → STAY Workflow
├── High volume, low variance? → STAY Workflow
├── Compliance/audit requirements? → STAY Workflow
│
├── Frequent edge cases? → ADD Agent nodes
├── Complex reasoning needed? → ADD Agent nodes
├── Need self-correction? → ADD Agent nodes
│
├── Open-ended goals? → FULL Agent
├── Unknown environments? → FULL Agent
└── Creativity required? → FULL Agent
```

### Key Insight

> "Start with workflows for reliability. Add agentic elements incrementally as trust in LLM decisions grows. The goal is not full autonomy—it's **appropriate autonomy** for each task."

**Practical Rule:** If you can draw a flowchart of the process, use a workflow. If the process requires "it depends" decisions at multiple points, use agents (or hybrid).

---

## When to Use Multi-Agent vs Single Agent

### Benchmarking Results (LangChain 2025)

**Key Finding**: Single agents suffer significant performance degradation with increased context size, even when context is irrelevant to the task.

#### Performance by Architecture

| Architecture | Token Efficiency | Performance (2+ Domains) | Maintainability |
|--------------|------------------|-------------------------|-----------------|
| Single Agent | ⚠️ Scales linearly with domains | ❌ Sharp decline | 🔧 Hard to update |
| Supervisor | ✅ Flat token usage | ✅ Good | ✅ Modular |
| Swarm | ✅ Flat token usage | ✅✅ Best | ✅✅ Most modular |

### Use Single Agent When:

✅ **Operating within single domain**
- Limited tool count (< 10 tools)
- Coherent context that fits in one window
- No parallelizable subtasks

✅ **Simple routing is sufficient**
- Clear if-then logic works
- No need for agent specialization

✅ **Only one distractor domain**
- Benchmarks show single agent slightly better with 1 distractor
- Performance drops sharply with 2+ distractors

### Use Multi-Agent When:

✅ **Multiple specialized domains**
- Research shows multi-agent excels with 2+ distinct domains
- Each agent becomes domain expert

✅ **Heavy parallelization needed**
- Independent research tasks
- Multiple data sources to query simultaneously
- Parallel analysis streams

✅ **Information exceeds single context window**
- Different agents can maintain separate context
- No single prompt can contain all necessary information

✅ **Numerous complex tools** (15+ tools)
- Tool confusion overwhelms single agent
- Specialized agents reduce tool selection errors

✅ **Engineering best practices**
- Modularity: Update agents independently
- Testability: Evaluate each agent separately
- Maintainability: Clear responsibility boundaries
- Scalability: Add new agents without refactoring

### When NOT to Use Multi-Agent:

❌ **Shared context requirement**
- All agents need identical information
- Context synchronization becomes bottleneck

❌ **Heavy inter-agent dependencies**
- Many sequential dependencies reduce parallelization benefits
- Example: Coding tasks (fewer truly parallel subtasks than research)

❌ **Simple, well-defined tasks**
- Overhead of multi-agent outweighs benefits
- Single agent with clear prompt is simpler

### Decision Framework

```python
def should_use_multi_agent(task):
    # Start with goals and constraints
    if task.tool_count < 10 and task.domains == 1:
        return "single_agent"

    if task.parallelizable_subtasks > 3:
        return "multi_agent_swarm"

    if task.domains >= 2 and task.tool_count > 15:
        return "multi_agent_supervisor"

    if task.requires_specialist_expertise:
        return "multi_agent_network"

    # Default: start simple
    return "single_agent"  # Can always migrate later
```

**Key Principle**: "Always start from your goals (definitions of success) and constraints when picking a design pattern."

---

## Three Core Multi-Agent Architectures

### 1. Multi-Agent Collaboration (Shared Scratchpad)

**Pattern**: All agents work on a shared message history, making all work visible to all agents.

```python
from langgraph.graph import StateGraph, MessagesState

# Shared state - all agents see all messages
class CollaborativeState(MessagesState):
    current_speaker: str
    conversation_round: int

def agent_a(state: CollaborativeState):
    # Agent A can see everything in state.messages
    response = llm_a.invoke([
        SystemMessage("You are Agent A, expert in topic X"),
        *state.messages  # Full message history
    ])
    return {"messages": [response]}

def agent_b(state: CollaborativeState):
    # Agent B also sees all messages
    response = llm_b.invoke([
        SystemMessage("You are Agent B, expert in topic Y"),
        *state.messages  # Same full history
    ])
    return {"messages": [response]}

# Build graph
graph = StateGraph(CollaborativeState)
graph.add_node("agent_a", agent_a)
graph.add_node("agent_b", agent_b)
graph.add_edge("agent_a", "agent_b")
graph.add_edge("agent_b", "agent_a")  # Can loop back
```

**Characteristics**:
- ✅ Full transparency: All agents see all work
- ✅ Natural for debates and discussions
- ✅ Good observability
- ⚠️ Context window grows quickly
- ⚠️ Agents see irrelevant information
- ⚠️ Prompt engineering crucial to avoid confusion

**Best For**:
- Debates and talkshows
- Collaborative brainstorming
- Peer review scenarios
- Small teams (2-4 agents)

---

### 2. Agent Supervisor (Hierarchical Control)

**Pattern**: Central supervisor coordinates specialized agents, each with isolated scratchpad. Only final responses go to global state.

```python
from langgraph_supervisor import create_supervisor

# Specialized agents with isolated context
research_agent = Agent(
    name="researcher",
    model="gpt-4o",
    tools=[web_search, arxiv_search],
    instructions="Research papers and web sources"
)

analyst_agent = Agent(
    name="analyst",
    model="claude-3-7-sonnet",
    tools=[data_analysis, visualization],
    instructions="Analyze data and create visualizations"
)

writer_agent = Agent(
    name="writer",
    model="gpt-4o",
    tools=[],
    instructions="Write clear, engaging reports"
)

# Supervisor coordinates everything
supervisor = create_supervisor(
    agents=[research_agent, analyst_agent, writer_agent],
    model="gpt-4o",
    instructions="""
    You coordinate a team of specialists:
    - researcher: For finding information
    - analyst: For analyzing data
    - writer: For creating final reports

    Delegate tasks appropriately and synthesize results.
    """
)

graph = supervisor.build_graph()
```

**Using langgraph-supervisor library** (2025):

```bash
pip install langgraph-supervisor
```

**Key Features**:
- **Handoff tools**: Prebuilt with `create_handoff_tool()`
- **Context filtering**: Sub-agents don't see routing logic
- **State isolation**: Each agent has own scratchpad
- **Only supervisor sees everything**

**Characteristics**:
- ✅ Clear hierarchy and responsibility
- ✅ Agents stay focused (no irrelevant context)
- ✅ Supervisor handles all routing logic
- ✅ Easier to reason about control flow
- ⚠️ Supervisor is single point of failure
- ⚠️ Can become bottleneck with many agents
- ⚠️ Supervisor quality critical to system success

**Best For**:
- Complex workflows with clear stages
- Task delegation scenarios
- Quality control (supervisor reviews work)
- Medium-sized teams (3-7 agents)

---

### 3. Agent Swarm/Network (Peer-to-Peer)

**Pattern**: Agents communicate directly with each other (many-to-many). Each agent can hand off to any other agent.

```python
from langgraph_swarm import create_swarm
from langgraph.types import Command

# Agents with handoff capabilities
def research_agent(state: MessagesState) -> Command:
    # Do research work
    result = perform_research(state.messages[-1])

    # Decide next agent
    if needs_analysis(result):
        return Command(
            goto="analyst_agent",
            update={"messages": [result]}
        )
    elif needs_writing(result):
        return Command(
            goto="writer_agent",
            update={"messages": [result]}
        )
    else:
        return Command(
            goto=END,
            update={"messages": [result]}
        )

def analyst_agent(state: MessagesState) -> Command:
    # Analyze data
    analysis = perform_analysis(state.messages)

    # Hand off to writer
    return Command(
        goto="writer_agent",
        update={"messages": [analysis]}
    )

def writer_agent(state: MessagesState) -> Command:
    # Write report
    report = write_report(state.messages)

    # Check if needs more research
    if needs_more_info(report):
        return Command(
            goto="research_agent",
            update={"messages": [report, "Need more info on: ..."]}
        )
    else:
        return Command(
            goto=END,
            update={"messages": [report]}
        )

# Build swarm
swarm = create_swarm(
    agents=[research_agent, analyst_agent, writer_agent],
    initial_agent="research_agent"
)
```

**Using langgraph-swarm library** (2025):

```bash
pip install langgraph-swarm
```

**Characteristics**:
- ✅ No single point of failure
- ✅ Maximum flexibility
- ✅ Agents self-organize
- ✅ Can parallelize dynamically
- ✅ Benchmarks show best performance
- ⚠️ Harder to debug (emergent behavior)
- ⚠️ Can loop infinitely if not careful
- ⚠️ Requires sophisticated termination logic

**Best For**:
- Dynamic, unpredictable workflows
- Peer collaboration (no clear hierarchy)
- Highly parallel tasks
- Exploration and creative tasks
- Large teams (5+ agents)

---

## 5-Axis Hierarchical MAS Taxonomy (arXiv:2508.12683)

**Reference:** arXiv:2508.12683 "A Taxonomy of Hierarchical Multi-Agent Systems"

This taxonomy provides a systematic framework for classifying hierarchical multi-agent systems (HMAS) across 5 orthogonal dimensions. It's essential for understanding where your multi-agent system falls on the design spectrum.

### The Five Axes

```
┌─────────────────────────────────────────────────────────────────────┐
│                    5-AXIS HMAS TAXONOMY                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  AXIS 1: CONTROL HIERARCHY                                          │
│  ├─ Centralized (single coordinator controls all)                  │
│  ├─ Decentralized (peer agents with local rules)                   │
│  └─ Federated/Hybrid (layered coordination)                        │
│                                                                     │
│  AXIS 2: INFORMATION FLOW                                           │
│  ├─ Top-Down (goals, specs, policies flow down)                    │
│  ├─ Bottom-Up (rewards, summaries, observations flow up)           │
│  └─ Bidirectional (with protocol constraints)                      │
│                                                                     │
│  AXIS 3: ROLE & TASK DELEGATION                                     │
│  ├─ Fixed Roles (pre-assigned, static duties)                      │
│  ├─ Emergent Roles (dynamic, runtime allocation)                   │
│  └─ Hybrid (mixed fixed + emergent)                                │
│                                                                     │
│  AXIS 4: COMMUNICATION STRUCTURE                                    │
│  ├─ Static Networks (fixed agent links)                            │
│  └─ Dynamic Networks (adaptive connections)                        │
│                                                                     │
│  AXIS 5: TEMPORAL LAYERING                                          │
│  ├─ Uniform (all agents same timescale)                            │
│  └─ Layered (different timescales: strategic/tactical/operational) │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Axis 1: Control Hierarchy

How decision-making authority is distributed:

| Type | Description | LLM Example | Trade-offs |
|------|-------------|-------------|------------|
| **Centralized** | Single coordinator dictates all policies | Supervisor pattern | Simple but bottleneck-prone |
| **Decentralized** | Autonomous agents with local rules | Swarm pattern | Scalable but coordination overhead |
| **Federated** | Layered coordination with local autonomy | Hierarchical teams | Balanced but complex |

**Decision Criterion:** Use centralized for simple orchestration, decentralized for swarms, federated for enterprise scale.

### Axis 2: Information Flow

How information moves through the hierarchy:

```
TOP-DOWN FLOW                    BOTTOM-UP FLOW
┌─────────────┐                  ┌─────────────┐
│ Coordinator │                  │ Coordinator │
│  (Goals,    │                  │  (Receives  │
│   Policies) │                  │   Summaries)│
└──────┬──────┘                  └──────▲──────┘
       │                                │
   ┌───┴───┐                        ┌───┴───┐
   ▼       ▼                        │       │
┌────┐  ┌────┐                   ┌────┐  ┌────┐
│ W1 │  │ W2 │                   │ W1 │  │ W2 │
│(Exec│  │(Exec│                  │(Report│ │(Report│
│ute) │  │ute) │                  │ s)   │  │ s)   │
└────┘  └────┘                   └────┘  └────┘

BIDIRECTIONAL FLOW
┌─────────────┐
│ Coordinator │◄────┐
│  (Goals ↓,  │     │
│   Summary ↑)│     │
└──────┬──────┘     │
       │            │
   ┌───┴───┐    Feedback
   ▼       ▼        │
┌────┐  ┌────┐      │
│ W1 │  │ W2 │──────┘
└────┘  └────┘
```

**LLM Implementation:**
```python
class HierarchicalState(TypedDict):
    # Top-down: Coordinator sets goals
    current_goal: str
    assigned_subtasks: list[str]

    # Bottom-up: Workers report results
    worker_summaries: dict[str, str]
    local_observations: list[str]

    # Bidirectional: State synced
    shared_context: str
```

### Axis 3: Role & Task Delegation

How roles are assigned to agents:

| Type | Description | When to Use |
|------|-------------|-------------|
| **Fixed Roles** | Pre-assigned at design time (researcher, analyst, writer) | Stable domains, well-defined expertise |
| **Emergent Roles** | Assigned at runtime based on context | Dynamic environments, unknown tasks |
| **Hybrid** | Core fixed roles + emergent specialists | Production systems (balance stability + flexibility) |

**Contract Net Protocol Example (Emergent):**
```python
def contract_net_delegation(task, available_agents):
    # Step 1: Announce task
    proposals = []
    for agent in available_agents:
        bid = agent.evaluate_task(task)
        if bid.can_handle:
            proposals.append((agent, bid))

    # Step 2: Select best bidder
    if proposals:
        winner = max(proposals, key=lambda x: x[1].confidence)
        return winner[0].execute(task)

    # Step 3: No suitable agent - escalate
    return escalate_to_coordinator(task)
```

### Axis 4: Communication Structure

Network topology between agents:

| Topology | Pattern | LLM Example | Characteristics |
|----------|---------|-------------|-----------------|
| **Star** | Hub-and-spoke | Supervisor pattern | Central control, easy routing |
| **Tree** | Hierarchical branching | Hierarchical teams | Scalable, clear escalation |
| **Mesh** | Full connectivity | Collaboration pattern | Flexible, high overhead |
| **Static** | Fixed connections | Predefined workflows | Predictable, limited |
| **Dynamic** | Runtime connections | Swarm pattern | Adaptive, complex |

```
STAR                 TREE                 MESH
    ┌───┐           ┌───┐           ┌───┐
    │ C │           │ M │           │ A │◄───────┐
    └─┬─┘           └─┬─┘           └─┬─┘        │
  ┌───┼───┐       ┌──┴──┐             │◄────────┐│
  ▼   ▼   ▼       ▼     ▼             ▼         ││
┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐       ┌───┐      ││
│ A │ │ B │ │ C │ │ C │ │ C │       │ B │◄─────┼┘
└───┘ └───┘ └───┘ └─┬─┘ └─┬─┘       └─┬─┘      │
                  ┌─┴─┐ ┌─┴─┐         │◄───────┘
                  ▼   ▼ ▼   ▼         ▼
                 W1  W2 W3  W4      ┌───┐
                                    │ C │
                                    └───┘
```

### Axis 5: Temporal Layering

Different agents operating at different timescales:

| Layer | Timescale | Responsibility | LLM Example |
|-------|-----------|----------------|-------------|
| **Strategic** | Days/weeks | Long-term planning, goal setting | Planning agent |
| **Tactical** | Hours/day | Task decomposition, coordination | Supervisor |
| **Operational** | Minutes | Real-time execution, tool use | Worker agents |

**Example: Enterprise Research System**
```
┌─────────────────────────────────────────────────────────────┐
│ STRATEGIC LAYER (Weekly)                                     │
│ "Set research priorities, allocate resources"                │
│                         │                                    │
│                         ▼                                    │
├─────────────────────────────────────────────────────────────┤
│ TACTICAL LAYER (Daily)                                       │
│ "Decompose research tasks, assign to specialists"           │
│                    ┌────┴────┐                               │
│                    ▼         ▼                               │
├─────────────────────────────────────────────────────────────┤
│ OPERATIONAL LAYER (Real-time)                                │
│ ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│ │ Search   │    │ Analysis │    │ Writing  │                │
│ │ Agent    │    │ Agent    │    │ Agent    │                │
│ └──────────┘    └──────────┘    └──────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Taxonomy Classification Matrix

Use this matrix to classify your multi-agent system:

| Your System | Axis 1 | Axis 2 | Axis 3 | Axis 4 | Axis 5 |
|-------------|--------|--------|--------|--------|--------|
| **Simple Supervisor** | Centralized | Top-Down | Fixed | Star | Uniform |
| **Hierarchical Teams** | Federated | Bidirectional | Hybrid | Tree | Layered |
| **Swarm** | Decentralized | Bottom-Up | Emergent | Dynamic Mesh | Uniform |
| **CrewAI Default** | Centralized | Bidirectional | Fixed | Star | Uniform |
| **AutoGen** | Decentralized | Bidirectional | Emergent | Dynamic | Uniform |
| **LangGraph Supervisor** | Centralized | Top-Down | Fixed | Star | Uniform |
| **LangGraph Swarm** | Decentralized | Bidirectional | Emergent | Dynamic | Uniform |

### When to Use Each Combination

**Centralized + Fixed + Static (Simple Supervisor):**
- Well-defined workflows
- Small teams (2-4 agents)
- Predictable tasks
- Easy debugging

**Federated + Hybrid + Dynamic (Enterprise Scale):**
- Complex organizations
- 10+ agents
- Mixed task types
- Needs both structure and flexibility

**Decentralized + Emergent + Dynamic (Swarm):**
- Highly dynamic environments
- Large agent populations
- Exploration tasks
- Resilience critical

### Trade-offs Summary

| Dimension | More Centralized/Fixed/Static | More Decentralized/Emergent/Dynamic |
|-----------|-------------------------------|-------------------------------------|
| **Complexity** | Lower | Higher |
| **Scalability** | Limited | Better |
| **Predictability** | Higher | Lower |
| **Flexibility** | Lower | Higher |
| **Debugging** | Easier | Harder |
| **Resilience** | Lower (single points of failure) | Higher |
| **Cost** | Lower (fewer LLM calls) | Higher |

### Key Insight

> "The 5-axis taxonomy isn't about finding the 'best' configuration—it's about consciously choosing trade-offs that match your domain constraints."

**Practical Application:** Before building a multi-agent system, explicitly classify it on all 5 axes. This forces clear architectural decisions and prevents accidental complexity.

---

## Agent Communication Patterns

### Handoffs: The Foundation

**Handoffs** allow one agent to transfer control to another, passing context and determining next steps.

#### Components of a Handoff

1. **Destination**: Which agent receives control
2. **Payload**: Information to pass
3. **Invocation**: How handoff is triggered

### Prebuilt Handoff Tools

LangGraph provides `create_handoff_tool()` for easy agent communication:

```python
from langgraph_supervisor import create_handoff_tool

# Create handoff tool for transitioning to analyst
handoff_to_analyst = create_handoff_tool(
    agent_name="analyst",
    description="Hand off to analyst when data needs analysis",
    include_messages=True,  # Pass full message history
    additional_instructions="Focus on statistical significance"
)

# Agent can use this tool
research_agent = Agent(
    name="researcher",
    tools=[web_search, arxiv_search, handoff_to_analyst],
    instructions="Research topics and hand off to analyst when data is ready"
)
```

**Default behavior** of `create_handoff_tool`:
- ✅ Passes full message history by default
- ✅ Adds tool message indicating successful handoff
- ✅ Can customize what context is passed

#### Custom Handoff Tools

For fine-grained control:

```python
def create_custom_handoff(target_agent: str, context_filter=None):
    @tool
    def handoff_tool(instructions: str = None):
        """Hand off to another agent with optional instructions."""
        return {
            "target": target_agent,
            "context": context_filter(state) if context_filter else state,
            "instructions": instructions
        }
    return handoff_tool

# Filter context to reduce token usage
def summarize_context(state):
    return {
        "summary": summarize(state.messages),
        "key_findings": extract_key_findings(state.messages)
    }

# Use custom handoff
handoff_to_writer = create_custom_handoff(
    target_agent="writer",
    context_filter=summarize_context
)
```

**Benefits of custom handoffs**:
- 🎯 Filter irrelevant context
- 💰 Reduce token usage
- 🎨 Add specific instructions per transition
- 🔧 Rename actions to influence LLM behavior

### Command: Modern Multi-Agent Communication (2025)

**Command** is LangGraph's latest feature for multi-agent coordination, announced in 2025.

#### What Command Does

Command allows nodes to specify:
1. **State update**: What to add/modify in state
2. **Next node**: Where to go next (dynamic routing)

```python
from langgraph.types import Command, END
from typing import Literal

def agent(state: MessagesState) -> Command[Literal["agent_a", "agent_b", END]]:
    # Process and generate response
    response = llm.invoke(state.messages)

    # Dynamically decide next agent
    if "needs_analysis" in response.content:
        next_agent = "agent_a"
    elif "needs_writing" in response.content:
        next_agent = "agent_b"
    else:
        next_agent = END

    return Command(
        goto=next_agent,
        update={"messages": [response]}
    )
```

#### Advantages Over Traditional Edges

**Before Command** (static edges):
```python
# Required separate routing function
def route_function(state):
    if condition:
        return "agent_a"
    else:
        return "agent_b"

graph.add_conditional_edges("router", route_function)
```

**With Command** (dynamic routing):
```python
# Agent decides directly in return value
def agent(state):
    return Command(
        goto="agent_a" if condition else "agent_b",
        update={...}
    )
```

**Key Benefits**:
- ✅ Less boilerplate (no separate routing functions)
- ✅ Agent controls own destiny
- ✅ Supports hierarchical jumps (child → parent graph)
- ✅ Maintains graph visualization
- ✅ Type hints for autocomplete: `Command[Literal["a", "b", END]]`

#### Hierarchical Communication

Command enables child agents to jump to parent graph nodes:

```python
# Child graph agent can return to parent
def sub_agent(state) -> Command[Literal["parent_node", END]]:
    if needs_parent_intervention:
        return Command(
            goto="parent_node",  # Jump to parent graph
            update={"escalation_reason": "..."}
        )
    else:
        return Command(goto=END, update={...})
```

**Use cases**:
- Escalation to supervisor
- Breaking out of sub-workflows
- Dynamic hierarchy traversal

---

## Talkshow & Debate Patterns

### Pattern 1: Round-Robin Debate

**Scenario**: Multiple agents debate a topic, taking turns responding.

```python
from langgraph.graph import StateGraph, MessagesState
from typing import Literal

class DebateState(MessagesState):
    topic: str
    rounds_completed: int
    max_rounds: int
    current_speaker: str
    speakers: list[str]

def moderator(state: DebateState):
    """Introduces topic and manages flow"""
    if state.rounds_completed == 0:
        intro = f"Today's debate topic: {state.topic}. Let's begin!"
        return {
            "messages": [AIMessage(content=intro, name="moderator")],
            "current_speaker": state.speakers[0]
        }
    else:
        # Check if debate should end
        if state.rounds_completed >= state.max_rounds:
            summary = "Let's wrap up. Each speaker, please provide closing statements."
            return {
                "messages": [AIMessage(content=summary, name="moderator")],
                "current_speaker": "closing"
            }

def debater_1(state: DebateState):
    """First debater with specific position"""
    prompt = f"""
    You are debating: {state.topic}
    Your position: SUPPORT

    Review the conversation so far:
    {format_messages(state.messages)}

    Provide your next argument (2-3 sentences).
    Counter previous opposing points if applicable.
    """
    response = llm_1.invoke(prompt)

    return {
        "messages": [AIMessage(content=response.content, name="debater_1")]
    }

def debater_2(state: DebateState):
    """Second debater with opposite position"""
    prompt = f"""
    You are debating: {state.topic}
    Your position: OPPOSE

    Review the conversation so far:
    {format_messages(state.messages)}

    Provide your next argument (2-3 sentences).
    Counter previous supporting points if applicable.
    """
    response = llm_2.invoke(prompt)

    return {
        "messages": [AIMessage(content=response.content, name="debater_2")]
    }

def judge(state: DebateState):
    """Evaluates debate and determines winner"""
    prompt = f"""
    Review this debate on: {state.topic}

    Full transcript:
    {format_messages(state.messages)}

    Evaluate based on:
    1. Argument strength
    2. Evidence quality
    3. Rebuttal effectiveness
    4. Logical coherence

    Provide:
    - Winner (debater_1 or debater_2)
    - Reasoning (3-4 sentences)
    - Key strengths of each side
    """
    judgment = llm_judge.invoke(prompt)

    return {
        "messages": [AIMessage(content=judgment.content, name="judge")]
    }

def router(state: DebateState) -> Literal["debater_1", "debater_2", "moderator", "judge", END]:
    """Routes to next speaker"""

    # First check if we should end
    if state.rounds_completed >= state.max_rounds:
        # Check if closing statements done
        if any(m.name == "judge" for m in state.messages):
            return END
        else:
            return "judge"

    # Determine next speaker
    current = state.current_speaker

    if current == "moderator":
        return "debater_1"
    elif current == "debater_1":
        return "debater_2"
    elif current == "debater_2":
        # Increment round
        state.rounds_completed += 1
        # Back to moderator for round transition
        return "moderator"
    else:
        return "moderator"

# Build graph
debate_graph = StateGraph(DebateState)
debate_graph.add_node("moderator", moderator)
debate_graph.add_node("debater_1", debater_1)
debate_graph.add_node("debater_2", debater_2)
debate_graph.add_node("judge", judge)

# Set entry point
debate_graph.set_entry_point("moderator")

# Add conditional routing
debate_graph.add_conditional_edges(
    "moderator",
    router
)
debate_graph.add_conditional_edges(
    "debater_1",
    router
)
debate_graph.add_conditional_edges(
    "debater_2",
    router
)
debate_graph.add_conditional_edges(
    "judge",
    router
)

app = debate_graph.compile()

# Run debate
result = app.invoke({
    "topic": "Should AI have legal rights?",
    "rounds_completed": 0,
    "max_rounds": 3,
    "current_speaker": "moderator",
    "speakers": ["debater_1", "debater_2"],
    "messages": []
})
```

**Key Features**:
- ✅ Structured turn-taking
- ✅ Round counting and termination
- ✅ Moderator manages flow
- ✅ Judge provides final evaluation

---

### Pattern 2: Talkshow with Host & Guests

**Scenario**: Host interviews multiple guests, managing conversation flow.

```python
class TalkshowState(MessagesState):
    topic: str
    host: str
    guests: list[str]
    current_guest: int
    questions_asked: int
    max_questions_per_guest: int

def host_agent(state: TalkshowState) -> Command[Literal["guest_1", "guest_2", "guest_3", END]]:
    """Host asks questions and manages discussion"""

    # Determine what to do
    if state.questions_asked == 0:
        # Opening
        question = f"Welcome everyone! Today we're discussing {state.topic}. Let's start with our first guest."
        next_guest = "guest_1"

    elif state.questions_asked >= state.max_questions_per_guest * len(state.guests):
        # Closing
        question = "That's all the time we have. Thank you to all our guests!"
        next_guest = END

    else:
        # Ask next question
        guest_idx = state.current_guest
        guest_name = state.guests[guest_idx]

        # LLM generates contextual question
        question_prompt = f"""
        You are a talkshow host discussing: {state.topic}

        Conversation so far:
        {format_messages(state.messages[-5:])}  # Last 5 messages for context

        Ask an insightful follow-up question to {guest_name}.
        Build on previous answers. Be engaging and specific.
        """

        question = llm_host.invoke(question_prompt).content

        # Rotate to next guest if needed
        questions_for_current = sum(
            1 for m in state.messages
            if m.name == f"guest_{guest_idx + 1}"
        )

        if questions_for_current >= state.max_questions_per_guest:
            state.current_guest = (state.current_guest + 1) % len(state.guests)

        next_guest = f"guest_{state.current_guest + 1}"

    return Command(
        goto=next_guest,
        update={
            "messages": [AIMessage(content=question, name="host")],
            "questions_asked": state.questions_asked + 1
        }
    )

def create_guest_agent(name: str, expertise: str):
    """Factory for creating guest agents"""
    def guest_agent(state: TalkshowState) -> Command[Literal["host"]]:
        # Get last question from host
        last_message = state.messages[-1]

        prompt = f"""
        You are a talkshow guest named {name}.
        Your expertise: {expertise}

        Topic: {state.topic}

        Recent conversation:
        {format_messages(state.messages[-3:])}

        Host just asked: {last_message.content}

        Provide a thoughtful, engaging answer (2-4 sentences).
        Reference previous points if relevant.
        """

        answer = llm_guest.invoke(prompt).content

        # Always hand back to host
        return Command(
            goto="host",
            update={"messages": [AIMessage(content=answer, name=name)]}
        )

    return guest_agent

# Create guest agents
guest_1 = create_guest_agent("Dr. Smith", "AI Ethics")
guest_2 = create_guest_agent("Prof. Lee", "Machine Learning")
guest_3 = create_guest_agent("Ms. Jones", "AI Policy")

# Build graph
talkshow = StateGraph(TalkshowState)
talkshow.add_node("host", host_agent)
talkshow.add_node("guest_1", guest_1)
talkshow.add_node("guest_2", guest_2)
talkshow.add_node("guest_3", guest_3)

talkshow.set_entry_point("host")

compiled_talkshow = talkshow.compile()

# Run talkshow
result = compiled_talkshow.invoke({
    "topic": "The Future of AI Governance",
    "host": "host",
    "guests": ["Dr. Smith", "Prof. Lee", "Ms. Jones"],
    "current_guest": 0,
    "questions_asked": 0,
    "max_questions_per_guest": 2,
    "messages": []
})
```

**Key Features**:
- ✅ Host controls conversation flow
- ✅ Guests respond to questions
- ✅ Dynamic question generation based on context
- ✅ Automatic guest rotation
- ✅ Configurable interview length

---

### Pattern 3: Panel Discussion with Cross-Talk

**Scenario**: Multiple experts discuss topic freely with occasional moderation.

```python
class PanelState(MessagesState):
    topic: str
    panelists: list[str]
    moderator: str
    speaking_turns: int
    max_turns: int

def panel_router(state: PanelState) -> str:
    """LLM-based routing: who should speak next?"""

    # Every N turns, moderator intervenes
    if state.speaking_turns % 5 == 0:
        return "moderator"

    # Otherwise, LLM decides based on conversation flow
    routing_prompt = f"""
    This is a panel discussion on: {state.topic}

    Panelists: {', '.join(state.panelists)}

    Recent conversation:
    {format_messages(state.messages[-4:])}

    Who should speak next? Consider:
    1. Who hasn't spoken recently
    2. Who has relevant expertise for current subtopic
    3. Who would provide interesting counterpoint

    Respond with just the panelist name.
    """

    next_speaker = llm_router.invoke(routing_prompt).content.strip()

    # Validate
    if next_speaker not in state.panelists:
        # Default to least recent speaker
        next_speaker = find_least_recent_speaker(state)

    return next_speaker

def moderator_agent(state: PanelState):
    """Moderator guides discussion and asks new questions"""

    prompt = f"""
    You are moderating a panel on: {state.topic}

    Conversation so far:
    {format_messages(state.messages[-6:])}

    Provide:
    1. Brief summary of discussion points so far
    2. New question to explore a different angle or deepen the discussion
    """

    response = llm_moderator.invoke(prompt)

    return {"messages": [AIMessage(content=response.content, name="moderator")]}

def create_panelist(name: str, perspective: str):
    def panelist_agent(state: PanelState):
        prompt = f"""
        You are {name}, a panelist with this perspective: {perspective}

        Panel topic: {state.topic}

        Recent discussion:
        {format_messages(state.messages[-5:])}

        Provide your thoughts (2-3 sentences):
        - Build on previous points
        - Offer your unique perspective
        - You may agree or disagree with others
        """

        response = llm.invoke(prompt)

        return {
            "messages": [AIMessage(content=response.content, name=name)],
            "speaking_turns": state.speaking_turns + 1
        }

    return panelist_agent

# Build panel
panel_graph = StateGraph(PanelState)

panel_graph.add_node("moderator", moderator_agent)
panel_graph.add_node("alice", create_panelist("Alice", "Tech optimist, AI acceleration"))
panel_graph.add_node("bob", create_panelist("Bob", "Cautious skeptic, AI safety focus"))
panel_graph.add_node("carol", create_panelist("Carol", "Ethicist, focuses on societal impact"))

panel_graph.set_entry_point("moderator")

# Dynamic routing from each node
for node in ["moderator", "alice", "bob", "carol"]:
    panel_graph.add_conditional_edges(
        node,
        panel_router,
        {
            "moderator": "moderator",
            "alice": "alice",
            "bob": "bob",
            "carol": "carol"
        }
    )

panel = panel_graph.compile()
```

**Key Features**:
- ✅ LLM-based dynamic routing (organic flow)
- ✅ Periodic moderator interventions
- ✅ Cross-talk between panelists
- ✅ Consideration of speaking history

---

### Pattern 4: GroupChat with Subgraphs

**Scenario**: Complex conversation with nested groups and private channels.

```python
from langgraph.graph import StateGraph

# Main discussion state
class MainDiscussionState(MessagesState):
    breakout_requested: bool
    breakout_group: str

# Breakout group state
class BreakoutState(MessagesState):
    group_name: str
    participants: list[str]

# Main discussion graph
def facilitator(state: MainDiscussionState):
    """Main facilitator manages overall discussion"""
    if should_have_breakout(state.messages):
        return {
            "messages": [AIMessage(content="Let's split into breakout groups", name="facilitator")],
            "breakout_requested": True
        }
    else:
        # Continue main discussion
        return {"messages": [generate_discussion_prompt()]}

# Breakout group subgraph
def create_breakout_subgraph(group_name: str, participants: list[str]):
    """Create isolated breakout discussion"""

    subgraph = StateGraph(BreakoutState)

    for participant in participants:
        subgraph.add_node(
            participant,
            create_participant_agent(participant)
        )

    # Add routing logic for breakout
    # ... (similar to panel pattern)

    return subgraph.compile()

# Main graph integrates breakout subgraphs
main_graph = StateGraph(MainDiscussionState)
main_graph.add_node("facilitator", facilitator)

# Add breakout subgraphs as nodes
main_graph.add_node("breakout_tech", create_breakout_subgraph(
    "tech_group",
    ["alice", "bob"]
))
main_graph.add_node("breakout_policy", create_breakout_subgraph(
    "policy_group",
    ["carol", "dave"]
))

# Conditional routing to breakouts
def route_breakouts(state):
    if state.breakout_requested:
        return ["breakout_tech", "breakout_policy"]  # Parallel
    else:
        return "facilitator"

main_graph.add_conditional_edges("facilitator", route_breakouts)

# After breakouts, return to main
main_graph.add_edge("breakout_tech", "facilitator")
main_graph.add_edge("breakout_policy", "facilitator")

compiled = main_graph.compile()
```

**Key Features**:
- ✅ Hierarchical conversation structure
- ✅ Parallel breakout sessions
- ✅ Isolated group contexts
- ✅ Reconvene to main discussion

---

## Production Case Studies

### LinkedIn: SQL Bot

**Challenge**: Enable non-technical employees to query data independently

**Solution**: Multi-agent system with LangGraph
- **Router agent**: Determines intent and required tables
- **Query agent**: Writes SQL queries
- **Validator agent**: Checks query correctness
- **Fixer agent**: Debugs and corrects errors
- **Explanation agent**: Translates results to natural language

**Architecture**: Supervisor pattern

**Results**:
- ✅ Employees across functions can access data insights independently
- ✅ Respects permissions automatically
- ✅ Self-healing (automatic error correction)

**Key Lesson**: "Careful network of specialized agents ensures each step handled with precision"

---

### Uber: Code Migration Assistant

**Challenge**: Large-scale code migrations across developer platform

**Solution**: Multi-agent unit test generation system

**Agents**:
- **Analysis agent**: Understands existing code
- **Generation agent**: Creates unit tests
- **Validation agent**: Runs and validates tests
- **Refinement agent**: Fixes failing tests

**Architecture**: Sequential with loops (test-fix-retest)

**Results**:
- ✅ Dramatically accelerated migration timelines
- ✅ "LangGraph greatly sped up development cycle when scaling complex workflows"

**Key Lesson**: Modularity allows updating individual agents without breaking system

---

### Replit: AI Copilot

**Challenge**: Build software from scratch with transparency

**Solution**: Multi-agent system with human-in-the-loop

**Agents**:
- **Planning agent**: Breaks down requirements
- **Coding agent**: Writes code
- **Package agent**: Manages dependencies
- **File agent**: Creates/modifies files
- **UI agent**: Updates user interface

**Architecture**: Swarm with user visibility

**Key Innovation**: Users see every agent action (package installations, file creation)

**Results**:
- ✅ Transparent development process
- ✅ Users can intervene at any step
- ✅ Builds entire applications autonomously

**Key Lesson**: Human-in-the-loop transparency builds trust in agentic systems

---

### Elastic: Security Threat Detection

**Challenge**: Real-time threat detection and response

**Solution**: Network of AI agents for security

**Agents**:
- **Detection agents**: Monitor for threats (parallel)
- **Analysis agents**: Investigate suspicious activity
- **Response agents**: Execute countermeasures
- **Escalation agents**: Alert security team

**Architecture**: Network/Swarm (parallel monitoring)

**Results**:
- ✅ Faster threat response
- ✅ More effective than single-agent approach
- ✅ Parallelization crucial for real-time performance

**Key Lesson**: Swarm architecture excels when parallelization is primary requirement

---

### Anthropic: Multi-Agent Research System (2025)

**Challenge**: Improve research depth and completeness in open-ended queries

**Solution**: Parallel multi-agent exploration with breadth-first search

**Architecture**: Swarm with parallel subagents
- **Orchestrator**: Breaks down research into parallel tracks
- **Research subagents**: Each explores different angles simultaneously
- **Synthesizer**: Combines findings into coherent output

**Research Findings** (arXiv/Anthropic 2025):
- 90.2% improvement on breadth-first queries vs single agent
- Multi-agent excels at **parallel exploration** (research, multiple data sources)
- Single agent better for **deep sequential reasoning** (math proofs, step-by-step logic)

**Key Lesson**: Match architecture to task nature—parallel exploration → multi-agent; deep reasoning → single agent

---

### Zapier: 800+ Agents in Production (2025)

**Challenge**: Scale agent deployment across enterprise automation workflows

**Solution**: Massive multi-agent deployment with centralized orchestration

**Scale**:
- 800+ specialized agents in production
- Handles diverse automation tasks
- Enterprise-grade reliability

**Architecture Insights**:
- **Agent specialization**: Each agent handles narrow domain
- **Central routing**: Intelligent task-to-agent matching
- **Error isolation**: Failed agents don't cascade to others
- **Monitoring**: Comprehensive observability across all agents

**Key Lesson**: Large-scale multi-agent is viable with proper orchestration and isolation

---

## Context Management Patterns (2025)

### The Context Bottleneck Problem

Research shows context management is the primary bottleneck in multi-agent systems (Google ADK 2025).

**Two-Layer Context Architecture**:

```
┌─────────────────────────────────────────┐
│           Session Context               │
│  (Persistent across conversations)      │
│  - User preferences                     │
│  - Long-term memory                     │
│  - Learned patterns                     │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│          Working Context                │
│  (Current task state)                   │
│  - Active messages                      │
│  - Tool results                         │
│  - Intermediate computations            │
└─────────────────────────────────────────┘
```

### JSPLIT: 100x Token Reduction

**Problem**: Full context passing between agents is expensive and slow

**Solution**: JSPLIT (arXiv 2025) compresses context for handoffs

```python
# Instead of passing full message history
handoff_context = jsplit_compress(
    messages=state.messages,
    relevance_threshold=0.7,
    max_tokens=1000
)

# Achieves 100x token reduction with minimal info loss
return Command(
    goto="next_agent",
    update={"compressed_context": handoff_context}
)
```

**Key Insight**: Agents don't need full history—summarized context often sufficient

### Context Degradation Patterns

**Error Cascade Pattern**:
```
Agent A (small error) → Agent B (amplifies) → Agent C (fails)
```

**Prevention**:
1. **Validation checkpoints**: Verify context quality at each handoff
2. **Error bounds**: Set thresholds for acceptable context quality
3. **Recovery hooks**: Allow agents to request clarification

```python
def validate_handoff(context):
    quality_score = assess_context_quality(context)
    if quality_score < 0.6:
        return Command(goto="clarification_agent")
    return Command(goto="next_agent", update={"context": context})
```

---

## Best Practices

### 1. State Management

**Keep State Minimal, Typed, and Validated**

```python
from typing import TypedDict
from pydantic import BaseModel, Field

# ❌ Bad: Untyped, bloated state
state = {
    "stuff": [...],
    "things": {...},
    "temp_val": None
}

# ✅ Good: Typed, minimal
class AgentState(TypedDict):
    messages: Annotated[list[Message], add_messages]
    current_agent: str
    iteration: int

# ✅ Even better: Pydantic with validation
class AgentState(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    current_agent: str = Field(pattern="^(agent_a|agent_b|agent_c)$")
    iteration: int = Field(ge=0, le=20)  # 0-20 iterations max
```

**Use Reducers Sparingly**

```python
# ✅ Use add_messages reducer for message accumulation
messages: Annotated[list[Message], add_messages]

# ❌ Don't use reducers for simple overwrites
iteration: Annotated[int, lambda x, y: x + y]  # Overkill

# ✅ Simple overwrite is fine
iteration: int  # Just overwrites
```

**Don't Dump Transient Values into State**

```python
# ❌ Bad: Putting temp values in state
def agent(state):
    temp_result = calculate_something()
    return {"temp_result": temp_result, "final": use_temp(temp_result)}

# ✅ Good: Use function scope
def agent(state):
    temp_result = calculate_something()  # Local variable
    return {"final": use_temp(temp_result)}  # Only final result in state
```

---

### 2. Persistence & Checkpointing

**Use Postgres Checkpointer for Production**

```python
from langgraph.checkpoint.postgres import PostgresSaver

# ✅ Production: Postgres for durability
checkpointer = PostgresSaver.from_conn_string("postgresql://...")

graph = graph.compile(checkpointer=checkpointer)

# Thread-scoped checkpoints
config = {"configurable": {"thread_id": "user-123-session-456"}}
result = graph.invoke(input, config=config)
```

**Benefits**:
- ✅ Error recovery: Resume from last checkpoint
- ✅ Human-in-the-loop: Pause and resume workflows
- ✅ Debugging: Inspect state at each step
- ✅ Multi-instance: Multiple workers share state

**For Development**:

```python
from langgraph.checkpoint.memory import MemorySaver

# ⚠️ Development only: In-memory (lost on restart)
checkpointer = MemorySaver()
```

---

### 3. Context Engineering

**Critical for Reliability**

> "Context engineering is critical to making agentic systems work reliably" - Anthropic/LangChain

**Problem**: Vague, short instructions lead to agent confusion

```python
# ❌ Vague instruction
agent = Agent(
    instructions="Help with analysis"
)
```

**Solution**: Detailed, specific instructions with examples

```python
# ✅ Detailed context
agent = Agent(
    instructions="""
    You are a data analysis specialist.

    Your role:
    - Analyze CSV and JSON datasets
    - Identify trends, outliers, correlations
    - Create visualizations with matplotlib
    - Provide statistical summaries

    When you receive data:
    1. First, inspect the schema and data types
    2. Check for missing values or anomalies
    3. Generate descriptive statistics
    4. Create 2-3 relevant visualizations
    5. Summarize findings in clear language

    Example flow:
    User: "Analyze sales.csv"
    You: *Load data, inspect schema*
    You: "Dataset has 3 columns (date, product, revenue), 1000 rows, no missing values"
    You: *Generate stats and visualizations*
    You: "Key findings: Revenue peaks on weekends, Product A is 60% of sales..."

    Available tools: pandas, matplotlib, numpy
    Output format: Markdown with embedded visualizations
    """
)
```

**Key Principles**:
- Explain business entities and overall flow
- Provide concrete examples
- Define expected outputs clearly
- Specify available tools and how to use them

---

### 4. Edge Design

**Use Simple Edges Where Possible**

```python
# ✅ Simple edge for deterministic flow
graph.add_edge("agent_a", "agent_b")

# Only use conditional when truly needed
def should_continue(state):
    return "agent_c" if state.iteration < 5 else END

graph.add_conditional_edges("agent_b", should_continue)
```

**Bound Cycles to Prevent Infinite Loops**

```python
# ❌ Unbounded cycle risk
def router(state):
    if not state.done:
        return "agent"  # Could loop forever
    return END

# ✅ Bounded cycle
def router(state):
    if state.iteration < MAX_ITERATIONS and not state.done:
        return "agent"
    return END

def agent(state):
    return {
        "iteration": state.iteration + 1,
        # ... other updates
    }
```

---

### 5. Communication Patterns

**Choose Appropriate Pattern for Use Case**

| Pattern | Use When | Avoid When |
|---------|----------|-----------|
| **Shared Scratchpad** | Debate, collaboration, transparency needed | Context window concerns, many agents |
| **Supervisor** | Clear hierarchy, quality control, moderate complexity | Need parallelization, supervisor becomes bottleneck |
| **Swarm/Network** | High parallelization, dynamic routing, peer collaboration | Need predictable flow, difficult debugging tolerance low |

**Message Filtering to Reduce Token Usage**

```python
# ❌ Pass all messages to every agent
def agent(state):
    response = llm.invoke(state.messages)  # Could be 100+ messages

# ✅ Filter to recent relevant messages
def agent(state):
    relevant_messages = filter_messages(
        state.messages,
        role="user",
        last_n=5,
        keywords=["analysis", "data"]
    )
    response = llm.invoke(relevant_messages)
```

---

### 6. Agent Specialization

**Focused Agents Outperform Generalists**

```python
# ❌ One agent with all tools (15+ tools)
generalist = Agent(
    name="do_everything",
    tools=[web_search, arxiv, calculator, sql_query, visualization,
           file_read, file_write, api_call, ...]  # 15 tools
)

# ✅ Specialized agents with focused tool sets
researcher = Agent(
    name="researcher",
    tools=[web_search, arxiv_search]  # 2 tools
)

analyst = Agent(
    name="analyst",
    tools=[calculator, data_analysis, visualization]  # 3 tools
)

coordinator = Agent(
    name="coordinator",
    tools=[handoff_to_researcher, handoff_to_analyst]  # 2 tools
)
```

**Benefits of Specialization**:
- ✅ Less tool confusion
- ✅ Better prompt engineering (focused context)
- ✅ Easier to evaluate and improve
- ✅ Benchmarks show superior performance

---

### 7. Monitoring & Debugging

**LangSmith Integration**

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-multi-agent-system"

# Traces automatically captured
result = graph.invoke(input)
```

**Log Agent Transitions**

```python
def supervisor(state):
    next_agent = decide_next_agent(state)

    # Log decision
    logger.info(f"Routing to {next_agent}", extra={
        "current_agent": state.current_agent,
        "iteration": state.iteration,
        "reason": get_routing_reason(state)
    })

    return {
        "current_agent": next_agent,
        ...
    }
```

**Track Agent Performance**

```python
import time

def timed_agent(agent_func):
    def wrapper(state):
        start = time.time()
        result = agent_func(state)
        duration = time.time() - start

        metrics.record("agent_duration", duration, tags={
            "agent": agent_func.__name__
        })

        return result
    return wrapper

@timed_agent
def research_agent(state):
    # ... agent logic
```

---

## Implementation Patterns

### Pattern: Supervisor with Error Recovery

```python
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.postgres import PostgresSaver

# Agents with error handling
def safe_agent_wrapper(agent_func):
    def wrapper(state):
        try:
            return agent_func(state)
        except Exception as e:
            return {
                "messages": [AIMessage(
                    content=f"Error: {str(e)}",
                    name=agent_func.__name__
                )],
                "error": str(e),
                "failed_agent": agent_func.__name__
            }
    return wrapper

research_agent = safe_agent_wrapper(research_agent_impl)
analyst_agent = safe_agent_wrapper(analyst_agent_impl)

supervisor = create_supervisor(
    agents=[research_agent, analyst_agent],
    model="gpt-4o"
)

# Checkpointing for recovery
checkpointer = PostgresSaver.from_conn_string(conn_string)
graph = supervisor.build_graph()
compiled = graph.compile(checkpointer=checkpointer)

# Use with error recovery
config = {"configurable": {"thread_id": "session-123"}}

try:
    result = compiled.invoke(input, config=config)
except Exception as e:
    # Can resume from last successful checkpoint
    state = compiled.get_state(config)
    # ... handle error, modify state, resume
    result = compiled.invoke(None, config=config)  # Resume
```

---

### Pattern: Swarm with Dynamic Tool Selection

```python
from langgraph_swarm import create_swarm
from typing import Literal

# Tools that agents can use
tools = {
    "search": web_search_tool,
    "analyze": data_analysis_tool,
    "visualize": visualization_tool
}

def agent_with_dynamic_tools(name: str, default_tools: list[str]):
    def agent(state) -> Command[Literal["agent_a", "agent_b", END]]:
        # Determine which tools needed for this task
        task = state.messages[-1].content
        needed_tools = infer_needed_tools(task)

        # Select tools dynamically
        agent_tools = [tools[t] for t in needed_tools if t in tools]

        # Use tools
        result = use_tools(task, agent_tools)

        # Decide next agent or end
        if task_complete(result):
            return Command(goto=END, update={"messages": [result]})
        elif needs_analysis(result):
            return Command(goto="agent_b", update={"messages": [result]})
        else:
            return Command(goto="agent_a", update={"messages": [result]})

    return agent

# Build swarm
swarm = create_swarm(
    agents=[
        agent_with_dynamic_tools("agent_a", ["search", "analyze"]),
        agent_with_dynamic_tools("agent_b", ["analyze", "visualize"])
    ]
)
```

---

## Common Pitfalls

### 1. Too Many Agents

**Problem**: >75% of systems with 5+ agents become difficult to manage

**Symptoms**:
- Exponential debugging complexity
- Unclear responsibility boundaries
- High coordination overhead

**Solution**:
```python
# ❌ 10 hyper-specialized agents
agents = [data_cleaner, data_validator, data_transformer,
          data_analyzer, data_visualizer, report_writer,
          report_editor, report_formatter, report_validator,
          report_publisher]

# ✅ 3-4 well-designed agents
agents = [
    data_processor,  # Combines cleaning, validating, transforming
    analyst,  # Analysis and visualization
    report_generator  # Writing, formatting, publishing
]
```

**Best Practice**: Start with 2-3 agents, add more only when clear benefit

---

### 2. Shared Context Anti-Pattern

**Problem**: All agents need identical information

**Why It's Bad**:
- Defeats purpose of multi-agent (no specialization)
- Better served by single agent with well-engineered prompt

**Example**:
```python
# ❌ All agents need full context
# Just use a single agent instead!

# ✅ Agents have distinct context needs
researcher: "Find papers on topic X"
analyst: "Analyze these papers (receives summaries, not full papers)"
writer: "Write report (receives analysis results, not raw data)"
```

---

### 3. Infinite Loops

**Problem**: Agents hand off in cycles without progress

**Prevention**:

```python
class LoopSafeState(MessagesState):
    iteration: int
    max_iterations: int = 20
    agents_visited: list[str]

def check_termination(state):
    # Hard limit
    if state.iteration >= state.max_iterations:
        return END

    # Detect cycles
    if len(state.agents_visited) > len(set(state.agents_visited)) * 2:
        # Same agents visited too many times
        return END

    # Progress check
    if state.iteration > 5 and not has_made_progress(state):
        return END

    return continue_routing(state)
```

---

### 4. Context Window Bloat

**Problem**: Message history grows unbounded in collaborative pattern

**Solution**: Implement summarization

```python
def manage_context(state):
    if len(state.messages) > 50:
        # Summarize old messages
        old_messages = state.messages[:30]
        summary = summarize_messages(old_messages)
        recent_messages = state.messages[30:]

        return {
            "messages": [
                AIMessage(content=f"Summary of previous discussion: {summary}"),
                *recent_messages
            ]
        }
    return state
```

---

### 5. Ignoring Benchmarks

**Mistake**: Adding agents without measuring impact

**Best Practice**: Benchmark before and after

```python
# Establish baseline
single_agent_performance = benchmark(single_agent, test_cases)

# Test multi-agent
multi_agent_performance = benchmark(multi_agent_system, test_cases)

# Compare
if multi_agent_performance.accuracy > single_agent_performance.accuracy:
    # Multi-agent justified
    deploy(multi_agent_system)
else:
    # Stick with single agent
    deploy(single_agent)
```

**LangChain's finding**: Multi-agent only shows clear advantage with 2+ distractor domains

---

### 8. Emergent Coordination (2025 Research)

**Finding**: Theory-of-mind prompts enable emergent coordination (arXiv:2510.05174)

**What Works**:
```python
# ✅ Add theory-of-mind to agent instructions
agent = Agent(
    instructions="""
    You are Agent A in a multi-agent system.

    Before acting, consider:
    - What might other agents be doing right now?
    - What information will they need from you?
    - How will your output affect downstream agents?

    [Rest of instructions...]
    """
)
```

**Why It Matters**:
- Agents that "think about others" coordinate better
- Reduces handoff errors and miscommunication
- Enables emergent collaborative behavior without explicit programming

**Anti-Pattern**:
```python
# ❌ Isolated agent thinking
agent = Agent(
    instructions="You are Agent A. Do X when asked."  # No awareness of system
)
```

---

## Summary: Decision Tree

```
START
  │
  ├─ Is task in single domain with <10 tools?
  │   └─ YES → Use Single Agent
  │
  ├─ Does task have 2+ distinct domains?
  │   └─ YES → Consider Multi-Agent
  │       │
  │       ├─ Need clear hierarchy & quality control?
  │       │   └─ YES → Use Supervisor Pattern
  │       │
  │       ├─ Need maximum parallelization & flexibility?
  │       │   └─ YES → Use Swarm Pattern
  │       │
  │       └─ Balanced requirements?
  │           └─ Use Network/Collaborative Pattern
  │
  ├─ All agents need same context?
  │   └─ YES → Use Single Agent (multi-agent won't help)
  │
  └─ Heavily dependent sequential tasks?
      └─ YES → Use Single Agent or Simple Sequential Workflow
```

**Key Takeaway**: "Start from your goals and constraints" - Don't default to multi-agent. Use when complexity justifies overhead.

---

## 10. Agent-to-Agent (A2A) Protocol

**Google's open standard for secure, interoperable agent communication**

### 10.1 A2A Protocol Overview

A2A (Agent-to-Agent) is an open protocol launched by Google Cloud in April 2025 for enabling secure communication between AI agents across different vendors, frameworks, and platforms. It's now hosted by the Linux Foundation.

```
A2A vs MCP Relationship:

┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Agent System                               │
│                                                                     │
│    ┌──────────────────────────────────────────────────────────┐    │
│    │                A2A Layer (Horizontal)                     │    │
│    │    Agent ←──────────→ Agent ←──────────→ Agent           │    │
│    │    (Task Delegation, Discovery, Coordination)             │    │
│    └──────────────────────────────────────────────────────────┘    │
│                        │               │                            │
│    ┌──────────────────────────────────────────────────────────┐    │
│    │                MCP Layer (Vertical)                       │    │
│    │    Agent ←──────→ Tools/Data ←──────→ Resources          │    │
│    │    (Tool Access, Context, Prompts)                        │    │
│    └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘

A2A: Agent-to-Agent (who talks to whom, task lifecycle)
MCP: Agent-to-Tools (what capabilities agents have)
```

### 10.2 A2A Protocol Specification

**Core Technologies:**
- **Transport**: HTTP/HTTPS
- **Message Format**: JSON-RPC 2.0
- **Streaming**: Server-Sent Events (SSE)
- **Discovery**: Agent Cards (JSON-LD)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import json

class TaskState(Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AgentCard:
    """
    Agent Card: JSON-LD metadata describing agent capabilities.
    Served at /.well-known/agent.json
    """
    context: str = "https://a2a-protocol.org/context"
    protocol_version: str = "2025-03-25"
    title: str = ""
    description: str = ""
    service_endpoints: Dict[str, str] = field(default_factory=dict)
    authentication: Dict[str, str] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    supported_modalities: List[str] = field(default_factory=list)

    def to_json(self) -> Dict:
        return {
            "@context": self.context,
            "protocolVersion": self.protocol_version,
            "title": self.title,
            "description": self.description,
            "serviceEndpoints": self.service_endpoints,
            "authentication": self.authentication,
            "capabilities": self.capabilities,
            "supportedModalities": self.supported_modalities
        }


@dataclass
class A2ATask:
    """
    A2A Task: The fundamental unit of work between agents.
    """
    task_id: str
    task_type: str
    status: TaskState
    message: str
    params: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_json_rpc(self, method: str) -> Dict:
        return {
            "jsonrpc": "2.0",
            "method": method,
            "params": {
                "taskId": self.task_id,
                "taskType": self.task_type,
                "message": self.message,
                "params": self.params
            },
            "id": self.task_id
        }
```

### 10.3 A2A Server Implementation

```python
from flask import Flask, jsonify, request, Response
import json
import uuid

app = Flask(__name__)

# Task storage
tasks: Dict[str, A2ATask] = {}

# Agent Card endpoint
@app.route('/.well-known/agent.json')
def agent_card():
    """
    Serve Agent Card for discovery.
    Other agents query this to understand capabilities.
    """
    card = AgentCard(
        title="Research Agent",
        description="Performs web research and summarization",
        service_endpoints={
            "tasks/send": "/a2a/tasks/send",
            "tasks/get": "/a2a/tasks/get",
            "tasks/cancel": "/a2a/tasks/cancel",
            "notifications/receive": "/a2a/notifications"
        },
        authentication={
            "type": "api-key",
            "header": "X-API-Key"
        },
        capabilities=["streaming", "push_notifications", "stateful"],
        supported_modalities=["text", "json"]
    )
    return jsonify(card.to_json())

# Task submission endpoint
@app.route('/a2a/tasks/send', methods=['POST'])
def send_task():
    """
    Receive new task from client agent.
    JSON-RPC 2.0 format.
    """
    data = request.json

    # Validate JSON-RPC format
    if data.get("jsonrpc") != "2.0":
        return jsonify({
            "jsonrpc": "2.0",
            "error": {"code": -32600, "message": "Invalid Request"},
            "id": None
        }), 400

    # Create task
    task_id = str(uuid.uuid4())
    params = data.get("params", {})

    task = A2ATask(
        task_id=task_id,
        task_type=params.get("taskType", "default"),
        status=TaskState.SUBMITTED,
        message=params.get("message", ""),
        params=params
    )

    tasks[task_id] = task

    # Check if streaming requested
    if params.get("stream"):
        return stream_task_execution(task)

    # Execute task asynchronously
    execute_task_async(task)

    return jsonify({
        "jsonrpc": "2.0",
        "result": {
            "taskId": task_id,
            "status": task.status.value
        },
        "id": data.get("id")
    })

def stream_task_execution(task: A2ATask) -> Response:
    """
    Stream task updates via Server-Sent Events.
    """
    def generate():
        # Update status to working
        task.status = TaskState.WORKING
        yield f"data: {json.dumps({'type': 'status', 'status': 'working'})}\\n\\n"

        # Execute task steps
        for i, step in enumerate(execute_task_steps(task)):
            yield f"data: {json.dumps({'type': 'progress', 'step': i+1, 'content': step})}\\n\\n"

        # Complete
        task.status = TaskState.COMPLETED
        yield f"data: {json.dumps({'type': 'completed', 'result': task.artifacts})}\\n\\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )

# Task status endpoint
@app.route('/a2a/tasks/get', methods=['POST'])
def get_task():
    """
    Get current task status.
    """
    data = request.json
    task_id = data.get("params", {}).get("taskId")

    if task_id not in tasks:
        return jsonify({
            "jsonrpc": "2.0",
            "error": {"code": -32602, "message": "Task not found"},
            "id": data.get("id")
        }), 404

    task = tasks[task_id]
    return jsonify({
        "jsonrpc": "2.0",
        "result": {
            "taskId": task.task_id,
            "status": task.status.value,
            "artifacts": task.artifacts,
            "updatedAt": task.updated_at.isoformat()
        },
        "id": data.get("id")
    })

# Notification endpoint
@app.route('/a2a/notifications', methods=['POST'])
def receive_notification():
    """
    Receive push notifications from other agents.
    """
    notification = request.json
    task_id = notification.get("taskId")

    # Process notification
    if task_id in tasks:
        tasks[task_id].status = TaskState(notification.get("status", "working"))

    return jsonify({"status": "received"})
```

### 10.4 A2A Client Implementation

```python
import requests
from typing import Optional, Generator
import json

class A2AClient:
    """
    Client for communicating with A2A-compatible agents.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.agent_card: Optional[Dict] = None

    def discover_agent(self) -> Dict:
        """
        Discover agent capabilities via Agent Card.
        """
        response = requests.get(
            f"{self.base_url}/.well-known/agent.json",
            headers=self._headers()
        )
        response.raise_for_status()
        self.agent_card = response.json()
        return self.agent_card

    def send_task(
        self,
        task_type: str,
        message: str,
        params: Optional[Dict] = None,
        stream: bool = False
    ) -> Dict:
        """
        Send task to remote agent.
        """
        # Ensure we have agent card
        if not self.agent_card:
            self.discover_agent()

        endpoint = self.agent_card["serviceEndpoints"]["tasks/send"]

        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "taskType": task_type,
                "message": message,
                **(params or {}),
                "stream": stream
            },
            "id": str(uuid.uuid4())
        }

        if stream:
            return self._send_streaming(endpoint, payload)

        response = requests.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()

    def _send_streaming(
        self,
        endpoint: str,
        payload: Dict
    ) -> Generator[Dict, None, None]:
        """
        Send task with streaming response.
        """
        response = requests.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            headers=self._headers(),
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    yield json.loads(line[6:])

    def get_task_status(self, task_id: str) -> Dict:
        """
        Get current status of a task.
        """
        if not self.agent_card:
            self.discover_agent()

        endpoint = self.agent_card["serviceEndpoints"]["tasks/get"]

        response = requests.post(
            f"{self.base_url}{endpoint}",
            json={
                "jsonrpc": "2.0",
                "method": "tasks/get",
                "params": {"taskId": task_id},
                "id": str(uuid.uuid4())
            },
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()

    def _headers(self) -> Dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers


# Usage example
async def coordinate_agents():
    """
    Example: Orchestrate multiple A2A agents.
    """
    # Connect to different specialized agents
    research_agent = A2AClient("https://research-agent.example.com", api_key="...")
    writing_agent = A2AClient("https://writing-agent.example.com", api_key="...")

    # Discover capabilities
    research_caps = research_agent.discover_agent()
    writing_caps = writing_agent.discover_agent()

    print(f"Research agent capabilities: {research_caps['capabilities']}")
    print(f"Writing agent capabilities: {writing_caps['capabilities']}")

    # Send research task
    research_result = research_agent.send_task(
        task_type="research",
        message="Find latest trends in AI agent protocols",
        params={"depth": "comprehensive"}
    )

    # Wait for completion
    task_id = research_result["result"]["taskId"]
    while True:
        status = research_agent.get_task_status(task_id)
        if status["result"]["status"] == "completed":
            break
        await asyncio.sleep(1)

    # Pass to writing agent
    writing_result = writing_agent.send_task(
        task_type="summarize",
        message="Write a summary of the research",
        params={"research": status["result"]["artifacts"]}
    )

    return writing_result
```

### 10.5 A2A vs MCP Comparison

| Aspect | A2A | MCP |
|--------|-----|-----|
| **Focus** | Agent-to-agent coordination | Agent-to-tool access |
| **Interaction** | Horizontal (agent ↔ agent) | Vertical (agent ↔ tools) |
| **State** | Stateful (task lifecycle) | Stateless (per-call) |
| **Discovery** | Agent Cards | Tool schemas |
| **Streaming** | SSE for task updates | SSE for results |
| **Security** | OAuth 2.0, API keys | Fine-grained access |
| **Use Case** | Multi-agent workflows | Enhancing single agent |

### 10.6 Combined A2A + MCP Architecture

```python
class HybridAgentArchitecture:
    """
    Combine A2A (agent coordination) with MCP (tool access).
    """

    def __init__(self):
        # A2A for agent-to-agent communication
        self.a2a_clients: Dict[str, A2AClient] = {}

        # MCP for tool access
        self.mcp_client = MCPClient()

    async def execute_complex_workflow(
        self,
        task: str,
        context: Dict
    ) -> Dict:
        """
        Execute workflow using both A2A and MCP.
        """
        # Step 1: Use MCP to access local tools/data
        local_data = await self.mcp_client.call_tool(
            "database_query",
            {"query": "SELECT * FROM knowledge_base WHERE topic = ?", "params": [task]}
        )

        # Step 2: Use A2A to delegate to specialized agent
        if self._requires_external_agent(task):
            research_agent = self._get_or_create_client("research-agent")

            # Send task via A2A
            result = research_agent.send_task(
                task_type="research",
                message=task,
                params={"context": local_data}
            )

            # Wait for completion
            while True:
                status = research_agent.get_task_status(result["result"]["taskId"])
                if status["result"]["status"] in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)

            external_data = status["result"]["artifacts"]
        else:
            external_data = None

        # Step 3: Use MCP to format and store results
        final_result = await self.mcp_client.call_tool(
            "format_response",
            {"local": local_data, "external": external_data}
        )

        return final_result

    def _get_or_create_client(self, agent_name: str) -> A2AClient:
        """
        Get or create A2A client for agent.
        """
        if agent_name not in self.a2a_clients:
            # Discover agent from registry
            agent_url = self._discover_agent_url(agent_name)
            self.a2a_clients[agent_name] = A2AClient(agent_url)

        return self.a2a_clients[agent_name]
```

### 10.7 A2A Security Considerations

```python
class A2ASecurityManager:
    """
    Security best practices for A2A implementations.
    """

    def __init__(self):
        self.rate_limits: Dict[str, int] = {}
        self.allowed_agents: set = set()

    def validate_request(
        self,
        request_data: Dict,
        headers: Dict
    ) -> bool:
        """
        Validate incoming A2A request.
        """
        # 1. Validate authentication
        if not self._validate_auth(headers):
            raise AuthenticationError("Invalid API key")

        # 2. Validate JSON-RPC format
        if not self._validate_json_rpc(request_data):
            raise ValidationError("Invalid JSON-RPC format")

        # 3. Check rate limits
        client_id = headers.get("X-Client-ID")
        if not self._check_rate_limit(client_id):
            raise RateLimitError("Rate limit exceeded")

        # 4. Validate agent is in allowlist (if configured)
        if self.allowed_agents and client_id not in self.allowed_agents:
            raise AuthorizationError("Agent not authorized")

        return True

    def _validate_json_rpc(self, data: Dict) -> bool:
        """
        Validate JSON-RPC 2.0 format.
        """
        required_fields = ["jsonrpc", "method", "params"]
        if not all(field in data for field in required_fields):
            return False

        if data.get("jsonrpc") != "2.0":
            return False

        return True

    def sanitize_params(self, params: Dict) -> Dict:
        """
        Sanitize task parameters to prevent injection.
        """
        sanitized = {}
        for key, value in params.items():
            if isinstance(value, str):
                # Remove potential injection patterns
                sanitized[key] = self._sanitize_string(value)
            else:
                sanitized[key] = value
        return sanitized
```

### 10.8 A2A Deployment Checklist

**Agent Card Setup:**
- [ ] Agent Card served at `/.well-known/agent.json`
- [ ] All endpoints documented in serviceEndpoints
- [ ] Authentication requirements specified
- [ ] Capabilities accurately described

**Task Handling:**
- [ ] JSON-RPC 2.0 format validated
- [ ] Task lifecycle states properly managed
- [ ] Streaming support via SSE (if needed)
- [ ] Error responses follow JSON-RPC format

**Security:**
- [ ] HTTPS enforced
- [ ] API key or OAuth authentication
- [ ] Rate limiting implemented
- [ ] Input sanitization on all params
- [ ] Audit logging for all requests

**Integration:**
- [ ] Agent discovery tested with client
- [ ] Task delegation tested end-to-end
- [ ] Notification handling implemented
- [ ] MCP integration tested (if hybrid)

---

## 11. Architecture Blueprints: 6 Universal Use Cases

**Production-ready reference architectures for the most common agent applications.**

### 11.1 Customer Support Bot

**Use Case:** Handle customer inquiries, route issues, process simple requests, escalate complex cases.

**Architecture:** Coordinator + Specialized Workers

```
                     ┌─────────────────────┐
                     │   Intent Router     │
                     │   (Coordinator)     │
                     └─────────┬───────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  Order Agent    │   │  Refund Agent   │   │ Escalation Agent│
│  (lookup,track) │   │ (process,verify)│   │  (human HITL)   │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

**Key Components:**

| Component | Responsibility | Tools |
|-----------|----------------|-------|
| **Intent Router** | Classify intent, route to specialist | intent_classifier |
| **Order Agent** | Lookup orders, track shipments | order_api, tracking_api |
| **Refund Agent** | Check eligibility, process refunds | refund_api, payment_api |
| **Escalation Agent** | Create tickets, notify humans | ticketing_api, slack_api |

**Implementation Pattern (LangGraph):**

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    customer_id: str
    resolved: bool

# Specialized agents
order_agent = create_react_agent(llm, [lookup_order, track_shipment])
refund_agent = create_react_agent(llm, [check_eligibility, process_refund])
escalation_agent = create_react_agent(llm, [create_ticket, notify_human])

def route_intent(state: SupportState) -> str:
    intent = classify_intent(state["messages"][-1].content)
    if intent == "order_status":
        return "order_agent"
    elif intent == "refund_request":
        return "refund_agent"
    else:
        return "escalation_agent"

# Build graph
graph = StateGraph(SupportState)
graph.add_node("router", route_intent)
graph.add_node("order_agent", order_agent)
graph.add_node("refund_agent", refund_agent)
graph.add_node("escalation_agent", escalation_agent)

graph.add_conditional_edges("router", route_intent)
graph.add_edge("order_agent", END)
graph.add_edge("refund_agent", END)
graph.add_edge("escalation_agent", END)
```

**Production Metrics:**
- First response time: <5s
- Intent classification accuracy: >95%
- Autonomous resolution rate: 70-85%
- Escalation rate: 15-30%
- Customer satisfaction: >4.0/5.0

**Key Design Decisions:**
1. **Stateless agents**: Each request is independent, context from CRM
2. **Human escalation as tool**: Not exception handler
3. **Audit trail**: Log every action for compliance
4. **Graceful degradation**: Fall back to human if agent fails

---

### 11.2 Code Assistant

**Use Case:** Analyze code, generate implementations, review changes, integrate with dev workflow.

**Architecture:** Pipeline + Human-in-Loop

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Analyzer   │────►│  Generator  │────►│  Reviewer   │────►│ Integrator  │
│  (context)  │     │   (code)    │     │  (quality)  │     │   (apply)   │
└─────────────┘     └─────────────┘     └──────┬──────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Human     │
                                        │  Approval   │
                                        └─────────────┘
```

**Key Components:**

| Component | Responsibility | Tools |
|-----------|----------------|-------|
| **Analyzer** | Understand codebase, gather context | read_file, search_code, get_symbols |
| **Generator** | Write code, create tests | write_file, run_tests |
| **Reviewer** | Check quality, security, style | lint, security_scan, type_check |
| **Integrator** | Apply changes, create PRs | git_commit, create_pr |

**Implementation Pattern:**

```python
class CodeAssistantState(TypedDict):
    task: str
    context: dict  # Relevant files, symbols
    generated_code: str
    review_results: dict
    approved: bool

async def analyze(state: CodeAssistantState) -> CodeAssistantState:
    """Gather context from codebase"""
    relevant_files = await search_codebase(state["task"])
    symbols = await get_related_symbols(relevant_files)
    return {"context": {"files": relevant_files, "symbols": symbols}}

async def generate(state: CodeAssistantState) -> CodeAssistantState:
    """Generate code based on context"""
    prompt = build_generation_prompt(state["task"], state["context"])
    code = await llm.generate(prompt)
    return {"generated_code": code}

async def review(state: CodeAssistantState) -> CodeAssistantState:
    """Review generated code"""
    lint_results = await run_linter(state["generated_code"])
    security_results = await security_scan(state["generated_code"])
    type_results = await type_check(state["generated_code"])

    return {"review_results": {
        "lint": lint_results,
        "security": security_results,
        "types": type_results,
        "passed": all_passed(lint_results, security_results, type_results)
    }}

def should_integrate(state: CodeAssistantState) -> str:
    if state["review_results"]["passed"] and state["approved"]:
        return "integrate"
    elif not state["review_results"]["passed"]:
        return "regenerate"
    else:
        return "await_approval"
```

**Production Metrics:**
- Context gathering: <10s
- Code generation: 10-60s depending on complexity
- Review pass rate: >80% on first attempt
- Human approval rate: >90%

**Key Design Decisions:**
1. **Always gather context first**: Don't generate blind
2. **Automated review before human**: Save human time
3. **Iterative refinement**: Loop back on failures
4. **Preserve existing style**: Match codebase conventions

---

### 11.3 Research Agent

**Use Case:** Gather information from multiple sources, synthesize findings, generate reports with citations.

**Architecture:** Parallel Execution + Synthesis

```
                     ┌─────────────────────┐
                     │   Query Planner     │
                     │   (decompose)       │
                     └─────────┬───────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   Web Search    │   │   Doc Search    │   │   API Search    │
│   Agent         │   │   Agent         │   │   Agent         │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                               ▼
                     ┌─────────────────────┐
                     │   Synthesizer       │
                     │   (dedupe, cite)    │
                     └─────────────────────┘
```

**Key Components:**

| Component | Responsibility | Tools |
|-----------|----------------|-------|
| **Query Planner** | Decompose query into sub-queries | query_decomposer |
| **Web Search Agent** | Search public web, news | web_search, news_api |
| **Doc Search Agent** | Search internal docs, PDFs | vector_search, doc_parser |
| **API Search Agent** | Query structured data APIs | api_client, database |
| **Synthesizer** | Deduplicate, resolve conflicts, cite | dedup, conflict_resolver |

**Implementation Pattern:**

```python
import asyncio

class ResearchState(TypedDict):
    query: str
    sub_queries: list[str]
    raw_findings: list[Finding]
    synthesized_report: str
    citations: list[Citation]

async def plan_research(state: ResearchState) -> ResearchState:
    """Decompose query into searchable sub-queries"""
    sub_queries = await llm.decompose_query(state["query"])
    return {"sub_queries": sub_queries}

async def parallel_search(state: ResearchState) -> ResearchState:
    """Run all search agents in parallel"""
    tasks = []
    for sub_query in state["sub_queries"]:
        tasks.extend([
            web_search_agent.search(sub_query),
            doc_search_agent.search(sub_query),
            api_search_agent.search(sub_query)
        ])

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out errors, flatten results
    findings = [r for r in results if not isinstance(r, Exception)]
    return {"raw_findings": findings}

async def synthesize(state: ResearchState) -> ResearchState:
    """Deduplicate, resolve conflicts, generate report"""
    # Deduplicate by semantic similarity
    unique_findings = deduplicate(state["raw_findings"], threshold=0.9)

    # Resolve conflicting information
    resolved = resolve_conflicts(unique_findings)

    # Generate report with citations
    report = await llm.synthesize(
        query=state["query"],
        findings=resolved
    )

    citations = extract_citations(resolved)

    return {"synthesized_report": report, "citations": citations}
```

**Production Metrics:**
- Query decomposition: <2s
- Parallel search: 5-30s (depends on sources)
- Synthesis: 10-30s
- Citation accuracy: >95%
- Deduplication effectiveness: >90%

**Key Design Decisions:**
1. **Parallel execution**: Speed up search phase
2. **Source attribution**: Every claim traced to source
3. **Conflict handling**: Explicitly note disagreements
4. **Freshness awareness**: Prefer recent sources

---

### 11.4 Data Analyst

**Use Case:** Answer questions about data, generate SQL, create visualizations, summarize insights.

**Architecture:** Tool-Heavy Single Agent

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Analyst Agent                       │
│                                                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐│
│  │  SQL    │ │ Schema  │ │  Chart  │ │ Insight │ │ Export  ││
│  │Generator│ │Explorer │ │Generator│ │Summarize│ │  Data   ││
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    Data Lake    │
                    │  (Snowflake,    │
                    │   BigQuery)     │
                    └─────────────────┘
```

**Key Components:**

| Tool | Responsibility | Safety |
|------|----------------|--------|
| **SQL Generator** | Natural language → SQL | Read-only, parameterized |
| **Schema Explorer** | List tables, columns, samples | Metadata only |
| **Chart Generator** | Create visualizations | Limited output size |
| **Insight Summarizer** | Explain findings | No PII exposure |
| **Data Exporter** | Export results | Row limits, approval |

**Implementation Pattern:**

```python
class DataAnalystState(TypedDict):
    question: str
    schema_context: dict
    generated_sql: str
    query_results: list[dict]
    visualization: str  # Chart config
    insights: str

tools = [
    Tool(
        name="explore_schema",
        description="Get table names, columns, and sample data",
        function=explore_schema
    ),
    Tool(
        name="generate_sql",
        description="Convert question to SQL query",
        function=generate_sql
    ),
    Tool(
        name="execute_query",
        description="Run SQL and return results (max 1000 rows)",
        function=execute_query_safe
    ),
    Tool(
        name="create_chart",
        description="Create visualization from query results",
        function=create_chart
    ),
    Tool(
        name="summarize_insights",
        description="Generate human-readable insights from data",
        function=summarize_insights
    )
]

# Safety wrappers
def execute_query_safe(sql: str) -> list[dict]:
    """Execute SQL with safety guards"""
    # 1. Parse and validate SQL
    if not is_read_only(sql):
        raise SecurityError("Only SELECT queries allowed")

    # 2. Add row limit
    sql = add_limit(sql, max_rows=1000)

    # 3. Execute with timeout
    with timeout(30):
        results = database.execute(sql)

    # 4. Redact sensitive columns
    results = redact_pii(results)

    return results
```

**Production Metrics:**
- Schema exploration: <2s
- SQL generation accuracy: >85%
- Query execution: <30s (with timeout)
- Visualization generation: <5s

**Key Design Decisions:**
1. **Read-only access**: No mutations ever
2. **Row limits**: Prevent memory issues
3. **PII redaction**: Automatic sensitive data handling
4. **Query review**: Option for human approval on complex queries

---

### 11.5 Workflow Automator

**Use Case:** Automate multi-step business processes, handle triggers, manage state, recover from failures.

**Architecture:** Event-Driven Orchestrator

```
┌─────────────────────────────────────────────────────────────┐
│                    Event Bus (Kafka/SQS)                      │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Trigger Handler │  │  Step Executor  │  │ Recovery Agent  │
│ (webhooks,cron) │  │  (run actions)  │  │  (retry,alert)  │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   State Store   │
                    │  (PostgreSQL)   │
                    └─────────────────┘
```

**Key Components:**

| Component | Responsibility | Triggers |
|-----------|----------------|----------|
| **Trigger Handler** | Start workflows from events | Webhook, cron, queue, email |
| **Step Executor** | Run individual workflow steps | Previous step completion |
| **Recovery Agent** | Handle failures, retries | Step failure, timeout |
| **State Store** | Persist workflow state | Every state change |

**Workflow Definition:**

```python
@dataclass
class WorkflowStep:
    name: str
    action: Callable
    retry_policy: RetryPolicy
    timeout: int
    on_failure: str  # "retry" | "skip" | "abort" | "human"

@dataclass
class Workflow:
    name: str
    trigger: Trigger
    steps: list[WorkflowStep]
    timeout: int

# Example: Invoice Processing Workflow
invoice_workflow = Workflow(
    name="invoice_processing",
    trigger=Trigger(type="email", filter="subject:Invoice*"),
    steps=[
        WorkflowStep(
            name="extract_data",
            action=extract_invoice_data,
            retry_policy=RetryPolicy(max_attempts=3, backoff="exponential"),
            timeout=60,
            on_failure="human"
        ),
        WorkflowStep(
            name="validate_vendor",
            action=validate_vendor,
            retry_policy=RetryPolicy(max_attempts=2),
            timeout=30,
            on_failure="skip"
        ),
        WorkflowStep(
            name="create_po",
            action=create_purchase_order,
            retry_policy=RetryPolicy(max_attempts=3),
            timeout=60,
            on_failure="abort"
        ),
        WorkflowStep(
            name="notify_approver",
            action=send_approval_request,
            retry_policy=RetryPolicy(max_attempts=5),
            timeout=30,
            on_failure="retry"
        )
    ],
    timeout=3600  # 1 hour total
)
```

**Implementation Pattern:**

```python
class WorkflowEngine:
    def __init__(self, state_store: StateStore, event_bus: EventBus):
        self.state_store = state_store
        self.event_bus = event_bus

    async def execute_workflow(self, workflow: Workflow, trigger_data: dict):
        """Execute workflow with state persistence and recovery"""

        # Create workflow instance
        instance = WorkflowInstance(
            workflow_id=workflow.name,
            trigger_data=trigger_data,
            status="running",
            current_step=0
        )
        await self.state_store.save(instance)

        for i, step in enumerate(workflow.steps):
            instance.current_step = i
            await self.state_store.save(instance)

            try:
                result = await self.execute_step(step, instance)
                instance.step_results[step.name] = result

            except Exception as e:
                if step.on_failure == "retry":
                    await self.schedule_retry(instance, step)
                elif step.on_failure == "skip":
                    instance.step_results[step.name] = {"skipped": True}
                elif step.on_failure == "human":
                    await self.escalate_to_human(instance, step, e)
                    return  # Pause until human resolves
                else:  # abort
                    instance.status = "failed"
                    await self.state_store.save(instance)
                    raise

        instance.status = "completed"
        await self.state_store.save(instance)
```

**Production Metrics:**
- Trigger latency: <1s
- Step execution: varies by action
- Recovery success rate: >95%
- End-to-end completion: >90%

**Key Design Decisions:**
1. **Idempotent steps**: Safe to retry
2. **State persistence**: Resume after crashes
3. **Timeout at every level**: Step, workflow, system
4. **Graceful degradation**: Skip non-critical steps

---

### 11.6 Content Generator

**Use Case:** Create content (blogs, emails, reports), maintain brand voice, iterate based on feedback.

**Architecture:** Generator-Critic Loop

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Planner      │────►│   Generator     │────►│     Critic      │
│   (outline)     │     │   (draft)       │     │   (review)      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         │ Feedback
                                                         │
                                                         ▼
                        ┌─────────────────────────────────────────┐
                        │              Pass?                       │
                        │   YES: Finalize   NO: Revise & Loop     │
                        └─────────────────────────────────────────┘
```

**Key Components:**

| Component | Responsibility | Model Choice |
|-----------|----------------|--------------|
| **Planner** | Create outline, structure | Fast model (GPT-4o-mini) |
| **Generator** | Write content sections | Best writing model (Claude Sonnet 4.5) |
| **Critic** | Review quality, voice, accuracy | Strong reasoning (o1, Opus 4.5) |
| **Finalizer** | Format, add metadata | Fast model |

**Implementation Pattern:**

```python
class ContentState(TypedDict):
    brief: str  # What to write
    outline: list[str]  # Section structure
    drafts: dict[str, str]  # Section drafts
    feedback: list[dict]  # Critic feedback
    revision_count: int
    final_content: str

async def plan_content(state: ContentState) -> ContentState:
    """Create content outline"""
    outline = await planner_llm.create_outline(
        brief=state["brief"],
        guidelines=brand_guidelines
    )
    return {"outline": outline}

async def generate_draft(state: ContentState) -> ContentState:
    """Generate content for each section"""
    drafts = {}
    for section in state["outline"]:
        draft = await writer_llm.write_section(
            section=section,
            brief=state["brief"],
            tone=brand_tone,
            previous_sections=drafts
        )
        drafts[section] = draft
    return {"drafts": drafts}

async def critique_content(state: ContentState) -> ContentState:
    """Review and provide feedback"""
    feedback = await critic_llm.review(
        content=state["drafts"],
        criteria={
            "brand_voice": brand_guidelines,
            "accuracy": fact_check_sources,
            "engagement": readability_targets,
            "seo": keyword_requirements
        }
    )
    return {"feedback": feedback}

def should_revise(state: ContentState) -> str:
    """Determine if revision needed"""
    if state["revision_count"] >= MAX_REVISIONS:
        return "finalize"  # Cap revisions

    if all(f["score"] >= 0.8 for f in state["feedback"]):
        return "finalize"  # Quality threshold met

    return "revise"

async def revise_content(state: ContentState) -> ContentState:
    """Apply feedback and regenerate"""
    improved_drafts = {}
    for section, draft in state["drafts"].items():
        section_feedback = [f for f in state["feedback"]
                          if f["section"] == section]

        improved = await writer_llm.revise(
            draft=draft,
            feedback=section_feedback
        )
        improved_drafts[section] = improved

    return {
        "drafts": improved_drafts,
        "revision_count": state["revision_count"] + 1
    }
```

**Production Metrics:**
- Planning: <10s
- Draft generation: 30-120s (depends on length)
- Critique: <20s
- Average revisions: 1-2
- Human approval rate: >85%

**Key Design Decisions:**
1. **Separate critic model**: Avoids self-confirmation bias
2. **Revision cap**: Prevent infinite loops
3. **Section-by-section**: Easier to iterate
4. **Brand guidelines as context**: Consistent voice

---

### 11.7 Blueprint Selection Guide

| Use Case | Architecture | When to Use |
|----------|--------------|-------------|
| **Customer Support** | Coordinator + Workers | Multiple domains, routing needed |
| **Code Assistant** | Pipeline + HITL | Sequential steps, human approval |
| **Research Agent** | Parallel + Synthesis | Multiple sources, time-sensitive |
| **Data Analyst** | Tool-Heavy Single | Complex tools, single domain |
| **Workflow Automator** | Event-Driven | Long-running, async, recovery |
| **Content Generator** | Generator-Critic | Creative, iterative refinement |

**Decision Factors:**

```
Need parallel execution?
  YES → Research Agent pattern
  NO  ↓

Need human approval?
  YES → Pipeline + HITL
  NO  ↓

Multiple specialized domains?
  YES → Coordinator + Workers
  NO  ↓

Long-running with recovery?
  YES → Event-Driven
  NO  ↓

Iterative quality improvement?
  YES → Generator-Critic
  NO  → Tool-Heavy Single Agent
```

---

## Resources

### Official Documentation
- LangGraph Multi-Agent Tutorial: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/
- Multi-Agent Concepts: https://github.com/langchain-ai/langgraph/blob/main/docs/docs/concepts/multi_agent.md
- Benchmarking Post: https://blog.langchain.com/benchmarking-multi-agent-architectures/

### Libraries
- `langgraph-supervisor`: https://github.com/langchain-ai/langgraph-supervisor-py
- `langgraph-swarm`: https://github.com/langchain-ai/langgraph-swarm-py

### Blog Posts
- "LangGraph: Multi-Agent Workflows": https://blog.langchain.com/langgraph-multi-agent-workflows/
- "How and When to Build Multi-Agent Systems": https://blog.langchain.com/how-and-when-to-build-multi-agent-systems/
- "Command Tool Announcement": https://blog.langchain.com/command-a-new-tool-for-multi-agent-architectures-in-langgraph/

---

**Last Updated**: 2025-12-25
**Maintained By**: Research synthesis from LangChain documentation, production case studies, academic research (arXiv 2025), and community best practices
**Key Sources**: LangChain benchmarks, Anthropic multi-agent research, Google ADK, arXiv:2512.08296, arXiv:2510.05174
