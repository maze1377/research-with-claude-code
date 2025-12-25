# Agentic Framework Comparison

**Consolidated analysis of LangGraph, CrewAI, AutoGPT, and design patterns**

---

## Framework Overview

| Framework | Type | Best For | Complexity |
|-----------|------|----------|------------|
| **LangGraph** | Graph-based | Complex workflows, state machines | Medium-High |
| **CrewAI** | Role-based | Team simulations, defined roles | Medium |
| **AutoGPT** | Autonomous | Fully autonomous, minimal oversight | High |
| **OpenManus** | Open-source | Research, customization | High |

---

## LangGraph

### Core Architecture
- **Graph-based**: Nodes = agents/functions, Edges = transitions
- **State Management**: Centralized TypedDict state
- **Persistence**: Built-in checkpointing for recovery

### Key Patterns (2025)

| Pattern | Use Case | Nodes | Complexity |
|---------|----------|-------|------------|
| Sequential | Simple pipeline | 2-3 | Low |
| Router | Query classification | 3-5 | Medium |
| Parallel | Independent tasks | 4+ | Medium |
| Supervisor | Central control | 3-10 | High |
| Swarm | Dynamic handoffs | 5+ | High |

### Supervisor Pattern
```
                    ┌─────────────┐
                    │  Supervisor │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ↓               ↓               ↓
      ┌─────────┐    ┌─────────┐    ┌─────────┐
      │ Agent A │    │ Agent B │    │ Agent C │
      └─────────┘    └─────────┘    └─────────┘
```
- Supervisor routes to specialists
- Collects and aggregates results
- Makes final decisions

### Swarm Pattern (New in 2025)
```
Agent A ──handoff──→ Agent B ──handoff──→ Agent C
    ↑                                        │
    └────────────────────────────────────────┘
```
- Agents hand off to each other dynamically
- No central coordinator
- 40% faster than Supervisor for certain tasks

### Strengths
- Full control over flow logic
- Built-in persistence and recovery
- Excellent debugging (graph visualization)
- Strong typing with TypedDict

### Limitations
- Steeper learning curve
- More boilerplate than simpler frameworks
- Requires explicit state management

---

## CrewAI

### Core Architecture
- **Role-based**: Agents defined by role, goal, backstory
- **Process**: Sequential, Hierarchical, or Parallel
- **Tasks**: Assigned to agents with expected outputs

### Agent Definition
```
Agent = {
    role: "Senior Researcher",
    goal: "Find comprehensive information",
    backstory: "Expert with 10 years experience",
    tools: [search, scrape, analyze]
}
```

### Process Types

| Process | Flow | Best For |
|---------|------|----------|
| Sequential | A → B → C | Linear pipelines |
| Hierarchical | Manager → Workers | Complex projects |
| Parallel | A, B, C simultaneously | Independent tasks |

### Strengths
- Intuitive role-based design
- Quick setup for standard patterns
- Good for team simulations
- 12M+ daily executions (production proven)

### Limitations
- Less flexibility than LangGraph
- Harder to customize flow logic
- Limited state management

---

## AutoGPT

### Core Architecture
- **Fully Autonomous**: Minimal human intervention
- **Goal-Driven**: Set goal, agent figures out steps
- **Persistent**: Maintains long-term memory

### Execution Loop
```
1. Set Goal
2. Think (plan next action)
3. Act (execute tool/command)
4. Observe (check result)
5. Repeat until goal achieved
```

### Strengths
- True autonomy for defined goals
- Long-running task support
- Self-correcting behavior

### Limitations
- Resource intensive (many API calls)
- Prone to loops without proper constraints
- Harder to debug/control
- Not recommended for production without guardrails

---

## Design Patterns Catalog

### 1. Single-Agent Pattern
```
Input → Agent → Output
```
**Use**: Simple tasks, single domain

### 2. Sequential Pattern
```
Input → Agent A → Agent B → Agent C → Output
```
**Use**: Pipeline processing, staged refinement

### 3. Parallel Pattern
```
       ┌→ Agent A ─┐
Input ─┼→ Agent B ─┼→ Merge → Output
       └→ Agent C ─┘
```
**Use**: Independent subtasks, faster execution

### 4. Loop Pattern
```
Input → Agent → Validate → [Pass] → Output
              ↓
           [Fail]
              ↓
          Feedback ───→ Agent
```
**Use**: Quality-critical tasks, iterative improvement

### 5. Review & Critique Pattern
```
Input → Generator → Critic → [Good] → Output
                      ↓
                   [Bad] → Generator (with feedback)
```
**Use**: High-accuracy requirements, creative tasks

### 6. Coordinator Pattern
```
              ┌─────────────┐
Input ──────→ │ Coordinator │
              └──────┬──────┘
      ┌───────────────┼───────────────┐
      ↓               ↓               ↓
 Specialist A   Specialist B   Specialist C
```
**Use**: Multi-domain tasks, dynamic routing

### 7. Hierarchical Pattern
```
                 Manager
              /     |     \
         Sub-Mgr  Sub-Mgr  Sub-Mgr
         /    \      |      /    \
       W1    W2     W3    W4    W5
```
**Use**: Large projects, organizational structure

### 8. Human-in-the-Loop Pattern
```
Agent → Action → [High Risk?] → Human Approval → Execute
                      ↓
                  [Low Risk] → Auto Execute
```
**Use**: Critical operations, compliance requirements

---

## Pattern Selection Guide

| Requirement | Recommended Pattern |
|-------------|---------------------|
| Simple task, single domain | Single-Agent |
| Pipeline with stages | Sequential |
| Independent subtasks | Parallel |
| Quality improvement | Loop + Critique |
| Multiple domains | Coordinator |
| Large project | Hierarchical |
| Critical operations | Human-in-the-Loop |
| Dynamic routing | Swarm |

### Decision Tree
```
Is task simple (1-2 steps)?
├── Yes → Single-Agent
└── No → Are subtasks independent?
    ├── Yes → Parallel
    └── No → Is there a natural sequence?
        ├── Yes → Sequential
        └── No → How many domains?
            ├── 2-3 → Coordinator
            └── 4+ → Hierarchical or Swarm
```

---

## Framework Selection Guide

| Factor | LangGraph | CrewAI | AutoGPT |
|--------|-----------|--------|---------|
| **Learning Curve** | High | Medium | Medium |
| **Flexibility** | Highest | Medium | Low |
| **Production Ready** | Yes | Yes | Caution |
| **State Management** | Excellent | Basic | Basic |
| **Debugging** | Excellent | Good | Difficult |
| **Autonomy** | Controlled | Controlled | Full |
| **Cost Efficiency** | High | Medium | Low |

### When to Use Each

**LangGraph**:
- Complex, custom workflows
- Need fine-grained control
- State persistence required
- Multi-agent orchestration

**CrewAI**:
- Role-based team simulations
- Standard patterns (research, analysis)
- Quick prototyping
- Team collaboration metaphor fits

**AutoGPT**:
- Research and experimentation
- Long-running autonomous tasks
- When human oversight is feasible
- NOT recommended for production without guardrails

---

## Industry Trends (2025)

### 1. MCP Protocol Adoption
- Standard for tool integration
- OpenAI, Google, Anthropic all support
- 2,000+ servers available

### 2. Swarm Patterns Rising
- OpenAI's Swarm library
- LangGraph Swarm support
- 40% faster for dynamic tasks

### 3. Memory Systems
- Mem0, GraphRAG integration
- Long-term context preservation
- 26% accuracy improvements

### 4. Multi-Model Architectures
- Different models for different agents
- Cost optimization through cascading
- Specialized models for specialized tasks

---

## Quick Reference

### Framework at a Glance
```
LangGraph  = Maximum control + Graph-based + Production-ready
CrewAI     = Role-based teams + Quick setup + Good for prototyping
AutoGPT    = Full autonomy + Research-focused + Use with caution
```

### Pattern at a Glance
```
Simple         → Single-Agent
Pipeline       → Sequential
Independent    → Parallel
Quality-first  → Loop + Critique
Multi-domain   → Coordinator/Supervisor
Large project  → Hierarchical
Dynamic        → Swarm
Critical       → Human-in-the-Loop
```

---

## Related Documents

- [multi-agent-patterns.md](multi-agent-patterns.md) - Detailed multi-agent patterns
- [patterns-and-antipatterns.md](patterns-and-antipatterns.md) - Common failures
- [topics.md](topics.md) - Q7-12 for architecture questions
- [2025-updates.md](2025-updates.md) - Latest 2025 developments

---

**Document Version**: 1.0 (Consolidated from findings-*.md)
**Last Updated**: December 2025
