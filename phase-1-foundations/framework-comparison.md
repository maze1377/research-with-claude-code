# Agentic Framework Comparison

**Consolidated analysis of LangGraph, CrewAI, AutoGPT, OpenAI Agents SDK, Claude Agent SDK, and design patterns**

**Last Updated:** 2025-12-26

---

## Framework Overview

| Framework | Type | Best For | Complexity | Version |
|-----------|------|----------|------------|---------|
| **LangGraph** | Graph-based | Complex workflows, state machines | Medium-High | v1.0 (Oct 2025) |
| **CrewAI** | Role-based | Team simulations, defined roles | Medium | 2025 IA40 |
| **OpenAI Agents SDK** | Primitives | Quick prototypes, production agents | Low | Mar 2025 |
| **Claude Agent SDK** | Skills-based | Long-running tasks, complex workflows | Medium | Dec 2025 |
| **AutoGen v0.4** | Layered | Enterprise, Azure integration | Medium | v0.4 |
| **Semantic Kernel** | Plugin-based | Enterprise, Microsoft ecosystem | Medium | 2025 |
| **DSPy** | Programmatic | Research, prompt optimization | Medium | 2025 |
| **Pydantic AI** | Type-safe | Python apps, structured outputs | Low-Medium | 2025 |
| **AWS Bedrock Agents** | Managed | AWS integration, MCP | Medium | AgentCore |

**Market Context (December 2025):**
- 78% of organizations using AI, 85% have agents in at least one workflow
- 23% scaling agentic AI, 39% experimenting
- AI agents: $0.25-0.50/interaction (vs $3-6 humans)
- Break-even: ~50,000 interactions annually

---

## LangGraph

### Core Architecture (v1.0 - October 2025)
- **Graph-based**: Nodes = agents/functions, Edges = transitions
- **State Management**: Centralized TypedDict state
- **Persistence**: Built-in checkpointing (Checkpoint 3.0 interface)
- **Cyclic Graphs**: Support for revisiting previous steps (not just linear)
- **Human-in-the-Loop**: Pause, inspect, and modify execution mid-workflow

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

## AutoGPT / AutoGen

### AutoGPT Core Architecture
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

### AutoGen v0.4 (Complete Redesign - 2025)

**Three-Layer Architecture:**
| Layer | Purpose |
|-------|---------|
| **Core** | Event-driven foundation, async messaging |
| **AgentChat** | High-level task-oriented API |
| **Extensions** | Third-party integrations, pluggable components |

**Key Features:**
- Asynchronous messaging (event-driven + request/response)
- Modular design with pluggable agents, tools, memory, models
- Built-in observability with OpenTelemetry integration
- AutoGen Studio: Visual team builder, real-time updates

**Migration Note:** Microsoft is unifying AutoGen + Semantic Kernel into **Microsoft Agent Framework**. AutoGen receives bug fixes only (no new features). Plan migration for new projects.

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

## OpenAI Agents SDK

### Core Architecture (March 2025)
**Four Core Primitives:**
| Primitive | Purpose |
|-----------|---------|
| **Agents** | Instruction-driven entities with tools and models |
| **Tools** | Python/TS functions with automatic schema generation |
| **Handoffs** | Native agent-to-agent task delegation |
| **Guardrails** | Input/output validation to constrain behavior |

### Key Features
- **Minimalist design**: Just 4 primitives, no overhead
- **Provider-agnostic**: Works with 100+ LLMs via Chat Completions API
- **Automatic tracing**: No custom instrumentation required
- **Responses API**: Unified tool-use with simpler streaming

### Production Features
```
Runner.run(agent, input)
    ↓
┌─────────────────────────────────┐
│   Trace (automatic)             │
│   ├─ Agent decisions            │
│   ├─ Tool calls                 │
│   ├─ Handoffs                   │
│   └─ Guardrail checks           │
└─────────────────────────────────┘
```

### Strengths
- Fastest time-to-prototype
- Built-in observability
- Works with any LLM
- Production-ready patterns

### Limitations
- Less flexibility than LangGraph for complex state
- Newer ecosystem (fewer examples)
- Handoffs simpler than full graph control

---

## AWS Bedrock AgentCore

### Core Components
| Component | Purpose |
|-----------|---------|
| **Gateway** | Converts REST APIs to MCP tools |
| **Identity** | Ingress/egress authentication |
| **Memory** | Long-term memory capabilities |
| **Runtime** | Execution with automatic scaling |
| **Observability** | Detailed metrics capture |

### Key Features
- MCP-standardized gateway for tool integration
- Compatible with LangGraph, CrewAI, AWS Strands
- Multi-agent collaboration built-in
- Enterprise security and compliance

### Strengths
- Native AWS integration
- Managed infrastructure
- Enterprise-grade security
- Multi-agent orchestration

### Limitations
- AWS vendor lock-in
- More complex setup than SDK approaches
- Cost varies with usage

---

## Google Vertex AI Agent Engine

### Core Features (December 2025)
- **Agent Designer**: Low-code visual designer (preview)
- **Code Execution**: Isolated sandbox environments
- **A2A Protocol**: Agent-to-agent coordination
- **Memory Bank**: Dynamic long-term memory generation
- **Tool Governance**: Admin-managed tool registries

### Gemini 3 Pro Integration
- Full ADK (Agent Development Kit) compatibility
- Long-horizon planning capabilities
- Production-ready agent building

### Strengths
- Google Cloud integration
- Low-code options
- Built-in tool governance

### Limitations
- GCP dependency
- Newer compared to LangGraph/CrewAI

---

## Emerging Frameworks (2025)

### Claude Agent SDK (Anthropic)

Built on Claude Code's production agent harness. Key primitives:

| Component | Purpose |
|-----------|---------|
| **Skills** | Markdown files teaching Claude domain expertise |
| **Hooks** | Shell commands on lifecycle events (format on edit, notify on idle) |
| **Plugins** | Package and share skills + MCP servers + hooks |
| **Context Compaction** | Auto-summarize when context fills |

**Strengths:**
- Planning-first architecture optimized for complex tasks
- Native MCP integration (2000+ servers)
- Skills system for domain expertise injection
- Production-tested in Claude Code

**Best For:** Long-running development tasks, complex multi-step workflows

### Semantic Kernel (Microsoft)

Enterprise-focused SDK for building AI agents.

| Feature | Capability |
|---------|------------|
| **Plugins** | Reusable function collections |
| **Planners** | Automatic task decomposition |
| **Memory** | Vector stores, semantic memory |
| **Connectors** | Azure, M365, custom |

**Strengths:**
- Enterprise integration (Azure, M365)
- Strong .NET and Python support
- Built-in planning and memory
- Microsoft backing

**Best For:** Enterprise applications, Microsoft ecosystem integration

### DSPy (Stanford)

Programmatic approach to prompt optimization.

```
Signature: Input → Output with constraints
Module: Composable prompt components
Compiler: Automatic prompt optimization
```

**Strengths:**
- Automatic prompt optimization
- Composable modules
- Reproducible results
- Research-backed methodology

**Best For:** Research applications, systematic prompt engineering

### Pydantic AI (Pydantic)

Type-safe agent framework using Pydantic models.

```python
from pydantic_ai import Agent

agent = Agent(
    system_prompt="You are helpful assistant.",
    model="openai:gpt-4o",
    result_type=ResponseModel  # Enforced output schema
)
```

**Strengths:**
- Type safety with Pydantic validation
- Familiar Python patterns
- Clean API design
- Built-in structured outputs

**Best For:** Python-heavy teams, type-safe production systems

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

| Factor | LangGraph | CrewAI | OpenAI SDK | AutoGen v0.4 | Bedrock |
|--------|-----------|--------|------------|--------------|---------|
| **Learning Curve** | High | Medium | Low | Medium | Medium |
| **Flexibility** | Highest | Medium | Medium | High | Medium |
| **Production Ready** | Yes | Yes | Yes | Yes | Yes |
| **State Management** | Excellent | Basic | Sessions | Good | Managed |
| **Debugging** | Excellent | Good | Good | Excellent | Good |
| **Multi-Agent** | Full control | Role-based | Handoffs | Full control | Built-in |
| **Cost Efficiency** | High | Medium | High | Medium | Varies |

### When to Use Each

**LangGraph**:
- Complex, custom workflows
- Need fine-grained control
- State persistence required
- Multi-agent orchestration
- Cyclic/adaptive workflows

**CrewAI**:
- Role-based team simulations
- Standard patterns (research, analysis)
- Quick prototyping
- Team collaboration metaphor fits

**OpenAI Agents SDK**:
- Fastest time-to-prototype
- Production with minimal config
- Provider-agnostic needed
- Simple multi-agent handoffs

**AutoGen v0.4 / MS Agent Framework**:
- Azure enterprise integration
- Comprehensive observability
- Large-scale distributed agents
- Visual prototyping (AutoGen Studio)

**AWS Bedrock AgentCore**:
- AWS infrastructure already in use
- Managed MCP gateway needed
- Enterprise compliance requirements

**AutoGPT**:
- Research and experimentation
- Long-running autonomous tasks
- When human oversight is feasible
- NOT recommended for production without guardrails

---

## Industry Trends (December 2025)

### 1. Protocol Standardization
- **MCP**: 10,000+ active servers, donated to Linux Foundation (Dec 2025)
- **A2A**: Google's Agent-to-Agent protocol, 50+ partners
- **AGENTS.md**: 60,000+ open source projects adopted
- **Agentic AI Foundation**: MCP, goose, AGENTS.md unified under Linux Foundation

### 2. Swarm Outperforms Supervisor
**LangChain Benchmark Results:**
| Metric | Swarm | Supervisor |
|--------|-------|------------|
| Token efficiency | Higher | Lower (translation tax) |
| Distractor domains | Better | Worse |
| Direct user response | Yes | Through supervisor |

### 3. Memory Systems
- Mem0: $24M Series A, 26% accuracy boost, 91% lower latency
- GraphRAG: Microsoft knowledge graph + RAG
- Memory Bank: Vertex AI dynamic memory generation

### 4. Multi-Model Architectures
- Different models for different agents
- Cost optimization through cascading
- Specialized models for specialized tasks

### 5. Analytics Framework Performance
| Framework | Accuracy | Efficiency | Tokens |
|-----------|----------|------------|--------|
| Swarm | 90% | 60% | ~1,000 |
| CrewAI | 87% | 21% | ~4,500 |
| LangChain | 78% | 42% | Varies |

### 6. Enterprise Adoption
- 23% scaling agentic AI across enterprise
- 39% experimenting with agents
- CrewAI ranked #7 on 2025 IA Enablers List (with $20M vs billions for peers)
- Technology, media, healthcare lead adoption

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
