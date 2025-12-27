# Cross-Framework Migration Guide

**Agentic AI Developer Onboarding Guide**

**Purpose:** Comprehensive guidance for teams evaluating, selecting, or migrating between AI agent frameworks. Covers decision criteria, migration patterns, abstraction strategies, and production lessons learned.

**Last Updated:** 2025-12-27

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Framework Selection Criteria](#framework-selection-criteria)
3. [Migration Patterns](#migration-patterns)
4. [Abstraction Strategies](#abstraction-strategies)
5. [Common Pitfalls](#common-pitfalls)
6. [Case Studies](#case-studies)
7. [Decision Framework](#decision-framework)
8. [Migration Checklists](#migration-checklists)

---

## Executive Summary

The AI agent framework landscape underwent significant transformation in 2024-2025. Key findings from production migrations:

| Metric | Value | Source |
|--------|-------|--------|
| Response latency improvement | 66% | Grid Dynamics LangGraph to Temporal |
| API cost reduction | 62% | Framework optimization case studies |
| Token consumption reduction | 40-60% | Context management optimization |
| Agentic AI project failure rate | 40% | Inadequate production planning |
| Organizations testing agents | 72% | Deloitte 2025 |
| Organizations in production | 11% | Deloitte 2025 |

**Key Insight:** The gap between testing (72%) and production deployment (11%) highlights that framework selection and migration are critical success factors.

### Framework Migration Landscape

```
LangChain Legacy Agents      AutoGen + Semantic Kernel
        |                              |
        v                              v
   LangGraph 1.0 GA  <------>  Microsoft Agent Framework
        ^                              ^
        |                              |
     CrewAI  <--------------------> Hybrid Approaches
        |
        v
   OpenAI Agents SDK / Custom Solutions
```

---

## Framework Selection Criteria

### 1. Use Case Alignment

Each framework excels in specific scenarios. Match your use case to framework strengths:

| Framework | Best For | Not Ideal For |
|-----------|----------|---------------|
| **LangGraph** | Complex workflows, state machines, cyclic graphs, fine-grained control | Simple prototypes, rapid experimentation |
| **CrewAI** | Role-based team simulations, standard patterns, quick prototyping | Complex conditional logic, custom state management |
| **OpenAI Agents SDK** | Fast prototypes, provider-agnostic agents, simple multi-agent handoffs | Complex state persistence, enterprise compliance |
| **AutoGen v0.4** | Conversational multi-agent, group chat dynamics, research exploration | Deterministic workflows, high-volume production |
| **MS Agent Framework** | Azure enterprise, unified orchestration, compliance requirements | Non-Microsoft ecosystems, rapid prototyping |
| **Claude Agent SDK** | Long-running tasks, complex multi-step workflows, skill-based expertise | Simple single-turn agents |

### Framework Decision Matrix

| Criterion | LangGraph | CrewAI | OpenAI SDK | MS Agent | Claude SDK |
|-----------|-----------|--------|------------|----------|------------|
| Learning Curve | High | Medium | Low | Medium | Medium |
| Flexibility | Highest | Medium | Medium | High | High |
| State Management | Excellent | Basic | Sessions | Excellent | Checkpoints |
| Observability | Excellent | Good | Good | Excellent | Good |
| Production Ready | Yes | Yes | Yes | Yes | Yes |
| Multi-Agent | Full control | Role-based | Handoffs | Full control | Subagents |

### 2. Team Expertise Considerations

**Assess Your Team's Skills:**

```
Skill Level Assessment:
+------------------------------------------+
| Infrastructure Engineers    | LangGraph, MS Agent Framework |
| ML/AI Specialists          | AutoGen, DSPy                 |
| Full-Stack Developers      | OpenAI SDK, CrewAI            |
| Business Analysts          | Visual Builders, CrewAI       |
| Enterprise Architects      | MS Agent Framework, Bedrock   |
+------------------------------------------+
```

**Team Skill Alignment:**

| Team Profile | Recommended Framework | Rationale |
|--------------|----------------------|-----------|
| DevOps-heavy | LangGraph | Explicit state, graph visualization |
| Rapid prototyping | CrewAI or OpenAI SDK | Minimal boilerplate, fast iteration |
| Microsoft ecosystem | MS Agent Framework | Azure integration, familiar patterns |
| Research-focused | AutoGen, DSPy | Flexible, experimental |
| Python experts | Pydantic AI, LangGraph | Type-safe, Pythonic APIs |

### 3. Ecosystem Requirements

**Integration Checklist:**

| Requirement | LangGraph | CrewAI | OpenAI SDK | MS Agent |
|-------------|-----------|--------|------------|----------|
| MCP Support | Yes | Yes | Yes | Yes |
| LangSmith Integration | Native | Plugin | External | Native |
| Vector DB Support | All major | All major | Via tools | Azure + third-party |
| Custom Tools | Python functions | Python classes | Decorated functions | Plugins |
| API Gateway | Custom | Custom | OpenAI native | Azure API Management |

**Community Size (December 2025):**

| Framework | GitHub Stars | npm/PyPI Downloads | Active Contributors |
|-----------|-------------|-------------------|---------------------|
| LangChain/LangGraph | 100K+ | 97M+ monthly | 500+ |
| CrewAI | 25K+ | Growing rapidly | 100+ |
| OpenAI SDK | N/A | OpenAI-managed | Core team |
| AutoGen | 35K+ | Moderate | 200+ |

### 4. Long-term Maintenance Considerations

**Vendor Stability Assessment:**

| Framework | Backing | Stability | Migration Risk |
|-----------|---------|-----------|----------------|
| LangGraph | LangChain Inc. (Series A $25M) | High - 1.0 GA | Low |
| CrewAI | CrewAI Inc. ($20M funding) | Medium - Active development | Medium |
| OpenAI SDK | OpenAI | High - Core product | Low |
| AutoGen | Microsoft | Medium - Bug fixes only, merging to Agent Framework | High |
| MS Agent Framework | Microsoft | High - Enterprise focus | Low |

**Deprecation Warnings (2025):**

| Framework/API | Status | Sunset Date | Migration Target |
|---------------|--------|-------------|------------------|
| LangChain `initialize_agent` | Deprecated | Already deprecated | LangGraph |
| OpenAI Assistants API | Deprecated | August 26, 2026 | Responses API / Agents SDK |
| AutoGen (new features) | Bug fixes only | Ongoing | MS Agent Framework |
| Semantic Kernel (standalone) | Merging | Ongoing | MS Agent Framework |

### 5. Cost Considerations

**Framework Cost Components:**

| Component | LangGraph | CrewAI | OpenAI SDK | MS Agent |
|-----------|-----------|--------|------------|----------|
| Licensing | Open source | Open source | Free SDK, API costs | Open source |
| Infrastructure | Self-managed | Self-managed | OpenAI-managed | Azure or self |
| Checkpoint Storage | PostgreSQL, Redis | External | Managed | Azure SQL, Cosmos |
| Observability | LangSmith ($$$) | External | Built-in | Azure Monitor |

**Total Cost of Ownership Estimate (1M monthly requests):**

| Framework | Infrastructure | API Costs | Observability | Total |
|-----------|---------------|-----------|---------------|-------|
| LangGraph + LangSmith | $500/mo | $2,000/mo | $200/mo | ~$2,700/mo |
| CrewAI + Langfuse | $300/mo | $2,500/mo | $100/mo | ~$2,900/mo |
| OpenAI SDK (managed) | $0 | $3,000/mo | Included | ~$3,000/mo |
| MS Agent (Azure) | $800/mo | $2,000/mo | $150/mo | ~$2,950/mo |

---

## Migration Patterns

### Pattern 1: LangChain to LangGraph (Most Common)

This is the most frequently executed migration path as organizations move from prototype to production.

**Why Migrate:**
- Legacy `initialize_agent` and `AgentExecutor` deprecated in LangChain v0.2
- Need for explicit state management and persistence
- Requirement for human-in-the-loop workflows
- Better debugging and observability

**Migration Steps:**

**Step 1: Inventory Legacy Code**

```python
# LEGACY: Find all deprecated imports
from langchain.agents import initialize_agent, AgentExecutor, Tool
from langchain.agents.types import AgentType

# Search your codebase for:
# - initialize_agent
# - AgentExecutor
# - AgentType.ZERO_SHOT_REACT_DESCRIPTION
# - AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
```

**Step 2: Define State Schema**

```python
# NEW: Explicit state definition with LangGraph
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Explicit state schema replaces implicit AgentExecutor state."""
    messages: Annotated[list, add_messages]
    context: dict
    tool_results: list
    iteration_count: int
```

**Step 3: Convert Agent to Graph**

```python
# LEGACY: Single function call
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
result = agent.invoke({"input": user_query})

# NEW: LangGraph equivalent
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver

# Create the agent with persistence
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=PostgresSaver(connection_string)  # Persistence!
)

# Compile once at startup (cache this!)
graph = agent.compile()

# Invoke with thread_id for state persistence
config = {"configurable": {"thread_id": user_session_id}}
result = graph.invoke({"messages": [HumanMessage(content=user_query)]}, config)
```

**Step 4: Add Production Features**

```python
from langgraph.graph import StateGraph, END

# Build custom graph with production features
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("guardrails", guardrail_node)  # NEW: Safety checks

# Add conditional routing
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
        "human_review": "human_approval"  # NEW: Human-in-the-loop
    }
)

# Add human approval node
workflow.add_node("human_approval", human_review_node)
```

**Migration Checklist:**

- [ ] Pin langchain>=0.2, langgraph>=0.2
- [ ] Replace all `initialize_agent` with `create_react_agent`
- [ ] Define explicit TypedDict state schemas
- [ ] Configure checkpointer (PostgresSaver for production)
- [ ] Cache compiled graph at application startup
- [ ] Add health probe endpoint (invoke with ping/pong)
- [ ] Configure LangSmith tracing (LANGSMITH_TRACING_V2=true)
- [ ] Export graph visualization for documentation
- [ ] Run regression tests against baseline
- [ ] Deploy with canary (10% traffic first)

---

### Pattern 2: CrewAI to LangGraph (Role-Based to Graph-Based)

Migrate when CrewAI's role-based abstractions become constraining.

**When to Migrate:**
- Need conditional routing beyond fixed processes
- Require fine-grained state management
- Want deterministic, auditable workflows
- Scaling beyond CrewAI's orchestration model

**Mapping CrewAI Concepts to LangGraph:**

| CrewAI Concept | LangGraph Equivalent |
|----------------|---------------------|
| Agent (role, goal, backstory) | Node with system prompt |
| Task | Node invocation |
| Crew (sequential/hierarchical) | Graph structure |
| Process.sequential | Linear edge connections |
| Process.hierarchical | Supervisor pattern |
| Tools | Tool functions bound to nodes |

**Migration Example:**

```python
# CREWAI: Role-based definition
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Research Analyst",
    goal="Find comprehensive market data",
    backstory="Expert researcher with 10 years experience",
    tools=[search_tool, scrape_tool]
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging reports",
    backstory="Award-winning technical writer",
    tools=[write_tool]
)

research_task = Task(
    description="Research the market for {topic}",
    agent=researcher,
    expected_output="Detailed research findings"
)

write_task = Task(
    description="Write report based on research",
    agent=writer,
    expected_output="Polished market report"
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential
)

# LANGGRAPH: Graph-based definition
from langgraph.graph import StateGraph, END

class ResearchState(TypedDict):
    topic: str
    research_findings: str
    final_report: str
    messages: Annotated[list, add_messages]

def researcher_node(state: ResearchState) -> ResearchState:
    """Researcher agent as a graph node."""
    prompt = """You are a Research Analyst with 10 years experience.
    Goal: Find comprehensive market data.

    Research the following topic: {topic}
    Provide detailed research findings."""

    result = llm_with_tools.invoke(
        prompt.format(topic=state["topic"]),
        tools=[search_tool, scrape_tool]
    )
    return {"research_findings": result.content}

def writer_node(state: ResearchState) -> ResearchState:
    """Writer agent as a graph node."""
    prompt = """You are an award-winning Content Writer.
    Goal: Create engaging reports.

    Based on these research findings:
    {research_findings}

    Write a polished market report."""

    result = llm.invoke(prompt.format(
        research_findings=state["research_findings"]
    ))
    return {"final_report": result.content}

# Build the graph (equivalent to sequential Crew)
workflow = StateGraph(ResearchState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)
workflow.set_entry_point("researcher")

# Now you can add conditional logic CrewAI cannot express
workflow.add_conditional_edges(
    "researcher",
    lambda state: "needs_more" if state["research_findings"] == "" else "continue",
    {
        "needs_more": "researcher",  # Retry with different approach
        "continue": "writer"
    }
)
```

**CrewAI to LangGraph Migration Checklist:**

- [ ] Map each Agent to a node function
- [ ] Convert role/goal/backstory to system prompts
- [ ] Define explicit state schema
- [ ] Map sequential process to linear edges
- [ ] Map hierarchical process to supervisor pattern
- [ ] Externalize any implicit delegation logic
- [ ] Add conditional routing for dynamic behavior
- [ ] Implement checkpointing for persistence
- [ ] Add error handling and retry logic
- [ ] Test with original CrewAI inputs/outputs

---

### Pattern 3: Custom Solutions to OpenAI SDK (Simplification)

Migrate custom orchestration to OpenAI SDK when simplicity outweighs flexibility.

**When to Migrate:**
- Custom code is mostly reimplementing SDK features
- Team bandwidth is limited
- OpenAI models are primary (or only) LLM
- Built-in tracing and guardrails are sufficient

**Migration Example:**

```python
# CUSTOM: Hand-rolled orchestration
class CustomAgent:
    def __init__(self, tools, model):
        self.tools = tools
        self.model = model
        self.messages = []

    def run(self, query):
        self.messages.append({"role": "user", "content": query})

        while True:
            response = self.model.chat.completions.create(
                messages=self.messages,
                tools=[self._format_tool(t) for t in self.tools]
            )

            if response.choices[0].finish_reason == "tool_calls":
                for tool_call in response.choices[0].message.tool_calls:
                    result = self._execute_tool(tool_call)
                    self.messages.append({"role": "tool", "content": result})
            else:
                return response.choices[0].message.content

# OPENAI SDK: Declarative equivalent
from agents import Agent, Runner, function_tool

@function_tool
def search(query: str) -> str:
    """Search for information."""
    return perform_search(query)

@function_tool
def calculate(expression: str) -> float:
    """Calculate mathematical expressions."""
    return eval(expression)

agent = Agent(
    name="Research Assistant",
    instructions="You are a helpful research assistant.",
    tools=[search, calculate],
    model="gpt-4o"
)

# Run with automatic tracing
result = Runner.run(agent, "Research the market for electric vehicles")

# Multi-agent with handoffs
from agents import handoff

analyst = Agent(
    name="Analyst",
    instructions="Analyze data and provide insights.",
    handoffs=[handoff(agent)]  # Can hand back to research agent
)
```

**Simplification Checklist:**

- [ ] Identify custom code that duplicates SDK features
- [ ] Map custom tools to `@function_tool` decorators
- [ ] Convert orchestration logic to Agent definitions
- [ ] Replace manual tracing with automatic tracing
- [ ] Implement handoffs for multi-agent workflows
- [ ] Add guardrails using SDK patterns
- [ ] Test equivalent behavior
- [ ] Measure latency and cost differences

---

### Pattern 4: Any Framework to Hybrid Approach (Gradual Migration)

When a full migration is too risky, run frameworks in parallel.

**Hybrid Architecture:**

```
                      ┌─────────────────────┐
                      │   Traffic Router    │
                      │  (Feature Flags)    │
                      └──────────┬──────────┘
                                 │
           ┌────────────────────┬┴────────────────────┐
           │                    │                     │
           ▼                    ▼                     ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │ Legacy Agent │    │ New LangGraph│    │   Fallback   │
    │  (CrewAI)    │    │    Agent     │    │   (Simple)   │
    └──────────────┘    └──────────────┘    └──────────────┘
           │                    │                     │
           └────────────────────┼─────────────────────┘
                                │
                      ┌─────────▼─────────┐
                      │  Unified State    │
                      │    (Redis/DB)     │
                      └───────────────────┘
```

**Implementation:**

```python
from enum import Enum
import random

class FrameworkChoice(Enum):
    LEGACY = "legacy"
    NEW = "new"
    FALLBACK = "fallback"

class HybridRouter:
    def __init__(self, legacy_agent, new_agent, fallback_agent):
        self.legacy = legacy_agent
        self.new = new_agent
        self.fallback = fallback_agent
        self.feature_flags = self._load_feature_flags()

    def route(self, request, user_id: str):
        """Route request based on feature flags and canary percentage."""

        # Check user-specific overrides
        if user_id in self.feature_flags.get("new_framework_users", []):
            return self._invoke_new(request)

        # Canary deployment percentage
        canary_percent = self.feature_flags.get("new_framework_canary", 0)
        if random.random() * 100 < canary_percent:
            try:
                return self._invoke_new(request)
            except Exception as e:
                self._log_canary_failure(e)
                return self._invoke_legacy(request)

        return self._invoke_legacy(request)

    def _invoke_new(self, request):
        """Invoke new framework with metrics."""
        start = time.time()
        try:
            result = self.new.invoke(request)
            self._record_metric("new_framework", "success", time.time() - start)
            return result
        except Exception as e:
            self._record_metric("new_framework", "error", time.time() - start)
            raise

    def _invoke_legacy(self, request):
        """Invoke legacy framework with metrics."""
        start = time.time()
        result = self.legacy.invoke(request)
        self._record_metric("legacy_framework", "success", time.time() - start)
        return result
```

**Gradual Migration Phases:**

| Phase | Canary % | Duration | Success Criteria |
|-------|----------|----------|------------------|
| 1. Shadow Mode | 0% (duplicate) | 2 weeks | Output parity >95% |
| 2. Canary | 5% | 1 week | Error rate <1%, latency <+10% |
| 3. Expand | 25% | 2 weeks | Metrics stable |
| 4. Majority | 75% | 2 weeks | No regressions |
| 5. Complete | 100% | 1 week | Legacy deprecated |

---

### Pattern 5: Migration Phases (Parallel Running and Gradual Cutover)

**Phase 1: Assessment (1-2 weeks)**

```
Tasks:
- Inventory all agent code and dependencies
- Document current architecture and data flows
- Identify critical paths and edge cases
- Establish baseline metrics (latency, cost, accuracy)
- Create test suite covering 80%+ of use cases
```

**Phase 2: Parallel Development (2-4 weeks)**

```
Tasks:
- Implement new framework version alongside legacy
- Create shared interfaces for state and tools
- Build comparison infrastructure (shadow mode)
- Validate output equivalence
```

**Phase 3: Shadow Mode (1-2 weeks)**

```
Tasks:
- Route 100% traffic to legacy
- Duplicate requests to new framework
- Compare outputs (do not return new results to users)
- Collect metrics and identify discrepancies
- Fix issues found in shadow testing
```

**Phase 4: Canary Deployment (2-4 weeks)**

```
Canary Progression:
Week 1: 5% traffic to new framework
Week 2: 25% traffic (if metrics stable)
Week 3: 50% traffic
Week 4: 75% traffic
```

**Phase 5: Full Cutover**

```
Tasks:
- Route 100% traffic to new framework
- Keep legacy in standby for 2 weeks
- Monitor for edge cases
- Decommission legacy after validation period
```

---

## Abstraction Strategies

### 1. Framework-Agnostic Agent Interfaces

Create interfaces that allow swapping frameworks without rewriting application code.

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class AgentInterface(ABC):
    """Framework-agnostic agent interface."""

    @abstractmethod
    async def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent with given input."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        pass

    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set agent state (for recovery/persistence)."""
        pass

class LangGraphAdapter(AgentInterface):
    """Adapter for LangGraph agents."""

    def __init__(self, graph, checkpointer):
        self.graph = graph
        self.checkpointer = checkpointer
        self._current_config = None

    async def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = input.get("thread_id", str(uuid4()))
        self._current_config = {"configurable": {"thread_id": thread_id}}

        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=input["query"])]},
            self._current_config
        )

        return {
            "response": result["messages"][-1].content,
            "thread_id": thread_id,
            "state": self.get_state()
        }

    def get_state(self) -> Dict[str, Any]:
        if self._current_config:
            return self.graph.get_state(self._current_config).values
        return {}

class CrewAIAdapter(AgentInterface):
    """Adapter for CrewAI crews."""

    def __init__(self, crew):
        self.crew = crew
        self._last_state = {}

    async def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        result = self.crew.kickoff(inputs=input)
        self._last_state = {"result": result, "input": input}

        return {
            "response": str(result),
            "state": self._last_state
        }

    def get_state(self) -> Dict[str, Any]:
        return self._last_state

class OpenAISDKAdapter(AgentInterface):
    """Adapter for OpenAI Agents SDK."""

    def __init__(self, agent):
        self.agent = agent
        self._run_result = None

    async def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        from agents import Runner

        self._run_result = await Runner.run(
            self.agent,
            input["query"]
        )

        return {
            "response": self._run_result.final_output,
            "trace_id": self._run_result.trace_id
        }

# Usage: swap frameworks without changing application code
def create_agent(framework: str, config: dict) -> AgentInterface:
    """Factory function for creating framework-agnostic agents."""
    if framework == "langgraph":
        graph = build_langgraph_agent(config)
        return LangGraphAdapter(graph, config.get("checkpointer"))
    elif framework == "crewai":
        crew = build_crewai_crew(config)
        return CrewAIAdapter(crew)
    elif framework == "openai":
        agent = build_openai_agent(config)
        return OpenAISDKAdapter(agent)
    else:
        raise ValueError(f"Unknown framework: {framework}")
```

### 2. Tool Portability (Reusable Tools Across Frameworks)

Define tools once, use everywhere:

```python
from typing import Callable, Any
from pydantic import BaseModel, Field

class ToolDefinition(BaseModel):
    """Framework-agnostic tool definition."""
    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable[..., Any]

    class Config:
        arbitrary_types_allowed = True

class PortableTool:
    """Tool that can be converted to any framework's format."""

    def __init__(self, definition: ToolDefinition):
        self.definition = definition

    def to_langgraph(self):
        """Convert to LangGraph tool format."""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            func=self.definition.handler,
            name=self.definition.name,
            description=self.definition.description
        )

    def to_crewai(self):
        """Convert to CrewAI tool format."""
        from crewai_tools import BaseTool

        class DynamicTool(BaseTool):
            name: str = self.definition.name
            description: str = self.definition.description

            def _run(self, **kwargs):
                return self.definition.handler(**kwargs)

        return DynamicTool()

    def to_openai_sdk(self):
        """Convert to OpenAI Agents SDK format."""
        from agents import function_tool

        @function_tool(name=self.definition.name)
        def tool_func(**kwargs):
            return self.definition.handler(**kwargs)

        tool_func.__doc__ = self.definition.description
        return tool_func

# Define tool once
search_tool = PortableTool(ToolDefinition(
    name="web_search",
    description="Search the web for information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    },
    handler=lambda query: perform_web_search(query)
))

# Use in any framework
langgraph_tool = search_tool.to_langgraph()
crewai_tool = search_tool.to_crewai()
openai_tool = search_tool.to_openai_sdk()
```

### 3. State Management Abstraction

Abstract state storage to enable framework switching:

```python
from abc import ABC, abstractmethod
from typing import Optional
import json

class StateStore(ABC):
    """Framework-agnostic state persistence."""

    @abstractmethod
    async def save(self, thread_id: str, state: dict) -> None:
        """Persist state for a thread."""
        pass

    @abstractmethod
    async def load(self, thread_id: str) -> Optional[dict]:
        """Load state for a thread."""
        pass

    @abstractmethod
    async def delete(self, thread_id: str) -> None:
        """Delete state for a thread."""
        pass

class PostgresStateStore(StateStore):
    """PostgreSQL state storage (recommended for production)."""

    def __init__(self, connection_string: str):
        self.pool = create_async_pool(connection_string)

    async def save(self, thread_id: str, state: dict) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO agent_state (thread_id, state, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (thread_id)
                DO UPDATE SET state = $2, updated_at = NOW()
            """, thread_id, json.dumps(state))

    async def load(self, thread_id: str) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state FROM agent_state WHERE thread_id = $1",
                thread_id
            )
            return json.loads(row["state"]) if row else None

class RedisStateStore(StateStore):
    """Redis state storage (for low-latency requirements)."""

    def __init__(self, redis_url: str, ttl_seconds: int = 86400):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl_seconds

    async def save(self, thread_id: str, state: dict) -> None:
        await self.redis.setex(
            f"agent_state:{thread_id}",
            self.ttl,
            json.dumps(state)
        )

    async def load(self, thread_id: str) -> Optional[dict]:
        data = await self.redis.get(f"agent_state:{thread_id}")
        return json.loads(data) if data else None

# Framework-specific adapters
class LangGraphStateAdapter:
    """Adapt generic state store to LangGraph checkpointer interface."""

    def __init__(self, store: StateStore):
        self.store = store

    # Implement LangGraph Checkpointer interface methods
    async def aget(self, config):
        thread_id = config["configurable"]["thread_id"]
        return await self.store.load(thread_id)

    async def aput(self, config, checkpoint, metadata):
        thread_id = config["configurable"]["thread_id"]
        await self.store.save(thread_id, {
            "checkpoint": checkpoint,
            "metadata": metadata
        })
```

### 4. Provider Abstraction (LLM Switching)

Abstract LLM providers for easy switching:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMProvider(ABC):
    """Framework-agnostic LLM interface."""

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        tools: List[dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion from messages."""
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o"):
        self.client = AsyncOpenAI()
        self.model = model

    async def complete(self, messages, tools=None, **kwargs):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            **kwargs
        )
        return self._format_response(response)

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-opus-4-5-20251222"):
        self.client = AsyncAnthropic()
        self.model = model

    async def complete(self, messages, tools=None, **kwargs):
        # Convert to Anthropic format
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]

        response = await self.client.messages.create(
            model=self.model,
            system=system,
            messages=user_messages,
            tools=self._convert_tools(tools) if tools else None,
            **kwargs
        )
        return self._format_response(response)

class LLMFactory:
    """Factory for creating LLM providers based on configuration."""

    @staticmethod
    def create(provider: str, **config) -> LLMProvider:
        providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "azure": AzureOpenAIProvider,
            "google": GoogleProvider,
        }
        return providers[provider](**config)

# Usage
config = load_config()
llm = LLMFactory.create(config["llm_provider"], **config["llm_config"])
```

### 5. Configuration Externalization

Keep framework-specific configuration external:

```yaml
# config/agent.yaml
agent:
  framework: langgraph  # Switch between: langgraph, crewai, openai, autogen

  langgraph:
    checkpointer: postgres
    checkpoint_connection: ${DATABASE_URL}
    compiled_cache: true

  crewai:
    process: sequential
    verbose: true
    memory: true

  openai:
    model: gpt-4o
    tracing: true

llm:
  primary:
    provider: anthropic
    model: claude-opus-4-5-20251222
    max_tokens: 4096
  fallback:
    provider: openai
    model: gpt-4o

state:
  store: postgres
  connection: ${DATABASE_URL}

tools:
  - name: web_search
    enabled: true
    rate_limit: 100/minute
  - name: code_execute
    enabled: true
    sandbox: true
```

```python
# Load and apply configuration
import yaml

def load_agent_config(path: str = "config/agent.yaml"):
    with open(path) as f:
        config = yaml.safe_load(f)

    # Resolve environment variables
    config = resolve_env_vars(config)

    return config

def create_agent_from_config(config: dict) -> AgentInterface:
    """Create agent based on external configuration."""
    framework = config["agent"]["framework"]
    framework_config = config["agent"].get(framework, {})

    # Create LLM provider
    llm = LLMFactory.create(**config["llm"]["primary"])

    # Create state store
    state_store = create_state_store(config["state"])

    # Create tools
    tools = [create_tool(t) for t in config["tools"] if t["enabled"]]

    # Create framework-specific agent
    return create_agent(framework, {
        "llm": llm,
        "tools": tools,
        "state_store": state_store,
        **framework_config
    })
```

---

## Common Pitfalls

### 1. State Format Differences (Serialization Issues)

**Problem:** State formats differ between frameworks, causing data loss during migration.

```python
# LangGraph: TypedDict with specific message types
class LangGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# CrewAI: Dictionary with string content
crewai_state = {"output": "string result", "inputs": {...}}

# OpenAI SDK: Run result objects
openai_state = RunResult(final_output="...", trace_id="...")
```

**Solution: Create State Adapters**

```python
class StateNormalizer:
    """Normalize state between frameworks."""

    @staticmethod
    def to_universal(state: Any, source: str) -> dict:
        """Convert framework state to universal format."""
        if source == "langgraph":
            return {
                "messages": [
                    {"role": m.type, "content": m.content}
                    for m in state.get("messages", [])
                ],
                "metadata": state.get("metadata", {})
            }
        elif source == "crewai":
            return {
                "messages": [{"role": "assistant", "content": str(state)}],
                "metadata": state.get("inputs", {})
            }
        elif source == "openai":
            return {
                "messages": [{"role": "assistant", "content": state.final_output}],
                "metadata": {"trace_id": state.trace_id}
            }

    @staticmethod
    def from_universal(universal: dict, target: str) -> Any:
        """Convert universal format to framework state."""
        if target == "langgraph":
            from langchain_core.messages import HumanMessage, AIMessage
            messages = []
            for m in universal["messages"]:
                if m["role"] == "human":
                    messages.append(HumanMessage(content=m["content"]))
                else:
                    messages.append(AIMessage(content=m["content"]))
            return {"messages": messages, "metadata": universal["metadata"]}
        # ... other frameworks
```

### 2. Tool Compatibility Issues (Schema Differences)

**Problem:** Tool schemas differ between frameworks.

| Framework | Tool Definition Style |
|-----------|----------------------|
| LangGraph | Pydantic models or @tool decorator |
| CrewAI | BaseTool class inheritance |
| OpenAI SDK | @function_tool decorator |
| AutoGen | FunctionTool wrappers |

**Solution: Universal Tool Schema**

```python
# Define tools using JSON Schema (universal format)
TOOL_SCHEMA = {
    "name": "search_database",
    "description": "Search a database for records",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10}
        },
        "required": ["query"]
    }
}

def tool_handler(query: str, limit: int = 10):
    return database.search(query, limit)

# Convert to any framework
def to_framework_tool(schema: dict, handler: Callable, framework: str):
    if framework == "langgraph":
        from langchain_core.tools import StructuredTool
        return StructuredTool.from_function(
            func=handler,
            name=schema["name"],
            description=schema["description"],
            args_schema=json_schema_to_pydantic(schema["parameters"])
        )
    elif framework == "openai":
        from agents import function_tool
        decorated = function_tool(name=schema["name"])(handler)
        decorated.__doc__ = schema["description"]
        return decorated
    # ... other frameworks
```

### 3. Prompt Format Changes (Different Syntax)

**Problem:** Prompt formats and variable injection differ.

```python
# LangGraph: ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are {role}"),
    ("human", "{input}")
])

# CrewAI: F-string in backstory/goal
agent = Agent(
    role="Analyst",
    backstory="You specialize in {domain}"  # Filled at runtime
)

# OpenAI SDK: Plain string instructions
agent = Agent(
    instructions="You are a helpful assistant."
)
```

**Solution: Prompt Template Abstraction**

```python
class PromptTemplate:
    """Universal prompt template."""

    def __init__(self, system: str, user: str = "{input}"):
        self.system = system
        self.user = user

    def to_langgraph(self):
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate.from_messages([
            ("system", self.system),
            ("human", self.user)
        ])

    def to_crewai_agent(self, role: str, goal: str):
        return {
            "role": role,
            "goal": goal,
            "backstory": self.system
        }

    def to_openai_instructions(self, **kwargs):
        return self.system.format(**kwargs)

# Define once
analyst_prompt = PromptTemplate(
    system="""You are a {specialty} analyst.
Your goal is to provide accurate, data-driven insights.
Always cite sources when making claims."""
)

# Use anywhere
langgraph_prompt = analyst_prompt.to_langgraph()
crewai_config = analyst_prompt.to_crewai_agent("Analyst", "Provide insights")
openai_instructions = analyst_prompt.to_openai_instructions(specialty="market")
```

### 4. Testing Regression (Behavior Changes)

**Problem:** Same inputs produce different outputs after migration.

**Solution: Regression Testing Framework**

```python
import pytest
from dataclasses import dataclass

@dataclass
class TestCase:
    input: dict
    expected_contains: List[str]  # Strings that should appear in output
    expected_not_contains: List[str] = None  # Strings that should NOT appear
    max_latency_ms: int = 5000
    max_cost_usd: float = 0.10

class MigrationRegressionSuite:
    """Test suite for framework migration validation."""

    def __init__(self, legacy_agent, new_agent, test_cases: List[TestCase]):
        self.legacy = legacy_agent
        self.new = new_agent
        self.test_cases = test_cases
        self.results = []

    async def run_comparison(self):
        """Run all test cases on both agents."""
        for case in self.test_cases:
            legacy_result = await self._run_with_metrics(self.legacy, case.input)
            new_result = await self._run_with_metrics(self.new, case.input)

            self.results.append({
                "input": case.input,
                "legacy": legacy_result,
                "new": new_result,
                "content_match": self._compare_outputs(
                    legacy_result["output"],
                    new_result["output"],
                    case
                ),
                "latency_regression": new_result["latency_ms"] > case.max_latency_ms,
                "cost_regression": new_result["cost_usd"] > case.max_cost_usd
            })

        return self._generate_report()

    def _compare_outputs(self, legacy: str, new: str, case: TestCase) -> dict:
        """Compare outputs for semantic equivalence."""
        issues = []

        for expected in case.expected_contains:
            if expected.lower() not in legacy.lower():
                issues.append(f"Legacy missing: {expected}")
            if expected.lower() not in new.lower():
                issues.append(f"New missing: {expected}")

        if case.expected_not_contains:
            for forbidden in case.expected_not_contains:
                if forbidden.lower() in new.lower():
                    issues.append(f"New incorrectly contains: {forbidden}")

        return {"passed": len(issues) == 0, "issues": issues}

# Example usage
test_cases = [
    TestCase(
        input={"query": "What is the capital of France?"},
        expected_contains=["Paris"],
        max_latency_ms=2000
    ),
    TestCase(
        input={"query": "Calculate 15% tip on $45.00"},
        expected_contains=["6.75", "tip"],
        max_latency_ms=3000
    )
]

suite = MigrationRegressionSuite(legacy_agent, new_agent, test_cases)
report = await suite.run_comparison()
```

### 5. Performance Differences

**Problem:** New framework has different performance characteristics.

**Common Performance Changes:**

| Change | Cause | Mitigation |
|--------|-------|------------|
| Cold start slower | Graph compilation | Cache compiled graph at startup |
| Latency higher | Additional state persistence | Use Redis for hot path, Postgres for durability |
| Memory higher | State object size | Implement state pruning |
| Cost higher | More LLM calls for routing | Optimize routing logic |

**Solution: Performance Monitoring**

```python
from dataclasses import dataclass
from prometheus_client import Histogram, Counter

# Define metrics
latency_histogram = Histogram(
    'agent_request_latency_seconds',
    'Agent request latency',
    ['framework', 'operation']
)

cost_counter = Counter(
    'agent_request_cost_usd',
    'Agent request cost in USD',
    ['framework', 'model']
)

@dataclass
class PerformanceBaseline:
    p50_latency_ms: float
    p99_latency_ms: float
    avg_cost_usd: float
    avg_tokens: int

class PerformanceMonitor:
    """Monitor for detecting performance regressions."""

    def __init__(self, baseline: PerformanceBaseline, tolerance: float = 0.1):
        self.baseline = baseline
        self.tolerance = tolerance

    def check_latency(self, latency_ms: float) -> bool:
        """Returns True if latency is within acceptable range."""
        max_allowed = self.baseline.p50_latency_ms * (1 + self.tolerance)
        return latency_ms <= max_allowed

    def check_cost(self, cost_usd: float) -> bool:
        """Returns True if cost is within acceptable range."""
        max_allowed = self.baseline.avg_cost_usd * (1 + self.tolerance)
        return cost_usd <= max_allowed

    async def alert_on_regression(self, metrics: dict):
        """Send alert if performance has regressed."""
        if not self.check_latency(metrics["latency_ms"]):
            await self._send_alert(
                "Latency Regression",
                f"Current: {metrics['latency_ms']}ms, Baseline: {self.baseline.p50_latency_ms}ms"
            )

        if not self.check_cost(metrics["cost_usd"]):
            await self._send_alert(
                "Cost Regression",
                f"Current: ${metrics['cost_usd']:.4f}, Baseline: ${self.baseline.avg_cost_usd:.4f}"
            )
```

---

## Case Studies

### Case Study 1: Grid Dynamics - LangGraph to Temporal Migration

**Context:** Research agent for deep market analysis.

**Initial Architecture:**
- LangGraph for workflow orchestration
- Redis for state management
- 100 concurrent workflows capacity

**Problems Encountered:**
1. Redis state lifecycle management complexity
2. Stale state issues with cache collisions
3. Debugging difficulties with distributed state

**Migration Decision:** Move to Temporal for workflow orchestration

**Results:**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Concurrent workflows | 100 | 10,000 | +9,900% |
| Average latency | 3.2s | 1.1s | -66% |
| Infrastructure cost | $X | $X * 0.38 | -62% |
| Custom retry code | 3,000 lines | 0 lines | -100% |

**Key Lessons:**
1. LangGraph excels at graph-based orchestration but requires additional infrastructure for durability
2. Temporal's event-sourcing model eliminated state management complexity
3. Choose frameworks that solve the complete problem, not just orchestration

---

### Case Study 2: Enterprise CRM - CrewAI to LangGraph Migration

**Context:** Customer service automation with 5 specialized agents.

**Initial Architecture (CrewAI):**
```
Crew (Hierarchical):
  - Manager Agent
  - Ticket Classifier Agent
  - Knowledge Base Agent
  - Response Generator Agent
  - Quality Assurance Agent
```

**Problems Encountered:**
1. Could not implement conditional routing (ticket type -> specific workflow)
2. State persistence required for multi-session conversations
3. Human approval needed for refund decisions

**Migration Approach:**

```python
# LangGraph implementation with conditional routing
def route_ticket(state: TicketState) -> str:
    """Route based on ticket classification."""
    ticket_type = state["ticket_type"]

    if ticket_type == "refund":
        if state["amount"] > 100:
            return "human_approval"
        return "auto_process_refund"
    elif ticket_type == "technical":
        return "technical_support"
    else:
        return "general_support"

workflow = StateGraph(TicketState)
workflow.add_node("classify", classify_ticket)
workflow.add_node("human_approval", human_approval_node)
workflow.add_node("auto_process_refund", refund_node)
workflow.add_node("technical_support", tech_support_node)
workflow.add_node("general_support", general_node)

# Conditional routing not possible in CrewAI
workflow.add_conditional_edges("classify", route_ticket, {
    "human_approval": "human_approval",
    "auto_process_refund": "auto_process_refund",
    "technical_support": "technical_support",
    "general_support": "general_support"
})
```

**Results:**

| Metric | CrewAI | LangGraph | Change |
|--------|--------|-----------|--------|
| Resolution time | 4.2 min | 2.8 min | -33% |
| Human escalation rate | 35% | 18% | -49% |
| Customer satisfaction | 3.8/5 | 4.3/5 | +13% |
| Development flexibility | Limited | Full control | Qualitative |

**Key Lessons:**
1. Role-based abstractions work until you need conditional logic
2. Explicit state management enables multi-session workflows
3. Human-in-the-loop patterns are first-class in LangGraph

---

### Case Study 3: Fintech Startup - Assistants API to Responses API Migration

**Context:** Financial advisory chatbot using OpenAI Assistants API.

**Migration Driver:** Assistants API deprecation (sunset August 26, 2026)

**Initial Architecture:**
```python
# Assistants API (deprecated)
assistant = client.beta.assistants.create(
    instructions="You are a financial advisor...",
    tools=[{"type": "retrieval"}],
    model="gpt-4-turbo"
)

thread = client.beta.threads.create()
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    content=user_query
)
run = client.beta.threads.runs.create(...)  # Blocks until complete
```

**Problems with Assistants API:**
1. Entire thread reprocessed every message (cost scaling)
2. No streaming (poor UX for long responses)
3. Unpredictable latency

**New Architecture (Responses API + Agents SDK):**

```python
from agents import Agent, Runner, function_tool

@function_tool
async def get_portfolio(user_id: str) -> dict:
    """Get user's current portfolio."""
    return await db.get_portfolio(user_id)

@function_tool
async def get_market_data(symbols: List[str]) -> dict:
    """Get current market data for symbols."""
    return await market_api.get_quotes(symbols)

advisor = Agent(
    name="Financial Advisor",
    instructions="""You are a certified financial advisor.
    Always consider the user's risk tolerance and goals.
    Never provide specific buy/sell recommendations without context.""",
    tools=[get_portfolio, get_market_data],
    model="gpt-4o"
)

# Explicit state management (now developer responsibility)
class ConversationManager:
    def __init__(self):
        self.store = PostgresStateStore()

    async def handle_message(self, user_id: str, message: str):
        # Load conversation history
        history = await self.store.load(user_id)

        # Intelligent context pruning (Assistants API couldn't do this)
        if len(history.get("messages", [])) > 20:
            history = await self._summarize_history(history)

        # Run with streaming
        async for chunk in Runner.run_streamed(
            advisor,
            message,
            context=history
        ):
            yield chunk

        # Save updated history
        await self.store.save(user_id, updated_history)
```

**Results:**

| Metric | Assistants API | Responses API | Change |
|--------|---------------|---------------|--------|
| Avg cost/conversation | $0.45 | $0.18 | -60% |
| Time to first token | 2.1s | 0.3s | -86% |
| Context control | None | Full | Qualitative |
| Streaming | No | Yes | Qualitative |

**Key Lessons:**
1. Explicit state management enables optimization (summarization, pruning)
2. Streaming dramatically improves perceived performance
3. Migration complexity was moderate (2 developer-weeks)

---

### Case Study 4: Healthcare Platform - Hybrid Multi-Framework Approach

**Context:** Clinical decision support with strict compliance requirements.

**Architecture Decision:** Use multiple frameworks for different components

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Triage Layer (OpenAI SDK)                           │
│ - Fast routing                                      │
│ - Simple handoffs                                   │
└─────────────────┬───────────────────────────────────┘
                  │
    ┌─────────────┼─────────────────┐
    ▼             ▼                 ▼
┌────────┐  ┌──────────┐    ┌──────────────┐
│ Billing│  │ Clinical │    │ Scheduling   │
│ (CrewAI)│  │(LangGraph)│   │ (OpenAI SDK) │
│ Simple │  │ Complex  │    │ Simple       │
│ workflow│  │ state    │    │ integrations │
└────────┘  └──────────┘    └──────────────┘
                  │
                  ▼
         ┌───────────────┐
         │ Audit Layer   │
         │ (LangGraph)   │
         │ Full state    │
         │ persistence   │
         └───────────────┘
```

**Framework Selection Rationale:**

| Component | Framework | Why |
|-----------|-----------|-----|
| Triage | OpenAI SDK | Fast routing, minimal latency |
| Billing | CrewAI | Standard workflow, quick to build |
| Clinical | LangGraph | Complex state, human-in-loop, audit trail |
| Scheduling | OpenAI SDK | Simple tool calling, external integrations |
| Audit | LangGraph | Full state persistence, compliance |

**Results:**

| Metric | Single Framework | Hybrid | Change |
|--------|-----------------|--------|--------|
| Development time | 6 months | 4 months | -33% |
| Team productivity | Bottlenecked | Parallelized | +50% |
| Compliance audit | Manual | Automated | -80% effort |
| Maintenance | Complex | Modular | Qualitative |

**Key Lessons:**
1. Not every component needs the most powerful framework
2. Hybrid approaches enable team parallelization
3. Clear interfaces between components are essential
4. Choose framework based on component requirements, not global optimization

---

## Decision Framework

### When to Stay vs. Migrate

**Stay with Current Framework When:**

| Factor | Indicator |
|--------|-----------|
| Stability | Current system is stable, no critical issues |
| ROI | Migration cost exceeds 6-month benefit |
| Team | No bandwidth for migration risk |
| Deprecation | No sunset date announced |
| Requirements | Current framework meets all requirements |

**Migrate When:**

| Factor | Indicator |
|--------|-----------|
| Deprecation | Sunset date announced (e.g., Assistants API) |
| Blocking Issues | Current framework cannot support requirements |
| Cost | Migration saves >30% in 6 months |
| Technical Debt | Workarounds exceeding framework value |
| Scale | Current framework at performance limits |

### Migration Cost Estimation

**Cost Estimation Formula:**

```
Total Migration Cost =
    Development Cost +
    Testing Cost +
    Rollout Cost +
    Risk Buffer

Where:
    Development Cost = (Engineer Days) * (Daily Rate) * 1.5 (unknowns)
    Testing Cost = Development Cost * 0.4
    Rollout Cost = Development Cost * 0.2
    Risk Buffer = (Development + Testing + Rollout) * 0.25
```

**Effort Estimation by Migration Type:**

| Migration Type | Complexity | Engineer-Days | Risk Level |
|---------------|------------|---------------|------------|
| LangChain -> LangGraph | Medium | 5-15 | Low |
| CrewAI -> LangGraph | Medium-High | 10-25 | Medium |
| Custom -> OpenAI SDK | Low-Medium | 3-10 | Low |
| Assistants -> Responses API | Medium | 5-15 | Low |
| Any -> Hybrid | High | 15-40 | Medium-High |
| Full Rewrite | Very High | 30-90 | High |

### Risk Assessment Checklist

**Pre-Migration Risk Assessment:**

```
Technical Risks:
□ State format compatibility verified
□ Tool schema compatibility verified
□ Performance baseline established
□ Rollback plan documented
□ Data migration strategy defined

Operational Risks:
□ Team trained on new framework
□ Monitoring updated for new patterns
□ Runbooks updated for new failure modes
□ On-call team briefed

Business Risks:
□ Stakeholder approval obtained
□ Rollout timeline communicated
□ Fallback plan approved
□ Success criteria defined
```

### Rollback Planning

**Rollback Strategy Template:**

```yaml
rollback_plan:
  triggers:
    - error_rate > 5% for 15 minutes
    - latency_p99 > baseline * 2 for 10 minutes
    - cost_per_request > baseline * 1.5 for 1 hour
    - customer_complaints > 10 in 1 hour

  procedure:
    immediate:
      - route 100% traffic to legacy
      - alert on-call team
      - preserve logs for analysis

    investigation:
      - capture state snapshots
      - collect error samples
      - identify root cause

    post_mortem:
      - document failure mode
      - update test cases
      - revise rollout criteria

  timeline:
    decision_to_rollback: < 5 minutes
    traffic_rerouted: < 2 minutes
    legacy_confirmed_healthy: < 5 minutes

  data_handling:
    in_flight_requests: complete on legacy
    state_divergence: manual reconciliation if needed
    audit_trail: preserve all decision logs
```

---

## Migration Checklists

### Pre-Migration Checklist

```
Assessment:
□ Current framework inventory complete
□ All agents, tools, and state documented
□ Baseline metrics collected (latency, cost, accuracy)
□ Test coverage at 80%+
□ Deprecation timeline understood

Planning:
□ Target framework selected with rationale
□ Architecture design reviewed
□ Migration phases defined
□ Team responsibilities assigned
□ Timeline and milestones set

Risk:
□ Risk assessment complete
□ Rollback plan documented
□ Success criteria defined
□ Stakeholder approval obtained
```

### During-Migration Checklist

```
Development:
□ Framework-agnostic interfaces implemented
□ State migration strategy implemented
□ Tools ported to new format
□ Prompts adapted to new syntax

Testing:
□ Unit tests passing
□ Integration tests passing
□ Regression tests against baseline
□ Performance tests within tolerance
□ Security review completed

Infrastructure:
□ Checkpoint storage configured
□ Observability enabled
□ Monitoring dashboards updated
□ Alerting configured
```

### Post-Migration Checklist

```
Validation:
□ Canary deployment successful
□ Metrics within acceptable range
□ No critical bugs in production
□ User feedback positive

Cleanup:
□ Legacy code removed (after stabilization period)
□ Documentation updated
□ Runbooks updated
□ Team training completed

Long-term:
□ Post-mortem conducted
□ Lessons learned documented
□ Framework abstraction maintained
□ Future migration path considered
```

---

## Appendix: Quick Reference

### Framework Comparison Summary

| Framework | Best For | Avoid When | Learning Curve |
|-----------|----------|------------|----------------|
| LangGraph | Complex workflows, production | Simple prototypes | High |
| CrewAI | Role-based teams, quick prototyping | Complex conditionals | Medium |
| OpenAI SDK | Fast development, simple agents | Fine-grained control | Low |
| AutoGen | Research, conversational agents | Deterministic needs | Medium |
| MS Agent | Azure enterprise, compliance | Non-Microsoft stack | Medium |

### Migration Path Quick Reference

```
Start Here:
    ↓
What's your current framework?
    ├── LangChain Agents → LangGraph (recommended path)
    ├── CrewAI → LangGraph (if hitting limits) or stay
    ├── Assistants API → Responses API + Agents SDK (required by 2026)
    ├── AutoGen → MS Agent Framework (if new projects)
    └── Custom → OpenAI SDK (for simplification)
```

### Key Metrics to Track

| Metric | Healthy Range | Alert Threshold |
|--------|--------------|-----------------|
| P50 Latency | <1s | >2s |
| P99 Latency | <5s | >10s |
| Error Rate | <1% | >5% |
| Cost per Request | <$0.05 | >$0.15 |
| Tokens per Request | <2000 | >5000 |

---

## Related Documents

- [framework-comparison.md](../phase-1-foundations/framework-comparison.md) - Detailed framework analysis
- [patterns-and-antipatterns.md](../phase-3-patterns/patterns-and-antipatterns.md) - Common failure modes
- [2025-updates.md](../reference/2025-updates.md) - Latest framework updates
- [evaluation-and-debugging.md](../phase-4-production/evaluation-and-debugging.md) - Testing strategies

---

**Document Version:** 1.0
**Last Updated:** 2025-12-27
**Authors:** Agentic AI Developer Onboarding Team
