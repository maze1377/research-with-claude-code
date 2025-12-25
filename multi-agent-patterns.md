# LangGraph Multi-Agent Patterns: Complete Guide (2025)

**Based on**: Production case studies (LinkedIn, Uber, Replit, Elastic), LangChain benchmarks, official documentation, and real-world lessons learned

**Last Updated**: 2025-11-08

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

**Last Updated**: 2025-11-08
**Maintained By**: Research synthesis from LangChain documentation, production case studies, and community best practices
