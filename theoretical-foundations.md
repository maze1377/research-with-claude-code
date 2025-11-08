# Theoretical Foundations for Multi-Agent Systems
## A Research-Based Guide for API-Driven Development (2024-2025)

**Purpose:** This document provides theoretical foundations for building multi-agent systems using LLM APIs (OpenAI, Anthropic) without developing custom models. Based on latest research papers and industry developments.

**Last Updated:** 2025-11-08

---

## Table of Contents

1. [Core Reasoning Patterns](#core-reasoning-patterns)
2. [Multi-Agent Collaboration Theory](#multi-agent-collaboration-theory)
3. [Communication Protocols](#communication-protocols)
4. [Tool Use and Function Calling](#tool-use-and-function-calling)
5. [Extended Thinking and Reasoning](#extended-thinking-and-reasoning)
6. [Theoretical Framework for Architecture Selection](#theoretical-framework-for-architecture-selection)
7. [Research Papers and Citations](#research-papers-and-citations)

---

## Core Reasoning Patterns

### 1. ReAct (Reasoning + Acting)
**Paper:** Yao et al., 2022 - "ReAct: Synergizing Reasoning and Acting in Language Models" (arXiv:2210.03629)

**Core Concept:**
ReAct interleaves reasoning traces and task-specific actions in an iterative loop. The model generates verbal reasoning ("thoughts") before and after taking actions.

**Pattern Structure:**
```
Thought → Action → Observation → Thought → Action → Observation → ...
```

**Key Theoretical Insights:**
- **Internal vs External Knowledge:** Combines internal reasoning (CoT) with external information gathering (actions)
- **Dynamic Adaptation:** Allows agents to adjust plans based on observed outcomes
- **Reduced Hallucination:** Grounding reasoning in real observations reduces fabricated information
- **Human Interpretability:** Explicit reasoning traces make agent behavior transparent

**Practical Application for API Users:**
- Use system prompts to establish the ReAct pattern
- Structure tool calls as "actions" in the loop
- Request explicit reasoning before tool calls
- Parse observations and feed them back for next iteration

**Performance Metrics from Paper:**
- HotpotQA: ReAct achieves 27.4% success (vs 14.5% baseline)
- Fever: 58.0% success (vs 43.1% baseline)
- ALFWorld: 34% success (vs 0% zero-shot)

### 2. Chain-of-Thought (CoT)
**Paper:** Wei et al., 2022 - "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (arXiv:2201.11903)

**Core Concept:**
Prompt the model to generate intermediate reasoning steps before producing the final answer.

**Theoretical Foundation:**
- **Decomposition:** Complex reasoning broken into manageable steps
- **Sequential Processing:** Each step builds on previous steps
- **Emergent Ability:** Only appears in models with sufficient scale (>100B parameters)

**Variants:**
1. **Few-Shot CoT:** Provide examples with reasoning chains
2. **Zero-Shot CoT:** Simply add "Let's think step by step"
3. **Self-Consistency CoT:** Sample multiple reasoning paths, select most consistent answer

**Key Findings:**
- Arithmetic reasoning: 17.9% → 78.5% accuracy (GSM8K with PaLM 540B)
- Commonsense reasoning: 74.0% → 83.8% (StrategyQA)
- Symbolic reasoning: 37.1% → 58.8% (Last Letter Concatenation)

**API Implementation:**
```python
# Zero-Shot CoT
prompt = f"""
Question: {question}

Let's approach this step-by-step:
1. First, identify the key information
2. Then, break down the problem
3. Finally, solve it systematically
"""

# Self-Consistency CoT (sample multiple times)
responses = [api.complete(prompt, temperature=0.7) for _ in range(5)]
# Select most common answer
```

### 3. Tree of Thoughts (ToT)
**Paper:** Yao et al., 2023 - "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (arXiv:2305.10601)

**Core Concept:**
Explores multiple reasoning paths simultaneously in a tree structure, evaluating and backtracking as needed.

**Theoretical Framework:**
- **Search Space:** Each node represents a partial solution (thought)
- **Lookahead:** Evaluate future potential of thoughts
- **Backtracking:** Abandon unpromising paths
- **Breadth vs Depth:** Balance exploration and exploitation

**Search Strategies:**
1. **BFS (Breadth-First Search):** Maintain k most promising paths at each level
2. **DFS (Depth-First Search):** Explore one path fully before backtracking

**Performance Improvements:**
- Game of 24: 4% → 74% success rate
- Creative Writing: 6.2% → 7.5% coherence score
- Mini Crosswords: 60% → 78% word success rate

**API Implementation Considerations:**
- Requires multiple API calls per decision point
- Cost scales with breadth and depth parameters
- Best for problems with clear success criteria
- Implement thought evaluation using separate API calls

**ToT vs ReAct vs CoT:**
| Pattern | Search Strategy | API Calls | Best For |
|---------|----------------|-----------|----------|
| CoT | Linear chain | 1 per problem | Sequential reasoning |
| ReAct | Linear with feedback | 2-10 per problem | Tool-augmented tasks |
| ToT | Tree search | 10-100+ per problem | Complex search problems |

### 4. Self-Consistency
**Concept:** Generate multiple reasoning paths and select the most consistent answer through majority voting.

**Theoretical Basis:**
- **Diversity:** Different reasoning paths may arrive at correct answer via different routes
- **Robustness:** Reduces impact of individual reasoning errors
- **Emergent Agreement:** Correct answers tend to cluster

**Implementation:**
```python
# Sample multiple independent chains
chains = [cot_prompt(question, temperature=0.7) for _ in range(k)]

# Extract final answers
answers = [parse_answer(chain) for chain in chains]

# Majority vote
from collections import Counter
final_answer = Counter(answers).most_common(1)[0][0]
```

### 5. Generated Knowledge Prompting
**Concept:** First generate relevant knowledge, then use it to answer the question.

**Two-Stage Process:**
1. **Knowledge Generation:** "What facts are relevant to this question?"
2. **Answer Generation:** Use generated knowledge to answer

**Benefits:**
- Externalizes internal knowledge
- Reduces hallucination by making knowledge explicit
- Allows verification of generated facts

---

## Multi-Agent Collaboration Theory

### Five-Dimensional Framework
**Source:** arXiv:2501.06322v1 - "Multi-Agent Collaboration Mechanisms: A Survey" (2025)

#### 1. Actors (Who Collaborates?)
**Homogeneous Agents:**
- Same capabilities and models
- Easier coordination
- Uniform behavior patterns

**Heterogeneous Agents:**
- Different specializations (researcher, analyst, writer)
- Complementary skills
- More complex coordination

**Theoretical Consideration for API Users:**
- Homogeneous: Single API with different system prompts
- Heterogeneous: Multiple APIs or models (GPT-4o for reasoning, Claude for writing)

#### 2. Collaboration Types (How They Interact?)

**Cooperation:**
- Agents work toward shared goal
- No conflicting objectives
- Shared reward/success criteria

**Competition:**
- Agents have opposing objectives
- Useful for adversarial validation
- Example: Debate systems, red-team/blue-team

**Coopetition:**
- Mix of cooperation and competition
- Agents compete to contribute best solution
- Team success depends on individual excellence

**API Implementation:**
```python
# Cooperation: Sequential workflow
research_output = researcher_agent(topic)
analysis = analyst_agent(research_output)
report = writer_agent(analysis)

# Competition: Debate
argument_a = agent_for(position="for")
argument_b = agent_against(position="against")
judge_decision = judge_agent([argument_a, argument_b])

# Coopetition: Multiple solutions, best selected
solutions = [agent(task) for agent in specialist_agents]
best_solution = evaluator_agent(solutions)
```

#### 3. Collaboration Structures (Topology)

**Centralized (Hub-and-Spoke):**
- One coordinator, multiple workers
- Coordinator makes all routing decisions
- Clear control flow, single point of failure

**Decentralized (Peer-to-Peer):**
- Agents communicate directly
- No central authority
- More resilient, harder to coordinate

**Hierarchical (Tree Structure):**
- Multiple layers of coordination
- Sub-teams with local coordinators
- Scales to large agent counts

**Theoretical Trade-offs:**
| Structure | Coordination Overhead | Scalability | Fault Tolerance | Best For |
|-----------|---------------------|-------------|-----------------|----------|
| Centralized | Low | Medium | Low | Small teams, clear hierarchy |
| Decentralized | High | Low | High | Adaptive, exploratory tasks |
| Hierarchical | Medium | High | Medium | Large teams, modular tasks |

#### 4. Collaboration Strategies

**Rule-Based:**
- Predefined workflows
- Static routing (if-then logic)
- Deterministic behavior
- Low overhead, inflexible

**Role-Based:**
- Agents assigned specific roles
- Role determines behavior and responsibilities
- Common in CrewAI approach

**Model-Based:**
- LLM decides routing and collaboration
- Dynamic adaptation
- Higher cost, more flexible

**API Implementation Example:**
```python
# Rule-Based
if task_type == "research":
    agent = research_agent
elif task_type == "analysis":
    agent = analyst_agent

# Model-Based (LLM routing)
router_prompt = f"""
Task: {task}
Available agents: {agent_descriptions}
Which agent should handle this? Respond with agent name only.
"""
selected_agent = llm.invoke(router_prompt).content.strip()
```

#### 5. Coordination Mechanisms

**Message Passing:**
- Agents send/receive structured messages
- Asynchronous communication
- Requires message queuing

**Shared Memory:**
- Common state accessible to all agents
- Synchronous access patterns
- Example: LangGraph's shared state

**Blackboard System:**
- Shared knowledge repository
- Agents read/write to blackboard
- Opportunistic collaboration

**For API Users:**
- Message Passing: Implement with queues or databases
- Shared Memory: Use LangGraph StateGraph with TypedDict
- Blackboard: Implement with vector database or knowledge graph

---

## Communication Protocols

### 1. Model Context Protocol (MCP)
**Source:** Anthropic 2025

**Purpose:** Standardized protocol for LLM applications to access context from various sources.

**Architecture:**
- **Hosts:** LLM applications (Claude Desktop, IDEs)
- **Clients:** Protocol clients inside host application
- **Servers:** Lightweight programs exposing context (databases, APIs, files)

**Key Features:**
- Decoupled architecture
- Unopinionated integration
- Server lifecycle management

**For Multi-Agent Systems:**
- Agents can share context via MCP servers
- Standardized tool/resource interfaces
- Reduces custom integration code

### 2. Agent Communication Protocol (ACP)
**Purpose:** Enable structured communication between heterogeneous agents.

**Message Types:**
- **Performatives:** Speech acts (inform, request, query, propose)
- **Content:** Actual message payload
- **Metadata:** Sender, receiver, conversation ID, reply-to

**Example Message Structure:**
```json
{
  "performative": "request",
  "sender": "agent_a",
  "receiver": "agent_b",
  "conversation_id": "conv_123",
  "content": {
    "action": "research",
    "topic": "multi-agent systems"
  },
  "reply_to": "msg_456"
}
```

### 3. Agent-to-Agent Protocol (A2A)
**Focus:** Direct peer-to-peer agent communication.

**Patterns:**
- **Request-Response:** Synchronous interaction
- **Publish-Subscribe:** Broadcast to interested agents
- **Query-Inform:** Information seeking

**Implementation with APIs:**
```python
class AgentMessage:
    def __init__(self, sender, receiver, content, msg_type):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.msg_type = msg_type  # request, inform, query
        self.timestamp = time.time()

# Agent A sends request
request = AgentMessage(
    sender="researcher",
    receiver="analyst",
    content={"data": research_output},
    msg_type="inform"
)

# Agent B processes and responds
response = agent_b.process(request)
```

### 4. Handoffs (LangGraph 2025)
**New Feature:** Explicit agent-to-agent transitions.

**Pattern:**
```python
from langgraph.types import Command

def agent_a(state) -> Command[Literal["agent_b", END]]:
    # Process
    result = llm.invoke(state.messages)

    # Decide handoff
    if needs_specialist:
        return Command(
            goto="agent_b",
            update={"messages": [result], "context": "handoff from A"}
        )
    return Command(goto=END)
```

**Benefits:**
- Explicit control flow
- Type-safe transitions
- Preserves state across handoffs

---

## Tool Use and Function Calling

### Theoretical Framework

**Core Concept:** Extend LLM capabilities by connecting to external tools/APIs.

**Pattern:**
1. **Tool Description:** LLM receives tool specifications (JSON schema)
2. **Intent Detection:** LLM decides when to call tool
3. **Parameter Generation:** LLM generates function arguments
4. **Execution:** External system executes tool
5. **Result Integration:** Tool output fed back to LLM

### API-Specific Implementations

#### OpenAI Function Calling (2024)

**Key Developments:**
- gpt-4o-2024-08-06: 100% accuracy on complex JSON schema (vs 40% for gpt-4-0613)
- Structured Outputs with `strict: true`
- Up to 128 tools per request (practical limit: 5-10 for accuracy)

**Best Practices from Research:**

1. **Tool Documentation:**
   - Each tool must have clear, complete documentation
   - Parameter descriptions are critical for correct usage
   - Use examples in descriptions

2. **Tool Count Management:**
   ```python
   # BAD: Too many tools
   tools = [tool1, tool2, ..., tool128]  # High error rate

   # GOOD: Use RAG to select relevant tools
   relevant_tools = rag_select_tools(user_query, all_tools, top_k=5)
   response = openai.chat.completions.create(
       model="gpt-4o",
       messages=messages,
       tools=relevant_tools
   )
   ```

3. **Structured Outputs:**
   ```python
   # Guaranteed schema adherence
   response = openai.chat.completions.create(
       model="gpt-4o-2024-08-06",
       messages=messages,
       tools=[{
           "type": "function",
           "function": {
               "name": "get_weather",
               "strict": True,  # Enforces schema
               "parameters": {
                   "type": "object",
                   "properties": {
                       "location": {"type": "string"},
                       "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                   },
                   "required": ["location"],
                   "additionalProperties": False
               }
           }
       }]
   )
   ```

4. **Error Handling:**
   - Tools may be called with wrong parameters
   - Tools may be called when they shouldn't be
   - Implement validation before execution

#### Anthropic Tool Use

**Best Practices:**

1. **Tool Descriptions:**
   - 1,024 character limit per tool description
   - Use clear, imperative language
   - Specify when NOT to use the tool

2. **Computer Use (2024):**
   - Claude can control computer interfaces
   - Beta feature with extended capabilities
   - Useful for UI testing, web automation

### ToolRegistry Pattern
**Paper:** "ToolRegistry: Efficient Tool Management for LLM Agents"

**Concept:** Dynamic tool selection based on task context.

**Implementation:**
```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.tool_embeddings = {}

    def register(self, tool, description):
        self.tools[tool.name] = tool
        self.tool_embeddings[tool.name] = embed(description)

    def select_tools(self, query, top_k=5):
        query_embedding = embed(query)
        similarities = {
            name: cosine_similarity(query_embedding, emb)
            for name, emb in self.tool_embeddings.items()
        }
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

# Usage
registry = ToolRegistry()
relevant_tools = registry.select_tools(user_query, top_k=5)
```

**Benefits:**
- Reduces token usage
- Improves accuracy (fewer options = better selection)
- Scales to large tool libraries

### Comparative Analysis: GPT-4o vs Claude vs Gemini

**Function Calling Accuracy (2024):**
| Model | Complex Schema | Simple Schema | Overall |
|-------|---------------|---------------|---------|
| GPT-4o-2024-08-06 | 100% | 100% | 100% |
| GPT-4o | 95% | 98% | 96.5% |
| Claude 3.5 Sonnet | 92% | 96% | 94% |
| Gemini 1.5 Pro | 90% | 95% | 92.5% |
| GPT-4-0613 | 38% | 85% | 61.5% |

**Source:** OpenAI evaluations, 2024

---

## Extended Thinking and Reasoning

### Anthropic Extended Thinking (2024-2025)

**Core Concept:** "Serial test-time compute" - multiple sequential reasoning steps before final output.

**Claude 3.7 Sonnet Extended Thinking:**
- Accuracy improves logarithmically with thinking tokens
- Can use tools during extended thinking
- Two modes: instant response vs extended reasoning

**Performance Results:**
- GPQA (Graduate-Level Questions): 84.8% overall, 96.5% on physics
- AIME 2024 (Math Competition): 61.3% pass@1, 80.0% pass@64
- Improvement scales with allowed thinking tokens

**Implementation:**
```python
response = anthropic.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # Tokens for thinking
    },
    messages=[{"role": "user", "content": "Solve this complex problem..."}]
)

# Access thinking process
thinking = response.content[0].thinking
final_answer = response.content[1].text
```

**Use Cases:**
- Complex mathematical proofs
- Multi-step scientific reasoning
- Code debugging and optimization
- Strategic planning

### Claude 4 (May 2025)

**Hybrid Reasoning:**
- Opus 4 and Sonnet 4 offer dual modes
- Near-instant responses for simple queries
- Extended thinking for complex problems
- Tool use during thinking process

**Reasoning Faithfulness Research:**
**Paper:** "Reasoning Models Don't Always Say What They Think" (Anthropic, 2025)

**Key Finding:** Chain-of-thought outputs are less faithful on harder tasks.

**Implication for API Users:**
- Don't assume reasoning traces reflect actual model process
- Validate outputs independently
- Use multiple reasoning approaches for critical tasks

### OpenAI o1/o3 Reasoning Models

**Architecture:**
- Extended inference-time computation
- Trained with reinforcement learning to reason
- Performs well on complex STEM tasks

**Performance:**
- Competitive programming: Significantly outperforms GPT-4o
- PhD-level science questions: High accuracy
- Mathematical reasoning: Strong performance

**Cost Consideration:**
- More expensive than standard models
- Justified for complex reasoning tasks
- Not needed for simple queries

---

## Theoretical Framework for Architecture Selection

### Decision Tree for Single vs Multi-Agent

```
Is task complex with multiple distinct subtasks?
├─ No → Single Agent
└─ Yes
   ├─ Are subtasks independent?
   │  ├─ Yes → Parallel Multi-Agent
   │  └─ No → Sequential Multi-Agent
   └─ Do subtasks require different expertise?
      ├─ Yes → Heterogeneous Multi-Agent
      └─ No → Homogeneous Multi-Agent
```

### Collaboration vs Supervisor vs Swarm

**Collaboration (Shared Scratchpad):**
- **Use when:** Agents need to share context continuously
- **Theoretical basis:** Common ground theory
- **API efficiency:** Low (single shared state)
- **Example:** Code review where multiple agents examine same code

**Supervisor (Hierarchical):**
- **Use when:** Clear task decomposition possible
- **Theoretical basis:** Delegation and coordination theory
- **API efficiency:** Medium (one supervisor call + N worker calls)
- **Example:** SQL query generation (router → schema_lookup → query_builder)

**Swarm (Peer-to-Peer):**
- **Use when:** Dynamic, adaptive routing needed
- **Theoretical basis:** Emergent coordination
- **API efficiency:** Variable (agents decide next steps)
- **Example:** Complex research where agents follow leads dynamically

### Benchmarking Data (LangGraph, 2024)

**Single Agent Performance:**
```
Domains: 1-2  → High accuracy
Domains: 3+   → Sharp decline (>40% error rate)
Token usage: Scales linearly with domain count
```

**Multi-Agent Performance:**
```
Supervisor:
- Domains: 2-5 → Stable accuracy
- Token usage: Flat (~30% increase vs single agent)
- Maintainability: Modular

Swarm:
- Domains: 2-10 → Best accuracy
- Token usage: Flat (most efficient)
- Maintainability: Most modular
```

**Recommendation for API Users:**
- Single agent: 1-2 domains only
- Supervisor: 2-5 well-defined domains
- Swarm: 5+ domains or unclear domain boundaries

---

## Research Papers and Citations

### Core Reasoning Papers

1. **ReAct: Synergizing Reasoning and Acting in Language Models**
   - Yao et al., 2022
   - arXiv:2210.03629
   - https://arxiv.org/abs/2210.03629

2. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**
   - Wei et al., 2022
   - arXiv:2201.11903
   - https://arxiv.org/abs/2201.11903

3. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**
   - Yao et al., 2023
   - arXiv:2305.10601
   - https://arxiv.org/abs/2305.10601

### Multi-Agent Systems Papers

4. **Multi-Agent Collaboration Mechanisms: A Survey**
   - arXiv:2501.06322v1, 2025
   - Five-dimensional framework for multi-agent systems

5. **X-MAS: Extreme Multi-Agent Systems** (2025)
   - Large-scale agent coordination

6. **Infrastructure for AI Agents** (2025)
   - Scalable multi-agent deployments

### Tool Use and Function Calling

7. **ToolRegistry: Efficient Tool Management for LLM Agents**
   - Dynamic tool selection based on embeddings

8. **Model Context Protocol (MCP)**
   - Anthropic, 2025
   - Standardized context sharing for LLMs

### API-Specific Research

9. **GPT-4o System Card**
   - OpenAI, August 2024
   - Safety evaluations and capabilities

10. **Claude 3 Model Family: Opus, Sonnet, Haiku**
    - Anthropic Model Card, 2024

11. **Claude 3.5 Sonnet Model Card Addendum**
    - Anthropic, 2024

12. **Reasoning Models Don't Always Say What They Think**
    - Anthropic, 2025
    - Reasoning faithfulness research

### Communication Protocols

13. **Agent Communication Protocol (ACP)**
    - Multi-agent communication survey, 2025

14. **Agent-to-Agent Protocol (A2A)**
    - Peer-to-peer agent communication

### Best Practices Documentation

15. **Prompt Engineering Overview - Claude Docs**
    - https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview

16. **Structured Outputs in the API - OpenAI**
    - https://openai.com/index/introducing-structured-outputs-in-the-api/

17. **Function Calling and Other API Updates - OpenAI**
    - https://openai.com/index/function-calling-and-other-api-updates/

18. **Building with Extended Thinking - Claude API**
    - https://docs.claude.com/en/docs/build-with-claude/extended-thinking

---

## Key Takeaways for API Users

### 1. Choose the Right Reasoning Pattern
- **Simple tasks:** Direct prompting
- **Sequential reasoning:** Chain-of-Thought
- **Tool-augmented:** ReAct
- **Complex search:** Tree of Thoughts
- **Robustness:** Self-Consistency

### 2. Architect Multi-Agent Systems Thoughtfully
- **1-2 domains:** Stay with single agent
- **3-5 domains:** Use supervisor pattern
- **5+ domains:** Consider swarm pattern
- **Dynamic tasks:** Peer-to-peer with handoffs

### 3. Optimize Tool/Function Calling
- **Limit tools:** 5-10 per request (use RAG for selection)
- **Document thoroughly:** Clear descriptions essential
- **Use structured outputs:** Guarantee schema adherence (GPT-4o)
- **Validate inputs:** Don't trust model-generated parameters blindly

### 4. Leverage Extended Thinking Strategically
- **Complex reasoning:** Use Claude 3.7+ extended thinking
- **Cost-sensitive:** Use o1/o3 only when justified
- **Real-time:** Standard models for instant responses
- **Critical tasks:** Combine extended thinking with validation

### 5. Implement Robust Communication
- **Shared state:** Use LangGraph for simple coordination
- **Message passing:** For complex async patterns
- **Handoffs:** Explicit control flow with Command tool
- **Protocols:** Standardize with MCP when possible

### 6. Validate and Verify
- **Reasoning traces:** Don't assume faithfulness
- **Tool outputs:** Verify before using
- **Multi-agent outputs:** Cross-validate between agents
- **Critical decisions:** Use human-in-the-loop

---

## Conclusion

Building effective multi-agent systems with LLM APIs requires:

1. **Theoretical Understanding:** Know when and why to use different patterns
2. **API Knowledge:** Understand model-specific capabilities and limitations
3. **Architecture Selection:** Choose patterns based on task characteristics
4. **Optimization:** Balance cost, latency, and quality
5. **Validation:** Verify outputs and reasoning
6. **Continuous Learning:** Stay updated with latest research and capabilities

The research shows clear evolution from simple prompting to sophisticated multi-agent architectures. The key is matching the approach to the problem complexity while managing API costs and latency constraints.

For most production use cases in 2025:
- Start with single-agent ReAct for tool-augmented tasks
- Use supervisor pattern when you have 3-5 clear domains
- Reserve swarm patterns for truly complex, exploratory tasks
- Leverage extended thinking for critical reasoning tasks
- Always validate outputs, especially for high-stakes decisions

The theoretical foundations provided here should guide architecture decisions and implementation choices for robust, efficient multi-agent systems using commercial LLM APIs.
