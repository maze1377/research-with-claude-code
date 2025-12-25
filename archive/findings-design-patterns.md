# Enterprise Agentic AI Design Patterns

**Sources**:
- Google Cloud Architecture Center
- Microsoft Azure AI Foundry
- Industry best practices (2025)

## Pattern Catalog

This document synthesizes design patterns from leading cloud providers for building production-grade agentic AI systems.

---

## Single-Agent Pattern

### Description
One AI model with comprehensive system prompt and defined tool set manages user requests autonomously.

### Architecture
```
User Query → [AI Agent + Tools + System Prompt] → Actions → Results
```

### Characteristics
- **Complexity**: Low
- **Latency**: Low (single model call)
- **Cost**: Low (minimal token usage)
- **Flexibility**: Medium

### Use Cases
- Customer support queries requiring data lookup
- Research assistance with web search
- Simple multi-step automation
- Personal assistant tasks

### Advantages
✓ Simplest to implement and maintain
✓ Ideal for tasks requiring multiple steps and external data access
✓ Low operational overhead
✓ Good starting point for agent development

### Limitations
✗ Effectiveness degrades with numerous tools (tool confusion)
✗ Increased errors in complex scenarios
✗ Limited specialization
✗ No quality control layer

### When to Use
- Tasks have clear, linear workflows
- Tool count < 10
- Complexity is moderate
- Cost/latency optimization critical

---

## Sequential Multi-Agent Pattern

### Description
Specialized agents execute in predefined linear order; output from one agent becomes input for next.

### Architecture
```
Input → Agent 1 → Agent 2 → Agent 3 → ... → Agent N → Output
```

### Characteristics
- **Complexity**: Medium
- **Latency**: Medium (sequential model calls)
- **Cost**: Medium (N model calls)
- **Flexibility**: Low (fixed path)

### Use Cases
- ETL pipelines (Extract → Transform → Load)
- Document processing (Parse → Analyze → Summarize)
- Content creation (Draft → Edit → Format → Publish)
- Compliance workflows (Check → Validate → Approve)

### Advantages
✓ Reduced latency vs. coordinator pattern (no orchestration overhead)
✓ Lower costs (no AI orchestration layer needed)
✓ Predictable execution path
✓ Clear responsibility boundaries
✓ Easy to debug and monitor

### Limitations
✗ Cannot skip unnecessary steps
✗ No dynamic adaptation
✗ Cannot handle branching logic
✗ All agents execute even if not needed

### When to Use
- Highly structured, repeatable processes
- Linear workflows with no branching
- Steps always execute in same order
- Predictability more important than flexibility

---

## Parallel Multi-Agent Pattern

### Description
Multiple specialized subagents execute simultaneously; outputs synthesized into consolidated response.

### Architecture
```
Input → Dispatcher
         ↓ (parallel)
      [Agent A, Agent B, Agent C, Agent D]
         ↓ (gather)
      Synthesizer → Output
```

### Characteristics
- **Complexity**: Medium
- **Latency**: Low (parallelization reduces wait time)
- **Cost**: High (multiple concurrent calls)
- **Flexibility**: Medium

### Use Cases
- Multi-source data gathering (news, social media, databases)
- Parallel research (multiple topics simultaneously)
- Option evaluation (comparing multiple approaches)
- Multi-modal analysis (text + image + audio simultaneously)

### Advantages
✓ Dramatically reduces overall latency
✓ Independent task isolation
✓ Efficient resource utilization
✓ Fault tolerance (one agent failure doesn't block others)

### Limitations
✗ Increased resource consumption
✗ Higher token usage and costs
✗ Synthesis complexity (combining diverse outputs)
✗ Requires conflict resolution logic

### When to Use
- Independent subtasks can run concurrently
- Latency is critical bottleneck
- Tasks require data from disparate sources
- Resources available for parallel execution

---

## Loop Pattern (Iterative Refinement)

### Description
Specialized subagents execute repeatedly until specific termination condition met.

### Architecture
```
Input → Agent → Evaluate → [Continue: Loop | Done: Output]
         ↑________________________|
```

### Characteristics
- **Complexity**: Medium-High
- **Latency**: Variable (depends on iterations)
- **Cost**: Variable (proportional to iterations)
- **Flexibility**: High

### Use Cases
- Iterative code debugging
- Progressive plan refinement
- Long-form document drafting with revisions
- Self-correction and quality improvement
- Simulation and optimization

### Advantages
✓ Produces sophisticated outputs difficult in single pass
✓ Self-improving through iterations
✓ Handles complex generation tasks effectively
✓ Adapts to quality requirements

### Limitations
✗ Cumulative latency with each iteration
✗ Costs increase linearly with iterations
✗ **Risk of infinite loops** if termination poorly defined
✗ Difficult to predict completion time

### Critical Implementation Detail
**Must have robust termination conditions**:
- Maximum iteration limit (safety net)
- Quality threshold metrics
- Timeout constraints
- Convergence detection

### When to Use
- Output quality more important than speed
- Single-pass generation insufficient
- Self-correction needed
- Iterative improvement natural fit

---

## Review and Critique Pattern

### Description
Generator agent creates output; separate critic agent evaluates against predefined criteria; provides approval or revision feedback.

### Architecture
```
Input → Generator → Output
                     ↓
                  Critic → [Approve: Done | Revise: Feedback]
                     ↓
                 Generator (with feedback)
```

### Characteristics
- **Complexity**: Medium
- **Latency**: Medium-High (2+ model calls)
- **Cost**: Medium-High
- **Flexibility**: High

### Use Cases
- **Code generation**: Security audits, style compliance
- **Content creation**: Fact-checking, tone validation
- **Compliance**: Regulatory requirement verification
- **Quality assurance**: Output validation against standards
- **Safety**: Harmful content detection

### Advantages
✓ Dramatically improves output quality, accuracy, and reliability
✓ Dedicated verification layer
✓ Catches errors before delivery
✓ Separates generation from validation concerns
✓ Enables specialized critic expertise

### Limitations
✗ Increased latency (minimum 2 model calls)
✗ Higher operational costs
✗ Potential disagreement loops
✗ Requires well-defined evaluation criteria

### Implementation Variants

**Single-pass critic**:
```
Generate → Critique → Done (accept or reject)
```

**Iterative critic** (combines with loop pattern):
```
Generate → Critique → [Pass: Done | Fail: Regenerate with feedback]
```

### When to Use
- High-accuracy requirements
- Output errors have significant consequences
- Regulatory compliance needed
- Specialized validation expertise required
- Quality more important than speed

---

## Coordinator Pattern (Dynamic Routing)

### Description
Central coordinator agent analyzes requests, decomposes into subtasks, dispatches to specialized agents using AI-based orchestration.

### Architecture
```
Input → Coordinator (LLM) → [Analyze → Decompose → Route]
                              ↓
                          [Agent A, Agent B, Agent C] (conditional)
                              ↓
                          Coordinator → Synthesize → Output
```

### Characteristics
- **Complexity**: High
- **Latency**: Medium-High (orchestration overhead + agent calls)
- **Cost**: Medium-High (coordinator + N agents)
- **Flexibility**: Very High

### Use Cases
- Customer service routing (billing/technical/sales)
- Adaptive business process automation
- Multi-domain question answering
- Dynamic workflow orchestration
- Context-specific task handling

### Advantages
✓ Maximum flexibility through dynamic routing
✓ No predefined scripts needed
✓ Adapts to diverse request types
✓ Efficient resource allocation (only needed agents execute)
✓ Handles unpredictable workflows

### Limitations
✗ Multiple model calls increase token throughput
✗ Higher costs (coordinator + agents)
✗ Increased latency (orchestration layer)
✗ Coordinator can make routing errors
✗ More complex to debug

### Key Differentiator
Unlike sequential pattern, **only necessary agents execute**, providing flexibility without waste.

### When to Use
- Diverse request types require different handling
- Cannot predetermine execution paths
- Specialized agents for different domains
- Flexibility more valuable than cost optimization

---

## Hierarchical Task Decomposition Pattern

### Description
Multi-level agent hierarchy; root agent decomposes complex tasks into subtasks delegated to lower-level specialized agents (recursive decomposition).

### Architecture
```
Complex Task
    ↓
Root Agent (Decompose)
    ↓
[Subtask 1, Subtask 2, Subtask 3]
    ↓
[Agent Layer 1] → Further decomposition if needed
    ↓
[Agent Layer 2] → Specialized execution
    ↓
Synthesis ← Aggregation ← Bottom-up
```

### Characteristics
- **Complexity**: Very High
- **Latency**: High (multiple hierarchical levels)
- **Cost**: High (many model calls across levels)
- **Flexibility**: Very High

### Use Cases
- Large-scale research and analysis projects
- Complex software development (architect → designer → coder → tester)
- Strategic planning (corporate → divisional → team → individual)
- Multi-stage synthesis tasks
- Ambiguous, open-ended problem solving

### Advantages
✓ Systematically manages highly complex problems
✓ Produces comprehensive, high-quality results
✓ Natural fit for naturally hierarchical tasks
✓ Enables specialization at each level
✓ Handles ambiguous requirements through progressive refinement

### Limitations
✗ **Significant architectural complexity**
✗ **Multiple layers multiply model calls exponentially**
✗ **Substantial latency** from cascading dependencies
✗ **Very high operational costs**
✗ Difficult to debug and monitor
✗ Coordination overhead across levels

### When to Use
- Extremely complex, multi-faceted problems
- Hierarchical decomposition is natural fit
- Quality and comprehensiveness justify cost
- Budget allows for extensive model usage

### Warning
This is the most resource-intensive pattern. Use only when simpler patterns insufficient.

---

## Swarm Pattern (Collaborative Debate)

### Description
Multiple specialized agents collaborate with all-to-all communication; dispatcher routes requests; agents iteratively refine solutions through debate and consensus.

### Architecture
```
Input → Dispatcher
         ↓
    [Agent A ←→ Agent B ←→ Agent C]
         ↓ (iterative communication)
    Consensus/Synthesis → Output
```

### Characteristics
- **Complexity**: Very High (highest)
- **Latency**: High (many iterative exchanges)
- **Cost**: Very High (N × M interactions)
- **Flexibility**: Extreme

### Use Cases
- Product design with competing requirements (cost/quality/speed)
- Complex decision-making requiring diverse perspectives
- Creative problem-solving
- Multi-objective optimization
- Scenarios requiring debate and consensus

### Advantages
✓ **Produces exceptionally high-quality and creative solutions**
✓ Simulates expert collaboration and debate
✓ Diverse perspectives improve outcomes
✓ Identifies edge cases through disagreement
✓ Novel solutions from agent interaction

### Limitations
✗ **Most complex pattern to implement**
✗ **Lacks centralized orchestration oversight**
✗ **Risk of unproductive loops** (agents disagree indefinitely)
✗ **Highest operational costs** (many agents × many interactions)
✗ Difficult to predict completion
✗ Requires sophisticated termination logic

### Critical Requirements
- Robust consensus mechanisms
- Deadlock prevention
- Maximum iteration limits
- Contribution quality monitoring

### When to Use
- Extremely complex problems benefit from debate
- Quality justifies very high cost
- Creative/novel solutions needed
- Multiple valid perspectives exist

### Warning
Most expensive and complex pattern. Reserve for cases where **exceptional quality justifies exceptional cost**.

---

## ReAct Pattern (Reason and Act)

### Description
Iterative loop combining reasoning traces with action execution; agent thinks, acts, observes, then repeats until task completion.

### Architecture
```
Input → Think (Reasoning) → Act (Tool/Query) → Observe (Results)
         ↑__________________________________________|
         (Loop until done)
```

### Characteristics
- **Complexity**: Medium
- **Latency**: Variable (depends on iterations)
- **Cost**: Variable (proportional to loops)
- **Flexibility**: Very High

### Workflow Detail
```
Thought: "I need to find the weather in Paris"
Action: search("Paris weather")
Observation: "Sunny, 22°C"
Thought: "Now I need to compare with London"
Action: search("London weather")
Observation: "Rainy, 15°C"
Thought: "I have enough information to answer"
Action: respond("Paris is warmer and sunnier than London today")
```

### Use Cases
- Complex, dynamic problem-solving
- Tasks requiring continuous adaptation
- Robotic path planning
- Research with unknown information needs
- Troubleshooting and debugging

### Advantages
✓ Dynamic planning adapts to real-time conditions
✓ Transparent reasoning transcript aids debugging
✓ Self-correcting through observations
✓ Handles unforeseen obstacles effectively
✓ Natural fit for exploratory tasks

### Limitations
✗ Higher end-to-end latency (iterative loops)
✗ Effectiveness dependent on model reasoning quality
✗ Errors can propagate through iterations
✗ Unpredictable token usage

### Key Insight
ReAct is **foundational to modern agentic AI**. Most agent frameworks (LangGraph, OpenManus, AutoGPT) use ReAct as default paradigm.

### When to Use
- Static plans insufficient
- Environment dynamic and unpredictable
- Task requires exploration
- Transparent reasoning valuable

---

## Human-in-the-Loop Pattern

### Description
Agent pauses execution at predefined checkpoints; waits for external human review/approval before continuing.

### Architecture
```
Input → Agent Processing → Checkpoint (Pause)
                              ↓
                          Human Review
                              ↓
                         [Approve | Reject | Modify]
                              ↓
                          Continue Execution → Output
```

### Characteristics
- **Complexity**: Medium-High
- **Latency**: High (human wait time)
- **Cost**: Medium (model + human time)
- **Flexibility**: Very High

### Use Cases
- **High-stakes decisions**: Financial transactions, legal documents
- **Safety-critical operations**: Medical diagnoses, infrastructure changes
- **Subjective approvals**: Creative content, brand compliance
- **Compliance**: Regulatory review requirements
- **Sensitive data**: Privacy, security validations

### Advantages
✓ Integrates human judgment at critical decision points
✓ Dramatically improves safety and reliability
✓ Ensures compliance with human oversight requirements
✓ Catches AI errors before consequences
✓ Builds user trust through transparency

### Limitations
✗ Requires building external user interaction systems
✗ Significantly increases architectural complexity
✗ Latency depends on human availability
✗ Bottleneck if approvals slow
✗ Scalability limited by human capacity

### Implementation Considerations
- **Checkpoint placement**: Where to pause for review
- **Context presentation**: What information reviewer needs
- **Approval interface**: How humans interact
- **Timeout handling**: What happens if no response
- **Escalation**: Who reviews if first reviewer unavailable

### When to Use
- Consequences of errors are severe
- Regulatory compliance requires human oversight
- Trust building critical for adoption
- Subjective judgment needed
- Safety paramount

---

## Custom Logic Pattern

### Description
Code-based orchestration using conditional statements for complex workflows with multiple branching paths; mix of programmatic rules and AI reasoning.

### Architecture
```python
if condition_A:
    result = agent_1(input)
elif condition_B:
    result = agent_2(input)
    if result.needs_validation:
        result = validator_agent(result)
else:
    result = fallback_agent(input)

# Further nested logic...
```

### Characteristics
- **Complexity**: Very High (implementation-dependent)
- **Latency**: Variable
- **Cost**: Variable
- **Flexibility**: Maximum

### Use Cases
- Workflows not fitting standard patterns
- Complex business logic with many branches
- Legacy system integration
- Hybrid AI + traditional automation
- Unique organizational requirements

### Advantages
✓ **Maximum flexibility** for custom requirements
✓ Enables complex branching logic mixing rules with reasoning
✓ Can optimize specific paths for performance
✓ Complete control over execution flow
✓ Can integrate non-AI components seamlessly

### Limitations
✗ **Increases development and maintenance complexity significantly**
✗ Requires comprehensive design, implementation, debugging responsibility
✗ Difficult to visualize and communicate
✗ Error-prone with many branches
✗ Testing complexity high

### When to Use
- Standard patterns genuinely insufficient
- Unique business requirements
- Need to mix AI with traditional logic extensively
- Have engineering capacity for custom development

### Warning
Default to standard patterns first. Use custom logic only when necessary—complexity cost is high.

---

## Pattern Selection Guide

### By Task Complexity

| Complexity | Pattern |
|-----------|---------|
| Simple | Single-Agent |
| Linear | Sequential Multi-Agent |
| Independent Parallel | Parallel Multi-Agent |
| Iterative | Loop / Review-Critique |
| Dynamic Routing | Coordinator |
| Hierarchical | Hierarchical Decomposition |
| Collaborative | Swarm |
| Adaptive | ReAct |

### By Priority

| Priority | Pattern |
|----------|---------|
| Low Latency | Single-Agent, Parallel |
| Low Cost | Single-Agent, Sequential |
| High Quality | Review-Critique, Swarm, Loop |
| Flexibility | Coordinator, ReAct, Custom Logic |
| Safety | Human-in-the-Loop, Review-Critique |
| Scalability | Sequential, Coordinator |

### By Use Case

| Use Case | Recommended Pattern |
|----------|-------------------|
| Customer Support | Single-Agent → Coordinator (as complexity grows) |
| Data Pipeline | Sequential Multi-Agent |
| Research | Parallel + ReAct |
| Code Generation | ReAct + Review-Critique |
| Content Creation | Loop + Review-Critique |
| Complex Analysis | Hierarchical Decomposition |
| Creative Design | Swarm |
| Compliance | Human-in-the-Loop + Review-Critique |

---

## Pattern Composition

Patterns can be **combined** for sophisticated workflows:

### Example: Enterprise Document Processing
```
Input (Coordinator)
   ↓
Route by document type
   ↓
[Legal: Sequential Pipeline] [Financial: Parallel Analysis]
   ↓                              ↓
Review-Critique                Review-Critique
   ↓                              ↓
Human-in-the-Loop (if high-stakes)
   ↓
Output
```

### Example: AI Software Development
```
Requirements (ReAct) → Architect (Hierarchical Decomposition)
                          ↓
                      [UI Team, Backend Team, DB Team] (Parallel)
                          ↓
                      Integration (Sequential)
                          ↓
                      Testing (Loop until pass)
                          ↓
                      Security Review (Review-Critique)
                          ↓
                      Deployment Approval (Human-in-the-Loop)
```

---

## Industry Trends (2025)

### Adoption Forecast
- **Gartner**: 33% of enterprise software will incorporate agentic AI by 2028 (up from <1% in 2024)
- **Current State**: Moving from prototypes to production

### Common Patterns in Production
1. **ReAct** - Default for most agents
2. **Coordinator** - Common for multi-domain applications
3. **Review-Critique** - Standard for quality-critical applications
4. **Human-in-the-Loop** - Required for regulated industries

### Emerging Patterns
- **Meta-agent coordination** - Agents managing other agents
- **Persistent memory** - Agents learning across sessions
- **Multi-agent planning loops** - Collaborative planning before execution
- **Semantic communication** - Agents using structured protocols

---

## Summary

| Pattern | Complexity | Cost | Latency | Quality | Use When |
|---------|-----------|------|---------|---------|----------|
| Single-Agent | ⭐ | $ | Fast | Good | Simple tasks |
| Sequential | ⭐⭐ | $$ | Medium | Good | Linear workflows |
| Parallel | ⭐⭐ | $$$ | Fast | Good | Independent tasks |
| Loop | ⭐⭐⭐ | $$-$$$ | Slow | Excellent | Iterative improvement |
| Review-Critique | ⭐⭐ | $$ | Medium | Excellent | Quality-critical |
| Coordinator | ⭐⭐⭐ | $$-$$$ | Medium | Good | Dynamic routing |
| Hierarchical | ⭐⭐⭐⭐ | $$$$ | Slow | Excellent | Complex decomposition |
| Swarm | ⭐⭐⭐⭐⭐ | $$$$$ | Slow | Outstanding | Creative/complex |
| ReAct | ⭐⭐ | $$-$$$ | Variable | Very Good | Adaptive tasks |
| Human-in-the-Loop | ⭐⭐⭐ | $$+ | Very Slow | Excellent | High-stakes |
| Custom Logic | ⭐⭐⭐⭐⭐ | Variable | Variable | Variable | Unique requirements |

**Key**: ⭐ = Complexity level, $ = Cost level
