# Agentic AI Developer Competency Checklist

**Purpose:** Track your progress from beginner to production-ready agent developer

**How to Use:**
- Check off skills as you develop them
- Each skill links to relevant learning material
- Use for self-assessment and identifying gaps
- Revisit periodically to track growth

---

## Phase 0: Prerequisites

### Python Fundamentals
- [ ] Can write and debug Python 3.10+ code
- [ ] Understand async/await and asyncio
- [ ] Can use type hints and Pydantic models
- [ ] Familiar with virtual environments (venv, poetry, uv)
- [ ] Can make HTTP requests with requests/httpx

**Learn:** [prerequisites.md](phase-0-prerequisites/prerequisites.md)

### Machine Learning Basics
- [ ] Can explain supervised vs unsupervised learning
- [ ] Understand what neural networks do (conceptually)
- [ ] Know when ML is overkill vs rule-based solutions
- [ ] Familiar with embeddings and vector similarity

**Learn:** [prerequisites.md](phase-0-prerequisites/prerequisites.md)

### LLM Fundamentals
- [ ] Understand tokenization and context windows
- [ ] Can explain temperature and sampling parameters
- [ ] Have made API calls to OpenAI, Anthropic, or similar
- [ ] Know the difference between completion and chat APIs

**Learn:** [llm-fundamentals.md](phase-1-foundations/llm-fundamentals.md)

---

## Phase 1: Foundations

### Agent Concepts
- [ ] Can explain what makes an agent different from a chatbot
- [ ] Understand the ReAct (Reason + Act) pattern
- [ ] Can describe Chain of Thought (CoT) prompting
- [ ] Know when to use Tree of Thought (ToT) vs linear reasoning
- [ ] Understand LATS (Language Agent Tree Search)

**Learn:** [theoretical-foundations.md](phase-1-foundations/theoretical-foundations.md)

### Framework Knowledge
- [ ] Can compare LangGraph, CrewAI, and OpenAI Agents SDK
- [ ] Know when to use each framework (use case mapping)
- [ ] Understand state management differences
- [ ] Can read and understand framework documentation

**Learn:** [framework-comparison.md](phase-1-foundations/framework-comparison.md)

### Decision Making
- [ ] Can decide when to use agents vs traditional automation
- [ ] Know the 5-question agent necessity filter
- [ ] Understand single vs multi-agent trade-offs
- [ ] Can evaluate build vs buy decisions

**Learn:** [reference/topics.md](reference/topics.md) Q82-Q87

---

## Phase 2: Building Agents

### Prompting
- [ ] Can write effective system prompts for agents
- [ ] Understand XML/structured prompting patterns
- [ ] Can design prompts that prevent common failures
- [ ] Know how to prompt for tool usage

**Learn:** [agent-prompting-guide.md](phase-2-building-agents/agent-prompting-guide.md)

### Tool Development
- [ ] Can create custom tools with proper schemas
- [ ] Understand tool input/output validation
- [ ] Can implement error handling in tools
- [ ] Know tool design best practices (single responsibility)

**Learn:** [tool-development-guide.md](phase-2-building-agents/tool-development-guide.md)

### MCP Servers
- [ ] Understand the MCP protocol architecture
- [ ] Can build a basic MCP server in Python
- [ ] Can build a basic MCP server in TypeScript
- [ ] Know how to register and expose tools via MCP

**Learn:** [tool-development-guide.md](phase-2-building-agents/tool-development-guide.md), [mcp-deep-dive.md](phase-6-advanced/mcp-deep-dive.md)

### Memory Systems
- [ ] Understand short-term vs long-term memory
- [ ] Can implement conversation memory
- [ ] Know vector database options (Pinecone, Chroma, etc.)
- [ ] Understand memory retrieval strategies

**Learn:** [memory-systems-guide.md](phase-2-building-agents/memory-systems-guide.md)

### First Agent
- [ ] Have built a working single-agent system
- [ ] Agent can use multiple tools
- [ ] Agent handles errors gracefully
- [ ] Agent maintains conversation context

**Learn:** [agentic-systems-cookbook.md](phase-2-building-agents/agentic-systems-cookbook.md)

---

## Phase 3: Patterns

### Multi-Agent Architecture
- [ ] Can implement Supervisor pattern
- [ ] Can implement Swarm pattern with handoffs
- [ ] Understand Hierarchical pattern for large systems
- [ ] Know when to use each pattern

**Learn:** [multi-agent-patterns.md](phase-3-patterns/multi-agent-patterns.md)

### Coordination
- [ ] Understand agent communication patterns
- [ ] Can implement shared state management
- [ ] Know how to handle agent conflicts
- [ ] Can design agent handoff protocols

**Learn:** [multi-agent-patterns.md](phase-3-patterns/multi-agent-patterns.md)

### Anti-Patterns
- [ ] Know the 14 common failure modes
- [ ] Can identify specification failures
- [ ] Understand role violation problems
- [ ] Know context management pitfalls

**Learn:** [patterns-and-antipatterns.md](phase-3-patterns/patterns-and-antipatterns.md)

### 12-Factor Agents
- [ ] Understand all 12 factors
- [ ] Can apply factors to agent design
- [ ] Know which factors are non-negotiable
- [ ] Can audit agents against the 12 factors

**Learn:** [patterns-and-antipatterns.md](phase-3-patterns/patterns-and-antipatterns.md) Section 11

---

## Phase 4: Production

### Testing
- [ ] Can mock LLM responses for testing
- [ ] Can write unit tests for agent tools
- [ ] Understand integration testing for agents
- [ ] Know how to create golden datasets

**Learn:** [testing-guide.md](phase-4-production/testing-guide.md)

### Evaluation
- [ ] Can use LangSmith for tracing
- [ ] Understand EDD (Evaluation-Driven Development)
- [ ] Know key benchmarks (SWE-bench, BFCL)
- [ ] Can evaluate agent performance objectively

**Learn:** [evaluation-and-debugging.md](phase-4-production/evaluation-and-debugging.md)

### Cost Optimization
- [ ] Understand model cascading for cost savings
- [ ] Can implement prompt caching
- [ ] Know tool RAG patterns for efficiency
- [ ] Can estimate and monitor costs

**Learn:** [api-optimization-guide.md](phase-4-production/api-optimization-guide.md)

### Deployment
- [ ] Can containerize agents with Docker
- [ ] Understand CI/CD for agent systems
- [ ] Know blue-green and canary deployment patterns
- [ ] Can set up monitoring and alerting

**Learn:** [ci-cd-guide.md](phase-4-production/ci-cd-guide.md)

### Agent Maturity
- [ ] Understand L1-L5 maturity levels
- [ ] Can assess current agent maturity
- [ ] Know minimum production-ready level (L3)
- [ ] Can plan maturity advancement

**Learn:** [evaluation-and-debugging.md](phase-4-production/evaluation-and-debugging.md) Section 12

---

## Phase 5: Security & Compliance

### Prompt Security
- [ ] Understand prompt injection attack vectors
- [ ] Can implement multi-layer injection defense
- [ ] Know input validation patterns
- [ ] Can detect and block malicious prompts

**Learn:** [security-essentials.md](phase-5-security-compliance/security-essentials.md)

### Tool Security
- [ ] Can implement tool sandboxing
- [ ] Understand permission scoping
- [ ] Know output filtering patterns
- [ ] Can audit tool access

**Learn:** [security-essentials.md](phase-5-security-compliance/security-essentials.md)

### Human-in-the-Loop
- [ ] Can implement risk-based approval flows
- [ ] Know when to require human oversight
- [ ] Understand escalation patterns
- [ ] Can design effective HITL interfaces

**Learn:** [security-essentials.md](phase-5-security-compliance/security-essentials.md)

### Compliance
- [ ] Understand EU AI Act requirements
- [ ] Know GDPR implications for agents
- [ ] Can implement audit logging
- [ ] Understand data retention requirements

**Learn:** [governance-compliance.md](phase-5-security-compliance/governance-compliance.md)

---

## Phase 6: Advanced

### Self-Improving Agents
- [ ] Understand Reflexion pattern
- [ ] Know LADDER and skill acquisition
- [ ] Can implement feedback loops
- [ ] Understand AlphaEvolve concepts

**Learn:** [advanced-agent-paradigms.md](phase-6-advanced/advanced-agent-paradigms.md)

### Advanced MCP
- [ ] Can build complex MCP servers
- [ ] Understand server federation
- [ ] Know dynamic tool discovery
- [ ] Can implement MCP authentication

**Learn:** [mcp-deep-dive.md](phase-6-advanced/mcp-deep-dive.md)

### DSPy
- [ ] Understand programmatic prompting
- [ ] Can create DSPy signatures and modules
- [ ] Know optimization strategies
- [ ] Can integrate DSPy with agent frameworks

**Learn:** [dspy-guide.md](phase-6-advanced/dspy-guide.md)

### Framework Migration
- [ ] Know when to switch frameworks
- [ ] Understand migration strategies
- [ ] Can abstract framework-agnostic interfaces
- [ ] Have migrated at least one agent

**Learn:** [cross-framework-migration.md](phase-6-advanced/cross-framework-migration.md)

---

## Product & Strategy

### Build vs Buy
- [ ] Can evaluate vendor solutions objectively
- [ ] Understand TCO calculations
- [ ] Know when to build custom
- [ ] Can create business cases for agents

**Learn:** [product-strategy-guide.md](product-strategy/product-strategy-guide.md)

### Team Workflows
- [ ] Know AI-assisted development patterns
- [ ] Can set up team AI workflows
- [ ] Understand reference application anchoring
- [ ] Can use RIPER or similar frameworks

**Learn:** [developer-productivity-guide.md](developer-productivity/developer-productivity-guide.md)

---

## Mastery Indicators

### Beginner Complete
You should have all Phase 0-1 items checked before building production agents.

### Intermediate Complete
You should have all Phase 0-3 items checked for multi-agent systems.

### Production Ready
You should have all Phase 0-5 items checked before deploying to production.

### Expert
All items checked + you're contributing back to the community.

---

## Related Documents

- [README.md](README.md) - Learning path overview
- [CLAUDE.md](CLAUDE.md) - Contribution guidelines
- [reference/topics.md](reference/topics.md) - Quick reference Q&A
