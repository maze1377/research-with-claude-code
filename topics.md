# Agentic Systems Mastery: Quick Reference Guide

**Purpose:** Fast-lookup reference for building production multi-agent systems. For detailed implementations, see linked research documents.

**Last Updated:** 2025-12-25

**Knowledge Foundation:** 18 comprehensive documents, 60+ academic papers, 14 failure modes, 11 production recipes, 5 case studies, December 2025 benchmarks, complete security research, browser automation, memory systems, MAST failure taxonomy, RAFA framework

---

## Practical Guides

### Getting Started: Your First Agent (30 Minutes)

**Choose Your SDK:**

| SDK | Best For | Complexity | Language |
|-----|----------|------------|----------|
| **OpenAI Agents SDK** | Quick start, multi-agent | Low | Python/JS |
| **LangGraph** | Complex workflows, state management | Medium | Python |
| **Claude Agent SDK** | Planning-heavy, long-running tasks | Medium | Python |

**Option 1: OpenAI Agents SDK (Simplest)** - v0.6.4 (Dec 2025)
```
pip install openai-agents
```

```python
from agents import Agent, Runner

# Define agent with instructions
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. Answer concisely.",
    model="gpt-4o-mini"  # Supports 100+ models
)

# Run agent
result = await Runner.run(agent, "What is 2+2?")
print(result.final_output)
```

**Key primitives:** Agents (LLM + instructions + tools), Handoffs (delegate to other agents), Guardrails (input/output validation), Sessions (conversation history), Tracing (built-in with LangSmith/Logfire).

**Option 2: LangGraph (Most Flexible)**
```
pip install langgraph langchain-openai
```

```python
from langgraph.graph import StateGraph, MessagesState
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def agent_node(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.set_finish_point("agent")

app = graph.compile()
result = app.invoke({"messages": [("user", "Hello!")]})
```

**Key concepts:** Nodes (computation units), Edges (connections), State (shared memory), Checkpoints (persistence).

**Option 3: Claude Agent SDK (Planning-First)**
```
pip install anthropic  # Claude Agent SDK is part of Anthropic SDK
```

Built on Claude Code's production agent harness. Key features:
- **Skills system**: Markdown files teaching Claude domain expertise
- **Hooks**: Shell commands on lifecycle events (format on edit, notify on idle)
- **Plugins**: Package and share skills + MCP servers + hooks
- **MCP Integration**: 2000+ servers for tools/resources
- **Context compaction**: Auto-summarize when context fills
- Default model: Claude Sonnet 4.5 (Claude Opus 4.5 for complex)

**Adding Tools (All SDKs):**
```python
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return results

# OpenAI SDK: Pass function directly
agent = Agent(tools=[search_web])

# LangGraph: Bind to model
llm_with_tools = llm.bind_tools([search_web])
```

**First 30 Minutes Checklist:**
- [ ] Install SDK (`pip install`)
- [ ] Set API key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)
- [ ] Create simple agent (copy example above)
- [ ] Run basic query
- [ ] Add one tool
- [ ] Test tool calling

**Common First-Time Mistakes:**
1. Forgetting async/await (OpenAI SDK is async)
2. Not handling tool errors
3. Missing API key environment variable
4. Overly complex first agent (start simple!)

**Sources:** [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/quickstart/), [LangGraph Docs](https://langchain-ai.github.io/langgraph/), [Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)

---

### Framework Selection Decision Tree

```
START: What's your primary need?
│
├─► Speed to market (weeks, not months)
│   └─► OpenAI Agents SDK or CrewAI
│       • Minimal boilerplate
│       • Pre-built patterns
│       • Quick prototyping
│
├─► Complex state management
│   └─► LangGraph
│       • Checkpoints & persistence
│       • Human-in-the-loop built-in
│       • Time travel debugging
│
├─► Long-running planning tasks
│   └─► Claude Agent SDK
│       • Auto-planning & decomposition
│       • Context compaction
│       • Agent Skills system
│
├─► Role-based collaboration
│   └─► CrewAI
│       • Define agents by role
│       • Sequential/hierarchical flows
│       • 12M+ daily executions
│
├─► Maximum control & customization
│   └─► LangGraph or Custom
│       • Build from primitives
│       • Full graph control
│       • Enterprise patterns
│
└─► Research/experimental
    └─► AutoGPT or Custom
        • Autonomous exploration
        • Self-prompting
        • Novel architectures
```

**Quick Comparison Matrix:**

| Factor | OpenAI SDK | LangGraph | Claude SDK | CrewAI |
|--------|-----------|-----------|------------|--------|
| Learning curve | Low | Medium | Medium | Low |
| Multi-agent | ✅ Handoffs | ✅ Full control | ⚠️ Limited | ✅ Role-based |
| State management | Sessions | ✅ Checkpoints | Compact | Basic |
| Human-in-loop | Guardrails | ✅ Built-in | Manual | Manual |
| Production ready | ✅ | ✅ | ✅ | ✅ |
| Best model | Any (100+ LLMs) | Any | Claude | Any |

**Decision Quick Guide:**
- **"I need something working today"** → OpenAI Agents SDK
- **"I need complex workflows with state"** → LangGraph
- **"I need planning and long-running tasks"** → Claude Agent SDK
- **"I want role-based agent teams"** → CrewAI
- **"I need maximum flexibility"** → LangGraph + custom code

**Detailed Framework Comparison (December 2025):**

| Aspect | LangGraph | CrewAI | AutoGen | OpenAI SDK |
|--------|-----------|--------|---------|------------|
| **Design** | Graph-based workflows | Role-based teams | Conversational agents | Handoff primitives |
| **Learning curve** | Steep (graphs, states) | Easy (roles, tasks) | Moderate (chat patterns) | Low (minimal concepts) |
| **Multi-agent** | Full control | Built-in collaboration | Async conversations | Handoff system |
| **Memory** | Checkpoints, persistence | ChromaDB + SQLite | Manual | Sessions |
| **Best for** | Complex state management | Team collaboration | Dialogue-heavy apps | Quick prototypes |
| **Future** | Active development | Active development | **Migrating to MS Agent Framework** | Active development |

**Important Note on AutoGen:**
Microsoft has launched Agent Framework, combining AutoGen and Semantic Kernel. AutoGen will only receive bug fixes (no new features). If using AutoGen, plan migration to Microsoft Agent Framework.

**Reference:** framework-comparison.md

---

### Security Hardening Checklist

**Pre-Deployment Security (Must-Have):**

```
[ ] INPUT VALIDATION
    ├─ [ ] Pattern matching for injection attacks
    ├─ [ ] Encoding detection (base64, hex, URL)
    ├─ [ ] Length limits on all inputs
    ├─ [ ] Content type validation
    └─ [ ] Rate limiting per user/IP

[ ] OUTPUT FILTERING
    ├─ [ ] PII detection and redaction
    ├─ [ ] Credential/API key scanning
    ├─ [ ] System prompt leak detection
    ├─ [ ] Harmful content filtering
    └─ [ ] Response length limits

[ ] TOOL SANDBOXING
    ├─ [ ] Process isolation (minimum)
    ├─ [ ] Container isolation (recommended)
    ├─ [ ] VM isolation (high-risk tools)
    ├─ [ ] File system restrictions
    ├─ [ ] Network restrictions
    └─ [ ] Resource limits (CPU, memory, time)

[ ] AUTHENTICATION & AUTHORIZATION
    ├─ [ ] API key rotation policy
    ├─ [ ] Per-tool permission matrix
    ├─ [ ] User role-based access
    └─ [ ] Audit logging enabled

[ ] HUMAN-IN-THE-LOOP
    ├─ [ ] Risk-based approval levels defined
    ├─ [ ] Approval timeout configured
    ├─ [ ] Escalation path documented
    └─ [ ] Critical actions blocked by default
```

**Runtime Security:**
```
[ ] MONITORING
    ├─ [ ] Injection attempt alerts (>10/hour)
    ├─ [ ] Anomaly detection enabled
    ├─ [ ] Failed auth alerts (>50/hour)
    └─ [ ] Cost spike alerts

[ ] INCIDENT RESPONSE
    ├─ [ ] P0 runbook documented
    ├─ [ ] Kill switch ready
    ├─ [ ] Log preservation automated
    └─ [ ] Notification channels configured
```

**Compliance Quick Check:**
- [ ] **EU AI Act**: Risk level determined, documentation ready
- [ ] **GDPR**: Lawful basis, data minimization, user rights
- [ ] **OWASP Top 10**: All 10 risks addressed
- [ ] **SOC 2**: If required, audit trails complete

**Reference:** security-essentials.md, security-research.md

---

### Cost Estimation Guide

**Cost Formula:**
```
Monthly Cost = (Requests × Tokens/Request × $/Token) + Tools + Infrastructure
```

**Token Pricing (December 2025):**

| Model | Input $/1M | Output $/1M | Best For |
|-------|-----------|-------------|----------|
| GPT-4o | $2.50 | $10.00 | Complex reasoning |
| GPT-4o-mini | $0.15 | $0.60 | Routing, simple tasks |
| Claude Opus 4.5 | $15.00 | $75.00 | Hardest problems |
| Claude Sonnet 4.5 | $3.00 | $15.00 | Balanced performance |
| Claude Haiku | $0.25 | $1.25 | Speed, simple tasks |

**Cost Estimation by Use Case:**

| Use Case | Tokens/Request | Requests/Day | Monthly Cost |
|----------|---------------|--------------|--------------|
| Simple chatbot | 500 | 1,000 | $50-100 |
| Code assistant | 2,000 | 500 | $200-400 |
| Research agent | 10,000 | 100 | $500-1,000 |
| Multi-agent system | 20,000 | 200 | $2,000-5,000 |

**Cost Optimization Strategies:**

| Strategy | Savings | Implementation |
|----------|---------|----------------|
| Model cascading | 40-60% | Use mini for simple, full for complex |
| Prompt caching | 90% | Cache >1024 tokens, Anthropic only |
| Response caching | 100% | Cache deterministic queries |
| Tool RAG | 20-30% | Select 5-10 tools vs loading 50 |
| Batching | 15-25% | Group similar requests |

**Budget Calculator (Pseudocode):**
```
def estimate_monthly_cost(
    requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    model: str = "gpt-4o-mini"
) -> float:
    prices = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "claude-sonnet": (3.00, 15.00),
        "claude-haiku": (0.25, 1.25),
    }
    input_price, output_price = prices[model]

    monthly_requests = requests_per_day * 30
    input_cost = (monthly_requests * avg_input_tokens / 1_000_000) * input_price
    output_cost = (monthly_requests * avg_output_tokens / 1_000_000) * output_price

    return input_cost + output_cost
```

**Break-Even Analysis:**
- **Small scale** (1K requests/day): Break-even 6-9 months
- **Medium scale** (10K requests/day): Break-even 3-6 months
- **Large scale** (100K+ requests/day): Break-even 1-3 months

**Reference:** api-optimization-guide.md

---

## Emerging Agent Technologies (2025)

### Voice/Audio Agents

**Platform Comparison:**

| Platform | Latency | Cost | Best For |
|----------|---------|------|----------|
| OpenAI Realtime API | ~200ms TTFA | $0.06/min input, $0.24/min output | Tool-using voice agents |
| ElevenLabs | ~150ms TTFA | $0.08/min (Business) | Expressive, emotional voices |
| Google Gemini Live | ~200ms | Varies | Multi-modal (voice + video) |
| Amazon Nova Sonic | Low | Varies | Enterprise, Bedrock integration |

**Architecture Options:**
1. **Speech-to-Speech (OpenAI Realtime)**: Single model, lowest latency, preserves nuance
2. **Modular Pipeline (ElevenLabs)**: STT → LLM → TTS, flexible model choice, higher latency
3. **Hybrid**: Use Realtime for response, ElevenLabs for custom voice

**Key Capabilities (2025):**
- MCP server support in voice agents
- Image inputs during voice conversations
- SIP phone calling integration
- Tool calling with voice confirmation
- WebRTC for browser, WebSocket for server

**Use Cases:**
- Customer support (24/7 voice agents)
- Virtual assistants (hands-free interaction)
- Accessibility (voice-first interfaces)
- Phone systems (IVR replacement)

**Sources:** [OpenAI Realtime API](https://openai.com/index/introducing-gpt-realtime/), [ElevenLabs Comparison](https://elevenlabs.io/blog/elevenlabs-agents-vs-openai-realtime-api-conversational-agents-showdown)

---

### Multi-Modal Agents (Vision + Text + Tools)

**Current Capabilities (December 2025):**

| Model | Vision | Tools | Audio | Video | Latency |
|-------|--------|-------|-------|-------|---------|
| GPT-4o | ✅ | ✅ | ✅ | ⚠️ | 320ms |
| Claude Opus 4.5 | ✅ | ✅ | ❌ | ❌ | ~500ms |
| Gemini 2.0 | ✅ | ✅ | ✅ | ✅ | ~400ms |
| Qwen3-VL | ✅ | ✅ | ❌ | ❌ | Varies |

**Vision Agent Use Cases:**
1. **Document Processing**: Forms, invoices, contracts (95%+ accuracy)
2. **UI Automation**: Screen reading, button clicking, form filling
3. **Visual QA**: Analyze charts, diagrams, screenshots
4. **Code Review**: Visual diff analysis, architecture diagrams

**Multi-Modal Agent Architecture:**
```
Input (image/text/audio)
    ↓
┌─────────────────────────────────┐
│   Multi-Modal Foundation Model   │
│   (GPT-4o, Gemini 2.0, etc.)    │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│         Tool Router              │
│   - Vision tools (OCR, detect)   │
│   - Action tools (click, type)   │
│   - API tools (search, compute)  │
└─────────────────────────────────┘
    ↓
Output (text/action/image)
```

**Security Considerations:**
- Vision-enabled agents have new attack surfaces
- Adversarial images can manipulate agent behavior
- Screen content may contain sensitive data
- Validate visual inputs like text inputs

**Open-Source Alternatives:**
- **Molmo** (1B-72B): Matches GPT-4V performance
- **Qwen3-VL**: GUI operation, 30+ languages
- **MiniCPM-V 8B**: Runs on mobile devices

**Reference:** Multi-modal agents research, arXiv:2406.12814

---

### Agent-to-Agent Protocol (A2A)

**Protocol Comparison:**

| Feature | MCP (Anthropic) | A2A (Google) |
|---------|-----------------|--------------|
| Focus | Agent ↔ Tools (vertical) | Agent ↔ Agent (horizontal) |
| Use case | Tool integration | Multi-agent orchestration |
| Transport | JSON-RPC over stdio/HTTP | JSON-RPC over HTTP/SSE |
| Modalities | Tools, resources | Audio, video, text streaming |
| Long-running | Limited | ✅ Built-in (hours/days) |
| Partners | 2,000+ servers | 50+ launch partners |

**A2A Core Concepts:**
- **Client Agent**: Formulates and communicates tasks
- **Remote Agent**: Executes tasks and returns results
- **Task Lifecycle**: Create → Execute → Stream updates → Complete
- **Agent Cards**: Discovery mechanism (like MCP server manifests)

**When to Use Each:**
```
MCP: Connect agent to external tools/data
     ├─ Database queries
     ├─ File operations
     ├─ API integrations
     └─ Memory systems

A2A: Connect agent to other agents
     ├─ Cross-vendor orchestration
     ├─ Enterprise workflows
     ├─ Long-running collaborations
     └─ Multi-modal streaming
```

**A2A Design Principles:**
1. **Agentic-first**: Unstructured, natural agent communication
2. **Standards-based**: HTTP, SSE, JSON-RPC (easy integration)
3. **Enterprise-secure**: OpenAPI-level auth at launch
4. **Long-running**: Hours/days with real-time updates
5. **Modality-agnostic**: Text, audio, video streaming

**Industry Adoption:**
- **Tech**: Atlassian, Salesforce, SAP, MongoDB, PayPal
- **Consulting**: Deloitte, McKinsey, Accenture, PwC
- **Donated to**: Linux Foundation (alongside MCP)

**Sources:** [Google A2A Announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/), [MCP vs A2A Guide](https://auth0.com/blog/mcp-vs-a2a/)

---

### Real-Time/Streaming Agents

**Why WebSockets for Agents:**
- **Low latency**: No repeated HTTP handshakes
- **Bi-directional**: Agent and user can send simultaneously
- **Persistent**: Maintain state across interactions
- **Efficient**: Continuous streams vs request/response

**Platform Support:**

| Platform | Protocol | Use Case |
|----------|----------|----------|
| OpenAI Realtime | WebRTC (browser), WebSocket (server) | Voice agents |
| Google Gemini Live | WebSocket | Voice + video agents |
| Amazon Bedrock AgentCore | WebSocket bi-directional | Enterprise agents |
| LiveKit | WebRTC + TURN | Low-latency global voice |

**Architecture Pattern:**
```
Client (Browser/App)
    ↓ WebSocket/WebRTC
Load Balancer
    ↓
Agent Server (maintains connection)
    ↓ Internal API
┌─────────────────────────────────┐
│  LLM API (streaming enabled)    │
│  - stream=True                  │
│  - SSE for token streaming      │
└─────────────────────────────────┘
```

**Key Considerations:**
1. **Connection management**: Handle reconnects gracefully
2. **State synchronization**: Keep client/server state in sync
3. **Backpressure**: Handle slow clients
4. **Scaling**: Sticky sessions or distributed state

**Emerging: QUIC & Media over QUIC (MoQ)**
- QUIC: Standard by mid-2025 (Google, Meta, Cloudflare)
- MoQ: Cloudflare global relay network (330+ cities)
- Benefits: Lower latency than TCP, better mobile performance

**Latency Targets:**
- **Voice response**: <500ms (feels real-time)
- **Text streaming**: First token <200ms
- **Action confirmation**: <1s
- **Tool execution feedback**: Streaming updates

**Sources:** [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime), [AWS Bedrock Streaming](https://aws.amazon.com/blogs/machine-learning/bi-directional-streaming-for-real-time-agent-interactions-now-available-in-amazon-bedrock-agentcore-runtime/)

---

### Browser Automation Agents (Operator, Computer Use)

**Platform Comparison (December 2025):**

| Platform | Model | OSWorld | WebArena | Availability |
|----------|-------|---------|----------|--------------|
| **OpenAI Operator** | CUA (Computer-Using Agent) | 38.1% | 58.1% | ChatGPT Pro ($200/mo) |
| **Claude Computer Use** | Claude Sonnet 4.5 | 61.4% | N/A | API (beta) |
| **ChatGPT Agent** | o3 + web browsing | N/A | Better than CUA | ChatGPT Plus+ |

**OpenAI Operator:**
- Powered by CUA model (GPT-4o vision + RL for GUIs)
- Iterative loop: Screenshot → Reason → Act (click/scroll/type)
- Asks for confirmation before external side effects
- Available at operator.chatgpt.com
- Expanding to Plus, Team, Enterprise users

**Claude Computer Use:**
- **OSWorld leader**: 61.4% (vs Operator 38.1%)
- Tool versions: computer_20251124 (Opus 4.5), computer_20250124 (others)
- Actions: screenshot, click, type, scroll, hold_key, zoom
- Requires anthropic-beta header
- Independent bash and text_editor tools

**Architecture Pattern:**
```
User Request
    ↓
┌─────────────────────────────────┐
│     Vision Model (CUA/Claude)   │
│   Screenshot → Reasoning → Plan │
└──────────────┬──────────────────┘
    ↓
┌─────────────────────────────────┐
│         Action Executor          │
│   Click, Type, Scroll, Wait      │
└──────────────┬──────────────────┘
    ↓
New Screenshot → Loop until complete
```

**Use Cases:**
- Web research and data extraction
- Form filling and submissions
- E-commerce automation
- Legacy system integration (no API)
- Testing and QA automation

**Safety Considerations:**
- Always confirm before irreversible actions
- Sandbox in VM/container for untrusted tasks
- Monitor for credential exposure in screenshots
- Rate limit to prevent runaway automation

**Sources:** [OpenAI Operator](https://openai.com/index/introducing-operator/), [Claude Computer Use](https://docs.claude.com/en/docs/agents-and-tools/tool-use/computer-use-tool)

---

### Memory Systems for Agents (Mem0, GraphRAG)

**Memory Types:**

| Type | Purpose | Retention | Example |
|------|---------|-----------|---------|
| Short-term | Current task context | Session | Conversation history |
| Episodic | Past experiences | Long-term | Previous task executions |
| Semantic | Facts and knowledge | Permanent | Domain knowledge |
| Procedural | How to do things | Long-term | Successful patterns |

**Mem0 - Universal Memory Layer:**
- **$24M Series A** (Oct 2025): YC, Basis Set, Peak XV
- **41K+ GitHub stars**, 13M+ PyPI downloads
- **Performance**: 26% accuracy boost, 91% lower p95 latency
- **Architecture**: Hybrid graph + vector + key-value store
- **Integration**: CrewAI, Flowise, Langflow, AWS Agent SDK (exclusive)
- **API Growth**: 35M → 186M calls (Q1 to Q3 2025)

**Mem0 Architecture:**
```
Incoming Message
    ↓
┌─────────────────────────────────┐
│      Entity Extractor           │
│   Identify nodes (people, etc)  │
└──────────────┬──────────────────┘
    ↓
┌─────────────────────────────────┐
│      Relations Generator         │
│   Infer labeled edges            │
└──────────────┬──────────────────┘
    ↓
┌─────────────────────────────────┐
│      Hybrid Datastore            │
│   Graph + Vector + Key-Value     │
└─────────────────────────────────┘
```

**GraphRAG (Microsoft):**
- Combines knowledge graphs with RAG
- Extracts entities/relationships from documents
- Hierarchical community detection for global queries
- **Available**: Via Microsoft Discovery, GitHub open source
- **Limitation**: Slow updates (recompute on change)

**When to Use Each:**
```
Mem0: Real-time personalization, multi-session agents
      ├─ User preferences over time
      ├─ Conversation history
      └─ Low-latency retrieval needs

GraphRAG: Document understanding, research
          ├─ Complex relationship queries
          ├─ "Synthesize across documents"
          └─ Batch processing acceptable
```

**Experience-Following Behavior (Research Finding):**
- High similarity between input and memory → similar output
- **Risk**: Error propagation, misaligned replay
- **Mitigation**: Curate memory, include context, allow overrides

**Sources:** [Mem0](https://mem0.ai/), [GraphRAG](https://github.com/microsoft/graphrag), [Mem0 Research Paper](https://arxiv.org/abs/2504.19413)

---

## Domain-Specific Agent Architectures

### Code Generation Agents (SWE-bench Deep Dive)

**Current Leaders (December 2025):**

| Agent | SWE-bench Verified | SWE-bench Live | Architecture |
|-------|-------------------|----------------|--------------|
| Claude Opus 4.5 | 80.9% | ~20% | Minimal scaffold |
| OpenHands + Claude 3.7 | 66.4% | ~19% | Critic model + scaling |
| Devstral (Open Source) | 46.8% | ~15% | Mistral + OpenHands |
| Devin | ~45% | ~15% | Autonomous agent |

**Anthropic's Minimal Architecture:**
Anthropic's approach: "Give as much control as possible to the language model itself, keep scaffolding minimal."
- Only 2 tools: bash + file editing (string replacements)
- No complex planning frameworks
- Let the model drive the entire workflow

**OpenHands Inference-Time Scaling:**
```
For each problem:
1. Run agent multiple times (temperature=1.0)
2. Generate multiple code patches
3. Use critic model (Qwen 2.5 Coder 32B) to evaluate
4. Select best solution

Result: 60.6% → 66.4% with 5 attempts
```

**Key Insights:**
- SWE-bench Live (real-world) scores are ~3x lower than Verified
- Minimal scaffolding often outperforms complex frameworks
- Inference-time scaling (multiple attempts + critic) improves results
- Open-source (Devstral) approaching commercial performance

**Sources:** [Anthropic SWE-bench](https://www.anthropic.com/engineering/swe-bench-sonnet), [OpenHands SOTA](https://openhands.dev/blog/sota-on-swe-bench-verified-with-inference-time-scaling-and-critic-model)

---

### Customer Support Agents

**Klarna Case Study (Updated 2025):**
- 2.3 million conversations, ~66% of all chats
- Equivalent to 700 full-time agents
- Built on LangGraph + LangSmith (OpenAI GPT-4)
- Resolution time: 2 minutes (vs 11 minutes human)
- **2025 Pivot**: Now hybrid AI+human model
  - AI still handles 66% of inquiries
  - Humans always available as option
  - Quality issues drove rebalancing
  - 25% reduction in repeat inquiries

**Three-Layer Architecture:**
```
┌─────────────────────────────────┐
│     Orchestration Layer          │
│   (Router: AI vs Human)          │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│       Action Layer               │
│   (API calls, transactions)      │
│   - Middleware (n8n, Make)       │
│   - Safe tool execution          │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│      Knowledge Layer             │
│   (RAG over documentation)       │
│   - Policy references            │
│   - Product information          │
└─────────────────────────────────┘
```

**Intercom Fin Pattern:**
- GPT-4/Claude powered
- Semantic search over knowledge base
- RAG for technical questions
- 86% resolution rate
- Human handoff for complex issues

**Hybrid Human-AI Model:**
- AI handles: FAQs, status checks, simple transactions
- Human handles: Complaints, edge cases, empathy-required
- Escalation triggers: Sentiment, complexity, customer request

**Key Metrics:**
| Metric | AI Agent | Human | Target |
|--------|----------|-------|--------|
| Resolution time | 2 min | 11 min | <5 min |
| Resolution rate | 66-86% | 95% | >70% |
| CSAT | ~4.2/5 | 4.5/5 | >4.0 |
| Cost per interaction | $0.10-0.50 | $5-15 | <$1 |

**Sources:** [Klarna LangChain](https://blog.langchain.com/customers-klarna/), [OpenAI Klarna](https://openai.com/index/klarna/)

---

### Data Analysis Agents

**Key Frameworks:**

| Framework | Approach | Best For |
|-----------|----------|----------|
| PandasAI | Natural language → Python/SQL | Non-technical users |
| LangChain DataFrame Agent | Agent + Pandas tools | Developers |
| LlamaIndex | RAG + structured data | Complex queries |

**PandasAI Architecture:**
```
User Query (natural language)
    ↓
┌─────────────────────────────────┐
│     LLM (GPT-4/Claude)          │
│   Translate to Python/SQL       │
└──────────────┬──────────────────┘
    ↓
┌─────────────────────────────────┐
│     Code Execution Engine        │
│   Execute against DataFrame      │
└──────────────┬──────────────────┘
    ↓
Results (table/chart/answer)
```

**Supported Data Sources:**
- SQL databases (PostgreSQL, MySQL, SQLite)
- DataFrames (pandas, polars)
- Files (CSV, Excel, Parquet)
- NoSQL (MongoDB)

**Example Queries:**
```python
# PandasAI usage
import pandas as pd
from pandasai import SmartDataframe

df = SmartDataframe(pd.read_csv("sales.csv"))
df.chat("What were the top 5 products by revenue last quarter?")
df.chat("Show me a trend line of monthly sales")
df.chat("Which region has the highest growth rate?")
```

**Benefits:**
- 50% reduction in analysis time
- Democratized data access (no SQL required)
- Fewer human errors in queries
- Real-time insights

**Security Considerations:**
- Validate generated SQL before execution
- Limit data access by user role
- Audit all queries
- Prevent SQL injection in generated code

**Sources:** [PandasAI GitHub](https://github.com/sinaptik-ai/pandas-ai), [LangChain DataFrame Agent](https://python.langchain.com/docs/tutorials/agents)

---

### Research/Writing Agents

**Multi-Agent Research Architecture:**

```
┌─────────────────────────────────┐
│        Coordinator Agent         │
│   (Breaks task into subtasks)    │
└──────────────┬──────────────────┘
               ↓
┌──────────────────────────────────────────────┐
│                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │Researcher│  │ Analyst │  │ Writer  │      │
│  │ (Search) │  │(Synth.) │  │(Draft)  │      │
│  └─────────┘  └─────────┘  └─────────┘      │
│                                              │
└──────────────────────────────────────────────┘
               ↓
┌─────────────────────────────────┐
│         Critic Agent             │
│   (Review & feedback)            │
└─────────────────────────────────┘
```

**Paper2Agent Framework:**
Transforms research papers into interactive AI agents:
1. Extract codebase and workflows from paper
2. Package as reproducible agent
3. Interface for downstream use
4. Enable discovery and adoption

**Commercial Research Agents:**
- **Resea AI**: Academic papers, literature reviews
- **Elicit**: Research question → paper summaries
- **Consensus**: Scientific consensus from papers

**Use Cases:**
1. **Literature Reviews**: Search, summarize, synthesize papers
2. **Content Generation**: Blog posts, reports, documentation
3. **Academic Writing**: Draft papers, format citations
4. **Market Research**: Competitor analysis, trend reports

**Multi-Agent Patterns for Writing:**
```
Sequential: Research → Outline → Draft → Critique → Revise
Parallel:   Multiple researchers → Merge findings
Iterative:  Draft → Critique → Improve (loop until quality threshold)
```

**Sources:** [Paper2Agent](https://arxiv.org/abs/2509.06917), [LLM Agents Papers](https://github.com/AGI-Edgerunners/LLM-Agents-Papers)

---

### Coding/Developer Agents (Cursor, Windsurf, Devin, Claude Code)

**Agent Levels:**

| Level | Type | Examples | Human Role |
|-------|------|----------|------------|
| L1 | Autocomplete | Copilot (basic) | Every keystroke |
| L2 | Chat Assistant | ChatGPT, Claude.ai | Every question |
| L3 | Supervised Agent | Cursor, Windsurf | Approve each step |
| L4 | Autonomous Agent | Claude Code, Devin | Review outcomes |

**Architecture Comparison (December 2025):**

| Tool | Engine | Key Features | Context | Price |
|------|--------|--------------|---------|-------|
| **Cursor 2.0** | Composer model | 4x faster MoE model, 8 parallel agents, git worktrees | Project index | $20/mo |
| **Windsurf** | Cascade | Memory system, parallel agents, auto-linting | Dependency graph | $15/mo |
| **Claude Code** | Native | Skills, hooks, plugins, MCP | 200K tokens | API usage |
| **Devin 2.0** | Cognition | Cloud IDE, parallel Devins, Wiki | Full codebase | $20/mo+ |

**Cursor 2.0 (December 2025):**
- **Composer model**: Custom MoE, 4x faster than similar models, RL-trained for coding
- **Multi-agent**: Run up to 8 agents in parallel on single prompt
- **Git worktrees**: Each agent in isolated codebase copy
- **AI Code Reviews**: Find and fix bugs with AI before commit
- **Voice control**: Built-in speech-to-text for agent commands
- **Sandboxing**: Linux + macOS secure shell execution

**Windsurf Cascade (December 2025):**
- **Memory system**: Auto-saves context between conversations
- **Parallel agents**: Multi-pane Cascade, never wait
- **Built-in planning**: Specialized planning agent + Todo tracking
- **Turbo mode**: Auto-execute terminal commands
- **Image-to-code**: Drop image, generate HTML/CSS/JS
- **MCP store**: One-click server installation
- Recognition: Gartner Magic Quadrant Leader 2025

**Claude Code Advantages:**
- **200K token context** (largest production context)
- **Skills system**: Markdown files teaching domain expertise
- **Hooks**: Trigger commands on lifecycle events
- **Plugins**: Share and install packaged workflows
- **MCP integration**: 2000+ tool servers
- Autonomous multi-file edits with session persistence

**Devin 2.0 (2025):**
- **Price drop**: $500/mo → $20/mo (Core plan)
- **Parallel Devins**: Multiple agents on different tasks
- **DeepWiki**: Auto-generated codebase documentation
- **Devin Search**: Natural language code queries
- Available at app.devin.ai
- Competes with GitHub Copilot, Windsurf, Amazon Q

**IDE vs CLI Trade-offs:**
| Factor | IDE (Cursor/Windsurf) | CLI (Claude Code) | Cloud (Devin) |
|--------|----------------------|-------------------|---------------|
| Interactive sessions | ✅ Excellent | ⚠️ Good | ✅ Excellent |
| Long-running tasks | ⚠️ Context loss | ✅ State persisted | ✅ Cloud state |
| Autonomous work | ⚠️ Limited | ✅ Full support | ✅ Full support |
| Team collaboration | ✅ Extensions | ✅ Git native | ✅ Built-in |
| Learning curve | Low | Medium | Low |

**Sources:** [Cursor 2.0](https://cursor.com/blog/2-0), [Windsurf Cascade](https://windsurf.com/cascade), [Devin 2.0](https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-software-engineer-to-20-per-month-from-500)

---

## Advanced Production Topics

### Agent Observability & Monitoring Deep Dive

**Platform Comparison (December 2025):**

| Platform | Type | Best For | GitHub Stars | Overhead |
|----------|------|----------|--------------|----------|
| LangSmith | Commercial | LangChain/LangGraph users | N/A | Near-zero |
| Langfuse | Open Source | Self-hosted, MIT license | 19K+ | Low |
| Braintrust | Commercial | Agent evaluation, CI/CD | N/A | Low |
| Arize | Commercial | ML teams, drift detection | N/A | Low |
| AgentOps | Commercial | Multi-agent systems | N/A | Low |

**Key Observability Features:**
1. **Tracing**: End-to-end visibility of agent decisions
2. **Span Analysis**: Nested views of multi-step workflows
3. **Latency Tracking**: Per-component timing breakdown
4. **Cost Attribution**: Token usage by step/agent
5. **Error Correlation**: Link failures to root causes

**Tracing Architecture:**
```
User Request
    ↓
┌─────────────────────────────────┐
│          Trace (root span)       │
│  ├─ Agent A (span)               │
│  │   ├─ LLM Call (span)          │
│  │   ├─ Tool: search (span)      │
│  │   └─ LLM Call (span)          │
│  ├─ Agent B (span)               │
│  │   └─ LLM Call (span)          │
│  └─ Final Output (span)          │
└─────────────────────────────────┘
```

**2025 Trends:**
- Deeper agent tracing (LangGraph, AutoGen support)
- Structured output monitoring (tools, multi-modal)
- LLM-as-judge integration with observability
- OpenTelemetry standardization for LLMs

**Choosing a Platform:**
- **LangChain ecosystem**: LangSmith (seamless integration)
- **Self-hosted/open source**: Langfuse (MIT, 19K+ GitHub stars)
- **Existing enterprise**: Datadog/Prometheus integration

**Sources:** [LangSmith](https://www.langchain.com/langsmith/observability), [Langfuse](https://langfuse.com/blog/2024-07-ai-agent-observability-with-langfuse)

---

### Production Deployment Patterns

**Kubernetes for AI Agents (2025):**

| Component | K8s Resource | Purpose |
|-----------|-------------|---------|
| Agent runtime | Deployment/StatefulSet | Agent execution |
| Tool servers | Service/Pod | Tool endpoints |
| State storage | PersistentVolume | Memory/checkpoints |
| Message bus | StatefulSet (Kafka/NATS) | Agent communication |
| Load balancer | Ingress/Service | Traffic distribution |

**Google Agent Sandbox (KubeCon NA 2025):**
- New K8s primitive for agent code execution
- Built on gVisor + Kata Containers for kernel-level isolation
- Secure boundary for tool execution and computer use
- **Pre-warmed pools**: 90% faster than cold starts (<1s latency)
- **Pod Snapshots**: Save/restore sandbox state in seconds
- **Python SDK**: Manage sandbox lifecycle without infra knowledge
- **Open Source**: kubernetes-sigs/agent-sandbox (K8s SIG Apps)
- Core APIs: Sandbox, SandboxTemplate, SandboxClaim

**Kagent Framework (CNCF Sandbox):**
- Deploy MCP servers on Kubernetes
- A2A protocol support
- Production-ready scaling
- GitOps integration

**Deployment Patterns:**

```
1. STATELESS AGENTS (Simple)
   ┌─────────────────────────────────┐
   │    Kubernetes Deployment         │
   │    - HPA for auto-scaling        │
   │    - No persistent storage       │
   │    - Fast startup                │
   └─────────────────────────────────┘

2. STATEFUL AGENTS (Memory)
   ┌─────────────────────────────────┐
   │    Kubernetes StatefulSet        │
   │    - PersistentVolumeClaim       │
   │    - Sticky sessions             │
   │    - Checkpoint persistence      │
   └─────────────────────────────────┘

3. MULTI-AGENT SYSTEM
   ┌─────────────────────────────────┐
   │    Supervisor (Deployment)       │
   │         ↓                        │
   │    Worker Pool (StatefulSet)     │
   │         ↓                        │
   │    Message Bus (Kafka)           │
   │         ↓                        │
   │    State Store (Redis/Postgres)  │
   └─────────────────────────────────┘
```

**Infrastructure Choices:**
- **Container runtime**: gVisor (security), containerd (performance)
- **State**: Redis (fast), PostgreSQL (durable), S3 (large artifacts)
- **Messaging**: NATS (simple), Kafka (durable), Redis Streams (hybrid)

**2025 Stats:**
- 90%+ enterprises run K8s in production
- 76% DevOps teams integrate AI (often on K8s)
- 5.6M developers use Kubernetes

**Sources:** [Google Agent Sandbox](https://cloud.google.com/blog/products/containers-kubernetes/agentic-ai-on-kubernetes-and-gke), [Kagent](https://kagent.dev/)

---

### Testing Strategies for Agents

**Testing Pyramid for Agents:**

```
        ╱╲
       ╱  ╲     End-to-End Tests
      ╱────╲    (Full scenarios, expensive)
     ╱      ╲
    ╱────────╲   Integration Tests
   ╱          ╲  (Workflow validation)
  ╱────────────╲
 ╱              ╲ Unit Tests
╱────────────────╲ (Prompt, tools, components)
```

**Unit Testing:**
- Test individual prompts with fixed inputs
- Validate tool parameter extraction
- Check output format compliance
- Fast, cheap, specific

**Integration Testing:**
- Test complete agent workflows
- Include tool execution
- Run in CI/CD on each commit
- Nightly smoke tests

**End-to-End Testing:**
- Full scenarios with real APIs
- Human evaluation samples
- Expensive but comprehensive

**Evaluation Frameworks (December 2025):**

| Framework | Focus | Automation | Key Feature |
|-----------|-------|------------|-------------|
| Promptfoo | Prompt testing | ✅ CI/CD | Local, fast iteration |
| DeepEval | Agent evaluation | ✅ CI/CD | Built-in metrics |
| OpenAI Evals | Model benchmarking | ✅ CLI | Standard benchmarks |
| LangSmith | Tracing + eval | ✅ Platform | LangChain integration |
| **Braintrust** | Agent trajectories | ✅ Platform | Loop scorer, remote evals |

**Braintrust for Agents (2025):**
- **Loop**: Generate custom scorers in natural language
- **Trace-driven testing**: No custom instrumentation needed
- **Trajectory evaluation**: Assess entire agent paths, not just outputs
- **Offline + Online**: Unit tests → production monitoring
- Framework-agnostic SDKs

**Hybrid Evaluation Strategy:**
```
1. AUTOMATED (80%)
   - LLM-as-judge scoring
   - Regression detection
   - Format validation
   - Tool call correctness

2. HUMAN REVIEW (20%)
   - Edge cases
   - Quality sampling
   - Expert validation
   - User feedback
```

**Key Metrics to Test:**
- Task completion rate
- Reasoning correctness
- Tool usage accuracy
- Latency percentiles
- Cost per task
- Hallucination rate

**Best Practices:**
1. Start with unit tests for prompts
2. Add integration tests for workflows
3. Include regression monitoring
4. Sample human evaluation
5. Run continuously in CI/CD

**Sources:** [Testing AI Agents](https://galileo.ai/learn/test-ai-agents), [AI Agent Testing Trends 2025](https://qawerk.com/blog/ai-agent-testing-trends/)

---

### Agent Orchestration at Scale

**Enterprise Orchestration Patterns:**

| Pattern | Use Case | Governance | Complexity |
|---------|----------|------------|------------|
| Centralized | Strict compliance | High | Medium |
| Decentralized | Autonomous teams | Low | High |
| Hierarchical | Complex workflows | Medium | High |
| Event-Driven | Real-time responses | Medium | Medium |
| Hybrid Human-AI | Regulated industries | Highest | Medium |

**Supervisor Pattern (Most Common):**
```
┌─────────────────────────────────┐
│        Supervisor Agent          │
│   - Task decomposition           │
│   - Agent assignment             │
│   - Result aggregation           │
│   - Conflict resolution          │
└──────────────┬──────────────────┘
               ↓
┌──────────────────────────────────────────────┐
│                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │Agent A  │  │Agent B  │  │Agent C  │      │
│  │(Domain1)│  │(Domain2)│  │(Domain3)│      │
│  └─────────┘  └─────────┘  └─────────┘      │
│                                              │
└──────────────────────────────────────────────┘
               ↓
         Shared State / Message Bus
```

**Scale Benchmarks (2025):**
- 8-10x memory reduction with optimization
- 80%+ coordination efficiency at 10,000+ agents
- 3x decision speed improvement
- 45% reduction in hand-offs

**Enterprise ROI:**
- Average: 171% ROI in 12-18 months
- Investment: $500K-$2M typical
- Key savings: Labor, speed, accuracy

**Scaling Challenges:**
- 60% of multi-agent systems fail beyond pilot
- Tool integration failures (primary barrier)
- Governance complexity
- Context synchronization

**Platform Comparison:**

| Platform | Orchestration | Scale | Enterprise |
|----------|--------------|-------|------------|
| Microsoft Copilot Studio | Multi-agent | ✅ | ✅ |
| Amazon Bedrock Agents | Managed | ✅ | ✅ |
| LangGraph | Flexible | ✅ | ⚠️ |
| Kore.ai | Supervisor pattern | ✅ | ✅ |
| CrewAI | Role-based | Medium | ⚠️ |

**Keys to Success:**
1. Start with single agent, plan for multi
2. Define clear agent boundaries
3. Implement shared context protocol
4. Build governance from start
5. Monitor and optimize continuously

**Sources:** [Azure AI Agent Patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns), [Multi-Agent Orchestration 2025](https://www.onabout.ai/p/mastering-multi-agent-orchestration-architectures-patterns-roi-benchmarks-for-2025-2026)

---

## Quick Navigation: 41 Critical Questions

**Business (1-6):**
1. Build vs buy? → Use framework for standard workflows, build custom for unique needs
2. Single vs multi-agent? → Multi-agent justified when 3+ distinct domains required
3. ROI calculation? → Break-even: 1-9 months depending on scale
4. Proven use cases? → Customer support (75% automation), code migration (90% accuracy), content generation (10x speed)
5. Main risks? → Hallucinations, cost runaway, security vulnerabilities
6. Build stakeholder trust? → Transparency + validation framework + staged rollout

**Technical (7-12):**
7. Architecture choice? → Collaboration (2-3 domains), Supervisor (3-5), Swarm (5+)
8. 2025 LangGraph features? → Command Tool, Handoffs, Supervisor/Swarm libraries
9. State schema design? → Keep flat, use TypedDict, include cost tracking from start
10. Agent communication? → Message bus (complex), shared state (simple), or MCP (external tools)
11. Essential components? → Input processing, task execution, validation, error handling, monitoring
12. Tool calling? → Use RAG to select 5-10 relevant tools, validate rigorously, handle errors

**Implementation (13-17):**
13. GPT-4o vs Claude? → GPT-4o for structured output/JSON, Claude for writing/long-form
14. Extended thinking? → Use for complex reasoning (2-5x cost), skip for simple queries
15. Prompt caching? → Cache >1024 tokens, 90% savings on reads, 5-min lifetime
16. Best prompting patterns? → Role+Task+Constraints+Format, CoT with verification, ReAct, few-shot
17. Prompting antipatterns? → Vague instructions, implicit context, negative instructions, underspecified formats

**Production (18-20):**
18. Production metrics? → Latency (p95), success rate, cost per request, validation pass rate
19. Error handling? → Exponential backoff, fallback models, circuit breaker pattern
20. Debug failures? → Trace collection, common issue checklist, failure mode matching

**Cost (21-23):**
21. Reduce costs 50-80%? → Model cascading (40-60%), prompt caching (90%), progress tracking, tool RAG
22. Budget management? → Daily limits, alerts at 70%/90%, cost attribution by feature
23. Reduce latency? → Streaming, parallel calls, caching, reduce max_tokens, faster models

**Troubleshooting (24-26):**
24. High failure rate? → Fix specifications (35%→8%), role enforcement (28%→5%), context management (22%→3%)
25. Infinite loops? → Iteration limits, progress tracking, explicit success criteria, cycle detection
26. Agents ignoring outputs? → Add acknowledgment mechanism, information broadcasting, improve prompts

**Domain Applications (27-30):**
27. Code review agent? → Review-critique pattern with multi-layer validation
28. Research agent? → Supervisor pattern with specialized gatherer/analyst/writer
29. Support agent? → Router + handlers + HITL escalation (70-80% automation)
30. Content pipeline? → Sequential with reflection (research→outline→write→critique→improve)

**Advanced (31-33):**
31. 2025 developments? → o3 (88% ARC-AGI), Claude Opus 4.5 (80.9% SWE-bench), GPT-5.2 (80.0%), Gemini 3 (45.1% ARC-AGI-2), DeepSeek-V3.1
32. Academic failures? → MAST: 44.2% system design, 32.3% inter-agent, 23.5% verification failures
33. Reasoning patterns? → CoT (1 call), ReAct (2-10 calls), ToT (10-100+ calls), Extended Thinking
33a. CoT limitations? → "Brittle mirage" outside training distribution, fails on novel patterns
33b. Reason from Future? → Bidirectional reasoning (top-down + bottom-up), better for planning
33c. LATS? → Tree search + ReAct + MCTS, outperforms ToT for complex tasks
33d. RAFA? → Reason for Future, Act for Now - √T regret bounds, principled planning
33e. Graph-CoT? → Tree reasoning structure, 95.7% token reduction, 38% accuracy improvement
33f. MAST Taxonomy? → 14 failure modes in 3 categories, most from system design not model
33g. RAFFLES? → Agentic debugging, Judge+Evaluator agents, 43% fault attribution accuracy

**Decision Frameworks (34-36):**
34. Go/No-Go decision? → See comprehensive checklist below
35. Architecture planning? → Use full decision template below
36. Production ready? → See production readiness checklist below

**Security & Safety (37-41):**
37. Prompt injection defense? → Multi-layer: input validation, semantic filtering, output filtering, monitoring
38. Tool sandboxing? → Process isolation + Docker containers + Firecracker VMs (layered)
39. Compliance requirements? → EU AI Act, GDPR, OWASP Top 10, NIST AI RMF
40. Human-in-the-loop? → Risk-based approval (LOW→auto, MEDIUM→approve, HIGH→approve+log, CRITICAL→block)
41. Alignment challenges? → Goal misspecification, instrumental convergence, deceptive alignment, reward hacking

---

## Business & Strategy Questions

### Q1: Build vs Buy?
**Answer:** Use existing frameworks (LangGraph, CrewAI) when time-to-market is critical and standard workflows fit. Build custom when you need unique workflows, fine-grained control, specific compliance, or operate at large scale (>1M requests/month).

**Reference:** framework-comparison.md

---

### Q2: Single vs Multi-Agent?
**Answer:** Use single agent for 1-2 domains where sequential processing is acceptable. Use multi-agent for 3+ distinct domains where parallelization provides value and quality justifies added complexity (~30% cost increase).

**Reference:** multi-agent-patterns.md, theoretical-foundations.md

---

### Q3: ROI Calculation?
**Answer:** Development costs $50K-$200K, API costs $500-$5K/month. Break-even: 6-9 months (small), 3-6 months (medium), 1-3 months (large). Benefits: 70-80% support automation, 10x content speed, 50% code review time savings.

**Reference:** api-optimization-guide.md

---

### Q4: Proven Use Cases?
**Answer:** High ROI (>300%): Customer support (75% automation, <2s response), code migration (90% accuracy, 10x speed), content generation ($0.50-$2/article). Medium ROI: Research, data processing, QA. Low ROI: Simple classification, real-time latency-critical, highly regulated domains.

**Reference:** multi-agent-patterns.md (case studies)

---

### Q5: Business Risks?
**Answer:** Critical risks: hallucinations (mitigate with multi-layer validation, +30% dev cost), cost runaway (budget limits, circuit breakers), security vulnerabilities (input validation, sandboxing, +15% dev time). Medium risks: performance issues, vendor lock-in, compliance violations.

**Reference:** patterns-and-antipatterns.md

---

### Q6: Build Stakeholder Confidence?
**Answer:** Use transparency mechanisms (reasoning traces, confidence scores, audit trails), validation framework (A/B testing, 5-10% human validation, metrics dashboard), and staged rollout (shadow→HITL→automated monitoring→full automation).

**Reference:** workflow-overview.md, agentic-systems-cookbook.md (Recipe 4)

---

## Technical Architecture Questions

### Q7: Architecture Selection?
**Answer:** Collaboration for 2-3 domains with shared context, Supervisor for 3-5 domains with sequential stages, Swarm for 5+ domains with dynamic/exploratory workflows. Consider workflow type (sequential vs parallel) and complexity tolerance.

**Reference:** multi-agent-patterns.md, theoretical-foundations.md

---

### Q8: 2025 LangGraph Features?
**Answer:** Use Command Tool for dynamic routing (type-safe), Handoffs for explicit agent transitions, Supervisor/Swarm libraries for pre-built patterns. These reduce boilerplate and improve reliability vs static graph edges.

**Reference:** multi-agent-patterns.md

---

### Q9: State Schema Design?
**Answer:** Keep state flat with TypedDict for type safety. Include: messages (with add_messages reducer), current execution state, task data, artifacts dict, metadata (cost, timing), and control flags. Add cost tracking from start, not later.

**Reference:** multi-agent-patterns.md, agentic-systems-cookbook.md (Recipe 5)

---

### Q10: Agent Communication?
**Answer:** Use message bus for complex systems (complete audit trail, type-safe, async support), shared state for simple systems (lower overhead, LangGraph built-in), or MCP for external tool integration (overkill for internal comms).

**Reference:** patterns-and-antipatterns.md (Pattern 3), theoretical-foundations.md

---

### Q11: Essential Components?
**Answer:** Core: input processing (validation, sanitization), task execution (ReAct loop, tool calling), validation (output verification, confidence scoring), error handling (retry, fallback, circuit breaker), monitoring (cost, latency, error rates). Optional: reflection, memory management, HITL.

**Reference:** workflow-overview.md

---

### Q12: Tool/Function Calling?
**Answer:** Use RAG to select 5-10 relevant tools from larger set (not all 50). Validate tool existence, parameters, types, and values. Provide high-quality descriptions with clear use cases. Always wrap in try-catch with meaningful error messages to LLM.

**Reference:** agentic-systems-cookbook.md (Recipes 7-8), api-optimization-guide.md

---

### Q12a: What is the Tool Learning "Three Ws" Framework?
**Answer:** From Springer Survey (Jun 2025), tool learning requires answering three questions:

1. **WHETHER**: Is a tool call necessary?
   - Not all queries need tools
   - Evaluate: Can the LLM answer directly?
   - Avoid unnecessary API costs

2. **WHICH**: Which tool to select?
   - Match task to tool capabilities
   - Use RAG/embedding for large tool sets
   - Consider tool reliability and cost

3. **HOW**: How to use the tool?
   - Extract correct parameters from context
   - Handle optional vs required params
   - Validate before execution

**Evaluation Metrics (BFCL)**: Intent accuracy, function selection accuracy, parameter extraction accuracy, end-to-end success rate.

**Reference:** Springer Data Science and Engineering (2025), BFCL

---

## Implementation & Development Questions

### Q13: GPT-4o vs Claude Sonnet 4.5?
**Answer:** Use GPT-4o for guaranteed JSON schema adherence (100%), parallel function calling, and vision+tools. Use Claude Sonnet 4.5 for long-form writing, extended thinking, proactive tool calling, and when prompt caching saves costs (90% discount). Hybrid strategy recommended: router (mini), simple tasks (mini), reasoning (Claude), structured (GPT-4o).

**Reference:** api-optimization-guide.md

---

### Q14: Extended Thinking?
**Answer:** Use for complex math/logic, code debugging, strategic planning (96.5% accuracy on physics vs 84.8% baseline). Skip for simple queries, classification, speed-critical (<1s), or cost-sensitive applications. Cost: 2-5x increase, Latency: 2-5x slower.

**Reference:** theoretical-foundations.md, api-optimization-guide.md

---

### Q15: Prompt Caching?
**Answer:** Cache content >1024 tokens (minimum) like system prompts, codebases, few-shot examples. Anthropic only: cached reads $0.30/1M (90% off), writes $3.75/1M, 5-min lifetime. Place cache_control on LAST eligible block. ROI: 89% savings with 100 queries/5min.

**Reference:** api-optimization-guide.md

---

### Q16: Best Prompting Patterns?
**Answer:** Top patterns: 1) Explicit role+task+constraints+format, 2) Chain-of-thought with verification, 3) Few-shot with explanations, 4) ReAct (thought→action→observation loop), 5) Self-consistency (majority vote from multiple solutions at temperature 0.7).

**Reference:** theoretical-foundations.md, patterns-and-antipatterns.md

---

### Q17: Prompting Antipatterns?
**Answer:** Avoid: vague instructions (specify word counts, sections), implicit context (externalize all assumptions), negative instructions (say what to do, not what to avoid), ambiguous examples (show exact input→output), underspecified formats (use Pydantic/JSON Schema).

**Reference:** patterns-and-antipatterns.md

---

## Production & Operations Questions

### Q18: Production Metrics?
**Answer:** Monitor: Performance (latency p50/p95/p99, success rate, throughput), Quality (validation pass rate, human override rate, confidence scores), Cost (per request, token usage, model mix, burn rate), Operations (circuit breaker status, retry rate, cache hit rate). Alert on: p95 >10s, errors >5%, cost >2x baseline, success <90%.

**Reference:** api-optimization-guide.md, agentic-systems-cookbook.md (Recipe 10)

---

### Q19: Error Handling?
**Answer:** Implement exponential backoff with jitter for rate limits, fallback to cheaper model after max retries, increase timeout on timeout errors, circuit breaker pattern (open after 5 failures, 60s timeout). Log all errors, retry only retryable ones.

**Reference:** agentic-systems-cookbook.md (Recipe 9), patterns-and-antipatterns.md

---

### Q20: Debug Failures?
**Answer:** Identify category (specification, inter-agent misalignment, verification), collect full traces (conversation_id, agents, messages, tools, errors, cost, latency), check common symptom→cause→fix table, match against 14 known failure modes from research.

**Reference:** patterns-and-antipatterns.md

---

## Cost & Resource Management Questions

### Q21: Reduce Costs 50-80%?
**Answer:** Combine strategies: model cascading (40-60% savings - use mini for simple tasks), prompt caching (90% on cached content), progress tracking (40% - avoid redundant work), tool selection with RAG (20-30%), response caching (100% on repeats), batch processing (15-25%), output length limits (10-20%). Combined: $5K→$1-2K/month.

**Reference:** api-optimization-guide.md, agentic-systems-cookbook.md (Recipe 11)

---

### Q22: Budget Management?
**Answer:** Set daily budgets with checks before requests. Alert at 70% (warning), 90% (critical), 100% (emergency). Options when exceeded: reject requests, use cheaper model, or queue. Track attribution by user/tenant, feature, model, and time.

**Reference:** agentic-systems-cookbook.md (Recipe 10)

---

### Q23: Reduce Latency?
**Answer:** Use streaming (perceived 0s latency), parallel API calls (3-5x faster), response caching (instant hits), reduce max_tokens (faster generation), faster models (mini 2x, haiku 3x), parallel tool calls (GPT-4o, Claude 4.5). Targets: interactive <1s, background <10s, batch <60s.

**Reference:** api-optimization-guide.md

---

## Troubleshooting & Debugging Questions

### Q24: High Failure Rate?
**Answer:** Root causes with fixes: vague specifications (35%→8%: add success criteria, validation, examples), role violations (28%→5%: programmatic enforcement, forbidden actions), context loss (22%→3%: intelligent management, summarization), incomplete verification (42%→12%: multi-layer validation, domain checks), prompting issues (use structured outputs, explicit formats).

**Reference:** patterns-and-antipatterns.md

---

### Q25: Infinite Loops?
**Answer:** Prevent with: iteration limits (max 20), progress-based termination (if delta <threshold after 5 iterations, abort), explicit success criteria checks, cost budget limits, cycle detection (same action 3x in row). Always include at least 2 of these mechanisms.

**Reference:** patterns-and-antipatterns.md (antipatterns 5, 10)

---

### Q26: Agents Ignoring Outputs?
**Answer:** Causes: no acknowledgment mechanism (add mandatory acknowledgment before proceeding), information not shared (implement broadcasting system with categories), poor prompts (explicitly include peer findings in context with "IMPORTANT: Other agents found...").

**Reference:** patterns-and-antipatterns.md (antipatterns 7-8)

---

## Domain-Specific Applications

### Q27: Code Review Agent?
**Answer:** Use review-critique pattern with three stages: automated checks (syntax, security, tests, coverage), LLM review (logic, performance, quality), senior engineer HITL for high/critical issues. Tools: security scanner, test runner, linter, AST parser.

**Reference:** framework-comparison.md, agentic-systems-cookbook.md

---

### Q28: Research Agent?
**Answer:** Use supervisor pattern: researcher gathers data (parallel web/academic/case study searches), analyst analyzes findings, writer creates report. Optimize costs: mini for gathering, gpt-4o for analysis, claude-sonnet for writing. Tools: Tavily, Semantic Scholar, arXiv, PDF reader.

**Reference:** multi-agent-patterns.md, agentic-systems-cookbook.md (Recipe 5)

---

### Q29: Support Agent?
**Answer:** Use router pattern: classify intent (FAQ, troubleshooting, account, complaint), route to specialized handlers (FAQ: cached fast responses, troubleshooting: ReAct, account: auto-resolve or escalate), HITL escalation for complaints/complexity. Targets: 70-80% automation, <2s response, >85% satisfaction.

**Reference:** multi-agent-patterns.md

---

### Q30: Content Pipeline?
**Answer:** Sequential with reflection: research→outline→write→critique→improve if score <0.8→fact-check if needed→format. Cost: ~$0.77/article (research $0.05, outline $0.02, writing $0.40, critique $0.10, revision $0.20).

**Reference:** agentic-systems-cookbook.md (Recipe 4)

---

## Advanced Topics & Research

### Q31: 2025 Developments?
**Answer:** Key innovations:

**Frontier Models:**
- **o3/o4-mini**: 88% ARC-AGI (high compute), 96.7% AIME, adjustable reasoning effort
- **Claude Opus 4.5**: 80.9% SWE-bench, 61.4% OSWorld, 1% prompt injection success
- **GPT-5/5.2**: 80.0% SWE-bench, 59.22% BFCL, 52.6% MCPMark
- **Gemini 3**: 45.1% ARC-AGI-2, 81% MMMU-Pro, 87.6% Video-MMMU, long-horizon planning
- **DeepSeek-V3.1**: Hybrid thinking/non-thinking mode (20-50% token reduction)

**Agent Platforms:**
- **ChatGPT Agent**: Integrated web browsing + tool use
- **OpenAI Operator**: Browser automation with CUA model
- **Cursor 2.0**: Composer model (4x faster), 8 parallel agents
- **Devin 2.0**: Price drop $500→$20/mo, 12x faster migrations (Nubank)

**Infrastructure:**
- **Mem0**: $24M Series A, 26% accuracy boost, 91% lower latency
- **MCP + A2A**: Donated to Linux Foundation, 2000+ MCP servers
- **Agent Sandbox**: Google's K8s primitive (KubeCon NA 2025)
- **Langfuse**: 19K+ GitHub stars, MIT license

**Research:**
- **MAST Taxonomy**: 14 failure modes, 44% from system design (not model)
- **RAFFLES**: Agentic debugging, 43% fault attribution accuracy

**Reference:** multi-agent-patterns.md, theoretical-foundations.md, api-optimization-guide.md, 2025-updates.md

---

### Q31a: What are Modern Memory Architectures?
**Answer:** See "Memory Systems for Agents (Mem0, GraphRAG)" section above for full details. Key points:

**Memory Types**: Short-term (session), Episodic (experiences), Semantic (facts), Procedural (patterns)

**Key Systems (2025)**:
- **Mem0**: $24M Series A, 26% accuracy boost, 91% lower latency, 41K GitHub stars
- **GraphRAG**: Microsoft's knowledge graph + RAG
- **MemGPT/Letta**: Self-editing memory blocks
- **MIRIX**: Multi-level (STM/MTM/LPM)

**Reference:** Memory Systems section, 2025-updates.md, arXiv:2404.13501

---

### Q31b: What is Experience-Following Behavior?
**Answer:** Research (arXiv:2505.16067, May 2025) reveals critical memory management insight:

**Finding**: High similarity between task input and retrieved memory record leads to highly similar agent outputs - agents "follow" past experiences closely.

**Challenges**:
1. **Error Propagation**: Mistakes in memory compound in future tasks
2. **Misaligned Replay**: Past experience may not fit current context
3. **Overreliance**: Agent trusts memory over current evidence

**Mitigation**:
- Curate memory carefully (don't store failures as successes)
- Include context with memories (when was this valid?)
- Validate retrieved memories against current task
- Allow overriding memory when evidence contradicts

**Reference:** arXiv:2505.16067

---

### Q32: Academic Failure Research?
**Answer:** Study (arXiv:2503.13657) found 25-75% failure rates across frameworks, identified 14 distinct failure modes (top: task violations 35%, role disobedience 28%, incomplete verification 42%). Key insight: prompt improvements alone only 14% gain, structural fixes (validation, protocols, state management) required.

**Reference:** patterns-and-antipatterns.md

---

### Q33: Reasoning Pattern Comparison?
**Answer:** Chain-of-Thought: 1 call, +334% on math (GSM8K). ReAct: 2-10 calls, +87% on multi-hop QA (HotpotQA). Tree-of-Thoughts: 10-100+ calls, +1750% on Game of 24. Selection: CoT for simple reasoning, ReAct for tool-augmented, ToT for complex search/planning.

**Reference:** theoretical-foundations.md

---

### Q33a: What are CoT Limitations?
**Answer:** Research (arXiv:2508.01191, Aug 2025) reveals CoT is a "brittle mirage" that fails outside training distributions. Key findings:
1. **Distribution Sensitivity**: CoT works on in-distribution problems but fails on novel patterns
2. **False Confidence**: Models produce convincing-looking reasoning that's factually wrong
3. **Prompt Dependence**: Small prompt changes cause large performance swings
4. **Verification Gap**: Models can't reliably verify their own reasoning
**Implication**: Don't rely solely on CoT for critical decisions - combine with validation, RAG grounding, and multi-agent verification.

**Reference:** arXiv:2508.01191, theoretical-foundations.md

---

### Q33b: What is Reason from Future (RFF)?
**Answer:** RFF (arXiv:2506.03673, Jun 2025) is a bidirectional reasoning paradigm:
1. **Top-Down Planning**: Start from goal, decompose backwards
2. **Bottom-Up Accumulation**: Build up from known facts
3. **Bidirectional Merge**: Combine both directions for answer
**Benefits**: Higher accuracy than unidirectional CoT, smaller search space for complex tasks.
**Use When**: Complex planning, goal-oriented tasks, multi-step problems.

**Reference:** arXiv:2506.03673

---

### Q33c: What is Language Agent Tree Search (LATS)?
**Answer:** LATS (arXiv:2310.04406) unifies reasoning, acting, and planning:
1. **Tree Structure**: Explore multiple reasoning paths simultaneously
2. **Monte Carlo Tree Search**: Use MCTS for path selection
3. **ReAct Integration**: Combine with tool use at each node
4. **Value Function**: Learn to predict path success
**Performance**: Outperforms both ToT and RAP with ReAct prompting. Best for complex, multi-step tasks requiring both reasoning and action.

**Reference:** arXiv:2310.04406, theoretical-foundations.md

---

### Q33d: What is RAFA (Reason for Future, Act for Now)?
**Answer:** RAFA (arXiv:2310.12346) provides provable regret guarantees for agent behavior:

**Core Mechanism:**
1. **Plan ahead**: At each state, plan trajectories multiple steps into future
2. **Act now**: Execute only the first action of that plan
3. **Observe**: Get environment feedback
4. **Update**: Revise beliefs about environment
5. **Replan**: Generate new plan from updated state

**Theoretical Guarantee**: Achieves √T regret bounds in Bayesian adaptive MDPs.

**Practical Benefit**: Principled framework for balancing planning depth vs execution speed. Optimal agent design involves specific ratios - over-planning wastes compute, under-planning increases errors.

**Reference:** arXiv:2310.12346, agentification.github.io/RAFA/

---

### Q33e: What is Graph Chain-of-Thought (Graph-CoT)?
**Answer:** Graph-CoT (arXiv:2511.01633) transforms sequential reasoning into tree structures:

**Key Innovation:**
- Convert linear CoT into branching tree structures
- Enable selective context sharing between branches
- Prune irrelevant reasoning paths early

**Performance:**
- **95.7% token reduction** compared to standard CoT
- **38% accuracy improvement** on complex reasoning tasks
- Particularly effective for problems requiring exploration of multiple solution paths

**Use When**: Complex reasoning with multiple valid approaches, resource-constrained deployments, tasks where exhaustive exploration is wasteful.

**Reference:** arXiv:2511.01633

---

### Q33f: What is the MAST Failure Taxonomy?
**Answer:** MAST (NeurIPS 2025, arXiv:2503.13657) analyzed 1,600+ execution traces across 7 frameworks to identify 14 failure modes in 3 categories:

**1. System Design Issues (44.2% of failures):**
| Failure Mode | Frequency | Description |
|--------------|-----------|-------------|
| Task disobedience | 15.7% | Agents ignore specifications |
| Step repetition/loops | 13.2% | Recursive loops, infinite cycling |
| Context window loss | 6.8% | History truncated mid-conversation |
| Tool misuse | 8.5% | Wrong tool selection or parameters |

**2. Inter-Agent Misalignment (32.3% of failures):**
| Failure Mode | Frequency | Description |
|--------------|-----------|-------------|
| Assumption failures | 12.4% | Agents fail to clarify assumptions |
| Task derailment | 11.8% | Agents pursue irrelevant discussions |
| Information withholding | 8.2% | Lossy state transfer between agents |

**3. Task Verification (23.5% of failures):**
| Failure Mode | Frequency | Description |
|--------------|-----------|-------------|
| Premature termination | 9.1% | Declare complete before criteria met |
| Incorrect verification | 2.8% | Generators can't see own errors |
| Missing validation | 11.6% | No output quality checks |

**Critical Insight**: Most failures stem from system design, not model capability. Focus on architectural patterns, not waiting for better models.

**Reference:** arXiv:2503.13657, NeurIPS 2025

---

### Q33g: What is the RAFFLES Debugging Framework?
**Answer:** RAFFLES (Capital One, 2025) treats agent debugging as an agentic task:

**Architecture:**
```
Execution Trace
    ↓
┌─────────────────────────────────┐
│         Judge Agent             │
│   Analyze trace, propose        │
│   hypotheses about failures     │
└──────────────┬──────────────────┘
    ↓
┌─────────────────────────────────┐
│       Evaluator Agents          │
│   Critique Judge's reasoning    │
│   Challenge weak hypotheses     │
└──────────────┬──────────────────┘
    ↓
Iterate until high-confidence attribution
```

**Performance**: 43% accuracy on fault attribution vs 16.6% for prior methods (2.6x improvement).

**Implication**: Debugging agents with agents is viable and effective. Build agentic observability systems.

**Reference:** NeurIPS 2025 Applied AI briefings

---

## Benchmarks & Evaluation

### Q42: What are the Key Agent Benchmarks?
**Answer:**
| Benchmark | Focus | Top Score (Dec 2025) | Tasks |
|-----------|-------|---------------------|-------|
| **SWE-bench Verified** | Code fixes | 80.9% (Claude Opus 4.5), 80.0% (GPT-5.2) | 2,294 GitHub issues |
| **SWE-bench+** | Harder code fixes | ~65% | Mitigates solution leakage |
| **GAIA** | General assistant | ~50% | 466 multi-step tasks |
| **WebArena** | Web automation | 58.1% (Operator) | 812 web tasks |
| **OSWorld** | Computer use | 61.4% (Claude), 38.1% (Operator) | GUI automation |
| **AgentBench** | Multi-domain | Varies | 8 environments |
| **BFCL** | Function calling | 90%+ (GPT-4o, Claude) | API accuracy |
| **MCPMark** | MCP implementation | 52.6% (GPT-5) | Tool integration |
| **ARC-AGI** | Abstract reasoning | 88% (o3 high-compute) | Novel patterns |
| **ARC-AGI-2** | Harder reasoning | 45.1% (Gemini 3), 2.9% (o3) | Harder abstract patterns |

**ARC-AGI vs ARC-AGI-2 Gap**: o3 scores 88% on ARC-AGI but only 2.9% on ARC-AGI-2, revealing generalization limits.

**Reference:** 2025-updates.md, leaderboard links in task.md

---

### Q43: What is AgentBench?
**Answer:** AgentBench (arXiv:2308.03688) evaluates agents across 8 diverse environments:
1. **Operating System** - File/system operations
2. **Database** - SQL query generation
3. **Knowledge Graph** - Entity reasoning
4. **Digital Card Game** - Strategic planning
5. **Lateral Thinking Puzzles** - Creative reasoning
6. **House-Holding** - Physical world simulation
7. **Web Shopping** - E-commerce navigation
8. **Web Browsing** - General web tasks

**Key Insight**: Tests planning, reasoning, tool use, and decision-making holistically. No single model excels across all 8 - reveals model-specific strengths/weaknesses.

**Reference:** arXiv:2308.03688

---

### Q44: What is BFCL (Berkeley Function Calling Leaderboard)?
**Answer:** BFCL V4 (ICML 2025) evaluates function calling accuracy:
- **Intent Recognition**: When is a function needed?
- **Function Selection**: Which function to call?
- **Parameter Mapping**: Extract correct arguments
- **Execution**: Proper invocation
- **Response Integration**: Process results correctly

**Top Performers (Dec 2025)**: GPT-4o, Claude Opus 4.5, Gemini 2.0 all score 90%+

**Use**: Benchmark your agent's tool use capability before production.

**Reference:** gorilla.cs.berkeley.edu/leaderboard.html

---

### Q45: What is SWE-bench+ and Why Does It Matter?
**Answer:** SWE-bench+ addresses flaws in original SWE-bench:
1. **Solution Leakage**: Original had test cases that leaked answers
2. **Weak Tests**: Some tests passed with incorrect solutions
3. **Difficulty Calibration**: More consistent challenge level

**Impact**: True agent performance is 15-20% lower than reported on original SWE-bench. Use SWE-bench+ for realistic capability assessment.

**Reference:** SWE-bench+ paper (2025)

---

## Security & Safety Questions

### Q37: How to Defend Against Prompt Injection?
**Answer:** Multi-layer defense:

**Detection Layers:**
1. **Input validation**: Pattern matching, encoding detection (base64, hex, URL), length limits
2. **Semantic filtering**: LLM-based intent classification
3. **Output filtering**: PII/credential redaction, system prompt leak detection
4. **Delimiter separation**: Clear boundaries between system and user content
5. **Monitor model**: Dedicated model watching for suspicious behavior (Operator approach)

**2025 State-of-the-Art:**
- **Claude Opus 4.5**: 1% attack success rate under Best-of-N adaptive attacks (Anthropic)
- **Key defenses**: RL exposure to injections during training, improved classifiers, human red teaming
- **Reality**: 1% is progress, not a solution - web remains adversarial environment

**OWASP Top 10 2025 Emphasis**: "Excessive Agency" - danger of unchecked LLM autonomy with external functions.

**Reference:** security-research.md, security-essentials.md, Anthropic prompt injection research

---

### Q38: What Sandboxing is Required for Tool Execution?
**Answer:** Multi-layer isolation with security/performance trade-offs:

**Layer Comparison:**
| Layer | Isolation | Performance | Best For |
|-------|-----------|-------------|----------|
| **Process** | Low | Fastest | Trusted internal code |
| **Container (LXC)** | Medium | Minimal overhead | Batch operations |
| **gVisor** | High | Moderate CPU cost | User-mode kernel isolation |
| **Firecracker microVM** | Highest | Slight overhead | Untrusted code execution |

**Firecracker Advantages (Recommended for Agent Tools):**
- Fresh VM creation per session
- Kernel-level isolation
- Deterministic cleanup
- Built-in network separation
- Optimal for agents executing untrusted commands

**Also Required:**
- Permission systems with whitelisted parameters
- Rate limiting per tool/user
- Parameter validation before execution
- Resource limits (CPU, memory, time)

**Reference:** security-research.md (Section 2.1), codeant.ai/agentic-rag-shell-sandboxing

---

### Q39: What Compliance Requirements Apply to AI Agents?
**Answer:**
- **EU AI Act**: High-risk systems (employment, credit, healthcare) require risk management, technical documentation, logging, human oversight, CE marking. Limited-risk (chatbots) require transparency disclosure.
- **GDPR**: Lawful basis, data minimization, user rights (access, deletion, portability), 72-hour breach notification.
- **OWASP Top 10 for LLM**: Prompt injection (#1), insecure output, training data poisoning, model DoS, supply chain, sensitive disclosure, insecure plugins, excessive agency, overreliance, model theft.
- **NIST AI RMF**: Mandates prompt injection controls and risk assessment.

**Reference:** security-essentials.md (Compliance Quick Check)

---

### Q40: How to Implement Human-in-the-Loop for Safety?
**Answer:** Risk-based approval framework: LOW risk (read operations) → auto-approve, MEDIUM risk (write, API calls) → require single approval, HIGH risk (delete, code execution) → require approval with logging, CRITICAL risk (financial, destructive) → block by default or require multi-party approval. Implement approval timeout (e.g., 5 minutes), audit logging, escalation procedures. Tools should have explicit permission policies with whitelisted parameters.

**Reference:** security-research.md (Section 2.3), security-essentials.md (SecureToolExecutor)

---

### Q41: What are the Key Alignment Challenges for Agents?
**Answer:**
1. **Goal Misspecification**: Agent optimizes for literal interpretation vs intended outcome
2. **Instrumental Convergence**: Agents develop unintended subgoals (self-preservation, resource acquisition)
3. **Deceptive Alignment**: Agent appears aligned during testing but not in production
4. **Reward Hacking**: Agent games metrics (e.g., offering unauthorized refunds to boost satisfaction)
5. **Multi-Agent Coordination**: Conflicting objectives, emergent unintended behaviors

**Mitigation**: Robust goal specification with success metrics, hard/soft constraints, resource limits, allowed/forbidden actions, verification tests. Use Constitutional AI principles and RLHF advances.

**Reference:** security-research.md (Section 3)

---

## Decision Frameworks & Checklists

### Q34: Go/No-Go Decision Checklist

**✅ GREEN LIGHTS (Proceed):**
- [ ] Task well-defined with clear success criteria
- [ ] Complex enough to justify cost (>5 manual hours per task)
- [ ] 5-10% error rate tolerance acceptable
- [ ] Budget allows $500-$5,000/month
- [ ] Team has LLM experience
- [ ] Human fallback available
- [ ] ROI timeline acceptable (3-9 months)

**⚠️ YELLOW LIGHTS (Caution):**
- [ ] Some task ambiguity (add clarification step)
- [ ] Regulatory/compliance constraints (add audit trails)
- [ ] Real-time requirements (<2s latency)
- [ ] Legacy system integration needed
- [ ] Limited budget (<$500/month)

**🛑 RED LIGHTS (Don't Proceed):**
- [ ] Zero error tolerance (life/safety critical)
- [ ] Task poorly defined or changing constantly
- [ ] Simple rule-based system would work
- [ ] Insufficient budget (<$200/month)
- [ ] No technical expertise
- [ ] No output validation method
- [ ] Legal/regulatory prohibits AI decisions

**Reference:** All documents (synthesized framework)

---

### Q35: Architecture Decision Template

```markdown
# Multi-Agent System Architecture

## 1. Requirements
- Task: [Specific description]
- Success Criteria: [Measurable outcomes]
- Targets: Latency <Xs, Accuracy >Y%, Cost <$Z/task

## 2. Single vs Multi-Agent
- Complexity: [1-2 / 3-5 / 5+ domains]
- Decision: [Single / Multi]
- Justification: [Why]

## 3. Architecture (if multi-agent)
- Pattern: [Collaboration / Supervisor / Swarm]
- Agents: [Agent name: role + responsibilities]
- Communication: [Message Bus / Shared State / MCP]

## 4. Models
- Primary: [GPT-4o / Claude Sonnet 4.5]
- Fallback: [gpt-4o-mini / claude-haiku]
- Routing: [Cascading / Fixed]

## 5. Tools
- Required: [5-10 specific tools]
- Selection: [Static / Dynamic RAG]
- Validation: [Method if yes]

## 6. State
- Schema: [TypedDict fields]
- Persistence: [PostgreSQL / Redis / Memory]
- Context: [Fixed window / Compression / Summarization]

## 7. Validation
- Layer 1: [Automated checks]
- Layer 2: [LLM critic]
- Layer 3: [HITL]
- Criteria: [Acceptance conditions]

## 8. Error Handling
- Retry: [Exponential backoff, max N]
- Fallback: [Cheaper model / Human]
- Circuit Breaker: [Threshold if yes]

## 9. Cost Management
- Budget: $X/day
- Controls: [Cascading / Caching / Limits]

## 10. Monitoring
- Metrics: [List]
- Alerts: [Thresholds]
- Dashboard: [Key displays]

## 11. Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
```

**Reference:** All documents (synthesized template)

---

### Q36: Production Readiness Checklist

**✅ FUNCTIONALITY (100% Required):**
- [ ] All features implemented
- [ ] Tests passing (>95% coverage)
- [ ] Multi-layer validation working
- [ ] Comprehensive error handling
- [ ] Tools/functions reliable

**✅ PERFORMANCE (100% Required):**
- [ ] Latency p95 < target
- [ ] Success rate >90%
- [ ] Load tested (2x peak)
- [ ] No memory leaks
- [ ] Caching implemented

**✅ COST (100% Required):**
- [ ] Cost tracking live
- [ ] Budget limits enforced
- [ ] Model cascading working
- [ ] Alerts configured
- [ ] Cost/request < target

**✅ MONITORING (100% Required):**
- [ ] Metrics dashboard live
- [ ] Alerts configured
- [ ] Comprehensive logging
- [ ] Tracing enabled
- [ ] Error tracking (Sentry)

**✅ SECURITY (100% Required):**
- [ ] Input validation
- [ ] Output sanitization
- [ ] Tool sandboxing
- [ ] API keys secured
- [ ] Rate limiting
- [ ] Audit trails

**✅ DOCUMENTATION (100% Required):**
- [ ] Architecture documented
- [ ] API documentation
- [ ] Runbooks for common issues
- [ ] Disaster recovery plan
- [ ] Escalation procedures

**✅ OPERATIONAL (Recommended):**
- [ ] Deployment automated
- [ ] Rollback tested
- [ ] Backup/restore tested
- [ ] On-call rotation
- [ ] Incident response plan

**✅ BUSINESS (Recommended):**
- [ ] Stakeholder sign-off
- [ ] User acceptance testing
- [ ] Success metrics defined
- [ ] ROI tracking plan
- [ ] Feedback mechanism

**Go-Live Criteria:**
- All "Required" items: ✅
- 80% of "Recommended": ✅
- Stakeholder approval: ✅
- Rollback ready: ✅

**Reference:** workflow-overview.md, patterns-and-antipatterns.md

---

## Quick Reference Card

**Cost Optimization (50-80% reduction):**
1. Model cascading (use mini for simple)
2. Prompt caching (90% off cached reads)
3. Progress tracking (avoid redundant work)
4. Tool RAG (5-10 vs 50 tools)
5. Response caching (deterministic queries)

**Common Failure Fixes:**
- Vague specs → Add success criteria + validation + examples
- Role violations → Programmatic enforcement + forbidden lists
- Context loss → Summarization + mark critical messages
- Infinite loops → Iteration limits + progress tracking + success criteria
- Agents ignoring → Acknowledgment + broadcasting + explicit prompts

**Model Selection Quick Guide (December 2025):**
- Router/Simple: gpt-4o-mini, claude-haiku
- Structured output: gpt-4o-2024-08-06 (100% adherence)
- Complex reasoning: o3/o4-mini (88% ARC-AGI), Claude Opus 4.5 + extended thinking
- Coding tasks: Claude Opus 4.5 (80.9% SWE-bench), GPT-5.2 (80.0%), o3 (69.1%)
- Long-form writing: Claude Sonnet 4.5
- Browser automation: Claude Computer Use (61.4% OSWorld), OpenAI Operator (38.1%)
- Multi-modal: Gemini 3 (81% MMMU-Pro, 87.6% Video-MMMU)
- Cost-efficient reasoning: DeepSeek-V3.1 (hybrid think/non-think)
- Speed critical: claude-haiku or gpt-4o-mini

**Architecture Quick Select:**
- 2-3 domains + shared context → Collaboration
- 3-5 domains + sequential → Supervisor
- 5+ domains + dynamic → Swarm

**Essential Production Components:**
1. Input validation → Prevent injection/abuse
2. Multi-layer validation → Catch errors early
3. Error handling → Retry + fallback + circuit breaker
4. Cost tracking → Budget limits + alerts
5. Monitoring → Metrics + alerts + logging

**Security Quick Reference:**
- Prompt injection defense: Multi-layer (pattern + semantic + output filtering)
- Tool sandboxing: Process → Container → VM (layered)
- HITL approval: Low=auto, Medium=approve, High=approve+log, Critical=block
- Compliance: EU AI Act (high-risk systems), GDPR, OWASP Top 10
- Incident response: P0(<immediate), P1(<15m), P2(<1h), P3(next day)

**Security Metrics Thresholds:**
- Blocked requests rate: >5% → Warning
- Prompt injection attempts: >10/hour → Alert
- Failed auth: >50/hour → Alert
- Error rate: >1% → Warning
- P95 latency: >3s → Warning

---

**You are now equipped to build production-grade multi-agent systems.**

**For detailed implementations, code examples, and deep dives, see:**

**Core Architecture & Patterns:**
- framework-comparison.md - LangGraph, CrewAI, AutoGPT comparison
- multi-agent-patterns.md - Multi-agent architectures (2025)
- patterns-and-antipatterns.md - 14 failure modes and fixes
- workflow-overview.md - 12-stage production workflow

**Implementation & Optimization:**
- agentic-systems-cookbook.md - 11 production-ready recipes
- api-optimization-guide.md - Model selection and cost strategies
- theoretical-foundations.md - Research citations and theory
- advanced-agent-paradigms.md - Self-improvement, planning patterns

**Evaluation & Security:**
- evaluation-and-debugging.md - Evaluation, tracing, improvement
- security-research.md - Complete security research (3,200+ lines)
- security-essentials.md - Production security checklists
- 2025-updates.md - Latest models, agents, MCP, memory

**Meta:**
- README.md - Entry point and navigation
- task.md - Research log and resources

**Last Updated:** 2025-12-25
