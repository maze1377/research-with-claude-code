# Agentic AI Developer Onboarding Guide

> **From Zero to Production-Ready Agent Developer**

**Last Updated:** 2025-12-27 | **Phase-Based Learning** | **100,000+ Words** | **30+ Documents**

---

## Quick Start by Experience Level

### Complete Beginner (No ML Background)
**Start here** → [Phase 0: Prerequisites](phase-0-prerequisites/prerequisites.md)
- Python essentials, ML basics, LLM fundamentals
- Development environment setup
- API basics and first API calls

### ML/AI Practitioner (Know ML, New to Agents)
**Start here** → [Phase 1: Foundations](phase-1-foundations/)
- What makes an agent different from a chatbot
- Framework landscape (LangGraph, CrewAI, OpenAI SDK)
- Core patterns: ReAct, CoT, ToT

### Experienced Developer (Fast Track)
**Start here** → [Phase 2: Building Agents](phase-2-building-agents/)
- Build your first production agent
- Tool development and MCP servers
- Memory systems and persistence

---

## Learning Path Overview

```
Phase 0: Prerequisites ─────────────────────────────────────────┐
  │ Python, ML basics, LLM fundamentals                        │
  ▼                                                             │
Phase 1: Foundations ─────────────────────────────────────────┐ │
  │ Frameworks, patterns, when to use agents                  │ │
  ▼                                                           │ │
Phase 2: Building Agents ─────────────────────────────────────┤ │
  │ Prompting, tools, memory, first agent                     │ │
  ▼                                                           │ │
Phase 3: Patterns ────────────────────────────────────────────┤ │
  │ Multi-agent, supervisor, swarm, anti-patterns             │ │
  ▼                                                           │ │
Phase 4: Production ──────────────────────────────────────────┤ │
  │ Testing, CI/CD, evaluation, debugging                     │ │
  ▼                                                           │ │
Phase 5: Security & Compliance ───────────────────────────────┤ │
  │ Prompt injection, sandboxing, governance                  │ │
  ▼                                                           │ │
Phase 6: Advanced ────────────────────────────────────────────┘ │
  │ Self-improvement, MCP deep dive, DSPy                       │
  └─────────────────────────────────────────────────────────────┘
```

**Parallel Tracks** (Any time after Phase 2):
- [Product Strategy](product-strategy/) - Build vs buy, ROI, team structure
- [Developer Productivity](developer-productivity/) - AI coding tools, workflows

---

## Competency Checklist

**Track your progress** → [COMPETENCY-CHECKLIST.md](COMPETENCY-CHECKLIST.md)

Quick self-assessment:
- [ ] Can explain ReAct pattern and when to use it
- [ ] Can build a custom tool with proper error handling
- [ ] Can implement Supervisor vs Swarm patterns
- [ ] Can defend against prompt injection attacks
- [ ] Can deploy an agent with CI/CD pipeline

---

## What's In This Guide

### Phase 0: Prerequisites
| Document | Purpose |
|----------|---------|
| [prerequisites.md](phase-0-prerequisites/prerequisites.md) | Python, ML basics, LLM fundamentals, environment setup |

### Phase 1: Foundations
| Document | Purpose |
|----------|---------|
| [theoretical-foundations.md](phase-1-foundations/theoretical-foundations.md) | ReAct, CoT, ToT, LATS paradigms with academic citations |
| [framework-comparison.md](phase-1-foundations/framework-comparison.md) | LangGraph, CrewAI, AutoGPT, OpenAI SDK, Claude SDK comparison |
| [llm-fundamentals.md](phase-1-foundations/llm-fundamentals.md) | Tokens, context windows, prompting basics, API patterns |

### Phase 2: Building Agents
| Document | Purpose |
|----------|---------|
| [agent-prompting-guide.md](phase-2-building-agents/agent-prompting-guide.md) | System prompts, XML patterns, multi-agent prompting |
| [agentic-systems-cookbook.md](phase-2-building-agents/agentic-systems-cookbook.md) | 11 production recipes with code |
| [tool-development-guide.md](phase-2-building-agents/tool-development-guide.md) | Creating production tools, MCP servers |
| [memory-systems-guide.md](phase-2-building-agents/memory-systems-guide.md) | Vector DBs, memory architectures, Mem0 |

### Phase 3: Patterns
| Document | Purpose |
|----------|---------|
| [multi-agent-patterns.md](phase-3-patterns/multi-agent-patterns.md) | Supervisor, Swarm, coordination + 6 architecture blueprints |
| [patterns-and-antipatterns.md](phase-3-patterns/patterns-and-antipatterns.md) | 14 failure modes, 12-Factor Agents |
| [workflow-overview.md](phase-3-patterns/workflow-overview.md) | 12-stage production workflow |

### Phase 4: Production
| Document | Purpose |
|----------|---------|
| [evaluation-and-debugging.md](phase-4-production/evaluation-and-debugging.md) | EDD, benchmarks, Agent Maturity Model |
| [api-optimization-guide.md](phase-4-production/api-optimization-guide.md) | Cost optimization, model selection |
| [testing-guide.md](phase-4-production/testing-guide.md) | Unit testing, integration testing, evaluation |
| [ci-cd-guide.md](phase-4-production/ci-cd-guide.md) | Deployment pipelines, monitoring |

### Phase 5: Security & Compliance
| Document | Purpose |
|----------|---------|
| [security-essentials.md](phase-5-security-compliance/security-essentials.md) | Prompt injection, sandboxing, HITL |
| [security-research.md](phase-5-security-compliance/security-research.md) | Full security research reference |
| [governance-compliance.md](phase-5-security-compliance/governance-compliance.md) | GDPR, HIPAA, EU AI Act, audit trails |

### Phase 6: Advanced
| Document | Purpose |
|----------|---------|
| [advanced-agent-paradigms.md](phase-6-advanced/advanced-agent-paradigms.md) | Reflexion, LADDER, AlphaEvolve |
| [mcp-deep-dive.md](phase-6-advanced/mcp-deep-dive.md) | Building MCP servers, advanced patterns |
| [dspy-guide.md](phase-6-advanced/dspy-guide.md) | Programmatic prompting and optimization |
| [cross-framework-migration.md](phase-6-advanced/cross-framework-migration.md) | When and how to switch frameworks |

### Product Strategy
| Document | Purpose |
|----------|---------|
| [product-strategy-guide.md](product-strategy/product-strategy-guide.md) | Build vs buy, ROI, vendor evaluation |
| [case-studies.md](product-strategy/case-studies.md) | Real production deployments and lessons |

### Developer Productivity
| Document | Purpose |
|----------|---------|
| [developer-productivity-guide.md](developer-productivity/developer-productivity-guide.md) | Cursor, Claude Code, RIPER, SDD |
| [collaboration-guide.md](developer-productivity/collaboration-guide.md) | Team workflows, open-source contribution |

### Reference
| Document | Purpose |
|----------|---------|
| [topics.md](reference/topics.md) | 102 Q&A quick reference |
| [2025-updates.md](reference/2025-updates.md) | Latest developments, model capabilities |
| [glossary.md](reference/glossary.md) | Terms and definitions |

---

## Quick Reference

### Key Numbers (December 2025)

| Metric | Value | Source |
|--------|-------|--------|
| Claude Opus 4.5 SWE-bench | **80.9%** | Anthropic |
| GPT-5 AIME (Math) | 94.6% | OpenAI |
| Agent S OSWorld | 72.6% | Beats human 72.36% |
| MCP Servers | 10,000+ | Linux Foundation |
| AI code globally | 41% | GitHub |
| AI code with vulnerabilities | 45% | Veracode |

### 12-Factor Agents (Production Essentials)
```
1. NL → Tool Calls      # LLM routes intent to actions
2. Own Your Prompts     # Version control, A/B test
3. Own Your Context     # Explicit token budgets
4. Tools = Schema       # Validation on everything
5. Unify State          # Single source of truth
6. Pause/Resume         # Interruptible workflows
7. Humans via Tools     # Escalation as explicit tool
8. Own Control Flow     # Explicit loops
9. Compact Errors       # Fit in context
10. Small Agents        # 5-10 tools max
11. Trigger Anywhere    # HTTP, queue, cron
12. Stateless Reducers  # (state, event) → state
```

### Architecture Quick Selection
```
2-3 domains + shared context → Collaboration Pattern
3-5 domains + sequential     → Supervisor Pattern
5+ domains + dynamic routing → Swarm Pattern
```

### Model Selection (December 2025)
```
Router/Simple     → gpt-4o-mini, claude-haiku
Structured output → gpt-4o-2024-08-06 (100% JSON)
Complex reasoning → GPT-5, Claude Opus 4.5 + extended thinking
Coding tasks      → Claude Opus 4.5, GPT-5
```

---

## Essential External Resources

### Leaderboards (Check Weekly)
- [Chatbot Arena](https://lmarena.ai/) - Model rankings
- [SWE-bench](https://www.swebench.com/) - Coding agents
- [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) - Function calling

### Research (Check Daily)
- [arXiv cs.AI](https://arxiv.org/list/cs.AI/recent) - AI papers
- [Hugging Face Papers](https://huggingface.co/papers) - Daily summaries

### Documentation
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Multi-agent framework
- [MCP Protocol](https://modelcontextprotocol.io/) - Tool integration standard
- [OWASP GenAI](https://genai.owasp.org/) - Security guidelines

---

## Contributing

See [CLAUDE.md](CLAUDE.md) for contribution guidelines.
See [task.md](task.md) for research tracking.

---

**Built with research from**: Anthropic, OpenAI, Google, LangChain, Microsoft, Stanford, OWASP, and 60+ academic papers
