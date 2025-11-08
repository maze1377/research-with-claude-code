# Agentic Systems Mastery: Quick Reference Guide

**Purpose:** Fast-lookup reference for building production multi-agent systems. For detailed implementations, see linked research documents.

**Last Updated:** 2025-11-08

**Knowledge Foundation:** 12 comprehensive documents, 18 academic papers, 14 failure modes, 11 production recipes, 4 case studies

---

## Quick Navigation: 36 Critical Questions

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
31. 2025 developments? → Command Tool, extended thinking, structured outputs (100% adherence), prompt caching
32. Academic failures? → 25-75% failure rates, 14 distinct modes, prompting alone only 14% improvement
33. Reasoning patterns? → CoT (1 call), ReAct (2-10 calls), ToT (10-100+ calls)

**Decision Frameworks (34-36):**
34. Go/No-Go decision? → See comprehensive checklist below
35. Architecture planning? → Use full decision template below
36. Production ready? → See production readiness checklist below

---

## Business & Strategy Questions

### Q1: Build vs Buy?
**Answer:** Use existing frameworks (LangGraph, CrewAI) when time-to-market is critical and standard workflows fit. Build custom when you need unique workflows, fine-grained control, specific compliance, or operate at large scale (>1M requests/month).

**Reference:** findings-langgraph.md, findings-crewai-autogpt.md

---

### Q2: Single vs Multi-Agent?
**Answer:** Use single agent for 1-2 domains where sequential processing is acceptable. Use multi-agent for 3+ distinct domains where parallelization provides value and quality justifies added complexity (~30% cost increase).

**Reference:** langgraph-multi-agent-patterns.md, theoretical-foundations.md

---

### Q3: ROI Calculation?
**Answer:** Development costs $50K-$200K, API costs $500-$5K/month. Break-even: 6-9 months (small), 3-6 months (medium), 1-3 months (large). Benefits: 70-80% support automation, 10x content speed, 50% code review time savings.

**Reference:** api-optimization-guide.md

---

### Q4: Proven Use Cases?
**Answer:** High ROI (>300%): Customer support (75% automation, <2s response), code migration (90% accuracy, 10x speed), content generation ($0.50-$2/article). Medium ROI: Research, data processing, QA. Low ROI: Simple classification, real-time latency-critical, highly regulated domains.

**Reference:** langgraph-multi-agent-patterns.md (case studies)

---

### Q5: Business Risks?
**Answer:** Critical risks: hallucinations (mitigate with multi-layer validation, +30% dev cost), cost runaway (budget limits, circuit breakers), security vulnerabilities (input validation, sandboxing, +15% dev time). Medium risks: performance issues, vendor lock-in, compliance violations.

**Reference:** patterns-and-antipatterns.md

---

### Q6: Build Stakeholder Confidence?
**Answer:** Use transparency mechanisms (reasoning traces, confidence scores, audit trails), validation framework (A/B testing, 5-10% human validation, metrics dashboard), and staged rollout (shadow→HITL→automated monitoring→full automation).

**Reference:** final-workflow.md, agentic-systems-cookbook.md (Recipe 4)

---

## Technical Architecture Questions

### Q7: Architecture Selection?
**Answer:** Collaboration for 2-3 domains with shared context, Supervisor for 3-5 domains with sequential stages, Swarm for 5+ domains with dynamic/exploratory workflows. Consider workflow type (sequential vs parallel) and complexity tolerance.

**Reference:** langgraph-multi-agent-patterns.md, theoretical-foundations.md

---

### Q8: 2025 LangGraph Features?
**Answer:** Use Command Tool for dynamic routing (type-safe), Handoffs for explicit agent transitions, Supervisor/Swarm libraries for pre-built patterns. These reduce boilerplate and improve reliability vs static graph edges.

**Reference:** langgraph-multi-agent-patterns.md

---

### Q9: State Schema Design?
**Answer:** Keep state flat with TypedDict for type safety. Include: messages (with add_messages reducer), current execution state, task data, artifacts dict, metadata (cost, timing), and control flags. Add cost tracking from start, not later.

**Reference:** findings-langgraph.md, agentic-systems-cookbook.md (Recipe 5)

---

### Q10: Agent Communication?
**Answer:** Use message bus for complex systems (complete audit trail, type-safe, async support), shared state for simple systems (lower overhead, LangGraph built-in), or MCP for external tool integration (overkill for internal comms).

**Reference:** patterns-and-antipatterns.md (Pattern 3), theoretical-foundations.md

---

### Q11: Essential Components?
**Answer:** Core: input processing (validation, sanitization), task execution (ReAct loop, tool calling), validation (output verification, confidence scoring), error handling (retry, fallback, circuit breaker), monitoring (cost, latency, error rates). Optional: reflection, memory management, HITL.

**Reference:** final-workflow.md, workflow-components.md

---

### Q12: Tool/Function Calling?
**Answer:** Use RAG to select 5-10 relevant tools from larger set (not all 50). Validate tool existence, parameters, types, and values. Provide high-quality descriptions with clear use cases. Always wrap in try-catch with meaningful error messages to LLM.

**Reference:** agentic-systems-cookbook.md (Recipes 7-8), api-optimization-guide.md

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

**Reference:** findings-design-patterns.md, agentic-systems-cookbook.md

---

### Q28: Research Agent?
**Answer:** Use supervisor pattern: researcher gathers data (parallel web/academic/case study searches), analyst analyzes findings, writer creates report. Optimize costs: mini for gathering, gpt-4o for analysis, claude-sonnet for writing. Tools: Tavily, Semantic Scholar, arXiv, PDF reader.

**Reference:** langgraph-multi-agent-patterns.md, agentic-systems-cookbook.md (Recipe 5)

---

### Q29: Support Agent?
**Answer:** Use router pattern: classify intent (FAQ, troubleshooting, account, complaint), route to specialized handlers (FAQ: cached fast responses, troubleshooting: ReAct, account: auto-resolve or escalate), HITL escalation for complaints/complexity. Targets: 70-80% automation, <2s response, >85% satisfaction.

**Reference:** langgraph-multi-agent-patterns.md

---

### Q30: Content Pipeline?
**Answer:** Sequential with reflection: research→outline→write→critique→improve if score <0.8→fact-check if needed→format. Cost: ~$0.77/article (research $0.05, outline $0.02, writing $0.40, critique $0.10, revision $0.20).

**Reference:** agentic-systems-cookbook.md (Recipe 4)

---

## Advanced Topics & Research

### Q31: 2025 Developments?
**Answer:** Key innovations: LangGraph Command Tool (dynamic routing), Supervisor/Swarm libraries (pre-built patterns), extended thinking (96.5% physics accuracy), GPT-4o structured outputs (100% schema adherence vs 40% before), Anthropic prompt caching (90% savings), Model Context Protocol (standardized integration).

**Reference:** langgraph-multi-agent-patterns.md, theoretical-foundations.md, api-optimization-guide.md

---

### Q32: Academic Failure Research?
**Answer:** Study (arXiv:2503.13657) found 25-75% failure rates across frameworks, identified 14 distinct failure modes (top: task violations 35%, role disobedience 28%, incomplete verification 42%). Key insight: prompt improvements alone only 14% gain, structural fixes (validation, protocols, state management) required.

**Reference:** patterns-and-antipatterns.md

---

### Q33: Reasoning Pattern Comparison?
**Answer:** Chain-of-Thought: 1 call, +334% on math (GSM8K). ReAct: 2-10 calls, +87% on multi-hop QA (HotpotQA). Tree-of-Thoughts: 10-100+ calls, +1750% on Game of 24. Selection: CoT for simple reasoning, ReAct for tool-augmented, ToT for complex search/planning.

**Reference:** theoretical-foundations.md

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

**Reference:** final-workflow.md, patterns-and-antipatterns.md

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

**Model Selection Quick Guide:**
- Router/Simple: gpt-4o-mini ($0.15/1M input)
- Structured output: gpt-4o-2024-08-06 (100% adherence)
- Complex reasoning: Claude Sonnet 4.5 + extended thinking
- Long-form writing: Claude Sonnet 4.5
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

---

**You are now equipped to build production-grade multi-agent systems.**

**For detailed implementations, code examples, and deep dives, see:**
- findings-langgraph.md
- langgraph-multi-agent-patterns.md
- patterns-and-antipatterns.md
- agentic-systems-cookbook.md
- api-optimization-guide.md
- theoretical-foundations.md

**Last Updated:** 2025-11-08
