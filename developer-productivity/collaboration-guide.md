# Team Collaboration Guide for AI Agent Development

**Best practices for teams building AI agents and LLM-powered applications**

**Last Updated:** 2025-12-27 | Version 1.0

---

## Table of Contents

1. [Team Workflows](#1-team-workflows)
2. [Git Workflows for Agents](#2-git-workflows-for-agents)
3. [Documentation Practices](#3-documentation-practices)
4. [Open-Source Contribution](#4-open-source-contribution)
5. [Communication Patterns](#5-communication-patterns)
6. [Team Structure](#6-team-structure)
7. [Quick Reference](#7-quick-reference)

---

## 1. Team Workflows

### Pair Programming with AI

AI coding assistants do not replace pair programming. They serve a fundamentally different purpose.

> **"Framing coding assistants as pair programmers ignores one of the key benefits of pairing: to make the team, not just individual contributors, better."** -- Thoughtworks Technology Radar

**What AI Assistants Provide:**
- Getting unstuck on technical problems
- Learning about new technologies
- Onboarding to codebases
- Accelerating tactical coding work

**What AI Assistants Do NOT Provide:**
- Keeping work-in-progress low
- Reducing handoffs and relearning
- Enabling continuous integration through collaboration
- Improving collective code ownership
- Knowledge transfer between team members

**Human-AI Collaboration Model:**

```
Traditional Pair Programming:
  Developer 1 (Driver) ←→ Developer 2 (Navigator)
  - Shared context
  - Knowledge transfer
  - Real-time feedback

Human-AI Pairing:
  Developer (Navigator) ←→ AI (Coder)
  - Human provides strategic direction
  - AI handles syntax and implementation
  - Human maintains architectural oversight
```

**Effective Patterns:**

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **Plan-Act-Reflect** | Human writes spec, AI implements, human reviews | Feature development |
| **Visibility+Control** | AI shows each step, human can interrupt | Complex refactoring |
| **Shared Knowledge Base** | Team standards in CLAUDE.md/.mdc files | Daily development |

### Code Review Patterns for AI-Generated Code

AI-generated code requires specialized review approaches because 45% contains security vulnerabilities and 62% has design flaws.

**Multi-Stage Validation Pipeline:**

```
Stage 1: AUTOMATED CHECKS
  ├── Linting (ESLint, Prettier)
  ├── Type checking (TypeScript)
  ├── Static analysis (SonarQube)
  └── Security scanning (Snyk, OWASP)

Stage 2: AI CODE REVIEW
  ├── Logic error detection
  ├── Pattern compliance
  └── Architectural fit

Stage 3: HUMAN REVIEW
  ├── Business logic correctness
  ├── Architectural decisions
  └── Security-sensitive areas
```

**Junior Developer Mental Model:**

Treat AI as a capable junior developer:
- **Capable:** Can write functional code quickly
- **Fast:** Completes tasks efficiently
- **Eager:** Will attempt anything asked
- **Needs guidance:** May miss edge cases and security issues
- **Requires review:** All output needs verification

**Code Review Checklist for AI-Generated Code:**

```markdown
## Logic and Correctness
- [ ] Does the code do what was requested?
- [ ] Are edge cases handled?
- [ ] Is error handling appropriate?

## Security
- [ ] No injection vulnerabilities?
- [ ] Secrets handled properly?
- [ ] Input validation present?

## Architecture
- [ ] Follows project patterns?
- [ ] No unnecessary complexity?
- [ ] Appropriate abstractions?

## Testing
- [ ] Tests cover happy path?
- [ ] Tests cover error cases?
- [ ] Integration tests if needed?
```

### Knowledge Sharing and Documentation

**Weekly AI Tools Sync (15 minutes):**
1. What prompts worked well this week?
2. What AI mistakes did we catch?
3. New patterns to share?
4. Updates to shared configuration files?

**Shared Prompt Library:**

```
.team/prompts/
├── new-endpoint.md      # Standard API endpoint pattern
├── component.md         # React component template
├── migration.md         # Database migration pattern
├── security-review.md   # Security audit checklist
└── refactor.md          # Refactoring workflow
```

**AI-Powered Knowledge Systems:**

Modern teams implement knowledge systems that:
- Actively organize and surface relevant information
- Identify duplicate content and flag outdated materials
- Extract insights from informal communications
- Provide real-time knowledge assistance integrated into tools

### Reference Application Anchoring

Martin Fowler's approach: anchor coding agents to a reference application as contextual ground truth.

```
┌─────────────────────────────────────────────────────────────┐
│              REFERENCE APPLICATION PATTERN                   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              REFERENCE APPLICATION                   │   │
│  │  A working implementation demonstrating:             │   │
│  │  - Architectural patterns                            │   │
│  │  - Coding standards                                  │   │
│  │  - Integration patterns                              │   │
│  │  - Test structure                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              AI CODING AGENT                         │   │
│  │  "Follow the patterns in the reference app when      │   │
│  │   implementing new features in the target codebase"  │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              TARGET CODEBASE                         │   │
│  │  New code follows established patterns automatically │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**

```markdown
<!-- CLAUDE.md for reference anchoring -->
## Reference Application

When implementing new features, follow patterns from:
- Reference app: /reference-app/
- Key patterns to follow:
  - API structure: /reference-app/src/api/
  - Component structure: /reference-app/src/components/
  - Test patterns: /reference-app/tests/

## Pattern Requirements
Always check reference app before implementing:
1. How similar features are structured
2. Error handling patterns used
3. Test coverage expectations
```

**Benefits:**
- Prevents code pattern drift across the codebase
- AI agents apply updated patterns consistently
- Maintains architectural coherence as teams scale

### Team AI Adoption Stages

Organizations progress through distinct maturity levels:

| Stage | Characteristics | Focus Areas |
|-------|-----------------|-------------|
| **Aware** | Recognize AI value, no clear path | Discovery workshops, use case identification |
| **Active** | Experimenting, prototypes only | Bridge prototyping to production, build MLOps |
| **Operational** | Deploying to production | Manage multiple deployments, scale across teams |
| **Systemic** | AI at scale across organization | Infrastructure, platforms, performance measurement |
| **Transformational** | AI drives strategy | Ethics, empowerment, next-generation challenges |

**Key Principle:** AI adoption follows "use > observe > refine > scale" rather than traditional "plan > buy > train > go live" patterns.

---

## 2. Git Workflows for Agents

### Prompt Versioning

Prompts and configurations require the same version control discipline as application code.

**Semantic Versioning for Prompts:**

| Version Part | When to Increment | Example |
|--------------|-------------------|---------|
| **Major (X.0.0)** | Breaking changes, fundamental restructuring | Output format changes |
| **Minor (0.X.0)** | Feature additions, backward-compatible | New reasoning steps |
| **Patch (0.0.X)** | Bug fixes, minor refinements | Typo corrections |

**Core Principle: Immutability**

Once a version is created, it must never be modified. If changes are required, create a new version.

**What to Version:**
- Prompt text
- Model selection (GPT-4, Claude 3.5 Sonnet)
- Model parameters (temperature, top-p, max tokens)
- System instructions
- Performance metrics for each version

**Example Prompt Version Record:**

```yaml
version: "2.3.1"
created: "2025-12-27"
author: "team-member"
model: "claude-sonnet-4-20250514"
parameters:
  temperature: 0.7
  max_tokens: 4096
prompt: |
  You are a customer service agent...
changelog: |
  Fixed edge case handling for multi-language requests
metrics:
  accuracy: 0.94
  latency_p50: 1.2s
```

### Configuration Management

**Configuration File Structure:**

```yaml
# agent-config.yaml
agent:
  name: "customer-support-agent"
  version: "1.2.0"
  description: "Handles tier-1 customer inquiries"

model:
  provider: "anthropic"
  model_id: "claude-sonnet-4-20250514"
  parameters:
    temperature: 0.7
    max_tokens: 4096
    top_p: 0.9

capabilities:
  - name: "ticket_lookup"
    tool: "crm_api"
    permissions: ["read"]
  - name: "refund_processing"
    tool: "payment_api"
    permissions: ["read", "write"]
    approval_required: true

guardrails:
  max_actions_per_session: 10
  require_human_approval:
    - "refund > $100"
    - "account_deletion"
```

**Validation:**
- Validate configurations against schemas before deployment
- Implement pre-commit hooks catching configuration errors
- Run validation in CI/CD pipelines

### Environment Handling

**Environment-Specific Configurations:**

| Environment | Purpose | Model Selection | Human Oversight |
|-------------|---------|-----------------|-----------------|
| **Development** | Rapid iteration | Faster/cheaper models | Minimal |
| **Staging** | Production mirror | Production models | Moderate |
| **Production** | User-facing | Validated configurations | Full |

**Environment Variable Patterns:**

```bash
# .env.example (documented, committed)
DEV_MODEL_NAME=claude-sonnet-4-20250514
DEV_TEMPERATURE=0.9

STAGING_MODEL_NAME=claude-sonnet-4-20250514
STAGING_TEMPERATURE=0.7

PROD_MODEL_NAME=claude-sonnet-4-20250514
PROD_TEMPERATURE=0.7
```

**Promotion Workflow:**

```
Development → Staging → Production

1. Development: Iterate freely
2. Staging: Validate with production-like data
3. Production: Deploy with explicit approval
4. Each promotion requires:
   - Passing automated tests
   - Performance metric validation
   - Explicit approval from designated reviewers
```

### Secret Management

**Never store secrets in:**
- Source code repositories
- Configuration files
- Environment variable defaults
- Any version-controlled location

**Use dedicated secret management:**
- AWS Secrets Manager
- Azure Key Vault
- Google Secret Manager
- HashiCorp Vault

**Best Practices:**

| Practice | Implementation |
|----------|----------------|
| Development secrets | Personal `.env` files in `.gitignore` |
| CI/CD secrets | Platform-provided secret injection |
| Production secrets | Dynamic retrieval from secret managers |
| API key rotation | Automated rotation every 30-90 days |
| Access audit | Log all secret access with identity |

**Pre-commit Hook for Secret Detection:**

```bash
#!/bin/bash
# .git/hooks/pre-commit
if git diff --cached --diff-filter=ACM | grep -E "(api_key|secret|password|token)" > /dev/null; then
    echo "ERROR: Potential secret detected in commit"
    exit 1
fi
```

### PR Review for Prompts

**What to Check in Prompt PRs:**

1. **Functional correctness:**
   - Does the prompt change achieve intended outcome?
   - Are edge cases handled?

2. **Performance impact:**
   - Evaluation results attached?
   - Comparison to baseline version?

3. **Security considerations:**
   - No hardcoded secrets?
   - Proper input validation?
   - Injection attack prevention?

4. **Consistency:**
   - Follows team prompt patterns?
   - Compatible with existing tools?

**Required Reviewers by Change Type:**

| Change Type | Required Reviewers |
|-------------|-------------------|
| Minor prompt refinements | 1 AI engineer |
| New system instructions | AI engineer + Product manager |
| Model selection changes | AI engineer + Tech lead |
| Security-related changes | AI engineer + Security team |

**Evaluation Results in PR:**

```markdown
## Prompt Change Evaluation

### Test Dataset: customer-support-v2 (n=500)

| Metric | Baseline (v2.2.0) | This PR (v2.3.0) | Change |
|--------|-------------------|------------------|--------|
| Accuracy | 0.92 | 0.94 | +2.2% |
| Latency (p50) | 1.4s | 1.2s | -14% |
| Token usage | 2,100 | 1,950 | -7% |
| Hallucination rate | 3.2% | 2.8% | -12% |
```

---

## 3. Documentation Practices

### Agent Behavior Documentation

**Model Card Template:**

```markdown
# Agent: Customer Support Bot v2.3

## Purpose
Handles tier-1 customer inquiries including order status,
return requests, and FAQ responses.

## Capabilities
- Order lookup via CRM API
- Return initiation (up to $100)
- FAQ responses from knowledge base
- Escalation to human agents

## Limitations
- Cannot process refunds over $100
- No access to payment information
- Cannot modify account settings
- May struggle with:
  - Highly technical questions
  - Multi-language conversations
  - Complex multi-order scenarios

## Known Failure Modes
- Hallucination rate: 2.8% on edge cases
- Confusion with similar product names
- Occasional incorrect date formatting

## Performance Characteristics
- Latency p50: 1.2s, p99: 3.4s
- Accuracy on test set: 94%
- Token cost per conversation: ~$0.02

## Fairness Considerations
- Tested across demographic groups
- No significant accuracy variance detected
- Ongoing monitoring for bias
```

### Tool Documentation

**Tool Documentation Template:**

```markdown
# Tool: CRM Order Lookup

## Purpose
Retrieves order details from the customer relationship
management system.

## Interface
```python
def lookup_order(
    order_id: str,           # Required: Order identifier
    include_history: bool = False  # Optional: Include order history
) -> OrderDetails:
```

## Inputs
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| order_id | string | Yes | Format: ORD-XXXXXXXX |
| include_history | boolean | No | Returns modification history |

## Outputs
```json
{
  "order_id": "ORD-12345678",
  "status": "shipped",
  "items": [...],
  "shipping_address": {...},
  "created_at": "2025-12-20T10:30:00Z"
}
```

## Side Effects
- Increments order view counter
- Logs access for audit trail
- May trigger cache refresh

## Error Handling
| Error Code | Meaning | Agent Action |
|------------|---------|--------------|
| 404 | Order not found | Ask customer to verify order ID |
| 403 | Access denied | Escalate to human agent |
| 429 | Rate limited | Wait and retry |
| 500 | System error | Apologize and escalate |

## Rate Limits
- 100 requests per minute
- 10,000 requests per day

## Performance
- Latency p50: 200ms
- Latency p99: 800ms
- Timeout: 5 seconds

## Idempotency
Read-only operation. Safe to retry.
```

### Production Runbooks

**Runbook Template:**

```markdown
# Runbook: Agent Performance Degradation

## Problem Statement
Agent task completion rate drops below 80% as measured
over a one-hour rolling window.

## Severity
- Level 2 (High) if affecting >10% of users
- Level 3 (Medium) if affecting <10% of users

## Detection
- Alert: "agent-completion-rate-low"
- Dashboard: ops.company.com/agent-health
- Metric: agent_task_completion_rate_1h

## Initial Diagnostics
1. Check recent deployments:
   ```bash
   git log --oneline -10 --since="2 hours ago"
   ```

2. Review error rates:
   ```bash
   kubectl logs -l app=agent --since=1h | grep ERROR
   ```

3. Check model provider status:
   - OpenAI: status.openai.com
   - Anthropic: status.anthropic.com

4. Verify data pipeline health:
   ```bash
   curl -s http://internal/health/data-pipeline
   ```

## Remediation Steps

### If recent deployment detected:
1. Initiate rollback:
   ```bash
   kubectl rollout undo deployment/agent
   ```
2. Monitor completion rate for 15 minutes
3. If recovered, investigate root cause

### If model provider issues:
1. Switch to backup model if available
2. Enable degraded mode with human fallback
3. Notify stakeholders

### If data pipeline issues:
1. Restart affected services
2. Clear caches if stale data suspected
3. Verify data freshness

## Escalation
- After 30 minutes without resolution: Page on-call lead
- If affecting >50% of users: Declare incident
- If security-related: Immediately escalate to security team

## Communication Templates

### Internal Slack:
"[INVESTIGATING] Agent completion rate below threshold.
Current: XX%. Investigating root cause. ETA for update: 15 min."

### Customer-facing (if prolonged):
"We're experiencing slower response times with our AI assistant.
Our team is working on a fix. Human agents are available."
```

### Incident Reports

**AI Agent Incident Report Template:**

```markdown
# Incident Report: Agent Hallucination Event

## Incident Summary
| Field | Value |
|-------|-------|
| Incident ID | INC-2025-1227-001 |
| Severity | Level 2 (High) |
| Duration | 2 hours 15 minutes |
| Users Affected | ~1,200 |
| Detection Time | 2025-12-27 14:30 UTC |
| Resolution Time | 2025-12-27 16:45 UTC |

## Classification
- [ ] Technical failure
- [x] AI behavior issue (hallucination)
- [ ] Data quality issue
- [ ] Policy violation
- [ ] Security incident

## Timeline
| Time (UTC) | Event |
|------------|-------|
| 14:30 | Alert triggered: hallucination rate spike |
| 14:35 | On-call engineer acknowledged |
| 14:45 | Root cause identified: corrupted context |
| 15:00 | Temporary fix deployed (context refresh) |
| 15:30 | Permanent fix prepared |
| 16:30 | Permanent fix deployed |
| 16:45 | Metrics returned to normal |

## Root Cause Analysis
The agent's context cache contained stale product information
following a failed database sync at 14:00 UTC. The agent
generated responses based on outdated product details,
resulting in factually incorrect information.

## Impact Assessment
- ~200 customers received incorrect product information
- 3 customers placed orders based on wrong pricing
- Estimated remediation cost: $1,500

## Corrective Actions
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| Add cache freshness check | @engineer | 2025-12-30 | In Progress |
| Implement context validation | @engineer | 2025-01-03 | Planned |
| Add hallucination detection | @ml-team | 2025-01-10 | Planned |
| Update runbook | @ops | 2025-12-28 | Complete |

## Lessons Learned
1. Need automated cache validity checking
2. Hallucination detection should trigger alerts
3. Consider redundant context sources
```

### Decision Logs (ADRs)

**Architecture Decision Record Template:**

```markdown
# ADR-007: Agent Autonomy Level

## Status
Accepted

## Context
We need to decide how much autonomy to grant our customer
service agent for processing refunds.

## Decision Drivers
- Customer experience (faster resolution)
- Risk management (financial exposure)
- Operational efficiency
- Regulatory requirements

## Options Considered

### Option A: Full Autonomy
Agent processes all refunds automatically.
- Pro: Fastest customer experience
- Con: High financial risk, compliance concerns

### Option B: Tiered Autonomy (Selected)
Agent auto-approves refunds under $100, escalates larger amounts.
- Pro: Balance of speed and safety
- Con: Some customers wait for human approval

### Option C: Human Approval Required
All refunds require human approval.
- Pro: Maximum control
- Con: Defeats automation benefits

## Decision
Implement tiered autonomy with $100 threshold.

## Consequences

### Positive
- 85% of refunds processed automatically
- Average resolution time reduced by 60%
- Human agents focus on complex cases

### Negative
- Need to maintain two code paths
- Threshold requires periodic review
- Some edge cases may need manual handling

### Risks
- Threshold may need adjustment based on fraud patterns
- Must monitor for gaming behavior

## Technical Implementation
- Add `approval_required: true` for refunds > $100
- Implement human-in-the-loop workflow
- Add audit logging for all refund actions

## Review Date
2026-03-27 (quarterly review)
```

---

## 4. Open-Source Contribution

### Contributing to Frameworks

**Major AI Agent Frameworks:**

| Framework | Focus | Contribution Process |
|-----------|-------|---------------------|
| **LangChain** | LLM integration | GitHub issues, PRs with tests |
| **LangGraph** | Multi-agent workflows | Style guides, documentation |
| **CrewAI** | Agent orchestration | Tool contributions welcome |
| **AutoGen** | Conversational agents | RFC process for features |

**Contribution Pipeline:**

1. **Identify opportunity:**
   - Search existing issues before creating new ones
   - Look for "good first issue" or "help wanted" labels
   - Discuss features before implementing

2. **Fork and branch:**
   - Create feature branch from main
   - Keep changes minimal and focused

3. **Implement:**
   - Write failing tests first
   - Follow project code style
   - Update documentation

4. **Submit PR:**
   - Reference issue with "Fixes #123"
   - Explain what and why
   - Respond to feedback promptly

**LangChain-Specific Guidelines:**
- Include minimal reproducible examples for bugs
- Document reasoning for feature requests
- CI checks must pass before review

### Building Community Tools

**High-Impact Contribution Areas:**

| Area | Examples | Impact Level |
|------|----------|--------------|
| **Integrations** | Model adapters, tool connectors | High |
| **Observability** | Debugging, tracing tools | High |
| **Documentation** | Tutorials, how-to guides | Very High |
| **Domain tools** | Industry-specific capabilities | Medium-High |

**Tool Documentation Requirements:**
- Clear purpose statement
- Input/output specifications
- Installation instructions
- Usage examples
- Known limitations

### Sharing Patterns

**Blog Post Best Practices:**

1. **Structure:**
   - Engaging problem statement
   - Step-by-step implementation
   - Working code examples
   - Synthesis and next steps

2. **Technical depth:**
   - Assume relevant background
   - Explain framework-specific concepts
   - Provide complete, runnable examples

3. **Value proposition:**
   - Focus on solving real problems
   - Explain why, not just how
   - Share lessons learned

**Conference Talk Guidelines:**

- Go deep on specific topics rather than surveying
- Use live demonstrations when possible
- Balance technical depth with accessibility
- Tailor complexity to audience

### License Considerations

**Common Open-Source Licenses:**

| License | Type | Commercial Use | Derivatives |
|---------|------|----------------|-------------|
| **MIT** | Permissive | Yes | Any license |
| **Apache 2.0** | Permissive | Yes | Any license, patent grant |
| **GPL** | Copyleft | Yes | Must be GPL |
| **AGPL** | Strong Copyleft | Yes | Must be AGPL |

**AI-Specific Licenses:**
- **OpenRAIL:** Open with responsible use provisions
- **Llama 2:** Restricts building competing models
- **Creative ML:** For creative AI applications

**Recommendations:**

| Scenario | Recommended License |
|----------|---------------------|
| Maximum adoption | MIT or Apache 2.0 |
| Require attribution | Apache 2.0 |
| Keep derivatives open | GPL |
| AI models with use restrictions | OpenRAIL |

### Community Engagement

**Building Developer Communities:**

1. **Inclusive contribution pathways:**
   - Tag issues for newcomers
   - Provide clear contribution guidelines
   - Celebrate first contributions

2. **Recognition:**
   - Mention contributors in release notes
   - Maintain CONTRIBUTORS file
   - Consider formal recognition programs

3. **Communication channels:**
   - Active Discord/Slack for discussions
   - Regular office hours or meetups
   - Documentation for async participation

**Sustainable Funding Models:**

| Model | Description | Trade-offs |
|-------|-------------|------------|
| Corporate sponsorship | Companies fund development | Risk of influence |
| Foundation | Neutral governance | Administrative overhead |
| Dual licensing | Open + commercial options | Complexity |
| Consulting/services | Support revenue | Time away from development |

---

## 5. Communication Patterns

### Async Collaboration Across Timezones

**Documentation-Centered Collaboration:**

1. **Process documents:** Capture workflows step-by-step with rationale
2. **Decision logs:** Prevent repeated discussions of settled matters
3. **Meeting notes:** Make insights accessible for all timezones

**Effective Async Communication:**

```
Weak async message:
"Should I use the existing auth system?"

Strong async message:
"For the admin dashboard, I'm deciding between using our
existing auth system or building a separate one. The existing
system would be faster but doesn't support role-based
permissions out of the box. I'm leaning toward extending it
with a roles table. Does that align with how you see this
evolving?"
```

**Key Principles:**
- Front-load context in every message
- Write for someone not in your head all day
- Make decisions, communicate clearly, avoid blocking

**Daily Updates Template:**

```markdown
## Daily Update - [Date]

### Completed
- [Task 1]
- [Task 2]

### In Progress
- [Current work with status]

### Blocked
- [Blocker] - Need [specific input] from [person]

### Decisions Made
- [Decision]: [Reasoning]

### Tomorrow
- [Planned work]
```

### Decision Documentation (ADRs for Agents)

**When to Write an ADR:**
- Model selection decisions
- Autonomy level choices
- Tool integration architecture
- Guardrail implementations
- Data access patterns

**ADR Workflow:**

```
1. Identify decision need
2. Document context and constraints
3. List options with trade-offs
4. Make decision with rationale
5. Document consequences
6. Set review date
7. Archive when superseded
```

**ADR Storage:**

```
docs/decisions/
├── 001-model-selection.md
├── 002-agent-autonomy-level.md
├── 003-tool-integration-pattern.md
├── 004-guardrail-architecture.md
└── template.md
```

### Knowledge Base Maintenance

**Keeping Documentation Current:**

| Practice | Frequency | Owner |
|----------|-----------|-------|
| Review high-traffic docs | Weekly | Doc owner |
| Audit for stale content | Monthly | Team lead |
| Update after incidents | Immediately | On-call |
| Archive deprecated docs | Quarterly | All team |

**AI-Enhanced Knowledge Management:**
- Semantic search for conceptual queries
- Automated stale content detection
- Real-time knowledge assistance in tools
- Meeting summarization and insight extraction

**Content Freshness Indicators:**

```markdown
---
last_reviewed: 2025-12-27
review_frequency: monthly
owner: @team-member
status: current  # current | needs-review | deprecated
---
```

### Stakeholder Communication

**Explaining AI to Non-Technical Audiences:**

**Effective Analogies:**

| Concept | Analogy |
|---------|---------|
| Generative AI | A chef who can creatively combine ingredients based on recipes learned |
| AI limitations | Like an intern: efficient but not always accurate |
| LLM hallucinations | Filling in blanks like autocomplete, sometimes wrong |
| Training data | The library of books the AI has read |
| Context window | The AI's short-term memory capacity |

**Communication Principles:**
1. Focus on business value, not technical details
2. Use analogies and metaphors familiar to audience
3. Explain both capabilities and limitations
4. Provide visual aids when possible
5. Invite questions frequently

**Customer-Facing Messaging:**
- Frame AI as enhancement, not replacement
- Focus on benefits (faster, personalized)
- Be transparent about AI involvement
- Communicate early and often about changes

### Incident Communication

**Internal Communication Template:**

```
[SEVERITY] Agent incident - [Brief description]

Status: [INVESTIGATING | MITIGATING | RESOLVED]
Impact: [Who is affected and how]
Current actions: [What we're doing]
ETA for update: [Time]

Point of contact: @on-call-engineer
War room: #incident-channel
```

**External Communication (If Needed):**

```
We're experiencing [impact description] with our AI assistant.
Our team is actively working to resolve this.
[Alternative option if available].
We'll provide an update by [time].

We apologize for any inconvenience.
```

**Post-Incident Communication:**
- Share lessons learned internally
- Update stakeholders on preventive measures
- Be transparent about root cause (appropriately)
- Demonstrate commitment to improvement

---

## 6. Team Structure

### Roles in an AI Agent Team

**Core Roles:**

| Role | Responsibility | Key Skills |
|------|---------------|------------|
| **AI/ML Engineer** | Model development, training, optimization | Python, PyTorch, ML algorithms |
| **AI Product Manager** | Strategy, requirements, stakeholder management | LLM understanding, data fluency |
| **Agentic Engineer** | Agent orchestration, tool integration | Systems thinking, prompt design |
| **AI QA Engineer** | Testing, validation, quality gates | Test design, critical thinking |
| **Data Engineer** | Data pipelines, infrastructure | SQL, ETL, data modeling |

**Emerging Roles:**

| Role | Description |
|------|-------------|
| **Prompt Engineer** | Specialized in prompt design and optimization |
| **AI Safety Engineer** | Guardrails, red teaming, alignment |
| **AI Ops Engineer** | Production deployment, monitoring, reliability |
| **Context Engineer** | Optimizing information flow to models |

**Team Composition by Stage:**

| Stage | Team Size | Key Roles |
|-------|-----------|-----------|
| **Prototype** | 2-3 | AI Engineer, Product Manager |
| **MVP** | 4-6 | + QA, Data Engineer |
| **Production** | 6-10 | + DevOps, Safety Engineer |
| **Scale** | 10+ | + Specialized roles |

### Skill Requirements and Hiring

**Technical Skills:**

| Skill | Importance | How to Assess |
|-------|------------|---------------|
| Python proficiency | Essential | Coding challenge |
| ML frameworks (PyTorch, TensorFlow) | High | Project review |
| LLM APIs and patterns | High | Technical interview |
| Cloud platforms | Medium-High | Experience discussion |
| Data fluency | High | Case study |

**Core Competencies:**

| Competency | Why It Matters |
|------------|----------------|
| Learning agility | AI landscape changes rapidly |
| Pattern recognition | Identifying AI behavior patterns |
| Communication | Bridging technical and business |
| Critical thinking | Evaluating AI outputs |
| Collaboration | Cross-functional teamwork |

**Hiring Practices:**

1. **Prioritize practical experience:**
   - GitHub repositories showing quality code
   - Open-source contributions
   - Kaggle or similar competition performance

2. **Skills-based assessment:**
   - Hands-on coding challenges
   - Realistic AI development scenarios
   - System design discussions

3. **Remote work capability:**
   - Clear written communication
   - Self-direction and autonomy
   - Prior remote experience

4. **AI attitude assessment:**
   - Willingness to experiment
   - Curiosity about new tools
   - Healthy skepticism about AI outputs

### Training and Upskilling

**Training Program Levels:**

| Level | Content | Duration | Audience |
|-------|---------|----------|----------|
| **L1: Basics** | AI tools overview, basic usage | 2 hours | All developers |
| **L2: Intermediate** | Context management, modes, patterns | 4 hours | Regular users |
| **L3: Advanced** | Prompt engineering, meta-prompting | 8 hours | Power users |
| **L4: Admin** | Security, governance, cost management | 4 hours | Team leads |

**Onboarding Checklist:**

```markdown
## AI Tools Onboarding

### Day 1
- [ ] Access to approved AI tools
- [ ] Walk through CLAUDE.md / .mdc files
- [ ] Review team prompt library

### Week 1
- [ ] Shadow experienced AI user
- [ ] Complete L1 training
- [ ] First AI-assisted PR (with review)

### Week 2-4
- [ ] L2 training
- [ ] Independent AI usage with review
- [ ] Contribute to shared prompt library

### Month 2+
- [ ] L3 training
- [ ] Help onboard next team member
- [ ] Propose improvements to AI workflows
```

**Continuous Learning:**
- Allocate dedicated learning time
- Peer-to-peer knowledge sharing
- Internal communities of practice
- External conference participation

### On-Call Patterns for Agents

**AI-Specific On-Call Considerations:**

| Aspect | Traditional Software | AI Agents |
|--------|---------------------|-----------|
| Failure modes | Clear errors, outages | Subtle quality degradation |
| Detection | Binary (up/down) | Gradual (accuracy drift) |
| Diagnosis | Stack traces, logs | Model behavior analysis |
| Recovery | Restart, rollback | May need retraining |

**On-Call Rotation:**

```
Week 1: Primary - @engineer-a, Secondary - @engineer-b
Week 2: Primary - @engineer-b, Secondary - @engineer-c
Week 3: Primary - @engineer-c, Secondary - @engineer-a
```

**Escalation Protocol:**

| Time Elapsed | Action |
|--------------|--------|
| 0-15 min | On-call investigates |
| 15-30 min | Engage secondary if needed |
| 30-60 min | Escalate to team lead |
| 60+ min | Incident commander |

**AI-Specific Monitoring:**

| Metric | Alert Threshold | Response |
|--------|-----------------|----------|
| Task completion rate | < 80% | Investigate immediately |
| Latency p99 | > 5s | Check model provider |
| Hallucination rate | > 5% | Consider rollback |
| Error rate | > 2% | Check integrations |
| Cost per request | 2x baseline | Audit token usage |

**Weekly Fire Drills:**
Practice agent shutdown procedures regularly to build muscle memory and identify process gaps before real incidents occur.

---

## 7. Quick Reference

### Team Workflow Checklist

```markdown
## Daily
- [ ] Check agent health dashboard
- [ ] Review overnight alerts
- [ ] Stand-up: AI-related blockers

## Weekly
- [ ] AI tools sync (15 min)
- [ ] Update shared prompt library
- [ ] Review cost metrics

## Monthly
- [ ] Update CLAUDE.md / .mdc files
- [ ] Audit documentation freshness
- [ ] Training assessment

## Quarterly
- [ ] Review AI governance policy
- [ ] Update agent capabilities docs
- [ ] Assess team AI maturity
```

### Git Workflow Quick Reference

```
Prompt Versioning:
  Major.Minor.Patch (e.g., 2.3.1)
  - Major: Breaking changes
  - Minor: New features
  - Patch: Bug fixes

Configuration Files:
  agent-config.yaml - Agent settings
  prompts/ - Versioned prompts
  .env.example - Environment documentation

Branch Strategy:
  main - Production prompts
  staging - Pre-production validation
  feature/* - Development work
```

### Documentation Quick Reference

| Document Type | Purpose | Update Frequency |
|---------------|---------|------------------|
| Model Cards | Agent capabilities/limitations | Per version |
| Tool Docs | Input/output/side effects | Per change |
| Runbooks | Operational procedures | After incidents |
| Incident Reports | Post-mortem analysis | Per incident |
| ADRs | Decision rationale | Per decision |

### Communication Templates

**Async Update:**
```
[Context] + [Decision/Question] + [Reasoning] + [Ask]
```

**Incident Alert:**
```
[SEVERITY] [Brief description]
Status: [STATE]
Impact: [Who/What affected]
Next update: [Time]
```

**Stakeholder Update:**
```
- What we did
- Why it matters
- What's next
- Questions welcome
```

### Role Summary

| Role | Focus | Reports To |
|------|-------|-----------|
| AI/ML Engineer | Model development | Tech Lead |
| AI Product Manager | Strategy, requirements | Product Director |
| Agentic Engineer | Agent orchestration | Tech Lead |
| AI QA Engineer | Quality, testing | QA Lead |
| AI Safety Engineer | Guardrails, alignment | Security Lead |

---

## Related Documents

- [developer-productivity-guide.md](developer-productivity-guide.md) - Tool-specific best practices
- [agent-prompting-guide.md](agent-prompting-guide.md) - Detailed prompting techniques
- [patterns-and-antipatterns.md](patterns-and-antipatterns.md) - Common patterns and failures
- [security-essentials.md](security-essentials.md) - Security implementation

---

**Document Version:** 1.0
**Last Updated:** 2025-12-27
**Sources:** Martin Fowler (Thoughtworks), Anthropic Engineering, GitHub, McKinsey, IBM, industry research
