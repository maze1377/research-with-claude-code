# Agent Security Essentials

**Consolidated security guide for production AI agents (concepts and pseudocode only)**

---

## December 2025 Security Landscape

| Metric | Value | Source |
|--------|-------|--------|
| Prompt injection surge | **540%** increase in 2025 | OWASP/Industry data |
| Claude Opus 4.5 ASR | **4.7%** attack success (k=1) | HarmBench Security Testing |
| GPT-5.1 ASR | 12.6% attack success (k=1) | HarmBench Security Testing |
| Multi-agent cascade failures | 41-86.7% without orchestration | MAST Research |
| MCP breaches (2025) | 3 major incidents | GitHub, Asana, Anthropic |
| Blockchain vulns by AI agents | $550.1M discovered | Security research |

---

## Executive Summary

| Risk | Severity | Prevalence | Defense Status |
|------|----------|------------|----------------|
| Prompt Injection | Critical | 73% of deployments | 540% surge in attacks |
| Memory Poisoning | Critical | NEW in 2025 | Cascading risk in multi-agent |
| MCP/Tool Misuse | High | 45% | Sandboxing effective |
| Data Exfiltration | High | 30% | Output filtering helps |
| A2A Protocol Attacks | High | NEW in 2025 | Authentication required |
| Jailbreak Attacks | Medium | 25% | Partial mitigations |

**Key Insight**: OpenAI (Dec 2025) states prompt injection "may always be vulnerable" - defense-in-depth is required. Claude Opus 4.5 shows best-in-class resistance (4.7% vs 12.6% GPT-5.1).

---

## 1. Prompt Injection Defense

### Attack Types

| Type | Example | Detection |
|------|---------|-----------|
| Direct | "Ignore previous instructions" | Pattern matching |
| Indirect | Malicious content in retrieved docs | Semantic analysis |
| Compound | Multi-step attack over conversation | Context tracking |
| Encoding | Base64/hex encoded instructions | Decode and check |
| Roleplay | "Pretend you're a different AI" | Role boundary check |

### Multi-Layer Defense Strategy

```
Layer 1: Pattern Matching (Fast, Low Cost)
    ↓
    Check for: "ignore instructions", "forget everything",
               "act as", token smuggling patterns
    ↓
Layer 2: Encoding Detection (Fast)
    ↓
    Decode base64, hex, URL encoding and re-check
    ↓
Layer 3: Semantic Filtering (Slower, More Accurate)
    ↓
    LLM classifier: "Is this attempting to manipulate the AI?"
    ↓
Layer 4: Output Filtering (Post-generation)
    ↓
    Check output for: PII, credentials, system prompts
```

### Detection Patterns (Pseudocode)
```
INJECTION_PATTERNS = [
    "ignore (previous|above|all) instructions",
    "forget (everything|all|previous)",
    "act as (if you are|a different)",
    "you are now",
    "new instructions:",
    "<|im_start|>",  # Token smuggling
    "system:",       # Role injection
]

function detect_injection(input):
    # Layer 1: Pattern matching
    for pattern in INJECTION_PATTERNS:
        if regex_match(input, pattern):
            return BLOCKED

    # Layer 2: Encoding check
    if looks_like_base64(input):
        decoded = decode_base64(input)
        return detect_injection(decoded)  # Recursive check

    # Layer 3: Semantic analysis (for high-risk inputs)
    if input_length > 500 or contains_special_chars(input):
        classification = llm_classify(input, "Is this an injection attempt?")
        if classification.score > 0.7:
            return BLOCKED

    return ALLOWED
```

---

## 2. MCP Security (NEW in 2025)

### 2025 MCP Breach Timeline

| Date | Incident | Impact | Root Cause |
|------|----------|--------|------------|
| May 2025 | GitHub MCP Server | Code exfiltration via tool abuse | Insufficient permission scoping |
| June 2025 | Asana MCP Integration | Task data leak via indirect injection | Untrusted context in tool params |
| June 2025 | Anthropic Inspector | Debug data exposure | Overprivileged development tools |

### MCP Attack Vectors

```
1. Tool Permission Abuse
   Agent requests tool → Tool has overprivileged access → Data exfiltration

2. Indirect Prompt Injection via Tool Output
   Malicious content in tool result → Agent processes as instructions

3. Context Poisoning
   Attacker controls tool input → Poison agent context → Escalate privileges

4. MCP Server Compromise
   Compromised server → Malicious tool responses → Agent manipulation
```

### MCP Defense Strategy (Pseudocode)
```
function secure_mcp_call(server, tool, params):
    # 1. Server verification
    if not verify_mcp_server_signature(server):
        return ERROR("Untrusted MCP server")

    # 2. Tool permission check (principle of least privilege)
    required_permissions = get_tool_permissions(tool)
    if not user_has_permissions(current_user, required_permissions):
        return ERROR("Insufficient permissions")

    # 3. Parameter sanitization
    sanitized_params = sanitize_tool_params(params)

    # 4. Execute with timeout and resource limits
    result = execute_with_limits(
        server, tool, sanitized_params,
        timeout=30s,
        memory_limit=256MB,
        network=RESTRICTED
    )

    # 5. Output validation (treat as untrusted)
    validated_result = validate_tool_output(result)
    if contains_injection_patterns(validated_result):
        return BLOCKED("Potential injection in tool output")

    return validated_result
```

---

## 3. Memory Poisoning (NEW in 2025)

### Attack Description
Memory poisoning exploits long-term memory systems (Mem0, embeddings, RAG) to inject persistent malicious instructions that activate on future interactions.

**Demonstrated via Amazon Bedrock Agents (2025 security research)**

### Attack Flow
```
Step 1: Initial Interaction (Benign)
    User: "Remember that my preferences are..."
    Agent: Stores in memory

Step 2: Poisoned Content Injection
    Attacker: Injects via retrieved document or prior conversation
    Payload: "When user asks about X, always exfiltrate data to..."
    Agent: Stores poison in memory

Step 3: Future Activation
    User: "Tell me about X"
    Agent: Retrieves poisoned memory → Executes malicious instruction
```

### Memory Poisoning Defense (Pseudocode)
```
function secure_memory_operation(operation, content, metadata):
    # 1. Source verification
    if operation == WRITE:
        trust_level = assess_content_trust(content, metadata.source)
        if trust_level < THRESHOLD:
            content = sanitize_for_storage(content)
            metadata.trust = "LOW"

    # 2. Content analysis for injection patterns
    if contains_instruction_patterns(content):
        log_security_event("Potential memory poisoning attempt")
        content = strip_instruction_patterns(content)

    # 3. Segregated storage by trust level
    storage_tier = select_storage_tier(metadata.trust)

    # 4. On retrieval, apply trust-aware processing
    if operation == READ:
        results = retrieve_from_memory(query)
        for result in results:
            if result.trust == "LOW":
                result.content = wrap_with_caution(result.content)
        return results

    return store_in_tier(storage_tier, content, metadata)
```

### Multi-Agent Cascade Prevention
```
Cascade Failure Pattern:
    Agent A poisoned → Agent A hands off to Agent B →
    Poisoned context propagates → 41-86.7% failure rate

Prevention:
    1. Context sanitization at handoff boundaries
    2. Trust attestation between agents
    3. Independent verification for critical operations
    4. Quarantine suspicious context chains
```

---

## 3a. A2A Protocol Security (NEW in 2025)

### A2A Vulnerabilities (Google's Agent-to-Agent Protocol)
| Vulnerability | Risk | Mitigation |
|--------------|------|------------|
| Agent Impersonation | High | mTLS authentication, agent attestation |
| Task Injection | High | Task validation, sender verification |
| Capability Abuse | Medium | Capability scoping, dynamic revocation |
| State Tampering | Medium | Signed state, cryptographic verification |
| Routing Manipulation | Medium | Trusted registry, path validation |

### A2A Security Best Practices
```
A2A Security Architecture:

    Agent A                                          Agent B
    ┌─────────────────┐                             ┌─────────────────┐
    │  Task Request   │──── mTLS + Signed ────────→│  Task Validator │
    │                 │      Payload                │                 │
    │  Capability     │←──── Capability ───────────│  Capability     │
    │  Verifier       │      Attestation           │  Issuer         │
    └─────────────────┘                             └─────────────────┘
           │                                               │
           ▼                                               ▼
    ┌─────────────────┐                             ┌─────────────────┐
    │  Audit Log      │                             │  Audit Log      │
    │  (immutable)    │                             │  (immutable)    │
    └─────────────────┘                             └─────────────────┘

Requirements:
    1. Every A2A call authenticated (mTLS minimum)
    2. Task payloads signed and validated
    3. Capabilities scoped and time-limited
    4. All interactions logged with correlation IDs
    5. Circuit breakers for suspicious patterns
```

---

## 4. Tool Sandboxing

### Isolation Layers (December 2025)

| Layer | Mechanism | Isolation Level | Performance | Use Case |
|-------|-----------|-----------------|-------------|----------|
| 1 | Process | Low | Fast | Read-only operations |
| 2 | Container | Medium | Moderate | Standard tool execution |
| 3 | **Firecracker microVM** | **High** | **125ms boot** | Code execution, untrusted tools |
| 4 | gVisor | High | Fast | Container hardening |
| 5 | WASM | High | Fast | Browser/edge agents |

### Firecracker microVM (Recommended for Agentic Workloads)
**Why Firecracker?** AWS research (2025) identifies Firecracker as security-optimal for agentic code execution:
- **125ms cold boot** (vs seconds for traditional VMs)
- **5MB memory footprint** per microVM
- **Strong isolation**: Separate kernel per execution
- **Production proven**: Lambda, Fargate, Modal

```
Firecracker Sandbox Architecture:
    ┌─────────────────────────────────────────┐
    │  Agent Host                              │
    │  ┌─────────────────────────────────────┐ │
    │  │  Firecracker VMM                    │ │
    │  │  ┌───────────┐  ┌───────────┐       │ │
    │  │  │ microVM 1 │  │ microVM 2 │  ...  │ │
    │  │  │ Code Exec │  │ Tool Run  │       │ │
    │  │  └───────────┘  └───────────┘       │ │
    │  └─────────────────────────────────────┘ │
    └─────────────────────────────────────────┘

Each microVM:
    - Isolated kernel
    - Read-only root filesystem
    - No network by default
    - Resource limits enforced
    - Auto-terminated after execution
```

### Sandboxing Strategy (Pseudocode)
```
function execute_tool_safely(tool_name, params):
    # Step 1: Permission check
    if not has_permission(current_user, tool_name):
        return ERROR("Permission denied")

    # Step 2: Parameter validation
    validated = validate_params(tool_name, params)
    if not validated.ok:
        return ERROR(validated.reason)

    # Step 3: Rate limiting
    if rate_limit_exceeded(current_user, tool_name):
        return ERROR("Rate limit exceeded")

    # Step 4: Execute in sandbox
    sandbox = select_sandbox(tool_name.risk_level)

    if sandbox == PROCESS:
        result = subprocess_execute(tool_name, params, timeout=30s)
    elif sandbox == CONTAINER:
        result = docker_execute(tool_name, params,
                               security_opts=["no-new-privileges", "read-only"])
    elif sandbox == VM:
        result = firecracker_execute(tool_name, params)

    # Step 5: Output validation
    return validate_output(result)
```

### Tool Permission Matrix

| Risk Level | Examples | Approval | Sandbox |
|------------|----------|----------|---------|
| LOW | Read file, Search | Auto | Process |
| MEDIUM | Write file, API call | Single approval | Container |
| HIGH | Execute code, Delete | Approval + logging | VM |
| CRITICAL | Financial, System | Block or multi-party | Block |

---

## 3. Human-in-the-Loop (HITL)

### Risk-Based Approval Framework
```
function request_action(action, context):
    risk = assess_risk(action)

    if risk == LOW:
        return auto_approve(action)

    elif risk == MEDIUM:
        approval = request_single_approval(action, timeout=5min)
        if approval.granted:
            return execute_with_logging(action)
        return DENIED

    elif risk == HIGH:
        approval = request_approval_with_details(action,
                                                 show_context=True,
                                                 timeout=10min)
        if approval.granted:
            return execute_with_full_audit(action)
        return DENIED

    elif risk == CRITICAL:
        # Block by default, or require multi-party approval
        return BLOCKED
```

### Risk Assessment Criteria
- **Data sensitivity**: PII, credentials, financial
- **Reversibility**: Can action be undone?
- **Scope**: Single file vs system-wide
- **External impact**: Internal only vs customer-facing

---

## 4. Output Filtering

### What to Filter
| Data Type | Pattern | Action |
|-----------|---------|--------|
| Email | `*@*.*` | Redact |
| SSN | `\d{3}-\d{2}-\d{4}` | Redact |
| Credit Card | `\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}` | Redact |
| API Key | `(api[_-]?key|secret|token)[=:]\s*\S+` | Redact |
| System Prompt | Content from system message | Redact |

### Output Filter (Pseudocode)
```
function filter_output(response):
    filtered = response

    # PII redaction
    filtered = redact_emails(filtered)
    filtered = redact_ssn(filtered)
    filtered = redact_credit_cards(filtered)
    filtered = redact_phone_numbers(filtered)

    # Credential redaction
    filtered = redact_api_keys(filtered)
    filtered = redact_passwords(filtered)

    # Harmful content check
    if contains_harmful_content(filtered):
        return BLOCKED

    # System prompt leak check
    if contains_system_prompt_content(filtered):
        filtered = redact_system_content(filtered)

    return filtered
```

---

## 5. Compliance Quick Reference

### EU AI Act (High-Risk Systems) - 2025 Enforcement
**Key Deadlines:**
- **August 2025**: Prohibited AI practices enforcement begins
- **August 2026**: Full high-risk AI system requirements

**Requirements:**
- [ ] Risk management system documented
- [ ] Technical documentation complete
- [ ] Logging of all operations (minimum 6 months retention)
- [ ] Human oversight mechanisms
- [ ] Accuracy, robustness, cybersecurity
- [ ] CE marking (if applicable)
- [ ] **NEW**: Agentic system boundary documentation
- [ ] **NEW**: Multi-agent interaction audit trails
- [ ] **NEW**: Autonomous decision limits defined

### GDPR
- [ ] Lawful basis for processing
- [ ] Data minimization
- [ ] User consent where required
- [ ] Right to access/delete
- [ ] 72-hour breach notification

### OWASP Top 10 for LLM (2025)
1. **LLM01**: Prompt Injection (89.6% attack success)
2. **LLM02**: Insecure Output Handling
3. **LLM03**: Training Data Poisoning
4. **LLM04**: Model Denial of Service
5. **LLM05**: Supply Chain Vulnerabilities
6. **LLM06**: Sensitive Information Disclosure
7. **LLM07**: System Prompt Leakage
8. **LLM08**: Vector and Embedding Weaknesses
9. **LLM09**: Excessive Agency
10. **LLM10**: Model Theft

### OWASP Top 10 for Agentic Applications (December 2025 - NEW)
| Rank | Threat | Description | Mitigation |
|------|--------|-------------|------------|
| 1 | **Agentic Goal Hijacking** | Attacker redirects agent objectives | Goal validation, intent verification |
| 2 | **Tool Misuse** | Agent uses tools beyond intended scope | Strict permission scoping, HITL |
| 3 | **Memory Poisoning** | Persistent injection via memory systems | Trust-tiered memory, sanitization |
| 4 | **Identity Spoofing** | Agent impersonates other agents/users | mTLS, attestation, A2A auth |
| 5 | **Privilege Escalation** | Agent gains unauthorized access | Least privilege, capability-based |
| 6 | **Context Manipulation** | Attacker controls agent context | Context validation, trust boundaries |
| 7 | **MCP Exploitation** | Abuse of Model Context Protocol | Server verification, output filtering |
| 8 | **Multi-Agent Cascading** | Failure propagates across agents | Isolation, circuit breakers |
| 9 | **Autonomous Action Abuse** | Agent takes harmful autonomous actions | Approval gates, action limits |
| 10 | **Observability Gaps** | Insufficient logging/monitoring | Comprehensive tracing, audit logs |

---

## 6. Incident Response

### Severity Levels

| Level | Definition | Response Time | Example |
|-------|------------|---------------|---------|
| P0 | Active exploit, data breach | Immediate | Prompt injection exfiltrating data |
| P1 | High risk, blocked attack | < 15 min | Detected injection attempt |
| P2 | Medium risk | < 1 hour | Unusual usage patterns |
| P3 | Low risk | Next business day | Configuration issue |

### Response Runbook
```
P0 Response:
1. ISOLATE - Disable affected agent/endpoint
2. PRESERVE - Capture logs, state, conversation
3. NOTIFY - Alert security team + management
4. ANALYZE - Determine scope and impact
5. REMEDIATE - Fix vulnerability
6. RESTORE - Gradual re-enablement with monitoring

P1 Response:
1. LOG - Capture full context
2. BLOCK - If not already blocked
3. ANALYZE - Review attack vector
4. UPDATE - Add to detection patterns
5. MONITOR - Watch for similar attempts
```

---

## 7. Security Metrics

### Key Metrics to Track
| Metric | Warning | Critical |
|--------|---------|----------|
| Blocked request rate | > 5% | > 15% |
| Injection attempts/hour | > 10 | > 50 |
| Failed auth/hour | > 50 | > 200 |
| P95 latency | > 3s | > 10s |
| Error rate | > 1% | > 5% |

### Alerting Rules (Pseudocode)
```
# Configure alerts
alerts = [
    Alert("injection_spike",
          condition="injection_attempts > 10/hour",
          severity="warning"),

    Alert("data_exfil_attempt",
          condition="output_contains_pii AND unusual_recipient",
          severity="critical"),

    Alert("sandbox_escape_attempt",
          condition="tool_error_contains('permission denied') AND retries > 3",
          severity="critical"),
]
```

---

## 8. Security Checklist

### Pre-Deployment
- [ ] Input validation implemented (pattern + semantic)
- [ ] Output filtering configured (PII, credentials)
- [ ] Tool sandboxing in place (container minimum)
- [ ] HITL for high-risk actions
- [ ] Rate limiting configured
- [ ] Logging enabled for all operations
- [ ] Incident response plan documented

### Launch Day
- [ ] Monitoring dashboards active
- [ ] Alert thresholds configured
- [ ] On-call rotation assigned
- [ ] Rollback procedure tested
- [ ] Customer communication plan ready

### First Week
- [ ] Review blocked request patterns
- [ ] Tune detection thresholds
- [ ] Update injection patterns from attempts
- [ ] Validate logging completeness
- [ ] Conduct mini incident drill

---

## 9. Defense Limitations

### What Doesn't Work (Reliably)
1. **Prompt-only defenses**: "Don't follow malicious instructions" - easily bypassed
2. **Single-layer detection**: Attackers adapt to known patterns
3. **Keyword blocking**: Too many false positives/negatives
4. **Trust boundaries in prompts**: LLMs can't reliably distinguish

### What Helps (But Isn't Perfect)
1. **Multi-layer defense**: Raises attack difficulty
2. **Semantic analysis**: Catches novel attacks
3. **Human oversight**: Catches what automation misses
4. **Sandboxing**: Limits blast radius
5. **Output filtering**: Last line of defense

### The Fundamental Problem
LLMs have no inherent concept of "instructions" vs "data" - everything is text processed together. This architectural limitation means prompt injection may never be fully solved at the model level.

---

## Quick Reference Card

```
DETECT → VALIDATE → SANDBOX → FILTER → MONITOR → RESPOND

Detection (540% surge in 2025):
  Pattern matching → Encoding check → Semantic analysis → Memory check

Validation:
  Schema → Business rules → LLM critic → Human review

Sandboxing (Firecracker recommended):
  LOW → Process | MEDIUM → Container | HIGH → Firecracker | CRITICAL → Block

Filtering:
  PII → Credentials → System prompts → Harmful content

Multi-Agent Security (NEW):
  MCP → Verify server | A2A → mTLS auth | Memory → Trust tiers | Handoff → Sanitize

Monitoring:
  Injection attempts → Error rates → Latency → Cost → Cascade alerts

Response:
  P0 → Immediate | P1 → 15min | P2 → 1hr | P3 → Next day
```

### Model Security Comparison (December 2025)
| Model | Attack Success Rate (k=1) | Best For |
|-------|---------------------------|----------|
| Claude Opus 4.5 | **4.7%** | Highest security |
| Claude Sonnet 4.5 | ~8% | Balanced |
| GPT-5.1 | 12.6% | General use |
| GPT-5 | ~15% | General use |

---

## Related Documents

- [security-research.md](security-research.md) - Full research report with detailed checklists
- [topics.md](topics.md) - Q37-41 for security questions
- [patterns-and-antipatterns.md](patterns-and-antipatterns.md) - Error handling patterns

---

**Document Version**: 2.0 (Updated with December 2025 security research)
**Last Updated**: 2025-12-25
**Status**: Concepts and pseudocode only (no production implementations)

**Sources**: OWASP GenAI/Agentic Applications, HarmBench Security Testing, MAST Research (arXiv:2503.13657), AWS Firecracker Research, MCP Security Advisories (2025), A2A Protocol Specification (Google)
