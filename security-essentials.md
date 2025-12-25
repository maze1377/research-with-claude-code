# Agent Security Essentials

**Consolidated security guide for production AI agents (concepts and pseudocode only)**

---

## Executive Summary

| Risk | Severity | Prevalence | Defense Status |
|------|----------|------------|----------------|
| Prompt Injection | Critical | 73% of deployments | No reliable defense |
| Tool Misuse | High | 45% | Sandboxing effective |
| Data Exfiltration | High | 30% | Output filtering helps |
| Jailbreak Attacks | Medium | 25% | Partial mitigations |

**Key Insight**: OpenAI (Dec 2025) states prompt injection "may always be vulnerable" - defense-in-depth is required.

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

## 2. Tool Sandboxing

### Isolation Layers

| Layer | Mechanism | Isolation Level | Performance |
|-------|-----------|-----------------|-------------|
| 1 | Process | Low | Fast |
| 2 | Container | Medium | Moderate |
| 3 | VM/Firecracker | High | Slower |
| 4 | WASM | High | Fast |

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

### EU AI Act (High-Risk Systems)
- [ ] Risk management system documented
- [ ] Technical documentation complete
- [ ] Logging of all operations
- [ ] Human oversight mechanisms
- [ ] Accuracy, robustness, cybersecurity
- [ ] CE marking (if applicable)

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

Detection:
  Pattern matching → Encoding check → Semantic analysis

Validation:
  Schema → Business rules → LLM critic → Human review

Sandboxing:
  LOW → Process | MEDIUM → Container | HIGH → VM | CRITICAL → Block

Filtering:
  PII → Credentials → System prompts → Harmful content

Monitoring:
  Injection attempts → Error rates → Latency → Cost

Response:
  P0 → Immediate | P1 → 15min | P2 → 1hr | P3 → Next day
```

---

## Related Documents

- [security-research.md](security-research.md) - Full research report with detailed checklists
- [topics.md](topics.md) - Q37-41 for security questions
- [patterns-and-antipatterns.md](patterns-and-antipatterns.md) - Error handling patterns

---

**Document Version**: 1.0 (Consolidated from agent-safety-*.md)
**Last Updated**: December 2025
**Status**: Concepts and pseudocode only (no production implementations)
