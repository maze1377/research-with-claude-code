# AI Governance and Compliance Guide

**Last Updated:** December 2025
**Difficulty:** Intermediate to Advanced

This guide covers governance frameworks, regulatory compliance, and enterprise considerations for deploying AI agent systems in production environments.

---

## Table of Contents

1. [Governance Frameworks](#governance-frameworks)
2. [Regulatory Landscape](#regulatory-landscape)
3. [Data Privacy Compliance](#data-privacy-compliance)
4. [Audit and Logging](#audit-and-logging)
5. [Access Control](#access-control)
6. [Model Risk Management](#model-risk-management)
7. [Documentation Requirements](#documentation-requirements)

---

## Governance Frameworks

### AI Governance Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Executive Oversight                          │
│                    (Board/C-Suite)                              │
├─────────────────────────────────────────────────────────────────┤
│                    AI Ethics Committee                          │
│            (Policy, Standards, Risk Assessment)                 │
├───────────────────┬───────────────────┬─────────────────────────┤
│  Model Governance │  Data Governance  │  Operations Governance  │
│  - Model registry │  - Data catalog   │  - Deployment policies  │
│  - Version control│  - Privacy rules  │  - Monitoring standards │
│  - Bias testing   │  - Retention      │  - Incident response    │
└───────────────────┴───────────────────┴─────────────────────────┘
```

### Governance Policies

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class UseCase(Enum):
    CUSTOMER_SERVICE = "customer_service"
    CONTENT_GENERATION = "content_generation"
    CODE_GENERATION = "code_generation"
    DECISION_SUPPORT = "decision_support"
    AUTONOMOUS_ACTION = "autonomous_action"

@dataclass
class GovernancePolicy:
    """Define governance requirements for AI use cases."""
    use_case: UseCase
    risk_level: RiskLevel
    requires_human_approval: bool
    max_autonomy_level: int  # 1-5 scale
    data_sensitivity_allowed: List[str]
    required_reviews: List[str]
    monitoring_level: str
    retention_days: int

GOVERNANCE_POLICIES = {
    UseCase.CUSTOMER_SERVICE: GovernancePolicy(
        use_case=UseCase.CUSTOMER_SERVICE,
        risk_level=RiskLevel.MEDIUM,
        requires_human_approval=False,
        max_autonomy_level=3,
        data_sensitivity_allowed=["public", "internal", "confidential"],
        required_reviews=["security", "privacy"],
        monitoring_level="standard",
        retention_days=90
    ),
    UseCase.DECISION_SUPPORT: GovernancePolicy(
        use_case=UseCase.DECISION_SUPPORT,
        risk_level=RiskLevel.HIGH,
        requires_human_approval=True,
        max_autonomy_level=2,
        data_sensitivity_allowed=["public", "internal"],
        required_reviews=["security", "privacy", "ethics", "legal"],
        monitoring_level="enhanced",
        retention_days=365
    ),
    UseCase.AUTONOMOUS_ACTION: GovernancePolicy(
        use_case=UseCase.AUTONOMOUS_ACTION,
        risk_level=RiskLevel.CRITICAL,
        requires_human_approval=True,
        max_autonomy_level=4,
        data_sensitivity_allowed=["public"],
        required_reviews=["security", "privacy", "ethics", "legal", "executive"],
        monitoring_level="comprehensive",
        retention_days=730
    )
}

class GovernanceManager:
    """Manage AI governance policies and approvals."""

    def __init__(self, policies: Dict[UseCase, GovernancePolicy]):
        self.policies = policies
        self.approvals: Dict[str, List[Dict]] = {}

    def get_requirements(self, use_case: UseCase) -> GovernancePolicy:
        """Get governance requirements for a use case."""
        return self.policies.get(use_case)

    def request_approval(self, agent_id: str, use_case: UseCase,
                        justification: str) -> Dict:
        """Request approval for agent deployment."""
        policy = self.get_requirements(use_case)

        approval_request = {
            "agent_id": agent_id,
            "use_case": use_case.value,
            "risk_level": policy.risk_level.value,
            "required_reviews": policy.required_reviews,
            "pending_reviews": policy.required_reviews.copy(),
            "completed_reviews": [],
            "status": "pending",
            "requested_at": datetime.utcnow().isoformat(),
            "justification": justification
        }

        self.approvals[agent_id] = self.approvals.get(agent_id, [])
        self.approvals[agent_id].append(approval_request)

        return approval_request

    def submit_review(self, agent_id: str, reviewer: str,
                     review_type: str, approved: bool, notes: str):
        """Submit a governance review."""
        if agent_id not in self.approvals:
            raise ValueError(f"No approval request for {agent_id}")

        request = self.approvals[agent_id][-1]

        if review_type not in request["pending_reviews"]:
            raise ValueError(f"Review {review_type} not required or already completed")

        request["pending_reviews"].remove(review_type)
        request["completed_reviews"].append({
            "type": review_type,
            "reviewer": reviewer,
            "approved": approved,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Update status
        if not approved:
            request["status"] = "rejected"
        elif not request["pending_reviews"]:
            request["status"] = "approved"

        return request

    def is_approved(self, agent_id: str) -> bool:
        """Check if agent is approved for deployment."""
        if agent_id not in self.approvals:
            return False
        return self.approvals[agent_id][-1]["status"] == "approved"
```

### Risk Assessment Framework

```python
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

class RiskCategory(Enum):
    SAFETY = "safety"
    PRIVACY = "privacy"
    SECURITY = "security"
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"

@dataclass
class RiskFactor:
    category: RiskCategory
    description: str
    likelihood: float  # 0-1
    impact: float      # 0-1
    mitigations: List[str]

    @property
    def risk_score(self) -> float:
        return self.likelihood * self.impact

@dataclass
class RiskAssessment:
    agent_id: str
    assessor: str
    assessment_date: str
    factors: List[RiskFactor]
    overall_risk: RiskLevel
    recommendations: List[str]
    approval_required: bool

class RiskAssessor:
    """Assess risks for AI agent deployments."""

    RISK_THRESHOLDS = {
        RiskLevel.LOW: 0.2,
        RiskLevel.MEDIUM: 0.4,
        RiskLevel.HIGH: 0.7,
        RiskLevel.CRITICAL: 1.0
    }

    def assess(self, agent_config: Dict[str, Any]) -> RiskAssessment:
        """Perform risk assessment for an agent."""
        factors = []

        # Safety risks
        factors.append(self._assess_safety(agent_config))

        # Privacy risks
        factors.append(self._assess_privacy(agent_config))

        # Security risks
        factors.append(self._assess_security(agent_config))

        # Fairness risks
        factors.append(self._assess_fairness(agent_config))

        # Calculate overall risk
        max_score = max(f.risk_score for f in factors)
        overall_risk = self._score_to_level(max_score)

        return RiskAssessment(
            agent_id=agent_config["id"],
            assessor="automated",
            assessment_date=datetime.utcnow().isoformat(),
            factors=factors,
            overall_risk=overall_risk,
            recommendations=self._generate_recommendations(factors),
            approval_required=overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        )

    def _assess_safety(self, config: Dict) -> RiskFactor:
        """Assess safety-related risks."""
        mitigations = []
        likelihood = 0.1
        impact = 0.5

        # Check for autonomous actions
        if config.get("can_execute_actions", False):
            likelihood += 0.3
            impact += 0.2
            mitigations.append("Implement human-in-the-loop for high-risk actions")

        # Check for external tool access
        if config.get("external_tools", []):
            likelihood += 0.2
            mitigations.append("Sandbox external tool execution")

        return RiskFactor(
            category=RiskCategory.SAFETY,
            description="Risk of unintended harmful actions",
            likelihood=min(likelihood, 1.0),
            impact=min(impact, 1.0),
            mitigations=mitigations
        )

    def _assess_privacy(self, config: Dict) -> RiskFactor:
        """Assess privacy-related risks."""
        mitigations = []
        likelihood = 0.2
        impact = 0.6

        # Check data access
        data_access = config.get("data_access", [])
        if "pii" in data_access:
            likelihood += 0.4
            impact += 0.2
            mitigations.append("Implement PII detection and masking")

        if "phi" in data_access:
            likelihood += 0.3
            impact += 0.3
            mitigations.append("Apply HIPAA-compliant data handling")

        return RiskFactor(
            category=RiskCategory.PRIVACY,
            description="Risk of privacy violations",
            likelihood=min(likelihood, 1.0),
            impact=min(impact, 1.0),
            mitigations=mitigations
        )

    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert risk score to level."""
        for level, threshold in sorted(
            self.RISK_THRESHOLDS.items(),
            key=lambda x: x[1]
        ):
            if score <= threshold:
                return level
        return RiskLevel.CRITICAL

    def _generate_recommendations(self, factors: List[RiskFactor]) -> List[str]:
        """Generate recommendations based on risk factors."""
        recommendations = []
        for factor in factors:
            if factor.risk_score > 0.5:
                recommendations.extend(factor.mitigations)
        return list(set(recommendations))
```

---

## Regulatory Landscape

### Key Regulations

| Regulation | Scope | Key Requirements | Agent Impact |
|------------|-------|------------------|--------------|
| **EU AI Act** | EU + global companies | Risk classification, transparency, human oversight | Must classify agents by risk level |
| **GDPR** | EU personal data | Data minimization, consent, right to explanation | Agent decisions must be explainable |
| **CCPA/CPRA** | California residents | Data access, deletion, opt-out | Must support data subject requests |
| **HIPAA** | US healthcare data | PHI protection, audit trails | Strict logging for health agents |
| **SOC 2** | Service providers | Security controls, availability | Compliance attestation needed |
| **PCI DSS** | Payment data | Data protection, access control | Cardholder data isolation |

### EU AI Act Compliance

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class EUAIActRiskCategory(Enum):
    UNACCEPTABLE = "unacceptable"  # Banned
    HIGH = "high"                   # Strict requirements
    LIMITED = "limited"             # Transparency obligations
    MINIMAL = "minimal"             # No specific requirements

@dataclass
class AISystemClassification:
    """EU AI Act classification for an AI system."""
    risk_category: EUAIActRiskCategory
    use_case: str
    requirements: List[str]
    documentation_needed: List[str]
    human_oversight_required: bool
    transparency_obligations: List[str]

class EUAIActClassifier:
    """Classify AI systems under EU AI Act."""

    HIGH_RISK_USE_CASES = [
        "biometric_identification",
        "critical_infrastructure",
        "education_scoring",
        "employment_decisions",
        "credit_scoring",
        "law_enforcement",
        "migration_control",
        "justice_administration"
    ]

    UNACCEPTABLE_USE_CASES = [
        "social_scoring",
        "real_time_biometric_public_spaces",
        "subliminal_manipulation",
        "exploitation_vulnerabilities"
    ]

    def classify(self, use_case: str, capabilities: List[str]) -> AISystemClassification:
        """Classify AI system under EU AI Act."""

        # Check for unacceptable risk
        if use_case in self.UNACCEPTABLE_USE_CASES:
            return AISystemClassification(
                risk_category=EUAIActRiskCategory.UNACCEPTABLE,
                use_case=use_case,
                requirements=["PROHIBITED - Cannot deploy in EU"],
                documentation_needed=[],
                human_oversight_required=True,
                transparency_obligations=[]
            )

        # Check for high risk
        if use_case in self.HIGH_RISK_USE_CASES:
            return AISystemClassification(
                risk_category=EUAIActRiskCategory.HIGH,
                use_case=use_case,
                requirements=[
                    "Risk management system",
                    "Data governance",
                    "Technical documentation",
                    "Record keeping",
                    "Transparency to users",
                    "Human oversight measures",
                    "Accuracy and robustness",
                    "Cybersecurity"
                ],
                documentation_needed=[
                    "Conformity assessment",
                    "Technical documentation",
                    "Quality management system",
                    "Risk assessment",
                    "Training data documentation"
                ],
                human_oversight_required=True,
                transparency_obligations=[
                    "Inform users of AI interaction",
                    "Explain decision-making process",
                    "Provide human oversight mechanism"
                ]
            )

        # Check for limited risk (chatbots, emotion recognition, etc.)
        if "conversational" in capabilities or "emotion_detection" in capabilities:
            return AISystemClassification(
                risk_category=EUAIActRiskCategory.LIMITED,
                use_case=use_case,
                requirements=["Transparency obligations only"],
                documentation_needed=["Basic system description"],
                human_oversight_required=False,
                transparency_obligations=[
                    "Inform users they are interacting with AI",
                    "Label AI-generated content"
                ]
            )

        # Minimal risk
        return AISystemClassification(
            risk_category=EUAIActRiskCategory.MINIMAL,
            use_case=use_case,
            requirements=["Voluntary codes of conduct"],
            documentation_needed=[],
            human_oversight_required=False,
            transparency_obligations=[]
        )


class EUAIActComplianceChecker:
    """Check compliance with EU AI Act requirements."""

    def __init__(self, classification: AISystemClassification):
        self.classification = classification

    def check_compliance(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if system meets EU AI Act requirements."""
        if self.classification.risk_category == EUAIActRiskCategory.UNACCEPTABLE:
            return {
                "compliant": False,
                "reason": "Unacceptable risk - prohibited use case",
                "violations": ["System use case is banned under EU AI Act"]
            }

        violations = []
        warnings = []

        # Check high-risk requirements
        if self.classification.risk_category == EUAIActRiskCategory.HIGH:
            # Risk management
            if not system_config.get("risk_management_system"):
                violations.append("Missing risk management system")

            # Human oversight
            if not system_config.get("human_oversight_enabled"):
                violations.append("Human oversight not implemented")

            # Technical documentation
            if not system_config.get("technical_documentation"):
                violations.append("Missing technical documentation")

            # Data governance
            if not system_config.get("data_governance"):
                violations.append("Missing data governance framework")

        # Check transparency obligations
        for obligation in self.classification.transparency_obligations:
            if not self._check_transparency(system_config, obligation):
                warnings.append(f"Transparency gap: {obligation}")

        return {
            "compliant": len(violations) == 0,
            "risk_category": self.classification.risk_category.value,
            "violations": violations,
            "warnings": warnings,
            "recommendations": self._generate_recommendations(violations, warnings)
        }

    def _check_transparency(self, config: Dict, obligation: str) -> bool:
        """Check if transparency obligation is met."""
        if "inform users" in obligation.lower():
            return config.get("ai_disclosure_enabled", False)
        if "label" in obligation.lower():
            return config.get("content_labeling_enabled", False)
        return True

    def _generate_recommendations(self, violations: List[str],
                                  warnings: List[str]) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []

        if violations:
            recommendations.append("Address all violations before deployment")

        if warnings:
            recommendations.append("Review and address transparency gaps")

        return recommendations
```

---

## Data Privacy Compliance

### GDPR Implementation

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum
import hashlib

class LegalBasis(Enum):
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"

@dataclass
class DataProcessingRecord:
    """Record of Processing Activities (ROPA) entry."""
    purpose: str
    legal_basis: LegalBasis
    data_categories: List[str]
    data_subjects: List[str]
    recipients: List[str]
    retention_period_days: int
    security_measures: List[str]
    third_country_transfers: bool
    automated_decision_making: bool

class GDPRComplianceManager:
    """Manage GDPR compliance for AI agent systems."""

    def __init__(self):
        self.processing_records: List[DataProcessingRecord] = []
        self.consent_records: Dict[str, Dict] = {}
        self.data_subject_requests: List[Dict] = []

    def register_processing(self, record: DataProcessingRecord):
        """Register a data processing activity."""
        self.processing_records.append(record)

    def record_consent(self, subject_id: str, purposes: List[str],
                      timestamp: datetime = None):
        """Record data subject consent."""
        self.consent_records[subject_id] = {
            "purposes": purposes,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "withdrawn": False
        }

    def withdraw_consent(self, subject_id: str):
        """Withdraw consent for a data subject."""
        if subject_id in self.consent_records:
            self.consent_records[subject_id]["withdrawn"] = True
            self.consent_records[subject_id]["withdrawn_at"] = datetime.utcnow().isoformat()

    def has_valid_consent(self, subject_id: str, purpose: str) -> bool:
        """Check if valid consent exists for a purpose."""
        if subject_id not in self.consent_records:
            return False

        record = self.consent_records[subject_id]
        if record["withdrawn"]:
            return False

        return purpose in record["purposes"]

    def process_dsr(self, request_type: str, subject_id: str,
                   data: Dict = None) -> Dict[str, Any]:
        """Process Data Subject Request (DSR)."""
        request = {
            "id": f"dsr_{datetime.utcnow().timestamp()}",
            "type": request_type,
            "subject_id_hash": hashlib.sha256(subject_id.encode()).hexdigest()[:16],
            "requested_at": datetime.utcnow().isoformat(),
            "status": "processing",
            "deadline": (datetime.utcnow() + timedelta(days=30)).isoformat()
        }

        if request_type == "access":
            request["result"] = self._handle_access_request(subject_id)
        elif request_type == "erasure":
            request["result"] = self._handle_erasure_request(subject_id)
        elif request_type == "rectification":
            request["result"] = self._handle_rectification_request(subject_id, data)
        elif request_type == "portability":
            request["result"] = self._handle_portability_request(subject_id)
        elif request_type == "restriction":
            request["result"] = self._handle_restriction_request(subject_id)

        request["status"] = "completed"
        self.data_subject_requests.append(request)

        return request

    def _handle_access_request(self, subject_id: str) -> Dict:
        """Handle right of access request."""
        # Collect all data related to subject
        return {
            "data_categories": ["conversation_history", "preferences"],
            "processing_purposes": ["customer_service"],
            "recipients": ["internal_teams"],
            "retention_period": "90 days",
            "source": "direct_collection"
        }

    def _handle_erasure_request(self, subject_id: str) -> Dict:
        """Handle right to erasure (right to be forgotten)."""
        # Delete or anonymize subject data
        systems_cleared = []

        # Clear from agent memory
        systems_cleared.append("agent_memory")

        # Clear from conversation history
        systems_cleared.append("conversation_history")

        # Clear from analytics
        systems_cleared.append("analytics")

        # Withdraw consent
        self.withdraw_consent(subject_id)

        return {
            "systems_cleared": systems_cleared,
            "completed_at": datetime.utcnow().isoformat()
        }

    def _handle_portability_request(self, subject_id: str) -> Dict:
        """Handle data portability request."""
        # Export data in machine-readable format
        return {
            "format": "json",
            "download_url": f"/exports/{subject_id}/data.json",
            "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat()
        }


class PIIMasker:
    """Mask PII in agent inputs/outputs."""

    PII_PATTERNS = {
        "email": r"[\w\.-]+@[\w\.-]+\.\w+",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    }

    def __init__(self, mask_char: str = "*"):
        self.mask_char = mask_char

    def mask(self, text: str, types: List[str] = None) -> str:
        """Mask PII in text."""
        import re

        types = types or list(self.PII_PATTERNS.keys())
        masked = text

        for pii_type in types:
            if pii_type in self.PII_PATTERNS:
                pattern = self.PII_PATTERNS[pii_type]
                masked = re.sub(
                    pattern,
                    lambda m: self.mask_char * len(m.group()),
                    masked
                )

        return masked

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII in text."""
        import re

        detections = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                detections.append({
                    "type": pii_type,
                    "start": match.start(),
                    "end": match.end(),
                    "value_preview": text[match.start():match.start()+3] + "..."
                })

        return detections
```

### Data Retention

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List
from enum import Enum

class DataCategory(Enum):
    AGENT_LOGS = "agent_logs"
    CONVERSATION_HISTORY = "conversation_history"
    AUDIT_TRAILS = "audit_trails"
    TRAINING_DATA = "training_data"
    USER_PREFERENCES = "user_preferences"
    PII_DATA = "pii_data"
    ANALYTICS = "analytics"

@dataclass
class RetentionPolicy:
    category: DataCategory
    retention_days: int
    legal_basis: str
    deletion_method: str
    archive_before_delete: bool = False

class DataRetentionManager:
    """Manage data retention policies."""

    REGULATORY_REQUIREMENTS = {
        "GDPR": {
            DataCategory.PII_DATA: 365,
            DataCategory.CONVERSATION_HISTORY: 90,
        },
        "HIPAA": {
            DataCategory.AUDIT_TRAILS: 2190,  # 6 years
            DataCategory.PII_DATA: 2190,
        },
        "SOC2": {
            DataCategory.AUDIT_TRAILS: 365,
            DataCategory.AGENT_LOGS: 365,
        },
        "PCI_DSS": {
            DataCategory.AUDIT_TRAILS: 365,
            DataCategory.AGENT_LOGS: 90,
        }
    }

    def __init__(self, applicable_regulations: List[str]):
        self.regulations = applicable_regulations
        self.policies = self._compute_policies()

    def _compute_policies(self) -> Dict[DataCategory, RetentionPolicy]:
        """Compute retention policies based on applicable regulations."""
        policies = {}

        for category in DataCategory:
            max_retention = 30  # Default 30 days
            legal_bases = []

            for regulation in self.regulations:
                reg_requirements = self.REGULATORY_REQUIREMENTS.get(regulation, {})
                if category in reg_requirements:
                    max_retention = max(max_retention, reg_requirements[category])
                    legal_bases.append(regulation)

            policies[category] = RetentionPolicy(
                category=category,
                retention_days=max_retention,
                legal_basis=", ".join(legal_bases) if legal_bases else "Business need",
                deletion_method="secure_erase" if category == DataCategory.PII_DATA else "standard",
                archive_before_delete=category in [
                    DataCategory.AUDIT_TRAILS,
                    DataCategory.TRAINING_DATA
                ]
            )

        return policies

    def get_policy(self, category: DataCategory) -> RetentionPolicy:
        """Get retention policy for a data category."""
        return self.policies.get(category)

    def get_expired_data(self, category: DataCategory,
                        data_store) -> List[Dict]:
        """Get data that has exceeded retention period."""
        policy = self.get_policy(category)
        cutoff = datetime.utcnow() - timedelta(days=policy.retention_days)

        return data_store.query(
            category=category.value,
            created_before=cutoff
        )

    def execute_retention(self, category: DataCategory, data_store):
        """Execute retention policy for a category."""
        policy = self.get_policy(category)
        expired = self.get_expired_data(category, data_store)

        results = {
            "category": category.value,
            "policy": policy,
            "records_processed": 0,
            "archived": 0,
            "deleted": 0
        }

        for record in expired:
            if policy.archive_before_delete:
                data_store.archive(record)
                results["archived"] += 1

            if policy.deletion_method == "secure_erase":
                data_store.secure_delete(record)
            else:
                data_store.delete(record)

            results["deleted"] += 1
            results["records_processed"] += 1

        return results
```

---

## Audit and Logging

### Comprehensive Audit Logging

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import json
import hashlib

class AuditEventType(Enum):
    # Agent lifecycle
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_ERROR = "agent.error"

    # Decisions
    DECISION_MADE = "decision.made"
    DECISION_OVERRIDDEN = "decision.overridden"

    # Tool usage
    TOOL_INVOKED = "tool.invoked"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"

    # Data access
    DATA_ACCESSED = "data.accessed"
    DATA_MODIFIED = "data.modified"
    DATA_DELETED = "data.deleted"

    # Authorization
    AUTH_REQUESTED = "auth.requested"
    AUTH_GRANTED = "auth.granted"
    AUTH_DENIED = "auth.denied"

    # Policy
    POLICY_EVALUATED = "policy.evaluated"
    POLICY_VIOLATED = "policy.violated"

@dataclass
class AuditEvent:
    """Comprehensive audit event for AI agent actions."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    agent_id: str

    # Context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    # Action details
    action: Optional[str] = None
    resource: Optional[str] = None
    resource_type: Optional[str] = None

    # Input/Output (hashed for privacy)
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None

    # Decision tracking
    decision_factors: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None

    # Authorization
    authorization_policy: Optional[str] = None
    authorization_result: Optional[str] = None

    # Performance
    duration_ms: Optional[int] = None

    # Error handling
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON for logging."""
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
        }

        # Add optional fields
        for field_name in [
            "user_id", "session_id", "request_id", "action",
            "resource", "resource_type", "input_hash", "output_hash",
            "decision_factors", "confidence_score", "authorization_policy",
            "authorization_result", "duration_ms", "error_code",
            "error_message", "metadata"
        ]:
            value = getattr(self, field_name)
            if value is not None:
                data[field_name] = value

        return json.dumps(data)


class ImmutableAuditLogger:
    """Tamper-evident audit logging with hash chain."""

    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.sequence = 0
        self.previous_hash = "genesis"

    def log(self, event: AuditEvent) -> Dict[str, str]:
        """Log an event with cryptographic integrity."""
        self.sequence += 1

        entry_data = event.to_json()
        timestamp = datetime.utcnow().isoformat()

        # Calculate hash including previous hash
        hash_input = f"{self.sequence}:{timestamp}:{entry_data}:{self.previous_hash}"
        entry_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        entry = {
            "sequence_number": self.sequence,
            "timestamp": timestamp,
            "event_data": entry_data,
            "previous_hash": self.previous_hash,
            "entry_hash": entry_hash
        }

        self.storage.write(entry)
        self.previous_hash = entry_hash

        return entry

    def verify_chain(self, entries: List[Dict]) -> bool:
        """Verify integrity of log chain."""
        if not entries:
            return True

        for i, entry in enumerate(entries):
            hash_input = (
                f"{entry['sequence_number']}:{entry['timestamp']}:"
                f"{entry['event_data']}:{entry['previous_hash']}"
            )
            calculated_hash = hashlib.sha256(hash_input.encode()).hexdigest()

            if calculated_hash != entry["entry_hash"]:
                return False

            if i > 0 and entry["previous_hash"] != entries[i-1]["entry_hash"]:
                return False

        return True


# Required audit events by regulation
MANDATORY_AUDIT_EVENTS = {
    "HIPAA": [
        AuditEventType.DATA_ACCESSED,
        AuditEventType.DATA_MODIFIED,
        AuditEventType.AUTH_GRANTED,
        AuditEventType.AUTH_DENIED,
    ],
    "GDPR": [
        AuditEventType.DATA_ACCESSED,
        AuditEventType.DATA_MODIFIED,
        AuditEventType.DATA_DELETED,
        AuditEventType.DECISION_MADE,
    ],
    "SOC2": [
        AuditEventType.AUTH_GRANTED,
        AuditEventType.AUTH_DENIED,
        AuditEventType.POLICY_VIOLATED,
        AuditEventType.AGENT_ERROR,
    ],
    "PCI_DSS": [
        AuditEventType.DATA_ACCESSED,
        AuditEventType.AUTH_GRANTED,
        AuditEventType.AUTH_DENIED,
        AuditEventType.POLICY_VIOLATED,
    ]
}
```

---

## Access Control

### Role-Based Access Control

```python
from dataclasses import dataclass
from typing import Set, Dict, List
from enum import Enum

class Permission(Enum):
    # Data permissions
    READ_PII = "data:pii:read"
    WRITE_PII = "data:pii:write"
    DELETE_PII = "data:pii:delete"

    READ_PHI = "data:phi:read"
    WRITE_PHI = "data:phi:write"

    READ_FINANCIAL = "data:financial:read"
    WRITE_FINANCIAL = "data:financial:write"

    # Tool permissions
    TOOL_WEB_SEARCH = "tool:web_search:execute"
    TOOL_DATABASE = "tool:database:execute"
    TOOL_EMAIL = "tool:email:execute"
    TOOL_FILE_SYSTEM = "tool:filesystem:execute"

    # Agent permissions
    AGENT_SPAWN = "agent:spawn"
    AGENT_DELEGATE = "agent:delegate"

@dataclass
class Role:
    name: str
    permissions: Set[Permission]
    max_data_sensitivity: str
    allowed_tools: Set[str]

@dataclass
class AgentIdentity:
    agent_id: str
    roles: Set[str]
    effective_permissions: Set[Permission]
    delegated_by: str = None
    delegation_scope: Set[Permission] = None

class AgentRBACManager:
    """Manage role-based access control for AI agents."""

    STANDARD_ROLES = {
        "customer_service_agent": Role(
            name="customer_service_agent",
            permissions={
                Permission.READ_PII,
                Permission.TOOL_DATABASE,
                Permission.TOOL_EMAIL,
            },
            max_data_sensitivity="confidential",
            allowed_tools={"customer_lookup", "ticket_management", "email_send"}
        ),
        "analytics_agent": Role(
            name="analytics_agent",
            permissions={
                Permission.READ_FINANCIAL,
                Permission.TOOL_DATABASE,
            },
            max_data_sensitivity="internal",
            allowed_tools={"database_query", "report_generator"}
        ),
        "healthcare_agent": Role(
            name="healthcare_agent",
            permissions={
                Permission.READ_PHI,
                Permission.WRITE_PHI,
                Permission.TOOL_DATABASE,
            },
            max_data_sensitivity="restricted",
            allowed_tools={"ehr_access", "clinical_notes", "lab_results"}
        ),
    }

    def __init__(self):
        self.roles: Dict[str, Role] = self.STANDARD_ROLES.copy()
        self.agent_identities: Dict[str, AgentIdentity] = {}

    def assign_role(self, agent_id: str, role_name: str):
        """Assign a role to an agent."""
        if agent_id not in self.agent_identities:
            self.agent_identities[agent_id] = AgentIdentity(
                agent_id=agent_id,
                roles=set(),
                effective_permissions=set()
            )

        identity = self.agent_identities[agent_id]
        identity.roles.add(role_name)
        self._recalculate_permissions(agent_id)

    def check_permission(self, agent_id: str, permission: Permission) -> bool:
        """Check if agent has a specific permission."""
        identity = self.agent_identities.get(agent_id)
        if not identity:
            return False

        if permission not in identity.effective_permissions:
            return False

        # Check delegation scope if applicable
        if identity.delegation_scope is not None:
            return permission in identity.delegation_scope

        return True

    def delegate_to_agent(self, delegating_user_id: str, agent_id: str,
                         scope: Set[Permission]):
        """Delegate specific permissions from user to agent."""
        identity = self.agent_identities.get(agent_id)
        if not identity:
            raise ValueError(f"Agent {agent_id} not found")

        # Can only delegate permissions the agent already has
        valid_scope = scope.intersection(identity.effective_permissions)

        identity.delegated_by = delegating_user_id
        identity.delegation_scope = valid_scope

    def _recalculate_permissions(self, agent_id: str):
        """Recalculate effective permissions from roles."""
        identity = self.agent_identities[agent_id]
        permissions = set()

        for role_name in identity.roles:
            if role_name in self.roles:
                permissions.update(self.roles[role_name].permissions)

        identity.effective_permissions = permissions
```

---

## Model Risk Management

### Model Governance

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

class ModelStatus(Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    APPROVED = "approved"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

@dataclass
class ModelRegistration:
    """Model registration for governance tracking."""
    model_id: str
    name: str
    version: str
    provider: str  # e.g., "anthropic", "openai"
    model_type: str  # e.g., "llm", "embedding"

    # Governance
    status: ModelStatus
    owner: str
    use_cases: List[str]
    risk_tier: str

    # Validation
    validation_date: Optional[datetime]
    validation_results: Optional[Dict[str, Any]]
    approved_by: Optional[str]

    # Monitoring
    performance_baseline: Optional[Dict[str, float]]
    drift_thresholds: Optional[Dict[str, float]]

    # Metadata
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

class ModelRegistry:
    """Central registry for model governance."""

    def __init__(self):
        self.models: Dict[str, ModelRegistration] = {}
        self.approval_history: List[Dict] = []

    def register_model(self, registration: ModelRegistration):
        """Register a new model."""
        self.models[registration.model_id] = registration

    def update_status(self, model_id: str, new_status: ModelStatus,
                     updated_by: str, reason: str):
        """Update model status with audit trail."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model = self.models[model_id]
        old_status = model.status

        model.status = new_status
        model.updated_at = datetime.utcnow()

        # Record status change
        self.approval_history.append({
            "model_id": model_id,
            "old_status": old_status.value,
            "new_status": new_status.value,
            "updated_by": updated_by,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        })

    def record_validation(self, model_id: str, results: Dict[str, Any]):
        """Record model validation results."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model = self.models[model_id]
        model.validation_date = datetime.utcnow()
        model.validation_results = results
        model.updated_at = datetime.utcnow()

        # Auto-transition to validation status
        if model.status == ModelStatus.DEVELOPMENT:
            model.status = ModelStatus.VALIDATION

    def approve_model(self, model_id: str, approved_by: str):
        """Approve model for production use."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model = self.models[model_id]

        # Check validation
        if not model.validation_results:
            raise ValueError("Model must be validated before approval")

        model.status = ModelStatus.APPROVED
        model.approved_by = approved_by
        model.updated_at = datetime.utcnow()

        self.update_status(model_id, ModelStatus.APPROVED, approved_by,
                          "Model approved for production")

    def get_production_models(self) -> List[ModelRegistration]:
        """Get all models approved for production."""
        return [
            m for m in self.models.values()
            if m.status in [ModelStatus.APPROVED, ModelStatus.PRODUCTION]
        ]

    def check_model_allowed(self, model_id: str, use_case: str) -> bool:
        """Check if model is allowed for a use case."""
        if model_id not in self.models:
            return False

        model = self.models[model_id]

        # Must be approved or in production
        if model.status not in [ModelStatus.APPROVED, ModelStatus.PRODUCTION]:
            return False

        # Check use case
        return use_case in model.use_cases
```

---

## Documentation Requirements

### Required Documentation

| Document | Purpose | Update Frequency |
|----------|---------|------------------|
| **System Description** | Technical architecture, data flows | On major changes |
| **Risk Assessment** | Risk identification and mitigations | Annually or on changes |
| **Data Processing Records** | ROPA for GDPR compliance | Continuous |
| **Model Cards** | Model capabilities, limitations, bias | Per model version |
| **Incident Reports** | Security/safety incidents | Per incident |
| **Audit Reports** | Compliance verification | Quarterly/annually |

### Model Card Template

```python
@dataclass
class ModelCard:
    """Standardized model documentation."""

    # Model Details
    model_name: str
    model_version: str
    model_type: str
    developer: str
    release_date: str

    # Intended Use
    primary_uses: List[str]
    primary_users: List[str]
    out_of_scope_uses: List[str]

    # Training Data
    training_data_description: str
    training_data_size: str
    preprocessing_steps: List[str]

    # Evaluation
    evaluation_metrics: Dict[str, float]
    evaluation_datasets: List[str]
    performance_by_group: Dict[str, Dict[str, float]]

    # Ethical Considerations
    potential_harms: List[str]
    mitigations: List[str]
    bias_evaluation: str

    # Limitations
    known_limitations: List[str]
    failure_modes: List[str]

    # Recommendations
    usage_recommendations: List[str]
    monitoring_recommendations: List[str]

    def to_markdown(self) -> str:
        """Generate markdown documentation."""
        sections = [
            f"# Model Card: {self.model_name} v{self.model_version}",
            "",
            "## Model Details",
            f"- **Type:** {self.model_type}",
            f"- **Developer:** {self.developer}",
            f"- **Release Date:** {self.release_date}",
            "",
            "## Intended Use",
            "### Primary Uses",
            *[f"- {use}" for use in self.primary_uses],
            "",
            "### Out of Scope",
            *[f"- {use}" for use in self.out_of_scope_uses],
            "",
            "## Evaluation Results",
            *[f"- **{metric}:** {value}" for metric, value in self.evaluation_metrics.items()],
            "",
            "## Ethical Considerations",
            "### Potential Harms",
            *[f"- {harm}" for harm in self.potential_harms],
            "",
            "### Mitigations",
            *[f"- {mitigation}" for mitigation in self.mitigations],
            "",
            "## Limitations",
            *[f"- {limitation}" for limitation in self.known_limitations],
            "",
            "## Recommendations",
            *[f"- {rec}" for rec in self.usage_recommendations],
        ]

        return "\n".join(sections)
```

---

## Compliance Checklist

### Pre-Deployment Checklist

- [ ] Risk assessment completed
- [ ] Governance approval obtained
- [ ] Data processing registered (ROPA)
- [ ] Privacy impact assessment completed
- [ ] Model card documented
- [ ] Audit logging configured
- [ ] Access controls implemented
- [ ] Human oversight mechanism in place
- [ ] Transparency disclosures ready
- [ ] Incident response plan documented
- [ ] Retention policies configured
- [ ] Security review completed

### Ongoing Compliance

- [ ] Regular risk assessments (quarterly)
- [ ] Audit log reviews (monthly)
- [ ] Model performance monitoring (continuous)
- [ ] Bias testing (quarterly)
- [ ] Data subject request handling (as needed)
- [ ] Policy updates (as regulations change)
- [ ] Training for operators (annually)
- [ ] Incident response drills (annually)

---

## Related Documents

- [Security Essentials](security-essentials.md) - Security implementation
- [Security Research](security-research.md) - Threat landscape
- [Evaluation and Debugging](../phase-4-production/evaluation-and-debugging.md) - Model evaluation
- [Testing Guide](../phase-4-production/testing-guide.md) - Compliance testing
