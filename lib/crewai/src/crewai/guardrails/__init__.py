"""
crewai.guardrails — Runtime security guardrail system for crewAI.

Reference implementation of the CCS (Correctover Conformance Shape) guardrail
interface. CCS is a vendor-neutral standard for runtime authorization, audit,
and conformance testing of AI agent frameworks.

Provides content-addressed decision audit chain for tool call authorization,
aligned with crewAI#4877.

CCS Spec: https://datatracker.ietf.org/doc/draft-correctover-ccs/
"""

from .guardrail_provider import (
    GuardrailDecisionV1,
    ActionEnvelopeV1,
    GuardrailProvider,
    AllowAllGuardrailProvider,
    DenyAllGuardrailProvider,
    ToolListGuardrailProvider,
    CKGGuardrailProvider,
    AuditTrail,
    GuardrailContext,
    ToolCallHookContext,
    make_guardrail_hook,
    detect_missing_guardrail,
    compute_decision_id,
    canonical_json,
)

__all__ = [
    "GuardrailDecisionV1",
    "ActionEnvelopeV1",
    "GuardrailProvider",
    "AllowAllGuardrailProvider",
    "DenyAllGuardrailProvider",
    "ToolListGuardrailProvider",
    "CKGGuardrailProvider",
    "AuditTrail",
    "GuardrailContext",
    "ToolCallHookContext",
    "make_guardrail_hook",
    "detect_missing_guardrail",
    "compute_decision_id",
    "canonical_json",
]
