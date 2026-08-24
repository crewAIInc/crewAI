# Runtime security guardrail implementation conforming to
# CCS — Correctover Conformance Shape
# Reference: https://datatracker.ietf.org/doc/draft-correctover-ccs/
# Reference implementation by Correctover (https://correctover.com)

"""
GuardrailProvider — Content-addressed decision audit chain for tool call authorization.

This module implements the CCS (Correctover Conformance Shape) guardrail
interface for crewAI. CCS defines a vendor-neutral runtime security
conformance standard; this is the reference implementation for the crewAI
framework.

Integrates through crewAI's existing BeforeToolCallHook infrastructure.
Aligned with crewAI#4877 converged spec (safal207's GuardrailDecisionV1,
babyblueviper1's recompute verification, Yarmoluk's CKG declarative authorization).

CCS Spec: https://datatracker.ietf.org/doc/draft-correctover-ccs/
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from crewai.hooks.tool_hooks import ToolCallHookContext as _CrewAIToolCallHookContext


# ---------------------------------------------------------------------------
# Canonical JSON serialization
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# decision_id computation
# ---------------------------------------------------------------------------

def compute_decision_id(claims: Dict[str, Any], expires_at: Optional[float] = None) -> str:
    """
    Content-addressed decision identifier.

    Algorithm:
    1. Build preimage: claims ∪ {"_expires_at": expires_at}
       If expires_at is None, the preimage is claims alone
       (no _expires_at key injected into the serialized payload).
    2. Serialize: json.dumps(preimage, sort_keys=True, separators=(",", ":"))
    3. Digest: SHA-256(canonical_bytes).hexdigest()

    Same claims + same expires_at → same decision_id.
    Tampering with either → verify_integrity() returns False.
    """
    preimage = claims.copy()
    if expires_at is not None:
        preimage["_expires_at"] = expires_at

    canonical_payload = canonical_json(preimage)
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuardrailDecisionV1:
    """
    Pre-execution authorization decision.

    decision_id is content-addressed: SHA-256 of canonical JSON over
    the claims dict (plus _expires_at if set). This enables:
    - O(1) audit dedup
    - Independent recompute verification without contacting the issuer
    - Tamper detection: modifying claims or expires_at invalidates the id
    """
    decision_id: str
    authorized: bool
    claims: Dict[str, Any]
    expires_at: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if this decision has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def verify_integrity(self) -> bool:
        """
        Recompute decision_id from claims + expires_at and compare.
        Returns False if claims or expires_at were tampered with.
        """
        expected_id = compute_decision_id(self.claims, self.expires_at)
        return self.decision_id == expected_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "authorized": self.authorized,
            "claims": self.claims,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class ActionEnvelopeV1:
    """
    Post-execution evidence envelope.

    Links back to the authorization decision via decision_id.
    Stores only the digest of the tool result (never raw value) to:
    - Prevent audit log bloat
    - Avoid leaking sensitive tool outputs into audit storage
    """
    decision_id: str
    tool_result_digest: str
    executed_at: float
    duration_ms: float

    @staticmethod
    def digest_result(result: Any) -> str:
        """SHA-256 digest of a tool result. Never stores raw value.

        Both ``None`` and an empty string produce the digest of an empty
        byte string, matching the convention that "no result" and "empty
        result" carry the same audit fingerprint.
        """
        if result is None or result == "":
            raw = ""
        else:
            raw = canonical_json(result)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "tool_result_digest": self.tool_result_digest,
            "executed_at": self.executed_at,
            "duration_ms": self.duration_ms,
        }


# ---------------------------------------------------------------------------
# GuardrailProvider protocol
# ---------------------------------------------------------------------------

class GuardrailProvider(ABC):
    """
    Abstract guardrail provider.

    Stateless protocol: receives a ToolCallHookContext, returns a
    GuardrailDecisionV1. No mutable state in the provider — keeps
    providers testable and composable.
    """

    @abstractmethod
    def authorize(self, context: "ToolCallHookContext") -> GuardrailDecisionV1:
        """
        Evaluate whether a tool call should be authorized.

        Args:
            context: crewAI's ToolCallHookContext containing tool_name,
                     tool_input, agent, etc.

        Returns:
            GuardrailDecisionV1 with authorized=True/False and claims
            capturing the decision rationale.
        """
        ...


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

class AllowAllGuardrailProvider(GuardrailProvider):
    """Default provider — authorizes all tool calls."""

    def authorize(self, context) -> GuardrailDecisionV1:
        claims = {
            "provider": "AllowAllGuardrailProvider",
            "tool_name": getattr(context, "tool_name", "unknown"),
            "agent_role": getattr(getattr(context, "agent", None), "role", "unknown"),
        }
        return GuardrailDecisionV1(
            decision_id=compute_decision_id(claims),
            authorized=True,
            claims=claims,
        )


class DenyAllGuardrailProvider(GuardrailProvider):
    """Safety lock — blocks all tool calls."""

    def authorize(self, context) -> GuardrailDecisionV1:
        claims = {
            "provider": "DenyAllGuardrailProvider",
            "tool_name": getattr(context, "tool_name", "unknown"),
            "reason": "deny-all safety lock active",
        }
        return GuardrailDecisionV1(
            decision_id=compute_decision_id(claims),
            authorized=False,
            claims=claims,
        )


class ToolListGuardrailProvider(GuardrailProvider):
    """
    Allowlist/blocklist provider — authorizes by tool name.

    If allowed_tools is set, only those tools pass.
    If denied_tools is set, those tools are blocked.
    If both are set, allowed takes precedence (explicit allow overrides deny).
    """

    def __init__(
        self,
        allowed_tools: Optional[Set[str]] = None,
        denied_tools: Optional[Set[str]] = None,
    ):
        self.allowed_tools = allowed_tools
        self.denied_tools = denied_tools or set()

    def authorize(self, context) -> GuardrailDecisionV1:
        tool_name = getattr(context, "tool_name", "unknown")
        agent_role = getattr(getattr(context, "agent", None), "role", "unknown")

        claims = {
            "provider": "ToolListGuardrailProvider",
            "tool_name": tool_name,
            "agent_role": agent_role,
        }

        # Explicit allow takes precedence
        if self.allowed_tools is not None:
            authorized = tool_name in self.allowed_tools
            claims["policy"] = "allowlist"
            claims["allowed_tools"] = sorted(self.allowed_tools)
        elif tool_name in self.denied_tools:
            authorized = False
            claims["policy"] = "denylist"
            claims["denied_tools"] = sorted(self.denied_tools)
        else:
            authorized = True
            claims["policy"] = "default-allow"

        return GuardrailDecisionV1(
            decision_id=compute_decision_id(claims),
            authorized=authorized,
            claims=claims,
        )


class CKGGuardrailProvider(GuardrailProvider):
    """
    Declarative authorization provider based on Yarmoluk's CKG proposal
    from crewAI#4877.

    6 built-in predicates:
    - tool_name_in: tool name must be in set
    - tool_name_not_in: tool name must not be in set
    - agent_role_in: agent role must be in set
    - param_matches: specific param must match value
    - has_param: tool input must contain specific param
    - no_param: tool input must NOT contain specific param

    Constraints are AND-combined. Extensible via add_constraint().
    """

    # Predicates accepted by :meth:`add_constraint`.  Any other name is
    # rejected up-front so that a typo cannot turn a deny rule into a
    # silent allow (fail-closed security semantics).
    SUPPORTED_PREDICATES = frozenset(
        {
            "tool_name_in",
            "tool_name_not_in",
            "agent_role_in",
            "param_matches",
            "has_param",
            "no_param",
        }
    )

    def __init__(self):
        self._constraints: List[Dict[str, Any]] = []

    def add_constraint(self, predicate: str, **kwargs) -> "CKGGuardrailProvider":
        """
        Add a declarative constraint. Returns self for chaining.

        Supported predicates:
        - tool_name_in(tools: Set[str])
        - tool_name_not_in(tools: Set[str])
        - agent_role_in(roles: Set[str])
        - param_matches(name: str, value: Any)
        - has_param(name: str)
        - no_param(name: str)

        Raises ``ValueError`` for unknown predicates so that a misspelled
        rule cannot be silently skipped.
        """
        if predicate not in self.SUPPORTED_PREDICATES:
            raise ValueError(
                f"Unsupported guardrail predicate: {predicate!r}. "
                f"Supported: {sorted(self.SUPPORTED_PREDICATES)}"
            )
        self._constraints.append({"predicate": predicate, **kwargs})
        return self

    def authorize(self, context) -> GuardrailDecisionV1:
        tool_name = getattr(context, "tool_name", "unknown")
        tool_input = getattr(context, "tool_input", {}) or {}
        agent_role = getattr(getattr(context, "agent", None), "role", "unknown")

        claims = {
            "provider": "CKGGuardrailProvider",
            "tool_name": tool_name,
            "agent_role": agent_role,
        }

        satisfied = True
        failed_predicate = None

        for constraint in self._constraints:
            pred = constraint["predicate"]

            if pred == "tool_name_in":
                if tool_name not in constraint["tools"]:
                    satisfied = False
                    failed_predicate = pred

            elif pred == "tool_name_not_in":
                if tool_name in constraint["tools"]:
                    satisfied = False
                    failed_predicate = pred

            elif pred == "agent_role_in":
                if agent_role not in constraint["roles"]:
                    satisfied = False
                    failed_predicate = pred

            elif pred == "param_matches":
                if tool_input.get(constraint["name"]) != constraint["value"]:
                    satisfied = False
                    failed_predicate = pred

            elif pred == "has_param":
                if constraint["name"] not in tool_input:
                    satisfied = False
                    failed_predicate = pred

            elif pred == "no_param":
                if constraint["name"] in tool_input:
                    satisfied = False
                    failed_predicate = pred

            else:
                # Defensive: unknown predicate names are rejected at
                # add_constraint time, but if a constraint somehow reaches
                # evaluation through another path, fail closed rather than
                # silently allowing the call.
                satisfied = False
                failed_predicate = pred

        if not satisfied:
            claims["failed_predicate"] = failed_predicate

        claims["constraints_count"] = len(self._constraints)

        return GuardrailDecisionV1(
            decision_id=compute_decision_id(claims),
            authorized=satisfied,
            claims=claims,
        )


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

class AuditTrail:
    """
    In-memory audit chain for decisions and action envelopes.

    Each decision can be independently verified via verify_integrity().
    Each envelope links back to its decision via decision_id.
    """

    def __init__(self):
        self._decisions: Dict[str, GuardrailDecisionV1] = {}
        # Each decision may have zero or more execution envelopes —
        # two identical tool calls produce the same content-addressed
        # decision_id but are distinct execution events that must both
        # be preserved in the audit trail.
        self._envelopes: Dict[str, List[ActionEnvelopeV1]] = {}

    def record_decision(self, decision: GuardrailDecisionV1) -> None:
        self._decisions[decision.decision_id] = decision

    def record_envelope(self, envelope: ActionEnvelopeV1) -> None:
        self._envelopes.setdefault(envelope.decision_id, []).append(envelope)

    def get_decision(self, decision_id: str) -> Optional[GuardrailDecisionV1]:
        return self._decisions.get(decision_id)

    def get_envelope(
        self, decision_id: str, index: int = -1
    ) -> Optional[ActionEnvelopeV1]:
        """Return the *index*-th envelope for *decision_id*.

        The default ``index=-1`` returns the most recent envelope, which
        matches the previous single-envelope behaviour for callers that
        only expect one.
        """
        envelopes = self._envelopes.get(decision_id)
        if not envelopes:
            return None
        try:
            return envelopes[index]
        except IndexError:
            return None

    def get_envelopes(self, decision_id: str) -> List[ActionEnvelopeV1]:
        """Return all execution envelopes for *decision_id*."""
        return list(self._envelopes.get(decision_id, []))

    def clear(self) -> None:
        self._decisions.clear()
        self._envelopes.clear()

    @property
    def decision_count(self) -> int:
        return len(self._decisions)

    @property
    def envelope_count(self) -> int:
        """Total number of recorded execution envelopes across all decisions."""
        return sum(len(v) for v in self._envelopes.values())


# ---------------------------------------------------------------------------
# ToolCallHookContext (minimal stub for standalone usage / testing)
#
# In crewAI integration, the real ToolCallHookContext from
# crewai.hooks.tool_hooks is used instead. The field name ``tool_input``
# matches crewAI's actual attribute (not ``tool_args``).
# ---------------------------------------------------------------------------

@dataclass
class ToolCallHookContext:
    """
    Minimal context object matching crewAI's BeforeToolCallHook interface.

    In production, crewAI provides ``crewai.hooks.tool_hooks.ToolCallHookContext``.
    This stub exists for testing and standalone usage.
    """
    tool_name: str
    tool_input: Dict[str, Any] = field(default_factory=dict)
    agent: Optional[Any] = None


# ---------------------------------------------------------------------------
# GuardrailContext — ties provider + trail + hook together
# ---------------------------------------------------------------------------

class GuardrailContext:
    """
    Runtime context that ties a GuardrailProvider to an AuditTrail.

    Usage:
        ctx = GuardrailContext(provider=MyProvider())
        decision = ctx.authorize(tool_call_context)
        # ... execute tool ...
        ctx.record_result(decision, result, start_time)
    """

    def __init__(
        self,
        provider: GuardrailProvider,
        trail: Optional[AuditTrail] = None,
        on_deny: Optional[Callable[[GuardrailDecisionV1], None]] = None,
    ):
        self.provider = provider
        self.trail = trail or AuditTrail()
        self.on_deny = on_deny

    def authorize(self, context) -> GuardrailDecisionV1:
        """Run provider authorization and record in audit trail."""
        decision = self.provider.authorize(context)
        self.trail.record_decision(decision)

        if not decision.authorized and self.on_deny:
            self.on_deny(decision)

        return decision

    def after_tool_call(
        self,
        decision: GuardrailDecisionV1,
        result: Any,
        start_time: float,
    ) -> ActionEnvelopeV1:
        """
        Record post-execution evidence envelope.
        Opt-in — not auto-registered; call manually or via after-hook.
        """
        duration_ms = (time.time() - start_time) * 1000
        envelope = ActionEnvelopeV1(
            decision_id=decision.decision_id,
            tool_result_digest=ActionEnvelopeV1.digest_result(result),
            executed_at=time.time(),
            duration_ms=duration_ms,
        )
        self.trail.record_envelope(envelope)
        return envelope


# ---------------------------------------------------------------------------
# Hook factory — one-line integration with crewAI BeforeToolCallHook
# ---------------------------------------------------------------------------

def make_guardrail_hook(
    provider: GuardrailProvider,
    trail: Optional[AuditTrail] = None,
    on_deny: Optional[Callable[[GuardrailDecisionV1], None]] = None,
) -> Callable:
    """
    Factory: creates a BeforeToolCallHook-compatible callable.

    The returned hook conforms to crewAI's hook signature — it receives a
    single ``ToolCallHookContext`` argument (as defined in
    ``crewai.hooks.tool_hooks``):

        def hook(context: ToolCallHookContext) -> bool | None:
            ...

    Behavior:
    1. Calls provider.authorize(context)
    2. Records GuardrailDecisionV1 in the audit trail
    3. Returns False to block, None to allow

    Usage:
        from crewai.hooks import register_before_tool_call_hook
        from crewai.guardrails import ToolListGuardrailProvider, make_guardrail_hook

        register_before_tool_call_hook(
            make_guardrail_hook(
                ToolListGuardrailProvider(allowed_tools={"read_file"})
            )
        )
    """
    ctx = GuardrailContext(provider=provider, trail=trail, on_deny=on_deny)

    def _hook(context: "ToolCallHookContext") -> Optional[bool]:
        decision = ctx.authorize(context)
        if not decision.authorized:
            return False  # Block execution
        return None  # Allow execution

    # Expose internal context for testing / advanced usage
    _hook._guardrail_context = ctx

    return _hook


# ---------------------------------------------------------------------------
# Gap analysis — corresponds to engine-v4 AS-GUARDRAIL-MISS-001
# ---------------------------------------------------------------------------

def detect_missing_guardrail(crew_or_agent) -> List[Dict[str, Any]]:
    """
    Scan a crew or agent for missing guardrail registration.

    Corresponds to engine-v4 seed pattern AS-GUARDRAIL-MISS-001:
    flags agents/tools operating without any registered GuardrailProvider.

    Returns list of findings (empty = all guarded).
    """
    findings = []

    agents = getattr(crew_or_agent, "agents", [crew_or_agent])

    for agent in agents:
        agent_name = getattr(agent, "role", getattr(agent, "name", "unknown"))
        hooks = getattr(agent, "_before_tool_call_hooks", None)

        if not hooks:
            findings.append({
                "severity": "CRITICAL",
                "pattern": "AS-GUARDRAIL-MISS-001",
                "agent": agent_name,
                "message": f"Agent '{agent_name}' has no registered GuardrailProvider",
                "remediation": "Register a GuardrailProvider via make_guardrail_hook()",
            })

    return findings
