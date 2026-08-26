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
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from crewai.hooks.tool_hooks import ToolCallHookContext as _CrewAIToolCallHookContext


# ---------------------------------------------------------------------------
# Canonical JSON serialization
# ---------------------------------------------------------------------------

# Domain separator prepended to every decision_id preimage so that a hash
# produced for one struct shape cannot be confused with another.
_DECISION_DOMAIN_SEPARATOR = "CCS-GuardrailDecisionV1"
_DECISION_SCHEMA_VERSION = 1

# Types that are safe for cross-language canonical JSON recompute.  ``set``
# and ``frozenset`` are *accepted* by the encoder but normalised to sorted
# lists so the output is reproducible in Go/Rust/JS.  Any other type raises
# ``TypeError`` rather than being silently stringified by ``default=str``.
_CANONICAL_ATOMS = (str, int, float, bool, type(None))


def _canonical_normalize(obj: Any) -> Any:
    """Recursively normalise *obj* into a JSON-safe canonical form.

    Sets/frozensets are converted to sorted lists.  Dicts are returned
    as-is (``json.dumps(sort_keys=True)`` handles key ordering).  Every
    leaf must be one of ``_CANONICAL_ATOMS``; unsupported types raise.

    Non-finite floats (``NaN``, ``+Infinity``, ``-Infinity``) are rejected
    because they are not valid JSON numbers per RFC 8259 §6 and Python's
    ``json.dumps`` defaults to ``allow_nan=True``, which emits the
    JavaScript literals ``NaN``/``Infinity`` — a non-canonical,
    language-specific leak into the signed preimage.  Raised by
    Aleksey Safonov (safal207) in PR review.
    """
    if isinstance(obj, bool):
        # bool must be checked before int (bool is a subclass of int).
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        import math as _math
        if _math.isnan(obj) or _math.isinf(obj):
            raise ValueError(
                f"canonical_json: non-finite float {obj!r} is not a valid "
                f"JSON number (RFC 8259 §6); use None, a string, or a finite "
                f"number instead."
            )
        return obj
    if isinstance(obj, str):
        return obj
    if obj is None:
        return None
    if isinstance(obj, (set, frozenset)):
        return sorted(_canonical_normalize(x) for x in obj)
    if isinstance(obj, (list, tuple)):
        return [_canonical_normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _canonical_normalize(v) for k, v in obj.items()}
    raise TypeError(
        f"canonical_json does not support type {type(obj).__name__!r}; "
        f"allowed: str, int, float, bool, None, list, tuple, set, frozenset, dict"
    )


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization: sorted keys, compact separators.

    Uses an explicit type allowlist so that unsupported Python types (e.g.
    ``datetime``, custom classes) raise ``TypeError`` instead of leaking a
    Python-specific ``str()`` representation into the canonical byte stream
    — which would make the hash non-reproducible in other languages.

    ``allow_nan=False`` is set defensively so that, even if a non-finite
    float bypasses ``_canonical_normalize`` in a future refactor,
    ``json.dumps`` raises ``ValueError`` rather than emitting the
    non-standard ``NaN``/``Infinity`` tokens.
    """
    return json.dumps(
        _canonical_normalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


# ---------------------------------------------------------------------------
# decision_id computation
# ---------------------------------------------------------------------------

def compute_decision_id(
    claims: Dict[str, Any],
    authorized: bool,
    expires_at: Optional[float] = None,
) -> str:
    """
    Content-addressed decision identifier.

    The preimage binds *five* fields, in this fixed order:

    1. ``_domain``    — ``"CCS-GuardrailDecisionV1"`` (cross-struct hash separation)
    2. ``_version``   — schema version integer
    3. ``authorized`` — the verdict itself
    4. ``claims``     — provider-specific decision rationale
    5. ``expires_at`` — optional expiration timestamp

    Binding ``authorized`` into the hash means a verdict cannot be flipped
    without invalidating ``decision_id``; ``verify_integrity()`` therefore
    detects tampering with the verdict, not just the claims.

    Same inputs → same decision_id.  Any mutation → verify_integrity() returns False.
    """
    preimage: Dict[str, Any] = {
        "_domain": _DECISION_DOMAIN_SEPARATOR,
        "_version": _DECISION_SCHEMA_VERSION,
        "authorized": authorized,
        "claims": claims,
    }
    if expires_at is not None:
        preimage["expires_at"] = expires_at

    canonical_payload = canonical_json(preimage)
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def _policy_digest(constraints: List[Dict[str, Any]]) -> str:
    """SHA-256 over the canonicalised constraint set.

    Constraints are normalised (sets → sorted lists) so the digest is
    stable across processes and independent of Python's hash seed.
    """
    canonical = canonical_json(constraints)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _action_digest(
    tool_name: str,
    agent_role: str,
    tool_input: Dict[str, Any],
) -> str:
    """SHA-256 over the concrete invocation: tool + agent + normalised input."""
    payload = {
        "tool_name": tool_name,
        "agent_role": agent_role,
        "tool_input": tool_input,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuardrailDecisionV1:
    """
    Pre-execution authorization decision.

    decision_id is content-addressed: SHA-256 of canonical JSON over
    the preimage ``{_domain, _version, authorized, claims, [expires_at]}``.
    This enables:
    - O(1) audit dedup
    - Independent recompute verification without contacting the issuer
    - Tamper detection: modifying claims, authorized, or expires_at
      invalidates the id
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
        Recompute decision_id from authorized + claims + expires_at and compare.
        Returns False if any of the verdict, claims, or expires_at were tampered with.
        """
        expected_id = compute_decision_id(self.claims, self.authorized, self.expires_at)
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

    ``attempt_id`` is a per-invocation UUIDv4.  ``decision_id`` is content
    identity (same claims+policy+verdict+action → same id, useful for
    deduplication); ``attempt_id`` distinguishes repeated executions of
    the same authorised call.
    """
    decision_id: str
    tool_result_digest: str
    executed_at: float
    duration_ms: float
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4()))

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
            "attempt_id": self.attempt_id,
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
            decision_id=compute_decision_id(claims, authorized=True),
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
            decision_id=compute_decision_id(claims, authorized=False),
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

        claims: Dict[str, Any] = {
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
            decision_id=compute_decision_id(claims, authorized=authorized),
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

    The decision claims include ``policy_digest`` (SHA-256 over the
    canonicalised constraint set) and ``action_digest`` (SHA-256 over
    tool_name + agent_role + normalised tool_input).  This means two
    materially different policies that both produce a passing call
    yield *different* ``decision_id`` values, so an independent
    verifier can reproduce why the concrete policy authorised the
    concrete action, not merely that the hash is intact.
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

    # Maps each predicate to the keyword arguments it requires.  A missing
    # or empty argument is rejected at registration time so that a typo such
    # as ``add_constraint("tool_name_in")`` (forgot ``tools=``) fails fast
    # instead of raising ``KeyError`` later during ``authorize()``.
    _PREDICATE_SCHEMA: Dict[str, Dict[str, Any]] = {
        "tool_name_in": {"tools": (set, list, frozenset, tuple)},
        "tool_name_not_in": {"tools": (set, list, frozenset, tuple)},
        "agent_role_in": {"roles": (set, list, frozenset, tuple)},
        "param_matches": {"name": (str,), "value": object},
        "has_param": {"name": (str,)},
        "no_param": {"name": (str,)},
    }

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

        Raises ``ValueError`` for unknown predicates or missing / empty
        required arguments, so that a malformed rule fails fast at
        registration time rather than crashing at authorization time.
        """
        if predicate not in self.SUPPORTED_PREDICATES:
            raise ValueError(
                f"Unsupported guardrail predicate: {predicate!r}. "
                f"Supported: {sorted(self.SUPPORTED_PREDICATES)}"
            )

        schema = self._PREDICATE_SCHEMA[predicate]
        for key, expected_types in schema.items():
            if key not in kwargs:
                raise ValueError(
                    f"Predicate {predicate!r} requires argument {key!r}."
                )
            value = kwargs[key]
            if expected_types is object:
                # ``value`` may be anything (including None), just verify
                # the key was explicitly provided (already checked above).
                continue
            if not isinstance(value, expected_types):
                type_names = (
                    t.__name__ for t in expected_types
                ) if isinstance(expected_types, tuple) else expected_types.__name__
                raise ValueError(
                    f"Predicate {predicate!r} argument {key!r} must be one of "
                    f"{list(type_names)}, got {type(value).__name__}."
                )
            if isinstance(value, (set, list, frozenset, tuple, str)) and len(value) == 0:
                raise ValueError(
                    f"Predicate {predicate!r} argument {key!r} must not be empty."
                )

        self._constraints.append({"predicate": predicate, **kwargs})
        return self

    def authorize(self, context) -> GuardrailDecisionV1:
        tool_name = getattr(context, "tool_name", "unknown")
        tool_input = getattr(context, "tool_input", {}) or {}
        agent_role = getattr(getattr(context, "agent", None), "role", "unknown")

        # Bind the concrete invocation (tool + agent + normalised input).
        action_digest = _action_digest(tool_name, agent_role, tool_input)
        # Bind the concrete policy (canonicalised constraint set).
        policy_digest = _policy_digest(self._constraints)

        claims: Dict[str, Any] = {
            "provider": "CKGGuardrailProvider",
            "tool_name": tool_name,
            "agent_role": agent_role,
            "policy_digest": policy_digest,
            "action_digest": action_digest,
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
            decision_id=compute_decision_id(claims, authorized=satisfied),
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
    Each envelope links back to its decision via decision_id and carries
    a unique attempt_id distinguishing repeated executions.
    """

    def __init__(self):
        self._decisions: Dict[str, GuardrailDecisionV1] = {}
        # Each decision may have zero or more execution envelopes —
        # two identical tool calls produce the same content-addressed
        # decision_id but are distinct execution events (different
        # attempt_id) that must both be preserved in the audit trail.
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

        Each envelope carries a fresh ``attempt_id`` (UUIDv4), so repeated
        executions of the same authorised decision are distinguishable.
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

    crewAI registers before-tool-call hooks globally via
    :func:`crewai.hooks.tool_hooks.register_before_tool_call_hook` (they are
    stored in a module-level list, not on individual agents).  To avoid
    false positives from unrelated hooks, a crew is considered guarded only
    when at least one registered hook carries the ``_guardrail_context``
    marker attached by :func:`make_guardrail_hook`.

    Returns list of findings (empty = all guarded).
    """
    findings = []

    agents = getattr(crew_or_agent, "agents", [crew_or_agent])

    # Pull the global hook list.  Import lazily so the module also works
    # standalone (outside a crewAI installation) for testing.
    try:
        from crewai.hooks.tool_hooks import get_before_tool_call_hooks
    except ImportError:  # pragma: no cover - exercised outside crewAI install
            def get_before_tool_call_hooks():  # type: ignore[no-redef]
                return []

    global_hooks = get_before_tool_call_hooks()
    has_guardrail_hook = any(
        getattr(hook, "_guardrail_context", None) is not None
        for hook in global_hooks
    )

    if has_guardrail_hook:
        return findings

    for agent in agents:
        agent_name = getattr(agent, "role", getattr(agent, "name", "unknown"))
        findings.append(
            {
                "severity": "CRITICAL",
                "pattern": "AS-GUARDRAIL-MISS-001",
                "agent": agent_name,
                "message": (
                    f"Agent '{agent_name}' has no registered GuardrailProvider "
                    "(no global before_tool_call hook with _guardrail_context marker)"
                ),
                "remediation": (
                    "Register a GuardrailProvider via "
                    "register_before_tool_call_hook(make_guardrail_hook(ctx))"
                ),
            }
        )

    return findings
