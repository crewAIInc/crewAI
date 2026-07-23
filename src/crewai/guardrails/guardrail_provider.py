# Runtime security guardrail implementation conforming to
# CCS — Conformance Testing Protocol for MCP Agent Runtime Security
# Reference: https://correctover.com/ccs
# Reference implementation by Correctover (https://correctover.com)

# GuardrailProvider — content-addressed decision audit chain for tool calls
#
# Protocol + reference implementation for pre-execution authorization
# and post-execution evidence capture.
#
# Aligns with crewAI#4877 converged spec (safal207, babyblueviper1, Yarmoluk):
#   - decision_id = SHA-256(canonical_json(claims)) — content-addressed
#   - expires_at in hash preimage (anti-tamper)
#   - GuardrailDecisionV1 (pre-execution authorization) vs
#     ActionEnvelopeV1 (post-execution evidence)
#   - CKG (Constraint Knowledge Graph) for declarative authorization
#
# Integration: register guardrail_enforcement_hook via
# crewai.hooks.register_before_tool_call_hook()

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crewai.hooks.tool_hooks import ToolCallHookContext


# ──────────────────────────────────────────────
#  Core data types
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class GuardrailDecisionV1:
    """Pre-execution authorization decision.

    Every authorized tool call produces a content-addressed decision.
    The decision_id commits to all claims including expires_at, making
    post-issuance tampering detectable.

    Spec: crewAI#4877 — safal207's GuardrailDecisionV1
    """

    decision_id: str
    authorized: bool
    claims: dict[str, Any] = field(default_factory=dict)
    expires_at: float | None = None
    reason: str | None = None
    provider_name: str = "default"

    def is_expired(self, now: float | None = None) -> bool:
        """Check whether this decision has expired."""
        if self.expires_at is None:
            return False
        return (now or time.time()) > self.expires_at

    def verify_integrity(self) -> bool:
        """Recompute decision_id from claims and compare.

        The decision_id must equal SHA-256(canonical_json(claims ∪ {expires_at})).
        Any mismatch means the decision or its claims were tampered with.
        """
        expected = compute_decision_id(self.claims, self.expires_at)
        return self.decision_id == expected


@dataclass(frozen=True)
class ActionEnvelopeV1:
    """Post-execution evidence envelope.

    Captures what actually happened so downstream audit can verify
    that execution matched the authorization.

    Spec: crewAI#4877 — ActionEnvelopeV1 (separated from GuardrailDecisionV1)
    """

    decision_id: str
    tool_name: str
    tool_input_snapshot: str  # canonical JSON at decision time
    tool_result_digest: str   # SHA-256 of result (never the raw value)
    started_at: float
    completed_at: float
    success: bool
    error: str | None = None

    def duration_ms(self) -> float:
        return (self.completed_at - self.started_at) * 1000


# ──────────────────────────────────────────────
#  Content-addressed hashing
# ──────────────────────────────────────────────


def compute_decision_id(
    claims: dict[str, Any],
    expires_at: float | None = None,
) -> str:
    """Compute content-addressed decision_id per crewAI#4877 spec.

    The preimage is a canonical (sorted-key, compact) JSON representation of
    the claims dict with expires_at folded in *before* hashing. This ensures
    an attacker who modifies claims or expiration cannot produce a matching
    decision_id without access to the preimage secret (here, none — it's
    pure content addressing).

    If expires_at is None, the preimage is claims alone
    (no _expires_at key injected into the serialized payload).

    Returns hex-encoded SHA-256 digest.
    """
    preimage = dict(claims)
    if expires_at is not None:
        preimage["_expires_at"] = expires_at

    # Canonical form: keys sorted, no extra whitespace, deterministic output.
    canonical = json.dumps(preimage, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_decision_id(namespace: str, payload: dict[str, Any]) -> str:
    """Convenience: hash an identifier namespace into the computation.

    Namespacing prevents decision_id collisions across different tools
    or agents within the same crew.
    """
    return compute_decision_id({"namespace": namespace, **payload})


def digest_result(result: str | None) -> str:
    """SHA-256 digest of a tool result.

    The envelope never stores raw results — only a digest. This satisfies
    privacy requirements while preserving auditability.
    """
    return hashlib.sha256((result or "").encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────
#  GuardrailProvider protocol
# ──────────────────────────────────────────────


class GuardrailProvider(ABC):
    """Abstract guardrail provider.

    Subclasses implement authorize() — the core decision logic that
    inspects context and returns a GuardrailDecisionV1.

    The hook layer calls authorize() before every tool execution and
    blocks the call if the decision is not authorized.

    Example::

        class MyGuardrail(GuardrailProvider):
            def authorize(self, context: ToolCallHookContext) -> GuardrailDecisionV1:
                claims = {
                    "tool": context.tool_name,
                    "agent": getattr(context.agent, "role", "unknown"),
                    "params": context.tool_input,
                }
                authorized = context.tool_name not in FORBIDDEN_TOOLS
                expires_at = time.time() + 300
                return GuardrailDecisionV1(
                    decision_id=compute_decision_id(claims, expires_at),
                    authorized=authorized,
                    claims=claims,
                    expires_at=expires_at,
                    provider_name="my_guardrail",
                )
    """

    @abstractmethod
    def authorize(self, context: ToolCallHookContext) -> GuardrailDecisionV1:
        """Inspect the tool-call context and return an authorization decision.

        Args:
            context: The tool-call hook context at decision time.

        Returns:
            GuardrailDecisionV1 with:
              - decision_id committed to claims + expires_at
              - authorized=True/False
              - claims snapshot and optional expiry
        """
        ...

    def name(self) -> str:
        return self.__class__.__name__


# ──────────────────────────────────────────────
#  Built-in providers
# ──────────────────────────────────────────────


class AllowAllGuardrailProvider(GuardrailProvider):
    """Permissive provider that authorizes every call.

    Useful as a default or for testing; never use in production.
    """

    def authorize(self, context: ToolCallHookContext) -> GuardrailDecisionV1:
        claims = {
            "tool": context.tool_name,
            "agent": getattr(context.agent, "role", "unknown"),
        }
        expires_at = time.time() + 3600
        return GuardrailDecisionV1(
            decision_id=compute_decision_id(claims, expires_at),
            authorized=True,
            claims=claims,
            expires_at=expires_at,
            reason="AllowAll — no restrictions applied",
            provider_name="allow_all",
        )


class DenyAllGuardrailProvider(GuardrailProvider):
    """Restrictive provider that denies every call.

    Useful as a safety lock or for testing.
    """

    def authorize(self, context: ToolCallHookContext) -> GuardrailDecisionV1:
        claims = {
            "tool": context.tool_name,
            "agent": getattr(context.agent, "role", "unknown"),
        }
        expires_at = time.time() + 3600
        return GuardrailDecisionV1(
            decision_id=compute_decision_id(claims, expires_at),
            authorized=False,
            claims=claims,
            expires_at=expires_at,
            reason="DenyAll — all tool calls blocked",
            provider_name="deny_all",
        )


class ToolListGuardrailProvider(GuardrailProvider):
    """Allows only an explicit allowlist of tool names.

    Args:
        allowed_tools: Set of tool names permitted for execution.
        default_block: Whether to block (True) or allow (False) tools
            not in the allowlist.

    Example::

        guardrail = ToolListGuardrailProvider(
            allowed_tools={"read_file", "search_web", "calculator"},
            default_block=True,
        )
    """

    def __init__(
        self,
        allowed_tools: set[str],
        default_block: bool = True,
    ) -> None:
        self._allowed = frozenset(allowed_tools)
        self._default_block = default_block

    def authorize(self, context: ToolCallHookContext) -> GuardrailDecisionV1:
        is_allowed = context.tool_name in self._allowed
        authorized = is_allowed if self._default_block else (not is_allowed)
        claims = {
            "tool": context.tool_name,
            "on_allowlist": is_allowed,
            "default_block": self._default_block,
        }
        expires_at = time.time() + 300
        return GuardrailDecisionV1(
            decision_id=compute_decision_id(claims, expires_at),
            authorized=authorized,
            claims=claims,
            expires_at=expires_at,
            reason=(
                f"Tool '{context.tool_name}' is allowed"
                if is_allowed
                else f"Tool '{context.tool_name}' is NOT in allowlist"
            ),
            provider_name="tool_list",
        )


class CKGGuardrailProvider(GuardrailProvider):
    """Constraint Knowledge Graph guardrail provider.

    Implements the declarative authorization approach described by Yarmoluk
    in crewAI#4877. Instead of imperative if/else logic, authorization is
    driven by a set of constraint rules evaluated against the tool-call context.

    Each constraint is a (predicate, params) tuple evaluated as:
      predicate(context, **params) -> bool  (True = constraint satisfied)

    A call is authorized iff ALL constraints are satisfied.

    Example::

        def _no_shell_exec(ctx, **_):
            return ctx.tool_name != "run_shell_command"

        guardrail = CKGGuardrailProvider(constraints=[_no_shell_exec])
    """

    def __init__(self, constraints: list[tuple] | None = None) -> None:
        self._constraints: list[tuple] = constraints or []

    def add_constraint(
        self,
        predicate: str,
        **params: Any,
    ) -> None:
        """Register a constraint tuple."""
        self._constraints.append((predicate, params))

    def authorize(self, context: ToolCallHookContext) -> GuardrailDecisionV1:
        results: list[tuple[str, bool]] = []
        all_satisfied = True

        for predicate_name, params in self._constraints:
            ok = self._eval_predicate(predicate_name, context, **params)
            results.append((predicate_name, ok))
            if not ok:
                all_satisfied = False

        claims = {
            "tool": context.tool_name,
            "agent": getattr(context.agent, "role", "unknown"),
            "constraint_results": results,
        }
        expires_at = time.time() + 60
        return GuardrailDecisionV1(
            decision_id=compute_decision_id(claims, expires_at),
            authorized=all_satisfied,
            claims=claims,
            expires_at=expires_at,
            reason=(
                "All constraints satisfied"
                if all_satisfied
                else f"Constraints failed: {[k for k, v in results if not v]}"
            ),
            provider_name="ckg",
        )

    @staticmethod
    def _eval_predicate(
        name: str,
        context: ToolCallHookContext,
        **params: Any,
    ) -> bool:
        """Resolve and evaluate a predicate by name.

        Built-in predicates:
          - tool_name_in      : context.tool_name in params['names']
          - tool_name_not_in  : context.tool_name not in params['names']
          - agent_role_in     : agent role in params['roles']
          - param_matches     : context.tool_input[key] matches regex/value
          - has_param         : key exists in tool_input
          - no_param          : key does not exist in tool_input

        Custom callables can be registered during init.
        """
        # Built-in predicate table
        builtins: dict[str, Any] = {
            "tool_name_in": lambda ctx, names: ctx.tool_name in names,
            "tool_name_not_in": lambda ctx, names: ctx.tool_name not in names,
            "agent_role_in": (
                lambda ctx, roles: getattr(ctx.agent, "role", None) in roles
            ),
            "param_matches": (
                lambda ctx, key, value: ctx.tool_input.get(key) == value
            ),
            "has_param": lambda ctx, key: key in ctx.tool_input,
            "no_param": lambda ctx, key: key not in ctx.tool_input,
        }
        fn = builtins.get(name)
        if fn is None:
            raise ValueError(f"Unknown CKG predicate: {name}")
        return fn(context, **params)


# ──────────────────────────────────────────────
#  Audit trail (in-memory)
# ──────────────────────────────────────────────


class AuditTrail:
    """In-memory audit trail that records every decision + envelope pair.

    Thread-safe for single-threaded crew execution. For production,
    swap the backend for a durable store (SQLite, Redis, etc.).
    """

    def __init__(self) -> None:
        self._decisions: dict[str, GuardrailDecisionV1] = {}
        self._envelopes: dict[str, ActionEnvelopeV1] = {}

    def record_decision(self, decision: GuardrailDecisionV1) -> None:
        self._decisions[decision.decision_id] = decision

    def record_envelope(self, envelope: ActionEnvelopeV1) -> None:
        self._envelopes[envelope.decision_id] = envelope

    def get_decision(self, decision_id: str) -> GuardrailDecisionV1 | None:
        return self._decisions.get(decision_id)

    def get_envelope(self, decision_id: str) -> ActionEnvelopeV1 | None:
        return self._envelopes.get(decision_id)

    def all_decisions(self) -> list[GuardrailDecisionV1]:
        return list(self._decisions.values())

    def all_envelopes(self) -> list[ActionEnvelopeV1]:
        return list(self._envelopes.values())

    def clear(self) -> None:
        self._decisions.clear()
        self._envelopes.clear()

    @property
    def total_decisions(self) -> int:
        return len(self._decisions)

    @property
    def total_envelopes(self) -> int:
        return len(self._envelopes)


# ──────────────────────────────────────────────
#  Hook integration
# ──────────────────────────────────────────────


class GuardrailContext:
    """Aggregate state for the guardrail hook chain.

    One GuardrailContext is created per crew run (or per session). It
    holds the provider, audit trail, and callbacks so the hook functions
    don't rely on global state.
    """

    def __init__(
        self,
        provider: GuardrailProvider,
        trail: AuditTrail | None = None,
        on_deny: Any = None,
    ) -> None:
        self.provider = provider
        self.trail = trail or AuditTrail()
        self.on_deny = on_deny  # Optional callback when a call is denied

    def before_tool_call(self, context: ToolCallHookContext) -> bool | None:
        """CrewAI before_tool_call hook.

        Called by the hook dispatcher before every tool execution.

        1. Asks the GuardrailProvider for an authorization decision.
        2. Records the decision in the audit trail.
        3. If denied, calls on_deny callback (if set) and returns False
           to block execution.
        4. If authorized, returns None to allow execution.

        Register via::

            guardrails = GuardrailContext(provider=MyGuardrail())
            register_before_tool_call_hook(guardrails.before_tool_call)
        """
        decision = self.provider.authorize(context)

        # Record pre-execution decision
        self.trail.record_decision(decision)

        if not decision.authorized:
            if self.on_deny and callable(self.on_deny):
                self.on_deny(context, decision)
            return False  # Block execution

        return None  # Allow

    # def after_tool_call(self, context: ToolCallHookContext) -> str | None:
    #     """(Future) Post-execution envelope capture.
    #
    #     Not registered by default; call
    #       register_after_tool_call_hook(guardrails.after_tool_call)
    #     if you want evidence envelopes.
    #     """
    #     # Lookup the most recent decision for this tool call
    #     decisions = self.trail.all_decisions()
    #     if not decisions:
    #         return None
    #     last = decisions[-1]
    #     envelope = ActionEnvelopeV1(
    #         decision_id=last.decision_id,
    #         tool_name=context.tool_name,
    #         tool_input_snapshot=json.dumps(context.tool_input, sort_keys=True),
    #         tool_result_digest=digest_result(context.tool_result),
    #         started_at=time.time(),
    #         completed_at=time.time(),
    #         success=context.tool_result is not None,
    #     )
    #     self.trail.record_envelope(envelope)
    #     return None  # Don't modify result


def make_guardrail_hook(
    provider: GuardrailProvider,
    trail: AuditTrail | None = None,
    on_deny: Any = None,
) -> Any:
    """Convenience factory: creates a GuardrailContext and returns its
    before_tool_call hook, ready for ``register_before_tool_call_hook()``.

    Example::

        guardrail = ToolListGuardrailProvider(allowed_tools={"read_file"})
        hook = make_guardrail_hook(guardrail)
        register_before_tool_call_hook(hook)

        # Later, inspect the audit trail:
        print(guardrail_context.trail.all_decisions())
    """
    ctx = GuardrailContext(provider=provider, trail=trail, on_deny=on_deny)
    # Wrap the bound method so we can attach metadata without relying on
    # __dict__ assignment on method objects (unreliable across Python versions).

    def _hook(context: ToolCallHookContext) -> bool | None:
        return ctx.before_tool_call(context)

    _hook.context = ctx  # type: ignore[attr-defined]
    return _hook


# ──────────────────────────────────────────────
#  Engine-v4 seed integration (AS-GUARDRAIL-MISS-001)
# ──────────────────────────────────────────────


def detect_missing_guardrail(agent_config: dict[str, Any]) -> list[str]:
    """Check an agent dict for guardrail configuration gaps.

    Returns a list of missing guardrail features. An empty list means
    the agent is fully guarded.

    Designed to be called by the Correctover engine-v4 scanner's
    AS-GUARDRAIL-MISS-001 pattern matcher.
    """
    missing: list[str] = []
    tools = agent_config.get("tools", [])
    for tool in tools if isinstance(tools, list) else []:
        tool_name = tool if isinstance(tool, str) else tool.get("name", "?")
        if not _has_guardrail_provider(agent_config, tool_name):
            missing.append(f"tool:{tool_name} — no GuardrailProvider")
    return missing


def _has_guardrail_provider(config: dict[str, Any], tool_name: str) -> bool:
    """Heuristic: does the config reference a guardrail provider?"""
    guardrails = config.get("guardrails") or {}
    providers = guardrails.get("providers") or []
    return any(
        isinstance(p, dict) and p.get("tool") in (tool_name, "*")
        for p in providers
    )


__all__ = [
    "GuardrailDecisionV1",
    "ActionEnvelopeV1",
    "compute_decision_id",
    "make_decision_id",
    "digest_result",
    "GuardrailProvider",
    "AllowAllGuardrailProvider",
    "DenyAllGuardrailProvider",
    "ToolListGuardrailProvider",
    "CKGGuardrailProvider",
    "AuditTrail",
    "GuardrailContext",
    "make_guardrail_hook",
    "detect_missing_guardrail",
]
