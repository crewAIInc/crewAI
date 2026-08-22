"""Action Gate & Cryptographic Action Ledger Module for CrewAI.

Provides deterministic execution boundaries, simulation fallbacks, and
append-only SHA-256 hash-chained action ledgers for CrewAI agent tool executions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
import functools
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, ClassVar


try:
    import fcntl  # POSIX advisory file locking.

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - Windows has no fcntl.
    _HAS_FCNTL = False

import datetime


# Argument keys matching these patterns are redacted before being logged or
# hashed into the ledger. The real, unredacted values are still passed to the
# guarded function -- redaction only affects what is written to the audit
# trail, never the actual tool execution.
_SENSITIVE_KEY_RE = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|apikey|credential|auth|private[_-]?key|access[_-]?key)",
    re.IGNORECASE,
)
_REDACTED = "***REDACTED***"


class ToolTier(str, Enum):
    READ = "read"
    WRITE_IDEMPOTENT = "write_idempotent"
    WRITE_MUTATING = "write_mutating"
    DESTRUCTIVE = "destructive"


class Disposition(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    SIMULATE = "simulate"


class LedgerIntegrityError(RuntimeError):
    """Raised when the action ledger's hash chain fails verification.

    A broken chain means the ledger may have been tampered with (edited,
    reordered, or truncated). Recovery refuses to silently continue writing
    new entries on top of a chain it cannot trust.
    """


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    disposition: Disposition
    tier: ToolTier
    reason: str
    tool_name: str
    receipt_hash: str
    simulation_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "disposition": self.disposition.value,
            "tier": self.tier.value,
            "reason": self.reason,
            "tool_name": self.tool_name,
            "receipt_hash": self.receipt_hash,
            "simulation_mode": self.simulation_mode,
        }


def _redact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``arguments`` with sensitive-looking values redacted."""
    return {
        key: (_REDACTED if _SENSITIVE_KEY_RE.search(str(key)) else value)
        for key, value in arguments.items()
    }


class ActionLedger:
    """Append-only SHA-256 hash-chained action ledger for compliance audit readiness.

    The chain is verified (not merely trusted) on recovery: each entry's
    ``receipt_hash`` is recomputed from its own fields and its stored
    ``prev_hash`` is checked against the actual previous entry's hash. Writes
    are guarded by an advisory file lock and re-read the true tail of the
    ledger under that lock, so multiple processes appending to the same
    ledger file cannot fork the chain.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self, ledger_path: Path | str = "artifacts/action_ledger.jsonl") -> None:
        self.ledger_path = Path(ledger_path)
        self._thread_lock = threading.Lock()
        self.last_hash = self._recover_last_hash()

    @staticmethod
    def _entry_hash(entry: dict[str, Any]) -> str:
        canonical_args = json.dumps(entry["arguments"], sort_keys=True, default=str)
        payload = (
            f"{entry['prev_hash']}|{entry['timestamp']}|{entry['tool_name']}|{canonical_args}|"
            f"{entry['decision']['disposition']}|{entry.get('agent_id') or ''}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_entries(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        entries = []
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def verify_chain(self) -> tuple[bool, str | None]:
        """Re-walk the whole ledger and verify every hash link.

        Returns:
            (True, None) if the chain is intact, or (False, reason) at the
            first entry whose stored hash does not match its recomputed hash,
            or whose stored prev_hash does not match the true previous entry.
        """
        expected_prev = self.GENESIS_HASH
        for index, entry in enumerate(self._read_entries()):
            if entry.get("prev_hash") != expected_prev:
                return False, f"entry {index}: prev_hash does not match the preceding entry's receipt_hash"
            recomputed = self._entry_hash(entry)
            if entry.get("receipt_hash") != recomputed:
                return False, f"entry {index}: receipt_hash does not match its recomputed hash (tampered content)"
            expected_prev = entry["receipt_hash"]
        return True, None

    def _recover_last_hash(self) -> str:
        if not self.ledger_path.exists():
            return self.GENESIS_HASH
        ok, reason = self.verify_chain()
        if not ok:
            raise LedgerIntegrityError(
                f"Refusing to recover ledger {self.ledger_path}: hash chain verification failed ({reason})"
            )
        entries = self._read_entries()
        return entries[-1]["receipt_hash"] if entries else self.GENESIS_HASH

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Serialize ledger writes across threads and (on POSIX) processes."""
        with self._thread_lock:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            # Open in append mode so the lock target always exists; the file
            # descriptor is only used for locking here, writes happen below.
            with open(self.ledger_path, "a", encoding="utf-8") as lock_fp:
                if _HAS_FCNTL:
                    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if _HAS_FCNTL:
                        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    def record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        decision: GateDecision,
        agent_id: str | None = None,
    ) -> str:
        redacted_arguments = _redact_arguments(arguments)
        with self._locked():
            # Re-read the true current tail under the lock so concurrent
            # writers (other threads or other processes) can never both
            # build on the same prev_hash and fork the chain.
            entries = self._read_entries()
            prev_hash = entries[-1]["receipt_hash"] if entries else self.GENESIS_HASH

            timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            entry = {
                "timestamp": timestamp,
                "prev_hash": prev_hash,
                "tool_name": tool_name,
                "arguments": redacted_arguments,
                "decision": decision.to_dict(),
                "agent_id": agent_id,
            }
            entry["receipt_hash"] = self._entry_hash(entry)

            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")

            self.last_hash = entry["receipt_hash"]
            return entry["receipt_hash"]


class ActionGate:
    """Deterministic security gate enforcing 'never_equate_intent_to_approval: true'."""

    # Tier -> keyword patterns, most severe first. classify_tool() checks
    # tiers in THIS order (not dict-insertion order) so a tool name matching
    # keywords from more than one tier is always classified by its most
    # dangerous match, never its first alphabetically-earlier one.
    DEFAULT_TIER_RULES: ClassVar[dict[ToolTier, tuple[str, ...]]] = {
        ToolTier.DESTRUCTIVE: ("delete", "drop", "terminate", "destroy", "exec", "bash", "shell", "purge", "wipe", "kill"),
        ToolTier.WRITE_MUTATING: ("post", "create", "update", "send"),
        ToolTier.WRITE_IDEMPOTENT: ("upsert", "put", "set"),
        ToolTier.READ: ("search", "read", "get", "list", "scrape", "query"),
    }
    _TIER_ORDER = (ToolTier.DESTRUCTIVE, ToolTier.WRITE_MUTATING, ToolTier.WRITE_IDEMPOTENT, ToolTier.READ)

    def __init__(
        self,
        prove_token: str | None = None,
        ledger: ActionLedger | None = None,
        custom_tier_rules: dict[ToolTier, tuple[str, ...]] | None = None,
        allow_simulation: bool = True,
    ) -> None:
        self.prove_token = prove_token or os.environ.get("AAG_PROVE_TOKEN")
        self.ledger = ledger or ActionLedger()
        self.tier_rules: dict[ToolTier, tuple[str, ...]] = {
            tier: patterns for tier, patterns in self.DEFAULT_TIER_RULES.items()
        }
        if custom_tier_rules:
            for tier, patterns in custom_tier_rules.items():
                self.tier_rules[tier] = tuple(patterns)
        self.allow_simulation = allow_simulation

    def classify_tool(self, tool_name: str) -> ToolTier:
        """Classify a tool name by severity, most dangerous tier first.

        Matches whole underscore/hyphen/space-separated tokens (not raw
        substrings), so e.g. "target_list" does not false-match the "get"
        pattern, and a name containing both a destructive and a benign
        keyword (e.g. "search_and_terminate_instance") is always classified
        by its most dangerous match.
        """
        tokens = set(re.split(r"[^a-z0-9]+", tool_name.lower()))
        for tier in self._TIER_ORDER:
            patterns = self.tier_rules.get(tier, ())
            if tokens & set(patterns):
                return tier
        return ToolTier.WRITE_MUTATING

    def is_kill_switch_active(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").strip() in ("1", "true", "TRUE"):
            return True
        return Path("artifacts/KILL").exists()

    def _token_proven(self, candidate: str | None) -> bool:
        """Decide whether this call is authorized to bypass simulation/deny.

        Two cases:
          - No per-call token was presented (``candidate is None``): fall
            back to whether the gate itself was configured with a secret
            (constructor arg or AAG_PROVE_TOKEN) -- that configuration
            represents authorization established out-of-band at deployment
            time, matching this gate's original, intended behavior.
          - A per-call token WAS presented: it must match the gate's
            configured secret exactly, via constant-time comparison. A
            caller-supplied token is a claim to be verified, never an
            override that is trusted merely for being non-empty -- that was
            the actual vulnerability (any string authorized destructive
            operations regardless of the gate's real configured secret).
        """
        if candidate is None:
            return bool(self.prove_token)
        if not self.prove_token:
            return False
        return hmac.compare_digest(str(candidate), str(self.prove_token))

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        prove_token: str | None = None,
        agent_id: str | None = None,
    ) -> GateDecision:
        # 1. Check kill switch
        if self.is_kill_switch_active():
            decision = GateDecision(
                allowed=False,
                disposition=Disposition.DENY,
                tier=ToolTier.DESTRUCTIVE,
                reason="Kill-switch engaged (AAG_KILL_SWITCH active or artifacts/KILL present)",
                tool_name=tool_name,
                receipt_hash="",
                simulation_mode=False,
            )
            receipt_hash = self.ledger.record(tool_name, arguments, decision, agent_id)
            return GateDecision(**{**asdict(decision), "receipt_hash": receipt_hash})

        tier = self.classify_tool(tool_name)
        token_proven = self._token_proven(prove_token)

        # 2. READ: Always allowed
        if tier == ToolTier.READ:
            decision = GateDecision(
                allowed=True,
                disposition=Disposition.ALLOW,
                tier=tier,
                reason="Read-only operation allowed",
                tool_name=tool_name,
                receipt_hash="",
                simulation_mode=False,
            )
        # 3. WRITE_IDEMPOTENT & WRITE_MUTATING: Allowed if token proven, else fallback to simulation
        elif tier in (ToolTier.WRITE_IDEMPOTENT, ToolTier.WRITE_MUTATING):
            if token_proven:
                decision = GateDecision(
                    allowed=True,
                    disposition=Disposition.ALLOW,
                    tier=tier,
                    reason="Mutating write authorized with proven token",
                    tool_name=tool_name,
                    receipt_hash="",
                    simulation_mode=False,
                )
            elif self.allow_simulation:
                decision = GateDecision(
                    allowed=True,
                    disposition=Disposition.SIMULATE,
                    tier=tier,
                    reason="Unapproved mutating write downgraded to simulation mode",
                    tool_name=tool_name,
                    receipt_hash="",
                    simulation_mode=True,
                )
            else:
                decision = GateDecision(
                    allowed=False,
                    disposition=Disposition.DENY,
                    tier=tier,
                    reason="Mutating write rejected: missing proven token and simulation disabled",
                    tool_name=tool_name,
                    receipt_hash="",
                    simulation_mode=False,
                )
        # 4. DESTRUCTIVE: Hard deny without a proven token
        elif tier == ToolTier.DESTRUCTIVE:
            if token_proven:
                decision = GateDecision(
                    allowed=True,
                    disposition=Disposition.ALLOW,
                    tier=tier,
                    reason="Destructive operation explicitly authorized with proven token",
                    tool_name=tool_name,
                    receipt_hash="",
                    simulation_mode=False,
                )
            else:
                decision = GateDecision(
                    allowed=False,
                    disposition=Disposition.DENY,
                    tier=tier,
                    reason="Destructive operation blocked by ActionGate (never_equate_intent_to_approval)",
                    tool_name=tool_name,
                    receipt_hash="",
                    simulation_mode=False,
                )
        else:
            decision = GateDecision(
                allowed=False,
                disposition=Disposition.DENY,
                tier=ToolTier.DESTRUCTIVE,
                reason="Unknown tool classification",
                tool_name=tool_name,
                receipt_hash="",
                simulation_mode=False,
            )

        receipt_hash = self.ledger.record(tool_name, arguments, decision, agent_id)
        return GateDecision(
            allowed=decision.allowed,
            disposition=decision.disposition,
            tier=decision.tier,
            reason=decision.reason,
            tool_name=decision.tool_name,
            receipt_hash=receipt_hash,
            simulation_mode=decision.simulation_mode,
        )


class ActionBoundary:
    """Tool wrapper & boundary decorator for CrewAI tools."""

    def __init__(
        self,
        gate: ActionGate | None = None,
        prove_token: str | None = None,
        ledger_path: str = "artifacts/action_ledger.jsonl",
    ) -> None:
        self.gate = gate or ActionGate(prove_token=prove_token, ledger=ActionLedger(ledger_path))

    def guard(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to wrap any function or CrewAI tool execution with ActionGate policy."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tool_name = getattr(func, "name", func.__name__)
            decision = self.gate.evaluate(tool_name=tool_name, arguments=kwargs)

            if not decision.allowed:
                raise PermissionError(
                    f"ActionGate blocked tool '{tool_name}': {decision.reason} [receipt={decision.receipt_hash}]"
                )

            if decision.simulation_mode:
                return {
                    "simulation_mode": True,
                    "disposition": "simulate",
                    "tool": tool_name,
                    "receipt_hash": decision.receipt_hash,
                    "message": "Simulated tool execution (no state mutation occurred).",
                }

            return func(*args, **kwargs)

        return wrapper
