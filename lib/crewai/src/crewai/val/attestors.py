"""Attestor backends for VAL records."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import re
from threading import Lock

from crewai.utilities.paths import db_storage_path
from crewai.val.types import VALRecord


class InMemoryVALAttestor:
    """Simple in-memory attestor, useful for tests and local debugging."""

    ledger = "memory"

    def __init__(self) -> None:
        self.records: list[VALRecord] = []
        self._lock = Lock()

    def attest(self, record: VALRecord) -> str:
        """Store record in memory and return a synthetic attestation id."""
        with self._lock:
            self.records.append(record)
            return f"{self.ledger}:{len(self.records)}"


class JSONLVALAttestor:
    """Append-only JSONL attestor for durable local audit storage."""

    ledger = "jsonl"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_topic(cls, topic_id: str | None = None) -> JSONLVALAttestor:
        """Create a default attestor path scoped by topic id."""
        raw_name = topic_id or "default"
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_name)
        val_dir = Path(db_storage_path()) / "val"
        return cls(val_dir / f"{safe_name}.jsonl")

    def attest(self, record: VALRecord) -> str:
        """Write one record to JSONL and return file-based attestation id."""
        with self._lock:
            line = json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")
                handle.flush()
                os.fsync(handle.fileno())
        return f"{self.path}:{record.sequence}"


class CallableVALAttestor:
    """Adapter to integrate any external attestation client or SDK."""

    def __init__(
        self,
        fn: Callable[[VALRecord], str | None],
        *,
        ledger: str = "custom",
    ) -> None:
        self._fn = fn
        self.ledger = ledger

    def attest(self, record: VALRecord) -> str | None:
        """Delegate attestation to user-provided callback."""
        return self._fn(record)
