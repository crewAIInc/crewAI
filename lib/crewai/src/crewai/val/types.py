"""Core types for Verifiable Audit Log (VAL) support."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import Any, Protocol

from crewai.utilities.serialization import to_serializable


VAL_SPEC_VERSION = "val/v1"


@dataclass(frozen=True, slots=True)
class VALRecord:
    """Represents one attested event in a tamper-evident hash chain."""

    spec_version: str
    sequence: int
    topic_id: str | None
    event_type: str
    event_id: str
    event_timestamp: str
    previous_hash: str | None
    record_hash: str
    payload: dict[str, Any]
    attestor: str
    attestation_id: str | None = None

    def with_attestation(self, attestation_id: str | None) -> VALRecord:
        """Return a copy with attestation metadata applied."""
        return replace(self, attestation_id=attestation_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable mapping."""
        return {
            "spec_version": self.spec_version,
            "sequence": self.sequence,
            "topic_id": self.topic_id,
            "event_type": self.event_type,
            "event_id": self.event_id,
            "event_timestamp": self.event_timestamp,
            "previous_hash": self.previous_hash,
            "record_hash": self.record_hash,
            "payload": self.payload,
            "attestor": self.attestor,
            "attestation_id": self.attestation_id,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> VALRecord:
        """Build a ``VALRecord`` from any mapping-like payload."""
        return cls(
            spec_version=str(data["spec_version"]),
            sequence=int(data["sequence"]),
            topic_id=(str(data["topic_id"]) if data.get("topic_id") is not None else None),
            event_type=str(data["event_type"]),
            event_id=str(data["event_id"]),
            event_timestamp=str(data["event_timestamp"]),
            previous_hash=(
                str(data["previous_hash"])
                if data.get("previous_hash") is not None
                else None
            ),
            record_hash=str(data["record_hash"]),
            payload=to_json_dict(data.get("payload", {})),
            attestor=str(data["attestor"]),
            attestation_id=(
                str(data["attestation_id"])
                if data.get("attestation_id") is not None
                else None
            ),
        )


class VALAttestor(Protocol):
    """Protocol for external attestation backends."""

    ledger: str

    def attest(self, record: VALRecord) -> str | None:
        """Persist an attestation record and return an external id."""


def to_json_dict(data: Any) -> dict[str, Any]:
    """Normalize arbitrary data into a JSON-compatible dictionary."""
    serialized = to_serializable(data)
    if isinstance(serialized, dict):
        return serialized
    return {"value": serialized}


def compute_record_hash(
    payload: Mapping[str, Any], previous_hash: str | None
) -> str:
    """Compute deterministic hash for a VAL record payload."""
    canonical = json.dumps(
        to_serializable({"previous_hash": previous_hash, "payload": payload}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def verify_val_chain(records: list[VALRecord | Mapping[str, Any]]) -> bool:
    """Verify record linkage and payload hashes for a VAL chain."""
    expected_previous: str | None = None
    for raw_record in records:
        record = (
            raw_record
            if isinstance(raw_record, VALRecord)
            else VALRecord.from_mapping(raw_record)
        )
        if record.previous_hash != expected_previous:
            return False

        expected_hash = compute_record_hash(record.payload, record.previous_hash)
        if record.record_hash != expected_hash:
            return False

        expected_previous = record.record_hash
    return True
