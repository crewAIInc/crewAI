"""Tests for tenant_id / user_id fields on MemoryRecord.

PR #1 of per-tenant memory isolation: the fields exist on the Pydantic model
with safe defaults. Backends do not yet persist or filter on them -- that lands
in PR #2 when the StorageBackend protocol gains the required tenant_id kwarg.

These tests pin the additive schema change and the default values that keep
single-tenant deployments working unchanged.
"""

from __future__ import annotations

import pytest

from crewai.memory.types import MemoryRecord


class TestMemoryRecordTenantFields:
    def test_default_tenant_id_is_underscore_default(self) -> None:
        rec = MemoryRecord(content="hello")
        assert rec.tenant_id == "_default"

    def test_default_user_id_is_none(self) -> None:
        rec = MemoryRecord(content="hello")
        assert rec.user_id is None

    def test_tenant_id_round_trips_via_model_dump(self) -> None:
        rec = MemoryRecord(content="hello", tenant_id="customer_42", user_id="alice")
        dumped = rec.model_dump()
        assert dumped["tenant_id"] == "customer_42"
        assert dumped["user_id"] == "alice"
        restored = MemoryRecord.model_validate(dumped)
        assert restored.tenant_id == "customer_42"
        assert restored.user_id == "alice"

    def test_tenant_id_round_trips_via_json(self) -> None:
        rec = MemoryRecord(content="hello", tenant_id="customer_42")
        restored = MemoryRecord.model_validate_json(rec.model_dump_json())
        assert restored.tenant_id == "customer_42"

    def test_legacy_record_without_tenant_id_loads_as_default(self) -> None:
        # Simulates reading an old row from disk that pre-dates this PR.
        # The default_factory must fire so the loaded record is non-leaking.
        legacy_payload = {"content": "old row from before this PR"}
        rec = MemoryRecord.model_validate(legacy_payload)
        assert rec.tenant_id == "_default"
        assert rec.user_id is None

    def test_tenant_id_must_be_string(self) -> None:
        # tenant_id is non-nullable str -- enforcement at the type level.
        with pytest.raises(Exception):
            MemoryRecord(content="x", tenant_id=None)  # type: ignore[arg-type]

    def test_user_id_accepts_none_and_string(self) -> None:
        assert MemoryRecord(content="x", user_id=None).user_id is None
        assert MemoryRecord(content="x", user_id="alice").user_id == "alice"
