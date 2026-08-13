"""Tests for OSS execution uuid creation and nesting inheritance."""

from __future__ import annotations

import pytest

from crewai.execution import (
    begin_execution,
    clear_execution_uuid,
    end_execution,
    ensure_execution_uuid,
    execution_uuid_scope,
    get_execution_uuid,
    set_execution_uuid,
)


@pytest.fixture(autouse=True)
def _clear_execution_uuid() -> None:
    clear_execution_uuid()
    yield
    clear_execution_uuid()


def test_ensure_creates_when_empty() -> None:
    assert get_execution_uuid() is None
    first = ensure_execution_uuid()
    second = ensure_execution_uuid()
    assert first
    assert first == second


def test_ensure_does_not_overwrite_existing() -> None:
    set_execution_uuid("enterprise-kickoff-id")
    assert ensure_execution_uuid("should-not-win") == "enterprise-kickoff-id"


def test_set_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        set_execution_uuid("")


def test_execution_uuid_scope_creates_and_clears() -> None:
    with execution_uuid_scope() as value:
        assert value
        assert get_execution_uuid() == value
    assert get_execution_uuid() is None


def test_execution_uuid_scope_inherits_without_clearing_parent() -> None:
    set_execution_uuid("parent")
    with execution_uuid_scope() as value:
        assert value == "parent"
    assert get_execution_uuid() == "parent"


def test_execution_uuid_scope_force_overrides() -> None:
    set_execution_uuid("parent")
    with execution_uuid_scope("celery-id", force=True) as value:
        assert value == "celery-id"
        assert get_execution_uuid() == "celery-id"
    assert get_execution_uuid() == "parent"


def test_nested_execution_inherits_and_only_owner_clears() -> None:
    parent_token = begin_execution()
    parent_id = get_execution_uuid()
    child_token = begin_execution()

    assert parent_id
    assert child_token is None
    assert get_execution_uuid() == parent_id

    end_execution(child_token)
    assert get_execution_uuid() == parent_id

    end_execution(parent_token)
    assert get_execution_uuid() is None


def test_two_sequential_outer_runs_get_distinct_uuids() -> None:
    first_token = begin_execution()
    first_id = get_execution_uuid()
    end_execution(first_token)
    second_token = begin_execution()
    second_id = get_execution_uuid()
    end_execution(second_token)

    assert first_id != second_id


def test_flow_kickoff_creates_and_clears_execution_uuid() -> None:
    from crewai.flow.flow import Flow, start

    seen: dict[str, str | None] = {}

    class ProbeFlow(Flow):
        @start()
        def begin(self) -> str:
            seen["during"] = get_execution_uuid()
            return "ok"

    flow = ProbeFlow()
    assert get_execution_uuid() is None
    flow.kickoff()
    assert seen["during"]
    assert get_execution_uuid() is None


def test_flow_kickoff_inherits_enterprise_execution_uuid() -> None:
    from crewai.flow.flow import Flow, start

    seen: dict[str, str | None] = {}

    class ProbeFlow(Flow):
        @start()
        def begin(self) -> str:
            seen["during"] = get_execution_uuid()
            return "ok"

    with execution_uuid_scope("celery-kickoff-id", force=True):
        ProbeFlow().kickoff()
        assert seen["during"] == "celery-kickoff-id"
        # Owner was enterprise scope, not the flow — still set here.
        assert get_execution_uuid() == "celery-kickoff-id"

    assert get_execution_uuid() is None
