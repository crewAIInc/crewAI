"""Tests for event context management."""

import pytest

from crewai.events.event_context import (
    SCOPE_ENDING_EVENTS,
    SCOPE_STARTING_EVENTS,
    VALID_EVENT_PAIRS,
    EmptyStackError,
    EventPairingError,
    MismatchBehavior,
    StackDepthExceededError,
    _event_context_config,
    EventContextConfig,
    get_current_parent_id,
    get_enclosing_parent_id,
    get_last_event_id,
    get_triggering_event_id,
    handle_empty_pop,
    handle_mismatch,
    pop_event_scope,
    push_event_scope,
    reset_last_event_id,
    set_last_event_id,
    set_triggering_event_id,
    triggered_by_scope,
)


class TestStackOperations:
    """Tests for stack push/pop operations."""

    def test_empty_stack_returns_none(self) -> None:
        assert get_current_parent_id() is None
        assert get_enclosing_parent_id() is None

    def test_push_and_get_parent(self) -> None:
        push_event_scope("event-1", "task_started")
        assert get_current_parent_id() == "event-1"

    def test_nested_push(self) -> None:
        push_event_scope("event-1", "crew_kickoff_started")
        push_event_scope("event-2", "task_started")
        assert get_current_parent_id() == "event-2"
        assert get_enclosing_parent_id() == "event-1"

    def test_pop_restores_parent(self) -> None:
        push_event_scope("event-1", "crew_kickoff_started")
        push_event_scope("event-2", "task_started")
        popped = pop_event_scope()
        assert popped == ("event-2", "task_started")
        assert get_current_parent_id() == "event-1"

    def test_pop_empty_stack_returns_none(self) -> None:
        assert pop_event_scope() is None


class TestStackDepthLimit:
    """Tests for stack depth limit."""

    def test_depth_limit_exceeded_raises(self) -> None:
        _event_context_config.set(EventContextConfig(max_stack_depth=3))

        push_event_scope("event-1", "type-1")
        push_event_scope("event-2", "type-2")
        push_event_scope("event-3", "type-3")

        with pytest.raises(StackDepthExceededError):
            push_event_scope("event-4", "type-4")


class TestMismatchHandling:
    """Tests for mismatch behavior."""

    def test_handle_mismatch_raises_when_configured(self) -> None:
        _event_context_config.set(
            EventContextConfig(mismatch_behavior=MismatchBehavior.RAISE)
        )

        with pytest.raises(EventPairingError):
            handle_mismatch("task_completed", "llm_call_started", "task_started")

    def test_handle_empty_pop_raises_when_configured(self) -> None:
        _event_context_config.set(
            EventContextConfig(empty_pop_behavior=MismatchBehavior.RAISE)
        )

        with pytest.raises(EmptyStackError):
            handle_empty_pop("task_completed")


class TestEventTypeSets:
    """Tests for event type set completeness."""

    def test_all_ending_events_have_pairs(self) -> None:
        for ending_event in SCOPE_ENDING_EVENTS:
            assert ending_event in VALID_EVENT_PAIRS

    def test_all_pairs_reference_starting_events(self) -> None:
        for ending_event, starting_event in VALID_EVENT_PAIRS.items():
            assert starting_event in SCOPE_STARTING_EVENTS

    def test_starting_and_ending_are_disjoint(self) -> None:
        overlap = SCOPE_STARTING_EVENTS & SCOPE_ENDING_EVENTS
        assert not overlap


class TestLastEventIdTracking:
    """Tests for linear chain event ID tracking."""

    def test_initial_last_event_id_is_none(self) -> None:
        reset_last_event_id()
        assert get_last_event_id() is None

    def test_set_and_get_last_event_id(self) -> None:
        reset_last_event_id()
        set_last_event_id("event-123")
        assert get_last_event_id() == "event-123"

    def test_reset_clears_last_event_id(self) -> None:
        set_last_event_id("event-123")
        reset_last_event_id()
        assert get_last_event_id() is None

    def test_overwrite_last_event_id(self) -> None:
        reset_last_event_id()
        set_last_event_id("event-1")
        set_last_event_id("event-2")
        assert get_last_event_id() == "event-2"


class TestTriggeringEventIdTracking:
    """Tests for causal chain event ID tracking."""

    def test_initial_triggering_event_id_is_none(self) -> None:
        set_triggering_event_id(None)
        assert get_triggering_event_id() is None

    def test_set_and_get_triggering_event_id(self) -> None:
        set_triggering_event_id("trigger-123")
        assert get_triggering_event_id() == "trigger-123"
        set_triggering_event_id(None)

    def test_set_none_clears_triggering_event_id(self) -> None:
        set_triggering_event_id("trigger-123")
        set_triggering_event_id(None)
        assert get_triggering_event_id() is None


class TestTriggeredByScope:
    """Tests for triggered_by_scope context manager."""

    def test_scope_sets_triggering_id(self) -> None:
        set_triggering_event_id(None)
        with triggered_by_scope("trigger-456"):
            assert get_triggering_event_id() == "trigger-456"

    def test_scope_restores_previous_value(self) -> None:
        set_triggering_event_id(None)
        with triggered_by_scope("trigger-456"):
            pass
        assert get_triggering_event_id() is None

    def test_nested_scopes(self) -> None:
        set_triggering_event_id(None)
        with triggered_by_scope("outer"):
            assert get_triggering_event_id() == "outer"
            with triggered_by_scope("inner"):
                assert get_triggering_event_id() == "inner"
            assert get_triggering_event_id() == "outer"
        assert get_triggering_event_id() is None

    def test_scope_restores_on_exception(self) -> None:
        set_triggering_event_id(None)
        try:
            with triggered_by_scope("trigger-789"):
                raise ValueError("test error")
        except ValueError:
            pass
        assert get_triggering_event_id() is None