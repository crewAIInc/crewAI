"""Unit tests for LoopDetector and loop detection integration.

Tests cover:
- LoopDetector class: tracking, detection, message generation, configuration
- ToolCallRecord equality and hashing
- Edge cases: empty history, threshold boundaries, window overflow
- Integration with CrewAgentExecutor._record_and_check_loop
- LoopDetectedEvent creation
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from crewai.agents.loop_detector import LoopDetector, ToolCallRecord
from crewai.events.types.loop_events import LoopDetectedEvent


# ---------------------------------------------------------------------------
# ToolCallRecord
# ---------------------------------------------------------------------------

class TestToolCallRecord:
    """Tests for ToolCallRecord model."""

    def test_equality_same_name_and_args(self):
        a = ToolCallRecord(tool_name="search", tool_args='{"q": "hello"}')
        b = ToolCallRecord(tool_name="search", tool_args='{"q": "hello"}')
        assert a == b

    def test_inequality_different_name(self):
        a = ToolCallRecord(tool_name="search", tool_args='{"q": "hello"}')
        b = ToolCallRecord(tool_name="fetch", tool_args='{"q": "hello"}')
        assert a != b

    def test_inequality_different_args(self):
        a = ToolCallRecord(tool_name="search", tool_args='{"q": "hello"}')
        b = ToolCallRecord(tool_name="search", tool_args='{"q": "world"}')
        assert a != b

    def test_hash_same_records(self):
        a = ToolCallRecord(tool_name="search", tool_args='{"q": "hello"}')
        b = ToolCallRecord(tool_name="search", tool_args='{"q": "hello"}')
        assert hash(a) == hash(b)

    def test_hash_different_records(self):
        a = ToolCallRecord(tool_name="search", tool_args='{"q": "hello"}')
        b = ToolCallRecord(tool_name="search", tool_args='{"q": "world"}')
        # Different records should (almost certainly) have different hashes
        assert hash(a) != hash(b)

    def test_equality_not_implemented_for_other_types(self):
        a = ToolCallRecord(tool_name="search", tool_args="{}")
        assert a != "not a record"
        assert a.__eq__("not a record") is NotImplemented


# ---------------------------------------------------------------------------
# LoopDetector — Initialization
# ---------------------------------------------------------------------------

class TestLoopDetectorInit:
    """Tests for LoopDetector initialization and defaults."""

    def test_default_values(self):
        ld = LoopDetector()
        assert ld.window_size == 5
        assert ld.repetition_threshold == 3
        assert ld.on_loop == "inject_reflection"

    def test_custom_values(self):
        ld = LoopDetector(window_size=10, repetition_threshold=4, on_loop="stop")
        assert ld.window_size == 10
        assert ld.repetition_threshold == 4
        assert ld.on_loop == "stop"

    def test_callable_on_loop(self):
        def my_cb(detector: LoopDetector) -> str:
            return "custom"

        ld = LoopDetector(on_loop=my_cb)
        assert callable(ld.on_loop)

    def test_window_size_minimum(self):
        with pytest.raises(Exception):
            LoopDetector(window_size=1)

    def test_repetition_threshold_minimum(self):
        with pytest.raises(Exception):
            LoopDetector(repetition_threshold=1)


# ---------------------------------------------------------------------------
# LoopDetector — Argument Normalization
# ---------------------------------------------------------------------------

class TestNormalizeArgs:
    """Tests for LoopDetector._normalize_args static method."""

    def test_dict_sorted_keys(self):
        result = LoopDetector._normalize_args({"b": 2, "a": 1})
        assert result == '{"a": 1, "b": 2}'

    def test_json_string_normalized(self):
        result = LoopDetector._normalize_args('{"b": 2, "a": 1}')
        assert result == '{"a": 1, "b": 2}'

    def test_plain_string_stripped(self):
        result = LoopDetector._normalize_args("  hello world  ")
        assert result == "hello world"

    def test_invalid_json_string(self):
        result = LoopDetector._normalize_args("not json")
        assert result == "not json"

    def test_empty_dict(self):
        result = LoopDetector._normalize_args({})
        assert result == "{}"

    def test_empty_string(self):
        result = LoopDetector._normalize_args("")
        assert result == ""


# ---------------------------------------------------------------------------
# LoopDetector — Recording and Detection
# ---------------------------------------------------------------------------

class TestLoopDetection:
    """Tests for loop recording and detection logic."""

    def test_no_loop_below_threshold(self):
        ld = LoopDetector(repetition_threshold=3)
        ld.record_tool_call("search", {"q": "test"})
        ld.record_tool_call("search", {"q": "test"})
        assert ld.is_loop_detected() is False

    def test_loop_at_threshold(self):
        ld = LoopDetector(repetition_threshold=3)
        for _ in range(3):
            ld.record_tool_call("search", {"q": "test"})
        assert ld.is_loop_detected() is True

    def test_loop_above_threshold(self):
        ld = LoopDetector(repetition_threshold=3)
        for _ in range(5):
            ld.record_tool_call("search", {"q": "test"})
        assert ld.is_loop_detected() is True

    def test_no_loop_with_different_tools(self):
        ld = LoopDetector(repetition_threshold=3)
        ld.record_tool_call("search", {"q": "test"})
        ld.record_tool_call("fetch", {"url": "http://example.com"})
        ld.record_tool_call("read", {"file": "data.txt"})
        assert ld.is_loop_detected() is False

    def test_no_loop_with_different_args(self):
        ld = LoopDetector(repetition_threshold=3)
        ld.record_tool_call("search", {"q": "test1"})
        ld.record_tool_call("search", {"q": "test2"})
        ld.record_tool_call("search", {"q": "test3"})
        assert ld.is_loop_detected() is False

    def test_loop_with_mixed_calls(self):
        """Loop detected even if other calls are interspersed, as long as
        threshold is met within the window."""
        ld = LoopDetector(window_size=5, repetition_threshold=3)
        ld.record_tool_call("search", {"q": "test"})
        ld.record_tool_call("fetch", {"url": "http://example.com"})
        ld.record_tool_call("search", {"q": "test"})
        ld.record_tool_call("search", {"q": "test"})
        assert ld.is_loop_detected() is True

    def test_window_overflow_removes_old_calls(self):
        """Old calls slide out of the window, potentially clearing the loop."""
        ld = LoopDetector(window_size=3, repetition_threshold=3)
        ld.record_tool_call("search", {"q": "test"})
        ld.record_tool_call("search", {"q": "test"})
        # Now add a different call — this pushes first "search" out when we add
        # one more different call
        ld.record_tool_call("fetch", {"url": "http://example.com"})
        # Only 2 "search" in window at this point, not 3
        assert ld.is_loop_detected() is False

    def test_loop_with_string_args(self):
        ld = LoopDetector(repetition_threshold=3)
        for _ in range(3):
            ld.record_tool_call("search", '{"q": "test"}')
        assert ld.is_loop_detected() is True

    def test_loop_with_equivalent_json_args(self):
        """Args with different key order should be normalized and treated as equal."""
        ld = LoopDetector(repetition_threshold=3)
        ld.record_tool_call("search", '{"b": 2, "a": 1}')
        ld.record_tool_call("search", '{"a": 1, "b": 2}')
        ld.record_tool_call("search", {"a": 1, "b": 2})
        assert ld.is_loop_detected() is True

    def test_empty_history(self):
        ld = LoopDetector()
        assert ld.is_loop_detected() is False

    def test_single_call(self):
        ld = LoopDetector()
        ld.record_tool_call("search", {"q": "test"})
        assert ld.is_loop_detected() is False


# ---------------------------------------------------------------------------
# LoopDetector — Reset
# ---------------------------------------------------------------------------

class TestLoopDetectorReset:
    """Tests for LoopDetector.reset()."""

    def test_reset_clears_history(self):
        ld = LoopDetector(repetition_threshold=3)
        for _ in range(3):
            ld.record_tool_call("search", {"q": "test"})
        assert ld.is_loop_detected() is True
        ld.reset()
        assert ld.is_loop_detected() is False

    def test_reset_allows_fresh_tracking(self):
        ld = LoopDetector(repetition_threshold=3)
        for _ in range(3):
            ld.record_tool_call("search", {"q": "test"})
        ld.reset()
        # Now record again — should not trigger until threshold
        ld.record_tool_call("search", {"q": "test"})
        ld.record_tool_call("search", {"q": "test"})
        assert ld.is_loop_detected() is False


# ---------------------------------------------------------------------------
# LoopDetector — Messages and Tool Info
# ---------------------------------------------------------------------------

class TestLoopDetectorMessages:
    """Tests for get_loop_message and get_repeated_tool_info."""

    def test_get_loop_message_with_callable(self):
        def my_callback(detector: LoopDetector) -> str:
            return "Custom loop message"

        ld = LoopDetector(on_loop=my_callback)
        assert ld.get_loop_message() == "Custom loop message"

    def test_get_loop_message_inject_reflection_returns_empty(self):
        ld = LoopDetector(on_loop="inject_reflection")
        assert ld.get_loop_message() == ""

    def test_get_loop_message_stop_returns_empty(self):
        ld = LoopDetector(on_loop="stop")
        assert ld.get_loop_message() == ""

    def test_get_repeated_tool_info_with_loop(self):
        ld = LoopDetector(repetition_threshold=3)
        for _ in range(3):
            ld.record_tool_call("search", {"q": "test"})
        info = ld.get_repeated_tool_info()
        assert info is not None
        assert "search" in info

    def test_get_repeated_tool_info_no_loop(self):
        ld = LoopDetector(repetition_threshold=3)
        ld.record_tool_call("search", {"q": "test"})
        assert ld.get_repeated_tool_info() is None

    def test_get_repeated_tool_info_empty_history(self):
        ld = LoopDetector()
        assert ld.get_repeated_tool_info() is None

    def test_callback_receives_detector_instance(self):
        received = {}

        def my_callback(detector: LoopDetector) -> str:
            received["detector"] = detector
            return "msg"

        ld = LoopDetector(on_loop=my_callback)
        ld.get_loop_message()
        assert received["detector"] is ld


# ---------------------------------------------------------------------------
# LoopDetectedEvent
# ---------------------------------------------------------------------------

class TestLoopDetectedEvent:
    """Tests for the LoopDetectedEvent model."""

    def test_event_creation(self):
        event = LoopDetectedEvent(
            agent_role="Researcher",
            agent_id="agent-1",
            task_id="task-1",
            repeated_tool="search({\"q\": \"test\"})",
            action_taken="inject_reflection",
            iteration=5,
        )
        assert event.agent_role == "Researcher"
        assert event.repeated_tool == 'search({"q": "test"})'
        assert event.action_taken == "inject_reflection"
        assert event.iteration == 5
        assert event.type == "loop_detected"

    def test_event_with_agent_fingerprint(self):
        mock_agent = MagicMock()
        mock_agent.fingerprint.uuid_str = "fp-123"
        mock_agent.fingerprint.metadata = {"key": "value"}

        event = LoopDetectedEvent(
            agent_role="Researcher",
            action_taken="stop",
            iteration=3,
            agent=mock_agent,
        )
        assert event.source_fingerprint == "fp-123"
        assert event.fingerprint_metadata == {"key": "value"}

    def test_event_without_agent(self):
        event = LoopDetectedEvent(
            agent_role="Researcher",
            action_taken="inject_reflection",
            iteration=1,
        )
        assert event.agent is None
        assert event.agent_id is None


# ---------------------------------------------------------------------------
# Integration: CrewAgentExecutor._record_and_check_loop
# ---------------------------------------------------------------------------

class TestRecordAndCheckLoop:
    """Tests for the _record_and_check_loop helper on CrewAgentExecutor."""

    @pytest.fixture
    def _make_executor(self):
        """Factory that builds a minimal CrewAgentExecutor-like object.

        We import the real class and patch its __init__ so we can
        test _record_and_check_loop without setting up the full executor.
        """
        from crewai.agents.crew_agent_executor import CrewAgentExecutor

        def _factory(
            loop_detector: LoopDetector | None = None,
            verbose: bool = False,
        ) -> CrewAgentExecutor:
            executor = object.__new__(CrewAgentExecutor)
            # Minimal attributes required by _record_and_check_loop
            executor.loop_detector = loop_detector

            mock_agent = Mock()
            mock_agent.role = "Test Agent"
            mock_agent.id = "agent-1"
            mock_agent.verbose = verbose
            executor.agent = mock_agent

            mock_task = Mock()
            mock_task.id = "task-1"
            executor.task = mock_task

            executor.iterations = 0
            executor.messages = []

            mock_printer = Mock()
            executor._printer = mock_printer

            mock_i18n = Mock()
            mock_i18n.slice.return_value = (
                "WARNING: You appear to be repeating the same action ({repeated_tool}). "
                "You have called it {count} times."
            )
            executor._i18n = mock_i18n

            mock_llm = Mock()
            executor.llm = mock_llm
            executor.callbacks = []

            return executor

        return _factory

    def test_no_loop_detector_returns_none(self, _make_executor):
        executor = _make_executor(loop_detector=None)
        result = executor._record_and_check_loop("search", {"q": "test"})
        assert result is None

    def test_no_loop_detected_returns_none(self, _make_executor):
        ld = LoopDetector(repetition_threshold=3)
        executor = _make_executor(loop_detector=ld)
        # Only 1 call — no loop
        result = executor._record_and_check_loop("search", {"q": "test"})
        assert result is None

    @patch("crewai.agents.crew_agent_executor.crewai_event_bus")
    def test_inject_reflection_appends_message(self, mock_bus, _make_executor):
        ld = LoopDetector(repetition_threshold=3, on_loop="inject_reflection")
        executor = _make_executor(loop_detector=ld)

        # Record 2 calls before the 3rd (which triggers detection)
        ld.record_tool_call("search", {"q": "test"})
        ld.record_tool_call("search", {"q": "test"})

        result = executor._record_and_check_loop("search", {"q": "test"})

        assert result is None  # inject_reflection does NOT stop
        assert len(executor.messages) == 1
        assert "repeating" in executor.messages[0]["content"].lower()
        # Event should have been emitted
        mock_bus.emit.assert_called_once()

    @patch("crewai.agents.crew_agent_executor.crewai_event_bus")
    @patch("crewai.agents.crew_agent_executor.handle_max_iterations_exceeded")
    def test_stop_returns_agent_finish(
        self, mock_handle_max, mock_bus, _make_executor
    ):
        from crewai.agents.parser import AgentFinish

        mock_handle_max.return_value = AgentFinish(
            thought="forced", output="Stopped", text="Stopped"
        )

        ld = LoopDetector(repetition_threshold=3, on_loop="stop")
        executor = _make_executor(loop_detector=ld)

        ld.record_tool_call("search", {"q": "test"})
        ld.record_tool_call("search", {"q": "test"})

        result = executor._record_and_check_loop("search", {"q": "test"})

        assert result is not None
        assert isinstance(result, AgentFinish)
        mock_handle_max.assert_called_once()
        mock_bus.emit.assert_called_once()

    @patch("crewai.agents.crew_agent_executor.crewai_event_bus")
    def test_callable_on_loop_injects_callback_message(
        self, mock_bus, _make_executor
    ):
        def custom_callback(detector: LoopDetector) -> str:
            return "CUSTOM: Stop repeating yourself!"

        ld = LoopDetector(repetition_threshold=3, on_loop=custom_callback)
        executor = _make_executor(loop_detector=ld)

        ld.record_tool_call("search", {"q": "test"})
        ld.record_tool_call("search", {"q": "test"})

        result = executor._record_and_check_loop("search", {"q": "test"})

        assert result is None
        assert len(executor.messages) == 1
        assert "CUSTOM: Stop repeating yourself!" in executor.messages[0]["content"]

    @patch("crewai.agents.crew_agent_executor.crewai_event_bus")
    def test_detection_resets_after_intervention(self, mock_bus, _make_executor):
        ld = LoopDetector(repetition_threshold=3, on_loop="inject_reflection")
        executor = _make_executor(loop_detector=ld)

        ld.record_tool_call("search", {"q": "test"})
        ld.record_tool_call("search", {"q": "test"})

        # 3rd call triggers loop
        executor._record_and_check_loop("search", {"q": "test"})

        # After intervention, detector should be reset
        assert ld.is_loop_detected() is False

    @patch("crewai.agents.crew_agent_executor.crewai_event_bus")
    def test_verbose_prints_message(self, mock_bus, _make_executor):
        ld = LoopDetector(repetition_threshold=3, on_loop="inject_reflection")
        executor = _make_executor(loop_detector=ld, verbose=True)

        ld.record_tool_call("search", {"q": "test"})
        ld.record_tool_call("search", {"q": "test"})

        executor._record_and_check_loop("search", {"q": "test"})

        executor._printer.print.assert_called_once()
        call_kwargs = executor._printer.print.call_args
        assert "Loop detected" in call_kwargs.kwargs.get(
            "content", call_kwargs[1].get("content", "")
        )

    @patch("crewai.agents.crew_agent_executor.crewai_event_bus")
    def test_event_contains_correct_data(self, mock_bus, _make_executor):
        ld = LoopDetector(repetition_threshold=3, on_loop="inject_reflection")
        executor = _make_executor(loop_detector=ld)

        ld.record_tool_call("search", {"q": "test"})
        ld.record_tool_call("search", {"q": "test"})

        executor._record_and_check_loop("search", {"q": "test"})

        event = mock_bus.emit.call_args[0][1]
        assert isinstance(event, LoopDetectedEvent)
        assert event.agent_role == "Test Agent"
        assert event.action_taken == "inject_reflection"
        assert event.iteration == 0  # executor.iterations was 0


# ---------------------------------------------------------------------------
# Public API import
# ---------------------------------------------------------------------------

class TestPublicImport:
    """Verify LoopDetector is importable from the public crewai package."""

    def test_import_from_crewai(self):
        from crewai import LoopDetector as LD

        assert LD is LoopDetector

    def test_in_all(self):
        import crewai

        assert "LoopDetector" in crewai.__all__
