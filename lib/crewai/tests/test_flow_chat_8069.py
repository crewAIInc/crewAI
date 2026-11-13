"""Tests for crewAIInc/crewAI#8069 — Flow.chat() must restore
``defer_trace_finalization`` even when ``finalize_session_traces()`` raises.

Issue repro (paraphrased from the report): chat() sets the flag, its cleanup
finalizes traces *before* restoring the flag, so a finalization error leaves
a reused Flow instance stuck with the temporary setting.

Baseline (crewai 1.15.40): test_config_restored_when_finalize_raises fails —
the attribute is still True after the failed session.
Patched: all tests pass, and the finalization error still propagates.
"""

from unittest.mock import patch

import pytest

from crewai.flow import ConversationConfig, Flow


@ConversationConfig(defer_trace_finalization=False)
class ChatFlow(Flow):
    pass


def _run_quit(flow):
    flow.chat(input_fn=lambda _: "quit", output_fn=lambda _: None)


def test_config_restored_when_finalize_raises():
    flow = ChatFlow()
    with patch.object(
        flow, "finalize_session_traces", side_effect=RuntimeError("boom")
    ):
        with pytest.raises(RuntimeError):
            _run_quit(flow)
    assert flow.defer_trace_finalization is False


def test_finalize_exception_still_propagates():
    flow = ChatFlow()
    with patch.object(
        flow, "finalize_session_traces", side_effect=RuntimeError("boom")
    ):
        with pytest.raises(RuntimeError, match="boom"):
            _run_quit(flow)


def test_config_restored_on_clean_exit():
    flow = ChatFlow()
    with patch.object(flow, "finalize_session_traces", return_value=None):
        _run_quit(flow)
    assert flow.defer_trace_finalization is False


def test_previous_true_value_is_restored_not_hardcoded():
    flow = ChatFlow()
    flow.defer_trace_finalization = True
    with patch.object(
        flow, "finalize_session_traces", side_effect=RuntimeError("boom")
    ):
        with pytest.raises(RuntimeError):
            _run_quit(flow)
    assert flow.defer_trace_finalization is True
