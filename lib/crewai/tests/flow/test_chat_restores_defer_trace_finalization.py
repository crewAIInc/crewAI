"""Regression test for #8069.

``Flow.chat()`` turns ``defer_trace_finalization`` on for the duration of the
terminal session and restores it afterwards. Finalisation ran before the
restore, so a finalizer that raised left the temporary value behind on a reused
Flow. No model call or provider credentials are needed.
"""

from unittest.mock import patch

import pytest

from crewai.flow import ConversationConfig, Flow


@ConversationConfig(defer_trace_finalization=False)
class _ChatFlow(Flow):
    pass


def _chat(flow: Flow) -> None:
    with patch.object(
        type(flow), "finalize_session_traces", side_effect=RuntimeError("boom")
    ):
        flow.chat(input_fn=lambda _prompt: "quit", output_fn=lambda _line: None)


def test_flag_is_restored_when_finalization_raises() -> None:
    flow = _ChatFlow()
    assert flow.defer_trace_finalization is False

    with pytest.raises(RuntimeError, match="boom"):
        _chat(flow)

    assert flow.defer_trace_finalization is False, (
        "the temporary defer_trace_finalization survived a failing finalizer"
    )


def test_flag_is_restored_when_finalization_succeeds() -> None:
    flow = _ChatFlow()
    flow.chat(input_fn=lambda _prompt: "quit", output_fn=lambda _line: None)
    assert flow.defer_trace_finalization is False


def test_finalization_still_runs() -> None:
    """The fix must not swallow or skip finalisation itself."""
    flow = _ChatFlow()
    with patch.object(type(flow), "finalize_session_traces") as finalize:
        flow.chat(input_fn=lambda _prompt: "quit", output_fn=lambda _line: None)
    assert finalize.called