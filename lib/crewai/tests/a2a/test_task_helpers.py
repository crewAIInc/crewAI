"""Tests for A2A task_helpers.py — specifically artifact text reassembly."""
from unittest.mock import MagicMock, patch

from a2a.types import Part, TaskState, TextPart

from crewai.a2a.task_helpers import process_task_state


def _make_text_part(text: str) -> Part:
    return Part(root=TextPart(text=text))


@patch("crewai.a2a.task_helpers.crewai_event_bus.emit")
def test_result_parts_are_concatenated_without_separator(mock_emit):
    """Per the A2A spec, artifact parts sent with append=True must be
    joined with NO separator. Previously this used ' '.join(...), which
    corrupted text by inserting spaces between streamed chunks."""
    a2a_task = MagicMock()
    a2a_task.status.state = TaskState.completed
    a2a_task.status.message.parts = [
        _make_text_part("Hel"),
        _make_text_part("lo, "),
        _make_text_part("world"),
    ]
    a2a_task.status.message.message_id = "msg-1"
    a2a_task.history = None
    a2a_task.context_id = "ctx-1"

    result = process_task_state(
        a2a_task=a2a_task,
        new_messages=[],
        agent_card=MagicMock(),
        turn_number=1,
        is_multiturn=False,
        agent_role=None,
        endpoint=None,
        a2a_agent_name=None,
        from_task=None,
        from_agent=None,
        is_final=True,
    )

    print("RESULT KEYS:", result)  # delete this line

    assert result is not None
    assert result["result"] == "Hello, world"