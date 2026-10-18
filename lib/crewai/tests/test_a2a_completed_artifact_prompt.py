from types import SimpleNamespace

from a2a.types import Message, Part, Role, TaskState, TextPart

from crewai.a2a.wrapper import (
    _augment_prompt_with_a2a,
    _handle_task_completion,
)


def test_completed_artifact_is_in_follow_up_prompt():
    artifact_text = "turn 1: echo: ping-external-peer"
    endpoint = "https://example.com/a2a"

    # Conversation history contains only the user's request.
    user_message = Message(
        role=Role.user,
        message_id="test-message",
        parts=[Part(root=TextPart(text="ping-external-peer"))],
    )

    a2a_result = {
        "status": TaskState.completed,
        "result": artifact_text,
        "history": [user_message],
    }

    trusted_result, _, _, remote_notice = _handle_task_completion(
        a2a_result=a2a_result,
        task=SimpleNamespace(config={}),
        task_id_config=None,
        reference_task_ids=[],
        agent_config=SimpleNamespace(trust_remote_completion_status=False),
        turn_num=0,
    )

    prompt, _, _ = _augment_prompt_with_a2a(
        a2a_agents=[SimpleNamespace(endpoint=endpoint)],
        task_description="Complete the original task.",
        agent_cards={
            endpoint: {
                "url": endpoint,
                "description": "Test A2A agent",
                "skills": [],
            }
        },
        conversation_history=[user_message],
        remote_status_notice=remote_notice,
    )

    assert trusted_result is None
    expected_block = (
        f"\n<REMOTE_AGENT_RESULT>\n{artifact_text}\n</REMOTE_AGENT_RESULT>\n"
    )
    assert expected_block in prompt
    assert "ping-external-peer" in prompt


def test_trusted_completed_artifact_returns_directly():
    artifact_text = "Remote task completed successfully"

    result, _, _, remote_notice = _handle_task_completion(
        a2a_result={
            "status": TaskState.completed,
            "result": artifact_text,
            "history": [],
        },
        task=SimpleNamespace(config={}),
        task_id_config=None,
        reference_task_ids=[],
        agent_config=SimpleNamespace(trust_remote_completion_status=True),
        turn_num=0,
    )

    assert result == artifact_text
    assert result == artifact_text
    assert "\n<REMOTE_AGENT_RESULT>\n" not in remote_notice
    assert "</REMOTE_AGENT_RESULT>" not in remote_notice


def test_empty_completed_artifact_is_not_injected():
    result, _, _, remote_notice = _handle_task_completion(
        a2a_result={
            "status": TaskState.completed,
            "result": "",
            "history": [],
        },
        task=SimpleNamespace(config={}),
        task_id_config=None,
        reference_task_ids=[],
        agent_config=SimpleNamespace(trust_remote_completion_status=False),
        turn_num=0,
    )

    assert result is None
    assert "\n<REMOTE_AGENT_RESULT>\n" not in remote_notice
    assert "</REMOTE_AGENT_RESULT>" not in remote_notice
