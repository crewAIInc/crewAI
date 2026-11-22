from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from crewai.task import Task
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.guardrail import GuardrailResult


def test_guardrail_retry_keeps_upstream_context_and_adds_acceptance_feedback(
    monkeypatch,
) -> None:
    import crewai.task as task_module

    process = Mock(
        side_effect=[
            GuardrailResult(success=False, error="GRF check failed: missing evidence"),
            GuardrailResult(success=True, result="corrected result"),
        ]
    )
    monkeypatch.setattr(task_module, "process_guardrail", process)
    retry_calls: list[str | None] = []
    agent = SimpleNamespace(
        role="test-agent",
        verbose=False,
        last_messages=[],
        execute_task=Mock(
            side_effect=lambda task, context, tools: (
                retry_calls.append(context) or "corrected result"
            )
        ),
    )
    task = Task(
        description="Use the upstream decision",
        expected_output="a supported answer",
        guardrail_max_retries=1,
    )
    task.prompt_context = "Upstream evidence: service code is A-17"
    output = TaskOutput(
        description=task.description,
        expected_output=task.expected_output,
        raw="unsupported answer",
        agent=agent.role,
    )

    result = task._invoke_guardrail_function(
        task_output=output,
        agent=agent,
        tools=[],
        guardrail=lambda _: (True, "unused"),
    )

    assert result.raw == "corrected result"
    assert len(retry_calls) == 1
    assert "Upstream evidence: service code is A-17" in retry_calls[0]
    assert "GRF check failed: missing evidence" in retry_calls[0]
    assert "unsupported answer" in retry_calls[0]


def test_async_guardrail_retry_keeps_upstream_context_and_adds_acceptance_feedback(
    monkeypatch,
) -> None:
    import crewai.task as task_module

    process = Mock(
        side_effect=[
            GuardrailResult(success=False, error="GRF check failed: missing evidence"),
            GuardrailResult(success=True, result="corrected result"),
        ]
    )
    monkeypatch.setattr(task_module, "process_guardrail", process)
    retry_calls: list[str | None] = []

    async def execute_task(task, context, tools):
        retry_calls.append(context)
        return "corrected result"

    agent = SimpleNamespace(
        role="test-agent",
        verbose=False,
        last_messages=[],
        aexecute_task=AsyncMock(side_effect=execute_task),
    )
    task = Task(
        description="Use the upstream decision",
        expected_output="a supported answer",
        guardrail_max_retries=1,
    )
    task.prompt_context = "Upstream evidence: service code is A-17"
    output = TaskOutput(
        description=task.description,
        expected_output=task.expected_output,
        raw="unsupported answer",
        agent=agent.role,
    )

    result = asyncio.run(
        task._ainvoke_guardrail_function(
            task_output=output,
            agent=agent,
            tools=[],
            guardrail=lambda _: (True, "unused"),
        )
    )

    assert result.raw == "corrected result"
    assert len(retry_calls) == 1
    assert "Upstream evidence: service code is A-17" in retry_calls[0]
    assert "GRF check failed: missing evidence" in retry_calls[0]
    assert "unsupported answer" in retry_calls[0]
