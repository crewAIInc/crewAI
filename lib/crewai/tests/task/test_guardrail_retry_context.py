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
            GuardrailResult(success=False, error="GRF check failed: stale receipt"),
            GuardrailResult(success=True, result="corrected result"),
        ]
    )
    # Controlled outcomes exercise Task retry handling without calling a CLI.
    monkeypatch.setattr(task_module, "process_guardrail", process)
    retry_calls: list[str | None] = []
    retry_results = iter(["second rejected answer", "corrected result"])
    agent = SimpleNamespace(
        role="test-agent",
        verbose=False,
        last_messages=[],
        execute_task=Mock(
            side_effect=lambda task, context, tools: (
                retry_calls.append(context) or next(retry_results)
            )
        ),
    )
    task = Task(
        description="Use the upstream decision",
        expected_output="a supported answer",
        guardrail_max_retries=2,
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
        guardrail=lambda _: (True, "unused"),  # Enables the patched guardrail path.
    )

    assert result.raw == "corrected result"
    assert len(retry_calls) == 2
    assert "Upstream evidence: service code is A-17" in retry_calls[0]
    assert "GRF check failed: missing evidence" in retry_calls[0]
    assert "unsupported answer" in retry_calls[0]
    assert "Upstream evidence: service code is A-17" in retry_calls[1]
    assert retry_calls[1].count("Upstream evidence: service code is A-17") == 1
    assert "GRF check failed: stale receipt" in retry_calls[1]
    assert "second rejected answer" in retry_calls[1]
    assert "missing evidence" not in retry_calls[1]
    assert "unsupported answer" not in retry_calls[1]
    assert [call.kwargs["output"].raw for call in process.call_args_list] == [
        "unsupported answer",
        "second rejected answer",
        "corrected result",
    ]


def test_async_guardrail_retry_keeps_upstream_context_and_adds_acceptance_feedback(
    monkeypatch,
) -> None:
    import crewai.task as task_module

    process = Mock(
        side_effect=[
            GuardrailResult(success=False, error="GRF check failed: missing evidence"),
            GuardrailResult(success=False, error="GRF check failed: stale receipt"),
            GuardrailResult(success=True, result="corrected result"),
        ]
    )
    # Controlled outcomes exercise Task retry handling without calling a CLI.
    monkeypatch.setattr(task_module, "process_guardrail", process)
    retry_calls: list[str | None] = []
    retry_results = iter(["second rejected answer", "corrected result"])

    async def execute_task(task, context, tools):
        retry_calls.append(context)
        return next(retry_results)

    agent = SimpleNamespace(
        role="test-agent",
        verbose=False,
        last_messages=[],
        aexecute_task=AsyncMock(side_effect=execute_task),
    )
    task = Task(
        description="Use the upstream decision",
        expected_output="a supported answer",
        guardrail_max_retries=2,
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
            guardrail=lambda _: (True, "unused"),  # Enables the patched guardrail path.
        )
    )

    assert result.raw == "corrected result"
    assert len(retry_calls) == 2
    assert "Upstream evidence: service code is A-17" in retry_calls[0]
    assert "GRF check failed: missing evidence" in retry_calls[0]
    assert "unsupported answer" in retry_calls[0]
    assert "Upstream evidence: service code is A-17" in retry_calls[1]
    assert retry_calls[1].count("Upstream evidence: service code is A-17") == 1
    assert "GRF check failed: stale receipt" in retry_calls[1]
    assert "second rejected answer" in retry_calls[1]
    assert "missing evidence" not in retry_calls[1]
    assert "unsupported answer" not in retry_calls[1]
    assert [call.kwargs["output"].raw for call in process.call_args_list] == [
        "unsupported answer",
        "second rejected answer",
        "corrected result",
    ]
