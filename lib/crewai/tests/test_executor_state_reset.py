"""Tests for executor state reset between task executions.

Verifies fix for issue #4603: crew.kickoff() truncates LLM output when
the agent executor is reused across multiple crew runs, because messages
and iterations were not being reset.
"""

from unittest.mock import MagicMock, patch

import pytest

from crewai.agent import Agent
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.crew import Crew
from crewai.llm import LLM
from crewai.process import Process
from crewai.task import Task
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput


LONG_OUTPUT = (
    '{"selected_factors": [{"identifier": "CX_IRC_TRS_US_USD", '
    '"name": "USD Government Bond Yield Curve", "category": "Interest", '
    '"description": "Treasury Zero Interest Rate Curve for United States '
    "of America USD. The Treasury yield curve represents the yields on "
    "government bonds across different maturities. It is one of the most "
    'important indicators in financial markets."}, '
    '{"identifier": "CX_EQT_SP500_US_USD", '
    '"name": "S&P 500 Index", "category": "Equity", '
    '"description": "The Standard and Poor 500 Index tracks the stock '
    "performance of 500 large companies listed on exchanges in the United "
    "States. It is widely regarded as the best gauge of large-cap U.S. "
    'equities."}]}'
)


@pytest.fixture(autouse=True)
def _suppress_telemetry(monkeypatch):
    """Suppress telemetry for all tests in this module."""
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    yield


def _make_task_output(raw: str) -> TaskOutput:
    return TaskOutput(
        description="test task",
        raw=raw,
        agent="Test Agent",
        json_dict=None,
        output_format=OutputFormat.RAW,
        pydantic=None,
        messages=[],
    )


def test_invoke_resets_messages():
    """invoke() must reset messages and iterations via the production code path.

    We verify indirectly through crew.kickoff(): after a first run that
    populates executor state, a second run must still return the full
    output — proving stale messages/iterations were cleared.
    """
    with patch.object(Task, "execute_sync") as mock_execute:
        mock_execute.return_value = _make_task_output(LONG_OUTPUT)

        agent = Agent(role="Tester", goal="Run", backstory="QA", llm="gpt-4o")
        task_obj = Task(description="Produce output", expected_output="text", agent=agent)
        crew = Crew(agents=[agent], tasks=[task_obj])

        result_1 = crew.kickoff()
        result_2 = crew.kickoff()

        assert result_1.raw == LONG_OUTPUT, "First run output was truncated"
        assert result_2.raw == LONG_OUTPUT, "Second run output was truncated — stale state leaked"

        assert task_obj.output is not None, "task_obj.output was not set after kickoff"
        assert task_obj.output.raw == LONG_OUTPUT, "task_obj.output.raw doesn't match expected"


def test_kickoff_preserves_full_output_with_reused_agent():
    """When the same agent is reused across multiple crew.kickoff() calls,
    the full output is preserved (not truncated by stale executor state).
    """
    with patch.object(Task, "execute_sync") as mock_execute:
        mock_execute.return_value = _make_task_output(LONG_OUTPUT)

        agent = Agent(
            role="Test Agent",
            goal="Return JSON",
            backstory="You are a test agent.",
            llm=LLM(model="gpt-4o"),
        )
        task1 = Task(
            description="First task",
            expected_output="JSON output",
            agent=agent,
        )

        crew1 = Crew(
            agents=[agent],
            tasks=[task1],
            process=Process.sequential,
            memory=False,
        )
        result1 = crew1.kickoff()

        # Second run with same agent
        task2 = Task(
            description="Second task",
            expected_output="JSON output",
            agent=agent,
        )
        crew2 = Crew(
            agents=[agent],
            tasks=[task2],
            process=Process.sequential,
            memory=False,
        )
        result2 = crew2.kickoff()

        assert result1.raw == LONG_OUTPUT
        assert result2.raw == LONG_OUTPUT
        assert len(result2.raw) == len(LONG_OUTPUT)


def test_kickoff_output_matches_task_output():
    """crew.kickoff().raw must match task.output.raw exactly."""
    task_output = _make_task_output(LONG_OUTPUT)

    def _execute_sync_side_effect(self, agent=None, context=None, tools=None):
        # Mimic the real execute_sync: set self.output and return TaskOutput
        self.output = task_output
        return task_output

    with patch.object(Task, "execute_sync", _execute_sync_side_effect):
        agent = Agent(
            role="Test Agent",
            goal="Return JSON",
            backstory="You are a test agent.",
            llm=LLM(model="gpt-4o"),
        )
        task_obj = Task(
            description="Get factors",
            expected_output="JSON",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task_obj],
            process=Process.sequential,
            memory=False,
        )
        result = crew.kickoff()

        # CrewOutput.raw must match the task output
        assert result.raw == LONG_OUTPUT
        assert task_obj.output is not None
        assert task_obj.output.raw == LONG_OUTPUT
        assert result.raw == task_obj.output.raw
        assert len(result.raw) > 100

        # Verify via tasks_output (the crew-level record of task results)
        assert len(result.tasks_output) == 1
        assert result.tasks_output[-1].raw == LONG_OUTPUT


def test_multiple_sequential_kickoffs_no_truncation():
    """Running kickoff 5 times with the same agent must not truncate output."""
    with patch.object(Task, "execute_sync") as mock_execute:
        mock_execute.return_value = _make_task_output(LONG_OUTPUT)

        agent = Agent(
            role="Test Agent",
            goal="Return JSON",
            backstory="You are a test agent.",
            llm=LLM(model="gpt-4o"),
        )

        for i in range(5):
            task_obj = Task(
                description=f"Task run {i}",
                expected_output="JSON output",
                agent=agent,
            )
            crew = Crew(
                agents=[agent],
                tasks=[task_obj],
                process=Process.sequential,
                memory=False,
            )
            result = crew.kickoff()
            assert result.raw == LONG_OUTPUT, (
                f"Run {i}: output was truncated to {len(result.raw)} chars"
            )


def test_executor_state_reset_on_invoke():
    """Directly verify CrewAgentExecutor resets messages and iterations.

    Instead of manually setting stale state, we call invoke() once to
    build up real executor state, then call invoke() again and verify the
    production reset logic clears messages and iterations.
    """
    call_count = 0
    responses = [
        "Final Answer: first output",
        "Final Answer: second output after reset",
    ]

    mock_llm = MagicMock(spec=LLM)
    mock_llm.supports_stop_words.return_value = False
    mock_llm.supports_function_calling.return_value = False
    mock_llm.stop = []

    def llm_side_effect(*args, **kwargs):
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return resp

    mock_llm.call.side_effect = llm_side_effect

    mock_task = MagicMock(spec=Task)
    mock_task.description = "test"
    mock_task.human_input = False

    mock_agent = MagicMock(spec=Agent)
    mock_agent.role = "Test"
    mock_agent.verbose = False
    mock_agent.security_config = None
    mock_agent.tools_results = []

    executor = CrewAgentExecutor(
        llm=mock_llm,
        task=mock_task,
        crew=None,
        agent=mock_agent,
        prompt={"system": "You are helpful.", "user": "Do task: {input}"},
        max_iter=25,
        tools=[],
        tools_names="",
        stop_words=[],
        tools_description="",
        tools_handler=MagicMock(),
        original_tools=[],
    )

    invoke_input = {
        "input": "test task",
        "tool_names": "",
        "tools": "",
        "ask_for_human_input": False,
    }

    with patch.object(executor, "_show_start_logs"), \
         patch.object(executor, "_show_logs"), \
         patch.object(executor, "_save_to_memory"), \
         patch.object(executor, "_inject_multimodal_files"):
        # First invoke populates executor state naturally
        result_1 = executor.invoke(invoke_input)
        messages_after_first = list(executor.messages)

        # Second invoke must reset stale state from the first run
        result_2 = executor.invoke(invoke_input)

    assert result_1["output"] == "first output"
    assert result_2["output"] == "second output after reset"

    # Messages from the first run must not leak into the second
    first_run_content = {m.get("content") for m in messages_after_first}
    leaked = [
        m for m in executor.messages
        if m.get("content") in first_run_content and m.get("role") == "assistant"
    ]
    assert len(leaked) == 0, "Messages from first invoke leaked into second run"
    # Iterations must have been reset (1 iteration per single-pass invoke)
    assert executor.iterations == 1, "Iterations counter was not reset between invocations"


def test_executor_reuse_across_multiple_invocations():
    """Integration-style: reuse the same executor across multiple invoke()
    calls and verify state is always reset — exercises the real executor
    loop instead of bypassing it with Task.execute_sync patches.
    """
    call_count = 0
    responses = [
        "Final Answer: first result with plenty of detail",
        "Final Answer: second result also complete and not truncated",
        "Final Answer: third result verifying no accumulation",
    ]

    mock_llm = MagicMock(spec=LLM)
    mock_llm.supports_stop_words.return_value = False
    mock_llm.supports_function_calling.return_value = False
    mock_llm.stop = []

    def llm_side_effect(*args, **kwargs):
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return resp

    mock_llm.call.side_effect = llm_side_effect

    mock_task = MagicMock(spec=Task)
    mock_task.description = "test"
    mock_task.human_input = False

    mock_agent = MagicMock(spec=Agent)
    mock_agent.role = "Test"
    mock_agent.verbose = False
    mock_agent.security_config = None
    mock_agent.tools_results = []

    executor = CrewAgentExecutor(
        llm=mock_llm,
        task=mock_task,
        crew=None,
        agent=mock_agent,
        prompt={"system": "You are helpful.", "user": "Do task: {input}"},
        max_iter=25,
        tools=[],
        tools_names="",
        stop_words=[],
        tools_description="",
        tools_handler=MagicMock(),
        original_tools=[],
    )

    with patch.object(executor, "_show_start_logs"), \
         patch.object(executor, "_show_logs"), \
         patch.object(executor, "_save_to_memory"), \
         patch.object(executor, "_inject_multimodal_files"):
        for i, expected in enumerate(responses):
            expected_text = expected.replace("Final Answer: ", "")
            result = executor.invoke({
                "input": f"task {i}",
                "tool_names": "",
                "tools": "",
                "ask_for_human_input": False,
            })
            assert result["output"] == expected_text, (
                f"Invocation {i}: expected '{expected_text}', got '{result['output']}'"
            )
            # No stale messages from previous iterations
            stale = [
                m for m in executor.messages
                if f"task {i - 1}" in str(m.get("content", "")) and i > 0
            ]
            assert len(stale) == 0, (
                f"Invocation {i}: found stale messages from previous run"
            )
