"""Tests for structured tool-failure signalling and the per-agent policy."""

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import (
    ToolFailureDetectedEvent,
    ToolUsageFinishedEvent,
)
from crewai.llm import LLM
from crewai.tools import BaseTool
from crewai.tools.tool_failure import (
    ToolExecutionFailedError,
    ToolFailure,
    ToolFailurePolicy,
    ToolFailureReason,
    ToolFailureRecord,
    detect_tool_failure,
    failure_from_exception,
    resolve_tool_failure_policy,
)


class SlackTool(BaseTool):
    """Mirrors an upstream API that answers 200 with an error body."""

    name: str = "slackbot_send_message"
    description: str = "Post a message to a Slack channel."

    def _run(self, channel: str) -> Any:
        return ToolFailure(
            message=f"Slack rejected the message to {channel}",
            code="channel_not_found",
        )


class WorkingTool(BaseTool):
    name: str = "echo"
    description: str = "Echo the input back."

    def _run(self, text: str) -> Any:
        return f"echoed: {text}"


class ScriptedLLM(LLM):
    """Emits a fixed sequence of ReAct steps without touching a provider."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "ScriptedLLM":
        return object.__new__(cls)

    def __init__(self, steps: list[str]) -> None:
        super().__init__(model="gpt-4o")
        self._steps = steps
        self._index = 0

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kw):  # noqa: ANN001, ANN003
        step = self._steps[min(self._index, len(self._steps) - 1)]
        self._index += 1
        return step

    def supports_function_calling(self) -> bool:
        return False


def _slack_steps() -> list[str]:
    call_step = (
        "Thought: posting\n"
        + "Action: slackbot_send_message\n"
        + 'Action Input: {"channel": "#joao-message"}'
    )
    return [
        call_step,
        "Thought: it failed\nFinal Answer: I could not post the message.",
    ]


def _build_crew(policy: ToolFailurePolicy | None = None, **task_kwargs: Any):
    agent_kwargs: dict[str, Any] = {
        "role": "Slack Messenger",
        "goal": "post a message",
        "backstory": "b",
        "llm": ScriptedLLM(_slack_steps()),
        "tools": [SlackTool()],
    }
    if policy is not None:
        agent_kwargs["tool_failure_policy"] = policy
    agent = Agent(**agent_kwargs)
    task = Task(
        description="post to slack",
        expected_output="confirmation",
        agent=agent,
        **task_kwargs,
    )
    return Crew(agents=[agent], tasks=[task]), agent


class TestToolFailureModel:
    def test_as_agent_message_includes_code(self) -> None:
        failure = ToolFailure(message="nope", code="channel_not_found")
        assert failure.as_agent_message() == "nope (code: channel_not_found)"

    def test_as_agent_message_without_code(self) -> None:
        assert ToolFailure(message="nope").as_agent_message() == "nope"

    def test_default_reason_is_tool_reported(self) -> None:
        assert ToolFailure(message="x").reason is ToolFailureReason.TOOL_REPORTED

    def test_detection_is_declarative_only(self) -> None:
        """A string that merely looks like an error is not a failure."""
        assert detect_tool_failure("Error: something went wrong") is None
        assert detect_tool_failure({"ok": False}) is None
        assert detect_tool_failure(ToolFailure(message="x")) is not None

    def test_failure_from_exception(self) -> None:
        failure = failure_from_exception(ValueError("bad input"))
        assert failure.reason is ToolFailureReason.EXCEPTION
        assert failure.code == "ValueError"
        assert "bad input" in failure.message

    def test_record_summary_mentions_tool_and_task(self) -> None:
        record = ToolFailureRecord(
            tool_name="slackbot_send_message",
            failure=ToolFailure(message="nope", code="channel_not_found"),
            task_name="post to slack",
        )
        summary = record.summary()
        assert "slackbot_send_message" in summary
        assert "post to slack" in summary
        assert "channel_not_found" in summary


class TestPolicyResolution:
    def test_defaults_to_warn(self) -> None:
        assert resolve_tool_failure_policy() is ToolFailurePolicy.WARN

    def test_agent_policy_used_when_no_narrower_scope(self) -> None:
        agent = Agent(
            role="r",
            goal="g",
            backstory="b",
            tool_failure_policy=ToolFailurePolicy.RAISE,
        )
        assert resolve_tool_failure_policy(agent=agent) is ToolFailurePolicy.RAISE

    def test_task_overrides_agent(self) -> None:
        agent = Agent(
            role="r",
            goal="g",
            backstory="b",
            tool_failure_policy=ToolFailurePolicy.WARN,
        )
        task = Task(
            description="d",
            expected_output="e",
            tool_failure_policy=ToolFailurePolicy.RAISE,
        )
        resolved = resolve_tool_failure_policy(agent=agent, task=task)
        assert resolved is ToolFailurePolicy.RAISE

    def test_unset_task_policy_falls_through_to_agent(self) -> None:
        agent = Agent(
            role="r",
            goal="g",
            backstory="b",
            tool_failure_policy=ToolFailurePolicy.IGNORE,
        )
        task = Task(description="d", expected_output="e")
        resolved = resolve_tool_failure_policy(agent=agent, task=task)
        assert resolved is ToolFailurePolicy.IGNORE

    def test_invalid_policy_is_ignored_rather_than_raising(self) -> None:
        """A bad policy value must never take down a tool call."""

        class Bogus:
            tool_failure_policy = "not-a-policy"

        assert resolve_tool_failure_policy(agent=Bogus()) is ToolFailurePolicy.WARN

    def test_invalid_policy_falls_through_to_next_scope(self) -> None:
        class Bogus:
            tool_failure_policy = object()

        agent = Agent(
            role="r",
            goal="g",
            backstory="b",
            tool_failure_policy=ToolFailurePolicy.IGNORE,
        )
        resolved = resolve_tool_failure_policy(tool=Bogus(), agent=agent)
        assert resolved is ToolFailurePolicy.IGNORE

    def test_tool_overrides_everything(self) -> None:
        class StrictTool(WorkingTool):
            tool_failure_policy: ToolFailurePolicy = ToolFailurePolicy.RAISE

        agent = Agent(
            role="r",
            goal="g",
            backstory="b",
            tool_failure_policy=ToolFailurePolicy.IGNORE,
        )
        resolved = resolve_tool_failure_policy(tool=StrictTool(), agent=agent)
        assert resolved is ToolFailurePolicy.RAISE


class TestAgentDefault:
    def test_agent_defaults_to_warn(self) -> None:
        agent = Agent(role="r", goal="g", backstory="b")
        assert agent.tool_failure_policy is ToolFailurePolicy.WARN

    def test_task_policy_defaults_to_none_so_it_inherits(self) -> None:
        assert Task(description="d", expected_output="e").tool_failure_policy is None


class TestEndToEndPolicies:
    def test_warn_records_and_emits_without_stopping(self) -> None:
        crew, agent = _build_crew(ToolFailurePolicy.WARN)
        events: list[ToolFailureDetectedEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ToolFailureDetectedEvent)
            def _(source: Any, event: ToolFailureDetectedEvent) -> None:
                events.append(event)

            result = crew.kickoff()

        assert len(events) == 1
        assert events[0].tool_name == "slackbot_send_message"
        assert events[0].failure.code == "channel_not_found"
        assert events[0].policy is ToolFailurePolicy.WARN

        assert result.has_tool_failures
        assert len(result.tool_failures) == 1
        assert result.tool_failures[0].failure.code == "channel_not_found"
        assert result.tasks_output[0].has_tool_failures

    def test_ignore_restores_previous_behaviour(self) -> None:
        crew, _ = _build_crew(ToolFailurePolicy.IGNORE)
        events: list[ToolFailureDetectedEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ToolFailureDetectedEvent)
            def _(source: Any, event: ToolFailureDetectedEvent) -> None:
                events.append(event)

            result = crew.kickoff()

        assert events == []
        assert not result.has_tool_failures
        assert result.tool_failures == []

    def test_raise_aborts_the_run(self) -> None:
        crew, _ = _build_crew(ToolFailurePolicy.RAISE)

        with pytest.raises(ToolExecutionFailedError) as exc_info:
            crew.kickoff()

        record = exc_info.value.record
        assert record.tool_name == "slackbot_send_message"
        assert record.failure.code == "channel_not_found"

    def test_event_is_emitted_before_raise(self) -> None:
        """Subscribers must observe the failure even on an aborting run."""
        crew, _ = _build_crew(ToolFailurePolicy.RAISE)
        events: list[ToolFailureDetectedEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ToolFailureDetectedEvent)
            def _(source: Any, event: ToolFailureDetectedEvent) -> None:
                events.append(event)

            with pytest.raises(ToolExecutionFailedError):
                crew.kickoff()

        assert len(events) == 1

    def test_task_policy_overrides_agent_end_to_end(self) -> None:
        crew, _ = _build_crew(
            ToolFailurePolicy.WARN,
            tool_failure_policy=ToolFailurePolicy.RAISE,
        )
        with pytest.raises(ToolExecutionFailedError):
            crew.kickoff()

    def test_default_agent_warns(self) -> None:
        """No explicit policy anywhere still records the failure."""
        crew, _ = _build_crew()
        result = crew.kickoff()
        assert result.has_tool_failures

    def test_finished_event_carries_the_failure(self) -> None:
        crew, _ = _build_crew(ToolFailurePolicy.WARN)
        finished: list[ToolUsageFinishedEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ToolUsageFinishedEvent)
            def _(source: Any, event: ToolUsageFinishedEvent) -> None:
                finished.append(event)

            crew.kickoff()

        slack_events = [e for e in finished if e.tool_name == "slackbot_send_message"]
        assert slack_events
        assert slack_events[0].failure is not None
        assert slack_events[0].failure.code == "channel_not_found"

    def test_agent_sees_the_failure_message_as_plain_text(self) -> None:
        """Model-facing behavior is unchanged: it still reads prose."""
        crew, _ = _build_crew(ToolFailurePolicy.WARN)
        result = crew.kickoff()
        tool_messages = [
            m
            for m in result.tasks_output[0].messages
            if "Slack rejected the message" in str(m.get("content", ""))
        ]
        assert tool_messages


class TestToolScopedPolicyReachesTheExecutor:
    """A tool-scoped policy must survive the CrewStructuredTool wrapper.

    The executors hand ``handle_tool_failure`` the wrapper, not the authored
    BaseTool, so a policy set on the tool used to be silently dropped.
    """

    def test_policy_survives_to_structured_tool(self) -> None:
        class StrictSlack(SlackTool):
            tool_failure_policy: ToolFailurePolicy | None = ToolFailurePolicy.RAISE

        wrapper = StrictSlack().to_structured_tool()
        assert wrapper.tool_failure_policy is ToolFailurePolicy.RAISE
        assert resolve_tool_failure_policy(tool=wrapper) is ToolFailurePolicy.RAISE

    def test_policy_resolves_through_original_tool_reference(self) -> None:
        """Even a wrapper that never copied the field resolves via _original_tool."""

        class StrictSlack(SlackTool):
            tool_failure_policy: ToolFailurePolicy | None = ToolFailurePolicy.RAISE

        wrapper = StrictSlack().to_structured_tool()
        wrapper.tool_failure_policy = None
        assert resolve_tool_failure_policy(tool=wrapper) is ToolFailurePolicy.RAISE

    def test_tool_policy_aborts_a_warn_agent_end_to_end(self) -> None:
        class StrictSlack(SlackTool):
            tool_failure_policy: ToolFailurePolicy | None = ToolFailurePolicy.RAISE

        agent = Agent(
            role="Slack Messenger",
            goal="post a message",
            backstory="b",
            llm=ScriptedLLM(_slack_steps()),
            tools=[StrictSlack()],
            tool_failure_policy=ToolFailurePolicy.WARN,
        )
        task = Task(description="post to slack", expected_output="c", agent=agent)
        with pytest.raises(ToolExecutionFailedError):
            Crew(agents=[agent], tasks=[task]).kickoff()

    def test_tool_policy_can_exempt_a_raising_agent(self) -> None:
        class ChattySlack(SlackTool):
            tool_failure_policy: ToolFailurePolicy | None = ToolFailurePolicy.IGNORE

        agent = Agent(
            role="Slack Messenger",
            goal="post a message",
            backstory="b",
            llm=ScriptedLLM(_slack_steps()),
            tools=[ChattySlack()],
            tool_failure_policy=ToolFailurePolicy.RAISE,
        )
        task = Task(description="post to slack", expected_output="c", agent=agent)
        result = Crew(agents=[agent], tasks=[task]).kickoff()
        assert not result.has_tool_failures

    def test_plain_tools_default_to_inheriting(self) -> None:
        assert SlackTool().tool_failure_policy is None


class TestConsolePanels:
    """Exactly one panel per failed call, and never a green one.

    A failed call used to print the green "Tool Execution Completed" panel --
    the terminal equivalent of the green checkmarks in the bug report.
    """

    @staticmethod
    def _formatter():
        from crewai.events.utils.console_formatter import ConsoleFormatter

        return ConsoleFormatter(verbose=True)

    def test_success_panel_suppressed_when_the_call_failed(self) -> None:
        failure = ToolFailure(message="nope", code="channel_not_found")
        assert self._formatter().should_render_success_panel(failure) is False

    def test_success_panel_still_shown_for_a_working_call(self) -> None:
        assert self._formatter().should_render_success_panel(None) is True

    def test_exception_failures_do_not_double_print(self) -> None:
        """ToolUsageErrorEvent already prints; the failure panel must not repeat it."""
        failure = failure_from_exception(ValueError("kaboom"))
        assert self._formatter().should_render_failure_panel(failure) is False

    def test_tool_reported_failures_do_print(self) -> None:
        failure = ToolFailure(message="nope", code="channel_not_found")
        assert self._formatter().should_render_failure_panel(failure) is True

    def test_mcp_failures_do_print(self) -> None:
        failure = ToolFailure(message="nope", reason=ToolFailureReason.MCP_ERROR)
        assert self._formatter().should_render_failure_panel(failure) is True

    def test_failure_panel_renders_without_raising(self) -> None:
        """The real formatter must handle the payload it is given."""
        self._formatter().handle_tool_failure_detected(
            "slackbot_send_message",
            ToolFailure(message="nope", code="channel_not_found"),
            ToolFailurePolicy.WARN,
        )

    def test_listener_consults_the_predicates(self) -> None:
        """The listener must route through the predicates, not its own logic."""
        import inspect

        from crewai.events.event_listener import EventListener

        source = inspect.getsource(EventListener.setup_listeners)
        assert "should_render_success_panel" in source
        assert "should_render_failure_panel" in source


class TestUnknownToolOnNativePaths:
    """The ReAct path reported unknown tools; the native paths did not."""

    def test_native_path_records_unknown_tool(self) -> None:
        from crewai.utilities.agent_utils import execute_single_native_tool_call

        agent = Agent(role="r", goal="g", backstory="b")
        recorded: list[ToolFailureDetectedEvent] = []

        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="does_not_exist", arguments="{}"),
        )

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ToolFailureDetectedEvent)
            def _(source: Any, event: ToolFailureDetectedEvent) -> None:
                recorded.append(event)

            execute_single_native_tool_call(
                tool_call,
                available_functions={},
                original_tools=[],
                structured_tools=[],
                tools_handler=None,
                agent=agent,
                task=None,
                crew=None,
                event_source=agent,
                printer=None,
                verbose=False,
            )
            # emit() dispatches sync handlers on a thread pool, so drain
            # before asserting on what subscribers saw.
            crewai_event_bus.flush(timeout=10.0)

        # The record is written synchronously, before the event is emitted.
        assert len(agent.last_tool_failures) == 1
        record = agent.last_tool_failures[0]
        assert record.tool_name == "does_not_exist"
        assert record.failure.reason is ToolFailureReason.UNKNOWN_TOOL
        assert record.failure.code == "does_not_exist"

        assert len(recorded) == 1
        assert recorded[0].failure.reason is ToolFailureReason.UNKNOWN_TOOL

    def test_unknown_tool_can_abort_under_raise(self) -> None:
        from crewai.utilities.agent_utils import execute_single_native_tool_call

        agent = Agent(
            role="r",
            goal="g",
            backstory="b",
            tool_failure_policy=ToolFailurePolicy.RAISE,
        )
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="does_not_exist", arguments="{}"),
        )

        with pytest.raises(ToolExecutionFailedError):
            execute_single_native_tool_call(
                tool_call,
                available_functions={},
                original_tools=[],
                structured_tools=[],
                tools_handler=None,
                agent=agent,
                task=None,
                crew=None,
                event_source=agent,
                printer=None,
                verbose=False,
            )


class TestExceptionFailuresStillRecorded:
    def test_raised_tool_produces_a_failure_record(self) -> None:
        class BoomTool(BaseTool):
            name: str = "boom"
            description: str = "Always explodes."

            def _run(self, x: str) -> Any:
                raise ValueError("kaboom")

        agent = Agent(
            role="Breaker",
            goal="break",
            backstory="b",
            llm=ScriptedLLM(
                [
                    'Thought: go\nAction: boom\nAction Input: {"x": "1"}',
                    "Thought: it broke\nFinal Answer: it broke.",
                ]
            ),
            tools=[BoomTool()],
        )
        task = Task(description="break it", expected_output="e", agent=agent)
        result = Crew(agents=[agent], tasks=[task]).kickoff()

        assert result.has_tool_failures
        reasons = {f.failure.reason for f in result.tool_failures}
        assert ToolFailureReason.EXCEPTION in reasons


class TestLiteAgentOutputParity:
    def test_has_tool_failures_exists_on_all_output_types(self) -> None:
        from crewai.crews.crew_output import CrewOutput
        from crewai.lite_agent_output import LiteAgentOutput
        from crewai.tasks.task_output import TaskOutput

        record = ToolFailureRecord(
            tool_name="t", failure=ToolFailure(message="nope")
        )
        assert LiteAgentOutput(agent_role="r").has_tool_failures is False
        assert (
            LiteAgentOutput(agent_role="r", tool_failures=[record]).has_tool_failures
            is True
        )
        assert TaskOutput(description="d", agent="a").has_tool_failures is False
        assert CrewOutput().has_tool_failures is False


class TestMCPIsErrorPlumbing:
    """An MCP server flags a failed tool with isError on a 200 response."""

    @staticmethod
    def _tool(is_error: bool) -> Any:
        from unittest.mock import AsyncMock

        from crewai.mcp.client import _MCPToolResult
        from crewai.tools.mcp_native_tool import MCPNativeTool

        client = AsyncMock()
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()
        client.call_tool_result = AsyncMock(
            return_value=_MCPToolResult("channel not found", is_error)
        )
        return MCPNativeTool(
            client_factory=lambda: client,
            tool_name="post",
            tool_schema={"description": "post a message"},
            server_name="slack",
        )

    def test_is_error_becomes_a_tool_failure(self) -> None:
        result = self._tool(is_error=True).run()
        assert isinstance(result, ToolFailure)
        assert result.reason is ToolFailureReason.MCP_ERROR
        assert result.message == "channel not found"
        assert result.details["server"] == "slack"

    def test_successful_call_still_returns_plain_text(self) -> None:
        assert self._tool(is_error=False).run() == "channel not found"


class TestPlatformActionTool:
    """CrewAI AMP agentic-app actions -- the Slack case from the bug report."""

    @staticmethod
    def _tool() -> Any:
        import crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool as mod

        return mod.CrewAIPlatformActionTool(
            description="Send a Slack message",
            action_name="slackbot_send_message",
            action_schema={
                "function": {
                    "name": "slackbot_send_message",
                    "parameters": {
                        "properties": {"channel": {"type": "string"}},
                        "required": [],
                    },
                }
            },
        )

    def test_non_ok_response_becomes_a_tool_failure(self, monkeypatch) -> None:  # noqa: ANN001
        from unittest.mock import Mock

        import crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool as mod

        response = Mock()
        response.ok = False
        response.status_code = 500
        response.json.return_value = {
            "error": "Failed to execute action: Slack API error: channel_not_found"
        }
        monkeypatch.setattr(mod.requests, "post", Mock(return_value=response))
        monkeypatch.setenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", "t")

        result = self._tool()._run(channel="#joao-message")

        assert isinstance(result, ToolFailure)
        assert "channel_not_found" in result.message
        assert result.retryable is True

    def test_ok_response_still_returns_json(self, monkeypatch) -> None:  # noqa: ANN001
        from unittest.mock import Mock

        import crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool as mod

        response = Mock()
        response.ok = True
        response.json.return_value = {"ts": "1234.5678"}
        monkeypatch.setattr(mod.requests, "post", Mock(return_value=response))
        monkeypatch.setenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", "t")

        result = self._tool()._run(channel="#general")

        assert not isinstance(result, ToolFailure)
        assert "1234.5678" in result


class TestSuccessfulToolsUnaffected:
    def test_no_failure_recorded_for_a_working_tool(self) -> None:
        agent = Agent(
            role="Echoer",
            goal="echo",
            backstory="b",
            llm=ScriptedLLM(
                [
                    'Thought: echo\nAction: echo\nAction Input: {"text": "hi"}',
                    "Thought: done\nFinal Answer: echoed: hi",
                ]
            ),
            tools=[WorkingTool()],
        )
        task = Task(description="echo hi", expected_output="hi", agent=agent)
        result = Crew(agents=[agent], tasks=[task]).kickoff()

        assert not result.has_tool_failures
        assert result.tool_failures == []

    def test_failures_reset_between_executions(self) -> None:
        crew, agent = _build_crew(ToolFailurePolicy.WARN)
        crew.kickoff()
        assert len(agent.last_tool_failures) == 1

        agent.llm = ScriptedLLM(_slack_steps())
        crew.kickoff()
        assert len(agent.last_tool_failures) == 1, "records must not accumulate"
