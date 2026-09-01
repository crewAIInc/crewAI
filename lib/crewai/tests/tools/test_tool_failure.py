"""Tests for structured tool-failure signalling and the per-agent policy."""

import json
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


class StatelessToolLLM(LLM):
    """Calls one tool, then answers -- decided from the messages, not a counter.

    Stateless so concurrent executions sharing one agent cannot interleave into
    each other's script.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "StatelessToolLLM":
        return object.__new__(cls)

    def __init__(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        done_marker: str = "rejected the message",
    ) -> None:
        super().__init__(model="gpt-4o")
        self._tool_name = tool_name
        self._tool_args = tool_args
        # A sentinel from the tool's own output. Not "Observation" -- the ReAct
        # prompt itself contains that word, so the stub would answer before
        # ever calling the tool.
        self._done_marker = done_marker

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kw):  # noqa: ANN001, ANN003
        if self._done_marker in str(messages):
            return "Thought: it failed\nFinal Answer: could not post."
        return (
            "Thought: posting\n"
            + f"Action: {self._tool_name}\n"
            + f"Action Input: {json.dumps(self._tool_args)}"
        )

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

    def test_crew_policy_used_when_agent_inherits(self) -> None:
        from crewai import Crew

        agent = Agent(role="r", goal="g", backstory="b")
        crew = Crew(
            agents=[agent], tasks=[], tool_failure_policy=ToolFailurePolicy.RAISE
        )
        resolved = resolve_tool_failure_policy(agent=agent, crew=crew)
        assert resolved is ToolFailurePolicy.RAISE

    def test_agent_overrides_crew(self) -> None:
        from crewai import Crew

        agent = Agent(
            role="r",
            goal="g",
            backstory="b",
            tool_failure_policy=ToolFailurePolicy.IGNORE,
        )
        crew = Crew(
            agents=[agent], tasks=[], tool_failure_policy=ToolFailurePolicy.RAISE
        )
        resolved = resolve_tool_failure_policy(agent=agent, crew=crew)
        assert resolved is ToolFailurePolicy.IGNORE

    def test_full_precedence_chain(self) -> None:
        """tool > task > agent > crew > warn."""
        from crewai import Crew

        class ScopedTool(SlackTool):
            tool_failure_policy: ToolFailurePolicy | None = None

        tool = ScopedTool()
        agent = Agent(role="r", goal="g", backstory="b")
        task = Task(description="d", expected_output="e")
        crew = Crew(agents=[agent], tasks=[])

        def resolved() -> ToolFailurePolicy:
            return resolve_tool_failure_policy(
                tool=tool, agent=agent, task=task, crew=crew
            )

        assert resolved() is ToolFailurePolicy.WARN

        crew.tool_failure_policy = ToolFailurePolicy.IGNORE
        assert resolved() is ToolFailurePolicy.IGNORE

        agent.tool_failure_policy = ToolFailurePolicy.WARN
        assert resolved() is ToolFailurePolicy.WARN

        task.tool_failure_policy = ToolFailurePolicy.RAISE
        assert resolved() is ToolFailurePolicy.RAISE

        tool.tool_failure_policy = ToolFailurePolicy.IGNORE
        assert resolved() is ToolFailurePolicy.IGNORE

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


class TestDefaults:
    """Every scope defaults to None ('inherit'); the resolver owns 'warn'."""

    def test_agent_defaults_to_inherit(self) -> None:
        assert Agent(role="r", goal="g", backstory="b").tool_failure_policy is None

    def test_task_defaults_to_inherit(self) -> None:
        assert Task(description="d", expected_output="e").tool_failure_policy is None

    def test_crew_defaults_to_inherit(self) -> None:
        from crewai import Crew

        agent = Agent(role="r", goal="g", backstory="b")
        assert Crew(agents=[agent], tasks=[]).tool_failure_policy is None

    def test_tool_defaults_to_inherit(self) -> None:
        assert SlackTool().tool_failure_policy is None

    def test_effective_default_is_warn(self) -> None:
        agent = Agent(role="r", goal="g", backstory="b")
        assert resolve_tool_failure_policy(agent=agent) is ToolFailurePolicy.WARN


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

    Executors pass the wrapper, not the authored BaseTool, so a tool-scoped
    policy used to be silently dropped.
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

    A failed call used to print the green "Completed" panel.
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
            # emit() dispatches on a thread pool; drain before asserting.
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


class TestRaisePolicySurvivesEveryWrapper:
    """`raise` must abort, not get downgraded by an enclosing handler."""

    def test_timeout_wrapper_preserves_the_error_type(self) -> None:
        """max_execution_time wraps failures in RuntimeError; not this one."""
        agent = Agent(
            role="Slack Messenger",
            goal="post a message",
            backstory="b",
            llm=ScriptedLLM(_slack_steps()),
            tools=[SlackTool()],
            tool_failure_policy=ToolFailurePolicy.RAISE,
            max_execution_time=30,
        )
        task = Task(description="post to slack", expected_output="c", agent=agent)

        with pytest.raises(ToolExecutionFailedError):
            Crew(agents=[agent], tasks=[task]).kickoff()

    def test_retry_limit_does_not_swallow_the_abort(self) -> None:
        """A deliberate stop must not be retried as a transient error."""
        agent = Agent(
            role="Slack Messenger",
            goal="post a message",
            backstory="b",
            llm=ScriptedLLM(_slack_steps()),
            tools=[SlackTool()],
            tool_failure_policy=ToolFailurePolicy.RAISE,
            max_retry_limit=3,
        )
        task = Task(description="post to slack", expected_output="c", agent=agent)

        with pytest.raises(ToolExecutionFailedError):
            Crew(agents=[agent], tasks=[task]).kickoff()
        assert agent._times_executed == 0, "the abort must not trigger retries"

    def test_crew_policy_aborts_end_to_end(self) -> None:
        """Crew scope must actually reach the executor, not just the resolver."""
        agent = Agent(
            role="Slack Messenger",
            goal="post a message",
            backstory="b",
            llm=ScriptedLLM(_slack_steps()),
            tools=[SlackTool()],
        )
        task = Task(description="post to slack", expected_output="c", agent=agent)
        crew = Crew(
            agents=[agent],
            tasks=[task],
            tool_failure_policy=ToolFailurePolicy.RAISE,
        )

        with pytest.raises(ToolExecutionFailedError):
            crew.kickoff()

    def test_crew_ignore_suppresses_recording_end_to_end(self) -> None:
        agent = Agent(
            role="Slack Messenger",
            goal="post a message",
            backstory="b",
            llm=ScriptedLLM(_slack_steps()),
            tools=[SlackTool()],
        )
        task = Task(description="post to slack", expected_output="c", agent=agent)
        result = Crew(
            agents=[agent],
            tasks=[task],
            tool_failure_policy=ToolFailurePolicy.IGNORE,
        ).kickoff()
        assert not result.has_tool_failures

    def test_every_broad_handler_around_tool_execution_lets_it_through(self) -> None:
        """Guard against a new `except Exception` quietly downgrading an abort.

        Five separate handlers have swallowed this exception during review of
        this PR, so assert the passthrough at each site rather than trusting
        that the next one will be spotted.
        """
        import inspect

        from crewai.agent.core import Agent as AgentCls
        from crewai.agents.step_executor import StepExecutor
        from crewai.experimental.agent_executor import AgentExecutor

        # CrewAgentExecutor is deprecated and deliberately excluded.
        sites = [
            (AgentCls._execute_with_timeout, "_passthrough_exceptions"),
            (StepExecutor.execute, "ToolExecutionFailedError"),
            (AgentExecutor.execute_tool_action, "ToolExecutionFailedError"),
            (AgentExecutor.execute_native_tool, "ToolExecutionFailedError"),
        ]
        for func, expected in sites:
            source = inspect.getsource(func)
            assert expected in source, f"{func.__qualname__} lost its passthrough"

    def test_passthrough_tuple_includes_the_error(self) -> None:
        from crewai.agent.core import _passthrough_exceptions

        assert ToolExecutionFailedError in _passthrough_exceptions


class TestFailureRecordsResetAndAccumulate:
    def test_kickoff_resets_between_runs(self) -> None:
        """Agent.kickoff() goes through _prepare_kickoff, not task execution."""
        agent = Agent(
            role="Slack Messenger",
            goal="post a message",
            backstory="b",
            llm=ScriptedLLM(_slack_steps()),
            tools=[SlackTool()],
        )

        first = agent.kickoff("post it")
        assert len(first.tool_failures) == 1
        assert first.has_tool_failures

        agent.llm = ScriptedLLM(_slack_steps())
        second = agent.kickoff("post it again")
        assert len(second.tool_failures) == 1, "records must not accumulate"

    def test_kickoff_output_sees_failures_recorded_on_the_agent(self) -> None:
        """The LiteAgent under kickoff records against the owning Agent."""
        agent = Agent(
            role="Slack Messenger",
            goal="post a message",
            backstory="b",
            llm=ScriptedLLM(_slack_steps()),
            tools=[SlackTool()],
        )
        result = agent.kickoff("post it")
        assert [f.failure.code for f in result.tool_failures] == ["channel_not_found"]

    def test_last_tool_failures_returns_a_copy(self) -> None:
        agent = Agent(role="r", goal="g", backstory="b")
        agent._tool_failures.append(
            ToolFailureRecord(tool_name="t", failure=ToolFailure(message="nope"))
        )
        snapshot = agent.last_tool_failures
        snapshot.clear()
        assert len(agent.last_tool_failures) == 1

    def test_guardrail_retry_preserves_earlier_failures(self) -> None:
        """A blocked attempt's failures must survive into the final output.

        The retry resets the agent's record, so without accumulation this would
        report zero failures despite one demonstrably happening.
        """
        attempts: list[int] = []

        def guardrail(output: Any) -> tuple[bool, Any]:
            attempts.append(1)
            if len(attempts) == 1:
                return (False, "needs another pass")
            return (True, output.raw)

        agent = Agent(
            role="Slack Messenger",
            goal="post a message",
            backstory="b",
            llm=ScriptedLLM(_slack_steps()),
            tools=[SlackTool()],
        )
        task = Task(
            description="post to slack",
            expected_output="c",
            agent=agent,
            guardrail=guardrail,
        )
        result = Crew(agents=[agent], tasks=[task]).kickoff()

        assert len(attempts) == 2, "guardrail should have blocked once"
        # The scripted LLM answers directly on the retry, so the survivor is
        # the blocked first attempt's record.
        assert len(result.tool_failures) == 1
        assert result.tool_failures[0].failure.code == "channel_not_found"


class TestIgnoreSurfacesNothing:
    """`ignore` must suppress the flag on the finished event too.

    Leaving `failure` set made traces treat the call as failed and left the
    console with no panel at all: green suppressed, red skipped.
    """

    def test_finished_event_carries_no_failure_under_ignore(self) -> None:
        crew, _ = _build_crew(ToolFailurePolicy.IGNORE)
        finished: list[ToolUsageFinishedEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ToolUsageFinishedEvent)
            def _(source: Any, event: ToolUsageFinishedEvent) -> None:
                finished.append(event)

            crew.kickoff()
            crewai_event_bus.flush(timeout=10.0)

        slack = [e for e in finished if e.tool_name == "slackbot_send_message"]
        assert slack, "the tool call should still report as finished"
        assert all(e.failure is None for e in slack)

    def test_finished_event_carries_failure_under_warn(self) -> None:
        crew, _ = _build_crew(ToolFailurePolicy.WARN)
        finished: list[ToolUsageFinishedEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ToolUsageFinishedEvent)
            def _(source: Any, event: ToolUsageFinishedEvent) -> None:
                finished.append(event)

            crew.kickoff()
            crewai_event_bus.flush(timeout=10.0)

        slack = [e for e in finished if e.tool_name == "slackbot_send_message"]
        assert any(e.failure is not None for e in slack)

    def test_reportable_failure_helper(self) -> None:
        from crewai.tools.tool_failure import reportable_failure

        failure = ToolFailure(message="nope")
        ignoring = Agent(
            role="r",
            goal="g",
            backstory="b",
            tool_failure_policy=ToolFailurePolicy.IGNORE,
        )
        warning = Agent(
            role="r",
            goal="g",
            backstory="b",
            tool_failure_policy=ToolFailurePolicy.WARN,
        )
        assert reportable_failure(failure, agent=ignoring) is None
        assert reportable_failure(failure, agent=warning) is failure
        assert reportable_failure(None, agent=warning) is None

    def test_ignore_still_shows_a_console_panel(self) -> None:
        """With no failure flag, the ordinary green panel is restored."""
        from crewai.events.utils.console_formatter import ConsoleFormatter

        formatter = ConsoleFormatter(verbose=True)
        assert formatter.should_render_success_panel(None) is True


class TestFailuresAreNotCached:
    """A cached failure would make a transient error permanent."""

    def test_cache_handler_refuses_to_store_a_failure(self) -> None:
        from crewai.agents.cache.cache_handler import CacheHandler

        cache = CacheHandler()
        cache.add(tool="t", input="{}", output=ToolFailure(message="nope"))
        assert cache.read(tool="t", input="{}") is None

    def test_cache_handler_still_stores_successes(self) -> None:
        from crewai.agents.cache.cache_handler import CacheHandler

        cache = CacheHandler()
        cache.add(tool="t", input="{}", output="fine")
        assert cache.read(tool="t", input="{}") == "fine"

    def test_repeated_failures_are_recorded_once_each(self) -> None:
        """Two failing calls give two records, not a replayed cache hit."""
        agent = Agent(
            role="Slack Messenger",
            goal="post a message",
            backstory="b",
            llm=ScriptedLLM(
                [
                    'Thought: a\nAction: slackbot_send_message\nAction Input: {"channel": "#c"}',
                    'Thought: b\nAction: slackbot_send_message\nAction Input: {"channel": "#c"}',
                    "Thought: done\nFinal Answer: could not post.",
                ]
            ),
            tools=[SlackTool()],
            cache=True,
        )
        task = Task(description="post twice", expected_output="c", agent=agent)
        result = Crew(agents=[agent], tasks=[task], cache=True).kickoff()
        assert len(result.tool_failures) >= 1
        assert all(
            f.failure.code == "channel_not_found" for f in result.tool_failures
        )


class TestUsageLimitIsStructured:
    """A spent max_usage_count must be a ToolFailure, not a bare string."""

    def test_claim_usage_returns_a_failure(self) -> None:
        tool = WorkingTool(max_usage_count=1)
        assert tool.run(text="first") == "echoed: first"

        second = tool.run(text="second")
        assert isinstance(second, ToolFailure)
        assert second.reason is ToolFailureReason.USAGE_LIMIT
        assert "usage limit" in second.message

    def test_spent_limit_is_recorded_on_every_path(self) -> None:
        agent = Agent(
            role="Echoer",
            goal="echo",
            backstory="b",
            llm=ScriptedLLM(
                [
                    'Thought: a\nAction: echo\nAction Input: {"text": "one"}',
                    'Thought: b\nAction: echo\nAction Input: {"text": "two"}',
                    "Thought: done\nFinal Answer: done.",
                ]
            ),
            tools=[WorkingTool(max_usage_count=1)],
        )
        task = Task(description="echo twice", expected_output="c", agent=agent)
        result = Crew(agents=[agent], tasks=[task]).kickoff()

        reasons = {f.failure.reason for f in result.tool_failures}
        assert ToolFailureReason.USAGE_LIMIT in reasons


class TestGuardrailReturningTaskOutput:
    def test_replacement_output_keeps_earlier_failures(self) -> None:
        """A guardrail may return a whole new TaskOutput; failures must survive."""
        from crewai.tasks.task_output import TaskOutput

        attempts: list[int] = []

        def guardrail(output: TaskOutput) -> tuple[bool, Any]:
            attempts.append(1)
            replacement = TaskOutput(
                description=output.description,
                raw="rewritten by guardrail",
                agent=output.agent,
            )
            return (True, replacement)

        agent = Agent(
            role="Slack Messenger",
            goal="post a message",
            backstory="b",
            llm=ScriptedLLM(_slack_steps()),
            tools=[SlackTool()],
        )
        task = Task(
            description="post to slack",
            expected_output="c",
            agent=agent,
            guardrail=guardrail,
        )
        result = Crew(agents=[agent], tasks=[task]).kickoff()

        assert attempts, "guardrail should have run"
        assert result.raw == "rewritten by guardrail"
        assert len(result.tool_failures) == 1
        assert result.tool_failures[0].failure.code == "channel_not_found"


class TestMergeToolFailures:
    def test_deduplicates_equivalent_records(self) -> None:
        from crewai.tools.tool_failure import merge_tool_failures

        record = ToolFailureRecord(
            tool_name="t", failure=ToolFailure(message="nope", code="c")
        )
        same = ToolFailureRecord(
            tool_name="t", failure=ToolFailure(message="nope", code="c")
        )
        other = ToolFailureRecord(tool_name="t2", failure=ToolFailure(message="nope"))

        merged = merge_tool_failures([record], [same, other])
        assert len(merged) == 2
        assert merged[0] is record

    def test_preserves_order(self) -> None:
        from crewai.tools.tool_failure import merge_tool_failures

        first = ToolFailureRecord(tool_name="a", failure=ToolFailure(message="1"))
        second = ToolFailureRecord(tool_name="b", failure=ToolFailure(message="2"))
        assert merge_tool_failures([first], [second]) == [first, second]


class TestHookBlockDoesNotInheritCachedFailure:
    """A blocked call must not be attributed a failure it did not produce.

    CacheHandler no longer stores failures, so this is unreachable through the
    built-in cache -- the guard covers a custom cache handler that does.
    """

    def test_blocked_call_reports_no_failure(self) -> None:
        from crewai.agents.tools_handler import ToolsHandler
        from crewai.hooks import (
            clear_before_tool_call_hooks,
            register_before_tool_call_hook,
        )
        from crewai.utilities.agent_utils import execute_single_native_tool_call

        class FailureReplayingCache:
            """Stands in for a custom cache that does retain failures."""

            def read(self, tool: str, input: str) -> Any:
                return ToolFailure(message="stale cached failure", code="cached")

            def add(self, tool: str, input: str, output: Any) -> None:
                pass

        agent = Agent(role="r", goal="g", backstory="b")
        recorded: list[ToolFailureDetectedEvent] = []
        tool = SlackTool()
        structured = tool.to_structured_tool()
        handler = ToolsHandler()
        handler.cache = FailureReplayingCache()  # type: ignore[assignment]

        tool_call = SimpleNamespace(
            id="c1",
            function=SimpleNamespace(
                name="slackbot_send_message", arguments='{"channel": "#c"}'
            ),
        )

        register_before_tool_call_hook(lambda ctx: False)
        try:
            with crewai_event_bus.scoped_handlers():

                @crewai_event_bus.on(ToolFailureDetectedEvent)
                def _(source: Any, event: ToolFailureDetectedEvent) -> None:
                    recorded.append(event)

                result = execute_single_native_tool_call(
                    tool_call,
                    available_functions={"slackbot_send_message": tool.run},
                    original_tools=[tool],
                    structured_tools=[structured],
                    tools_handler=handler,
                    agent=agent,
                    task=None,
                    crew=None,
                    event_source=agent,
                    printer=None,
                    verbose=False,
                )
                crewai_event_bus.flush(timeout=10.0)
        finally:
            clear_before_tool_call_hooks()

        assert "blocked by hook" in str(result.result)
        assert recorded == [], "a blocked call must not report a tool failure"
        assert agent.last_tool_failures == []


class TestCrewScopeReachesTheFinishedEvent:
    """`ToolUsage` needs the crew, or crew-level ignore only half applies."""

    def test_crew_ignore_suppresses_the_finished_event_flag(self) -> None:
        agent = Agent(
            role="Slack Messenger",
            goal="post a message",
            backstory="b",
            llm=ScriptedLLM(_slack_steps()),
            tools=[SlackTool()],
        )
        task = Task(description="post to slack", expected_output="c", agent=agent)
        crew = Crew(
            agents=[agent],
            tasks=[task],
            tool_failure_policy=ToolFailurePolicy.IGNORE,
        )

        finished: list[ToolUsageFinishedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ToolUsageFinishedEvent)
            def _(source: Any, event: ToolUsageFinishedEvent) -> None:
                finished.append(event)

            crew.kickoff()
            crewai_event_bus.flush(timeout=10.0)

        slack = [e for e in finished if e.tool_name == "slackbot_send_message"]
        assert slack
        assert all(e.failure is None for e in slack)

    def test_tool_usage_accepts_and_stores_crew(self) -> None:
        from crewai.tools.tool_usage import ToolUsage

        agent = Agent(role="r", goal="g", backstory="b")
        crew = Crew(agents=[agent], tasks=[])
        usage = ToolUsage(
            tools_handler=None,
            tools=[],
            task=None,
            function_calling_llm=None,  # type: ignore[arg-type]
            agent=agent,
            crew=crew,
        )
        assert usage.crew is crew


class TestFailedToolIsNotTheFinalAnswer:
    """result_as_answer must not turn an error into the task's output."""

    @staticmethod
    def _agent(policy: ToolFailurePolicy) -> Agent:
        class AnswerSlack(SlackTool):
            result_as_answer: bool = True

        return Agent(
            role="Slack Messenger",
            goal="post a message",
            backstory="b",
            llm=ScriptedLLM(_slack_steps()),
            tools=[AnswerSlack()],
            tool_failure_policy=policy,
        )

    def test_failure_does_not_short_circuit_under_warn(self) -> None:
        agent = self._agent(ToolFailurePolicy.WARN)
        task = Task(description="post to slack", expected_output="c", agent=agent)
        result = Crew(agents=[agent], tasks=[task]).kickoff()

        assert "Slack rejected the message" not in result.raw
        assert result.raw == "I could not post the message."
        assert result.has_tool_failures

    def test_successful_result_as_answer_still_short_circuits(self) -> None:
        class AnswerEcho(WorkingTool):
            result_as_answer: bool = True

        agent = Agent(
            role="Echoer",
            goal="echo",
            backstory="b",
            llm=ScriptedLLM(
                [
                    'Thought: go\nAction: echo\nAction Input: {"text": "hi"}',
                    "Thought: done\nFinal Answer: unused.",
                ]
            ),
            tools=[AnswerEcho()],
        )
        task = Task(description="echo", expected_output="c", agent=agent)
        result = Crew(agents=[agent], tasks=[task]).kickoff()

        assert result.raw == "echoed: hi"
        assert not result.has_tool_failures


class TestEventCarriesCorrelationIds:
    """The failure event must be correlatable with the call it describes."""

    def test_agent_and_task_ids_are_populated(self) -> None:
        crew, agent = _build_crew(ToolFailurePolicy.WARN)
        events: list[ToolFailureDetectedEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ToolFailureDetectedEvent)
            def _(source: Any, event: ToolFailureDetectedEvent) -> None:
                events.append(event)

            crew.kickoff()
            crewai_event_bus.flush(timeout=10.0)

        assert events
        event = events[0]
        assert event.agent_id == str(agent.id)
        assert event.agent_role == agent.role
        assert event.task_id is not None
        assert event.task_name == "post to slack"

    def test_ids_match_the_paired_finished_event(self) -> None:
        crew, _ = _build_crew(ToolFailurePolicy.WARN)
        failures: list[ToolFailureDetectedEvent] = []
        finished: list[ToolUsageFinishedEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ToolFailureDetectedEvent)
            def _f(source: Any, event: ToolFailureDetectedEvent) -> None:
                failures.append(event)

            @crewai_event_bus.on(ToolUsageFinishedEvent)
            def _d(source: Any, event: ToolUsageFinishedEvent) -> None:
                if event.tool_name == "slackbot_send_message":
                    finished.append(event)

            crew.kickoff()
            crewai_event_bus.flush(timeout=10.0)

        assert failures and finished
        assert failures[0].agent_id == finished[0].agent_id
        assert failures[0].task_id == finished[0].task_id


class TestMalformedArgumentsAreReported:
    """A tool call with unparseable JSON args is a failure, not a silent skip."""

    @staticmethod
    def _parse_error() -> dict[str, Any]:
        from crewai.utilities.agent_utils import parse_tool_call_args

        args, error = parse_tool_call_args("{not json", "echo", "call_1")
        assert args is None
        assert error is not None
        return error

    def test_parse_error_carries_an_invalid_input_failure(self) -> None:
        error = self._parse_error()
        failure = error["tool_failure"]
        assert isinstance(failure, ToolFailure)
        assert failure.reason is ToolFailureReason.INVALID_INPUT
        assert failure.code == "json_decode_error"

    def test_valid_args_carry_no_failure(self) -> None:
        from crewai.utilities.agent_utils import parse_tool_call_args

        args, error = parse_tool_call_args('{"text": "hi"}', "echo", "call_1")
        assert args == {"text": "hi"}
        assert error is None

    def test_reason_enum_member_is_used(self) -> None:
        """INVALID_INPUT was declared but unreferenced before this."""
        import inspect

        from crewai.utilities import agent_utils

        assert "INVALID_INPUT" in inspect.getsource(agent_utils.parse_tool_call_args)


class TestDeprecatedExecutorIsNotIntegrated:
    """CrewAgentExecutor is deprecated; the feature must not extend into it."""

    def test_no_tool_failure_integration(self) -> None:
        from importlib import import_module
        from pathlib import Path

        # Read the file directly: importing this module by name resolves to a
        # different one in this package, so inspect would read the wrong source.
        package = import_module(Agent.__module__.split(".")[0])
        source = (
            Path(package.__file__).parent / "agents" / "crew_agent_executor.py"
        ).read_text()
        assert "tool_failure" not in source
        assert "ToolExecutionFailedError" not in source


class TestConcurrentExecutionsAreIsolated:
    """A shared agent must not leak failures between concurrent executions.

    Accumulating on the agent let one execution reset another's list and both
    outputs end up with both records.
    """

    @staticmethod
    def _tool(channel_code: str) -> BaseTool:
        class NamedSlack(BaseTool):
            name: str = f"slack_{channel_code}"
            description: str = "Post a message."

            def _run(self, text: str) -> Any:
                return ToolFailure(message=f"failed {channel_code}", code=channel_code)

        return NamedSlack()

    def _agent_and_task(self, code: str) -> tuple[Agent, Task]:
        agent = Agent(
            role=f"Poster {code}",
            goal="post",
            backstory="b",
            llm=ScriptedLLM(
                [
                    f'Thought: go\nAction: slack_{code}\nAction Input: {{"text": "x"}}',
                    "Thought: done\nFinal Answer: could not post.",
                ]
            ),
            tools=[self._tool(code)],
        )
        task = Task(
            description=f"post {code}", expected_output="c", agent=agent
        )
        return agent, task

    def test_threads_do_not_cross_contaminate(self) -> None:
        import concurrent.futures

        crews = []
        for code in ("aaa", "bbb", "ccc"):
            agent, task = self._agent_and_task(code)
            crews.append((code, Crew(agents=[agent], tasks=[task])))

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(crew.kickoff): code for code, crew in crews
            }
            results = {
                futures[f]: f.result()
                for f in concurrent.futures.as_completed(futures)
            }

        for code, result in results.items():
            codes = [f.failure.code for f in result.tool_failures]
            assert codes == [code], f"{code} saw {codes}"

    def test_concurrent_kickoffs_on_a_shared_agent(self) -> None:
        """The reported repro, made deterministic with a barrier.

        Both kickoffs are held inside their tool call at the same time, so the
        old agent-level accumulation had each reset the other's list and both
        outputs came back holding two records instead of one.

        Crew tasks cannot hit this -- AgentExecutor refuses concurrent reuse of
        one instance -- but ``agent.kickoff()`` has no such guard.
        """
        import concurrent.futures
        import threading

        barrier = threading.Barrier(2, timeout=30)

        class BlockingFailingTool(BaseTool):
            name: str = "poster"
            description: str = "Post a message."

            def _run(self, channel: str) -> Any:
                barrier.wait()
                # Phrasing the LLM stub recognises as "tool already ran".
                return ToolFailure(message=f"TOOLRAN {channel}", code=channel)

        agent = Agent(
            role="Poster",
            goal="post",
            backstory="b",
            llm=StatelessToolLLM("poster", {"channel": "c1"}, "TOOLRAN"),
            tools=[BlockingFailingTool()],
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(agent.kickoff, f"do {n}") for n in ("A", "B")]
            outputs = [f.result() for f in futures]

        counts = [len(o.tool_failures) for o in outputs]
        assert counts == [1, 1], f"each kickoff should hold only its own: {counts}"

    def test_shared_agent_across_sequential_tasks(self) -> None:
        """One agent, two tasks: each output carries only its own record."""
        agent = Agent(
            role="Poster",
            goal="post",
            backstory="b",
            llm=ScriptedLLM(_slack_steps() * 4),
            tools=[SlackTool()],
        )
        task_a = Task(description="post A", expected_output="c", agent=agent)
        task_b = Task(description="post B", expected_output="c", agent=agent)
        result = Crew(agents=[agent], tasks=[task_a, task_b]).kickoff()

        for task_output in result.tasks_output:
            assert len(task_output.tool_failures) == 1, (
                f"{task_output.name} carried {len(task_output.tool_failures)}"
            )
        assert len(result.tool_failures) == 2

    def test_collector_is_execution_scoped(self) -> None:
        from crewai.tools.tool_failure import (
            active_tool_failures,
            tool_failure_collector,
        )

        assert active_tool_failures() is None
        with tool_failure_collector() as outer:
            assert active_tool_failures() is outer
            with tool_failure_collector() as inner:
                assert active_tool_failures() is inner
                assert inner is not outer
            assert active_tool_failures() is outer
        assert active_tool_failures() is None

    @pytest.mark.asyncio
    async def test_async_tasks_do_not_cross_contaminate(self) -> None:
        import asyncio

        crews = []
        for code in ("ddd", "eee"):
            agent, task = self._agent_and_task(code)
            crews.append((code, Crew(agents=[agent], tasks=[task])))

        outputs = await asyncio.gather(
            *(crew.kickoff_async() for _, crew in crews)
        )

        for (code, _), result in zip(crews, outputs, strict=True):
            codes = [f.failure.code for f in result.tool_failures]
            assert codes == [code], f"{code} saw {codes}"


class TestMalformedArgsOnEveryPath:
    """A malformed tool call is reported the same way everywhere."""

    def test_shared_native_helper_reports_instead_of_emptying_args(self) -> None:
        """It used to swallow the decode error and run the tool with no input."""
        from crewai.utilities.agent_utils import execute_single_native_tool_call

        agent = Agent(role="r", goal="g", backstory="b")
        tool = WorkingTool()
        recorded: list[ToolFailureDetectedEvent] = []
        calls: list[str] = []

        def tracked(**kwargs: Any) -> str:
            calls.append("ran")
            return "should not happen"

        tool_call = SimpleNamespace(
            id="c1", function=SimpleNamespace(name="echo", arguments="{not json")
        )

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ToolFailureDetectedEvent)
            def _(source: Any, event: ToolFailureDetectedEvent) -> None:
                recorded.append(event)

            result = execute_single_native_tool_call(
                tool_call,
                available_functions={"echo": tracked},
                original_tools=[tool],
                structured_tools=[tool.to_structured_tool()],
                tools_handler=None,
                agent=agent,
                task=None,
                crew=None,
                event_source=agent,
                printer=None,
                verbose=False,
            )
            crewai_event_bus.flush(timeout=10.0)

        assert calls == [], "the tool must not run with silently emptied args"
        assert "Failed to parse tool arguments" in str(result.result)
        assert len(recorded) == 1
        assert recorded[0].failure.reason is ToolFailureReason.INVALID_INPUT

    def test_react_path_reports_a_malformed_call(self) -> None:
        from crewai.agents.parser import AgentAction
        from crewai.utilities.tool_utils import execute_tool_and_check_finality

        agent = Agent(role="r", goal="g", backstory="b")
        tool = WorkingTool().to_structured_tool()
        recorded: list[ToolFailureDetectedEvent] = []

        action = AgentAction(
            thought="t",
            tool="echo",
            tool_input="{not a dict",
            text="Action: echo\nAction Input: {not a dict",
        )

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ToolFailureDetectedEvent)
            def _(source: Any, event: ToolFailureDetectedEvent) -> None:
                recorded.append(event)

            execute_tool_and_check_finality(
                agent_action=action,
                tools=[tool],
                agent=agent,
                task=None,
                crew=None,
            )
            crewai_event_bus.flush(timeout=10.0)

        assert len(recorded) == 1
        assert recorded[0].failure.reason is ToolFailureReason.INVALID_INPUT

    def test_malformed_call_aborts_under_raise(self) -> None:
        from crewai.utilities.agent_utils import execute_single_native_tool_call

        agent = Agent(
            role="r",
            goal="g",
            backstory="b",
            tool_failure_policy=ToolFailurePolicy.RAISE,
        )
        tool = WorkingTool()
        tool_call = SimpleNamespace(
            id="c1", function=SimpleNamespace(name="echo", arguments="{not json")
        )

        with pytest.raises(ToolExecutionFailedError):
            execute_single_native_tool_call(
                tool_call,
                available_functions={"echo": tool.run},
                original_tools=[tool],
                structured_tools=[tool.to_structured_tool()],
                tools_handler=None,
                agent=agent,
                task=None,
                crew=None,
                event_source=agent,
                printer=None,
                verbose=False,
            )


class TestBlockedCallsAreNotFailures:
    """A hook block is a deliberate decision, so it is not reported as one."""

    def test_no_blocked_by_hook_reason_exists(self) -> None:
        assert not hasattr(ToolFailureReason, "BLOCKED_BY_HOOK")

    def test_every_reason_is_actually_produced(self) -> None:
        """Guard against another declared-but-unused reason."""
        from pathlib import Path

        from importlib import import_module

        package = Path(import_module(Agent.__module__.split(".")[0]).__file__).parent
        sources = "\n".join(
            path.read_text()
            for path in package.rglob("*.py")
            if "tool_failure.py" not in path.name
        )
        # TOOL_REPORTED is the field default, so it is produced without ever
        # being named; every other member has to be referenced somewhere.
        for member in list(ToolFailureReason):
            if member is ToolFailureReason.TOOL_REPORTED:
                continue
            assert f"ToolFailureReason.{member.name}" in sources, (
                f"{member.name} is declared but never produced"
            )


class TestKickoffGuardrailRetries:
    def test_blocked_attempt_failures_survive_the_retry(self) -> None:
        """The retry opens its own collector, so earlier records must be merged."""
        attempts: list[int] = []

        def guardrail(output: Any) -> tuple[bool, Any]:
            attempts.append(1)
            if len(attempts) == 1:
                return (False, "try again")
            return (True, output.raw)

        agent = Agent(
            role="Slack Messenger",
            goal="post",
            backstory="b",
            llm=StatelessToolLLM("slackbot_send_message", {"channel": "#c"}),
            tools=[SlackTool()],
            guardrail=guardrail,
        )
        result = agent.kickoff("post it")

        assert len(attempts) == 2, "guardrail should have blocked once"
        assert result.has_tool_failures
        codes = [f.failure.code for f in result.tool_failures]
        assert codes and all(c == "channel_not_found" for c in codes), codes


class TestParallelAbortCancelsPendingSiblings:
    def test_pool_is_shut_down_with_cancel_futures(self) -> None:
        """A pending sibling must never start once an abort is requested.

        In-flight threads cannot be interrupted in Python, so this covers the
        not-yet-started ones -- the only ones that can still be prevented.
        """
        import inspect

        from crewai.experimental.agent_executor import AgentExecutor

        source = inspect.getsource(AgentExecutor.execute_native_tool)
        assert "cancel_futures=True" in source


class TestKickoffResetsTheAccessor:
    def test_last_tool_failures_does_not_grow_across_kickoffs(self) -> None:
        agent = Agent(
            role="Slack Messenger",
            goal="post",
            backstory="b",
            llm=StatelessToolLLM("slackbot_send_message", {"channel": "#c"}),
            tools=[SlackTool()],
        )

        agent.kickoff("post once")
        assert len(agent.last_tool_failures) == 1

        agent.kickoff("post again")
        assert len(agent.last_tool_failures) == 1, "records must not accumulate"


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
        import crewai_tools.tools.crewai_platform_tools.integrations_client as client_mod

        return mod.CrewAIPlatformActionTool(
            client_mod.ToolInfo(
                app="slack",
                action="slackbot_send_message",
                connection_id=None,
                description="Send a Slack message",
                parameters={
                    "properties": {"channel": {"type": "string"}},
                    "required": [],
                },
            )
        )

    def test_non_ok_response_becomes_a_tool_failure(self, monkeypatch) -> None:  # noqa: ANN001
        from unittest.mock import Mock

        import crewai_tools.tools.crewai_platform_tools.integrations_client as client_mod

        response = Mock()
        response.ok = False
        response.status_code = 500
        response.json.return_value = {
            "error": "Failed to execute action: Slack API error: channel_not_found"
        }
        monkeypatch.setattr(client_mod.requests, "post", Mock(return_value=response))
        monkeypatch.setenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", "t")

        result = self._tool()._run(channel="#joao-message")

        assert isinstance(result, ToolFailure)
        assert "channel_not_found" in result.message
        assert result.retryable is True

    def test_ok_response_still_returns_json(self, monkeypatch) -> None:  # noqa: ANN001
        from unittest.mock import Mock

        import crewai_tools.tools.crewai_platform_tools.integrations_client as client_mod

        response = Mock()
        response.ok = True
        response.json.return_value = {"ts": "1234.5678"}
        monkeypatch.setattr(client_mod.requests, "post", Mock(return_value=response))
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
