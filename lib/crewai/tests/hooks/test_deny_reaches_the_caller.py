"""A deny raised inside a run has to reach whoever started the run.

The sibling propagation tests all call the frame that makes the model call
directly, so they prove a deny escapes *that* function and nothing about what
its callers do with it. Every regression in this area has lived one or more
frames up, in a broad ``except Exception`` that turned the deny into a degraded
result. These tests drive the public entry points instead, and count model calls
so a deny that gets retried reads as a failure rather than as a pass.
"""

from __future__ import annotations

from typing import Any

from crewai.agent import Agent
from crewai.agents.step_executor import StepExecutor
from crewai.crew import Crew
from crewai.experimental.agent_executor import AgentExecutor
from crewai.hooks.dispatch import HookAborted, InterceptionPoint, clear_all, on
from crewai.lite_agent import LiteAgent
from crewai.llms.base_llm import BaseLLM
from crewai.task import Task
from crewai.utilities.planning_types import TodoItem
from crewai.utilities.step_execution_context import StepExecutionContext
from crewai.utilities.types import LLMMessage
import pytest

from ..utils import wait_for_event_handlers


class StubProviderLLM(BaseLLM):
    """Answers without a network, dispatching the before hooks like a provider."""

    def __init__(self, fail_first_call: bool = False) -> None:
        super().__init__(model="stub")
        self.fail_first_call = fail_first_call
        self.answered = 0

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
        **kwargs: Any,
    ) -> str:
        formatted: list[LLMMessage] = (
            messages
            if isinstance(messages, list)
            else [{"role": "user", "content": messages}]
        )
        self._invoke_before_llm_call_hooks(formatted, from_agent)
        self.answered += 1
        if self.fail_first_call and self.answered == 1:
            raise RuntimeError("the provider blipped")
        return "Thought: done\nFinal Answer: ok"

    def supports_function_calling(self) -> bool:
        return False


class DenyingMemory:
    """Stands in for the memory whose own model call a hook denied."""

    read_only = False
    root_scope = None

    def __init__(self, deny_on: str, error: Exception | None = None) -> None:
        self.deny_on = deny_on
        self.error = error or HookAborted(
            reason="memory is off limits", source="policy"
        )
        self.touched: list[str] = []

    def _step(self, name: str) -> None:
        self.touched.append(name)
        if name == self.deny_on:
            raise self.error

    def drain_writes(self) -> None:
        pass

    def recall(self, *args: Any, **kwargs: Any) -> list[Any]:
        self._step("recall")
        return []

    def extract_memories(self, *args: Any, **kwargs: Any) -> list[str]:
        self._step("extract_memories")
        return ["a memory"]

    def remember_many(self, *args: Any, **kwargs: Any) -> None:
        self._step("remember_many")

    def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


class StubKnowledge:
    def query(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


class DenyingStepExecutor:
    """Stands in for the per-step executor whose own model call was denied."""

    def __init__(self) -> None:
        self.executed = 0

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.executed += 1
        raise HookAborted(reason="no model calls allowed", source="policy")


@pytest.fixture(autouse=True)
def _clean_hooks():
    clear_all()
    yield
    # A kickoff emits events whose handlers run on a pool; draining them here
    # keeps a straggler from firing inside an unrelated test.
    wait_for_event_handlers()
    clear_all()


def deny_nth_model_call(n: int) -> list[str]:
    """Deny the nth model call of the run, returning the log of attempts."""
    attempts: list[str] = []

    @on(InterceptionPoint.PRE_MODEL_CALL)
    def gate(_ctx: Any) -> None:
        attempts.append("attempt")
        if len(attempts) == n:
            raise HookAborted(reason="no model calls allowed", source="policy")

    return attempts


def build_agent(**kwargs: Any) -> Agent:
    return Agent(
        role="Worker",
        goal="Answer",
        backstory="You answer.",
        llm=StubProviderLLM(),
        **kwargs,
    )


def build_crew(agent: Agent, **task_kwargs: Any) -> Crew:
    task = Task(
        description="Say ok", expected_output="ok", agent=agent, **task_kwargs
    )
    return Crew(agents=[agent], tasks=[task])


def run_agent_kickoff() -> Any:
    return build_agent().kickoff("say ok")


def run_agent_kickoff_with_planning() -> Any:
    return build_agent(planning=True).kickoff("say ok")


def run_crew_kickoff() -> Any:
    return build_crew(build_agent()).kickoff()


def run_crew_kickoff_with_planning() -> Any:
    return build_crew(build_agent(planning=True)).kickoff()


def run_crew_kickoff_with_knowledge() -> Any:
    agent = build_agent()
    agent.knowledge = StubKnowledge()
    return build_crew(agent).kickoff()


def run_lite_agent_kickoff() -> Any:
    return LiteAgent(
        role="Worker", goal="Answer", backstory="You answer.", llm=StubProviderLLM()
    ).kickoff("say ok")


@pytest.mark.parametrize(
    "entry_point",
    [
        run_agent_kickoff,
        run_agent_kickoff_with_planning,
        run_crew_kickoff,
        run_crew_kickoff_with_planning,
        run_crew_kickoff_with_knowledge,
        run_lite_agent_kickoff,
    ],
    ids=[
        "agent.kickoff",
        "agent.kickoff-planning",
        "crew.kickoff",
        "crew.kickoff-planning",
        "crew.kickoff-knowledge",
        "lite_agent.kickoff",
    ],
)
def test_a_denied_model_call_reaches_the_caller(entry_point):
    attempts = deny_nth_model_call(1)

    with pytest.raises(HookAborted):
        entry_point()

    assert len(attempts) == 1


def test_a_denied_guardrail_stops_the_crew_instead_of_retrying_the_task():
    attempts = deny_nth_model_call(2)
    crew = build_crew(build_agent(), guardrail="The answer must be polite")

    with pytest.raises(HookAborted):
        crew.kickoff()

    # the answer, then the denied validation, and nothing after it
    assert len(attempts) == 2


@pytest.mark.parametrize("deny_on", ["recall", "extract_memories"])
def test_a_denied_memory_step_reaches_the_caller(deny_on):
    memory = DenyingMemory(deny_on)
    agent = build_agent()
    agent.memory = memory

    with pytest.raises(HookAborted):
        agent.kickoff("say ok")

    assert memory.touched.count(deny_on) == 1


def test_a_denied_memory_save_stops_the_crew_instead_of_retrying_the_task():
    memory = DenyingMemory("extract_memories")
    agent = build_agent()
    agent.memory = memory

    with pytest.raises(HookAborted):
        build_crew(agent).kickoff()

    assert memory.touched.count("extract_memories") == 1


def build_step_executor(llm: StubProviderLLM) -> StepExecutor:
    return StepExecutor(llm=llm, tools=[], agent=build_agent())


def a_step() -> tuple[TodoItem, StepExecutionContext]:
    return (
        TodoItem(step_number=1, description="Say ok"),
        StepExecutionContext(task_description="Say ok", task_goal="ok"),
    )


def test_a_denied_step_stops_the_plan_instead_of_reporting_a_failed_step():
    attempts = deny_nth_model_call(1)
    todo, context = a_step()

    with pytest.raises(HookAborted):
        build_step_executor(StubProviderLLM()).execute(todo, context)

    assert len(attempts) == 1


def test_an_ordinary_step_failure_still_reports_a_failed_step():
    todo, context = a_step()

    result = build_step_executor(StubProviderLLM(fail_first_call=True)).execute(
        todo, context
    )

    assert result.success is False


@pytest.mark.asyncio
async def test_a_denied_parallel_step_reaches_the_caller():
    agent = build_agent()
    executor = AgentExecutor(agent=agent, llm=agent.llm, task=None)
    executor.state.todos.items = [
        TodoItem(step_number=1, description="first"),
        TodoItem(step_number=2, description="second"),
    ]
    step_executor = DenyingStepExecutor()
    object.__setattr__(executor, "_ensure_step_executor", lambda: step_executor)

    with pytest.raises(HookAborted):
        await executor.execute_todos_parallel()


def test_an_ordinary_model_failure_is_still_retried():
    agent = Agent(
        role="Worker",
        goal="Answer",
        backstory="You answer.",
        llm=StubProviderLLM(fail_first_call=True),
    )

    assert "ok" in str(build_crew(agent).kickoff())


def test_an_ordinary_memory_failure_still_degrades():
    memory = DenyingMemory("extract_memories", error=RuntimeError("storage is down"))
    agent = build_agent()
    agent.memory = memory

    assert str(agent.kickoff("say ok")) == "ok"
