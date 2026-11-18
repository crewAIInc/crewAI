from unittest.mock import AsyncMock, Mock, patch

import pytest

from crewai import Agent, Crew, Task
from crewai.agents.agent_executor import AgentExecutor
from crewai.llms.base_llm import BaseLLM
from crewai.tasks.task_output import TaskOutput
from crewai.types.usage_metrics import UsageMetrics
from crewai.utilities.cost_calculator import calculate_token_cost, get_model_pricing
from crewai.utilities.exceptions.task_security_exceptions import (
    IterationLimitExceeded,
    PreTaskAttestationError,
    SpendLimitExceeded,
)


class MockLLM(BaseLLM):
    """Mock LLM for security and spend testing."""

    def __init__(self, response="Attested output", model="gpt-4o", usage=None):
        super().__init__(model=model)
        self.response = response
        self.custom_usage = usage or UsageMetrics(
            prompt_tokens=1000, completion_tokens=200
        )

    def call(self, messages, **kwargs):
        return self.response

    def get_token_usage_summary(self):
        return self.custom_usage


def _create_agent(
    role: str = "security_agent", model: str = "gpt-4o", usage=None, response="Attested output"
):
    llm = MockLLM(response=response, model=model, usage=usage)
    return Agent(
        role=role,
        goal="Test security goal",
        backstory="Test security backstory",
        llm=llm,
    )


class TestPreTaskAttestation:
    """Tests for pre-flight task attestation hooks."""

    def test_pre_task_callback_success(self):
        called = []

        def guard(task, agent, context):
            called.append((task.description, context.get("task_id")))
            return True

        agent = _create_agent()

        task = Task(
            description="Process security batch",
            expected_output="Done",
            agent=agent,
            pre_task_callback=guard,
        )

        with patch.object(Agent, "execute_task", return_value="Attested output"):
            output = task.execute_sync(agent=agent)

        assert isinstance(output, TaskOutput)
        assert output.raw == "Attested output"
        assert len(called) == 1
        assert called[0][0] == "Process security batch"

    def test_pre_task_callback_fails_and_raises(self):
        def deny_guard(task, agent, context):
            return False

        agent = _create_agent()
        task = Task(
            description="High privilege task",
            expected_output="Done",
            agent=agent,
            pre_task_callback=deny_guard,
        )

        with pytest.raises(PreTaskAttestationError) as exc_info:
            task.execute_sync(agent=agent)

        assert "Pre-flight task attestation failed" in str(exc_info.value)
        assert "High privilege task" in str(exc_info.value)
        assert exc_info.value.task == task

    def test_pre_task_callback_raising_exception_not_swallowed(self):
        def crashing_guard(task, agent, context):
            raise PermissionError("Access token revoked")

        agent = _create_agent()
        task = Task(
            description="Classified task",
            expected_output="Done",
            agent=agent,
            pre_task_callback=crashing_guard,
        )

        with pytest.raises(PermissionError) as exc_info:
            task.execute_sync(agent=agent)

        assert "Access token revoked" in str(exc_info.value)

    def test_crew_level_pre_task_callback_enforcement(self):
        def crew_guard(task, agent, context):
            if "forbidden" in task.description.lower():
                return False
            return True

        agent = _create_agent()

        t1 = Task(description="Safe task", expected_output="Done", agent=agent)
        t2 = Task(description="Forbidden task", expected_output="Done", agent=agent)

        crew = Crew(
            agents=[agent],
            tasks=[t1, t2],
            pre_task_callback=crew_guard,
        )

        with patch.object(Agent, "execute_task", return_value="Attested output"):
            # Safe task executes cleanly
            out1 = t1.execute_sync(agent=agent)
            assert out1.raw == "Attested output"

            # Forbidden task fails attestation
            with pytest.raises(PreTaskAttestationError) as exc_info:
                t2.execute_sync(agent=agent)
            assert "Forbidden task" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_pre_task_callback(self):
        async def async_guard(task, agent, context):
            return False

        agent = _create_agent()

        task = Task(
            description="Async sensitive task",
            expected_output="Done",
            agent=agent,
            pre_task_callback=async_guard,
        )

        with patch.object(Agent, "aexecute_task", new_callable=AsyncMock) as mock_aexec:
            mock_aexec.return_value = "Attested output"
            with pytest.raises(PreTaskAttestationError):
                await task.aexecute_sync(agent=agent)


class TestSpendContainment:
    """Tests for spend containment and budget limits."""

    def test_pricing_catalog_lookup(self):
        gpt4o = get_model_pricing("gpt-4o")
        assert gpt4o["input_cost_per_token"] == pytest.approx(2.5e-6)
        assert gpt4o["output_cost_per_token"] == pytest.approx(1e-5)

        mini = get_model_pricing("openai/gpt-4o-mini")
        assert mini["input_cost_per_token"] == pytest.approx(1.5e-7)

    def test_calculate_token_cost(self):
        metrics = UsageMetrics(prompt_tokens=1000, completion_tokens=500)
        # gpt-4o: 1000 * 2.5e-6 ($0.0025) + 500 * 1e-5 ($0.005) = $0.0075
        cost = calculate_token_cost(metrics, "gpt-4o")
        assert cost == pytest.approx(0.0075)

    def test_task_budget_exceeded(self):
        initial_metrics = UsageMetrics(prompt_tokens=0, completion_tokens=0)
        final_metrics = UsageMetrics(prompt_tokens=50_000, completion_tokens=10_000)

        agent = _create_agent(model="gpt-4o", usage=initial_metrics)

        def fake_execute(*args, **kwargs):
            agent.llm.custom_usage = final_metrics
            return "Done"

        task = Task(
            description="Token heavy task",
            expected_output="Done",
            agent=agent,
            max_execution_budget_usd=0.05,  # $0.05 budget ceiling (50k tokens on gpt-4o > $0.125)
        )

        with patch.object(Agent, "execute_task", side_effect=fake_execute):
            with pytest.raises(SpendLimitExceeded) as exc_info:
                task.execute_sync(agent=agent)

        assert "exceeded maximum budget" in str(exc_info.value)
        assert exc_info.value.max_budget == 0.05
        assert exc_info.value.current_spend > 0.05

    def test_crew_spend_containment_preflight_check(self):
        # 500,000 prompt tokens on gpt-4o = $1.25
        high_usage = UsageMetrics(prompt_tokens=500_000, completion_tokens=10_000)
        agent = _create_agent(model="gpt-4o", usage=high_usage)

        t1 = Task(description="Step 1", expected_output="Done", agent=agent)
        crew = Crew(
            agents=[agent],
            tasks=[t1],
            max_execution_budget_usd=0.50,
        )

        with pytest.raises(SpendLimitExceeded) as exc_info:
            crew._execute_tasks([t1])

        assert "Crew execution spend" in str(exc_info.value)
        assert exc_info.value.max_budget == 0.50


class TestIterationCeilings:
    """Tests for deterministic iteration limit enforcement."""

    def test_agent_executor_iteration_limit_exceeded(self):
        executor = AgentExecutor()
        executor.state.iterations = 3

        mock_task = Mock(spec=Task)
        mock_task.description = "Looping task"
        mock_task.max_task_iterations = 3
        mock_task.max_execution_budget_usd = None
        executor.task = mock_task
        executor.crew = None

        with pytest.raises(IterationLimitExceeded) as exc_info:
            executor.check_max_iterations()

        assert "reached maximum iteration ceiling of 3" in str(exc_info.value)
        assert exc_info.value.iterations == 3
        assert exc_info.value.max_iterations == 3

    def test_agent_executor_crew_iteration_limit_inheritance(self):
        executor = AgentExecutor()
        executor.state.iterations = 5

        mock_task = Mock(spec=Task)
        mock_task.description = "Task under crew ceiling"
        mock_task.max_task_iterations = None
        mock_task.max_execution_budget_usd = None
        executor.task = mock_task

        mock_crew = Mock(spec=Crew)
        mock_crew.max_task_iterations = 5
        mock_crew.max_execution_budget_usd = None
        executor.crew = mock_crew

        with pytest.raises(IterationLimitExceeded) as exc_info:
            executor.check_max_iterations()

        assert "reached maximum iteration ceiling of 5" in str(exc_info.value)
        assert exc_info.value.iterations == 5
        assert exc_info.value.max_iterations == 5

    def test_agent_executor_spend_limit_exceeded(self):
        executor = AgentExecutor()
        executor.state.iterations = 1

        mock_task = Mock(spec=Task)
        mock_task.description = "Expensive tool task"
        mock_task.max_task_iterations = 10
        mock_task.max_execution_budget_usd = 0.01
        executor.task = mock_task
        executor.crew = None

        # Mock spend calculation to return $0.05
        executor._calculate_current_task_spend = Mock(return_value=0.05)

        with pytest.raises(SpendLimitExceeded) as exc_info:
            executor.check_max_iterations()

        assert "exceeded budget ceiling of $0.0100" in str(exc_info.value)
        assert exc_info.value.current_spend == 0.05
        assert exc_info.value.max_budget == 0.01
