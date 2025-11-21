import threading

import pytest
from crewai.agent import Agent
from crewai.crew import Crew
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentEvaluationCompletedEvent,
    AgentEvaluationFailedEvent,
    AgentEvaluationStartedEvent,
)
from crewai.experimental.evaluation import (
    EvaluationScore,
    GoalAlignmentEvaluator,
    MetricCategory,
    ParameterExtractionEvaluator,
    ReasoningEfficiencyEvaluator,
    SemanticQualityEvaluator,
    ToolInvocationEvaluator,
    ToolSelectionEvaluator,
    create_default_evaluator,
)
from crewai.experimental.evaluation.agent_evaluator import AgentEvaluator
from crewai.experimental.evaluation.base_evaluator import (
    AgentEvaluationResult,
    BaseEvaluator,
)
from crewai.task import Task


class TestAgentEvaluator:
    @pytest.fixture
    def mock_crew(self):
        agent = Agent(
            role="Test Agent",
            goal="Complete test tasks successfully",
            backstory="An agent created for testing purposes",
            allow_delegation=False,
            verbose=False,
        )

        task = Task(
            description="Test task description",
            agent=agent,
            expected_output="Expected test output",
        )

        crew = Crew(agents=[agent], tasks=[task])
        return crew

    def test_set_iteration(self):
        agent_evaluator = AgentEvaluator(agents=[])

        agent_evaluator.set_iteration(3)
        assert agent_evaluator._execution_state.iteration == 3

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_evaluate_current_iteration(self, mock_crew):
        from crewai.events.types.task_events import TaskCompletedEvent

        agent_evaluator = AgentEvaluator(
            agents=mock_crew.agents, evaluators=[GoalAlignmentEvaluator()]
        )

        task_completed_condition = threading.Condition()
        task_completed = False

        @crewai_event_bus.on(TaskCompletedEvent)
        async def on_task_completed(source, event):
            # TaskCompletedEvent fires AFTER evaluation results are stored
            nonlocal task_completed
            with task_completed_condition:
                task_completed = True
                task_completed_condition.notify()

        mock_crew.kickoff()

        with task_completed_condition:
            assert task_completed_condition.wait_for(
                lambda: task_completed, timeout=5
            ), "Timeout waiting for task completion"

        results = agent_evaluator.get_evaluation_results()

        assert isinstance(results, dict)

        (agent,) = mock_crew.agents
        (task,) = mock_crew.tasks

        assert len(mock_crew.agents) == 1
        assert agent.role in results
        assert len(results[agent.role]) == 1

        (result,) = results[agent.role]
        assert isinstance(result, AgentEvaluationResult)

        assert result.agent_id == str(agent.id)
        assert result.task_id == str(task.id)

        (goal_alignment,) = result.metrics.values()
        assert goal_alignment.score == 5.0

        expected_feedback = "The agent's output demonstrates an understanding of the need for a comprehensive document outlining task"
        assert expected_feedback in goal_alignment.feedback

        assert goal_alignment.raw_response is not None
        assert '"score": 5' in goal_alignment.raw_response

    def test_create_default_evaluator(self, mock_crew):
        agent_evaluator = create_default_evaluator(agents=mock_crew.agents)
        assert isinstance(agent_evaluator, AgentEvaluator)
        assert agent_evaluator.agents == mock_crew.agents

        expected_types = [
            GoalAlignmentEvaluator,
            SemanticQualityEvaluator,
            ToolSelectionEvaluator,
            ParameterExtractionEvaluator,
            ToolInvocationEvaluator,
            ReasoningEfficiencyEvaluator,
        ]

        assert len(agent_evaluator.evaluators) == len(expected_types)
        for evaluator, expected_type in zip(
            agent_evaluator.evaluators, expected_types, strict=False
        ):
            assert isinstance(evaluator, expected_type)

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_eval_specific_agents_from_crew(self, mock_crew):
        from crewai.events.types.task_events import TaskCompletedEvent

        agent = Agent(
            role="Test Agent Eval",
            goal="Complete test tasks successfully",
            backstory="An agent created for testing purposes",
        )
        task = Task(
            description="Test task description",
            agent=agent,
            expected_output="Expected test output",
        )
        mock_crew.agents.append(agent)
        mock_crew.tasks.append(task)

        events = {}
        results_condition = threading.Condition()
        results_ready = False

        agent_evaluator = AgentEvaluator(
            agents=[agent], evaluators=[GoalAlignmentEvaluator()]
        )

        @crewai_event_bus.on(AgentEvaluationStartedEvent)
        async def capture_started(source, event):
            if event.agent_id == str(agent.id):
                events["started"] = event

        @crewai_event_bus.on(AgentEvaluationCompletedEvent)
        async def capture_completed(source, event):
            if event.agent_id == str(agent.id):
                events["completed"] = event

        @crewai_event_bus.on(AgentEvaluationFailedEvent)
        def capture_failed(source, event):
            events["failed"] = event

        @crewai_event_bus.on(TaskCompletedEvent)
        async def on_task_completed(source, event):
            nonlocal results_ready
            if event.task and event.task.id == task.id:
                while not agent_evaluator.get_evaluation_results().get(agent.role):
                    pass
                with results_condition:
                    results_ready = True
                    results_condition.notify()

        mock_crew.kickoff()

        with results_condition:
            assert results_condition.wait_for(
                lambda: results_ready, timeout=5
            ), "Timeout waiting for evaluation results"

        assert events.keys() == {"started", "completed"}
        assert events["started"].agent_id == str(agent.id)
        assert events["started"].agent_role == agent.role
        assert events["started"].task_id == str(task.id)
        assert events["started"].iteration == 1

        assert events["completed"].agent_id == str(agent.id)
        assert events["completed"].agent_role == agent.role
        assert events["completed"].task_id == str(task.id)
        assert events["completed"].iteration == 1
        assert events["completed"].metric_category == MetricCategory.GOAL_ALIGNMENT
        assert isinstance(events["completed"].score, EvaluationScore)
        assert events["completed"].score.score == 5.0

        results = agent_evaluator.get_evaluation_results()

        assert isinstance(results, dict)
        assert len(results.keys()) == 1
        (result,) = results[agent.role]
        assert isinstance(result, AgentEvaluationResult)

        assert result.agent_id == str(agent.id)
        assert result.task_id == str(task.id)

        (goal_alignment,) = result.metrics.values()
        assert goal_alignment.score == 5.0

        expected_feedback = "The agent provided a thorough guide on how to conduct a test task but failed to produce specific expected output"
        assert expected_feedback in goal_alignment.feedback

        assert goal_alignment.raw_response is not None
        assert '"score": 5' in goal_alignment.raw_response

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_failed_evaluation(self, mock_crew):
        (agent,) = mock_crew.agents
        (task,) = mock_crew.tasks

        events = {}
        started_event = threading.Event()
        failed_event = threading.Event()

        @crewai_event_bus.on(AgentEvaluationStartedEvent)
        def capture_started(source, event):
            events["started"] = event
            started_event.set()

        @crewai_event_bus.on(AgentEvaluationCompletedEvent)
        def capture_completed(source, event):
            events["completed"] = event

        @crewai_event_bus.on(AgentEvaluationFailedEvent)
        def capture_failed(source, event):
            events["failed"] = event
            failed_event.set()

        class FailingEvaluator(BaseEvaluator):
            metric_category = MetricCategory.GOAL_ALIGNMENT

            def evaluate(self, agent, task, execution_trace, final_output):
                raise ValueError("Forced evaluation failure")

        agent_evaluator = AgentEvaluator(
            agents=[agent], evaluators=[FailingEvaluator()]
        )
        mock_crew.kickoff()

        assert started_event.wait(timeout=5), "Timeout waiting for started event"
        assert failed_event.wait(timeout=5), "Timeout waiting for failed event"

        assert events.keys() == {"started", "failed"}
        assert events["started"].agent_id == str(agent.id)
        assert events["started"].agent_role == agent.role
        assert events["started"].task_id == str(task.id)
        assert events["started"].iteration == 1

        assert events["failed"].agent_id == str(agent.id)
        assert events["failed"].agent_role == agent.role
        assert events["failed"].task_id == str(task.id)
        assert events["failed"].iteration == 1
        assert events["failed"].error == "Forced evaluation failure"

        results = agent_evaluator.get_evaluation_results()
        (result,) = results[agent.role]
        assert isinstance(result, AgentEvaluationResult)

        assert result.agent_id == str(agent.id)
        assert result.task_id == str(task.id)

        assert result.metrics == {}
