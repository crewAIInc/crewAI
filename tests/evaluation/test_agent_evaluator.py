import pytest

from crewai.agent import Agent
from crewai.task import Task
from crewai.crew import Crew
from crewai.evaluation.agent_evaluator import AgentEvaluator
from crewai.evaluation.base_evaluator import AgentEvaluationResult
from crewai.evaluation import (
    GoalAlignmentEvaluator,
    SemanticQualityEvaluator,
    ToolSelectionEvaluator,
    ParameterExtractionEvaluator,
    ToolInvocationEvaluator,
    ReasoningEfficiencyEvaluator
)

from crewai.evaluation import create_default_evaluator
class TestAgentEvaluator:
    @pytest.fixture
    def mock_crew(self):
        agent = Agent(
            role="Test Agent",
            goal="Complete test tasks successfully",
            backstory="An agent created for testing purposes",
            allow_delegation=False,
            verbose=False
        )

        task = Task(
            description="Test task description",
            agent=agent,
            expected_output="Expected test output"
        )

        crew = Crew(
            agents=[agent],
            tasks=[task]
        )
        return crew

    def test_set_iteration(self):
        agent_evaluator = AgentEvaluator()

        agent_evaluator.set_iteration(3)
        assert agent_evaluator.iteration == 3

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_evaluate_current_iteration(self, mock_crew):
        agent_evaluator = AgentEvaluator(crew=mock_crew, evaluators=[GoalAlignmentEvaluator()])

        mock_crew.kickoff()

        results = agent_evaluator.evaluate_current_iteration()

        assert isinstance(results, dict)

        agent, = mock_crew.agents
        task, = mock_crew.tasks

        assert len(mock_crew.agents) == 1
        assert agent.role in results
        assert len(results[agent.role]) == 1

        result, = results[agent.role]
        assert isinstance(result, AgentEvaluationResult)

        assert result.agent_id == str(agent.id)
        assert result.task_id == str(task.id)

        goal_alignment, = result.metrics.values()
        assert goal_alignment.score == 5.0

        expected_feedback = "The agent's output demonstrates an understanding of the need for a comprehensive document"
        assert expected_feedback in goal_alignment.feedback

        assert goal_alignment.raw_response is not None
        assert '"score": 5' in goal_alignment.raw_response

    def test_create_default_evaluator(self, mock_crew):
        agent_evaluator = create_default_evaluator(crew=mock_crew)
        assert isinstance(agent_evaluator, AgentEvaluator)
        assert agent_evaluator.crew == mock_crew

        expected_types = [
            GoalAlignmentEvaluator,
            SemanticQualityEvaluator,
            ToolSelectionEvaluator,
            ParameterExtractionEvaluator,
            ToolInvocationEvaluator,
            ReasoningEfficiencyEvaluator
        ]

        assert len(agent_evaluator.evaluators) == len(expected_types)
        for evaluator, expected_type in zip(agent_evaluator.evaluators, expected_types):
            assert isinstance(evaluator, expected_type)
