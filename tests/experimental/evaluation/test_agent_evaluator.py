import pytest

from crewai.agent import Agent
from crewai.task import Task
from crewai.crew import Crew
from crewai.experimental.evaluation.agent_evaluator import AgentEvaluator
from crewai.experimental.evaluation.base_evaluator import AgentEvaluationResult
from crewai.experimental.evaluation import (
    GoalAlignmentEvaluator,
    SemanticQualityEvaluator,
    ToolSelectionEvaluator,
    ParameterExtractionEvaluator,
    ToolInvocationEvaluator,
    ReasoningEfficiencyEvaluator
)

from crewai.experimental.evaluation import create_default_evaluator

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
        agent_evaluator = AgentEvaluator(agents=[])

        agent_evaluator.set_iteration(3)
        assert agent_evaluator._execution_state.iteration == 3

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_evaluate_current_iteration(self, mock_crew):
        agent_evaluator = AgentEvaluator(agents=mock_crew.agents, evaluators=[GoalAlignmentEvaluator()])

        mock_crew.kickoff()

        results = agent_evaluator.get_evaluation_results()

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
            ReasoningEfficiencyEvaluator
        ]

        assert len(agent_evaluator.evaluators) == len(expected_types)
        for evaluator, expected_type in zip(agent_evaluator.evaluators, expected_types):
            assert isinstance(evaluator, expected_type)

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_eval_lite_agent(self):
        agent = Agent(
            role="Test Agent",
            goal="Complete test tasks successfully",
            backstory="An agent created for testing purposes",
        )
        agent_evaluator = AgentEvaluator(agents=[agent], evaluators=[GoalAlignmentEvaluator()])

        agent.kickoff(messages="Complete this task successfully")

        results = agent_evaluator.get_evaluation_results()

        assert isinstance(results, dict)

        result, = results[agent.role]
        assert isinstance(result, AgentEvaluationResult)

        assert result.agent_id == str(agent.id)
        assert result.task_id == "lite_task"

        goal_alignment, = result.metrics.values()
        assert goal_alignment.score == 2.0

        expected_feedback = "The agent did not demonstrate a clear understanding of the task goal, which is to complete test tasks successfully"
        assert expected_feedback in goal_alignment.feedback

        assert goal_alignment.raw_response is not None
        assert '"score": 2' in goal_alignment.raw_response

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_eval_specific_agents_from_crew(self, mock_crew):
        agent = Agent(
            role="Test Agent Eval",
            goal="Complete test tasks successfully",
            backstory="An agent created for testing purposes",
        )
        task = Task(
            description="Test task description",
            agent=agent,
            expected_output="Expected test output"
        )
        mock_crew.agents.append(agent)
        mock_crew.tasks.append(task)

        agent_evaluator = AgentEvaluator(agents=[agent], evaluators=[GoalAlignmentEvaluator()])

        mock_crew.kickoff()

        results = agent_evaluator.get_evaluation_results()

        assert isinstance(results, dict)
        assert len(results.keys()) == 1
        result, = results[agent.role]
        assert isinstance(result, AgentEvaluationResult)

        assert result.agent_id == str(agent.id)
        assert result.task_id == str(task.id)

        goal_alignment, = result.metrics.values()
        assert goal_alignment.score == 5.0

        expected_feedback = "The agent provided a thorough guide on how to conduct a test task but failed to produce specific expected output"
        assert expected_feedback in goal_alignment.feedback

        assert goal_alignment.raw_response is not None
        assert '"score": 5' in goal_alignment.raw_response