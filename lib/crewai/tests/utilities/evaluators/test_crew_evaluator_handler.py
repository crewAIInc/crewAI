from unittest import mock

import pytest
from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.evaluators.crew_evaluator_handler import (
    CrewEvaluator,
    TaskEvaluationPydanticOutput,
)


class InternalCrewEvaluator:
    @pytest.fixture
    def crew_planner(self):
        agent = Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1")
        task = Task(
            description="Task 1",
            expected_output="Output 1",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task])

        return CrewEvaluator(crew, openai_model_name="gpt-4o-mini")

    def test_setup_for_evaluating(self, crew_planner):
        crew_planner._setup_for_evaluating()
        assert crew_planner.crew.tasks[0].callback == crew_planner.evaluate

    def test_set_iteration(self, crew_planner):
        crew_planner.set_iteration(1)
        assert crew_planner.iteration == 1

    def test_evaluator_agent(self, crew_planner):
        agent = crew_planner._evaluator_agent()
        assert agent.role == "Task Execution Evaluator"
        assert (
            agent.goal
            == "Your goal is to evaluate the performance of the agents in the crew based on the tasks they have performed using score from 1 to 10 evaluating on completion, quality, and overall performance."
        )
        assert (
            agent.backstory
            == "Evaluator agent for crew evaluation with precise capabilities to evaluate the performance of the agents in the crew based on the tasks they have performed"
        )
        assert agent.verbose is False
        assert agent.llm.model == "gpt-4o-mini"

    def test_evaluation_task(self, crew_planner):
        evaluator_agent = Agent(
            role="Evaluator Agent",
            goal="Evaluate the performance of the agents in the crew",
            backstory="Master in Evaluation",
        )
        task_to_evaluate = Task(
            description="Task 1",
            expected_output="Output 1",
            agent=Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1"),
        )
        task_output = "Task Output 1"
        task = crew_planner._evaluation_task(
            evaluator_agent, task_to_evaluate, task_output
        )

        assert task.description.startswith(
            "Based on the task description and the expected output, compare and evaluate the performance of the agents in the crew based on the Task Output they have performed using score from 1 to 10 evaluating on completion, quality, and overall performance."
        )

        assert task.agent == evaluator_agent
        assert (
            task.description
            == "Based on the task description and the expected output, compare and evaluate "
            "the performance of the agents in the crew based on the Task Output they have "
            "performed using score from 1 to 10 evaluating on completion, quality, and overall "
            "performance.task_description: Task 1 task_expected_output: Output 1 "
            "agent: Agent 1 agent_goal: Goal 1 Task Output: Task Output 1"
        )

    @mock.patch("crewai.utilities.evaluators.crew_evaluator_handler.Console")
    @mock.patch("crewai.utilities.evaluators.crew_evaluator_handler.Table")
    def test_print_crew_evaluation_result(self, table, console, crew_planner):
        crew_planner.tasks_scores = {
            1: [10, 9, 8],
            2: [9, 8, 7],
        }
        crew_planner.run_execution_times = {
            1: [24, 45, 66],
            2: [55, 33, 67],
        }

        crew_planner.crew.agents = [
            mock.Mock(role="Agent 1"),
            mock.Mock(role="Agent 2"),
        ]
        crew_planner.crew.tasks = [
            mock.Mock(
                agent=crew_planner.crew.agents[0], processed_by_agents=["Agent 1"]
            ),
            mock.Mock(
                agent=crew_planner.crew.agents[1], processed_by_agents=["Agent 2"]
            ),
        ]

        crew_planner.print_crew_evaluation_result()

        table.assert_has_calls(
            [
                mock.call(
                    title="Tasks Scores \n (1-10 Higher is better)", box=mock.ANY
                ),
                mock.call().add_column("Tasks/Crew/Agents", style="cyan"),
                mock.call().add_column("Run 1", justify="center"),
                mock.call().add_column("Run 2", justify="center"),
                mock.call().add_column("Avg. Total", justify="center"),
                mock.call().add_column("Agents", style="green"),
                mock.call().add_row("Task 1", "10.0", "9.0", "9.5", "- Agent 1"),
                mock.call().add_row("", "", "", "", "", ""),  # Blank row between tasks
                mock.call().add_row("Task 2", "9.0", "8.0", "8.5", "- Agent 2"),
                mock.call().add_row("Crew", "9.00", "8.00", "8.5", ""),
                mock.call().add_row("Execution Time (s)", "135", "155", "145", ""),
            ]
        )

        # Ensure the console prints the table
        console.assert_has_calls([mock.call(), mock.call().print(table())])

    def test_evaluate(self, crew_planner):
        task_output = TaskOutput(
            description="Task 1", agent=str(crew_planner.crew.agents[0])
        )

        with mock.patch.object(Task, "execute_sync") as execute:
            execute().pydantic = TaskEvaluationPydanticOutput(quality=9.5)
            crew_planner.evaluate(task_output)
            assert crew_planner.tasks_scores[0] == [9.5]


class TestCrewEvaluatorSkippedConditionalTasks:
    """Score bookkeeping when a ConditionalTask is skipped during Crew.test.

    A skipped ConditionalTask never runs ``Task.execute_sync``, so the
    evaluator callback installed by ``CrewEvaluator._setup_for_evaluating``
    is never invoked for it. Scores must stay attributed to the tasks that
    produced them instead of shifting into the skipped slot.
    """

    @pytest.fixture
    def evaluator(self):
        agent = Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1")
        tasks = [
            Task(description=f"Task {i}", expected_output=f"Output {i}", agent=agent)
            for i in range(1, 4)
        ]
        crew = Crew(agents=[agent], tasks=tasks)

        return CrewEvaluator(crew, openai_model_name="gpt-4o-mini")

    def _set_mock_tasks(self, evaluator):
        evaluator.crew.tasks = [
            mock.Mock(processed_by_agents=[f"Agent {i}"]) for i in range(1, 4)
        ]

    @mock.patch("crewai.utilities.evaluators.crew_evaluator_handler.Console")
    @mock.patch("crewai.utilities.evaluators.crew_evaluator_handler.Table")
    def test_print_result_with_middle_task_skipped_in_one_run(
        self, table, console, evaluator
    ):
        """A skipped middle task shows a placeholder and keeps other rows intact."""
        evaluator.tasks_scores = {1: [9.0, None, 8.0], 2: [9.0, 7.0, 8.0]}
        evaluator.run_execution_times = {1: [1.0, 1.0], 2: [1.0, 1.0, 1.0]}
        self._set_mock_tasks(evaluator)

        evaluator.print_crew_evaluation_result()

        table.assert_has_calls(
            [
                mock.call().add_row("Task 1", "9.0", "9.0", "9.0", "- Agent 1"),
                mock.call().add_row("", "", "", "", "", ""),
                mock.call().add_row("Task 2", "-", "7.0", "7.0", "- Agent 2"),
                mock.call().add_row("", "", "", "", "", ""),
                mock.call().add_row("Task 3", "8.0", "8.0", "8.0", "- Agent 3"),
                mock.call().add_row("Crew", "8.50", "8.00", "8.0", ""),
                mock.call().add_row("Execution Time (s)", "2", "3", "2", ""),
            ]
        )

    @mock.patch("crewai.utilities.evaluators.crew_evaluator_handler.Console")
    @mock.patch("crewai.utilities.evaluators.crew_evaluator_handler.Table")
    def test_print_result_with_last_task_skipped_in_one_run(
        self, table, console, evaluator
    ):
        """A skipped last task leaves its run score list short; no IndexError."""
        evaluator.tasks_scores = {1: [9.0, 7.0], 2: [9.0, 7.0, 10.0]}
        evaluator.run_execution_times = {1: [1.0, 1.0], 2: [1.0, 1.0, 1.0]}
        self._set_mock_tasks(evaluator)

        evaluator.print_crew_evaluation_result()

        table.assert_has_calls(
            [
                mock.call().add_row("Task 1", "9.0", "9.0", "9.0", "- Agent 1"),
                mock.call().add_row("", "", "", "", "", ""),
                mock.call().add_row("Task 2", "7.0", "7.0", "7.0", "- Agent 2"),
                mock.call().add_row("", "", "", "", "", ""),
                mock.call().add_row("Task 3", "-", "10.0", "10.0", "- Agent 3"),
                mock.call().add_row("Crew", "8.00", "8.67", "8.7", ""),
                mock.call().add_row("Execution Time (s)", "2", "3", "2", ""),
            ]
        )

    @mock.patch("crewai.utilities.evaluators.crew_evaluator_handler.Console")
    @mock.patch("crewai.utilities.evaluators.crew_evaluator_handler.Table")
    def test_print_result_with_no_scores_at_all(self, table, console, evaluator):
        """Every task skipped in every run must not crash on empty averages."""
        evaluator.tasks_scores = {1: [], 2: []}
        evaluator.run_execution_times = {1: [], 2: []}
        self._set_mock_tasks(evaluator)

        evaluator.print_crew_evaluation_result()

        table.assert_has_calls(
            [
                mock.call().add_row("Task 1", "-", "-", "-", "- Agent 1"),
                mock.call().add_row("", "", "", "", "", ""),
                mock.call().add_row("Task 2", "-", "-", "-", "- Agent 2"),
                mock.call().add_row("", "", "", "", "", ""),
                mock.call().add_row("Task 3", "-", "-", "-", "- Agent 3"),
                mock.call().add_row("Crew", "-", "-", "-", ""),
                mock.call().add_row("Execution Time (s)", "0", "0", "0", ""),
            ]
        )

    def _run_crew_test(self, task1_raw_outputs):
        """Run a 3-task crew (middle one conditional) through Crew.test.

        Returns the evaluator instance created by ``Crew.test``. Only the
        agent LLM call (``Agent.execute_task``) and the evaluator's scoring
        LLM call are mocked; the rest is the real ``Crew.test``/``kickoff``
        path, including the conditional-skip logic and the task callbacks.
        """
        quality_by_task = {"Task 1": 9.0, "Task 2": 7.0, "Task 3": 8.0}
        evaluator_instances = []

        class RecordingEvaluator(CrewEvaluator):
            def __init__(self, crew, eval_llm=None, **kwargs):
                super().__init__(crew, eval_llm, **kwargs)
                evaluator_instances.append(self)

            def _evaluation_task(
                self, evaluator_agent, task_to_evaluate, task_output
            ):
                result = mock.Mock()
                result.execute_sync.return_value.pydantic = (
                    TaskEvaluationPydanticOutput(
                        quality=quality_by_task[task_to_evaluate.description]
                    )
                )
                return result

        agent = Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1")
        task1 = Task(description="Task 1", expected_output="Output 1", agent=agent)
        task2 = ConditionalTask(
            description="Task 2",
            expected_output="Output 2",
            agent=agent,
            condition=lambda output: output.raw != "skip",
        )
        task3 = Task(description="Task 3", expected_output="Output 3", agent=agent)
        crew = Crew(agents=[agent], tasks=[task1, task2, task3])

        raw_outputs = iter(task1_raw_outputs)

        def fake_execute_task(task, context=None, tools=None):
            if task.description == "Task 1":
                return next(raw_outputs)
            return f"done {task.description}"

        with (
            mock.patch("crewai.crew.CrewEvaluator", RecordingEvaluator),
            mock.patch("crewai.Agent.execute_task", side_effect=fake_execute_task),
        ):
            crew.test(2, "gpt-4o-mini")

        return evaluator_instances[0]

    @pytest.mark.block_network(allowed_hosts=[r"^127\.0\.0\.1$"])
    @mock.patch("crewai.utilities.evaluators.crew_evaluator_handler.Console")
    @mock.patch("crewai.utilities.evaluators.crew_evaluator_handler.Table")
    def test_crew_test_with_conditional_task_skipped_in_one_run(
        self, table, console
    ):
        """Scores stay attributed to the tasks that produced them.

        The middle ConditionalTask is skipped in run 1 (previous output raw
        is ``"skip"``) and executed in run 2.
        """
        evaluator = self._run_crew_test(["skip", "go"])

        assert dict(evaluator.tasks_scores) == {
            1: [9.0, None, 8.0],
            2: [9.0, 7.0, 8.0],
        }

        table.assert_has_calls(
            [
                mock.call().add_row("Task 1", "9.0", "9.0", "9.0", "- Agent 1"),
                mock.call().add_row("", "", "", "", "", ""),
                mock.call().add_row("Task 2", "-", "7.0", "7.0", "- Agent 1"),
                mock.call().add_row("", "", "", "", "", ""),
                mock.call().add_row("Task 3", "8.0", "8.0", "8.0", "- Agent 1"),
                mock.call().add_row("Crew", "8.50", "8.00", "8.0", ""),
                mock.call().add_row("Execution Time (s)", "0", "0", "0", ""),
            ]
        )

    @pytest.mark.block_network(allowed_hosts=[r"^127\.0\.0\.1$"])
    @mock.patch("crewai.utilities.evaluators.crew_evaluator_handler.Console")
    @mock.patch("crewai.utilities.evaluators.crew_evaluator_handler.Table")
    def test_crew_test_with_conditional_task_skipped_in_all_runs(
        self, table, console
    ):
        """A ConditionalTask skipped in every run shows placeholders, no crash."""
        evaluator = self._run_crew_test(["skip", "skip"])

        assert dict(evaluator.tasks_scores) == {
            1: [9.0, None, 8.0],
            2: [9.0, None, 8.0],
        }

        table.assert_has_calls(
            [
                mock.call().add_row("Task 1", "9.0", "9.0", "9.0", "- Agent 1"),
                mock.call().add_row("", "", "", "", "", ""),
                mock.call().add_row("Task 2", "-", "-", "-", ""),
                mock.call().add_row("", "", "", "", "", ""),
                mock.call().add_row("Task 3", "8.0", "8.0", "8.0", "- Agent 1"),
                mock.call().add_row("Crew", "8.50", "8.50", "8.5", ""),
                mock.call().add_row("Execution Time (s)", "0", "0", "0", ""),
            ]
        )
