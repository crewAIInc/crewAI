from collections import defaultdict

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from crewai.agent import Agent
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput
from crewai.telemetry import Telemetry


class TaskEvaluationPydanticOutput(BaseModel):
    quality: float = Field(
        description="A score from 1 to 10 evaluating on completion, quality, and overall performance from the task_description and task_expected_output to the actual Task Output."
    )


class CrewEvaluator:
    """
    A class to evaluate the performance of the agents in the crew based on the tasks they have performed.

    Attributes:
        crew (Crew): The crew of agents to evaluate.
        openai_model_name (str): The model to use for evaluating the performance of the agents (for now ONLY OpenAI accepted).
        tasks_scores (defaultdict): A dictionary to store the scores of the agents for each task.
        iteration (int): The current iteration of the evaluation.
    """

    tasks_scores: defaultdict = defaultdict(list)
    run_execution_times: defaultdict = defaultdict(list)
    iteration: int = 0

    def __init__(self, crew, openai_model_name: str):
        self.crew = crew
        self.openai_model_name = openai_model_name
        self._telemetry = Telemetry()
        self._setup_for_evaluating()

    def _setup_for_evaluating(self) -> None:
        """Sets up the crew for evaluating."""
        for task in self.crew.tasks:
            task.callback = self.evaluate

    def _evaluator_agent(self):
        return Agent(
            role="Task Execution Evaluator",
            goal=(
                "Your goal is to evaluate the performance of the agents in the crew based on the tasks they have performed using score from 1 to 10 evaluating on completion, quality, and overall performance."
            ),
            backstory="Evaluator agent for crew evaluation with precise capabilities to evaluate the performance of the agents in the crew based on the tasks they have performed",
            verbose=False,
            llm=ChatOpenAI(model=self.openai_model_name),
        )

    def _evaluation_task(
        self, evaluator_agent: Agent, task_to_evaluate: Task, task_output: str
    ) -> Task:
        return Task(
            description=(
                "Based on the task description and the expected output, compare and evaluate the performance of the agents in the crew based on the Task Output they have performed using score from 1 to 10 evaluating on completion, quality, and overall performance."
                f"task_description: {task_to_evaluate.description} "
                f"task_expected_output: {task_to_evaluate.expected_output} "
                f"agent: {task_to_evaluate.agent.role if task_to_evaluate.agent else None} "
                f"agent_goal: {task_to_evaluate.agent.goal if task_to_evaluate.agent else None} "
                f"Task Output: {task_output}"
            ),
            expected_output="Evaluation Score from 1 to 10 based on the performance of the agents on the tasks",
            agent=evaluator_agent,
            output_pydantic=TaskEvaluationPydanticOutput,
        )

    def set_iteration(self, iteration: int) -> None:
        self.iteration = iteration

    def print_crew_evaluation_result(self) -> None:
        """
        Prints the evaluation result of the crew in a table.
        A Crew with 2 tasks using the command crewai test -n 2
        will output the following table:

                        Task Scores
                    (1-10 Higher is better)
            ┏━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
            ┃ Tasks/Crew ┃ Run 1 ┃ Run 2 ┃ Avg. Total ┃
            ┡━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
            │ Task 1     │ 10.0  │ 9.0   │ 9.5        │
            │ Task 2     │ 9.0   │ 9.0   │ 9.0        │
            │ Crew       │ 9.5   │ 9.0   │ 9.2        │
            └────────────┴───────┴───────┴────────────┘
        """
        task_averages = [
            sum(scores) / len(scores) for scores in zip(*self.tasks_scores.values())
        ]
        crew_average = sum(task_averages) / len(task_averages)

        # Create a table
        table = Table(title="Tasks Scores \n (1-10 Higher is better)")

        # Add columns for the table
        table.add_column("Tasks/Crew")
        for run in range(1, len(self.tasks_scores) + 1):
            table.add_column(f"Run {run}")
        table.add_column("Avg. Total")

        # Add rows for each task
        for task_index in range(len(task_averages)):
            task_scores = [
                self.tasks_scores[run][task_index]
                for run in range(1, len(self.tasks_scores) + 1)
            ]
            avg_score = task_averages[task_index]
            table.add_row(
                f"Task {task_index + 1}", *map(str, task_scores), f"{avg_score:.1f}"
            )

        # Add a row for the crew average
        crew_scores = [
            sum(self.tasks_scores[run]) / len(self.tasks_scores[run])
            for run in range(1, len(self.tasks_scores) + 1)
        ]
        table.add_row("Crew", *map(str, crew_scores), f"{crew_average:.1f}")

        run_exec_times = [
            int(sum(tasks_exec_times))
            for _, tasks_exec_times in self.run_execution_times.items()
        ]
        execution_time_avg = int(sum(run_exec_times) / len(run_exec_times))
        table.add_row(
            "Execution Time (s)",
            *map(str, run_exec_times),
            f"{execution_time_avg}",
        )
        # Display the table in the terminal
        console = Console()
        console.print(table)

    def evaluate(self, task_output: TaskOutput):
        """Evaluates the performance of the agents in the crew based on the tasks they have performed."""
        current_task = None
        for task in self.crew.tasks:
            if task.description == task_output.description:
                current_task = task
                break

        if not current_task or not task_output:
            raise ValueError(
                "Task to evaluate and task output are required for evaluation"
            )

        evaluator_agent = self._evaluator_agent()
        evaluation_task = self._evaluation_task(
            evaluator_agent, current_task, task_output.raw
        )

        evaluation_result = evaluation_task.execute_sync()

        if isinstance(evaluation_result.pydantic, TaskEvaluationPydanticOutput):
            self._test_result_span = self._telemetry.individual_test_result_span(
                self.crew,
                evaluation_result.pydantic.quality,
                current_task._execution_time,
                self.openai_model_name,
            )
            self.tasks_scores[self.iteration].append(evaluation_result.pydantic.quality)
            self.run_execution_times[self.iteration].append(
                current_task._execution_time
            )
        else:
            raise ValueError("Evaluation result is not in the expected format")
