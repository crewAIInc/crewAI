from collections import defaultdict

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from crewai.agent import Agent
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput


class TaskEvaluationPydanticOutput(BaseModel):
    quality: float = Field(
        description="A score from 0 to 10 evaluating on completion, quality, and overall performance from the task_description and task_expected_output to the actual Task Output."
    )


class CrewEvaluator:
    tasks_scores = defaultdict(list)
    iteration = 0

    def __init__(self, crew, model: str):
        self.crew = crew
        self.model = model
        self._setup_for_evaluating()

    def _setup_for_evaluating(self) -> None:
        """Sets up the crew for evaluating."""
        for task in self.crew.tasks:
            task.callback = self.evaluate

    def set_iteration(self, iteration: int) -> None:
        self.iteration = iteration

    def _evaluator_agent(self):
        return Agent(
            role="Task Execution Evaluator",
            goal=(
                "Your goal is to evaluate the performance of the agents in the crew based on the tasks they have performed."
            ),
            backstory="Evaluator agent for crew evaluation",
            verbose=False,
            llm=ChatOpenAI(model=self.model),
        )

    def _evaluation_task(
        self, evaluator_agent: Agent, task_to_evaluate: Task, task_output: str
    ) -> Task:
        return Task(
            description=(
                "Based on the task description and the expected output, compare and evaluate the performance of the agents in the crew based on the Task Output they have performed."
                f"task_description: {task_to_evaluate.description}"
                f"task_expected_output: {task_to_evaluate.expected_output}"
                f"agent: {task_to_evaluate.agent.role if task_to_evaluate.agent else None}"
                f"agent_goal: {task_to_evaluate.agent.goal if task_to_evaluate.agent else None}"
                f"Task Output: {task_output}"
            ),
            expected_output="Evaluation score based on the performance of the agents on the tasks",
            agent=evaluator_agent,
            output_pydantic=TaskEvaluationPydanticOutput,
        )

    def print_crew_evaluation_result(self) -> None:
        self.tasks_scores
        results = self.tasks_scores

        task_averages = [sum(scores) / len(scores) for scores in zip(*results.values())]
        crew_average = sum(task_averages) / len(task_averages)

        # Create a table
        table = Table(title="Task Scores")

        # Add columns for the table
        table.add_column("Task")
        for run in range(1, len(results) + 1):
            table.add_column(f"Run {run}")
        table.add_column("Avg. Total")

        # Add rows for each task
        for task_index in range(len(task_averages)):
            task_scores = [
                results[run][task_index] for run in range(1, len(results) + 1)
            ]
            avg_score = task_averages[task_index]
            table.add_row(
                f"Task {task_index + 1}", *map(str, task_scores), f"{avg_score:.1f}"
            )

        # Add a row for the crew average
        crew_scores = [
            sum(results[run]) / len(results[run]) for run in range(1, len(results) + 1)
        ]
        table.add_row("Crew", *map(str, crew_scores), f"{crew_average:.1f}")

        # Display the table in the terminal
        console = Console()
        console.print(table)

    def evaluate(self, task_output: TaskOutput):
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
        self.tasks_scores[self.iteration].append(evaluation_result.pydantic.quality)
