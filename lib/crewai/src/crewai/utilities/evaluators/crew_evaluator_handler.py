from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from rich.box import HEAVY_EDGE
from rich.console import Console
from rich.table import Table

from crewai.agent import Agent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.crew_events import CrewTestResultEvent
from crewai.llms.base_llm import BaseLLM
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput


if TYPE_CHECKING:
    from crewai.crew import Crew


class TaskEvaluationPydanticOutput(BaseModel):
    quality: float = Field(
        description="A score from 1 to 10 evaluating on completion, quality, and overall performance from the task_description and task_expected_output to the actual Task Output."
    )


class CrewEvaluator:
    """A class to evaluate the performance of the agents in the crew based on the tasks they have performed.

    Attributes:
        crew: The crew of agents to evaluate.
        tasks_scores: A dictionary to store the scores of the agents for each task.
            Scores are stored per task position; a skipped task (e.g. a
            ConditionalTask whose condition was not met) leaves a ``None``
            placeholder at its position.
        run_execution_times: A dictionary to store execution times for each run.
        iteration: The current iteration of the evaluation.
    """

    def __init__(
        self,
        crew: Crew,
        eval_llm: BaseLLM | str | None = None,
        openai_model_name: str | None = None,
        llm: BaseLLM | str | None = None,
    ) -> None:
        self.crew = crew
        self.llm = eval_llm
        self.tasks_scores: defaultdict[int, list[float | None]] = defaultdict(list)
        self.run_execution_times: defaultdict[int, list[float]] = defaultdict(list)
        self.iteration: int = 0
        self._setup_for_evaluating()

    def _setup_for_evaluating(self) -> None:
        """Sets up the crew for evaluating."""
        for task in self.crew.tasks:
            task.callback = self.evaluate

    def _evaluator_agent(self) -> Agent:
        return Agent(
            role="Task Execution Evaluator",
            goal=(
                "Your goal is to evaluate the performance of the agents in the crew based on the tasks they have performed using score from 1 to 10 evaluating on completion, quality, and overall performance."
            ),
            backstory="Evaluator agent for crew evaluation with precise capabilities to evaluate the performance of the agents in the crew based on the tasks they have performed",
            verbose=False,
            llm=self.llm,
        )

    @staticmethod
    def _evaluation_task(
        evaluator_agent: Agent, task_to_evaluate: Task, task_output: str
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
        """Sets the current iteration of the evaluation.

        Args:
            iteration: The current iteration number.
        """
        self.iteration = iteration

    def print_crew_evaluation_result(
        self, token_usage: list[dict[str, Any]] | None = None
    ) -> None:
        """
        Prints the evaluation result of the crew in a table.
        A Crew with 2 tasks using the command crewai test -n 3
        will output the following table:

                        Tasks Scores
                    (1-10 Higher is better)
        ┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ Tasks/Crew/Agents  ┃ Run 1 ┃ Run 2 ┃ Run 3 ┃ Avg. Total ┃ Agents                       ┃
        ┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ Task 1             │ 9.0   │ 10.0  │ 9.0   │ 9.3        │ - AI LLMs Senior Researcher  │
        │                    │       │       │       │            │ - AI LLMs Reporting Analyst  │
        │                    │       │       │       │            │                              │
        │ Task 2             │ 9.0   │ 9.0   │ 9.0   │ 9.0        │ - AI LLMs Senior Researcher  │
        │                    │       │       │       │            │ - AI LLMs Reporting Analyst  │
        │                    │       │       │       │            │                              │
        │ Crew               │ 9.0   │ 9.5   │ 9.0   │ 9.2        │                              │
        │ Execution Time (s) │ 42    │ 79    │ 52    │ 57         │                              │
        └────────────────────┴───────┴───────┴───────┴────────────┴──────────────────────────────┘

        Tasks skipped in a run (e.g. a ConditionalTask whose condition was not
        met) are shown as ``-`` and are excluded from averages.
        """
        runs = sorted(self.tasks_scores)
        run_scores_by_run = [self.tasks_scores[run] for run in runs]

        task_averages: list[float | None] = []
        for task_index in range(len(self.crew.tasks)):
            scores_at_position = (
                run_scores[task_index]
                for run_scores in run_scores_by_run
                if task_index < len(run_scores)
            )
            scores = [score for score in scores_at_position if score is not None]
            task_averages.append(
                sum(scores) / len(scores) if scores else None,
            )

        valid_task_averages = [avg for avg in task_averages if avg is not None]
        crew_average = (
            sum(valid_task_averages) / len(valid_task_averages)
            if valid_task_averages
            else None
        )

        table = Table(title="Tasks Scores \n (1-10 Higher is better)", box=HEAVY_EDGE)

        table.add_column("Tasks/Crew/Agents", style="cyan")
        for run_index in range(1, len(runs) + 1):
            table.add_column(f"Run {run_index}", justify="center")
        table.add_column("Avg. Total", justify="center")
        table.add_column("Agents", style="green")

        for task_index, task in enumerate(self.crew.tasks):
            task_scores = [
                run_scores[task_index] if task_index < len(run_scores) else None
                for run_scores in run_scores_by_run
            ]
            avg_score = task_averages[task_index]
            agents = list(task.processed_by_agents)

            table.add_row(
                f"Task {task_index + 1}",
                *[
                    f"{score:.1f}" if score is not None else "-"
                    for score in task_scores
                ],
                f"{avg_score:.1f}" if avg_score is not None else "-",
                f"- {agents[0]}" if agents else "",
            )

            for agent in agents[1:]:
                table.add_row("", "", "", "", "", f"- {agent}")

            if task_index < len(self.crew.tasks) - 1:
                table.add_row("", "", "", "", "", "")

        crew_scores: list[float | None] = []
        for run_scores in run_scores_by_run:
            valid_scores = [score for score in run_scores if score is not None]
            crew_scores.append(
                sum(valid_scores) / len(valid_scores) if valid_scores else None
            )
        table.add_row(
            "Crew",
            *[f"{score:.2f}" if score is not None else "-" for score in crew_scores],
            f"{crew_average:.1f}" if crew_average is not None else "-",
            "",
        )

        run_exec_times = [
            int(sum(tasks_exec_times))
            for _, tasks_exec_times in self.run_execution_times.items()
        ]
        execution_time_avg = (
            int(sum(run_exec_times) / len(run_exec_times)) if run_exec_times else 0
        )
        table.add_row(
            "Execution Time (s)", *map(str, run_exec_times), f"{execution_time_avg}", ""
        )

        console = Console()
        console.print("\n")
        console.print(table)

    def evaluate(self, task_output: TaskOutput) -> None:
        """Evaluates the performance of the agents in the crew based on the tasks they have performed.

        Args:
            task_output: The output of the task to evaluate.
        """
        current_task = None
        current_task_index = -1
        for task_index, task in enumerate(self.crew.tasks):
            if task.output is task_output:
                current_task = task
                current_task_index = task_index
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
            quality_score = evaluation_result.pydantic.quality
            if quality_score is None:
                raise ValueError("Evaluation quality score cannot be None")

            crewai_event_bus.emit(
                self.crew,
                CrewTestResultEvent(
                    quality=quality_score,
                    execution_duration=current_task.execution_duration,
                    model=getattr(self.llm, "model", str(self.llm)),
                    crew_name=self.crew.name,
                    crew=self.crew,
                ),
            )
            # Store the score at the task's position so tasks skipped in this
            # run (which never invoke this callback) don't shift the scores
            # of the tasks that did execute.
            run_scores = self.tasks_scores[self.iteration]
            while len(run_scores) <= current_task_index:
                run_scores.append(None)
            run_scores[current_task_index] = quality_score
            if current_task.execution_duration is not None:
                self.run_execution_times[self.iteration].append(
                    current_task.execution_duration
                )
        else:
            raise ValueError("Evaluation result is not in the expected format")
