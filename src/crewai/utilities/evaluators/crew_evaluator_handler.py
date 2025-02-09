from collections import defaultdict
from typing import Any, Dict, List, Union

from pydantic import (
    BaseModel,
    Field,
    InstanceOf,
    PrivateAttr,
    model_validator,
)
from rich.box import HEAVY_EDGE
from rich.console import Console
from rich.table import Table

from crewai.agent import Agent
from crewai.llm import LLM
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput
from crewai.telemetry import Telemetry


class TaskEvaluationPydanticOutput(BaseModel):
    quality: float = Field(
        description="A score from 1 to 10 evaluating on completion, quality, and overall performance from the task_description and task_expected_output to the actual Task Output."
    )


class CrewEvaluator(BaseModel):
    """
    A class to evaluate the performance of the agents in the crew based on the tasks they have performed.

    Attributes:
        crew (Crew): The crew of agents to evaluate.
        llm (Union[str, InstanceOf[LLM], Any]): The language model to use for evaluating the performance of the agents.
        tasks_scores (defaultdict): A dictionary to store the scores of the agents for each task.
        iteration (int): The current iteration of the evaluation.
    """

    crew: Any = Field(description="The crew of agents to evaluate.")
    llm: Union[str, InstanceOf[LLM], Any] = Field(
        description="Language model that will run the evaluation."
    )
    tasks_scores: Dict[int, List[float]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Dictionary to store the scores of the agents for each task."
    )
    run_execution_times: Dict[int, List[int]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Dictionary to store execution times for each run."
    )
    iteration: int = Field(
        default=0,
        description="Current iteration of the evaluation."
    )

    @model_validator(mode="after")
    def validate_llm(self):
        """Validates that the LLM is properly configured."""
        if not self.llm:
            raise ValueError("LLM configuration is required")
        return self

    _telemetry: Telemetry = PrivateAttr(default_factory=Telemetry)

    def __init__(self, crew, llm: Union[str, InstanceOf[LLM], Any]):
        # Initialize Pydantic model with validated fields
        super().__init__(crew=crew, llm=llm)
        self._setup_for_evaluating()

    @model_validator(mode="before")
    def init_llm(cls, values):
        """Initialize LLM before Pydantic validation."""
        llm = values.get("llm")
        try:
            if isinstance(llm, str):
                values["llm"] = LLM(model=llm)
            elif isinstance(llm, LLM):
                values["llm"] = llm
            else:
                # For any other type, attempt to extract relevant attributes
                llm_params = {
                    "model": getattr(llm, "model_name", None)
                    or getattr(llm, "deployment_name", None)
                    or str(llm),
                    "temperature": getattr(llm, "temperature", None),
                    "max_tokens": getattr(llm, "max_tokens", None),
                    "timeout": getattr(llm, "timeout", None),
                }
                # Remove None values
                llm_params = {k: v for k, v in llm_params.items() if v is not None}
                values["llm"] = LLM(**llm_params)
        except Exception as e:
            raise ValueError(f"Invalid LLM configuration: {str(e)}") from e
        return values

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
            llm=self.llm,
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
        """
        task_averages = [
            sum(scores) / len(scores) for scores in zip(*self.tasks_scores.values())
        ]
        crew_average = sum(task_averages) / len(task_averages)

        table = Table(title="Tasks Scores \n (1-10 Higher is better)", box=HEAVY_EDGE)

        table.add_column("Tasks/Crew/Agents", style="cyan")
        for run in range(1, len(self.tasks_scores) + 1):
            table.add_column(f"Run {run}", justify="center")
        table.add_column("Avg. Total", justify="center")
        table.add_column("Agents", style="green")

        for task_index, task in enumerate(self.crew.tasks):
            task_scores = [
                self.tasks_scores[run][task_index]
                for run in range(1, len(self.tasks_scores) + 1)
            ]
            avg_score = task_averages[task_index]
            agents = list(task.processed_by_agents)

            # Add the task row with the first agent
            table.add_row(
                f"Task {task_index + 1}",
                *[f"{score:.1f}" for score in task_scores],
                f"{avg_score:.1f}",
                f"- {agents[0]}" if agents else "",
            )

            # Add rows for additional agents
            for agent in agents[1:]:
                table.add_row("", "", "", "", "", f"- {agent}")

            # Add a blank separator row if it's not the last task
            if task_index < len(self.crew.tasks) - 1:
                table.add_row("", "", "", "", "", "")

        # Add Crew and Execution Time rows
        crew_scores = [
            sum(self.tasks_scores[run]) / len(self.tasks_scores[run])
            for run in range(1, len(self.tasks_scores) + 1)
        ]
        table.add_row(
            "Crew",
            *[f"{score:.2f}" for score in crew_scores],
            f"{crew_average:.1f}",
            "",
        )

        run_exec_times = [
            int(sum(tasks_exec_times))
            for _, tasks_exec_times in self.run_execution_times.items()
        ]
        execution_time_avg = int(sum(run_exec_times) / len(run_exec_times))
        table.add_row(
            "Execution Time (s)", *map(str, run_exec_times), f"{execution_time_avg}", ""
        )

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
                self.llm.model if isinstance(self.llm, LLM) else self.llm,
            )
            self.tasks_scores[self.iteration].append(evaluation_result.pydantic.quality)
            self.run_execution_times[self.iteration].append(
                current_task._execution_time
            )
        else:
            raise ValueError("Evaluation result is not in the expected format")
