from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.task_events import TaskEvaluationEvent
from crewai.llm import LLM
from crewai.utilities.converter import Converter
from crewai.utilities.pydantic_schema_parser import PydanticSchemaParser
from crewai.utilities.training_converter import TrainingConverter

if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.task import Task


class Entity(BaseModel):
    name: str = Field(description="The name of the entity.")
    type: str = Field(description="The type of the entity.")
    description: str = Field(description="Description of the entity.")
    relationships: list[str] = Field(description="Relationships of the entity.")


class TaskEvaluation(BaseModel):
    suggestions: list[str] = Field(
        description="Suggestions to improve future similar tasks."
    )
    quality: float = Field(
        description="A score from 0 to 10 evaluating on completion, quality, and overall performance, all taking into account the task description, expected output, and the result of the task."
    )
    entities: list[Entity] = Field(
        description="Entities extracted from the task output."
    )


class TrainingTaskEvaluation(BaseModel):
    suggestions: list[str] = Field(
        description="List of clear, actionable instructions derived from the Human Feedbacks to enhance the Agent's performance. Analyze the differences between Initial Outputs and Improved Outputs to generate specific action items for future tasks. Ensure all key and specific points from the human feedback are incorporated into these instructions."
    )
    quality: float = Field(
        description="A score from 0 to 10 evaluating on completion, quality, and overall performance from the improved output to the initial output based on the human feedback."
    )
    final_summary: str = Field(
        description="A step by step action items to improve the next Agent based on the human-feedback and improved output."
    )


class TaskEvaluator:
    """A class to evaluate the performance of an agent based on the tasks they have performed.

    Attributes:
        llm: The LLM to use for evaluation.
        original_agent: The agent to evaluate.
    """

    def __init__(self, original_agent: Agent) -> None:
        """Initializes the TaskEvaluator with the given LLM and agent.

        Args:
            original_agent: The agent to evaluate.
        """
        self.llm = cast(LLM, original_agent.llm)
        self.original_agent = original_agent

    def evaluate(self, task: Task, output: str) -> TaskEvaluation:
        """

        Args:
          task: The task to be evaluated.
          output: The output of the task.

        Returns:
            TaskEvaluation: The evaluation of the task.

        Notes:
            - Investigate the Converter.to_pydantic signature, returns BaseModel strictly?
        """
        crewai_event_bus.emit(
            self, TaskEvaluationEvent(evaluation_type="task_evaluation", task=task)
        )
        evaluation_query = (
            f"Assess the quality of the task completed based on the description, expected output, and actual results.\n\n"
            f"Task Description:\n{task.description}\n\n"
            f"Expected Output:\n{task.expected_output}\n\n"
            f"Actual Output:\n{output}\n\n"
            "Please provide:\n"
            "- Bullet points suggestions to improve future similar tasks\n"
            "- A score from 0 to 10 evaluating on completion, quality, and overall performance"
            "- Entities extracted from the task output, if any, their type, description, and relationships"
        )

        instructions = "Convert all responses into valid JSON output."

        if not self.llm.supports_function_calling():
            model_schema = PydanticSchemaParser(model=TaskEvaluation).get_schema()
            instructions = f"{instructions}\n\nReturn only valid JSON with the following schema:\n```json\n{model_schema}\n```"

        converter = Converter(
            llm=self.llm,
            text=evaluation_query,
            model=TaskEvaluation,
            instructions=instructions,
        )

        return cast(TaskEvaluation, converter.to_pydantic())

    def evaluate_training_data(
        self, training_data: dict, agent_id: str
    ) -> TrainingTaskEvaluation:
        """
        Evaluate the training data based on the llm output, human feedback, and improved output.

        Args:
            - training_data: The training data to be evaluated.
            - agent_id: The ID of the agent.

        Notes:
            - Investigate the Converter.to_pydantic signature, returns BaseModel strictly?
        """
        crewai_event_bus.emit(
            self, TaskEvaluationEvent(evaluation_type="training_data_evaluation")
        )

        output_training_data = training_data[agent_id]
        final_aggregated_data = ""

        for iteration, data in output_training_data.items():
            improved_output = data.get("improved_output")
            initial_output = data.get("initial_output")
            human_feedback = data.get("human_feedback")

            if not all([improved_output, initial_output, human_feedback]):
                missing_fields = [
                    field
                    for field in ["improved_output", "initial_output", "human_feedback"]
                    if not data.get(field)
                ]
                error_msg = (
                    f"Critical training data error: Missing fields ({', '.join(missing_fields)}) "
                    f"for agent {agent_id} in iteration {iteration}.\n"
                    "This indicates a broken training process. "
                    "Cannot proceed with evaluation.\n"
                    "Please check your training implementation."
                )
                raise ValueError(error_msg)

            final_aggregated_data += (
                f"Iteration: {iteration}\n"
                f"Initial Output:\n{initial_output}\n\n"
                f"Human Feedback:\n{human_feedback}\n\n"
                f"Improved Output:\n{improved_output}\n\n"
                "------------------------------------------------\n\n"
            )

        evaluation_query = (
            "Assess the quality of the training data based on the llm output, human feedback , and llm output improved result.\n\n"
            f"{final_aggregated_data}"
            "Please provide:\n"
            "- Provide a list of clear, actionable instructions derived from the Human Feedbacks to enhance the Agent's performance. Analyze the differences between Initial Outputs and Improved Outputs to generate specific action items for future tasks. Ensure all key and specificpoints from the human feedback are incorporated into these instructions.\n"
            "- A score from 0 to 10 evaluating on completion, quality, and overall performance from the improved output to the initial output based on the human feedback\n"
        )
        instructions = "I'm gonna convert this raw text into valid JSON."

        if not self.llm.supports_function_calling():
            model_schema = PydanticSchemaParser(
                model=TrainingTaskEvaluation
            ).get_schema()
            instructions = f"{instructions}\n\nThe json should have the following structure, with the following keys:\n{model_schema}"

        converter = TrainingConverter(
            llm=self.llm,
            text=evaluation_query,
            model=TrainingTaskEvaluation,
            instructions=instructions,
        )

        return cast(TrainingTaskEvaluation, converter.to_pydantic())
