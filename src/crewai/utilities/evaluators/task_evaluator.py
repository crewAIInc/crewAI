from typing import List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from crewai.utilities import Converter
from crewai.utilities.pydantic_schema_parser import PydanticSchemaParser

agentops = None
try:
    from agentops import track_agent
except ImportError:

    def track_agent(name):
        def noop(f):
            return f

        return noop


class Entity(BaseModel):
    name: str = Field(description="The name of the entity.")
    type: str = Field(description="The type of the entity.")
    description: str = Field(description="Description of the entity.")
    relationships: List[str] = Field(description="Relationships of the entity.")


class TaskEvaluation(BaseModel):
    suggestions: List[str] = Field(
        description="Suggestions to improve future similar tasks."
    )
    quality: float = Field(
        description="A score from 0 to 10 evaluating on completion, quality, and overall performance, all taking into account the task description, expected output, and the result of the task."
    )
    entities: List[Entity] = Field(
        description="Entities extracted from the task output."
    )


class TrainingTaskEvaluation(BaseModel):
    suggestions: List[str] = Field(
        description="Based on the Human Feedbacks and the comparison between Initial Outputs and Improved outputs provide action items based on human_feedback for future tasks."
    )
    quality: float = Field(
        description="A score from 0 to 10 evaluating on completion, quality, and overall performance from the improved output to the initial output based on the human feedback."
    )
    final_summary: str = Field(
        description="A step by step action items to improve the next Agent based on the human-feedback and improved output."
    )


@track_agent(name="Task Evaluator")
class TaskEvaluator:
    def __init__(self, original_agent):
        self.llm = original_agent.llm

    def evaluate(self, task, ouput) -> TaskEvaluation:
        evaluation_query = (
            f"Assess the quality of the task completed based on the description, expected output, and actual results.\n\n"
            f"Task Description:\n{task.description}\n\n"
            f"Expected Output:\n{task.expected_output}\n\n"
            f"Actual Output:\n{ouput}\n\n"
            "Please provide:\n"
            "- Bullet points suggestions to improve future similar tasks\n"
            "- A score from 0 to 10 evaluating on completion, quality, and overall performance"
            "- Entities extracted from the task output, if any, their type, description, and relationships"
        )

        instructions = "I'm gonna convert this raw text into valid JSON."

        if not self._is_gpt(self.llm):
            model_schema = PydanticSchemaParser(model=TaskEvaluation).get_schema()
            instructions = f"{instructions}\n\nThe json should have the following structure, with the following keys:\n{model_schema}"

        converter = Converter(
            llm=self.llm,
            text=evaluation_query,
            model=TaskEvaluation,
            instructions=instructions,
        )

        return converter.to_pydantic()

    def _is_gpt(self, llm) -> bool:
        return isinstance(llm, ChatOpenAI) and llm.openai_api_base is None

    def evaluate_training_data(
        self, training_data: dict, agent_id: str
    ) -> TrainingTaskEvaluation:
        """
        Evaluate the training data based on the llm output, human feedback, and improved output.

        Parameters:
            - training_data (dict): The training data to be evaluated.
            - agent_id (str): The ID of the agent.
        """

        output_training_data = training_data[agent_id]

        final_aggregated_data = ""
        for _, data in output_training_data.items():
            final_aggregated_data += (
                f"Initial Output:\n{data['initial_output']}\n\n"
                f"Human Feedback:\n{data['human_feedback']}\n\n"
                f"Improved Output:\n{data['improved_output']}\n\n"
            )

        evaluation_query = (
            "Assess the quality of the training data based on the llm output, human feedback , and llm output improved result.\n\n"
            f"{final_aggregated_data}"
            "Please provide:\n"
            "- Based on the Human Feedbacks and the comparison between Initial Outputs and Improved outputs provide action items based on human_feedback for future tasks\n"
            "- A score from 0 to 10 evaluating on completion, quality, and overall performance from the improved output to the initial output based on the human feedback\n"
        )
        instructions = "I'm gonna convert this raw text into valid JSON."

        if not self._is_gpt(self.llm):
            model_schema = PydanticSchemaParser(
                model=TrainingTaskEvaluation
            ).get_schema()
            instructions = f"{instructions}\n\nThe json should have the following structure, with the following keys:\n{model_schema}"

        converter = Converter(
            llm=self.llm,
            text=evaluation_query,
            model=TrainingTaskEvaluation,
            instructions=instructions,
        )

        pydantic_result = converter.to_pydantic()
        return pydantic_result
