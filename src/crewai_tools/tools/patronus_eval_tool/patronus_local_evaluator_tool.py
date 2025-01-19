from typing import Any, Type

from crewai.tools import BaseTool
from patronus import Client
from pydantic import BaseModel, Field


class FixedLocalEvaluatorToolSchema(BaseModel):
    evaluated_model_input: str = Field(
        ..., description="The agent's task description in simple text"
    )
    evaluated_model_output: str = Field(
        ..., description="The agent's output of the task"
    )
    evaluated_model_retrieved_context: str = Field(
        ..., description="The agent's context"
    )
    evaluated_model_gold_answer: str = Field(
        ..., description="The agent's gold answer only if available"
    )
    evaluator: str = Field(..., description="The registered local evaluator")


class PatronusLocalEvaluatorTool(BaseTool):
    name: str = "Patronus Local Evaluator Tool"
    evaluator: str = "The registered local evaluator"
    evaluated_model_gold_answer: str = "The agent's gold answer"
    description: str = "This tool is used to evaluate the model input and output using custom function evaluators."
    client: Any = None
    args_schema: Type[BaseModel] = FixedLocalEvaluatorToolSchema

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        patronus_client: Client,
        evaluator: str,
        evaluated_model_gold_answer: str,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.client = patronus_client
        if evaluator:
            self.evaluator = evaluator
            self.evaluated_model_gold_answer = evaluated_model_gold_answer
            self.description = f"This tool calls the Patronus Evaluation API that takes an additional argument in addition to the following new argument:\n evaluators={evaluator}, evaluated_model_gold_answer={evaluated_model_gold_answer}"
            self._generate_description()
            print(
                f"Updating judge evaluator, gold_answer to: {self.evaluator}, {self.evaluated_model_gold_answer}"
            )

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        evaluated_model_input = kwargs.get("evaluated_model_input")
        evaluated_model_output = kwargs.get("evaluated_model_output")
        evaluated_model_retrieved_context = kwargs.get(
            "evaluated_model_retrieved_context"
        )
        evaluated_model_gold_answer = self.evaluated_model_gold_answer
        evaluator = self.evaluator

        result = self.client.evaluate(
            evaluator=evaluator,
            evaluated_model_input=(
                evaluated_model_input
                if isinstance(evaluated_model_input, str)
                else evaluated_model_input.get("description")
            ),
            evaluated_model_output=(
                evaluated_model_output
                if isinstance(evaluated_model_output, str)
                else evaluated_model_output.get("description")
            ),
            evaluated_model_retrieved_context=(
                evaluated_model_retrieved_context
                if isinstance(evaluated_model_retrieved_context, str)
                else evaluated_model_retrieved_context.get("description")
            ),
            evaluated_model_gold_answer=(
                evaluated_model_gold_answer
                if isinstance(evaluated_model_gold_answer, str)
                else evaluated_model_gold_answer.get("description")
            ),
            tags={},  # Optional metadata, supports arbitrary kv pairs
        )
        output = f"Evaluation result: {result.pass_}, Explanation: {result.explanation}"
        return output
