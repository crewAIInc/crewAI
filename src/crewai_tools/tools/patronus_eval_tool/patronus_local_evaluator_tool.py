from typing import TYPE_CHECKING, Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from patronus import Client, EvaluationResult

try:
    import patronus

    PYPATRONUS_AVAILABLE = True
except ImportError:
    PYPATRONUS_AVAILABLE = False


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
    description: str = (
        "This tool is used to evaluate the model input and output using custom function evaluators."
    )
    args_schema: Type[BaseModel] = FixedLocalEvaluatorToolSchema
    client: "Client" = None
    evaluator: str
    evaluated_model_gold_answer: str

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        patronus_client: "Client" = None,
        evaluator: str = "",
        evaluated_model_gold_answer: str = "",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if PYPATRONUS_AVAILABLE:
            self.client = patronus_client
            self.evaluator = evaluator
            self.evaluated_model_gold_answer = evaluated_model_gold_answer
            self._generate_description()
            print(
                f"Updating evaluator and gold_answer to: {self.evaluator}, {self.evaluated_model_gold_answer}"
            )
        else:
            raise ImportError(
                "The 'patronus' package is not installed. "
                "Please install it by running `uv add patronus` to use PatronusLocalEvaluatorTool."
            )

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        if not PYPATRONUS_AVAILABLE:
            raise ImportError(
                "The 'patronus' package is not installed. "
                "Please install it by running `uv add patronus` to use PatronusLocalEvaluatorTool."
            )

        evaluated_model_input = kwargs.get("evaluated_model_input")
        evaluated_model_output = kwargs.get("evaluated_model_output")
        evaluated_model_retrieved_context = kwargs.get(
            "evaluated_model_retrieved_context"
        )
        evaluated_model_gold_answer = self.evaluated_model_gold_answer
        evaluator = self.evaluator

        result: "EvaluationResult" = self.client.evaluate(
            evaluator=evaluator,
            evaluated_model_input=evaluated_model_input,
            evaluated_model_output=evaluated_model_output,
            evaluated_model_retrieved_context=evaluated_model_retrieved_context,
            evaluated_model_gold_answer=evaluated_model_gold_answer,
            tags={},  # Optional metadata, supports arbitrary key-value pairs
        )
        output = f"Evaluation result: {result.pass_}, Explanation: {result.explanation}"
        return output


try:
    # Only rebuild if the class hasn't been initialized yet
    if not hasattr(PatronusLocalEvaluatorTool, "_model_rebuilt"):
        PatronusLocalEvaluatorTool.model_rebuild()
        PatronusLocalEvaluatorTool._model_rebuilt = True
except Exception:
    pass
