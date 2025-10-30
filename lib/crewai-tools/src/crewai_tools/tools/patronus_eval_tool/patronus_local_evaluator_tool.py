from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from patronus import Client, EvaluationResult  # type: ignore[import-untyped]

try:
    import patronus  # noqa: F401

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
    description: str = "This tool is used to evaluate the model input and output using custom function evaluators."
    args_schema: type[BaseModel] = FixedLocalEvaluatorToolSchema
    client: Client = None
    evaluator: str
    evaluated_model_gold_answer: str

    model_config = ConfigDict(arbitrary_types_allowed=True)
    package_dependencies: list[str] = Field(default_factory=lambda: ["patronus"])

    def __init__(
        self,
        patronus_client: Client = None,
        evaluator: str = "",
        evaluated_model_gold_answer: str = "",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.evaluator = evaluator
        self.evaluated_model_gold_answer = evaluated_model_gold_answer
        self._initialize_patronus(patronus_client)

    def _initialize_patronus(self, patronus_client: Client) -> None:
        try:
            if PYPATRONUS_AVAILABLE:
                self.client = patronus_client
                self._generate_description()
            else:
                raise ImportError
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'patronus' package. Would you like to install it?"
            ):
                import subprocess

                try:
                    subprocess.run(["uv", "add", "patronus"], check=True)  # noqa: S607
                    self.client = patronus_client
                    self._generate_description()
                except subprocess.CalledProcessError as e:
                    raise ImportError("Failed to install 'patronus' package") from e
            else:
                raise ImportError(
                    "`patronus` package not found, please run `uv add patronus`"
                ) from None

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

        result: EvaluationResult = self.client.evaluate(
            evaluator=evaluator,
            evaluated_model_input=evaluated_model_input,
            evaluated_model_output=evaluated_model_output,
            evaluated_model_retrieved_context=evaluated_model_retrieved_context,
            evaluated_model_gold_answer=evaluated_model_gold_answer,
            tags={},  # Optional metadata, supports arbitrary key-value pairs
        )
        return f"Evaluation result: {result.pass_}, Explanation: {result.explanation}"


try:
    # Only rebuild if the class hasn't been initialized yet
    if not hasattr(PatronusLocalEvaluatorTool, "_model_rebuilt"):
        PatronusLocalEvaluatorTool.model_rebuild()
        PatronusLocalEvaluatorTool._model_rebuilt = True  # type: ignore[attr-defined]
except Exception:  # noqa: S110
    pass
