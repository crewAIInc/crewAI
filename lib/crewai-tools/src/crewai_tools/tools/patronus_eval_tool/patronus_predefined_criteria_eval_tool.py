import json
import os
from typing import Any, Dict, List, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class FixedBaseToolSchema(BaseModel):
    evaluated_model_input: Dict = Field(
        ..., description="The agent's task description in simple text"
    )
    evaluated_model_output: Dict = Field(
        ..., description="The agent's output of the task"
    )
    evaluated_model_retrieved_context: Dict = Field(
        ..., description="The agent's context"
    )
    evaluated_model_gold_answer: Dict = Field(
        ..., description="The agent's gold answer only if available"
    )
    evaluators: List[Dict[str, str]] = Field(
        ...,
        description="List of dictionaries containing the evaluator and criteria to evaluate the model input and output. An example input for this field: [{'evaluator': '[evaluator-from-user]', 'criteria': '[criteria-from-user]'}]",
    )


class PatronusPredefinedCriteriaEvalTool(BaseTool):
    """
    PatronusEvalTool is a tool to automatically evaluate and score agent interactions.

    Results are logged to the Patronus platform at app.patronus.ai
    """

    name: str = "Call Patronus API tool for evaluation of model inputs and outputs"
    description: str = """This tool calls the Patronus Evaluation API that takes the following arguments:"""
    evaluate_url: str = "https://api.patronus.ai/v1/evaluate"
    args_schema: Type[BaseModel] = FixedBaseToolSchema
    evaluators: List[Dict[str, str]] = []

    def __init__(self, evaluators: List[Dict[str, str]], **kwargs: Any):
        super().__init__(**kwargs)
        if evaluators:
            self.evaluators = evaluators
            self.description = f"This tool calls the Patronus Evaluation API that takes an additional argument in addition to the following new argument:\n evaluators={evaluators}"
            self._generate_description()
            print(f"Updating judge criteria to: {self.evaluators}")

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        evaluated_model_input = kwargs.get("evaluated_model_input")
        evaluated_model_output = kwargs.get("evaluated_model_output")
        evaluated_model_retrieved_context = kwargs.get(
            "evaluated_model_retrieved_context"
        )
        evaluated_model_gold_answer = kwargs.get("evaluated_model_gold_answer")
        evaluators = self.evaluators

        headers = {
            "X-API-KEY": os.getenv("PATRONUS_API_KEY"),
            "accept": "application/json",
            "content-type": "application/json",
        }

        data = {
            "evaluated_model_input": (
                evaluated_model_input
                if isinstance(evaluated_model_input, str)
                else evaluated_model_input.get("description")
            ),
            "evaluated_model_output": (
                evaluated_model_output
                if isinstance(evaluated_model_output, str)
                else evaluated_model_output.get("description")
            ),
            "evaluated_model_retrieved_context": (
                evaluated_model_retrieved_context
                if isinstance(evaluated_model_retrieved_context, str)
                else evaluated_model_retrieved_context.get("description")
            ),
            "evaluated_model_gold_answer": (
                evaluated_model_gold_answer
                if isinstance(evaluated_model_gold_answer, str)
                else evaluated_model_gold_answer.get("description")
            ),
            "evaluators": (
                evaluators
                if isinstance(evaluators, list)
                else evaluators.get("description")
            ),
        }

        response = requests.post(
            self.evaluate_url, headers=headers, data=json.dumps(data)
        )
        if response.status_code != 200:
            raise Exception(
                f"Failed to evaluate model input and output. Status code: {response.status_code}. Reason: {response.text}"
            )

        return response.json()
