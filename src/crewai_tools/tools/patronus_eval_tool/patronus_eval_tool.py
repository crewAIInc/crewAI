import os
import json
import requests
import warnings
from typing import Any, List, Dict, Optional, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from patronus import Client


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


class FixedLocalEvaluatorToolSchema(BaseModel):
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
    evaluator: str = Field(..., description="The registered local evaluator")


class PatronusEvalTool(BaseTool):
    name: str = "Patronus Evaluation Tool"
    evaluate_url: str = "https://api.patronus.ai/v1/evaluate"
    evaluators: List[Dict[str, str]] = []
    criteria: List[Dict[str, str]] = []
    description: str = ""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        temp_evaluators, temp_criteria = self._init_run()
        self.evaluators = temp_evaluators
        self.criteria = temp_criteria
        self.description = self._generate_description()
        warnings.warn("You are allowing the agent to select the best evaluator and criteria when you use the `PatronusEvalTool`. If this is not intended then please use `PatronusPredifinedCriteriaEvalTool` instead.")

    def _init_run(self):
        content = json.loads(
            requests.get(
                "https://api.patronus.ai/v1/evaluators",
                headers={
                    "accept": "application/json",
                    "X-API-KEY": os.environ["PATRONUS_API_KEY"],
                },
            ).text
        )["evaluators"]
        ids, evaluators = set(), []
        for i in content:
            if not i["deprecated"] and i["id"] not in ids:
                evaluators.append(
                    {
                        "id": i["id"],
                        "name": i["name"],
                        "description": i["description"],
                        "aliases": i["aliases"],
                    }
                )
                ids.add(i["id"])

        content = json.loads(
            requests.get(
                "https://api.patronus.ai/v1/evaluator-criteria",
                headers={
                    "accept": "application/json",
                    "X-API-KEY": os.environ["PATRONUS_API_KEY"],
                },
            ).text
        )["evaluator_criteria"]
        criteria = []
        for i in content:
            if i["config"].get("pass_criteria", None):
                if i["config"].get("rubric", None):
                    criteria.append(
                        {
                            "evaluator": i["evaluator_family"],
                            "name": i["name"],
                            "pass_criteria": i["config"]["pass_criteria"],
                            "rubric": i["config"]["rubric"],
                        }
                    )
                else:
                    criteria.append(
                        {
                            "evaluator": i["evaluator_family"],
                            "name": i["name"],
                            "pass_criteria": i["config"]["pass_criteria"],
                        }
                    )
            elif i["description"]:
                criteria.append(
                    {
                        "evaluator": i["evaluator_family"],
                        "name": i["name"],
                        "description": i["description"],
                    }
                )

        return evaluators, criteria

    def _generate_description(self) -> str:
        criteria = "\n".join([json.dumps(i) for i in self.criteria])
        return f"""This tool calls the Patronus Evaluation API that takes the following arguments:
1. evaluated_model_input: str: The agent's task description in simple text
2. evaluated_model_output: str: The agent's output of the task
3. evaluated_model_retrieved_context: str: The agent's context
4. evaluators: This is a list of dictionaries containing one of the following evaluators and the corresponding criteria. An example input for this field: [{{"evaluator": "Judge", "criteria": "patronus:is-code"}}] 

Evaluators: 
{criteria}

You must ONLY choose the most appropriate evaluator and criteria based on the "pass_criteria" or "description" fields for your evaluation task and nothing from outside of the options present."""

    def _run(
        self,
        evaluated_model_input: Optional[str],
        evaluated_model_output: Optional[str],
        evaluated_model_retrieved_context: Optional[str],
        evaluators: List[Dict[str, str]],
    ) -> Any:

        # Assert correct format of evaluators
        evals = []
        for e in evaluators:
            evals.append(
                {
                    "evaluator": e["evaluator"].lower(),
                    "criteria": e["name"] if "name" in e else e["criteria"],
                }
            )

        data = {
            "evaluated_model_input": evaluated_model_input,
            "evaluated_model_output": evaluated_model_output,
            "evaluated_model_retrieved_context": evaluated_model_retrieved_context,
            "evaluators": evals,
        }

        headers = {
            "X-API-KEY": os.getenv("PATRONUS_API_KEY"),
            "accept": "application/json",
            "content-type": "application/json",
        }

        response = requests.post(
            self.evaluate_url, headers=headers, data=json.dumps(data)
        )
        if response.status_code != 200:
            raise Exception(
                f"Failed to evaluate model input and output. Response status code: {response.status_code}. Reason: {response.text}"
            )

        return response.json()


class PatronusLocalEvaluatorTool(BaseTool):
    name: str = "Patronus Local Evaluator Tool"
    evaluator: str = "The registered local evaluator"
    evaluated_model_gold_answer: str = "The agent's gold answer"
    description: str = (
        "This tool is used to evaluate the model input and output using custom function evaluators."
    )
    client: Any = None
    args_schema: Type[BaseModel] = FixedLocalEvaluatorToolSchema

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, evaluator: str, evaluated_model_gold_answer: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = Client()
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
            tags={},
        )
        output = f"Evaluation result: {result.pass_}, Explanation: {result.explanation}"
        return output


class PatronusPredifinedCriteriaEvalTool(BaseTool):
    """
    PatronusEvalTool is a tool to automatically evaluate and score agent interactions.

    Results are logged to the Patronus platform at app.patronus.ai
    """

    name: str = "Call Patronus API tool for evaluation of model inputs and outputs"
    description: str = (
        """This tool calls the Patronus Evaluation API that takes the following arguments:"""
    )
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
