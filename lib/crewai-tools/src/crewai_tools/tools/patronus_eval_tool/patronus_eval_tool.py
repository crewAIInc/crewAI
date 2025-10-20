import json
import os
from typing import Any
import warnings

from crewai.tools import BaseTool, EnvVar
from pydantic import Field
import requests


class PatronusEvalTool(BaseTool):
    name: str = "Patronus Evaluation Tool"
    evaluate_url: str = "https://api.patronus.ai/v1/evaluate"
    evaluators: list[dict[str, str]] = Field(default_factory=list)
    criteria: list[dict[str, str]] = Field(default_factory=list)
    description: str = ""
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="PATRONUS_API_KEY",
                description="API key for Patronus evaluation services",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        temp_evaluators, temp_criteria = self._init_run()
        self.evaluators = temp_evaluators
        self.criteria = temp_criteria
        warnings.warn(
            "You are allowing the agent to select the best evaluator and criteria when you use the `PatronusEvalTool`. If this is not intended then please use `PatronusPredefinedCriteriaEvalTool` instead.",
            stacklevel=2,
        )

    def _init_run(self):
        evaluators_set = json.loads(
            requests.get(
                "https://api.patronus.ai/v1/evaluators",
                headers={
                    "accept": "application/json",
                    "X-API-KEY": os.environ["PATRONUS_API_KEY"],
                },
                timeout=30,
            ).text
        )["evaluators"]
        ids, evaluators = set(), []
        for ev in evaluators_set:
            if not ev["deprecated"] and ev["id"] not in ids:
                evaluators.append(
                    {
                        "id": ev["id"],
                        "name": ev["name"],
                        "description": ev["description"],
                        "aliases": ev["aliases"],
                    }
                )
                ids.add(ev["id"])

        criteria_set = json.loads(
            requests.get(
                "https://api.patronus.ai/v1/evaluator-criteria",
                headers={
                    "accept": "application/json",
                    "X-API-KEY": os.environ["PATRONUS_API_KEY"],
                },
                timeout=30,
            ).text
        )["evaluator_criteria"]
        criteria = []
        for cr in criteria_set:
            if cr["config"].get("pass_criteria", None):
                if cr["config"].get("rubric", None):
                    criteria.append(
                        {
                            "evaluator": cr["evaluator_family"],
                            "name": cr["name"],
                            "pass_criteria": cr["config"]["pass_criteria"],
                            "rubric": cr["config"]["rubric"],
                        }
                    )
                else:
                    criteria.append(
                        {
                            "evaluator": cr["evaluator_family"],
                            "name": cr["name"],
                            "pass_criteria": cr["config"]["pass_criteria"],
                        }
                    )
            elif cr["description"]:
                criteria.append(
                    {
                        "evaluator": cr["evaluator_family"],
                        "name": cr["name"],
                        "description": cr["description"],
                    }
                )

        return evaluators, criteria

    def _generate_description(self) -> None:
        criteria = "\n".join([json.dumps(i) for i in self.criteria])
        self.description = f"""This tool calls the Patronus Evaluation API that takes the following arguments:
        1. evaluated_model_input: str: The agent's task description in simple text
        2. evaluated_model_output: str: The agent's output of the task
        3. evaluated_model_retrieved_context: str: The agent's context
        4. evaluators: This is a list of dictionaries containing one of the following evaluators and the corresponding criteria. An example input for this field: [{{"evaluator": "Judge", "criteria": "patronus:is-code"}}]

        Evaluators:
        {criteria}

        You must ONLY choose the most appropriate evaluator and criteria based on the "pass_criteria" or "description" fields for your evaluation task and nothing from outside of the options present."""

    def _run(
        self,
        evaluated_model_input: str | None,
        evaluated_model_output: str | None,
        evaluated_model_retrieved_context: str | None,
        evaluators: list[dict[str, str]],
    ) -> Any:
        # Assert correct format of evaluators
        evals = []
        for ev in evaluators:
            evals.append(  # noqa: PERF401
                {
                    "evaluator": ev["evaluator"].lower(),
                    "criteria": ev["name"] if "name" in ev else ev["criteria"],
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
            self.evaluate_url,
            headers=headers,
            data=json.dumps(data),
            timeout=30,
        )
        if response.status_code != 200:
            raise Exception(
                f"Failed to evaluate model input and output. Response status code: {response.status_code}. Reason: {response.text}"
            )

        return response.json()
