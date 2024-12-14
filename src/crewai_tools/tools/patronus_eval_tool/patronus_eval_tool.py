import os
import json
import requests

from typing import Any, List, Dict
from crewai.tools import BaseTool


class PatronusEvalTool(BaseTool):
    """
    PatronusEvalTool is a tool to automatically evaluate and score agent interactions.

    Results are logged to the Patronus platform at app.patronus.ai
    """

    name: str = "Call Patronus API tool for evaluation of model inputs and outputs"
    description: str = (
        """This tool calls the Patronus Evaluation API that takes the following arguments:
1. evaluated_model_input: str: The agent's task description 
2. evaluated_model_output: str: The agent's output code
3. evaluators: list[dict[str,str]]: list of dictionaries, each with a an evaluator (such as `judge`) and a criteria (like `patronus:[criteria-name-here]`)."""
    )
    evaluate_url: str = "https://api.patronus.ai/v1/evaluate"

    def _run(
        self,
        evaluated_model_input: str,
        evaluated_model_output: str,
        evaluators: List[Dict[str, str]],
        tags: dict,
    ) -> Any:

        api_key = os.getenv("PATRONUS_API_KEY")
        headers = {
            "X-API-KEY": api_key,
            "accept": "application/json",
            "content-type": "application/json",
        }
        data = {
            "evaluated_model_input": evaluated_model_input,
            "evaluated_model_output": evaluated_model_output,
            "evaluators": evaluators,
            "tags": tags,
        }

        response = requests.post(
            self.evaluate_url, headers=headers, data=json.dumps(data)
        )
        if response.status_code != 200:
            raise Exception(
                f"Failed to evaluate model input and output. Reason: {response.text}"
            )

        return response.json()
