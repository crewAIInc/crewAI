from typing import Any, Optional, Type, cast, ClassVar

from crewai.tools import BaseTool
import json
import os
import requests


class PatronusEvalTool(BaseTool):
    """
    PatronusEvalTool is a tool to automatically evaluate and score agent interactions.
    
    Results are logged to the Patronus platform at app.patronus.ai
    """

    name: str = "Call Patronus API tool"
    description: str = (
        "This tool calls the Patronus Evaluation API. This function returns the response from the API."
    )
    evaluate_url: str = "https://api.patronus.ai/v1/evaluate"

    
    def _run(
        self,
        evaluated_model_input: str,
        evaluated_model_output: str,
        evaluators: list,
        tags: dict
    ) -> Any:
        
        api_key = os.getenv("PATRONUS_API_KEY")
        headers = {
            "X-API-KEY": api_key,
            "accept": "application/json",
            "content-type": "application/json"
        }
        data = {
            "evaluated_model_input": evaluated_model_input,
            "evaluated_model_output": evaluated_model_output,
            "evaluators": evaluators,
            "tags": tags
        }

        # Make the POST request
        response = requests.post(self.evaluate_url, headers=headers, data=json.dumps(data))