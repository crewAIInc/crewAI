from typing import Any, Dict, List

from pydantic import BaseModel, Field

from crewai.crews.crew_output import CrewOutput


class PipelineOutput(BaseModel):
    final_outputs: List[CrewOutput] = Field(
        description="List of final outputs from the last crew in the pipeline",
        default=[],
    )
    token_usage: List[List[Dict[str, Any]]] = Field(
        description="Token usage for each crew in each stream", default=[]
    )

    def add_final_output(self, output: CrewOutput):
        self.final_outputs.append(output)

    def add_token_usage(self, usage: List[Dict[str, Any]]):
        self.token_usage.append(usage)
