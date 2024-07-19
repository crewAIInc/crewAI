from typing import Any, Dict, List

from pydantic import BaseModel, Field

from crewai.crews.crew_output import CrewOutput


class PipelineRunResult(BaseModel):
    final_output: CrewOutput = Field(
        description="Final output from the last crew in the run"
    )
    token_usage: Dict[str, Any] = Field(
        description="Token usage for each crew in the run"
    )
    trace: List[Any] = Field(
        description="Trace of the journey of inputs through the run"
    )
    # TODO: Should we store all outputs from crews along the way?
    crews_output: List[CrewOutput] = Field(
        description="Output from each crew in the run",
        default=[],
    )
