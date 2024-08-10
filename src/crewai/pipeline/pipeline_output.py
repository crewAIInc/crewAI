import uuid
from typing import List

from pydantic import UUID4, BaseModel, Field

from crewai.pipeline.pipeline_kickoff_result import PipelineKickoffResult


class PipelineOutput(BaseModel):
    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )
    run_results: List[PipelineKickoffResult] = Field(
        description="List of results for each run through the pipeline", default=[]
    )

    def add_run_result(self, result: PipelineKickoffResult):
        self.run_results.append(result)
