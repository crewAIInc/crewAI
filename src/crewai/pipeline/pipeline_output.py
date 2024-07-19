from typing import List

from pydantic import BaseModel, Field

from crewai.pipeline.pipeline_run_result import PipelineRunResult


class PipelineOutput(BaseModel):
    run_results: List[PipelineRunResult] = Field(
        description="List of results for each run through the pipeline", default=[]
    )

    def add_run_result(self, result: PipelineRunResult):
        self.run_results.append(result)
