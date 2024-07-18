import asyncio
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field

from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput

"""
Pipeline Terminology:
Pipeline: The overall structure that defines a sequence of operations.
Stage: A distinct part of the pipeline, which can be either sequential or parallel.
Branch: Parallel executions within a stage (e.g., concurrent crew operations).
Stream: The journey of an individual input through the entire pipeline.

Example pipeline structure:
crew1 >> [crew2, crew3] >> crew4

This represents a pipeline with three stages:
1. A sequential stage (crew1)
2. A parallel stage with two branches (crew2 and crew3 executing concurrently)
3. Another sequential stage (crew4)

Each input creates its own stream, flowing through all stages of the pipeline.
Multiple streams can be processed concurrently, each following the defined pipeline structure.
"""


class Pipeline(BaseModel):
    stages: List[Union[Crew, List[Crew]]] = Field(
        ..., description="List of crews representing stages to be executed in sequence"
    )

    async def process_streams(
        self, stream_inputs: List[Dict[str, Any]]
    ) -> List[List[CrewOutput]]:
        """
        Process multiple streams in parallel, with each stream going through all stages.
        """

        async def process_single_stream(
            stream_input: Dict[str, Any]
        ) -> List[CrewOutput]:
            print("current_input in stream", stream_input)
            stage_outputs = []

            for stage in self.stages:
                if isinstance(stage, Crew):
                    # Process single crew
                    stage_output = await stage.kickoff_async(inputs=stream_input)
                    stage_outputs = [stage_output]
                else:
                    # Process each crew in parallel
                    parallel_outputs = await asyncio.gather(
                        *[crew.kickoff_async(inputs=stream_input) for crew in stage]
                    )
                    stage_outputs = parallel_outputs

                # Convert all CrewOutputs from stage into a dictionary for next stage
                # and update original stream_input dictionary with new values
                stage_output_dicts = [output.to_dict() for output in stage_outputs]
                for stage_dict in stage_output_dicts:
                    stream_input.update(stage_dict)
                    print("UPDATING stream_input - new values:", stream_input)

            # Return all CrewOutputs from this stream
            return stage_outputs

        # Process all streams in parallel
        return await asyncio.gather(
            *(process_single_stream(input_data) for input_data in stream_inputs)
        )

    def __rshift__(self, other: Any) -> "Pipeline":
        """
        Implements the >> operator to add another Stage (Crew or List[Crew]) to an existing Pipeline.
        """
        if isinstance(other, Crew):
            return type(self)(stages=self.stages + [other])
        elif isinstance(other, list) and all(isinstance(crew, Crew) for crew in other):
            return type(self)(stages=self.stages + [other])
        else:
            raise TypeError(
                f"Unsupported operand type for >>: '{type(self).__name__}' and '{type(other).__name__}'"
            )


# Helper function to run the pipeline
async def run_pipeline(
    pipeline: Pipeline, inputs: List[Dict[str, Any]]
) -> List[List[CrewOutput]]:
    return await pipeline.process_streams(inputs)
