import asyncio
from collections import deque
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field, model_validator
from pydantic_core import PydanticCustomError

from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.pipeline.pipeline_run_result import PipelineRunResult

"""
Pipeline Terminology:
Pipeline: The overall structure that defines a sequence of operations.
Stage: A distinct part of the pipeline, which can be either sequential or parallel.
Run: A specific execution of the pipeline for a given set of inputs, representing a single instance of processing through the pipeline.
Branch: Parallel executions within a stage (e.g., concurrent crew operations).
Trace: The journey of an individual input through the entire pipeline.

Example pipeline structure:
crew1 >> crew2 >> crew3

This represents a pipeline with three sequential stages:
1. crew1 is the first stage, which processes the input and passes its output to crew2.
2. crew2 is the second stage, which takes the output from crew1 as its input, processes it, and passes its output to crew3.
3. crew3 is the final stage, which takes the output from crew2 as its input and produces the final output of the pipeline.

Each input creates its own run, flowing through all stages of the pipeline.
Multiple runs can be processed concurrently, each following the defined pipeline structure.

Another example pipeline structure:
crew1 >> [crew2, crew3] >> crew4

This represents a pipeline with three stages:
1. A sequential stage (crew1)
2. A parallel stage with two branches (crew2 and crew3 executing concurrently)
3. Another sequential stage (crew4)

Each input creates its own run, flowing through all stages of the pipeline.
Multiple runs can be processed concurrently, each following the defined pipeline structure.
"""

Trace = Union[Union[str, Dict[str, Any]], List[Union[str, Dict[str, Any]]]]


class Pipeline(BaseModel):
    stages: List[Union[Crew, List[Crew]]] = Field(
        ..., description="List of crews representing stages to be executed in sequence"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_stages(cls, values):
        stages = values.get("stages", [])

        def check_nesting_and_type(item, depth=0):
            if depth > 1:
                raise ValueError("Double nesting is not allowed in pipeline stages")
            if isinstance(item, list):
                for sub_item in item:
                    check_nesting_and_type(sub_item, depth + 1)
            elif not isinstance(item, Crew):
                raise ValueError(
                    f"Expected Crew instance or list of Crews, got {type(item)}"
                )

        for stage in stages:
            check_nesting_and_type(stage)
        return values

    async def process_runs(
        self, run_inputs: List[Dict[str, Any]]
    ) -> List[PipelineRunResult]:
        """
        Process multiple runs in parallel, with each run going through all stages.
        """
        pipeline_results = []

        def format_traces(
            traces: List[List[Union[str, Dict[str, Any]]]],
        ) -> List[List[Trace]]:
            print("INCOMING TRACES: ", traces)
            formatted_traces: List[Trace] = []

            # Process all traces except the last one
            for trace in traces[:-1]:
                if len(trace) == 1:
                    formatted_traces.append(trace[0])
                else:
                    formatted_traces.append(trace)

            print("FORMATTED TRACES PRE LAST TRACE: ", formatted_traces)

            # Handle the final stage trace
            traces_to_return: List[List[Trace]] = []

            final_trace = traces[-1]
            print("FINAL TRACE: ", final_trace)
            if len(final_trace) == 1:
                formatted_traces.append(final_trace[0])
                traces_to_return.append(formatted_traces)
            else:
                for trace in final_trace:
                    copied_traces = formatted_traces.copy()
                    copied_traces.append(trace)
                    traces_to_return.append(copied_traces)

            print("TRACES TO RETURN", traces_to_return)

            return traces_to_return

        def format_crew_outputs(
            all_stage_outputs: List[List[CrewOutput]],
        ) -> List[List[CrewOutput]]:
            formatted_crew_outputs: List[List[CrewOutput]] = []

            # Handle all output stages except the final one
            crew_outputs: List[CrewOutput] = []
            for stage_outputs in all_stage_outputs[:-1]:
                for output in stage_outputs:
                    crew_outputs.append(output)

            final_stage = all_stage_outputs[-1]
            for output in final_stage:
                copied_crew_outputs = crew_outputs.copy()
                copied_crew_outputs.append(output)
                formatted_crew_outputs.append(copied_crew_outputs)

            return formatted_crew_outputs

        def build_pipeline_run_results(
            all_stage_outputs: List[List[CrewOutput]],
            traces: List[List[Union[str, Dict[str, Any]]]],
            token_usage: Dict[str, Any],
        ) -> List[PipelineRunResult]:
            """
            Build PipelineRunResult objects from the final stage outputs and traces.
            """

            pipeline_run_results: List[PipelineRunResult] = []

            formatted_traces = format_traces(traces)
            formatted_crew_outputs = format_crew_outputs(all_stage_outputs)

            for crews_outputs, formatted_trace in zip(
                formatted_crew_outputs, formatted_traces
            ):

                final_crew = crews_outputs[-1]
                new_pipeline_run_result = PipelineRunResult(
                    token_usage=token_usage,
                    trace=formatted_trace,
                    raw=final_crew.raw,
                    pydantic=final_crew.pydantic,
                    json_dict=final_crew.json_dict,
                    crews_outputs=crews_outputs,
                )

                pipeline_run_results.append(new_pipeline_run_result)

            return pipeline_run_results

        async def process_single_run(
            run_input: Dict[str, Any]
        ) -> List[PipelineRunResult]:
            initial_input = run_input.copy()  # Create a copy of the initial input
            current_input = (
                run_input.copy()
            )  # Create a working copy that will be updated
            stages_queue = deque(self.stages)
            usage_metrics = {}
            stage_outputs: List[CrewOutput] = []
            all_stage_outputs: List[List[CrewOutput]] = []
            traces: List[List[Union[str, Dict[str, Any]]]] = [
                [initial_input]
            ]  # Use the initial input here

            stage = None
            while stages_queue:
                stage = stages_queue.popleft()

                if isinstance(stage, Crew):
                    # Process single crew
                    output = await stage.kickoff_async(inputs=current_input)
                    # Update usage metrics and setup inputs for next stage
                    usage_metrics[stage.name or stage.id] = output.token_usage
                    current_input.update(output.to_dict())  # Update the working copy
                    # Update traces for single crew stage
                    traces.append([stage.name or str(stage.id)])
                    # Store output for final results
                    stage_outputs = [output]

                else:
                    # Process each crew in parallel
                    parallel_outputs = await asyncio.gather(
                        *[crew.kickoff_async(inputs=current_input) for crew in stage]
                    )
                    # Update usage metrics and setup inputs for next stage
                    for crew, output in zip(stage, parallel_outputs):
                        usage_metrics[crew.name] = output.token_usage
                        current_input.update(
                            output.to_dict()
                        )  # Update the working copy
                    # Update traces for parallel stage
                    traces.append([crew.name or str(crew.id) for crew in stage])
                    # Store output for final results
                    stage_outputs = parallel_outputs

                all_stage_outputs.append(stage_outputs)

            # print("STAGE OUTPUTS: ", stage_outputs)
            # print("TRACES: ", traces)
            # print("TOKEN USAGE: ", usage_metrics)
            # print("ALL STAGE OUTPUTS: ", all_stage_outputs)

            # Build final pipeline run results
            final_results = build_pipeline_run_results(
                all_stage_outputs=all_stage_outputs,
                traces=traces,
                token_usage=usage_metrics,
            )
            print("FINAL RESULTS: ", final_results)

            # prepare traces for final results
            return final_results

        # Process all runs in parallel
        all_run_results = await asyncio.gather(
            *(process_single_run(input_data) for input_data in run_inputs)
        )

        # Flatten the list of lists into a single list of results
        pipeline_results.extend(
            result for run_result in all_run_results for result in run_result
        )

        return pipeline_results

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
) -> List[PipelineRunResult]:
    return await pipeline.process_runs(inputs)
