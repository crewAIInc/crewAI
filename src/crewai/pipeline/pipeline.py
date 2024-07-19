import asyncio
from collections import deque
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field

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
            formatted_traces: List[Trace] = []

            # Process all traces except the last one
            for trace in traces[:-1]:
                if len(trace) == 1:
                    formatted_traces.append(trace[0])
                else:
                    formatted_traces.append(trace)

            # Handle the final stage trace
            traces_to_return: List[List[Trace]] = []

            final_trace = traces[-1]
            if len(final_trace) == 1:
                formatted_traces.append(final_trace)
                traces_to_return.append(formatted_traces)
            else:
                for trace in final_trace:
                    copied_traces = formatted_traces.copy()
                    copied_traces.append(trace)
                    traces_to_return.append(copied_traces)

            return traces_to_return

        def build_pipeline_run_results(
            final_stage_outputs: List[CrewOutput],
            traces: List[List[Union[str, Dict[str, Any]]]],
            token_usage: Dict[str, Any],
        ) -> List[PipelineRunResult]:
            """
            Build PipelineRunResult objects from the final stage outputs and traces.
            """

            pipeline_run_results: List[PipelineRunResult] = []

            # Format traces
            formatted_traces = format_traces(traces)

            for output, formatted_trace in zip(final_stage_outputs, formatted_traces):
                # FORMAT TRACE

                new_pipeline_run_result = PipelineRunResult(
                    final_output=output,
                    token_usage=token_usage,
                    trace=formatted_trace,
                )

                pipeline_run_results.append(new_pipeline_run_result)

            return pipeline_run_results

        async def process_single_run(
            run_input: Dict[str, Any]
        ) -> List[PipelineRunResult]:
            stages_queue = deque(self.stages)
            usage_metrics = {}
            stage_outputs: List[CrewOutput] = []
            traces: List[List[Union[str, Dict[str, Any]]]] = [[run_input]]

            stage = None
            while stages_queue:
                stage = stages_queue.popleft()

                if isinstance(stage, Crew):
                    # Process single crew
                    output = await stage.kickoff_async(inputs=run_input)
                    # Update usage metrics and setup inputs for next stage
                    usage_metrics[stage.name] = output.token_usage
                    run_input.update(output.to_dict())
                    # Update traces for single crew stage
                    traces.append([stage.name or "No name"])
                    # Store output for final results
                    stage_outputs = [output]

                else:
                    # Process each crew in parallel
                    parallel_outputs = await asyncio.gather(
                        *[crew.kickoff_async(inputs=run_input) for crew in stage]
                    )
                    # Update usage metrics and setup inputs for next stage
                    for crew, output in zip(stage, parallel_outputs):
                        usage_metrics[crew.name] = output.token_usage
                        run_input.update(output.to_dict())
                    # Update traces for parallel stage
                    traces.append([crew.name or "No name" for crew in stage])
                    # Store output for final results
                    stage_outputs = parallel_outputs

            print("STAGE OUTPUTS: ", stage_outputs)
            print("TRACES: ", traces)
            print("TOKEN USAGE: ", usage_metrics)

            # Build final pipeline run results
            final_results = build_pipeline_run_results(
                final_stage_outputs=stage_outputs,
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
