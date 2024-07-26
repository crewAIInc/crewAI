from __future__ import annotations

import asyncio
import copy
from typing import Any, Dict, List, Tuple, Union

from pydantic import BaseModel, Field, model_validator

from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.pipeline.pipeline_run_result import PipelineRunResult
from crewai.routers.pipeline_router import PipelineRouter

Trace = Union[Union[str, Dict[str, Any]], List[Union[str, Dict[str, Any]]]]

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


class Pipeline(BaseModel):
    stages: List[Union[Crew, "Pipeline", "PipelineRouter"]] = Field(
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

        # Process all runs in parallel
        all_run_results = await asyncio.gather(
            *(self.process_single_run(input_data) for input_data in run_inputs)
        )

        # Flatten the list of lists into a single list of results
        pipeline_results.extend(
            result for run_result in all_run_results for result in run_result
        )

        return pipeline_results

    async def process_single_run(
        self, run_input: Dict[str, Any]
    ) -> List[PipelineRunResult]:
        initial_input = copy.deepcopy(run_input)
        current_input = copy.deepcopy(run_input)
        usage_metrics = {}
        all_stage_outputs: List[List[CrewOutput]] = []
        traces: List[List[Union[str, Dict[str, Any]]]] = [[initial_input]]

        stage_index = 0
        while stage_index < len(self.stages):
            stage = self.stages[stage_index]
            stage_input = copy.deepcopy(current_input)

            if isinstance(stage, PipelineRouter):
                next_stage = stage.route(stage_input)
                traces.append([f"Routed to {next_stage.__class__.__name__}"])
                stage = next_stage

            if isinstance(stage, Crew):
                stage_outputs, stage_trace = await self._process_crew(
                    stage, stage_input
                )
            elif isinstance(stage, Pipeline):
                stage_outputs, stage_trace = await self._process_pipeline(
                    stage, stage_input
                )
            else:
                raise ValueError(f"Unsupported stage type: {type(stage)}")

            self._update_metrics_and_input(
                usage_metrics, current_input, stage, stage_outputs
            )
            traces.append(stage_trace)
            all_stage_outputs.append(stage_outputs)

            stage_index += 1

        return self._build_pipeline_run_results(
            all_stage_outputs, traces, usage_metrics
        )

    async def _process_crew(
        self, crew: Crew, current_input: Dict[str, Any]
    ) -> Tuple[List[CrewOutput], List[Union[str, Dict[str, Any]]]]:
        output = await crew.kickoff_async(inputs=current_input)
        return [output], [crew.name or str(crew.id)]

    async def _process_pipeline(
        self, pipeline: "Pipeline", current_input: Dict[str, Any]
    ) -> Tuple[List[CrewOutput], List[Union[str, Dict[str, Any]]]]:
        results = await pipeline.process_single_run(current_input)
        outputs = [result.crews_outputs[-1] for result in results]
        traces: List[Union[str, Dict[str, Any]]] = [
            f"Nested Pipeline: {pipeline.__class__.__name__}"
        ]
        return outputs, traces

    async def _process_stage(
        self, stage: Union[Crew, List[Crew]], current_input: Dict[str, Any]
    ) -> Tuple[List[CrewOutput], List[Union[str, Dict[str, Any]]]]:
        if isinstance(stage, Crew):
            return await self._process_single_crew(stage, current_input)
        else:
            return await self._process_parallel_crews(stage, current_input)

    async def _process_single_crew(
        self, crew: Crew, current_input: Dict[str, Any]
    ) -> Tuple[List[CrewOutput], List[Union[str, Dict[str, Any]]]]:
        output = await crew.kickoff_async(inputs=current_input)
        return [output], [crew.name or str(crew.id)]

    async def _process_parallel_crews(
        self, crews: List[Crew], current_input: Dict[str, Any]
    ) -> Tuple[List[CrewOutput], List[Union[str, Dict[str, Any]]]]:
        parallel_outputs = await asyncio.gather(
            *[crew.kickoff_async(inputs=current_input) for crew in crews]
        )
        return parallel_outputs, [crew.name or str(crew.id) for crew in crews]

    def _update_metrics_and_input(
        self,
        usage_metrics: Dict[str, Any],
        current_input: Dict[str, Any],
        stage: Union[Crew, "Pipeline"],
        outputs: List[CrewOutput],
    ) -> None:
        for output in outputs:
            if isinstance(stage, Crew):
                usage_metrics[stage.name or str(stage.id)] = output.token_usage
            current_input.update(output.to_dict())

    def _build_pipeline_run_results(
        self,
        all_stage_outputs: List[List[CrewOutput]],
        traces: List[List[Union[str, Dict[str, Any]]]],
        token_usage: Dict[str, Any],
    ) -> List[PipelineRunResult]:
        formatted_traces = self._format_traces(traces)
        formatted_crew_outputs = self._format_crew_outputs(all_stage_outputs)

        return [
            PipelineRunResult(
                token_usage=token_usage,
                trace=formatted_trace,
                raw=crews_outputs[-1].raw,
                pydantic=crews_outputs[-1].pydantic,
                json_dict=crews_outputs[-1].json_dict,
                crews_outputs=crews_outputs,
            )
            for crews_outputs, formatted_trace in zip(
                formatted_crew_outputs, formatted_traces
            )
        ]

    def _format_traces(
        self, traces: List[List[Union[str, Dict[str, Any]]]]
    ) -> List[List[Trace]]:
        formatted_traces: List[Trace] = []
        for trace in traces[:-1]:
            formatted_traces.append(trace[0] if len(trace) == 1 else trace)

        traces_to_return: List[List[Trace]] = []
        final_trace = traces[-1]
        if len(final_trace) == 1:
            formatted_traces.append(final_trace[0])
            traces_to_return.append(formatted_traces)
        else:
            for trace in final_trace:
                copied_traces = formatted_traces.copy()
                copied_traces.append(trace)
                traces_to_return.append(copied_traces)

        return traces_to_return

    def _format_crew_outputs(
        self, all_stage_outputs: List[List[CrewOutput]]
    ) -> List[List[CrewOutput]]:
        crew_outputs: List[CrewOutput] = [
            output
            for stage_outputs in all_stage_outputs[:-1]
            for output in stage_outputs
        ]
        return [crew_outputs + [output] for output in all_stage_outputs[-1]]

    def __rshift__(self, other: Any) -> "Pipeline":
        if isinstance(other, (Crew, Pipeline, PipelineRouter)):
            return type(self)(stages=self.stages + [other])
        else:
            raise TypeError(
                f"Unsupported operand type for >>: '{type(self).__name__}' and '{type(other).__name__}'"
            )


# TODO: CHECK IF NECESSARY
from crewai.routers.pipeline_router import PipelineRouter

Pipeline.model_rebuild()
