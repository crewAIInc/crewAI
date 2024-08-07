import asyncio
import copy
from typing import Any, Dict, List, Tuple, Union

from pydantic import BaseModel, Field, model_validator

from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.pipeline.pipeline_kickoff_result import PipelineKickoffResult
from crewai.routers.router import Router
from crewai.types.usage_metrics import UsageMetrics

Trace = Union[Union[str, Dict[str, Any]], List[Union[str, Dict[str, Any]]]]
PipelineStage = Union[Crew, List[Crew], Router]

"""
Developer Notes:

This module defines a Pipeline class that represents a sequence of operations (stages)
to process inputs. Each stage can be either sequential or parallel, and the pipeline
can process multiple kickoffs concurrently.

Core Loop Explanation:
1. The `process_kickoffs` method processes multiple kickoffs in parallel, each going through
   all pipeline stages.
2. The `process_single_kickoff` method handles the processing of a single kickouff through
   all stages, updating metrics and input data along the way.
3. The `_process_stage` method determines whether a stage is sequential or parallel
   and processes it accordingly.
4. The `_process_single_crew` and `_process_parallel_crews` methods handle the
   execution of single and parallel crew stages.
5. The `_update_metrics_and_input` method updates usage metrics and the current input
   with the outputs from a stage.
6. The `_build_pipeline_kickoff_results` method constructs the final results of the
   pipeline kickoff, including traces and outputs.

Handling Traces and Crew Outputs:
- During the processing of stages, we handle the results (traces and crew outputs)
  for all stages except the last one differently from the final stage.
- For intermediate stages, the primary focus is on passing the input data between stages.
  This involves merging the output dictionaries from all crews in a stage into a single
  dictionary and passing it to the next stage. This merged dictionary allows for smooth
  data flow between stages.
- For the final stage, in addition to passing the input data, we also need to prepare
  the final outputs and traces to be returned as the overall result of the pipeline kickoff.
  In this case, we do not merge the results, as each result needs to be included
  separately in its own pipeline kickoff result.

Pipeline Terminology:
- Pipeline: The overall structure that defines a sequence of operations.
- Stage: A distinct part of the pipeline, which can be either sequential or parallel.
- Kickoff: A specific execution of the pipeline for a given set of inputs, representing a single instance of processing through the pipeline.
- Branch: Parallel executions within a stage (e.g., concurrent crew operations).
- Trace: The journey of an individual input through the entire pipeline.

Example pipeline structure:
crew1 >> crew2 >> crew3 

This represents a pipeline with three sequential stages:
1. crew1 is the first stage, which processes the input and passes its output to crew2.
2. crew2 is the second stage, which takes the output from crew1 as its input, processes it, and passes its output to crew3.
3. crew3 is the final stage, which takes the output from crew2 as its input and produces the final output of the pipeline.

Each input creates its own kickoff, flowing through all stages of the pipeline.
Multiple kickoffss can be processed concurrently, each following the defined pipeline structure.

Another example pipeline structure:
crew1 >> [crew2, crew3] >> crew4

This represents a pipeline with three stages:
1. A sequential stage (crew1)
2. A parallel stage with two branches (crew2 and crew3 executing concurrently)
3. Another sequential stage (crew4)

Each input creates its own kickoff, flowing through all stages of the pipeline.
Multiple kickoffs can be processed concurrently, each following the defined pipeline structure.
"""


class Pipeline(BaseModel):
    stages: List[PipelineStage] = Field(
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
            elif not isinstance(item, (Crew, Router)):
                raise ValueError(
                    f"Expected Crew instance, Router instance, or list of Crews, got {type(item)}"
                )

        for stage in stages:
            check_nesting_and_type(stage)
        return values

    async def kickoff(
        self, inputs: List[Dict[str, Any]]
    ) -> List[PipelineKickoffResult]:
        """
        Processes multiple runs in parallel, each going through all pipeline stages.

        Args:
            inputs (List[Dict[str, Any]]): List of inputs for each run.

        Returns:
            List[PipelineKickoffResult]: List of results from each run.
        """
        pipeline_results: List[PipelineKickoffResult] = []

        # Process all runs in parallel
        all_run_results = await asyncio.gather(
            *(self.process_single_kickoff(input_data) for input_data in inputs)
        )

        # Flatten the list of lists into a single list of results
        pipeline_results.extend(
            result for run_result in all_run_results for result in run_result
        )

        return pipeline_results

    async def process_single_kickoff(
        self, kickoff_input: Dict[str, Any]
    ) -> List[PipelineKickoffResult]:
        """
        Processes a single run through all pipeline stages.

        Args:
            input (Dict[str, Any]): The input for the run.

        Returns:
            List[PipelineKickoffResult]: The results of processing the run.
        """
        initial_input = copy.deepcopy(kickoff_input)
        current_input = copy.deepcopy(kickoff_input)
        stages = self._copy_stages()
        pipeline_usage_metrics: Dict[str, UsageMetrics] = {}
        all_stage_outputs: List[List[CrewOutput]] = []
        traces: List[List[Union[str, Dict[str, Any]]]] = [[initial_input]]

        stage_index = 0
        while stage_index < len(stages):
            stage = stages[stage_index]
            stage_input = copy.deepcopy(current_input)

            if isinstance(stage, Router):
                next_pipeline, route_taken = stage.route(stage_input)
                stages = (
                    stages[: stage_index + 1]
                    + list(next_pipeline.stages)
                    + stages[stage_index + 1 :]
                )
                traces.append([{"route_taken": route_taken}])
                stage_index += 1
                continue

            stage_outputs, stage_trace = await self._process_stage(stage, stage_input)

            self._update_metrics_and_input(
                pipeline_usage_metrics, current_input, stage, stage_outputs
            )
            traces.append(stage_trace)
            all_stage_outputs.append(stage_outputs)
            stage_index += 1

        return self._build_pipeline_kickoff_results(
            all_stage_outputs, traces, pipeline_usage_metrics
        )

    async def _process_stage(
        self, stage: PipelineStage, current_input: Dict[str, Any]
    ) -> Tuple[List[CrewOutput], List[Union[str, Dict[str, Any]]]]:
        """
        Processes a single stage of the pipeline, which can be either sequential or parallel.

        Args:
            stage (Union[Crew, List[Crew]]): The stage to process.
            current_input (Dict[str, Any]): The input for the stage.

        Returns:
            Tuple[List[CrewOutput], List[Union[str, Dict[str, Any]]]]: The outputs and trace of the stage.
        """
        if isinstance(stage, Crew):
            return await self._process_single_crew(stage, current_input)
        elif isinstance(stage, list) and all(isinstance(crew, Crew) for crew in stage):
            return await self._process_parallel_crews(stage, current_input)
        else:
            raise ValueError(f"Unsupported stage type: {type(stage)}")

    async def _process_single_crew(
        self, crew: Crew, current_input: Dict[str, Any]
    ) -> Tuple[List[CrewOutput], List[Union[str, Dict[str, Any]]]]:
        """
        Processes a single crew.

        Args:
            crew (Crew): The crew to process.
            current_input (Dict[str, Any]): The input for the crew.

        Returns:
            Tuple[List[CrewOutput], List[Union[str, Dict[str, Any]]]]: The output and trace of the crew.
        """
        output = await crew.kickoff_async(inputs=current_input)
        return [output], [crew.name or str(crew.id)]

    async def _process_parallel_crews(
        self, crews: List[Crew], current_input: Dict[str, Any]
    ) -> Tuple[List[CrewOutput], List[Union[str, Dict[str, Any]]]]:
        """
        Processes multiple crews in parallel.

        Args:
            crews (List[Crew]): The list of crews to process in parallel.
            current_input (Dict[str, Any]): The input for the crews.

        Returns:
            Tuple[List[CrewOutput], List[Union[str, Dict[str, Any]]]]: The outputs and traces of the crews.
        """
        parallel_outputs = await asyncio.gather(
            *[crew.kickoff_async(inputs=current_input) for crew in crews]
        )
        return parallel_outputs, [crew.name or str(crew.id) for crew in crews]

    def _update_metrics_and_input(
        self,
        usage_metrics: Dict[str, UsageMetrics],
        current_input: Dict[str, Any],
        stage: PipelineStage,
        outputs: List[CrewOutput],
    ) -> None:
        """
        Updates metrics and current input with the outputs of a stage.

        Args:
            usage_metrics (Dict[str, Any]): The usage metrics to update.
            current_input (Dict[str, Any]): The current input to update.
            stage (Union[Crew, List[Crew]]): The stage that was processed.
            outputs (List[CrewOutput]): The outputs of the stage.
        """
        if isinstance(stage, Crew):
            usage_metrics[stage.name or str(stage.id)] = outputs[0].token_usage
            current_input.update(outputs[0].to_dict())
        elif isinstance(stage, list) and all(isinstance(crew, Crew) for crew in stage):
            for crew, output in zip(stage, outputs):
                usage_metrics[crew.name or str(crew.id)] = output.token_usage
                current_input.update(output.to_dict())
        else:
            raise ValueError(f"Unsupported stage type: {type(stage)}")

    def _build_pipeline_kickoff_results(
        self,
        all_stage_outputs: List[List[CrewOutput]],
        traces: List[List[Union[str, Dict[str, Any]]]],
        token_usage: Dict[str, UsageMetrics],
    ) -> List[PipelineKickoffResult]:
        """
        Builds the results of a pipeline run.

        Args:
            all_stage_outputs (List[List[CrewOutput]]): All stage outputs.
            traces (List[List[Union[str, Dict[str, Any]]]]): All traces.
            token_usage (Dict[str, Any]): Token usage metrics.

        Returns:
            List[PipelineKickoffResult]: The results of the pipeline run.
        """
        formatted_traces = self._format_traces(traces)
        formatted_crew_outputs = self._format_crew_outputs(all_stage_outputs)

        return [
            PipelineKickoffResult(
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
        """
        Formats the traces of a pipeline run.

        Args:
            traces (List[List[Union[str, Dict[str, Any]]]]): The traces to format.

        Returns:
            List[List[Trace]]: The formatted traces.
        """
        formatted_traces: List[Trace] = self._format_single_trace(traces[:-1])
        return self._format_multiple_traces(formatted_traces, traces[-1])

    def _format_single_trace(
        self, traces: List[List[Union[str, Dict[str, Any]]]]
    ) -> List[Trace]:
        """
        Formats single traces.

        Args:
            traces (List[List[Union[str, Dict[str, Any]]]]): The traces to format.

        Returns:
            List[Trace]: The formatted single traces.
        """
        formatted_traces: List[Trace] = []
        for trace in traces:
            formatted_traces.append(trace[0] if len(trace) == 1 else trace)
        return formatted_traces

    def _format_multiple_traces(
        self,
        formatted_traces: List[Trace],
        final_trace: List[Union[str, Dict[str, Any]]],
    ) -> List[List[Trace]]:
        """
        Formats multiple traces.

        Args:
            formatted_traces (List[Trace]): The formatted single traces.
            final_trace (List[Union[str, Dict[str, Any]]]): The final trace to format.

        Returns:
            List[List[Trace]]: The formatted multiple traces.
        """
        traces_to_return: List[List[Trace]] = []
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
        """
        Formats the outputs of all stages into a list of crew outputs.

        Args:
            all_stage_outputs (List[List[CrewOutput]]): All stage outputs.

        Returns:
            List[List[CrewOutput]]: Formatted crew outputs.
        """
        crew_outputs: List[CrewOutput] = [
            output
            for stage_outputs in all_stage_outputs[:-1]
            for output in stage_outputs
        ]
        return [crew_outputs + [output] for output in all_stage_outputs[-1]]

    def _copy_stages(self):
        """Create a deep copy of the Pipeline's stages."""
        new_stages = []
        for stage in self.stages:
            if isinstance(stage, list):
                new_stages.append(
                    [
                        crew.copy() if hasattr(crew, "copy") else copy.deepcopy(crew)
                        for crew in stage
                    ]
                )
            elif hasattr(stage, "copy"):
                new_stages.append(stage.copy())
            else:
                new_stages.append(copy.deepcopy(stage))

        return new_stages

    def __rshift__(self, other: PipelineStage) -> "Pipeline":
        """
        Implements the >> operator to add another Stage (Crew or List[Crew]) to an existing Pipeline.

        Args:
            other (Any): The stage to add.

        Returns:
            Pipeline: A new pipeline with the added stage.
        """
        if isinstance(other, (Crew, Router)) or (
            isinstance(other, list) and all(isinstance(item, Crew) for item in other)
        ):
            return type(self)(stages=self.stages + [other])
        else:
            raise TypeError(
                f"Unsupported operand type for >>: '{type(self).__name__}' and '{type(other).__name__}'"
            )
