from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union

from pydantic import BaseModel

from crewai.crew import Crew


class PipelineRouter(BaseModel):
    conditions: List[
        Tuple[Callable[[Dict[str, Any]], bool], Union[Crew, "Pipeline"]]
    ] = []
    default: Union[Crew, "Pipeline", None] = None

    def add_condition(
        self,
        condition: Callable[[Dict[str, Any]], bool],
        next_stage: Union[Crew, "Pipeline"],
    ):
        """
        Add a condition and its corresponding next stage to the router.

        Args:
            condition: A function that takes the input dictionary and returns a boolean.
            next_stage: The Crew or Pipeline to execute if the condition is met.
        """
        self.conditions.append((condition, next_stage))

    def set_default(self, default_stage: Union[Crew, "Pipeline"]):
        """Set the default stage to be executed if no conditions are met."""
        self.default = default_stage

    def route(self, input_dict: Dict[str, Any]) -> Union[Crew, "Pipeline"]:
        """
        Evaluate the input against the conditions and return the appropriate next stage.

        Args:
            input_dict: The input dictionary to be evaluated.

        Returns:
            The next Crew or Pipeline to be executed.

        Raises:
            ValueError: If no conditions are met and no default stage was set.
        """
        for condition, next_stage in self.conditions:
            if condition(input_dict):
                self._update_trace(input_dict, next_stage)
                return next_stage

        if self.default is not None:
            self._update_trace(input_dict, self.default)
            return self.default

        raise ValueError("No conditions were met and no default stage was set.")

    def _update_trace(
        self, input_dict: Dict[str, Any], next_stage: Union[Crew, "Pipeline"]
    ):
        """Update the trace to show that the input went through the router."""
        if "trace" not in input_dict:
            input_dict["trace"] = []
        input_dict["trace"].append(
            {
                "router": self.__class__.__name__,
                "next_stage": next_stage.__class__.__name__,
            }
        )


# TODO: See if this is necessary
from crewai.pipeline.pipeline import Pipeline

# This line should be at the end of the file
PipelineRouter.model_rebuild()
