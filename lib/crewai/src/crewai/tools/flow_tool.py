"""Wrap Flow classes as callable tools so agents can invoke them."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool
from crewai.utilities.string_utils import sanitize_tool_name


class FlowToolInputSchema(BaseModel):
    """Default input schema for a FlowTool."""

    inputs: str = Field(
        default="{}",
        description=(
            "JSON string of key-value pairs to pass as inputs to the flow. "
            "Use '{}' if the flow requires no inputs."
        ),
    )


class FlowTool(BaseTool):
    """Wraps a Flow class as a BaseTool so an agent can invoke it.

    The tool instantiates the Flow, calls ``kickoff(inputs=...)`` and returns
    the result as a string.
    """

    name: str = ""
    description: str = ""
    flow_class: Any = Field(
        default=None,
        description="The Flow class (not instance) to wrap.",
        exclude=True,
    )
    args_schema: Any = FlowToolInputSchema

    def _run(self, inputs: str = "{}") -> str:
        """Instantiate the Flow, run kickoff, and return the result."""
        try:
            parsed_inputs = json.loads(inputs) if isinstance(inputs, str) else inputs
        except (json.JSONDecodeError, TypeError):
            parsed_inputs = {}

        if not isinstance(parsed_inputs, dict):
            parsed_inputs = {}

        flow_instance = self.flow_class()
        result = flow_instance.kickoff(inputs=parsed_inputs if parsed_inputs else None)
        return str(result)


def create_flow_tools(flows: list[type] | None) -> list[BaseTool]:
    """Convert a list of Flow classes into BaseTool wrappers.

    Args:
        flows: Flow classes (not instances) to wrap as tools.

    Returns:
        A list of FlowTool instances ready for agent use.
    """
    if not flows:
        return []

    tools: list[BaseTool] = []
    for flow_cls in flows:
        name = sanitize_tool_name(flow_cls.__name__)
        docstring = (flow_cls.__doc__ or "").strip()
        description = docstring if docstring else f"Run the {flow_cls.__name__} flow."

        tools.append(
            FlowTool(
                name=name,
                description=description,
                flow_class=flow_cls,
            )
        )
    return tools
