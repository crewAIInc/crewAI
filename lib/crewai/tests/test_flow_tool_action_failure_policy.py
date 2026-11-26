"""Flow tool actions must honor the tool's explicit ``tool_failure_policy``.

``BaseTool.tool_failure_policy`` documents "Overrides the agent's and task's
tool_failure_policy for this tool only", and the agent execution path resolves
it (``handle_tool_failure`` / ``resolve_tool_failure_policy``): WARN records
the failure, emits ``ToolFailureDetectedEvent`` and continues; IGNORE stays
silent; only RAISE aborts. A tool action that raises on every detected
``ToolFailure`` regardless of the declared policy breaks that contract and
gives flows zero observability (no ``ToolFailureDetectedEvent`` at all).
"""

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolFailureDetectedEvent
from crewai.flow import Flow
from crewai.flow.flow_definition import FlowDefinition
from crewai.tools import BaseTool, ToolFailure
from crewai.tools.tool_failure import ToolExecutionFailedError, ToolFailurePolicy


class WarnPolicyTool(BaseTool):
    name: str = "WarnPolicyTool"
    description: str = "Reports a soft failure under an explicit WARN policy."
    tool_failure_policy: ToolFailurePolicy = ToolFailurePolicy.WARN

    def _run(self) -> ToolFailure:
        return ToolFailure(message="soft failure", code="soft")


class RaisePolicyTool(BaseTool):
    name: str = "RaisePolicyTool"
    description: str = "Reports a failure under an explicit RAISE policy."
    tool_failure_policy: ToolFailurePolicy = ToolFailurePolicy.RAISE

    def _run(self) -> ToolFailure:
        return ToolFailure(message="fatal failure", code="fatal")


def _tool_flow(ref: str) -> Flow:
    definition = FlowDefinition.from_declaration(
        contents={
            "schema": "crewai.flow/v1",
            "name": "ToolFailurePolicyFlow",
            "methods": {
                "run_tool": {
                    "start": True,
                    "do": {"call": "tool", "ref": ref, "with": {}},
                }
            },
        }
    )
    return Flow.from_declaration(contents=definition)


def test_warn_policy_tool_does_not_kill_the_flow():
    events: list[ToolFailureDetectedEvent] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(ToolFailureDetectedEvent)
        def on_failure(source, event):
            events.append(event)

        flow = _tool_flow(f"{__name__}:WarnPolicyTool")
        result = flow.kickoff()

    # Under the tool's explicit WARN policy the flow completes (the soft
    # failure is the step's output) and the failure is observable.
    assert isinstance(result, ToolFailure), "WARN policy must not abort the flow"
    assert events, "expected a ToolFailureDetectedEvent under the WARN policy"


def test_raise_policy_tool_still_aborts_the_flow():
    flow = _tool_flow(f"{__name__}:RaisePolicyTool")
    with pytest.raises(ToolExecutionFailedError, match="fatal failure"):
        flow.kickoff()
