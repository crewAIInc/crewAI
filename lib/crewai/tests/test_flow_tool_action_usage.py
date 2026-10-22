"""Flow tool actions must get a fresh tool usage budget for every kickoff.

Tool actions build their tool instances once per flow object (in
``_flow_post_init``) and reuse them across kickoffs. Without resetting
``current_usage_count`` at the kickoff boundary, a ``max_usage_count`` budget
exhausted in an earlier kickoff makes every later kickoff of the same flow
object fail with ``ToolExecutionFailedError`` (``BaseTool.run`` answers with a
``ToolFailure(USAGE_LIMIT)`` and the tool action turns it fatal).
"""

import pytest

from crewai.flow import Flow
from crewai.flow.flow_definition import FlowDefinition
from crewai.tools import BaseTool
from crewai.tools.tool_failure import ToolExecutionFailedError


class BudgetedEchoTool(BaseTool):
    name: str = "BudgetedEchoTool"
    description: str = "Echo tool with a per-run usage budget."
    max_usage_count: int = 2

    def _run(self, text: str) -> str:
        return f"ok:{text}"


class SingleShotTool(BaseTool):
    name: str = "SingleShotTool"
    description: str = "Single-shot tool."
    max_usage_count: int = 1

    def _run(self, text: str) -> str:
        return f"ok:{text}"


def _tool_flow(ref: str, with_: dict) -> Flow:
    definition = FlowDefinition.from_declaration(
        contents={
            "schema": "crewai.flow/v1",
            "name": "ToolUsageBudgetFlow",
            "methods": {
                "run_tool": {
                    "start": True,
                    "do": {"call": "tool", "ref": ref, "with": with_},
                }
            },
        }
    )
    return Flow.from_declaration(contents=definition)


def test_tool_action_usage_budget_resets_between_kickoffs():
    flow = _tool_flow(f"{__name__}:BudgetedEchoTool", {"text": "${state.text}"})
    assert flow.kickoff(inputs={"text": "a"}) == "ok:a"
    assert flow.kickoff(inputs={"text": "b"}) == "ok:b"
    # Third kickoff on the same flow object: without the per-kickoff reset the
    # shared tool instance's current_usage_count (2) already meets the budget,
    # so BaseTool.run returns ToolFailure(USAGE_LIMIT) and the flow crashes.
    assert flow.kickoff(inputs={"text": "c"}) == "ok:c"


def test_tool_action_usage_budget_still_enforced_within_a_kickoff():
    # The budget is per kickoff, not per call: an each: loop invoking a
    # max_usage_count=1 tool twice in ONE kickoff still exhausts it mid-run.
    definition = FlowDefinition.from_declaration(
        contents={
            "schema": "crewai.flow/v1",
            "name": "EachBudgetFlow",
            "methods": {
                "loop": {
                    "start": True,
                    "do": {
                        "call": "each",
                        "in": "${state.items}",
                        "do": [
                            {
                                "name": "call_tool",
                                "action": {
                                    "call": "tool",
                                    "ref": f"{__name__}:SingleShotTool",
                                    "with": {"text": "${item}"},
                                },
                            }
                        ],
                    },
                }
            },
        }
    )
    flow = Flow.from_declaration(contents=definition)
    with pytest.raises(ToolExecutionFailedError, match="usage limit"):
        flow.kickoff(inputs={"items": ["a", "b"]})
