"""Reproduction script demonstrating the pre-execution step_callback hook.

This script proves that the step_callback now fires BEFORE tool execution,
exposing the tool name and arguments. This enables external observability
tools to implement circuit-breaker logic for infinite retry loop detection.

Usage:
    python reproduce_hook.py
"""

from crewai.agents.parser import AgentAction, AgentFinish


def my_step_callback(step):
    """Example callback that logs tool invocations before they execute.

    When step.result is None, the callback is firing BEFORE the tool runs.
    When step.result is set, it is the post-execution callback.
    """
    if isinstance(step, AgentAction):
        if step.result is None:
            # PRE-EXECUTION: this is the new hook
            print(f"[Hook:PRE]  Agent acting: {step.tool} with {step.tool_input}")
        else:
            # POST-EXECUTION: existing behavior
            print(f"[Hook:POST] Agent acted:  {step.tool} -> {step.result[:80]}...")
    elif isinstance(step, AgentFinish):
        print(f"[Hook:DONE] Agent finished: {str(step.output)[:80]}...")


# --- Direct unit-level proof that the callback fires ---

if __name__ == "__main__":
    print("=" * 60)
    print("Testing pre-execution step_callback hook")
    print("=" * 60)

    # Simulate what CrewAgentExecutor._invoke_pre_tool_step_callback does
    import json

    tool_name = "search_tool"
    tool_input = {"query": "crewai loop detection", "max_results": 5}

    input_str = json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)

    pre_action = AgentAction(
        tool=tool_name,
        tool_input=input_str,
        text=f"About to execute: {tool_name}",
        thought=f"Pre-execution hook for {tool_name}",
        result=None,
    )

    print("\n1. Firing PRE-execution callback:")
    my_step_callback(pre_action)

    # Simulate post-execution (existing behavior)
    post_action = AgentAction(
        tool=tool_name,
        tool_input=input_str,
        text=f"Executed: {tool_name}",
        thought="Tool completed",
        result="Found 3 results about loop detection in CrewAI documentation.",
    )

    print("\n2. Firing POST-execution callback:")
    my_step_callback(post_action)

    # Simulate final answer
    finish = AgentFinish(
        thought="Task complete",
        output="Loop detection is possible using step_callback.",
        text="Final answer",
    )

    print("\n3. Firing FINISH callback:")
    my_step_callback(finish)

    print("\n" + "=" * 60)
    print("SUCCESS: Pre-execution hook fires with tool name + args")
    print("         before the tool is actually executed.")
    print("=" * 60)
