#!/usr/bin/env python3
"""
Demonstration of Token-Based Tool Execution Verification

This script demonstrates how the token-based verification system
solves CrewAI Issue #3154: "Agent does not actually invoke tools,
only simulates tool usage with fabricated output"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from crewai.utilities.tool_execution_verifier import (
    ToolExecutionWrapper,
    execution_registry,
    verify_observation_token,
)


# Example tool that actually performs work
def web_search_tool(query: str) -> str:
    """Simulate a real web search tool"""
    import time
    time.sleep(0.01)  # Simulate network delay
    return f"Search results for '{query}': Found 10 relevant articles"

def file_write_tool(filename: str, content: str) -> str:
    """Simulate a file writing tool"""
    import time
    time.sleep(0.02)  # Simulate file I/O
    return f"Successfully wrote {len(content)} characters to {filename}"

def demonstrate_legitimate_execution():
    """Demonstrate legitimate tool execution with verification"""
    print("🟢 Scenario 1: Legitimate Tool Execution")
    print("-" * 40)

    # Agent requests tool execution
    token = execution_registry.request_execution(
        tool_name="WebSearchTool",
        agent_id="research_agent_1",
        task_id="task_123"
    )
    print(f"📝 Agent requested execution with token: {token.token_id[:8]}...")

    # Wrap the tool with verification
    verified_tool = ToolExecutionWrapper(web_search_tool, "WebSearchTool")

    # Execute the tool with the token
    try:
        result = verified_tool.execute_with_token(token, "AI in Healthcare")
        print(f"✅ Tool executed successfully: {result}")

        # Agent verifies the observation
        is_valid = verify_observation_token(token.token_id)
        print(f"🔍 Observation verification: {'PASSED' if is_valid else 'FAILED'}")

        return True

    except Exception as e:
        print(f"❌ Tool execution failed: {e}")
        return False

def demonstrate_fabrication_prevention():
    """Demonstrate prevention of fabricated observations"""
    print("\n🔴 Scenario 2: Attempted Fabrication")
    print("-" * 40)

    # Agent tries to use a fabricated observation
    fake_token_id = "test-token-id-for-demo"  # noqa: S105
    print(f"📝 Agent attempts to use fabricated token: {fake_token_id}")

    # Try to verify the fake token
    is_valid = verify_observation_token(fake_token_id)
    print(f"🔍 Observation verification: {'PASSED' if is_valid else 'FAILED'}")

    if not is_valid:
        print("❌ Fabrication detected - no valid execution record!")
        return True
    print("⚠️  Unexpected: Fabrication was not detected")
    return False

def demonstrate_multiple_executions():
    """Demonstrate multiple concurrent tool executions"""
    print("\n🔵 Scenario 3: Multiple Tool Executions")
    print("-" * 40)

    # Request multiple executions
    tokens = []
    tools = [
        ("WebSearchTool", web_search_tool, "AI in Healthcare"),
        ("FileWriteTool", file_write_tool, "output.txt", "Research findings"),
        ("WebSearchTool", web_search_tool, "Machine Learning Trends")
    ]

    # Request all executions
    for i, tool_info in enumerate(tools):
        tool_name = tool_info[0]
        token = execution_registry.request_execution(
            tool_name=tool_name,
            agent_id="multi_tool_agent",
            task_id=f"multi_task_{i}"
        )
        tokens.append((token, tool_info))
        print(f"📝 Requested {tool_name} with token: {token.token_id[:8]}...")

    # Execute all tools
    results = []
    for token, tool_info in tokens:
        tool_name, tool_func = tool_info[0], tool_info[1]
        tool_args = tool_info[2:]

        verified_tool = ToolExecutionWrapper(tool_func, tool_name)
        try:
            result = verified_tool.execute_with_token(token, *tool_args)
            results.append((token, result, True))
            print(f"✅ {tool_name} executed successfully")
        except Exception as e:
            results.append((token, str(e), False))
            print(f"❌ {tool_name} execution failed: {e}")

    # Verify all results
    print("\n🔍 Verifying all executions:")
    all_valid = True
    for token, _result, _success in results:
        is_valid = verify_observation_token(token.token_id)
        status = "PASSED" if is_valid else "FAILED"
        print(f"   Token {token.token_id[:8]}: {status}")
        if not is_valid:
            all_valid = False

    return all_valid

def main():
    """Main demonstration function"""
    print("🔍 Token-Based Tool Execution Verification Demo")
    print("=" * 50)
    print("Solving CrewAI Issue #3154: Agent Tool Fabrication")
    print()

    # Run all scenarios
    scenario1 = demonstrate_legitimate_execution()
    scenario2 = demonstrate_fabrication_prevention()
    scenario3 = demonstrate_multiple_executions()

    print("\n" + "=" * 50)
    print("📊 Results Summary:")
    print(f"   Legitimate Execution: {'✅ PASS' if scenario1 else '❌ FAIL'}")
    print(f"   Fabrication Prevention: {'✅ PASS' if scenario2 else '❌ FAIL'}")
    print(f"   Multiple Executions: {'✅ PASS' if scenario3 else '❌ FAIL'}")

    if all([scenario1, scenario2, scenario3]):
        print("\n🎉 ALL TESTS PASSED!")
        print("The token-based system successfully prevents tool fabrication")
        print("while allowing legitimate executions.")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("There may be issues with the implementation.")

if __name__ == "__main__":
    main()
