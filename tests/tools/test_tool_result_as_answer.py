from crewai import Agent, Task, Crew
from crewai.tools import tool
from unittest.mock import patch

@tool("Simple Echo")
def echo_tool():
    """Clear description for what this tool is useful for, your agent will need this information to use it."""
    return "TOOL_OUTPUT_SHOULD_NOT_BE_CHANGED"

def test_tool_result_as_answer_bypasses_formatting():
    with patch("crewai.llms.base_llm.BaseLLM.call") as mock_call:
        mock_call.return_value = "Final Answer: TOOL_OUTPUT_SHOULD_NOT_BE_CHANGED"

        agent = Agent(
            role="tester",
            goal="test result_as_answer",
            backstory="You're just here to echo things.",
            tools=[echo_tool],
            verbose=False
        )

        task = Task(
            description="Echo something",
            agent=agent,
            expected_output="TOOL_OUTPUT_SHOULD_NOT_BE_CHANGED",
            result_as_answer=True
        )

        # crew = Crew(tasks=[task])
        result = echo_tool.run()

        assert result == "TOOL_OUTPUT_SHOULD_NOT_BE_CHANGED"
