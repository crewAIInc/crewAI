"""
Example script demonstrating how to use the LiteAgent.

This example shows how to create and use a LiteAgent for simple interactions
without the need for a full crew or task-based workflow.
"""

import asyncio
from typing import Any, Dict, cast

from pydantic import BaseModel, Field

from crewai.lite_agent import LiteAgent
from crewai.tools.base_tool import BaseTool


# Define custom tools
class WebSearchTool(BaseTool):
    """Tool for searching the web for information."""

    name: str = "search_web"
    description: str = "Search the web for information about a topic."

    def _run(self, query: str) -> str:
        """Search the web for information about a topic."""
        # This is a mock implementation
        if "tokyo" in query.lower():
            return "Tokyo's population in 2023 was approximately 14 million people in the city proper, and 37 million in the greater metropolitan area."
        elif "climate change" in query.lower() and "coral" in query.lower():
            return "Climate change severely impacts coral reefs through: 1) Ocean warming causing coral bleaching, 2) Ocean acidification reducing calcification, 3) Sea level rise affecting light availability, 4) Increased storm frequency damaging reef structures. Sources: NOAA Coral Reef Conservation Program, Global Coral Reef Alliance."
        else:
            return f"Found information about {query}: This is a simulated search result for demonstration purposes."


class CalculatorTool(BaseTool):
    """Tool for performing calculations."""

    name: str = "calculate"
    description: str = "Calculate the result of a mathematical expression."

    def _run(self, expression: str) -> str:
        """Calculate the result of a mathematical expression."""
        try:
            # CAUTION: eval can be dangerous in production code
            # This is just for demonstration purposes
            result = eval(expression, {"__builtins__": {}})
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"


# Define a custom response format using Pydantic
class ResearchResult(BaseModel):
    """Structure for research results."""

    main_findings: str = Field(description="The main findings from the research")
    key_points: list[str] = Field(description="List of key points")
    sources: list[str] = Field(description="List of sources used")


async def main():
    # Create tools
    web_search_tool = WebSearchTool()
    calculator_tool = CalculatorTool()

    # Create a LiteAgent with a specific role, goal, and backstory
    agent = LiteAgent(
        role="Research Analyst",
        goal="Provide accurate and concise information on requested topics",
        backstory="You are an expert research analyst with years of experience in gathering and synthesizing information from various sources.",
        llm="gpt-4",  # You can use any supported LLM
        tools=[web_search_tool, calculator_tool],
        verbose=True,
        response_format=ResearchResult,  # Optional: Use a structured output format
    )

    # Example 1: Simple query with raw text response
    print("\n=== Example 1: Simple Query ===")
    result = await agent.kickoff_async("What is the population of Tokyo in 2023?")
    print(f"Raw response: {result.raw}")

    # # Example 2: Query with structured output
    # print("\n=== Example 2: Structured Output ===")
    # structured_query = """
    # Research the impact of climate change on coral reefs.

    # YOU MUST format your response as a valid JSON object with the following structure:
    # {
    #   "main_findings": "A summary of the main findings",
    #   "key_points": ["Point 1", "Point 2", "Point 3"],
    #   "sources": ["Source 1", "Source 2"]
    # }

    # Include at least 3 key points and 2 sources. Wrap your JSON in ```json and ``` tags.
    # """

    # result = await agent.kickoff_async(structured_query)

    # if result.pydantic:
    #     # Cast to the specific type for better IDE support
    #     research_result = cast(ResearchResult, result.pydantic)
    #     print(f"Main findings: {research_result.main_findings}")
    #     print("\nKey points:")
    #     for i, point in enumerate(research_result.key_points, 1):
    #         print(f"{i}. {point}")
    #     print("\nSources:")
    #     for i, source in enumerate(research_result.sources, 1):
    #         print(f"{i}. {source}")
    # else:
    #     print(f"Raw response: {result.raw}")
    #     print(
    #         "\nNote: Structured output was not generated. The LLM may need more explicit instructions to format the response as JSON."
    #     )

    # # Example 3: Multi-turn conversation
    # print("\n=== Example 3: Multi-turn Conversation ===")
    # messages = [
    #     {"role": "user", "content": "I'm planning a trip to Japan."},
    #     {
    #         "role": "assistant",
    #         "content": "That sounds exciting! Japan is a beautiful country with rich culture, delicious food, and stunning landscapes. What would you like to know about Japan to help with your trip planning?",
    #     },
    #     {
    #         "role": "user",
    #         "content": "What are the best times to visit Tokyo and Kyoto?",
    #     },
    # ]

    # result = await agent.kickoff_async(messages)
    # print(f"Response: {result.raw}")

    # # Print usage metrics if available
    # if result.usage_metrics:
    #     print("\nUsage metrics:")
    #     for key, value in result.usage_metrics.items():
    #         print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
