#!/usr/bin/env python
"""Minimal Flow runner: Agent.kickoff() + response_format + tool.

  cd lib/crewai
  uv run python scripts/structured_output_with_tools_runner.py
"""

from pydantic import BaseModel, Field

from crewai import Agent
from crewai.flow import Flow, start
from crewai.tools import tool


class CalculationResult(BaseModel):
    operation: str = Field(description="Operation performed")
    result: int = Field(description="Numeric result")
    explanation: str = Field(description="Short explanation")


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class StructuredOutputFlow(Flow):
    @start()
    def run(self):
        agent = Agent(
            role="Calculator",
            goal="Use tools and return structured results",
            backstory="Compute with tools, then answer in the required schema.",
            tools=[add_numbers],
            verbose=True,
            llm='anthropic/claude-sonnet-4-6'
        )
        return agent.kickoff(
            "Use add_numbers to compute 15 + 27.",
            response_format=CalculationResult,
        )


if __name__ == "__main__":
    result = StructuredOutputFlow().kickoff()
    print(result.pydantic)
