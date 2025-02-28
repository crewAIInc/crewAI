#!/usr/bin/env python
"""
Test module for token tracking functionality in CrewAI.
This tests both direct LangChain models and LiteLLM integration.
"""

import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from crewai import Crew, Process, Task
from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.agents.langchain_agent_adapter import LangChainAgentAdapter
from crewai.utilities.token_counter_callback import (
    LangChainTokenCounter,
    LiteLLMTokenCounter,
)


def get_weather(location: str = "San Francisco"):
    """Simulates fetching current weather data for a given location."""
    # In a real implementation, you could replace this with an API call.
    return f"Current weather in {location}: Sunny, 25°C"


class TestTokenTracking:
    """Test suite for token tracking functionality."""

    @pytest.fixture
    def weather_tool(self):
        """Create a simple weather tool for testing."""
        return Tool(
            name="Weather",
            func=get_weather,
            description="Useful for fetching current weather information for a given location.",
        )

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI response with token usage information."""
        return {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }

    def test_token_process_basic(self):
        """Test basic functionality of TokenProcess class."""
        token_process = TokenProcess()

        # Test adding prompt tokens
        token_process.sum_prompt_tokens(100)
        assert token_process.prompt_tokens == 100

        # Test adding completion tokens
        token_process.sum_completion_tokens(50)
        assert token_process.completion_tokens == 50

        # Test adding successful requests
        token_process.sum_successful_requests(1)
        assert token_process.successful_requests == 1

        # Test getting summary
        summary = token_process.get_summary()
        assert summary.prompt_tokens == 100
        assert summary.completion_tokens == 50
        assert summary.total_tokens == 150
        assert summary.successful_requests == 1

    @patch("litellm.completion")
    def test_litellm_token_counter(self, mock_completion):
        """Test LiteLLMTokenCounter with a mock response."""
        # Setup
        token_process = TokenProcess()
        counter = LiteLLMTokenCounter(token_process)

        # Mock the response
        mock_completion.return_value = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            }
        }

        # Simulate a successful LLM call
        counter.log_success_event(
            kwargs={},
            response_obj=mock_completion.return_value,
            start_time=0,
            end_time=1,
        )

        # Verify token counts were updated
        assert token_process.prompt_tokens == 100
        assert token_process.completion_tokens == 50
        assert token_process.successful_requests == 1

    def test_langchain_token_counter(self):
        """Test LangChainTokenCounter with a mock response."""
        # Setup
        token_process = TokenProcess()
        counter = LangChainTokenCounter(token_process)

        # Create a mock LangChain response
        mock_response = MagicMock()
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            }
        }

        # Simulate a successful LLM call
        counter.on_llm_end(mock_response)

        # Verify token counts were updated
        assert token_process.prompt_tokens == 100
        assert token_process.completion_tokens == 50
        assert token_process.successful_requests == 1

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )
    def test_langchain_agent_adapter_token_tracking(self, weather_tool):
        """
        Integration test for token tracking with LangChainAgentAdapter.
        This test requires an OpenAI API key.
        """
        # Initialize a ChatOpenAI model
        llm = ChatOpenAI(model="gpt-3.5-turbo")

        # Create a LangChainAgentAdapter with the direct LLM
        agent = LangChainAgentAdapter(
            langchain_agent=llm,
            tools=[weather_tool],
            role="Weather Agent",
            goal="Provide current weather information for the requested location.",
            backstory="An expert weather provider that fetches current weather information using simulated data.",
            verbose=True,
        )

        # Create a weather task for the agent
        task = Task(
            description="Fetch the current weather for San Francisco.",
            expected_output="A weather report showing current conditions in San Francisco.",
            agent=agent,
        )

        # Create a crew with the single agent and task
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential,
        )

        # Execute the crew
        result = crew.kickoff()

        # Verify token usage was tracked
        assert result.token_usage is not None
        assert result.token_usage.total_tokens > 0
        assert result.token_usage.prompt_tokens > 0
        assert result.token_usage.completion_tokens > 0
        assert result.token_usage.successful_requests > 0

        # Also verify token usage directly from the agent
        usage = agent.token_process.get_summary()
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens > 0
        assert usage.successful_requests > 0


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
