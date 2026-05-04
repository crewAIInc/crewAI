"""Tests for Agent.kickoff() with A2A delegation using VCR cassettes."""

from __future__ import annotations

import os

import pytest

from crewai import Agent
from crewai.a2a.config import A2AClientConfig


A2A_TEST_ENDPOINT = os.getenv(
    "A2A_TEST_ENDPOINT", "http://localhost:9999/.well-known/agent-card.json"
)


class TestAgentA2AKickoff:
    """Tests for Agent.kickoff() with A2A delegation."""

    @pytest.fixture
    def researcher_agent(self) -> Agent:
        """Create a research agent with A2A configuration."""
        return Agent(
            role="Research Analyst",
            goal="Find and analyze information about AI developments",
            backstory="Expert researcher with access to remote specialized agents",
            verbose=True,
            a2a=[
                A2AClientConfig(
                    endpoint=A2A_TEST_ENDPOINT,
                    fail_fast=False,
                    max_turns=3,  # Limit turns for testing
                )
            ],
        )

    @pytest.mark.skip(reason="VCR cassette matching issue with agent card caching")
    @pytest.mark.vcr()
    def test_agent_kickoff_delegates_to_a2a(self, researcher_agent: Agent) -> None:
        """Test that agent.kickoff() delegates to A2A server."""
        result = researcher_agent.kickoff(
            "Use the remote A2A agent to find out what the current time is in New York."
        )

        assert result is not None
        assert result.raw is not None
        assert isinstance(result.raw, str)
        assert len(result.raw) > 0

    @pytest.mark.skip(reason="VCR cassette matching issue with agent card caching")
    @pytest.mark.vcr()
    def test_agent_kickoff_with_calculator_skill(
        self, researcher_agent: Agent
    ) -> None:
        """Test that agent can delegate calculation to A2A server."""
        result = researcher_agent.kickoff(
            "Ask the remote A2A agent to calculate 25 times 17."
        )

        assert result is not None
        assert result.raw is not None
        assert "425" in result.raw or "425.0" in result.raw

    @pytest.mark.skip(reason="VCR cassette matching issue with agent card caching")
    @pytest.mark.vcr()
    def test_agent_kickoff_with_conversation_skill(
        self, researcher_agent: Agent
    ) -> None:
        """Test that agent can have a conversation with A2A server."""
        result = researcher_agent.kickoff(
            "Delegate to the remote A2A agent to explain quantum computing in simple terms."
        )

        assert result is not None
        assert result.raw is not None
        assert isinstance(result.raw, str)
        assert len(result.raw) > 50  # Should have a meaningful response

    @pytest.mark.vcr()
    def test_agent_kickoff_returns_lite_agent_output(
        self, researcher_agent: Agent
    ) -> None:
        """Test that kickoff returns LiteAgentOutput with correct structure."""
        from crewai.lite_agent_output import LiteAgentOutput

        result = researcher_agent.kickoff(
            "Use the A2A agent to tell me what time it is."
        )

        assert isinstance(result, LiteAgentOutput)
        assert result.raw is not None
        assert result.agent_role == "Research Analyst"
        assert isinstance(result.messages, list)

    @pytest.mark.skip(reason="VCR cassette matching issue with agent card caching")
    @pytest.mark.vcr()
    def test_agent_kickoff_handles_multi_turn_conversation(
        self, researcher_agent: Agent
    ) -> None:
        """Test that agent handles multi-turn A2A conversations."""
        # This should trigger multiple turns of conversation
        result = researcher_agent.kickoff(
            "Ask the remote A2A agent about recent developments in AI agent communication protocols."
        )

        assert result is not None
        assert result.raw is not None
        # The response should contain information about A2A or agent protocols
        assert isinstance(result.raw, str)

    @pytest.mark.vcr()
    def test_agent_without_a2a_works_normally(self) -> None:
        """Test that agent without A2A config works normally."""
        agent = Agent(
            role="Simple Assistant",
            goal="Help with basic tasks",
            backstory="A helpful assistant",
            verbose=False,
        )

        # This should work without A2A delegation
        result = agent.kickoff("Say hello")

        assert result is not None
        assert result.raw is not None

    @pytest.mark.vcr()
    def test_agent_kickoff_with_failed_a2a_endpoint(self) -> None:
        """Test that agent handles failed A2A connection gracefully."""
        agent = Agent(
            role="Research Analyst",
            goal="Find information",
            backstory="Expert researcher",
            verbose=False,
            a2a=[
                A2AClientConfig(
                    endpoint="http://nonexistent:9999/.well-known/agent-card.json",
                    fail_fast=False,
                )
            ],
        )

        # Should fallback to local LLM when A2A fails
        result = agent.kickoff("What is 2 + 2?")

        assert result is not None
        assert result.raw is not None

    @pytest.mark.skip(reason="VCR cassette matching issue with agent card caching")
    @pytest.mark.vcr()
    def test_agent_kickoff_with_list_messages(
        self, researcher_agent: Agent
    ) -> None:
        """Test that agent.kickoff() works with list of messages."""
        messages = [
            {
                "role": "user",
                "content": "Delegate to the A2A agent to find the current time in Tokyo.",
            },
        ]

        result = researcher_agent.kickoff(messages)

        assert result is not None
        assert result.raw is not None
        assert isinstance(result.raw, str)


class TestAgentA2AKickoffAsync:
    """Tests for async Agent.kickoff_async() with A2A delegation."""

    @pytest.fixture
    def researcher_agent(self) -> Agent:
        """Create a research agent with A2A configuration."""
        return Agent(
            role="Research Analyst",
            goal="Find and analyze information",
            backstory="Expert researcher with access to remote agents",
            verbose=True,
            a2a=[
                A2AClientConfig(
                    endpoint=A2A_TEST_ENDPOINT,
                    fail_fast=False,
                    max_turns=3,
                )
            ],
        )

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_agent_kickoff_async_delegates_to_a2a(
        self, researcher_agent: Agent
    ) -> None:
        """Test that agent.kickoff_async() delegates to A2A server."""
        result = await researcher_agent.kickoff_async(
            "Use the remote A2A agent to calculate 10 plus 15."
        )

        assert result is not None
        assert result.raw is not None
        assert isinstance(result.raw, str)

    @pytest.mark.skip(reason="Test assertion needs fixing - not capturing final answer")
    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_agent_kickoff_async_with_calculator(
        self, researcher_agent: Agent
    ) -> None:
        """Test async delegation with calculator skill."""
        result = await researcher_agent.kickoff_async(
            "Ask the A2A agent to calculate 100 divided by 4."
        )

        assert result is not None
        assert result.raw is not None
        assert "25" in result.raw or "25.0" in result.raw
