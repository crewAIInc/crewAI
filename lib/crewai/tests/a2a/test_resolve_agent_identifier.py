"""Test resolve_agent_identifier function for A2A skill ID resolution."""

import pytest
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from crewai.a2a.config import A2AConfig
from crewai.a2a.utils import resolve_agent_identifier


@pytest.fixture
def sample_agent_configs():
    """Create sample A2A agent configurations."""
    return [
        A2AConfig(endpoint="http://localhost:10001/.well-known/agent-card.json"),
        A2AConfig(endpoint="http://localhost:10002/.well-known/agent-card.json"),
    ]


@pytest.fixture
def sample_agent_cards():
    """Create sample AgentCards with skills."""
    card1 = AgentCard(
        name="Research Agent",
        description="An expert research agent",
        url="http://localhost:10001",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="Research",
                name="Research",
                description="Conduct comprehensive research",
                tags=["research", "analysis"],
                examples=["Research quantum computing"],
            )
        ],
    )

    card2 = AgentCard(
        name="Writing Agent",
        description="An expert writing agent",
        url="http://localhost:10002",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="Writing",
                name="Writing",
                description="Write high-quality content",
                tags=["writing", "content"],
                examples=["Write a blog post"],
            )
        ],
    )

    return {
        "http://localhost:10001/.well-known/agent-card.json": card1,
        "http://localhost:10002/.well-known/agent-card.json": card2,
    }


def test_resolve_endpoint_passthrough(sample_agent_configs, sample_agent_cards):
    """Test that endpoint URLs are returned as-is."""
    endpoint = "http://localhost:10001/.well-known/agent-card.json"
    result = resolve_agent_identifier(endpoint, sample_agent_configs, sample_agent_cards)
    assert result == endpoint


def test_resolve_unique_skill_id(sample_agent_configs, sample_agent_cards):
    """Test that a unique skill ID resolves to the correct endpoint."""
    result = resolve_agent_identifier("Research", sample_agent_configs, sample_agent_cards)
    assert result == "http://localhost:10001/.well-known/agent-card.json"

    result = resolve_agent_identifier("Writing", sample_agent_configs, sample_agent_cards)
    assert result == "http://localhost:10002/.well-known/agent-card.json"


def test_resolve_unknown_identifier(sample_agent_configs, sample_agent_cards):
    """Test that unknown identifiers raise a descriptive error."""
    with pytest.raises(ValueError) as exc_info:
        resolve_agent_identifier("UnknownSkill", sample_agent_configs, sample_agent_cards)

    error_msg = str(exc_info.value)
    assert "Unknown A2A agent identifier 'UnknownSkill'" in error_msg
    assert "Available endpoints:" in error_msg
    assert "Available skill IDs:" in error_msg
    assert "Research" in error_msg
    assert "Writing" in error_msg


def test_resolve_ambiguous_skill_id():
    """Test that ambiguous skill IDs raise a descriptive error."""
    configs = [
        A2AConfig(endpoint="http://localhost:10001/.well-known/agent-card.json"),
        A2AConfig(endpoint="http://localhost:10002/.well-known/agent-card.json"),
    ]

    card1 = AgentCard(
        name="Research Agent 1",
        description="First research agent",
        url="http://localhost:10001",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="Research",
                name="Research",
                description="Conduct research",
                tags=["research"],
                examples=["Research topic"],
            )
        ],
    )

    card2 = AgentCard(
        name="Research Agent 2",
        description="Second research agent",
        url="http://localhost:10002",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="Research",
                name="Research",
                description="Conduct research",
                tags=["research"],
                examples=["Research topic"],
            )
        ],
    )

    cards = {
        "http://localhost:10001/.well-known/agent-card.json": card1,
        "http://localhost:10002/.well-known/agent-card.json": card2,
    }

    with pytest.raises(ValueError) as exc_info:
        resolve_agent_identifier("Research", configs, cards)

    error_msg = str(exc_info.value)
    assert "Ambiguous skill ID 'Research'" in error_msg
    assert "found in multiple agents" in error_msg
    assert "http://localhost:10001/.well-known/agent-card.json" in error_msg
    assert "http://localhost:10002/.well-known/agent-card.json" in error_msg
    assert "Please use the specific endpoint URL to disambiguate" in error_msg


def test_resolve_with_no_skills():
    """Test resolution when agent cards have no skills."""
    configs = [
        A2AConfig(endpoint="http://localhost:10001/.well-known/agent-card.json"),
    ]

    card = AgentCard(
        name="Agent Without Skills",
        description="An agent without skills",
        url="http://localhost:10001",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[],
    )

    cards = {
        "http://localhost:10001/.well-known/agent-card.json": card,
    }

    result = resolve_agent_identifier(
        "http://localhost:10001/.well-known/agent-card.json", configs, cards
    )
    assert result == "http://localhost:10001/.well-known/agent-card.json"

    with pytest.raises(ValueError) as exc_info:
        resolve_agent_identifier("SomeSkill", configs, cards)

    error_msg = str(exc_info.value)
    assert "Unknown A2A agent identifier 'SomeSkill'" in error_msg
    assert "Available skill IDs: none" in error_msg


def test_resolve_with_multiple_skills_same_card(sample_agent_configs):
    """Test resolution when a card has multiple skills."""
    card = AgentCard(
        name="Multi-Skill Agent",
        description="An agent with multiple skills",
        url="http://localhost:10001",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="Research",
                name="Research",
                description="Conduct research",
                tags=["research"],
                examples=["Research topic"],
            ),
            AgentSkill(
                id="Analysis",
                name="Analysis",
                description="Analyze data",
                tags=["analysis"],
                examples=["Analyze data"],
            ),
        ],
    )

    cards = {
        "http://localhost:10001/.well-known/agent-card.json": card,
    }

    result1 = resolve_agent_identifier("Research", sample_agent_configs[:1], cards)
    assert result1 == "http://localhost:10001/.well-known/agent-card.json"

    result2 = resolve_agent_identifier("Analysis", sample_agent_configs[:1], cards)
    assert result2 == "http://localhost:10001/.well-known/agent-card.json"


def test_resolve_empty_agent_cards():
    """Test resolution with empty agent cards dictionary."""
    configs = [
        A2AConfig(endpoint="http://localhost:10001/.well-known/agent-card.json"),
    ]
    cards = {}

    result = resolve_agent_identifier(
        "http://localhost:10001/.well-known/agent-card.json", configs, cards
    )
    assert result == "http://localhost:10001/.well-known/agent-card.json"

    with pytest.raises(ValueError) as exc_info:
        resolve_agent_identifier("SomeSkill", configs, cards)

    error_msg = str(exc_info.value)
    assert "Unknown A2A agent identifier 'SomeSkill'" in error_msg
