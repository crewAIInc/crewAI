"""Integration test for A2A skill ID resolution (issue #3897)."""

import pytest
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from pydantic import BaseModel

from crewai.a2a.config import A2AConfig
from crewai.a2a.utils import (
    create_agent_response_model,
    extract_agent_identifiers_from_cards,
    resolve_agent_identifier,
)


def test_skill_id_resolution_integration():
    """Test the complete flow of skill ID resolution as described in issue #3897.
    
    This test replicates the exact scenario from the bug report:
    1. User creates A2A config with endpoint URL
    2. Remote agent has AgentCard with skill.id="Research"
    3. LLM returns a2a_ids=["Research"] instead of the endpoint URL
    4. System should resolve "Research" to the endpoint and proceed successfully
    """
    a2a_config = A2AConfig(
        endpoint="http://localhost:10001/.well-known/agent-card.json"
    )
    a2a_agents = [a2a_config]
    
    agent_card = AgentCard(
        name="Research Agent",
        description="An expert research agent that can conduct thorough research",
        url="http://localhost:10001",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="Research",
                name="Research",
                description="Conduct comprehensive research on any topic",
                tags=["research", "analysis", "information-gathering"],
                examples=[
                    "Research the latest developments in quantum computing",
                    "What are the current trends in renewable energy?",
                ],
            )
        ],
    )
    
    agent_cards = {
        "http://localhost:10001/.well-known/agent-card.json": agent_card
    }
    
    identifiers = extract_agent_identifiers_from_cards(a2a_agents, agent_cards)
    
    assert "http://localhost:10001/.well-known/agent-card.json" in identifiers
    assert "Research" in identifiers
    
    agent_response_model = create_agent_response_model(identifiers)
    
    agent_response_data = {
        "a2a_ids": ["Research"],  # LLM uses skill ID instead of endpoint
        "message": "Please research quantum computing developments",
        "is_a2a": True,
    }
    
    agent_response = agent_response_model.model_validate(agent_response_data)
    assert agent_response.a2a_ids == ("Research",)
    assert agent_response.message == "Please research quantum computing developments"
    assert agent_response.is_a2a is True
    
    resolved_endpoint = resolve_agent_identifier(
        "Research", a2a_agents, agent_cards
    )
    assert resolved_endpoint == "http://localhost:10001/.well-known/agent-card.json"
    
    resolved_endpoint_direct = resolve_agent_identifier(
        "http://localhost:10001/.well-known/agent-card.json",
        a2a_agents,
        agent_cards,
    )
    assert resolved_endpoint_direct == "http://localhost:10001/.well-known/agent-card.json"


def test_skill_id_validation_error_before_fix():
    """Test that demonstrates the original bug (for documentation purposes).
    
    Before the fix, creating an AgentResponse model with only endpoints
    would cause a validation error when the LLM returned a skill ID.
    """
    endpoints_only = ("http://localhost:10001/.well-known/agent-card.json",)
    agent_response_model_old = create_agent_response_model(endpoints_only)
    
    agent_response_data = {
        "a2a_ids": ["Research"],
        "message": "Please research quantum computing",
        "is_a2a": True,
    }
    
    with pytest.raises(Exception) as exc_info:
        agent_response_model_old.model_validate(agent_response_data)
    
    error_msg = str(exc_info.value)
    assert "validation error" in error_msg.lower() or "literal" in error_msg.lower()


def test_multiple_agents_with_unique_skill_ids():
    """Test that multiple agents with unique skill IDs work correctly."""
    a2a_agents = [
        A2AConfig(endpoint="http://localhost:10001/.well-known/agent-card.json"),
        A2AConfig(endpoint="http://localhost:10002/.well-known/agent-card.json"),
    ]
    
    card1 = AgentCard(
        name="Research Agent",
        description="Research agent",
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
            )
        ],
    )
    
    card2 = AgentCard(
        name="Writing Agent",
        description="Writing agent",
        url="http://localhost:10002",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="Writing",
                name="Writing",
                description="Write content",
                tags=["writing"],
            )
        ],
    )
    
    agent_cards = {
        "http://localhost:10001/.well-known/agent-card.json": card1,
        "http://localhost:10002/.well-known/agent-card.json": card2,
    }
    
    identifiers = extract_agent_identifiers_from_cards(a2a_agents, agent_cards)
    
    assert len(identifiers) == 4
    assert "http://localhost:10001/.well-known/agent-card.json" in identifiers
    assert "http://localhost:10002/.well-known/agent-card.json" in identifiers
    assert "Research" in identifiers
    assert "Writing" in identifiers
    
    agent_response_model = create_agent_response_model(identifiers)
    
    response1 = agent_response_model.model_validate({
        "a2a_ids": ["Research"],
        "message": "Do research",
        "is_a2a": True,
    })
    assert response1.a2a_ids == ("Research",)
    
    response2 = agent_response_model.model_validate({
        "a2a_ids": ["Writing"],
        "message": "Write content",
        "is_a2a": True,
    })
    assert response2.a2a_ids == ("Writing",)
    
    endpoint1 = resolve_agent_identifier("Research", a2a_agents, agent_cards)
    assert endpoint1 == "http://localhost:10001/.well-known/agent-card.json"
    
    endpoint2 = resolve_agent_identifier("Writing", a2a_agents, agent_cards)
    assert endpoint2 == "http://localhost:10002/.well-known/agent-card.json"
