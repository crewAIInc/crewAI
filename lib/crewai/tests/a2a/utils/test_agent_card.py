"""Tests for A2A agent card utilities."""

from __future__ import annotations

from a2a.types import AgentCard, AgentSkill

from crewai import Agent
from crewai.a2a.config import A2AClientConfig, A2AServerConfig
from crewai.a2a.utils.agent_card import inject_a2a_server_methods


class TestInjectA2AServerMethods:
    """Tests for inject_a2a_server_methods function."""

    def test_agent_with_server_config_gets_to_agent_card_method(self) -> None:
        """Agent with A2AServerConfig should have to_agent_card method injected."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            a2a=A2AServerConfig(),
        )

        assert hasattr(agent, "to_agent_card")
        assert callable(agent.to_agent_card)

    def test_agent_without_server_config_no_injection(self) -> None:
        """Agent without A2AServerConfig should not get to_agent_card method."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            a2a=A2AClientConfig(endpoint="http://example.com"),
        )

        assert not hasattr(agent, "to_agent_card")

    def test_agent_without_a2a_no_injection(self) -> None:
        """Agent without any a2a config should not get to_agent_card method."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
        )

        assert not hasattr(agent, "to_agent_card")

    def test_agent_with_mixed_configs_gets_injection(self) -> None:
        """Agent with list containing A2AServerConfig should get to_agent_card."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            a2a=[
                A2AClientConfig(endpoint="http://example.com"),
                A2AServerConfig(name="My Agent"),
            ],
        )

        assert hasattr(agent, "to_agent_card")
        assert callable(agent.to_agent_card)

    def test_manual_injection_on_plain_agent(self) -> None:
        """inject_a2a_server_methods should work when called manually."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
        )
        # Manually set server config and inject
        object.__setattr__(agent, "a2a", A2AServerConfig())
        inject_a2a_server_methods(agent)

        assert hasattr(agent, "to_agent_card")
        assert callable(agent.to_agent_card)


class TestToAgentCard:
    """Tests for the injected to_agent_card method."""

    def test_returns_agent_card(self) -> None:
        """to_agent_card should return an AgentCard instance."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            a2a=A2AServerConfig(),
        )

        card = agent.to_agent_card("http://localhost:8000")

        assert isinstance(card, AgentCard)

    def test_uses_agent_role_as_name(self) -> None:
        """AgentCard name should default to agent role."""
        agent = Agent(
            role="Data Analyst",
            goal="Analyze data",
            backstory="Expert analyst",
            a2a=A2AServerConfig(),
        )

        card = agent.to_agent_card("http://localhost:8000")

        assert card.name == "Data Analyst"

    def test_uses_server_config_name(self) -> None:
        """AgentCard name should prefer A2AServerConfig.name over role."""
        agent = Agent(
            role="Data Analyst",
            goal="Analyze data",
            backstory="Expert analyst",
            a2a=A2AServerConfig(name="Custom Agent Name"),
        )

        card = agent.to_agent_card("http://localhost:8000")

        assert card.name == "Custom Agent Name"

    def test_uses_goal_as_description(self) -> None:
        """AgentCard description should include agent goal."""
        agent = Agent(
            role="Test Agent",
            goal="Accomplish important tasks",
            backstory="Has extensive experience",
            a2a=A2AServerConfig(),
        )

        card = agent.to_agent_card("http://localhost:8000")

        assert "Accomplish important tasks" in card.description

    def test_uses_server_config_description(self) -> None:
        """AgentCard description should prefer A2AServerConfig.description."""
        agent = Agent(
            role="Test Agent",
            goal="Accomplish important tasks",
            backstory="Has extensive experience",
            a2a=A2AServerConfig(description="Custom description"),
        )

        card = agent.to_agent_card("http://localhost:8000")

        assert card.description == "Custom description"

    def test_uses_provided_url(self) -> None:
        """AgentCard url should use the provided URL."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            a2a=A2AServerConfig(),
        )

        card = agent.to_agent_card("http://my-server.com:9000")

        assert card.url == "http://my-server.com:9000"

    def test_uses_server_config_url(self) -> None:
        """AgentCard url should prefer A2AServerConfig.url over provided URL."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            a2a=A2AServerConfig(url="http://configured-url.com"),
        )

        card = agent.to_agent_card("http://fallback-url.com")

        assert card.url == "http://configured-url.com/"

    def test_generates_default_skill(self) -> None:
        """AgentCard should have at least one skill based on agent role."""
        agent = Agent(
            role="Research Assistant",
            goal="Help with research",
            backstory="Skilled researcher",
            a2a=A2AServerConfig(),
        )

        card = agent.to_agent_card("http://localhost:8000")

        assert len(card.skills) >= 1
        skill = card.skills[0]
        assert skill.name == "Research Assistant"
        assert skill.description == "Help with research"

    def test_uses_server_config_skills(self) -> None:
        """AgentCard skills should prefer A2AServerConfig.skills."""
        custom_skill = AgentSkill(
            id="custom-skill",
            name="Custom Skill",
            description="A custom skill",
            tags=["custom"],
        )
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            a2a=A2AServerConfig(skills=[custom_skill]),
        )

        card = agent.to_agent_card("http://localhost:8000")

        assert len(card.skills) == 1
        assert card.skills[0].id == "custom-skill"
        assert card.skills[0].name == "Custom Skill"

    def test_includes_custom_version(self) -> None:
        """AgentCard should include version from A2AServerConfig."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            a2a=A2AServerConfig(version="2.0.0"),
        )

        card = agent.to_agent_card("http://localhost:8000")

        assert card.version == "2.0.0"

    def test_default_version(self) -> None:
        """AgentCard should have default version 1.0.0."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            a2a=A2AServerConfig(),
        )

        card = agent.to_agent_card("http://localhost:8000")

        assert card.version == "1.0.0"


class TestAgentCardJsonStructure:
    """Tests for the JSON structure of AgentCard."""

    def test_json_has_required_fields(self) -> None:
        """AgentCard JSON should contain all required A2A protocol fields."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            a2a=A2AServerConfig(),
        )

        card = agent.to_agent_card("http://localhost:8000")
        json_data = card.model_dump()

        assert "name" in json_data
        assert "description" in json_data
        assert "url" in json_data
        assert "version" in json_data
        assert "skills" in json_data
        assert "capabilities" in json_data
        assert "defaultInputModes" in json_data
        assert "defaultOutputModes" in json_data

    def test_json_skills_structure(self) -> None:
        """Each skill in JSON should have required fields."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            a2a=A2AServerConfig(),
        )

        card = agent.to_agent_card("http://localhost:8000")
        json_data = card.model_dump()

        assert len(json_data["skills"]) >= 1
        skill = json_data["skills"][0]
        assert "id" in skill
        assert "name" in skill
        assert "description" in skill
        assert "tags" in skill

    def test_json_capabilities_structure(self) -> None:
        """Capabilities in JSON should have expected fields."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            a2a=A2AServerConfig(),
        )

        card = agent.to_agent_card("http://localhost:8000")
        json_data = card.model_dump()

        capabilities = json_data["capabilities"]
        assert "streaming" in capabilities
        assert "pushNotifications" in capabilities

    def test_json_serializable(self) -> None:
        """AgentCard should be JSON serializable."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            a2a=A2AServerConfig(),
        )

        card = agent.to_agent_card("http://localhost:8000")
        json_str = card.model_dump_json()

        assert isinstance(json_str, str)
        assert "Test Agent" in json_str
        assert "http://localhost:8000" in json_str

    def test_json_excludes_none_values(self) -> None:
        """AgentCard JSON with exclude_none should omit None fields."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            a2a=A2AServerConfig(),
        )

        card = agent.to_agent_card("http://localhost:8000")
        json_data = card.model_dump(exclude_none=True)

        assert "provider" not in json_data
        assert "documentationUrl" not in json_data
        assert "iconUrl" not in json_data
