"""
Tests for capability-based agent hierarchy.
"""

import pytest
from unittest.mock import Mock

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.responsibility.models import AgentCapability, CapabilityType, TaskRequirement
from crewai.responsibility.hierarchy import CapabilityHierarchy


class TestCapabilityHierarchy:
    @pytest.fixture
    def hierarchy(self):
        return CapabilityHierarchy()
        
    @pytest.fixture
    def mock_agent(self):
        agent = Mock(spec=BaseAgent)
        agent.role = "Test Agent"
        return agent
        
    @pytest.fixture
    def python_capability(self):
        return AgentCapability(
            name="Python Programming",
            capability_type=CapabilityType.TECHNICAL,
            proficiency_level=0.8,
            confidence_score=0.9,
            keywords=["python", "programming"]
        )
        
    @pytest.fixture
    def analysis_capability(self):
        return AgentCapability(
            name="Data Analysis",
            capability_type=CapabilityType.ANALYTICAL,
            proficiency_level=0.7,
            confidence_score=0.8,
            keywords=["data", "analysis", "statistics"]
        )
        
    def test_add_agent(self, hierarchy, mock_agent, python_capability):
        capabilities = [python_capability]
        hierarchy.add_agent(mock_agent, capabilities)
        
        assert len(hierarchy.agents) == 1
        assert len(hierarchy.agent_capabilities) == 1
        assert "Python Programming" in hierarchy.capability_index
        
    def test_remove_agent(self, hierarchy, mock_agent, python_capability):
        capabilities = [python_capability]
        hierarchy.add_agent(mock_agent, capabilities)
        
        assert len(hierarchy.agents) == 1
        
        hierarchy.remove_agent(mock_agent)
        
        assert len(hierarchy.agents) == 0
        assert len(hierarchy.agent_capabilities) == 0
        assert len(hierarchy.capability_index["Python Programming"]) == 0
        
    def test_supervision_relationship(self, hierarchy):
        supervisor = Mock(spec=BaseAgent)
        supervisor.role = "Supervisor"
        subordinate = Mock(spec=BaseAgent)
        subordinate.role = "Subordinate"
        
        hierarchy.add_agent(supervisor, [])
        hierarchy.add_agent(subordinate, [])
        
        hierarchy.set_supervision_relationship(supervisor, subordinate)
        
        subordinates = hierarchy.get_subordinates(supervisor)
        assert len(subordinates) == 1
        assert subordinates[0] == subordinate
        
    def test_update_agent_capability(self, hierarchy, mock_agent, python_capability):
        hierarchy.add_agent(mock_agent, [python_capability])
        
        success = hierarchy.update_agent_capability(
            mock_agent, "Python Programming", 0.9, 0.95
        )
        
        assert success is True
        
        capabilities = hierarchy.get_agent_capabilities(mock_agent)
        updated_cap = next(cap for cap in capabilities if cap.name == "Python Programming")
        assert updated_cap.proficiency_level == 0.9
        assert updated_cap.confidence_score == 0.95
        
    def test_find_capable_agents(self, hierarchy, mock_agent, python_capability):
        hierarchy.add_agent(mock_agent, [python_capability])
        
        requirements = [
            TaskRequirement(
                capability_name="Python Programming",
                capability_type=CapabilityType.TECHNICAL,
                minimum_proficiency=0.5,
                weight=1.0
            )
        ]
        
        capable_agents = hierarchy.find_capable_agents(requirements)
        
        assert len(capable_agents) == 1
        assert capable_agents[0][0] == mock_agent
        assert capable_agents[0][1] > 0.5  # Should have a good match score
        
    def test_get_best_agent_for_task(self, hierarchy, python_capability, analysis_capability):
        agent1 = Mock(spec=BaseAgent)
        agent1.role = "Python Developer"
        agent2 = Mock(spec=BaseAgent)
        agent2.role = "Data Analyst"
        
        hierarchy.add_agent(agent1, [python_capability])
        hierarchy.add_agent(agent2, [analysis_capability])
        
        requirements = [
            TaskRequirement(
                capability_name="Python Programming",
                capability_type=CapabilityType.TECHNICAL,
                minimum_proficiency=0.5,
                weight=1.0
            )
        ]
        
        result = hierarchy.get_best_agent_for_task(requirements)
        
        assert result is not None
        best_agent, score, matches = result
        assert best_agent == agent1  # Python developer should be chosen
        assert "Python Programming" in matches
        
    def test_capability_distribution(self, hierarchy, python_capability, analysis_capability):
        agent1 = Mock(spec=BaseAgent)
        agent1.role = "Developer"
        agent2 = Mock(spec=BaseAgent)
        agent2.role = "Analyst"
        
        hierarchy.add_agent(agent1, [python_capability])
        hierarchy.add_agent(agent2, [analysis_capability])
        
        distribution = hierarchy.get_capability_distribution()
        
        assert CapabilityType.TECHNICAL in distribution
        assert CapabilityType.ANALYTICAL in distribution
        assert distribution[CapabilityType.TECHNICAL]["high"] == 1  # Python capability is high proficiency
        assert distribution[CapabilityType.ANALYTICAL]["medium"] == 1  # Analysis capability is medium proficiency
        
    def test_hierarchy_path(self, hierarchy):
        manager = Mock(spec=BaseAgent)
        manager.role = "Manager"
        supervisor = Mock(spec=BaseAgent)
        supervisor.role = "Supervisor"
        worker = Mock(spec=BaseAgent)
        worker.role = "Worker"
        
        hierarchy.add_agent(manager, [])
        hierarchy.add_agent(supervisor, [])
        hierarchy.add_agent(worker, [])
        
        hierarchy.set_supervision_relationship(manager, supervisor)
        hierarchy.set_supervision_relationship(supervisor, worker)
        
        path = hierarchy.get_hierarchy_path(manager, worker)
        
        assert path is not None
        assert len(path) == 3
        assert path[0] == manager
        assert path[1] == supervisor
        assert path[2] == worker
        
    def test_capabilities_match(self, hierarchy, python_capability):
        requirement = TaskRequirement(
            capability_name="Python Programming",
            capability_type=CapabilityType.TECHNICAL,
            minimum_proficiency=0.5
        )
        
        assert hierarchy._capabilities_match(python_capability, requirement) is True
        
        requirement2 = TaskRequirement(
            capability_name="Different Name",
            capability_type=CapabilityType.TECHNICAL,
            minimum_proficiency=0.5
        )
        
        assert hierarchy._capabilities_match(python_capability, requirement2) is True
        
        requirement3 = TaskRequirement(
            capability_name="Different Name",
            capability_type=CapabilityType.ANALYTICAL,
            minimum_proficiency=0.5,
            keywords=["python"]
        )
        
        assert hierarchy._capabilities_match(python_capability, requirement3) is True
        
        requirement4 = TaskRequirement(
            capability_name="Different Name",
            capability_type=CapabilityType.ANALYTICAL,
            minimum_proficiency=0.5,
            keywords=["java"]
        )
        
        assert hierarchy._capabilities_match(python_capability, requirement4) is False
