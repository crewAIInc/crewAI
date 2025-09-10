"""
Tests for mathematical responsibility assignment.
"""

import pytest
from unittest.mock import Mock

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.task import Task
from crewai.responsibility.models import AgentCapability, CapabilityType, TaskRequirement
from crewai.responsibility.hierarchy import CapabilityHierarchy
from crewai.responsibility.assignment import ResponsibilityCalculator, AssignmentStrategy


class TestResponsibilityCalculator:
    @pytest.fixture
    def hierarchy(self):
        return CapabilityHierarchy()
        
    @pytest.fixture
    def calculator(self, hierarchy):
        return ResponsibilityCalculator(hierarchy)
        
    @pytest.fixture
    def mock_task(self):
        task = Mock(spec=Task)
        task.id = "test_task_1"
        task.description = "Test task description"
        return task
        
    @pytest.fixture
    def python_agent(self, hierarchy):
        agent = Mock(spec=BaseAgent)
        agent.role = "Python Developer"
        
        capability = AgentCapability(
            name="Python Programming",
            capability_type=CapabilityType.TECHNICAL,
            proficiency_level=0.9,
            confidence_score=0.8,
            keywords=["python", "programming"]
        )
        
        hierarchy.add_agent(agent, [capability])
        return agent
        
    @pytest.fixture
    def analysis_agent(self, hierarchy):
        agent = Mock(spec=BaseAgent)
        agent.role = "Data Analyst"
        
        capability = AgentCapability(
            name="Data Analysis",
            capability_type=CapabilityType.ANALYTICAL,
            proficiency_level=0.8,
            confidence_score=0.9,
            keywords=["data", "analysis"]
        )
        
        hierarchy.add_agent(agent, [capability])
        return agent
        
    def test_greedy_assignment(self, calculator, mock_task, python_agent):
        requirements = [
            TaskRequirement(
                capability_name="Python Programming",
                capability_type=CapabilityType.TECHNICAL,
                minimum_proficiency=0.5,
                weight=1.0
            )
        ]
        
        assignment = calculator.calculate_responsibility_assignment(
            mock_task, requirements, AssignmentStrategy.GREEDY
        )
        
        assert assignment is not None
        assert assignment.task_id == "test_task_1"
        assert assignment.responsibility_score > 0.5
        assert "Python Programming" in assignment.capability_matches
        assert "Greedy assignment" in assignment.reasoning
        
    def test_balanced_assignment(self, calculator, mock_task, python_agent, analysis_agent):
        calculator.update_workload(python_agent, 5)  # High workload
        calculator.update_workload(analysis_agent, 1)  # Low workload
        
        requirements = [
            TaskRequirement(
                capability_name="General Programming",
                capability_type=CapabilityType.TECHNICAL,
                minimum_proficiency=0.3,
                weight=1.0
            )
        ]
        
        assignment = calculator.calculate_responsibility_assignment(
            mock_task, requirements, AssignmentStrategy.BALANCED
        )
        
        assert assignment is not None
        assert "Balanced assignment" in assignment.reasoning
        
    def test_optimal_assignment(self, calculator, mock_task, python_agent):
        requirements = [
            TaskRequirement(
                capability_name="Python Programming",
                capability_type=CapabilityType.TECHNICAL,
                minimum_proficiency=0.5,
                weight=1.0
            )
        ]
        
        assignment = calculator.calculate_responsibility_assignment(
            mock_task, requirements, AssignmentStrategy.OPTIMAL
        )
        
        assert assignment is not None
        assert "Optimal assignment" in assignment.reasoning
        
    def test_multi_agent_assignment(self, calculator, mock_task, python_agent, analysis_agent):
        requirements = [
            TaskRequirement(
                capability_name="Python Programming",
                capability_type=CapabilityType.TECHNICAL,
                minimum_proficiency=0.5,
                weight=1.0
            ),
            TaskRequirement(
                capability_name="Data Analysis",
                capability_type=CapabilityType.ANALYTICAL,
                minimum_proficiency=0.5,
                weight=0.8
            )
        ]
        
        assignments = calculator.calculate_multi_agent_assignment(
            mock_task, requirements, max_agents=2
        )
        
        assert len(assignments) <= 2
        assert len(assignments) > 0
        
        agent_ids = [assignment.agent_id for assignment in assignments]
        assert len(agent_ids) == len(set(agent_ids))
        
    def test_workload_update(self, calculator, python_agent):
        initial_workload = calculator.current_workloads.get(
            calculator.hierarchy._get_agent_id(python_agent), 0
        )
        
        calculator.update_workload(python_agent, 3)
        
        new_workload = calculator.current_workloads.get(
            calculator.hierarchy._get_agent_id(python_agent), 0
        )
        
        assert new_workload == initial_workload + 3
        
        calculator.update_workload(python_agent, -2)
        
        final_workload = calculator.current_workloads.get(
            calculator.hierarchy._get_agent_id(python_agent), 0
        )
        
        assert final_workload == new_workload - 2
        
    def test_workload_distribution(self, calculator, python_agent, analysis_agent):
        calculator.update_workload(python_agent, 3)
        calculator.update_workload(analysis_agent, 1)
        
        distribution = calculator.get_workload_distribution()
        
        python_id = calculator.hierarchy._get_agent_id(python_agent)
        analysis_id = calculator.hierarchy._get_agent_id(analysis_agent)
        
        assert distribution[python_id] == 3
        assert distribution[analysis_id] == 1
        
    def test_exclude_agents(self, calculator, mock_task, python_agent, analysis_agent):
        requirements = [
            TaskRequirement(
                capability_name="Programming",
                capability_type=CapabilityType.TECHNICAL,
                minimum_proficiency=0.3,
                weight=1.0
            )
        ]
        
        assignment = calculator.calculate_responsibility_assignment(
            mock_task, requirements, AssignmentStrategy.GREEDY, 
            exclude_agents=[python_agent]
        )
        
        if assignment:  # If any agent was assigned
            python_id = calculator.hierarchy._get_agent_id(python_agent)
            assert assignment.agent_id != python_id
            
    def test_no_capable_agents(self, calculator, mock_task):
        requirements = [
            TaskRequirement(
                capability_name="Quantum Computing",
                capability_type=CapabilityType.TECHNICAL,
                minimum_proficiency=0.9,
                weight=1.0
            )
        ]
        
        assignment = calculator.calculate_responsibility_assignment(
            mock_task, requirements, AssignmentStrategy.GREEDY
        )
        
        assert assignment is None
        
    def test_workload_penalty_calculation(self, calculator):
        assert calculator._calculate_workload_penalty(0) == 0.0
        
        penalty_1 = calculator._calculate_workload_penalty(1)
        penalty_5 = calculator._calculate_workload_penalty(5)
        
        assert penalty_1 < penalty_5  # Higher workload should have higher penalty
        assert penalty_5 <= 0.8  # Should not exceed maximum penalty
