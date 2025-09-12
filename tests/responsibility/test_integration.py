"""
Integration tests for the responsibility tracking system.
"""

from unittest.mock import Mock

import pytest

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.responsibility.assignment import AssignmentStrategy
from crewai.responsibility.models import (
    AgentCapability,
    CapabilityType,
    TaskRequirement,
)
from crewai.responsibility.system import ResponsibilitySystem
from crewai.task import Task


class TestResponsibilitySystemIntegration:
    @pytest.fixture
    def system(self):
        return ResponsibilitySystem()

    @pytest.fixture
    def python_agent(self):
        agent = Mock(spec=BaseAgent)
        agent.role = "Python Developer"
        return agent

    @pytest.fixture
    def analysis_agent(self):
        agent = Mock(spec=BaseAgent)
        agent.role = "Data Analyst"
        return agent

    @pytest.fixture
    def python_capability(self):
        return AgentCapability(
            name="Python Programming",
            capability_type=CapabilityType.TECHNICAL,
            proficiency_level=0.9,
            confidence_score=0.8,
            keywords=["python", "programming", "development"]
        )

    @pytest.fixture
    def analysis_capability(self):
        return AgentCapability(
            name="Data Analysis",
            capability_type=CapabilityType.ANALYTICAL,
            proficiency_level=0.8,
            confidence_score=0.9,
            keywords=["data", "analysis", "statistics"]
        )

    @pytest.fixture
    def mock_task(self):
        task = Mock(spec=Task)
        task.id = "integration_test_task"
        task.description = "Complex data processing task requiring Python skills"
        return task

    def test_full_workflow(self, system, python_agent, python_capability, mock_task):
        """Test complete workflow from agent registration to task completion."""

        system.register_agent(python_agent, [python_capability])

        status = system.get_agent_status(python_agent)
        assert status["role"] == "Python Developer"
        assert len(status["capabilities"]) == 1
        assert status["capabilities"][0]["name"] == "Python Programming"

        requirements = [
            TaskRequirement(
                capability_name="Python Programming",
                capability_type=CapabilityType.TECHNICAL,
                minimum_proficiency=0.5,
                weight=1.0
            )
        ]

        assignment = system.assign_task_responsibility(mock_task, requirements)

        assert assignment is not None
        assert assignment.task_id == "integration_test_task"
        assert assignment.responsibility_score > 0.5

        updated_status = system.get_agent_status(python_agent)
        assert updated_status["current_workload"] == 1

        system.complete_task(
            agent=python_agent,
            task=mock_task,
            success=True,
            completion_time=1800.0,
            quality_score=0.9,
            outcome_description="Task completed successfully"
        )

        final_status = system.get_agent_status(python_agent)
        assert final_status["performance"]["total_tasks"] == 1
        assert final_status["performance"]["success_rate"] == 1.0
        assert final_status["current_workload"] == 0  # Should be decremented

    def test_multi_agent_scenario(self, system, python_agent, analysis_agent,
                                 python_capability, analysis_capability, mock_task):
        """Test scenario with multiple agents and capabilities."""

        system.register_agent(python_agent, [python_capability])
        system.register_agent(analysis_agent, [analysis_capability])

        requirements = [
            TaskRequirement(
                capability_name="Python Programming",
                capability_type=CapabilityType.TECHNICAL,
                minimum_proficiency=0.7,
                weight=1.0
            ),
            TaskRequirement(
                capability_name="Data Analysis",
                capability_type=CapabilityType.ANALYTICAL,
                minimum_proficiency=0.6,
                weight=0.8
            )
        ]

        greedy_assignment = system.assign_task_responsibility(
            mock_task, requirements, AssignmentStrategy.GREEDY
        )

        assert greedy_assignment is not None

        system.calculator.update_workload(python_agent, 5)

        balanced_assignment = system.assign_task_responsibility(
            mock_task, requirements, AssignmentStrategy.BALANCED
        )

        assert balanced_assignment is not None

    def test_delegation_workflow(self, system, python_agent, analysis_agent,
                                python_capability, analysis_capability, mock_task):
        """Test task delegation between agents."""

        system.register_agent(python_agent, [python_capability], supervisor=None)
        system.register_agent(analysis_agent, [analysis_capability], supervisor=python_agent)

        system.delegate_task(
            delegating_agent=python_agent,
            receiving_agent=analysis_agent,
            task=mock_task,
            reason="Analysis expertise required"
        )

        analysis_status = system.get_agent_status(analysis_agent)

        assert analysis_status["current_workload"] > 0

        delegation_records = system.accountability.get_agent_records(
            python_agent, action_type="delegation"
        )
        assert len(delegation_records) > 0

    def test_performance_based_capability_adjustment(self, system, python_agent,
                                                   python_capability, mock_task):
        """Test that capabilities are adjusted based on performance."""

        system.register_agent(python_agent, [python_capability])

        for i in range(5):
            task = Mock(spec=Task)
            task.id = f"task_{i}"
            task.description = f"Task {i}"

            system.complete_task(
                agent=python_agent,
                task=task,
                success=True,
                completion_time=1800.0,
                quality_score=0.9
            )

        updated_capabilities = system.hierarchy.get_agent_capabilities(python_agent)

        assert len(updated_capabilities) == 1

    def test_system_overview_and_recommendations(self, system, python_agent,
                                               analysis_agent, python_capability,
                                               analysis_capability):
        """Test system overview and recommendation generation."""

        system.register_agent(python_agent, [python_capability])
        system.register_agent(analysis_agent, [analysis_capability])

        overview = system.get_system_overview()

        assert overview["enabled"] is True
        assert overview["total_agents"] == 2
        assert "capability_distribution" in overview
        assert "system_performance" in overview

        recommendations = system.generate_recommendations()

        assert isinstance(recommendations, list)

    def test_system_enable_disable(self, system, python_agent, python_capability, mock_task):
        """Test enabling and disabling the responsibility system."""

        assert system.enabled is True

        system.register_agent(python_agent, [python_capability])

        requirements = [
            TaskRequirement(
                capability_name="Python Programming",
                capability_type=CapabilityType.TECHNICAL,
                minimum_proficiency=0.5,
                weight=1.0
            )
        ]

        assignment = system.assign_task_responsibility(mock_task, requirements)
        assert assignment is not None

        system.disable_system()
        assert system.enabled is False

        disabled_assignment = system.assign_task_responsibility(mock_task, requirements)
        assert disabled_assignment is None

        disabled_status = system.get_agent_status(python_agent)
        assert disabled_status == {}

        system.enable_system()
        assert system.enabled is True

        enabled_assignment = system.assign_task_responsibility(mock_task, requirements)
        assert enabled_assignment is not None

    def test_accountability_tracking_integration(self, system, python_agent,
                                               python_capability, mock_task):
        """Test that all operations are properly logged for accountability."""

        system.register_agent(python_agent, [python_capability])

        registration_records = system.accountability.get_agent_records(
            python_agent, action_type="registration"
        )
        assert len(registration_records) == 1

        requirements = [
            TaskRequirement(
                capability_name="Python Programming",
                capability_type=CapabilityType.TECHNICAL,
                minimum_proficiency=0.5,
                weight=1.0
            )
        ]

        system.assign_task_responsibility(mock_task, requirements)

        assignment_records = system.accountability.get_agent_records(
            python_agent, action_type="task_assignment"
        )
        assert len(assignment_records) == 1

        system.complete_task(
            agent=python_agent,
            task=mock_task,
            success=True,
            completion_time=1800.0,
            quality_score=0.9
        )

        completion_records = system.accountability.get_agent_records(
            python_agent, action_type="task_completion"
        )
        assert len(completion_records) == 1

        report = system.accountability.generate_accountability_report(agent=python_agent)

        assert report["total_records"] >= 3  # At least registration, assignment, completion
        assert "registration" in report["action_counts"]
        assert "task_assignment" in report["action_counts"]
        assert "task_completion" in report["action_counts"]
