"""
Tests for responsibility tracking data models.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from crewai.responsibility.models import (
    AgentCapability,
    CapabilityType,
    ResponsibilityAssignment,
    AccountabilityRecord,
    PerformanceMetrics,
    TaskRequirement
)


class TestAgentCapability:
    def test_create_capability(self):
        capability = AgentCapability(
            name="Python Programming",
            capability_type=CapabilityType.TECHNICAL,
            proficiency_level=0.8,
            confidence_score=0.9,
            description="Expert in Python development",
            keywords=["python", "programming", "development"]
        )
        
        assert capability.name == "Python Programming"
        assert capability.capability_type == CapabilityType.TECHNICAL
        assert capability.proficiency_level == 0.8
        assert capability.confidence_score == 0.9
        assert "python" in capability.keywords
        
    def test_update_proficiency(self):
        capability = AgentCapability(
            name="Data Analysis",
            capability_type=CapabilityType.ANALYTICAL,
            proficiency_level=0.5,
            confidence_score=0.6
        )
        
        old_updated = capability.last_updated
        capability.update_proficiency(0.7, 0.8)
        
        assert capability.proficiency_level == 0.7
        assert capability.confidence_score == 0.8
        assert capability.last_updated > old_updated
        
    def test_proficiency_bounds(self):
        capability = AgentCapability(
            name="Test",
            capability_type=CapabilityType.TECHNICAL,
            proficiency_level=0.5,
            confidence_score=0.5
        )
        
        capability.update_proficiency(1.5, 1.2)
        assert capability.proficiency_level == 1.0
        assert capability.confidence_score == 1.0
        
        capability.update_proficiency(-0.5, -0.2)
        assert capability.proficiency_level == 0.0
        assert capability.confidence_score == 0.0


class TestResponsibilityAssignment:
    def test_create_assignment(self):
        assignment = ResponsibilityAssignment(
            agent_id="agent_1",
            task_id="task_1",
            responsibility_score=0.85,
            capability_matches=["Python Programming", "Data Analysis"],
            reasoning="Best match for technical requirements"
        )
        
        assert assignment.agent_id == "agent_1"
        assert assignment.task_id == "task_1"
        assert assignment.responsibility_score == 0.85
        assert len(assignment.capability_matches) == 2
        assert assignment.success is None
        
    def test_mark_completed(self):
        assignment = ResponsibilityAssignment(
            agent_id="agent_1",
            task_id="task_1",
            responsibility_score=0.85,
            reasoning="Test assignment"
        )
        
        assert assignment.completed_at is None
        assert assignment.success is None
        
        assignment.mark_completed(True)
        
        assert assignment.completed_at is not None
        assert assignment.success is True


class TestAccountabilityRecord:
    def test_create_record(self):
        record = AccountabilityRecord(
            agent_id="agent_1",
            action_type="task_execution",
            action_description="Executed data analysis task",
            task_id="task_1",
            context={"complexity": "high", "duration": 3600}
        )
        
        assert record.agent_id == "agent_1"
        assert record.action_type == "task_execution"
        assert record.context["complexity"] == "high"
        assert record.outcome is None
        
    def test_set_outcome(self):
        record = AccountabilityRecord(
            agent_id="agent_1",
            action_type="decision",
            action_description="Chose algorithm X"
        )
        
        record.set_outcome("Algorithm performed well", True)
        
        assert record.outcome == "Algorithm performed well"
        assert record.success is True


class TestPerformanceMetrics:
    def test_create_metrics(self):
        metrics = PerformanceMetrics(agent_id="agent_1")
        
        assert metrics.agent_id == "agent_1"
        assert metrics.total_tasks == 0
        assert metrics.success_rate == 0.0
        assert metrics.quality_score == 0.5
        
    def test_update_metrics_success(self):
        metrics = PerformanceMetrics(agent_id="agent_1")
        
        metrics.update_metrics(True, 1800, 0.8)
        
        assert metrics.total_tasks == 1
        assert metrics.successful_tasks == 1
        assert metrics.failed_tasks == 0
        assert metrics.success_rate == 1.0
        assert metrics.average_completion_time == 1800
        assert metrics.reliability_score == 1.0
        
    def test_update_metrics_failure(self):
        metrics = PerformanceMetrics(agent_id="agent_1")
        
        metrics.update_metrics(False, 3600)
        
        assert metrics.total_tasks == 1
        assert metrics.successful_tasks == 0
        assert metrics.failed_tasks == 1
        assert metrics.success_rate == 0.0
        
    def test_update_metrics_mixed(self):
        metrics = PerformanceMetrics(agent_id="agent_1")
        
        metrics.update_metrics(True, 1800, 0.8)
        metrics.update_metrics(False, 3600, 0.3)
        metrics.update_metrics(True, 2400, 0.9)
        
        assert metrics.total_tasks == 3
        assert metrics.successful_tasks == 2
        assert metrics.failed_tasks == 1
        assert abs(metrics.success_rate - 2/3) < 0.001


class TestTaskRequirement:
    def test_create_requirement(self):
        requirement = TaskRequirement(
            capability_name="Python Programming",
            capability_type=CapabilityType.TECHNICAL,
            minimum_proficiency=0.7,
            weight=1.5,
            keywords=["python", "coding"]
        )
        
        assert requirement.capability_name == "Python Programming"
        assert requirement.capability_type == CapabilityType.TECHNICAL
        assert requirement.minimum_proficiency == 0.7
        assert requirement.weight == 1.5
        assert "python" in requirement.keywords
