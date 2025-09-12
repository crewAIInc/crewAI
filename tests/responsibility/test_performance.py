"""
Tests for performance-based capability adjustment.
"""

import pytest
from datetime import timedelta
from unittest.mock import Mock

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.responsibility.models import AgentCapability, CapabilityType, PerformanceMetrics
from crewai.responsibility.hierarchy import CapabilityHierarchy
from crewai.responsibility.performance import PerformanceTracker


class TestPerformanceTracker:
    @pytest.fixture
    def hierarchy(self):
        return CapabilityHierarchy()
        
    @pytest.fixture
    def tracker(self, hierarchy):
        return PerformanceTracker(hierarchy)
        
    @pytest.fixture
    def mock_agent(self, hierarchy):
        agent = Mock(spec=BaseAgent)
        agent.role = "Test Agent"
        
        capability = AgentCapability(
            name="Python Programming",
            capability_type=CapabilityType.TECHNICAL,
            proficiency_level=0.7,
            confidence_score=0.8
        )
        
        hierarchy.add_agent(agent, [capability])
        return agent
        
    def test_record_task_completion_success(self, tracker, mock_agent):
        tracker.record_task_completion(
            agent=mock_agent,
            task_success=True,
            completion_time=1800.0,
            quality_score=0.9
        )
        
        metrics = tracker.get_performance_metrics(mock_agent)
        
        assert metrics is not None
        assert metrics.total_tasks == 1
        assert metrics.successful_tasks == 1
        assert metrics.failed_tasks == 0
        assert metrics.success_rate == 1.0
        assert metrics.average_completion_time == 1800.0
        assert metrics.quality_score > 0.5  # Should be updated towards 0.9
        
    def test_record_task_completion_failure(self, tracker, mock_agent):
        tracker.record_task_completion(
            agent=mock_agent,
            task_success=False,
            completion_time=3600.0,
            quality_score=0.3
        )
        
        metrics = tracker.get_performance_metrics(mock_agent)
        
        assert metrics is not None
        assert metrics.total_tasks == 1
        assert metrics.successful_tasks == 0
        assert metrics.failed_tasks == 1
        assert metrics.success_rate == 0.0
        
    def test_multiple_task_completions(self, tracker, mock_agent):
        tracker.record_task_completion(mock_agent, True, 1800.0, 0.8)
        tracker.record_task_completion(mock_agent, False, 3600.0, 0.4)
        tracker.record_task_completion(mock_agent, True, 2400.0, 0.9)
        
        metrics = tracker.get_performance_metrics(mock_agent)
        
        assert metrics.total_tasks == 3
        assert metrics.successful_tasks == 2
        assert metrics.failed_tasks == 1
        assert abs(metrics.success_rate - 2/3) < 0.001
        
    def test_capability_adjustment_on_success(self, tracker, mock_agent):
        initial_capabilities = tracker.hierarchy.get_agent_capabilities(mock_agent)
        initial_proficiency = initial_capabilities[0].proficiency_level
        
        tracker.record_task_completion(
            agent=mock_agent,
            task_success=True,
            completion_time=1800.0,
            quality_score=0.9,
            capability_used="Python Programming"
        )
        
        updated_capabilities = tracker.hierarchy.get_agent_capabilities(mock_agent)
        updated_proficiency = updated_capabilities[0].proficiency_level
        
        assert updated_proficiency >= initial_proficiency
        
    def test_capability_adjustment_on_failure(self, tracker, mock_agent):
        initial_capabilities = tracker.hierarchy.get_agent_capabilities(mock_agent)
        initial_proficiency = initial_capabilities[0].proficiency_level
        
        tracker.record_task_completion(
            agent=mock_agent,
            task_success=False,
            completion_time=3600.0,
            quality_score=0.2,
            capability_used="Python Programming"
        )
        
        updated_capabilities = tracker.hierarchy.get_agent_capabilities(mock_agent)
        updated_proficiency = updated_capabilities[0].proficiency_level
        
        assert updated_proficiency <= initial_proficiency
        
    def test_adjust_capabilities_based_on_performance(self, tracker, mock_agent):
        for _ in range(5):
            tracker.record_task_completion(mock_agent, True, 1800.0, 0.9)
        for _ in range(2):
            tracker.record_task_completion(mock_agent, False, 3600.0, 0.3)
            
        adjustments = tracker.adjust_capabilities_based_on_performance(mock_agent)
        
        assert isinstance(adjustments, list)
        
    def test_get_performance_trends(self, tracker, mock_agent):
        tracker.record_task_completion(mock_agent, True, 1800.0, 0.8)
        tracker.record_task_completion(mock_agent, True, 2000.0, 0.9)
        
        trends = tracker.get_performance_trends(mock_agent)
        
        assert "success_rate" in trends
        assert "quality_score" in trends
        assert "efficiency_score" in trends
        assert "reliability_score" in trends
        
        assert len(trends["success_rate"]) > 0
        
    def test_identify_improvement_opportunities(self, tracker, mock_agent):
        tracker.record_task_completion(mock_agent, False, 7200.0, 0.3)  # Long time, low quality
        tracker.record_task_completion(mock_agent, False, 6000.0, 0.4)
        tracker.record_task_completion(mock_agent, True, 5400.0, 0.5)
        
        opportunities = tracker.identify_improvement_opportunities(mock_agent)
        
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        areas = [opp["area"] for opp in opportunities]
        assert "success_rate" in areas or "quality" in areas or "efficiency" in areas
        
    def test_compare_agent_performance(self, tracker, hierarchy):
        agent1 = Mock(spec=BaseAgent)
        agent1.role = "Agent 1"
        agent2 = Mock(spec=BaseAgent)
        agent2.role = "Agent 2"
        
        capability = AgentCapability(
            name="Test Capability",
            capability_type=CapabilityType.TECHNICAL,
            proficiency_level=0.7,
            confidence_score=0.8
        )
        
        hierarchy.add_agent(agent1, [capability])
        hierarchy.add_agent(agent2, [capability])
        
        tracker.record_task_completion(agent1, True, 1800.0, 0.9)  # Good performance
        tracker.record_task_completion(agent1, True, 2000.0, 0.8)
        
        tracker.record_task_completion(agent2, False, 3600.0, 0.4)  # Poor performance
        tracker.record_task_completion(agent2, True, 4000.0, 0.5)
        
        comparison = tracker.compare_agent_performance([agent1, agent2], metric="overall")
        
        assert len(comparison) == 2
        assert comparison[0][1] > comparison[1][1]  # First agent should have higher score
        
        success_comparison = tracker.compare_agent_performance([agent1, agent2], metric="success_rate")
        assert len(success_comparison) == 2
        
    def test_learning_rate_effect(self, tracker, mock_agent):
        original_learning_rate = tracker.learning_rate
        
        tracker.learning_rate = 0.5
        
        initial_capabilities = tracker.hierarchy.get_agent_capabilities(mock_agent)
        initial_proficiency = initial_capabilities[0].proficiency_level
        
        tracker.record_task_completion(
            mock_agent, True, 1800.0, 0.9, capability_used="Python Programming"
        )
        
        high_lr_capabilities = tracker.hierarchy.get_agent_capabilities(mock_agent)
        high_lr_proficiency = high_lr_capabilities[0].proficiency_level
        
        tracker.hierarchy.update_agent_capability(
            mock_agent, "Python Programming", initial_proficiency, 0.8
        )
        tracker.learning_rate = 0.01
        
        tracker.record_task_completion(
            mock_agent, True, 1800.0, 0.9, capability_used="Python Programming"
        )
        
        low_lr_capabilities = tracker.hierarchy.get_agent_capabilities(mock_agent)
        low_lr_proficiency = low_lr_capabilities[0].proficiency_level
        
        high_lr_change = abs(high_lr_proficiency - initial_proficiency)
        low_lr_change = abs(low_lr_proficiency - initial_proficiency)
        
        assert high_lr_change > low_lr_change
        
        tracker.learning_rate = original_learning_rate
        
    def test_performance_metrics_creation(self, tracker, mock_agent):
        assert tracker.get_performance_metrics(mock_agent) is None
        
        tracker.record_task_completion(mock_agent, True, 1800.0)
        
        metrics = tracker.get_performance_metrics(mock_agent)
        assert metrics is not None
        assert metrics.agent_id == tracker._get_agent_id(mock_agent)
