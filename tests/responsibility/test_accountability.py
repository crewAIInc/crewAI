"""
Tests for accountability logging system.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.task import Task
from crewai.responsibility.accountability import AccountabilityLogger


class TestAccountabilityLogger:
    @pytest.fixture
    def logger(self):
        return AccountabilityLogger()
        
    @pytest.fixture
    def mock_agent(self):
        agent = Mock(spec=BaseAgent)
        agent.role = "Test Agent"
        return agent
        
    @pytest.fixture
    def mock_task(self):
        task = Mock(spec=Task)
        task.id = "test_task_1"
        task.description = "Test task description"
        return task
        
    def test_log_action(self, logger, mock_agent, mock_task):
        context = {"complexity": "high", "priority": "urgent"}
        
        record = logger.log_action(
            agent=mock_agent,
            action_type="task_execution",
            action_description="Executed data processing task",
            task=mock_task,
            context=context
        )
        
        assert record.agent_id == "Test Agent_" + str(id(mock_agent))
        assert record.action_type == "task_execution"
        assert record.action_description == "Executed data processing task"
        assert record.task_id == "test_task_1"
        assert record.context["complexity"] == "high"
        assert len(logger.records) == 1
        
    def test_log_decision(self, logger, mock_agent, mock_task):
        alternatives = ["Option A", "Option B", "Option C"]
        
        record = logger.log_decision(
            agent=mock_agent,
            decision="Chose Option A",
            reasoning="Best performance characteristics",
            task=mock_task,
            alternatives_considered=alternatives
        )
        
        assert record.action_type == "decision"
        assert record.action_description == "Chose Option A"
        assert record.context["reasoning"] == "Best performance characteristics"
        assert record.context["alternatives_considered"] == alternatives
        
    def test_log_delegation(self, logger, mock_task):
        delegating_agent = Mock(spec=BaseAgent)
        delegating_agent.role = "Manager"
        
        receiving_agent = Mock(spec=BaseAgent)
        receiving_agent.role = "Developer"
        
        record = logger.log_delegation(
            delegating_agent=delegating_agent,
            receiving_agent=receiving_agent,
            task=mock_task,
            delegation_reason="Specialized expertise required"
        )
        
        assert record.action_type == "delegation"
        assert "Delegated task to Developer" in record.action_description
        assert record.context["receiving_agent_role"] == "Developer"
        assert record.context["delegation_reason"] == "Specialized expertise required"
        
    def test_log_task_completion(self, logger, mock_agent, mock_task):
        record = logger.log_task_completion(
            agent=mock_agent,
            task=mock_task,
            success=True,
            outcome_description="Task completed successfully with high quality",
            completion_time=1800.0
        )
        
        assert record.action_type == "task_completion"
        assert record.success is True
        assert record.outcome == "Task completed successfully with high quality"
        assert record.context["completion_time"] == 1800.0
        
    def test_get_agent_records(self, logger, mock_agent, mock_task):
        logger.log_action(mock_agent, "action1", "Description 1", mock_task)
        logger.log_action(mock_agent, "action2", "Description 2", mock_task)
        logger.log_decision(mock_agent, "decision1", "Reasoning", mock_task)
        
        all_records = logger.get_agent_records(mock_agent)
        assert len(all_records) == 3
        
        decision_records = logger.get_agent_records(mock_agent, action_type="decision")
        assert len(decision_records) == 1
        assert decision_records[0].action_type == "decision"
        
        recent_time = datetime.utcnow() - timedelta(minutes=1)
        recent_records = logger.get_agent_records(mock_agent, since=recent_time)
        assert len(recent_records) == 3  # All should be recent
        
    def test_get_task_records(self, logger, mock_agent, mock_task):
        other_task = Mock(spec=Task)
        other_task.id = "other_task"
        
        logger.log_action(mock_agent, "action1", "Description 1", mock_task)
        logger.log_action(mock_agent, "action2", "Description 2", other_task)
        logger.log_action(mock_agent, "action3", "Description 3", mock_task)
        
        task_records = logger.get_task_records(mock_task)
        assert len(task_records) == 2
        
        for record in task_records:
            assert record.task_id == "test_task_1"
            
    def test_get_delegation_chain(self, logger, mock_task):
        manager = Mock(spec=BaseAgent)
        manager.role = "Manager"
        
        supervisor = Mock(spec=BaseAgent)
        supervisor.role = "Supervisor"
        
        developer = Mock(spec=BaseAgent)
        developer.role = "Developer"
        
        logger.log_delegation(manager, supervisor, mock_task, "Initial delegation")
        logger.log_delegation(supervisor, developer, mock_task, "Further delegation")
        
        chain = logger.get_delegation_chain(mock_task)
        assert len(chain) == 2
        
        assert chain[0].context["receiving_agent_role"] == "Supervisor"
        assert chain[1].context["receiving_agent_role"] == "Developer"
        
    def test_generate_accountability_report(self, logger, mock_agent, mock_task):
        record1 = logger.log_action(mock_agent, "task_execution", "Task 1", mock_task)
        record1.set_outcome("Success", True)
        
        record2 = logger.log_action(mock_agent, "task_execution", "Task 2", mock_task)
        record2.set_outcome("Failed", False)
        
        record3 = logger.log_decision(mock_agent, "Decision 1", "Reasoning", mock_task)
        record3.set_outcome("Good decision", True)
        
        report = logger.generate_accountability_report(agent=mock_agent)
        
        assert report["total_records"] == 3
        assert report["action_counts"]["task_execution"] == 2
        assert report["action_counts"]["decision"] == 1
        assert report["success_counts"]["task_execution"] == 1
        assert report["failure_counts"]["task_execution"] == 1
        assert report["success_rates"]["task_execution"] == 0.5
        assert report["success_rates"]["decision"] == 1.0
        
        assert len(report["recent_actions"]) == 3
        
    def test_generate_system_wide_report(self, logger, mock_task):
        agent1 = Mock(spec=BaseAgent)
        agent1.role = "Agent 1"
        
        agent2 = Mock(spec=BaseAgent)
        agent2.role = "Agent 2"
        
        logger.log_action(agent1, "task_execution", "Task 1", mock_task)
        logger.log_action(agent2, "task_execution", "Task 2", mock_task)
        
        report = logger.generate_accountability_report()
        
        assert report["agent_id"] == "all_agents"
        assert report["total_records"] == 2
        assert report["action_counts"]["task_execution"] == 2
        
    def test_time_filtered_report(self, logger, mock_agent, mock_task):
        logger.log_action(mock_agent, "old_action", "Old action", mock_task)
        
        report = logger.generate_accountability_report(
            agent=mock_agent, 
            time_period=timedelta(hours=1)
        )
        
        assert report["total_records"] == 1
        
        report = logger.generate_accountability_report(
            agent=mock_agent, 
            time_period=timedelta(seconds=1)
        )
