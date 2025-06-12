"""Tests for TaskInfo dataclass."""

import pytest
from datetime import datetime
from unittest.mock import Mock

try:
    from crewai.a2a.crew_agent_executor import TaskInfo
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A integration not available")
class TestTaskInfo:
    """Test TaskInfo dataclass functionality."""
    
    def test_task_info_creation(self):
        """Test TaskInfo creation with required fields."""
        mock_task = Mock()
        started_at = datetime.now()
        
        task_info = TaskInfo(task=mock_task, started_at=started_at)
        
        assert task_info.task == mock_task
        assert task_info.started_at == started_at
        assert task_info.status == "running"
    
    def test_task_info_with_custom_status(self):
        """Test TaskInfo creation with custom status."""
        mock_task = Mock()
        started_at = datetime.now()
        
        task_info = TaskInfo(
            task=mock_task,
            started_at=started_at,
            status="completed"
        )
        
        assert task_info.status == "completed"
    
    def test_task_info_status_update(self):
        """Test TaskInfo status can be updated."""
        mock_task = Mock()
        started_at = datetime.now()
        
        task_info = TaskInfo(task=mock_task, started_at=started_at)
        assert task_info.status == "running"
        
        task_info.status = "cancelled"
        assert task_info.status == "cancelled"
