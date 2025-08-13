import pytest
from unittest.mock import Mock, patch
from crewai.utilities.events.utils.console_formatter import ConsoleFormatter


class TestConsoleFormatterVerbose:
    """Test verbose output functionality in console formatter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ConsoleFormatter(verbose=True)
        
    def test_get_task_display_name_with_name(self):
        """Test task display name when task has a name."""
        task = Mock()
        task.name = "Research Market Trends"
        task.id = "12345678-1234-5678-9012-123456789abc"
        
        result = self.formatter._get_task_display_name(task)
        assert "Research Market Trends" in result
        assert "12345678" in result
        
    def test_get_task_display_name_with_description_only(self):
        """Test task display name when task has no name but has description."""
        task = Mock()
        task.name = None
        task.description = "Analyze current market trends and provide insights"
        task.id = "12345678-1234-5678-9012-123456789abc"
        
        result = self.formatter._get_task_display_name(task)
        assert "Analyze current market trends" in result
        assert "12345678" in result
        
    def test_get_task_display_name_long_description_truncated(self):
        """Test task display name truncates long descriptions."""
        task = Mock()
        task.name = None
        task.description = "This is a very long task description that should be truncated because it exceeds the maximum length"
        task.id = "12345678-1234-5678-9012-123456789abc"
        
        result = self.formatter._get_task_display_name(task)
        assert len(result.split("(ID:")[0].strip()) <= 53
        assert "..." in result
        
    def test_get_task_display_name_fallback_to_id(self):
        """Test task display name falls back to ID when no name or description."""
        task = Mock()
        task.name = None
        task.description = None
        task.id = "12345678-1234-5678-9012-123456789abc"
        
        result = self.formatter._get_task_display_name(task)
        assert result == "12345678-1234-5678-9012-123456789abc"
        
    @patch('crewai.utilities.events.utils.console_formatter.ConsoleFormatter.print')
    def test_create_task_branch_uses_task_name(self, mock_print):
        """Test create_task_branch displays task name instead of ID."""
        task = Mock()
        task.name = "Write Blog Post"
        task.id = "12345678-1234-5678-9012-123456789abc"
        
        crew_tree = Mock()
        crew_tree.add.return_value = Mock()
        
        self.formatter.create_task_branch(crew_tree, task)
        
        call_args = crew_tree.add.call_args[0][0]
        assert "Write Blog Post" in str(call_args)
        assert "12345678" in str(call_args)
        
    @patch('crewai.utilities.events.utils.console_formatter.ConsoleFormatter.print')
    def test_update_task_status_uses_task_name(self, mock_print):
        """Test update_task_status displays task name instead of ID."""
        task = Mock()
        task.name = "Data Analysis"
        task.id = "12345678-1234-5678-9012-123456789abc"
        
        crew_tree = Mock()
        branch = Mock()
        branch.label = "12345678-1234-5678-9012-123456789abc"
        crew_tree.children = [branch]
        
        self.formatter.update_task_status(crew_tree, task, "Data Analyst", "completed")
        
        updated_label = branch.label
        assert "Data Analysis" in str(updated_label)

    def test_verbose_disabled_returns_none(self):
        """Test that methods return None when verbose is disabled."""
        formatter = ConsoleFormatter(verbose=False)
        task = Mock()
        
        result = formatter.create_task_branch(Mock(), task)
        assert result is None
        
        formatter.update_task_status(Mock(), task, "Agent", "completed")
