"""Tests for console formatter retry attempt display functionality.

This module tests the retry attempt display feature that shows attempt numbers
when guardrails fail and tasks are retried.
"""

from unittest.mock import patch

from rich.tree import Tree
from rich.text import Text

from crewai.events.utils.console_formatter import ConsoleFormatter


class TestConsoleFormatterRetryDisplay:
    """Test console formatter retry attempt display functionality."""

    def setup_method(self):
        """Setup test environment for each test."""
        self.formatter = ConsoleFormatter(verbose=True)
        
    def test_retry_display_with_task_name_and_id(self):
        """Test retry attempt display for task with name and ID."""
        # Setup crew tree and task branch
        crew_tree = Tree("Test Crew")
        self.formatter.current_crew_tree = crew_tree
        
        # Create task branch with name and ID
        task_content = Text()
        task_content.append("📋 Task: ", style="yellow bold")
        task_content.append("Research Task", style="yellow bold")
        task_content.append(" (ID: task_123)", style="yellow dim")
        
        task_branch = crew_tree.add(task_content)
        self.formatter.current_task_branch = task_branch
        
        with patch.object(self.formatter, 'print'):
            # Test retry attempt display
            self.formatter.update_task_branch_for_retry(2)
            
            # Verify the label shows attempt number and preserves task info
            updated_label = str(task_branch.label)
            assert "Research Task [Attempt 2]" in updated_label
            assert "task_123" in updated_label
            assert "Retrying Task..." in updated_label

    def test_retry_display_with_id_only(self):
        """Test retry attempt display for task with only ID (no name)."""
        # Setup crew tree and task branch
        crew_tree = Tree("Test Crew")
        self.formatter.current_crew_tree = crew_tree
        
        # Create task branch with ID only
        task_content = Text("📋 Task: task_456")
        task_branch = crew_tree.add(task_content)
        self.formatter.current_task_branch = task_branch
        
        with patch.object(self.formatter, 'print'):
            # Test retry attempt display
            self.formatter.update_task_branch_for_retry(3)
            
            # Verify the label shows attempt number
            updated_label = str(task_branch.label)
            assert "task_456 [Attempt 3]" in updated_label
            assert "Retrying Task..." in updated_label

    def test_multiple_retry_attempts_progression(self):
        """Test that multiple retry attempts show correct progression."""
        # Setup crew tree and task branch
        crew_tree = Tree("Test Crew")
        self.formatter.current_crew_tree = crew_tree
        
        task_content = Text("📋 Task: Progressive Task (ID: task_prog)")
        task_branch = crew_tree.add(task_content)
        self.formatter.current_task_branch = task_branch
        
        with patch.object(self.formatter, 'print'):
            # Test progression through multiple attempts
            for attempt in [2, 3, 4]:
                self.formatter.update_task_branch_for_retry(attempt)
                
                updated_label = str(task_branch.label)
                assert f"[Attempt {attempt}]" in updated_label
                assert "Progressive Task" in updated_label
                
                # Verify previous attempt numbers are not present
                for prev_attempt in range(2, attempt):
                    assert f"[Attempt {prev_attempt}]" not in updated_label

    def test_guardrail_failure_integration(self):
        """Test integration with guardrail failure events."""
        # Setup crew tree and task branch
        crew_tree = Tree("Test Crew")
        self.formatter.current_crew_tree = crew_tree
        
        task_content = Text("📋 Task: Validation Task (ID: task_validation)")
        task_branch = crew_tree.add(task_content)
        self.formatter.current_task_branch = task_branch
        
        with patch.object(self.formatter, 'print'), \
             patch.object(self.formatter, 'print_panel'):
            
            # Simulate guardrail failure (retry_count=1 means attempt 3)
            self.formatter.handle_guardrail_completed(False, "Validation failed", 1)
            
            # Verify that the task branch was updated with attempt number
            updated_label = str(task_branch.label)
            assert "[Attempt 3]" in updated_label  # retry_count + 2 = 3
            assert "Validation Task" in updated_label
            assert "task_validation" in updated_label
            assert "Retrying Task..." in updated_label

    def test_non_verbose_mode_no_display(self):
        """Test that retry display is disabled in non-verbose mode."""
        # Setup non-verbose formatter
        self.formatter = ConsoleFormatter(verbose=False)
        
        crew_tree = Tree("Test Crew")
        task_branch = crew_tree.add(Text("📋 Task: Test"))
        
        self.formatter.current_crew_tree = crew_tree
        self.formatter.current_task_branch = task_branch
        
        original_label = str(task_branch.label)
        
        with patch.object(self.formatter, 'print') as mock_print:
            # Should do nothing in non-verbose mode
            self.formatter.update_task_branch_for_retry(2)
            
            # Verify nothing changed
            assert str(task_branch.label) == original_label
            mock_print.assert_not_called()

    def test_task_id_preservation_across_retries(self):
        """Test that task IDs are preserved across multiple retry attempts."""
        # This test caught a real bug in the label parsing logic
        crew_tree = Tree("Test Crew")
        self.formatter.current_crew_tree = crew_tree
        
        task_content = Text("📋 Task: ID Test (ID: task_preserve_123)")
        task_branch = crew_tree.add(task_content)
        self.formatter.current_task_branch = task_branch
        
        with patch.object(self.formatter, 'print'), \
             patch.object(self.formatter, 'print_panel'):
            
            # Trigger multiple guardrail failures
            for retry_count in range(0, 3):
                self.formatter.handle_guardrail_completed(False, f"Failure {retry_count + 1}", retry_count)
                
                # Verify task ID is always preserved
                updated_label = str(task_branch.label)
                assert "task_preserve_123" in updated_label
                assert "ID Test" in updated_label