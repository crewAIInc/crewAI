import unittest
from unittest.mock import patch

from crewai.task import Task


class TestMultiRoundDialogue(unittest.TestCase):
    """Test the multi-round dialogue functionality."""

    def test_task_max_dialogue_rounds_default(self):
        """Test that Task has a default max_dialogue_rounds of 10."""
        # Create a task with default max_dialogue_rounds
        task = Task(
            description="Test task",
            expected_output="Test output",
            human_input=True
        )
        
        # Verify the default value
        self.assertEqual(task.max_dialogue_rounds, 10)

    def test_task_max_dialogue_rounds_custom(self):
        """Test that Task accepts a custom max_dialogue_rounds."""
        # Create a task with custom max_dialogue_rounds
        task = Task(
            description="Test task",
            expected_output="Test output",
            human_input=True,
            max_dialogue_rounds=5
        )
        
        # Verify the custom value
        self.assertEqual(task.max_dialogue_rounds, 5)


if __name__ == "__main__":
    unittest.main()
