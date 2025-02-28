import unittest
from unittest.mock import patch, MagicMock

from crewai.task import Task
from langchain_core.agents import AgentFinish


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
        
    def test_task_max_dialogue_rounds_validation(self):
        """Test that Task validates max_dialogue_rounds as a positive integer."""
        # Create a task with invalid max_dialogue_rounds
        with self.assertRaises(ValueError):
            task = Task(
                description="Test task",
                expected_output="Test output",
                human_input=True,
                max_dialogue_rounds=0
            )
            
    def test_handle_regular_feedback_rounds(self):
        """Test that _handle_regular_feedback correctly handles multiple rounds."""
        from crewai.agents.crew_agent_executor import CrewAgentExecutor
        
        # Create a simple mock executor
        executor = MagicMock()
        executor.ask_for_human_input = True
        executor._ask_human_input = MagicMock(side_effect=["Feedback", ""])
        executor._process_feedback_iteration = MagicMock(return_value=MagicMock())
        
        # Create a sample initial answer
        initial_answer = MagicMock()
        
        # Call the method directly
        CrewAgentExecutor._handle_regular_feedback(
            executor, 
            initial_answer, 
            "Initial feedback", 
            max_rounds=3
        )
        
        # Verify the correct number of iterations occurred
        # First call for initial feedback, second call for empty feedback to end loop
        self.assertEqual(executor._ask_human_input.call_count, 2)
        # The _process_feedback_iteration is called for the initial feedback and the first round
        self.assertEqual(executor._process_feedback_iteration.call_count, 2)


if __name__ == "__main__":
    unittest.main()
