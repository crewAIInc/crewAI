import unittest

from crewai import Agent, Task


class TestTaskInitFix(unittest.TestCase):
    """Test the fix for issue #2219 where agent methods are not handled correctly in tasks."""

    def test_task_init_handles_callable_agent(self):
        """Test that the Task.__init__ method correctly handles callable agents."""
        
        # Create an agent instance
        agent_instance = Agent(
            role="Test Agent",
            goal="Test Goal",
            backstory="Test Backstory"
        )
        
        # Create a callable that returns the agent instance
        def callable_agent():
            return agent_instance
        
        # Create a task with the callable agent
        task = Task(
            description="Test Task",
            expected_output="Test Output",
            agent=callable_agent
        )
        
        # Verify that the agent in the task is an instance, not a callable
        self.assertIsInstance(task.agent, Agent)
        self.assertEqual(task.agent.role, "Test Agent")
        self.assertIs(task.agent, agent_instance)
        
    def test_task_init_handles_invalid_callable_agent(self):
        """Test that the Task.__init__ method correctly handles invalid callable agents."""
        
        # Create a callable that returns an invalid agent (not an Agent instance)
        def invalid_callable_agent():
            return "Not an agent"
        
        # Create a task with the invalid callable agent
        with self.assertRaises(ValueError) as context:
            task = Task(
                description="Test Task",
                expected_output="Test Output",
                agent=invalid_callable_agent
            )
        
        # Verify that the error message is correct
        self.assertIn("Expected BaseAgent instance", str(context.exception))
