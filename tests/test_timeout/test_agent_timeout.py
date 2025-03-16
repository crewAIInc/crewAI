import time
from unittest.mock import MagicMock, patch

import pytest

from crewai import Agent, Task
from crewai.utilities import AgentExecutionTimeoutError


def test_agent_max_execution_time():
    """Test that max_execution_time parameter is enforced."""
    # Create a simple test function that will be used to simulate a long-running task
    def test_timeout():
        # Create an agent with a 1-second timeout
        with patch('crewai.agent.Agent.create_agent_executor'):
            agent = Agent(
                role="Test Agent",
                goal="Test timeout functionality",
                backstory="I am testing the timeout functionality",
                max_execution_time=1,
                verbose=True
            )
            
            # Create a task that will take longer than 1 second
            task = Task(
                description="Sleep for 5 seconds and then return a result",
                expected_output="The result after sleeping",
                agent=agent
            )
            
            # Mock the agent_executor to simulate a long-running task
            mock_executor = MagicMock()
            def side_effect(*args, **kwargs):
                # Sleep for longer than the timeout to trigger the timeout mechanism
                time.sleep(2)
                return {"output": "This should never be returned due to timeout"}
            
            mock_executor.invoke.side_effect = side_effect
            mock_executor.tools_names = []
            mock_executor.tools_description = []
            
            # Replace the agent's executor with our mock
            agent.agent_executor = mock_executor
            
            # Mock the event bus to avoid any real event emissions
            with patch('crewai.agent.crewai_event_bus'):
                # Execute the task and measure the time
                start_time = time.time()
                
                # We expect an Exception to be raised due to timeout
                with pytest.raises(Exception) as excinfo:
                    agent.execute_task(task)
                
                # Check that the execution time is close to 1 second (the timeout)
                execution_time = time.time() - start_time
                assert execution_time <= 2.1, f"Execution took {execution_time:.2f} seconds, expected ~1 second"
                
                # Check that the exception message mentions timeout or execution time
                error_message = str(excinfo.value).lower()
                assert any(term in error_message for term in ["timeout", "execution time", "exceeded maximum"])
    
    # Run the test function
    test_timeout()


def test_agent_timeout_error_message():
    """Test that the timeout error message includes agent and task information."""
    # Create an agent with a very short timeout
    with patch('crewai.agent.Agent.create_agent_executor'):
        agent = Agent(
            role="Test Agent",
            goal="Test timeout error messaging",
            backstory="I am testing the timeout error messaging",
            max_execution_time=1,  # Short timeout
            verbose=True
        )
        
        # Create a task
        task = Task(
            description="This task should timeout quickly",
            expected_output="This should never be returned",
            agent=agent
        )
        
        # Mock the agent_executor
        mock_executor = MagicMock()
        def side_effect(*args, **kwargs):
            # Sleep to trigger timeout
            time.sleep(2)
            return {"output": "This should never be returned"}
        
        mock_executor.invoke.side_effect = side_effect
        mock_executor.tools_names = []
        mock_executor.tools_description = []
        
        # Replace the agent's executor
        agent.agent_executor = mock_executor
        
        # Execute the task and expect an exception
        with patch('crewai.agent.crewai_event_bus'):
            with pytest.raises(Exception) as excinfo:
                agent.execute_task(task)
            
            # Verify error message contains agent name and task description
            error_message = str(excinfo.value).lower()
            assert "test agent" in error_message or "agent: test agent" in error_message
            assert "this task should timeout" in error_message or "task: this task should timeout" in error_message
