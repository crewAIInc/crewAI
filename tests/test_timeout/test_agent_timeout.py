import pytest
import time
from unittest.mock import MagicMock, patch

from crewai import Agent, Task

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
                
                # Check that the exception message mentions timeout
                assert "timeout" in str(excinfo.value).lower() or "execution time" in str(excinfo.value).lower()
    
    # Run the test function
    test_timeout()
