"""Test that context=[] is respected and doesn't include previous task outputs."""
import pytest
from unittest import mock
from crewai import Agent, Task, Crew, Process
from crewai.tasks.task_output import TaskOutput, OutputFormat
from crewai.utilities.formatter import aggregate_raw_outputs_from_tasks, aggregate_raw_outputs_from_task_outputs

def test_context_empty_list():
    """Test that context=[] is respected and doesn't include previous task outputs."""
    
    
    researcher = Agent(
        role='Researcher',
        goal='Research thoroughly',
        backstory='You are an expert researcher'
    )
    
    task_with_empty_context = Task(
        description='Task with empty context',
        expected_output='Output',
        agent=researcher,
        context=[]  # Explicitly set context to empty list
    )
    
    task_outputs = [
        TaskOutput(
            description="Previous task output",
            raw="Previous task result",
            agent="Researcher",
            json_dict=None,
            output_format=OutputFormat.RAW,
            pydantic=None,
            summary="Previous task result",
        )
    ]
    
    crew = Crew(
        agents=[researcher],
        tasks=[task_with_empty_context],
        process=Process.sequential,
        verbose=False
    )
    
    with mock.patch('crewai.agent.Agent.execute_task') as mock_execute:
        mock_execute.return_value = "Mocked execution result"
        
        context = crew._get_context(task_with_empty_context, task_outputs)
        
        # So it should return the aggregated task_outputs
        expected_context = aggregate_raw_outputs_from_task_outputs(task_outputs)
        
        assert context == expected_context
        
        assert not (task_with_empty_context.context and len(task_with_empty_context.context) > 0)
        
        other_task = Task(
            description='Other task',
            expected_output='Output',
            agent=researcher
        )
        
        task_with_context = Task(
            description='Task with context',
            expected_output='Output',
            agent=researcher,
            context=[other_task]  # Non-empty context
        )
        
        assert task_with_context.context and len(task_with_context.context) > 0
