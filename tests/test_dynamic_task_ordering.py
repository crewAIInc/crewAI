import pytest
from unittest.mock import Mock

from crewai import Agent, Crew, Task
from crewai.process import Process
from crewai.task import TaskOutput


@pytest.fixture
def agents():
    return [
        Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1"),
        Agent(role="Agent 2", goal="Goal 2", backstory="Backstory 2"),
        Agent(role="Agent 3", goal="Goal 3", backstory="Backstory 3"),
    ]


@pytest.fixture
def tasks(agents):
    return [
        Task(description="Task 1", expected_output="Output 1", agent=agents[0]),
        Task(description="Task 2", expected_output="Output 2", agent=agents[1]),
        Task(description="Task 3", expected_output="Output 3", agent=agents[2]),
    ]


def test_sequential_process_with_reverse_ordering(agents, tasks):
    """Test sequential process with reverse task ordering."""
    execution_order = []
    
    def reverse_ordering_callback(all_tasks, completed_outputs, current_index):
        completed_tasks = {id(task) for task in all_tasks if task.output is not None}
        remaining_indices = [i for i in range(len(all_tasks)) 
                           if id(all_tasks[i]) not in completed_tasks]
        if remaining_indices:
            next_index = max(remaining_indices)
            execution_order.append(next_index)
            return next_index
        return None
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        task_ordering_callback=reverse_ordering_callback,
        verbose=False
    )
    
    result = crew.kickoff()
    
    assert len(result.tasks_output) == 3
    assert execution_order == [2, 1, 0]


def test_hierarchical_process_with_priority_ordering(agents, tasks):
    """Test hierarchical process with priority-based task ordering."""
    
    tasks[0].priority = 3
    tasks[1].priority = 1  
    tasks[2].priority = 2
    
    execution_order = []
    
    def priority_ordering_callback(all_tasks, completed_outputs, current_index):
        completed_tasks = {id(task) for task in all_tasks if task.output is not None}
        remaining_tasks = [
            (i, task) for i, task in enumerate(all_tasks)
            if id(task) not in completed_tasks
        ]
        
        if remaining_tasks:
            remaining_tasks.sort(key=lambda x: getattr(x[1], 'priority', 999))
            next_index = remaining_tasks[0][0]
            execution_order.append(next_index)
            return next_index
        
        return None
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.hierarchical,
        manager_llm="gpt-4o",
        task_ordering_callback=priority_ordering_callback,
        verbose=False
    )
    
    result = crew.kickoff()
    assert len(result.tasks_output) == 3
    assert execution_order == [1, 2, 0]


def test_task_ordering_callback_with_task_object_return():
    """Test callback returning Task object instead of index."""
    
    agents = [Agent(role="Agent", goal="Goal", backstory="Backstory")]
    tasks = [
        Task(description="Task A", expected_output="Output A", agent=agents[0]),
        Task(description="Task B", expected_output="Output B", agent=agents[0]),
    ]
    
    execution_order = []
    
    def task_object_callback(all_tasks, completed_outputs, current_index):
        if len(completed_outputs) == 0:
            execution_order.append(1)
            return all_tasks[1]
        elif len(completed_outputs) == 1:
            execution_order.append(0)
            return all_tasks[0]
        return None
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        task_ordering_callback=task_object_callback,
        verbose=False
    )
    
    result = crew.kickoff()
    assert len(result.tasks_output) == 2
    assert execution_order == [1, 0]


def test_invalid_task_ordering_callback_index():
    """Test handling of invalid task index from callback."""
    
    agents = [Agent(role="Agent", goal="Goal", backstory="Backstory")]
    tasks = [Task(description="Task", expected_output="Output", agent=agents[0])]
    
    def invalid_callback(all_tasks, completed_outputs, current_index):
        return 999
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        task_ordering_callback=invalid_callback,
        verbose=False
    )
    
    result = crew.kickoff()
    assert len(result.tasks_output) == 1


def test_task_ordering_callback_exception_handling():
    """Test handling of exceptions in task ordering callback."""
    
    agents = [Agent(role="Agent", goal="Goal", backstory="Backstory")]
    tasks = [Task(description="Task", expected_output="Output", agent=agents[0])]
    
    def failing_callback(all_tasks, completed_outputs, current_index):
        raise ValueError("Callback error")
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        task_ordering_callback=failing_callback,
        verbose=False
    )
    
    result = crew.kickoff()
    assert len(result.tasks_output) == 1


def test_task_ordering_callback_validation():
    """Test validation of task ordering callback signature."""
    
    agents = [Agent(role="Agent", goal="Goal", backstory="Backstory")]
    tasks = [Task(description="Task", expected_output="Output", agent=agents[0])]
    
    def invalid_signature_callback(only_one_param):
        return 0
    
    with pytest.raises(ValueError, match="task_ordering_callback must accept exactly 3 parameters"):
        Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            task_ordering_callback=invalid_signature_callback
        )


def test_no_task_ordering_callback_default_behavior():
    """Test that default behavior is unchanged when no callback is provided."""
    
    agents = [Agent(role="Agent", goal="Goal", backstory="Backstory")]
    tasks = [
        Task(description="Task 1", expected_output="Output 1", agent=agents[0]),
        Task(description="Task 2", expected_output="Output 2", agent=agents[0]),
    ]
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False
    )
    
    result = crew.kickoff()
    assert len(result.tasks_output) == 2


def test_task_ordering_callback_with_none_return():
    """Test callback returning None for default ordering."""
    
    agents = [Agent(role="Agent", goal="Goal", backstory="Backstory")]
    tasks = [
        Task(description="Task 1", expected_output="Output 1", agent=agents[0]),
        Task(description="Task 2", expected_output="Output 2", agent=agents[0]),
    ]
    
    def none_callback(all_tasks, completed_outputs, current_index):
        return None
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        task_ordering_callback=none_callback,
        verbose=False
    )
    
    result = crew.kickoff()
    assert len(result.tasks_output) == 2


def test_task_ordering_callback_invalid_task_object():
    """Test handling of invalid Task object from callback."""
    
    agents = [Agent(role="Agent", goal="Goal", backstory="Backstory")]
    tasks = [Task(description="Task", expected_output="Output", agent=agents[0])]
    
    invalid_task = Task(description="Invalid", expected_output="Invalid", agent=agents[0])
    
    def invalid_task_callback(all_tasks, completed_outputs, current_index):
        return invalid_task
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        task_ordering_callback=invalid_task_callback,
        verbose=False
    )
    
    result = crew.kickoff()
    assert len(result.tasks_output) == 1


def test_task_ordering_callback_invalid_return_type():
    """Test handling of invalid return type from callback."""
    
    agents = [Agent(role="Agent", goal="Goal", backstory="Backstory")]
    tasks = [Task(description="Task", expected_output="Output", agent=agents[0])]
    
    def invalid_type_callback(all_tasks, completed_outputs, current_index):
        return "invalid"
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        task_ordering_callback=invalid_type_callback,
        verbose=False
    )
    
    result = crew.kickoff()
    assert len(result.tasks_output) == 1


def test_task_ordering_prevents_infinite_loops():
    """Test that task ordering prevents infinite loops by tracking executed tasks."""
    
    agents = [Agent(role="Agent", goal="Goal", backstory="Backstory")]
    tasks = [
        Task(description="Task 1", expected_output="Output 1", agent=agents[0]),
        Task(description="Task 2", expected_output="Output 2", agent=agents[0]),
    ]
    
    call_count = 0
    
    def loop_callback(all_tasks, completed_outputs, current_index):
        nonlocal call_count
        call_count += 1
        if call_count > 10:
            pytest.fail("Callback called too many times, possible infinite loop")
        return 0
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        task_ordering_callback=loop_callback,
        verbose=False
    )
    
    result = crew.kickoff()
    assert len(result.tasks_output) == 2
    assert call_count <= 4
