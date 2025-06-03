import pytest
from crewai import Agent, Crew, Task, Process

def test_selective_execution_basic():
    """Test basic selective execution functionality without VCR."""
    
    researcher = Agent(
        role="Researcher",
        goal="Research topics",
        backstory="Expert researcher"
    )
    
    writer = Agent(
        role="Writer", 
        goal="Write content",
        backstory="Expert writer"
    )
    
    forecast_task = Task(
        description="Analyze forecast data",
        expected_output="Forecast analysis",
        agent=researcher,
        tags=["forecast", "analysis"]
    )
    
    news_task = Task(
        description="Summarize news",
        expected_output="News summary", 
        agent=writer,
        tags=["news", "summary"]
    )
    
    crew = Crew(
        agents=[researcher, writer],
        tasks=[forecast_task, news_task],
        task_selector=Crew.create_tag_selector()
    )
    
    assert crew.task_selector is not None
    
    selector = crew.task_selector
    
    inputs = {"action": "forecast"}
    assert selector(inputs, forecast_task) == True
    assert selector(inputs, news_task) == False
    
    inputs = {"action": "news"}
    assert selector(inputs, forecast_task) == False
    assert selector(inputs, news_task) == True
    
    print("All selective execution tests passed!")

def test_selective_process_validation():
    """Test that selective process requires task_selector."""
    from pydantic import ValidationError
    
    researcher = Agent(
        role="Researcher",
        goal="Research topics", 
        backstory="Expert researcher"
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=researcher
    )
    
    try:
        crew = Crew(
            agents=[researcher],
            tasks=[task],
            process=Process.selective
        )
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "task_selector" in str(e)
        print("Validation error correctly raised for missing task_selector")

def test_tag_selector_edge_cases():
    """Test edge cases for tag selector."""
    
    researcher = Agent(
        role="Researcher",
        goal="Research topics",
        backstory="Expert researcher"
    )
    
    tagged_task = Task(
        description="Tagged task",
        expected_output="Output",
        agent=researcher,
        tags=["test"]
    )
    
    untagged_task = Task(
        description="Untagged task", 
        expected_output="Output",
        agent=researcher
    )
    
    selector = Crew.create_tag_selector()
    
    assert selector({}, tagged_task) == True
    assert selector({}, untagged_task) == True
    
    assert selector({"action": "anything"}, untagged_task) == True
    
    print("Edge case tests passed!")

if __name__ == "__main__":
    test_selective_execution_basic()
    test_selective_process_validation()
    test_tag_selector_edge_cases()
    print("All tests completed successfully!")
