#!/usr/bin/env python3
"""Test script to verify verbose output shows task names instead of IDs."""

from crewai import Agent, Task, Crew

def test_verbose_output():
    """Test that verbose output shows task names instead of UUIDs."""
    print("Testing verbose output with task names...")
    
    agent = Agent(
        role="Research Analyst",
        goal="Analyze data and provide insights",
        backstory="You are an experienced data analyst.",
        verbose=True
    )

    task = Task(
        name="Market Research Analysis",
        description="Research current market trends in AI technology",
        expected_output="A comprehensive report on AI market trends",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )

    print("Task name:", task.name)
    print("Task ID:", task.id)
    print("\nRunning crew with verbose=True...")
    print("Expected: Should show task name 'Market Research Analysis' instead of UUID")
    
    try:
        result = crew.kickoff()
        print("\nCrew execution completed successfully!")
        return True
    except Exception as e:
        print(f"Error during execution: {e}")
        return False

if __name__ == "__main__":
    test_verbose_output()
