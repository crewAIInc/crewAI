"""
Reproduction script for issue #3152 - mem0 external memory format error
Based on the code provided in the GitHub issue
"""

import os
from crewai import Agent, Task, Crew
from crewai.memory.external.external_memory import ExternalMemory

def test_mem0_external_memory():
    """Test that reproduces the mem0 external memory format error"""
    
    embedder_config = {
        "provider": "mem0",
        "config": {
            "user_id": "test_user_123",
        }
    }
    
    external_memory = ExternalMemory(embedder_config=embedder_config)
    
    agent = Agent(
        role="Test Agent",
        goal="Test external memory functionality",
        backstory="A test agent for reproducing the mem0 issue",
        verbose=True
    )
    
    task = Task(
        description="Test task for external memory",
        expected_output="Test output",
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        external_memory=external_memory,
        verbose=True
    )
    
    print("Testing mem0 external memory integration...")
    
    try:
        result = crew.kickoff()
        print("SUCCESS: External memory integration worked!")
        print(f"Result: {result}")
    except Exception as e:
        print(f"ERROR: {e}")
        if "Expected a list of items but got type" in str(e):
            print("CONFIRMED: This is the mem0 format error from issue #3152")
        raise

if __name__ == "__main__":
    test_mem0_external_memory()
