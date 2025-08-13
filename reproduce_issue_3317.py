#!/usr/bin/env python3
"""
Reproduction script for issue #3317
CrewAI injects stop sequences causing Bedrock GPT-OSS 400 errors
"""

import os
from crewai import Agent, Task, Crew
from crewai.llm import LLM

def test_bedrock_stop_sequence_issue():
    """
    Reproduce the issue where CrewAI automatically injects stop sequences
    that cause Bedrock models to fail with 'stopSequences not supported' error.
    """
    print("Testing Bedrock stop sequence issue...")
    
    llm = LLM(
        model="bedrock/converse/openai.gpt-oss-20b-1:0",
        litellm_params={
            "aws_region_name": "us-east-1",
            "drop_params": True
        }
    )
    
    print(f"Model supports stop words: {llm.supports_stop_words()}")
    
    agent = Agent(
        role="Test Agent",
        goal="Test the stop sequence issue",
        backstory="A test agent to reproduce the issue",
        llm=llm,
        verbose=True
    )
    
    task = Task(
        description="Say hello",
        expected_output="A simple greeting",
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        print("SUCCESS: No error occurred")
        print(f"Result: {result}")
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR: {error_msg}")
        
        if "stopSequences not supported" in error_msg or "Unsupported parameter" in error_msg and "'stop'" in error_msg:
            print("CONFIRMED: This is the expected stop sequence error from issue #3317")
            return False
        else:
            print("UNEXPECTED: This is a different error")
            raise e

if __name__ == "__main__":
    test_bedrock_stop_sequence_issue()
