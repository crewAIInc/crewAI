#!/usr/bin/env python3
"""
Reproduction script for issue #3165: LLM Failed with Custom OpenAI-Compatible Endpoint

This script reproduces the bug where CrewAI shows generic "LLM Failed" errors
instead of propagating specific error details from custom endpoints.
"""

import os
import sys
from crewai import Agent, Task, Crew
from crewai.llm import LLM

def test_custom_endpoint_error_handling():
    """Test error handling with a custom OpenAI-compatible endpoint."""
    
    print("Testing custom endpoint error handling...")
    
    custom_llm = LLM(
        model="gpt-3.5-turbo",
        base_url="https://non-existent-endpoint.example.com/v1",
        api_key="fake-api-key-for-testing"
    )
    
    agent = Agent(
        role="Test Agent",
        goal="Test custom endpoint error handling",
        backstory="A test agent for reproducing issue #3165",
        llm=custom_llm,
        verbose=True
    )
    
    task = Task(
        description="Say hello world",
        expected_output="A simple greeting",
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    try:
        print("\nAttempting to run crew with custom endpoint...")
        result = crew.kickoff()
        print(f"Unexpected success: {result}")
    except Exception as e:
        print(f"\nCaught exception: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        
        if "LLM Failed" in str(e) and "connection" not in str(e).lower():
            print("\n❌ BUG CONFIRMED: Generic 'LLM Failed' error without specific details")
            print("Expected: Specific connection/authentication error details")
            return False
        else:
            print("\n✅ Good: Specific error details preserved")
            return True

def test_direct_llm_call():
    """Test direct LLM call with custom endpoint."""
    
    print("\n" + "="*60)
    print("Testing direct LLM call with custom endpoint...")
    
    custom_llm = LLM(
        model="gpt-3.5-turbo", 
        base_url="https://non-existent-endpoint.example.com/v1",
        api_key="fake-api-key-for-testing"
    )
    
    try:
        print("Attempting direct LLM call...")
        response = custom_llm.call("Hello world")
        print(f"Unexpected success: {response}")
    except Exception as e:
        print(f"\nCaught exception: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["connection", "resolve", "network", "timeout", "unreachable"]):
            print("\n✅ Good: Specific connection error details preserved")
            return True
        else:
            print("\n❌ BUG CONFIRMED: Generic error without connection details")
            print("Expected: Specific connection error details")
            return False

if __name__ == "__main__":
    print("Reproducing issue #3165: LLM Failed with Custom OpenAI-Compatible Endpoint")
    print("="*80)
    
    crew_test_passed = test_custom_endpoint_error_handling()
    direct_test_passed = test_direct_llm_call()
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print(f"Crew-level test: {'PASSED' if crew_test_passed else 'FAILED (bug confirmed)'}")
    print(f"Direct LLM test: {'PASSED' if direct_test_passed else 'FAILED (bug confirmed)'}")
    
    if not crew_test_passed or not direct_test_passed:
        print("\n❌ Issue #3165 reproduced successfully")
        print("CrewAI is showing generic errors instead of specific endpoint error details")
        sys.exit(1)
    else:
        print("\n✅ Issue #3165 appears to be fixed")
        sys.exit(0)
