#!/usr/bin/env python3
"""
Test script for issue #3559 - TraceBatchManager authentication handling
"""

import sys
sys.path.insert(0, '/home/ubuntu/repos/crewAI/src')

def test_tracing_auth_issue():
    """Test that tracing authentication issue is fixed"""
    
    try:
        from crewai.events.listeners.tracing.trace_batch_manager import TraceBatchManager
        from crewai.cli.authentication.token import AuthError
        from unittest.mock import patch
        
        print("Test 1: TraceBatchManager creation without authentication")
        
        with patch(
            "crewai.events.listeners.tracing.trace_batch_manager.get_auth_token",
            side_effect=AuthError("No token found, make sure you are logged in")
        ):
            batch_manager = TraceBatchManager()
            print(f"✓ TraceBatchManager created successfully with empty API key: '{batch_manager.plus_api.api_key}'")
            
            batch = batch_manager.initialize_batch({"user_id": "test"}, {"crew_name": "test"})
            if batch is not None:
                print(f"✓ Batch initialized successfully: {batch.batch_id}")
            else:
                print("✗ Batch initialization returned None")
                return False
                
    except Exception as e:
        print(f"✗ TraceBatchManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from crewai import Agent, Crew, Task, LLM
        
        print("\nTest 2: Crew creation without authentication")
        
        with patch(
            "crewai.events.listeners.tracing.trace_batch_manager.get_auth_token",
            side_effect=AuthError("No token found, make sure you are logged in")
        ):
            agent = Agent(
                role="Test Agent",
                goal="Complete a simple task",
                backstory="A test agent for reproducing the bug",
                llm=LLM(model="gpt-4o-mini", api_key="fake-key")
            )
            
            task = Task(
                description="Say hello world",
                expected_output="A greeting message",
                agent=agent
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=False
            )
            
            print("✓ Crew created successfully without authentication errors")
            
    except Exception as e:
        print(f"✗ Crew creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("Testing TraceBatchManager authentication handling...")
    success = test_tracing_auth_issue()
    if not success:
        print("\nFAILED: Issue #3559 still exists")
        exit(1)
    else:
        print("\nPASSED: Issue #3559 appears to be fixed")
