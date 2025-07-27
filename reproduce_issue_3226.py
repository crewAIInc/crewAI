#!/usr/bin/env python3
"""
Reproduction script for issue #3226: Cannot Register Custom Tools with Agents in CrewAI 0.150.0
This script tests all the failing patterns mentioned in the issue.
"""

import sys
import traceback

def test_function_tool():
    """Test 1: Function Tool with @tool decorator"""
    print("=== Test 1: Function Tool with @tool decorator ===")
    try:
        from crewai.tools import tool
        from crewai import Agent
        
        @tool
        def fetch_logs(query: str) -> str:
            """Fetch logs from New Relic based on query"""
            return f"Logs for query: {query}"
        
        agent = Agent(
            role='CrashFetcher',
            goal='Extract logs',
            backstory='An agent that fetches logs',
            tools=[fetch_logs],
            allow_delegation=False
        )
        assert len(agent.tools) == 1, f"Expected 1 tool, got {len(agent.tools)}"
        print("✅ Function tool with @tool decorator: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Function tool with @tool decorator: FAILED - {e}")
        traceback.print_exc()
        return False

def test_dict_tool():
    """Test 2: Dict-based tool definition"""
    print("\n=== Test 2: Dict-based tool definition ===")
    try:
        from crewai import Agent
        
        def fetch_logs_func(query: str) -> str:
            return f"Logs for query: {query}"
        
        fetch_logs_dict = {
            'name': 'fetch_logs',
            'description': 'Fetch logs from New Relic',
            'func': fetch_logs_func
        }
        
        agent = Agent(
            role='CrashFetcher',
            goal='Extract logs',
            backstory='An agent that fetches logs',
            tools=[fetch_logs_dict],
            allow_delegation=False
        )
        assert len(agent.tools) == 1, f"Expected 1 tool, got {len(agent.tools)}"
        print("✅ Dict-based tool: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Dict-based tool: FAILED - {e}")
        traceback.print_exc()
        return False

def test_basetool_class():
    """Test 3: BaseTool class inheritance"""
    print("\n=== Test 3: BaseTool class inheritance ===")
    try:
        from crewai.tools import BaseTool
        from crewai import Agent
        
        class FetchLogsTool(BaseTool):
            name: str = "fetch_logs"
            description: str = "Fetch logs from New Relic based on query"
            
            def _run(self, query: str) -> str:
                return f"Logs for query: {query}"
        
        agent = Agent(
            role='CrashFetcher',
            goal='Extract logs',
            backstory='An agent that fetches logs',
            tools=[FetchLogsTool()],
            allow_delegation=False
        )
        assert len(agent.tools) == 1, f"Expected 1 tool, got {len(agent.tools)}"
        print("✅ BaseTool class inheritance: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ BaseTool class inheritance: FAILED - {e}")
        traceback.print_exc()
        return False

def test_direct_function():
    """Test 4: Direct function assignment"""
    print("\n=== Test 4: Direct function assignment ===")
    try:
        from crewai import Agent
        
        def fetch_logs(query: str) -> str:
            """Fetch logs from New Relic based on query"""
            return f"Logs for query: {query}"
        
        agent = Agent(
            role='CrashFetcher',
            goal='Extract logs',
            backstory='An agent that fetches logs',
            tools=[fetch_logs],
            allow_delegation=False
        )
        assert len(agent.tools) == 1, f"Expected 1 tool, got {len(agent.tools)}"
        print("✅ Direct function assignment: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Direct function assignment: FAILED - {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests and report results"""
    print("Testing custom tool registration patterns from issue #3226\n")
    
    results = []
    results.append(test_function_tool())
    results.append(test_dict_tool())
    results.append(test_basetool_class())
    results.append(test_direct_function())
    
    print("\n=== SUMMARY ===")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All custom tool patterns are working!")
        return 0
    else:
        print("💥 Some custom tool patterns are still broken")
        return 1

if __name__ == "__main__":
    sys.exit(main())
