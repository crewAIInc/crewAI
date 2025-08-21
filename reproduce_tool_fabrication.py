#!/usr/bin/env python3
"""
Minimal reproduction of CrewAI tool fabrication issue #3154

This script demonstrates the problem where agents generate fake tool outputs
instead of actually executing tools.
"""

import os
import tempfile
from pathlib import Path

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class FileWriteInput(BaseModel):
    filename: str = Field(description="Name of the file to write")
    content: str = Field(description="Content to write to the file")


class FileWriteTool(BaseTool):
    name: str = "File Writer"
    description: str = "Writes content to a file and returns confirmation"
    args_schema: type[BaseModel] = FileWriteInput

    def _run(self, filename: str, content: str) -> str:
        """Actually write to a file - this should leave filesystem evidence"""
        temp_dir = Path(tempfile.gettempdir()) / "crewai_test"
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / filename
        
        print(f"🔧 REAL TOOL EXECUTION: Writing to {file_path}")
        
        with open(file_path, "w") as f:
            f.write(content)
            
        # Return evidence of real execution
        file_size = file_path.stat().st_size
        return f"File {filename} written successfully. Size: {file_size} bytes. Path: {file_path}"


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query to find information")


class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Searches the web for information"
    args_schema: type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        """Simulate a web search - should NOT actually search"""
        print(f"🔧 REAL TOOL EXECUTION: Searching for '{query}'")
        
        # In real scenario, this would make HTTP requests
        # For testing, we'll just return a distinctive marker
        return f"REAL_SEARCH_RESULT: Found information about '{query}' from actual web search"


def test_tool_execution_authenticity():
    """Test if tools are actually executed or fabricated"""
    
    # Create agent with tools
    agent = Agent(
        role="File Manager",
        goal="Write files and search the web based on user requests",
        backstory="You are an expert at managing files and finding information.",
        tools=[FileWriteTool(), WebSearchTool()],
        verbose=True,
        allow_delegation=False
    )
    
    # Task that requires tool usage
    task = Task(
        description="""
        1. Write a file called 'test_output.txt' containing 'Hello from CrewAI tool test!'
        2. Search the web for 'CrewAI tool execution verification'
        
        Be specific about what files you created and what search results you found.
        """,
        expected_output="Confirmation of file creation and web search results",
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    print("🔍 STARTING TOOL EXECUTION TEST")
    print("=" * 50)
    
    # Execute the task
    result = crew.kickoff()
    
    print("\n" * 2)
    print("🔍 EXECUTION RESULT:")
    print("=" * 50)
    print(result)
    
    # Check for evidence of real execution
    print("\n" * 2)
    print("🔍 EVIDENCE ANALYSIS:")
    print("=" * 50)
    
    # Check if file was actually created
    temp_dir = Path(tempfile.gettempdir()) / "crewai_test"
    test_file = temp_dir / "test_output.txt"
    
    if test_file.exists():
        print("✅ REAL EXECUTION: File was actually created")
        print(f"   File path: {test_file}")
        print(f"   File content: {test_file.read_text()}")
        print(f"   File size: {test_file.stat().st_size} bytes")
    else:
        print("❌ FABRICATED EXECUTION: No file was actually created")
        print(f"   Expected file path: {test_file}")
        print("   This indicates the agent fabricated the file writing result")
    
    # Analyze the result text for fabrication indicators
    result_str = str(result)
    
    print("\n🔍 RESULT TEXT ANALYSIS:")
    
    if "REAL_SEARCH_RESULT:" in result_str:
        print("✅ REAL EXECUTION: Found actual web search marker")
    else:
        print("❌ FABRICATED EXECUTION: No real web search marker found")
        print("   Agent likely fabricated search results")
    
    if "🔧 REAL TOOL EXECUTION:" in result_str:
        print("✅ TOOL PRINT DETECTED: Tool execution print statements found")
    else:
        print("❌ NO TOOL PRINTS: Tool _run methods may not have executed")
    
    # Check for common fabrication patterns
    fabrication_indicators = [
        "successfully created",
        "file has been written",
        "I have created",
        "search results show",
        "I found information",
    ]
    
    fabrication_count = sum(1 for indicator in fabrication_indicators if indicator.lower() in result_str.lower())
    
    if fabrication_count > 2:
        print(f"⚠️  HIGH FABRICATION LIKELIHOOD: {fabrication_count} fabrication indicators found")
        print("   Result contains multiple 'success' claims without evidence")
    
    return {
        "file_created": test_file.exists(),
        "real_search_marker": "REAL_SEARCH_RESULT:" in result_str,
        "tool_prints_found": "🔧 REAL TOOL EXECUTION:" in result_str,
        "fabrication_indicators": fabrication_count,
        "result": result_str
    }


if __name__ == "__main__":
    # Run the test
    test_results = test_tool_execution_authenticity()
    
    print("\n" * 2)
    print("🔍 FINAL ANALYSIS:")
    print("=" * 50)
    
    if test_results["file_created"] and test_results["real_search_marker"]:
        print("✅ TOOLS EXECUTED AUTHENTICALLY")
    else:
        print("❌ TOOL EXECUTION FABRICATED")
        print("\nEvidence:")
        print(f"  - File actually created: {test_results['file_created']}")
        print(f"  - Real search executed: {test_results['real_search_marker']}")
        print(f"  - Tool prints found: {test_results['tool_prints_found']}")
        print(f"  - Fabrication indicators: {test_results['fabrication_indicators']}")
        
        print("\n🎯 ISSUE #3154 CONFIRMED: CrewAI agents fabricate tool results")