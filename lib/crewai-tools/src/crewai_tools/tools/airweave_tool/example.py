#!/usr/bin/env python3
"""
Example usage of the AirweaveSearchTool with CrewAI.

This script demonstrates how to use the Airweave tool to search
organizational data and generate insights using AI agents.

Prerequisites:
    1. Install dependencies: pip install 'crewai[tools]' airweave-sdk python-dotenv
    2. Set environment variables in .env file:
       - AIRWEAVE_API_KEY="your-api-key"
       - AIRWEAVE_COLLECTION_ID="your-collection-id"
       - OPENAI_API_KEY="your-openai-key"
    3. Have an Airweave collection with data synced
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import AirweaveSearchTool

# Load environment variables from .env file
# Try to find .env in parent directories
current_dir = Path(__file__).resolve().parent
for parent in [current_dir] + list(current_dir.parents):
    env_file = parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"📝 Loaded .env from: {parent}")
        break
else:
    load_dotenv()

# Get collection ID from environment
DEFAULT_COLLECTION_ID = os.getenv("AIRWEAVE_COLLECTION_ID", "sales-data-2024")
print(f"🗂️  Using collection: {DEFAULT_COLLECTION_ID}")
print()


def example_basic_search():
    """Basic example: Simple search without answer generation."""
    
    print("="*60)
    print("Example 1: Basic Search")
    print("="*60)
    
    # Initialize the tool
    airweave_search = AirweaveSearchTool()
    
    # Create a research agent
    researcher = Agent(
        role='Research Analyst',
        goal='Find relevant information from company data',
        backstory='Expert at finding insights in organizational data',
        tools=[airweave_search],
        verbose=True
    )
    
    # Create a search task - using env variable for collection
    search_task = Task(
        description=f"""
        Search the '{DEFAULT_COLLECTION_ID}' collection for information. 
        Use these parameters:
        - query: "What information is available?"
        - collection_id: "{DEFAULT_COLLECTION_ID}"
        - limit: 10
        
        Provide a summary of key findings.
        """,
        agent=researcher,
        expected_output='Summary of available information'
    )
    
    # Create and run the crew
    crew = Crew(
        agents=[researcher],
        tasks=[search_task],
        verbose=True
    )
    
    result = crew.kickoff()
    print("\n📊 Result:")
    print(result)


def example_with_answer_generation():
    """Advanced example: Search with AI-powered answer generation."""
    
    print("\n" + "="*60)
    print("Example 2: Search with Answer Generation")
    print("="*60)
    
    # Initialize with custom settings
    airweave_search = AirweaveSearchTool(
        max_content_length_per_result=2000,  # Longer content per result
        timeout=90  # Longer timeout
    )
    
    # Create a specialized analyst
    data_analyst = Agent(
        role='Senior Data Analyst',
        goal='Extract actionable insights from company data',
        backstory='Expert analyst with deep knowledge of business metrics and trends',
        tools=[airweave_search],
        verbose=True
    )
    
    # Advanced search task - using env variable
    analysis_task = Task(
        description=f"""
        Using the AirweaveSearchTool, research available data by:
        
        1. Searching the '{DEFAULT_COLLECTION_ID}' collection
        2. Using query: "Tell me about the most important information"
        3. Enabling these features:
           - generate_answer: True (get AI-generated insights)
           - expand_query: True (improve recall)
           - rerank: True (better relevance)
           - temporal_relevance: 0.3 (slightly favor recent data)
           - limit: 15
        
        Analyze the results and provide:
        - Key findings
        - Important patterns
        - Notable insights
        - Strategic recommendations
        """,
        agent=data_analyst,
        expected_output='Comprehensive analysis with AI-generated insights and recommendations'
    )
    
    crew = Crew(
        agents=[data_analyst],
        tasks=[analysis_task],
        verbose=True
    )
    
    result = crew.kickoff()
    print("\n📊 Result:")
    print(result)


def example_comprehensive_research():
    """Example: Comprehensive research of available data."""
    
    print("\n" + "="*60)
    print("Example 3: Comprehensive Research")
    print("="*60)
    
    airweave_search = AirweaveSearchTool()
    
    # Create researcher
    researcher = Agent(
        role='Comprehensive Researcher',
        goal='Gather and synthesize information from data sources',
        backstory='Skilled at connecting insights across diverse datasets',
        tools=[airweave_search],
        verbose=True
    )
    
    # Research task - using env variable
    research_task = Task(
        description=f"""
        Research the available data by searching the '{DEFAULT_COLLECTION_ID}' collection:
        
        Search for:
           - Available information and data
           - Key documents and content
           - Recent updates and changes
        
        Use these parameters:
        - expand_query: True
        - rerank: True
        - generate_answer: True
        - limit: 10
        
        Provide a comprehensive report showing:
        - What data is available
        - Key insights found
        - Summary of contents
        - Recommendations based on findings
        """,
        agent=researcher,
        expected_output='Comprehensive report of available data and insights'
    )
    
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True
    )
    
    result = crew.kickoff()
    print("\n📊 Result:")
    print(result)


def example_with_memory():
    """Example: Using Airweave search with CrewAI memory."""
    
    print("\n" + "="*60)
    print("Example 4: Search with Memory")
    print("="*60)
    
    airweave_search = AirweaveSearchTool()
    
    # Agent with memory enabled
    knowledge_agent = Agent(
        role='Knowledge Manager',
        goal='Build and maintain comprehensive organizational knowledge',
        backstory='Expert at connecting information',
        tools=[airweave_search],
        memory=True,  # Enable CrewAI memory
        verbose=True
    )
    
    # First task - using env variable
    task1 = Task(
        description=f"""
        Search the '{DEFAULT_COLLECTION_ID}' collection for general information.
        Remember key details for future queries.
        Query: "What data is available?"
        """,
        agent=knowledge_agent,
        expected_output='Summary of available information'
    )
    
    # Second task - using env variable
    task2 = Task(
        description=f"""
        Now search the '{DEFAULT_COLLECTION_ID}' collection again for more specific
        details related to what we found in the first search. Connect the insights.
        Query: "Tell me more about the topics we just discovered"
        """,
        agent=knowledge_agent,
        expected_output='Analysis connecting information from both searches'
    )
    
    crew = Crew(
        agents=[knowledge_agent],
        tasks=[task1, task2],
        verbose=True
    )
    
    result = crew.kickoff()
    print("\n📊 Result:")
    print(result)


def example_error_handling():
    """Example: Demonstrating graceful error handling."""
    
    print("\n" + "="*60)
    print("Example 5: Error Handling")
    print("="*60)
    
    airweave_search = AirweaveSearchTool()
    
    agent = Agent(
        role='Robust Researcher',
        goal='Handle search operations gracefully',
        backstory='Expert at error recovery and troubleshooting',
        tools=[airweave_search],
        verbose=True
    )
    
    # Using env variable
    task = Task(
        description=f"""
        Search the '{DEFAULT_COLLECTION_ID}' collection for available data.
        Handle any errors gracefully and provide helpful feedback.
        Query: "Show me what you have"
        """,
        agent=agent,
        expected_output='Search results or error handling report'
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        print("\n📊 Result:")
        print(result)
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("This demonstrates how the tool handles errors gracefully.")


if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          AirweaveSearchTool Examples for CrewAI             ║
╚══════════════════════════════════════════════════════════════╝

These examples demonstrate various ways to use the Airweave tool
with CrewAI agents for searching organizational data.

📝 Collection ID from environment: {DEFAULT_COLLECTION_ID}

Prerequisites:
  1. Set AIRWEAVE_API_KEY environment variable (or in .env file)
  2. Set AIRWEAVE_COLLECTION_ID environment variable (or in .env file)
  3. Install: pip install 'crewai[tools]' airweave-sdk python-dotenv
  4. Have Airweave collections with synced data
    """)
    
    # Choose which example to run
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_basic_search()
        elif example_num == "2":
            example_with_answer_generation()
        elif example_num == "3":
            example_comprehensive_research()
        elif example_num == "4":
            example_with_memory()
        elif example_num == "5":
            example_error_handling()
        else:
            print(f"Unknown example: {example_num}")
            print("Usage: python example.py [1-5]")
    else:
        print("\nUsage:")
        print("  python example.py 1  # Basic search")
        print("  python example.py 2  # With answer generation")
        print("  python example.py 3  # Comprehensive research")
        print("  python example.py 4  # With memory")
        print("  python example.py 5  # Error handling")
        print("\nOr uncomment one of the examples below to run it directly.")
        
        # Uncomment the example you want to run:
        # example_basic_search()
        # example_with_answer_generation()
        # example_comprehensive_research()
        # example_with_memory()
        # example_error_handling()
