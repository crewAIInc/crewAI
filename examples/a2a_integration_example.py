"""Example: CrewAI A2A Integration

This example demonstrates how to expose a CrewAI crew as an A2A (Agent-to-Agent)
protocol server for remote interoperability.

Requirements:
    pip install crewai[a2a]
"""

from crewai import Agent, Crew, Task
from crewai.a2a import CrewAgentExecutor, start_a2a_server


def main():
    """Create and start an A2A server with a CrewAI crew."""
    
    researcher = Agent(
        role="Research Analyst",
        goal="Provide comprehensive research and analysis on any topic",
        backstory=(
            "You are an experienced research analyst with expertise in "
            "gathering, analyzing, and synthesizing information from various sources."
        ),
        verbose=True
    )
    
    research_task = Task(
        description=(
            "Research and analyze the topic: {query}\n"
            "Provide a comprehensive overview including:\n"
            "- Key concepts and definitions\n"
            "- Current trends and developments\n"
            "- Important considerations\n"
            "- Relevant examples or case studies"
        ),
        agent=researcher,
        expected_output="A detailed research report with analysis and insights"
    )
    
    research_crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True
    )
    
    executor = CrewAgentExecutor(
        crew=research_crew,
        supported_content_types=['text', 'text/plain', 'application/json']
    )
    
    print("Starting A2A server with CrewAI research crew...")
    print("Server will be available at http://localhost:10001")
    print("Use the A2A CLI or SDK to interact with the crew remotely")
    
    start_a2a_server(
        executor,
        host="0.0.0.0",
        port=10001,
        transport="starlette"
    )


if __name__ == "__main__":
    main()
