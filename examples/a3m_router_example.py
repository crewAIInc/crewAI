"""
Example: Using A3M Router with CrewAI for cost-optimized multi-agent systems.

A3M Router provides intelligent routing for CrewAI agents with:
- 70-95% cost savings vs GPT-4o
- Automatic model selection based on task complexity
- Built-in fallback handling
- Support for 47+ LLM providers

Installation:
    pip install adaptive-memory-multi-model-router crewai

Usage:
    from crewai import Agent, Task, Crew
    from crewai.llms import A3MCompletion

    # Create A3M-powered agent
    researcher = Agent(
        role="Research Analyst",
        goal="Research and summarize market trends",
        backstory="Expert market analyst with years of experience",
        llm=A3MCompletion(model="auto"),
    )

    # Create crew with A3M routing
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
    )

    result = crew.kickoff()
"""

from crewai import Agent, Crew, Task
from crewai.llms import A3MCompletion


def example_basic():
    """Basic A3M Router usage with CrewAI."""
    # Create A3M router with automatic model selection
    llm = A3MCompletion(model="auto")

    # Create research agent
    researcher = Agent(
        role="Research Analyst",
        goal="Research the latest AI trends and provide insights",
        backstory="""You are an expert research analyst specializing in 
        artificial intelligence and machine learning. You have deep 
        knowledge of the AI industry, emerging technologies, and market trends.""",
        llm=llm,
        verbose=True,
    )

    # Create task
    research_task = Task(
        description="Research the current state of LLM routing technologies and summarize key findings",
        agent=researcher,
        expected_output="A comprehensive summary of LLM routing technologies",
    )

    # Create crew
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True,
    )

    # Run
    result = crew.kickoff()
    print(f"Result: {result}")


def example_multi_agent():
    """Multi-agent system with A3M Router for different task types."""
    # Different agents can use different routing strategies
    researcher_llm = A3MCompletion(model="auto", temperature=0.7)
    writer_llm = A3MCompletion(model="auto", temperature=0.9)
    critic_llm = A3MCompletion(model="auto", temperature=0.3)

    researcher = Agent(
        role="Research Analyst",
        goal="Gather and analyze information",
        backstory="Expert researcher with analytical mindset",
        llm=researcher_llm,
    )

    writer = Agent(
        role="Content Writer",
        goal="Create engaging content from research",
        backstory="Skilled writer with creative flair",
        llm=writer_llm,
    )

    critic = Agent(
        role="Quality Assurance",
        goal="Ensure content quality and accuracy",
        backstory="Detail-oriented editor with high standards",
        llm=critic_llm,
    )

    # Define tasks
    research_task = Task(description="Research AI trends", agent=researcher)
    writing_task = Task(description="Write article", agent=writer)
    review_task = Task(description="Review content", agent=critic)

    # Create crew
    crew = Crew(
        agents=[researcher, writer, critic],
        tasks=[research_task, writing_task, review_task],
        process="hierarchical",  # Manager coordinates others
    )

    result = crew.kickoff()
    print(f"Multi-agent result: {result}")


def example_cost_tracking():
    """Example demonstrating A3M cost optimization."""
    llm = A3MCompletion(model="auto")

    agent = Agent(
        role="Data Analyst",
        goal="Analyze datasets and provide insights",
        backstory="Expert data scientist",
        llm=llm,
    )

    task = Task(
        description="Analyze this sales data and provide recommendations",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    print(f"Result: {result}")


if __name__ == "__main__":
    print("Running A3M Router + CrewAI examples...")
    example_basic()
