"""
Example demonstrating prompt caching with CrewAI for cost optimization.

This example shows how to use prompt caching with kickoff_for_each() and
kickoff_async() to reduce costs when processing multiple similar inputs.
"""

from crewai import Agent, Crew, Task, LLM
import asyncio


def create_crew_with_caching():
    """Create a crew with prompt caching enabled."""
    
    llm = LLM(
        model="anthropic/claude-3-5-sonnet-20240620",
        enable_prompt_caching=True,
        temperature=0.1
    )
    
    analyst = Agent(
        role="Data Analyst",
        goal="Analyze data and provide insights",
        backstory="""You are an experienced data analyst with expertise in 
        statistical analysis, data visualization, and business intelligence. 
        You have worked with various industries including finance, healthcare, 
        and technology. Your approach is methodical and you always provide 
        actionable insights based on data patterns.""",
        llm=llm
    )
    
    analysis_task = Task(
        description="""Analyze the following dataset: {dataset}
        
        Please provide:
        1. Summary statistics
        2. Key patterns and trends
        3. Actionable recommendations
        4. Potential risks or concerns
        
        Be thorough in your analysis and provide specific examples.""",
        expected_output="A comprehensive analysis report with statistics, trends, and recommendations",
        agent=analyst
    )
    
    return Crew(agents=[analyst], tasks=[analysis_task])


def example_kickoff_for_each():
    """Example using kickoff_for_each with prompt caching."""
    print("Running kickoff_for_each example with prompt caching...")
    
    crew = create_crew_with_caching()
    
    datasets = [
        {"dataset": "Q1 2024 sales data showing 15% growth in mobile segment"},
        {"dataset": "Q2 2024 customer satisfaction scores with 4.2/5 average rating"},
        {"dataset": "Q3 2024 website traffic data with 25% increase in organic search"},
        {"dataset": "Q4 2024 employee engagement survey with 78% satisfaction rate"}
    ]
    
    results = crew.kickoff_for_each(datasets)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Analysis {i} ---")
        print(result.raw)
        
    if crew.usage_metrics:
        print(f"\nTotal usage metrics:")
        print(f"Total tokens: {crew.usage_metrics.total_tokens}")
        print(f"Prompt tokens: {crew.usage_metrics.prompt_tokens}")
        print(f"Completion tokens: {crew.usage_metrics.completion_tokens}")


async def example_kickoff_for_each_async():
    """Example using kickoff_for_each_async with prompt caching."""
    print("Running kickoff_for_each_async example with prompt caching...")
    
    crew = create_crew_with_caching()
    
    datasets = [
        {"dataset": "Marketing campaign A: 12% CTR, 3.5% conversion rate"},
        {"dataset": "Marketing campaign B: 8% CTR, 4.1% conversion rate"},
        {"dataset": "Marketing campaign C: 15% CTR, 2.8% conversion rate"}
    ]
    
    results = await crew.kickoff_for_each_async(datasets)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Async Analysis {i} ---")
        print(result.raw)
        
    if crew.usage_metrics:
        print(f"\nTotal async usage metrics:")
        print(f"Total tokens: {crew.usage_metrics.total_tokens}")


def example_bedrock_caching():
    """Example using AWS Bedrock with prompt caching."""
    print("Running Bedrock example with prompt caching...")
    
    llm = LLM(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        enable_prompt_caching=True
    )
    
    agent = Agent(
        role="Legal Analyst",
        goal="Review legal documents and identify key clauses",
        backstory="Expert legal analyst with 10+ years experience in contract review",
        llm=llm
    )
    
    task = Task(
        description="Review this contract section: {contract_section}",
        expected_output="Summary of key legal points and potential issues",
        agent=agent
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    
    contract_sections = [
        {"contract_section": "Section 1: Payment terms and conditions"},
        {"contract_section": "Section 2: Intellectual property rights"},
        {"contract_section": "Section 3: Termination clauses"}
    ]
    
    results = crew.kickoff_for_each(contract_sections)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Legal Review {i} ---")
        print(result.raw)


def example_openai_caching():
    """Example using OpenAI with prompt caching."""
    print("Running OpenAI example with prompt caching...")
    
    llm = LLM(
        model="gpt-4o",
        enable_prompt_caching=True
    )
    
    agent = Agent(
        role="Content Writer",
        goal="Create engaging content for different audiences",
        backstory="Professional content writer with expertise in various writing styles and formats",
        llm=llm
    )
    
    task = Task(
        description="Write a {content_type} about: {topic}",
        expected_output="Well-structured and engaging content piece",
        agent=agent
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    
    content_requests = [
        {"content_type": "blog post", "topic": "benefits of renewable energy"},
        {"content_type": "social media post", "topic": "importance of cybersecurity"},
        {"content_type": "newsletter", "topic": "latest AI developments"}
    ]
    
    results = crew.kickoff_for_each(content_requests)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Content Piece {i} ---")
        print(result.raw)


if __name__ == "__main__":
    print("=== CrewAI Prompt Caching Examples ===\n")
    
    example_kickoff_for_each()
    
    print("\n" + "="*50 + "\n")
    
    asyncio.run(example_kickoff_for_each_async())
    
    print("\n" + "="*50 + "\n")
    
    example_bedrock_caching()
    
    print("\n" + "="*50 + "\n")
    
    example_openai_caching()
