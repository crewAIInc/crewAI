"""
Example demonstrating Google Batch Mode support in CrewAI.

This example shows how to use batch mode with Gemini models to reduce costs
by up to 50% for non-urgent LLM calls.
"""

import os
from crewai import Agent, Task, Crew
from crewai.llm import LLM

os.environ["GOOGLE_API_KEY"] = "your-google-api-key-here"

def main():
    batch_llm = LLM(
        model="gemini/gemini-1.5-pro",
        batch_mode=True,
        batch_size=5,  # Process 5 requests at once
        batch_timeout=300,  # Wait up to 5 minutes for batch completion
        temperature=0.7
    )
    
    research_agent = Agent(
        role="Research Analyst",
        goal="Analyze market trends and provide insights",
        backstory="You are an expert market analyst with years of experience.",
        llm=batch_llm,
        verbose=True
    )
    
    tasks = []
    topics = [
        "artificial intelligence market trends",
        "renewable energy investment opportunities", 
        "cryptocurrency regulatory landscape",
        "e-commerce growth projections",
        "healthcare technology innovations"
    ]
    
    for topic in topics:
        task = Task(
            description=f"Research and analyze {topic}. Provide a brief summary of key trends and insights.",
            agent=research_agent,
            expected_output="A concise analysis with key findings and trends"
        )
        tasks.append(task)
    
    crew = Crew(
        agents=[research_agent],
        tasks=tasks,
        verbose=True
    )
    
    print("Starting batch processing...")
    print("Note: Batch requests will be queued until batch_size is reached")
    
    result = crew.kickoff()
    
    print("Batch processing completed!")
    print("Results:", result)

if __name__ == "__main__":
    main()
