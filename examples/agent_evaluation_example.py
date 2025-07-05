"""Agent evaluation example with crewAI.

This example demonstrates how to use the agent evaluation framework to
assess the performance of agents across various metrics.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileReadTool
from crewai.evaluation import (
    AgentEvaluator,
    GoalAlignmentEvaluator,
    KnowledgeRetrievalEvaluator,
    SemanticQualityEvaluator,
    ToolUsageEvaluator,
    create_evaluation_callbacks
)

# Load environment variables
load_dotenv()

# Initialize tools
search_tool = SerperDevTool()
file_tool = FileReadTool()

def run_crew_evaluation_example():
    """Run an example crew with evaluation for each agent."""

    # Create agents
    researcher = Agent(
        role="Market Researcher",
        goal="Gather comprehensive market data on electric vehicles",
        backstory="""You are an expert at finding and compiling market data.
        You know how to search for the most relevant information.""",
        verbose=True,
        tools=[search_tool]
    )

    analyst = Agent(
        role="Data Analyst",
        goal="Analyze market data and extract key insights",
        backstory="""You are skilled at analyzing market data and identifying
        important trends and patterns that others might miss.""",
        verbose=True
    )

    writer = Agent(
        role="Report Writer",
        goal="Create clear, concise reports from market analysis",
        backstory="""You excel at synthesizing complex information into
        well-structured, easy-to-understand reports.""",
        verbose=True
    )

    # Create tasks
    research_task = Task(
        description="Research the current state of the electric vehicle market in the US",
        expected_output="Raw data on market size, key players, and trends",
        agent=researcher
    )

    analysis_task = Task(
        description="""Analyze the raw market data to identify key trends,
        market leaders, and growth opportunities""",
        expected_output="Analysis of market trends with key insights",
        agent=analyst,
    )

    report_task = Task(
        description="Create a comprehensive market report based on the analysis",
        expected_output="A professional market report on the US EV market",
        agent=writer,
    )

    # Create the crew
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, report_task],
        verbose=True,
        process=Process.sequential
    )

    # Create the evaluator for the crew
    agent_evaluator = AgentEvaluator(
        evaluators=[
            GoalAlignmentEvaluator(),
            ToolUsageEvaluator(),
            KnowledgeRetrievalEvaluator(),
            SemanticQualityEvaluator()
        ],
        crew=crew
    )

    # Create and register the evaluation callback
    evaluation_callback = create_evaluation_callbacks()
    agent_evaluator.set_callback(evaluation_callback)

    # Execute the crew tasks with evaluation tracing
    print("\n🔍 Executing crew tasks with evaluation tracing...\n")
    crew.kickoff()

    # After execution is complete, get the evaluation results
    evaluation_results = agent_evaluator.get_evaluation_results()

    # Print evaluation results
    print("\n📋 Evaluation Results:\n")
    for role, results in evaluation_results.items():
        print(f"=== {role} ===")
        # Each role might have multiple evaluation results (one per task)
        for result in results:
            # Use the string representation of the evaluation result
            print(str(result))
        print("\n")

    return evaluation_results


if __name__ == "__main__":
    print("\n\n" + "=" * 50)
    print("CREW EVALUATION EXAMPLE")
    print("=" * 50)

    # Uncomment to run crew evaluation example
    crew_results = run_crew_evaluation_example()
