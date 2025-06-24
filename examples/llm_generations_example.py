"""
Example demonstrating the new LLM generations and logprobs functionality.
"""

from crewai import Agent, Task, LLM
from crewai.utilities.xml_parser import extract_xml_content


def example_multiple_generations():
    """Example of using multiple generations with an agent."""
    
    llm = LLM(
        model="gpt-3.5-turbo",
        n=3,  # Request 3 generations
        temperature=0.8,  # Higher temperature for variety
        return_full_completion=True
    )
    
    agent = Agent(
        role="Creative Writer",
        goal="Write engaging content",
        backstory="You are a creative writer who generates multiple ideas",
        llm=llm,
        return_completion_metadata=True
    )
    
    task = Task(
        description="Write a short story opening about a mysterious door",
        agent=agent,
        expected_output="A compelling story opening"
    )
    
    result = agent.execute_task(task)
    
    print("Primary result:")
    print(result)
    print("\n" + "="*50 + "\n")
    
    if hasattr(task, 'output') and task.output.completion_metadata:
        generations = task.output.get_generations()
        if generations:
            print(f"Generated {len(generations)} alternatives:")
            for i, generation in enumerate(generations, 1):
                print(f"\nGeneration {i}:")
                print(generation)
                print("-" * 30)


def example_xml_extraction():
    """Example of extracting structured content from agent output."""
    
    agent = Agent(
        role="Problem Solver",
        goal="Solve problems systematically",
        backstory="You think step by step and show your reasoning",
        llm=LLM(model="gpt-3.5-turbo")
    )
    
    task = Task(
        description="""
        Solve this problem: How can we reduce energy consumption in an office building?
        
        Please structure your response with:
        - <thinking> tags for your internal reasoning
        - <analysis> tags for your analysis of the problem
        - <solution> tags for your proposed solution
        """,
        agent=agent,
        expected_output="A structured solution with reasoning"
    )
    
    result = agent.execute_task(task)
    
    print("Full agent output:")
    print(result)
    print("\n" + "="*50 + "\n")
    
    thinking = extract_xml_content(result, "thinking")
    analysis = extract_xml_content(result, "analysis")
    solution = extract_xml_content(result, "solution")
    
    if thinking:
        print("Agent's thinking process:")
        print(thinking)
        print("\n" + "-"*30 + "\n")
    
    if analysis:
        print("Problem analysis:")
        print(analysis)
        print("\n" + "-"*30 + "\n")
    
    if solution:
        print("Proposed solution:")
        print(solution)


def example_logprobs_analysis():
    """Example of accessing log probabilities for analysis."""
    
    llm = LLM(
        model="gpt-3.5-turbo",
        logprobs=5,  # Request top 5 log probabilities
        top_logprobs=3,  # Show top 3 alternatives
        return_full_completion=True
    )
    
    agent = Agent(
        role="Decision Analyst",
        goal="Make confident decisions",
        backstory="You analyze confidence levels in your responses",
        llm=llm,
        return_completion_metadata=True
    )
    
    task = Task(
        description="Should we invest in renewable energy? Give a yes/no answer with confidence.",
        agent=agent,
        expected_output="A clear yes/no decision"
    )
    
    result = agent.execute_task(task)
    
    print("Decision:")
    print(result)
    print("\n" + "="*50 + "\n")
    
    if hasattr(task, 'output') and task.output.completion_metadata:
        logprobs = task.output.get_logprobs()
        usage = task.output.get_usage_metrics()
        
        if logprobs:
            print("Confidence analysis (log probabilities):")
            print(f"Available logprobs data: {len(logprobs)} choices")
            
        if usage:
            print("\nToken usage:")
            print(f"Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"Total tokens: {usage.get('total_tokens', 'N/A')}")


if __name__ == "__main__":
    print("=== CrewAI LLM Generations and XML Extraction Examples ===\n")
    
    print("1. Multiple Generations Example:")
    print("-" * 40)
    try:
        example_multiple_generations()
    except Exception as e:
        print(f"Example requires actual LLM API access: {e}")
    
    print("\n\n2. XML Content Extraction Example:")
    print("-" * 40)
    try:
        example_xml_extraction()
    except Exception as e:
        print(f"Example requires actual LLM API access: {e}")
    
    print("\n\n3. Log Probabilities Analysis Example:")
    print("-" * 40)
    try:
        example_logprobs_analysis()
    except Exception as e:
        print(f"Example requires actual LLM API access: {e}")
    
    print("\n=== Examples completed ===")
