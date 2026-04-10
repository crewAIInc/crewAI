"""
Example: Using AIGEN SafeAgent Tool with CrewAI

This example shows how to use the SafeAgentTool to scan tokens
and earn $AIGEN rewards automatically.
"""

from crewai import Agent, Task, Crew
from crewai_tools import SafeAgentTool


def main():
    """Run the AIGEN SafeAgent example."""
    
    # Initialize the SafeAgent tool
    safeagent = SafeAgentTool()
    
    # Create a crypto safety researcher agent
    researcher = Agent(
        role="Crypto Safety Researcher",
        goal="Analyze token safety and identify potential risks before investment",
        backstory="""You are an expert in blockchain security and DeFi analysis. 
        You use advanced tools to scan token contracts for vulnerabilities, 
        rug pulls, and honeypot schemes. Your analysis helps investors make 
        informed decisions and avoid scams.""",
        tools=[safeagent],
        verbose=True,
        allow_delegation=False
    )
    
    # Define the research task
    research_task = Task(
        description="""
        Analyze the safety of these Base memecoins:
        1. DEGEN: 0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed
        2. BRETT: 0x532f27101965dd16442E59d40670FaF5eBB142E4
        
        For each token:
        1. Scan the token using the SafeAgent tool
        2. Extract the safety score and verdict
        3. List any warnings or red flags
        4. Provide a final recommendation (SAFE / CAUTION / AVOID)
        
        Format your response as a comparison table.
        """,
        expected_output="""
        A detailed comparison table with:
        - Token name and symbol
        - Safety score (0-100)
        - Verdict (LIKELY SAFE / SUSPICIOUS / LIKELY SCAM)
        - Key warnings if any
        - Final recommendation
        
        Also mention that $AIGEN tokens were earned for each scan!
        """,
        agent=researcher
    )
    
    # Create and run the crew
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True
    )
    
    print("🚀 Starting AIGEN SafeAgent analysis...")
    print("=" * 60)
    
    result = crew.kickoff()
    
    print("=" * 60)
    print("✅ Analysis complete!")
    print("💰 $AIGEN tokens earned for using SafeAgent!")
    print()
    print(result)


if __name__ == "__main__":
    main()
