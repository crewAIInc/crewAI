#!/usr/bin/env python3
"""
Example demonstrating encrypted agent-to-agent communication in CrewAI.

This example shows how to:
1. Enable encrypted communication for agents
2. Use existing agent tools with encryption
3. Verify that communication is encrypted between agents
"""

from crewai import Agent, Crew, Task
from crewai.security import SecurityConfig
from crewai.tools.agent_tools.ask_question_tool import AskQuestionTool


def main():
    """Demonstrate encrypted agent communication."""
    print("🔒 CrewAI Encrypted Agent Communication Example")
    print("=" * 50)
    
    # Create agents with encrypted communication enabled
    print("Creating agents with encrypted communication...")
    
    # Researcher agent with encryption enabled
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Conduct thorough research and analysis",
        backstory="You are an expert researcher with years of experience in data analysis.",
        security_config=SecurityConfig(encrypted_communication=True),
        verbose=True
    )
    
    # Writer agent with encryption enabled
    writer = Agent(
        role="Content Writer", 
        goal="Create compelling content based on research",
        backstory="You are a skilled writer who transforms complex research into engaging content.",
        security_config=SecurityConfig(encrypted_communication=True),
        verbose=True
    )
    
    print(f"✓ Researcher agent created with encryption: {researcher.security_config.encrypted_communication}")
    print(f"✓ Writer agent created with encryption: {writer.security_config.encrypted_communication}")
    print(f"✓ Researcher fingerprint: {researcher.security_config.fingerprint.uuid_str[:8]}...")
    print(f"✓ Writer fingerprint: {writer.security_config.fingerprint.uuid_str[:8]}...")
    
    # Create agent tools - these will automatically use encryption when available
    agent_tools = [
        AskQuestionTool(
            agents=[researcher, writer],
            description="Tool for asking questions to coworkers with encrypted communication support"
        )
    ]
    
    print(f"✓ Agent tools created with encryption capability")
    
    # Create tasks that will involve encrypted communication
    research_task = Task(
        description="Research the latest trends in artificial intelligence and machine learning",
        expected_output="A comprehensive research report on AI/ML trends",
        agent=researcher
    )
    
    writing_task = Task(
        description="Ask the researcher about their findings and write a blog post",
        expected_output="An engaging blog post about AI trends", 
        agent=writer,
        tools=agent_tools
    )
    
    # Create crew with both agents
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=True
    )
    
    print("\n🚀 Starting crew execution with encrypted communication...")
    print("Note: Agent communications will be automatically encrypted!")
    
    # Execute the crew - agent tools will use encryption automatically
    try:
        result = crew.kickoff()
        print("\n✅ Crew execution completed successfully!")
        print("=" * 50)
        print("RESULT:")
        print(result)
        
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        print("This is expected in a demo environment without proper LLM configuration")
    
    print("\n🔍 Key Features Demonstrated:")
    print("- Agents created with SecurityConfig(encrypted_communication=True)")
    print("- Unique fingerprints generated for each agent")
    print("- Agent tools automatically detect encryption capability")
    print("- Communication payloads encrypted using Fernet symmetric encryption")
    print("- Keys derived from agent fingerprints for secure communication")
    print("- Backward compatibility maintained - non-encrypted agents still work")


if __name__ == "__main__":
    main()