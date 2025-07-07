"""
Example demonstrating OAuth2 authentication with custom LLM providers in CrewAI.

This example shows how to configure and use OAuth2-authenticated LLM providers.
"""

import json
from pathlib import Path
from crewai import Agent, Task, Crew, LLM

def create_example_config():
    """Create an example OAuth2 configuration file."""
    config = {
        "oauth2_providers": {
            "my_custom_provider": {
                "client_id": "your_client_id_here",
                "client_secret": "your_client_secret_here",
                "token_url": "https://your-provider.com/oauth/token",
                "scope": "llm.read llm.write"
            }
        }
    }
    
    config_path = Path("example_oauth2_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created example config at {config_path}")
    return config_path

def main():
    config_path = create_example_config()
    
    try:
        llm = LLM(
            model="my_custom_provider/my-model",
            oauth2_config_path=str(config_path)
        )
        
        agent = Agent(
            role="Research Assistant",
            goal="Provide helpful research insights",
            backstory="An AI assistant specialized in research and analysis",
            llm=llm
        )
        
        task = Task(
            description="Research the benefits of OAuth2 authentication in AI systems",
            agent=agent,
            expected_output="A comprehensive summary of OAuth2 benefits"
        )
        
        crew = Crew(agents=[agent], tasks=[task])
        
        print("Running crew with OAuth2-authenticated LLM...")
        result = crew.kickoff()
        print(f"Result: {result}")
        
    finally:
        if config_path.exists():
            config_path.unlink()

if __name__ == "__main__":
    main()
