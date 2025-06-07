"""A2A (Agent-to-Agent) protocol integration for CrewAI.

This module provides integration with the A2A protocol to enable remote agent
interoperability. It allows CrewAI crews to be exposed as A2A-compatible agents
that can communicate with other agents following the A2A protocol standard.

The integration is optional and requires the 'a2a' extra dependency:
    pip install crewai[a2a]

Example:
    from crewai import Agent, Crew, Task
    from crewai.a2a import CrewAgentExecutor, start_a2a_server
    
    agent = Agent(role="Assistant", goal="Help users", backstory="Helpful AI")
    task = Task(description="Help with {query}", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])
    
    executor = CrewAgentExecutor(crew)
    start_a2a_server(executor, host="localhost", port=8080)
"""

try:
    from .crew_agent_executor import CrewAgentExecutor
    from .server import start_a2a_server, create_a2a_app
    
    __all__ = [
        "CrewAgentExecutor",
        "start_a2a_server", 
        "create_a2a_app"
    ]
except ImportError:
    import warnings
    warnings.warn(
        "A2A integration requires the 'a2a' extra dependency. "
        "Install with: pip install crewai[a2a]",
        ImportWarning
    )
    
    def _missing_dependency(*args, **kwargs):
        raise ImportError(
            "A2A integration requires the 'a2a' extra dependency. "
            "Install with: pip install crewai[a2a]"
        )
    
    CrewAgentExecutor = _missing_dependency  # type: ignore
    start_a2a_server = _missing_dependency  # type: ignore
    create_a2a_app = _missing_dependency  # type: ignore
    
    __all__ = []
