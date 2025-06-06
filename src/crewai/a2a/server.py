"""A2A Server utilities for CrewAI integration.

This module provides convenience functions for starting A2A servers with CrewAI
crews, supporting multiple transport protocols and configurations.
"""

import logging
from typing import Optional

try:
    from a2a.server.agent_execution.agent_executor import AgentExecutor
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import AgentCard, AgentCapabilities, AgentSkill
except ImportError:
    raise ImportError(
        "A2A integration requires the 'a2a' extra dependency. "
        "Install with: pip install crewai[a2a]"
    )

logger = logging.getLogger(__name__)


def start_a2a_server(
    agent_executor: AgentExecutor,
    host: str = "localhost",
    port: int = 10001,
    transport: str = "starlette",
    **kwargs
) -> None:
    """Start an A2A server with the given agent executor.
    
    This is a convenience function that creates and starts an A2A server
    with the specified configuration.
    
    Args:
        agent_executor: The A2A agent executor to serve
        host: Host address to bind the server to
        port: Port number to bind the server to  
        transport: Transport protocol to use ("starlette" or "fastapi")
        **kwargs: Additional arguments passed to the server
        
    Example:
        from crewai import Agent, Crew, Task
        from crewai.a2a import CrewAgentExecutor, start_a2a_server
        
        agent = Agent(role="Assistant", goal="Help users", backstory="Helpful AI")
        task = Task(description="Help with {query}", agent=agent)
        crew = Crew(agents=[agent], tasks=[task])
        
        executor = CrewAgentExecutor(crew)
        start_a2a_server(executor, host="0.0.0.0", port=8080)
    """
    app = create_a2a_app(agent_executor, transport=transport, **kwargs)
    
    logger.info(f"Starting A2A server on {host}:{port} using {transport} transport")
    
    try:
        import uvicorn
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        raise ImportError("uvicorn is required to run the A2A server. Install with: pip install uvicorn")


def create_a2a_app(
    agent_executor: AgentExecutor,
    transport: str = "starlette",
    agent_name: Optional[str] = None,
    agent_description: Optional[str] = None,
    **kwargs
):
    """Create an A2A application with the given agent executor.
    
    This function creates an A2A server application that can be run
    with any ASGI server.
    
    Args:
        agent_executor: The A2A agent executor to serve
        transport: Transport protocol to use ("starlette" or "fastapi")
        agent_name: Optional name for the agent
        agent_description: Optional description for the agent
        **kwargs: Additional arguments passed to the transport
        
    Returns:
        ASGI application ready to be served
        
    Example:
        from crewai.a2a import CrewAgentExecutor, create_a2a_app
        
        executor = CrewAgentExecutor(crew)
        app = create_a2a_app(
            executor, 
            agent_name="My Crew Agent",
            agent_description="A helpful CrewAI agent"
        )
        
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8080)
    """
    agent_card = AgentCard(
        name=agent_name or "CrewAI Agent",
        description=agent_description or "A CrewAI agent exposed via A2A protocol",
        version="1.0.0",
        supportedContentTypes=getattr(agent_executor, 'supported_content_types', ['text', 'text/plain']),
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False
        ),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[
            AgentSkill(
                id="crew_execution",
                name="Crew Execution",
                description="Execute CrewAI crew tasks with multiple agents",
                examples=["Process user queries", "Coordinate multi-agent workflows"],
                tags=["crewai", "multi-agent", "workflow"]
            )
        ],
        url="https://github.com/crewAIInc/crewAI"
    )
    
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(agent_executor, task_store)
    
    if transport.lower() == "fastapi":
        raise ValueError("FastAPI transport is not available in the current A2A SDK version")
    else:
        app_instance = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
            **kwargs
        )
    
    return app_instance.build()
