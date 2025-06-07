"""Server configuration for A2A integration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ServerConfig:
    """Configuration for A2A server.
    
    This class encapsulates server settings to improve readability
    and flexibility for server setups.
    
    Attributes:
        host: Host address to bind the server to
        port: Port number to bind the server to
        transport: Transport protocol to use ("starlette" or "fastapi")
        agent_name: Optional name for the agent
        agent_description: Optional description for the agent
    """
    host: str = "localhost"
    port: int = 10001
    transport: str = "starlette"
    agent_name: Optional[str] = None
    agent_description: Optional[str] = None
