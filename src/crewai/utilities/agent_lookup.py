"""Agent lookup utilities for CrewAI."""

from typing import List, Optional, Union
from ..agents.agent_builder.base_agent import BaseAgent
from ..exceptions import AgentLookupError


class AgentLookupMixin:
    """Mixin class for agent lookup functionality."""
    
    def get_agent_by_role(self, role: str, agents: List[BaseAgent]) -> Union[BaseAgent, None]:
        """Find an agent by role, case-insensitive.
        
        Args:
            role: The role to search for
            agents: List of agents to search through
            
        Returns:
            The found agent or None
        """
        normalized_role = role.casefold().replace('"', "").replace("\n", "")
        return next(
            (agent for agent in agents 
             if agent.role.casefold().replace("\n", "") == normalized_role),
            None
        )
