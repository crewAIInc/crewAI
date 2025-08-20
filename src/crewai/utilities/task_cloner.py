"""
Task cloning service for domain-specific replication logic.

Separates task cloning business logic from Pydantic BaseModel concerns,
allowing Task to remain a pure data model while providing rich cloning capabilities.
"""

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.task import Task


class TaskCloner:
    """
    Service for creating deep copies of tasks with agent and context mapping.
    
    Separates domain-specific cloning logic from data model concerns,
    providing clean separation of responsibilities.
    """
    
    @staticmethod
    def deep_copy(
        task: "Task", 
        agents: List["BaseAgent"], 
        task_mapping: Dict[str, "Task"]
    ) -> "Task":
        """
        Create a deep copy of a task with agent and task mapping.
        
        Args:
            task: Source task to clone
            agents: List of agents for the cloned task
            task_mapping: Mapping of task IDs to task instances
            
        Returns:
            Deep copy of the task with updated references
        """
        # Create base copy using Pydantic's copy method
        new_task = task.model_copy()
        
        # Reset fields that need special handling
        new_task.agent = None
        new_task.context = None
        if hasattr(new_task, 'tools'):
            new_task.tools = []
        
        # Handle agent assignment
        if task.agent:
            # Find matching agent by role
            matching_agents = [
                agent for agent in agents 
                if agent.role == task.agent.role
            ]
            new_task.agent = matching_agents[0] if matching_agents else None
        
        # Handle context task references
        if task.context:
            new_context = []
            for context_task in task.context:
                if hasattr(context_task, 'key') and context_task.key in task_mapping:
                    new_context.append(task_mapping[context_task.key])
                else:
                    new_context.append(context_task)
            new_task.context = new_context
        
        # Clone tools if present
        if hasattr(task, 'tools') and task.tools:
            new_task.tools = [tool for tool in task.tools]
        
        return new_task