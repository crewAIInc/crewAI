"""
Task cloning service for domain-specific replication logic.

Separates task cloning business logic from Pydantic BaseModel concerns,
allowing Task to remain a pure data model while providing rich cloning capabilities.
"""

import copy
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
            task_mapping: Mapping of task keys (Task.key) to cloned Task instances
            
        Returns:
            Deep copy of the task with updated references and fresh ID
        """
        # Create new task instance with fresh ID to avoid shared state
        # Extract all fields except those we need to handle specially
        task_data = task.model_dump(exclude={'id', 'agent', 'context', 'tools'})
        
        # Import here to avoid circular imports
        from crewai.task import Task
        new_task = Task(**task_data)
        
        # Handle agent assignment
        if task.agent:
            # Find matching agent by role
            matching_agents = [
                agent for agent in agents 
                if agent.role == task.agent.role
            ]
            new_task.agent = matching_agents[0] if matching_agents else None
        
        # Handle context task references with safe fallback
        if task.context:
            new_context = []
            for context_task in task.context:
                if hasattr(context_task, 'key'):
                    # Use .get() to safely handle missing keys
                    mapped_task = task_mapping.get(context_task.key)
                    if mapped_task:
                        new_context.append(mapped_task)
                    else:
                        # Fallback to original context task if not in mapping
                        new_context.append(context_task)
                else:
                    new_context.append(context_task)
            new_task.context = new_context
        
        # Deep copy tools to prevent cross-run interference
        if hasattr(task, 'tools') and task.tools:
            new_task.tools = copy.deepcopy(task.tools)
        
        return new_task