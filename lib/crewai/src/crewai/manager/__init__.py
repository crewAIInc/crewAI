"""CrewManager - Programmatic CRUD API for CrewAI.

This module provides a Python library interface for managing CrewAI
agents, tasks, crews, and tools programmatically. It's designed to
be used by UI applications and other tools that need to create and
manage CrewAI configurations without using the CLI.

Example:
    ```python
    from crewai.manager import CrewManager, AgentConfig, TaskConfig, CrewConfig
    from crewai.manager.storage import SQLiteStorage

    # Initialize with persistent storage
    manager = CrewManager(storage=SQLiteStorage("my_crews.db"))

    # Create an agent
    agent = AgentConfig(
        role="Researcher",
        goal="Find and synthesize information",
        backstory="Expert researcher with attention to detail",
        llm="gpt-4",
    )
    manager.create_agent(agent)

    # Create a task
    task = TaskConfig(
        description="Research the topic: {topic}",
        expected_output="A comprehensive report on the topic",
        agent_id=agent.id,
    )
    manager.create_task(task)

    # Create and execute a crew
    crew = CrewConfig(
        name="Research Crew",
        agent_ids=[agent.id],
        task_ids=[task.id],
    )
    manager.create_crew(crew)

    result = manager.execute_crew(crew.id, inputs={"topic": "AI trends"})
    print(result.raw_output)
    ```

Storage Backends:
    - InMemoryStorage: For development and testing (data lost on restart)
    - JSONFileStorage: Persistent storage in JSON format
    - YAMLFileStorage: Persistent storage in YAML format (human-readable)
    - SQLiteStorage: Robust database storage with concurrent access
"""

from crewai.manager.crew_manager import CrewManager
from crewai.manager.models import (
    AgentConfig,
    CrewConfig,
    ExecutionProgress,
    ExecutionResult,
    KnowledgeSourceConfig,
    ListResult,
    MCPServerConfig,
    OperationResult,
    TaskConfig,
    ToolConfig,
)
from crewai.manager.storage import (
    BaseStorage,
    InMemoryStorage,
    JSONFileStorage,
    SQLiteStorage,
    YAMLFileStorage,
)

__all__ = [
    # Main manager
    "CrewManager",
    # Config models
    "AgentConfig",
    "TaskConfig",
    "CrewConfig",
    "ToolConfig",
    "KnowledgeSourceConfig",
    "MCPServerConfig",
    # Response models
    "OperationResult",
    "ListResult",
    "ExecutionResult",
    "ExecutionProgress",
    # Storage backends
    "BaseStorage",
    "InMemoryStorage",
    "JSONFileStorage",
    "YAMLFileStorage",
    "SQLiteStorage",
]
