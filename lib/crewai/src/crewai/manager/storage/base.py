"""Base storage interface for CrewManager persistence."""

from abc import ABC, abstractmethod
from typing import Any

from crewai.manager.models.agent_config import AgentConfig
from crewai.manager.models.crew_config import CrewConfig
from crewai.manager.models.knowledge_config import KnowledgeSourceConfig
from crewai.manager.models.mcp_config import MCPServerConfig
from crewai.manager.models.task_config import TaskConfig
from crewai.manager.models.tool_config import ToolConfig


class BaseStorage(ABC):
    """Abstract base class for CrewManager storage backends.

    All storage implementations must implement this interface to provide
    CRUD operations for agents, tasks, crews, and tools.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the storage backend.

        Called once when the storage is first used. Implementations should
        set up any required resources (files, databases, etc.).
        """
        pass

    # ==================== Agent Operations ====================

    @abstractmethod
    def save_agent(self, agent: AgentConfig) -> str:
        """Save an agent configuration.

        Args:
            agent: The agent configuration to save

        Returns:
            The ID of the saved agent
        """
        pass

    @abstractmethod
    def get_agent(self, agent_id: str) -> AgentConfig | None:
        """Get an agent by ID.

        Args:
            agent_id: The ID of the agent to retrieve

        Returns:
            The agent configuration, or None if not found
        """
        pass

    @abstractmethod
    def get_agent_by_role(self, role: str) -> AgentConfig | None:
        """Get an agent by role.

        Args:
            role: The role of the agent to find

        Returns:
            The first agent with matching role, or None if not found
        """
        pass

    @abstractmethod
    def list_agents(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[AgentConfig], int]:
        """List agents with pagination.

        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
            tags: Optional filter by tags

        Returns:
            Tuple of (list of agents, total count)
        """
        pass

    @abstractmethod
    def update_agent(
        self, agent_id: str, updates: dict[str, Any]
    ) -> AgentConfig | None:
        """Update an agent configuration.

        Args:
            agent_id: The ID of the agent to update
            updates: Dictionary of fields to update

        Returns:
            The updated agent configuration, or None if not found
        """
        pass

    @abstractmethod
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent by ID.

        Args:
            agent_id: The ID of the agent to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    # ==================== Task Operations ====================

    @abstractmethod
    def save_task(self, task: TaskConfig) -> str:
        """Save a task configuration.

        Args:
            task: The task configuration to save

        Returns:
            The ID of the saved task
        """
        pass

    @abstractmethod
    def get_task(self, task_id: str) -> TaskConfig | None:
        """Get a task by ID.

        Args:
            task_id: The ID of the task to retrieve

        Returns:
            The task configuration, or None if not found
        """
        pass

    @abstractmethod
    def list_tasks(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[TaskConfig], int]:
        """List tasks with pagination.

        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
            tags: Optional filter by tags

        Returns:
            Tuple of (list of tasks, total count)
        """
        pass

    @abstractmethod
    def update_task(
        self, task_id: str, updates: dict[str, Any]
    ) -> TaskConfig | None:
        """Update a task configuration.

        Args:
            task_id: The ID of the task to update
            updates: Dictionary of fields to update

        Returns:
            The updated task configuration, or None if not found
        """
        pass

    @abstractmethod
    def delete_task(self, task_id: str) -> bool:
        """Delete a task by ID.

        Args:
            task_id: The ID of the task to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    # ==================== Crew Operations ====================

    @abstractmethod
    def save_crew(self, crew: CrewConfig) -> str:
        """Save a crew configuration.

        Args:
            crew: The crew configuration to save

        Returns:
            The ID of the saved crew
        """
        pass

    @abstractmethod
    def get_crew(self, crew_id: str) -> CrewConfig | None:
        """Get a crew by ID.

        Args:
            crew_id: The ID of the crew to retrieve

        Returns:
            The crew configuration, or None if not found
        """
        pass

    @abstractmethod
    def get_crew_by_name(self, name: str) -> CrewConfig | None:
        """Get a crew by name.

        Args:
            name: The name of the crew to find

        Returns:
            The first crew with matching name, or None if not found
        """
        pass

    @abstractmethod
    def list_crews(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[CrewConfig], int]:
        """List crews with pagination.

        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
            tags: Optional filter by tags

        Returns:
            Tuple of (list of crews, total count)
        """
        pass

    @abstractmethod
    def update_crew(
        self, crew_id: str, updates: dict[str, Any]
    ) -> CrewConfig | None:
        """Update a crew configuration.

        Args:
            crew_id: The ID of the crew to update
            updates: Dictionary of fields to update

        Returns:
            The updated crew configuration, or None if not found
        """
        pass

    @abstractmethod
    def delete_crew(self, crew_id: str) -> bool:
        """Delete a crew by ID.

        Args:
            crew_id: The ID of the crew to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    # ==================== Tool Operations ====================

    @abstractmethod
    def save_tool(self, tool: ToolConfig) -> str:
        """Save a tool configuration.

        Args:
            tool: The tool configuration to save

        Returns:
            The ID of the saved tool
        """
        pass

    @abstractmethod
    def get_tool(self, tool_id: str) -> ToolConfig | None:
        """Get a tool by ID.

        Args:
            tool_id: The ID of the tool to retrieve

        Returns:
            The tool configuration, or None if not found
        """
        pass

    @abstractmethod
    def get_tool_by_name(self, name: str) -> ToolConfig | None:
        """Get a tool by name.

        Args:
            name: The name of the tool to find

        Returns:
            The first tool with matching name, or None if not found
        """
        pass

    @abstractmethod
    def list_tools(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[ToolConfig], int]:
        """List tools with pagination.

        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
            tags: Optional filter by tags

        Returns:
            Tuple of (list of tools, total count)
        """
        pass

    @abstractmethod
    def update_tool(
        self, tool_id: str, updates: dict[str, Any]
    ) -> ToolConfig | None:
        """Update a tool configuration.

        Args:
            tool_id: The ID of the tool to update
            updates: Dictionary of fields to update

        Returns:
            The updated tool configuration, or None if not found
        """
        pass

    @abstractmethod
    def delete_tool(self, tool_id: str) -> bool:
        """Delete a tool by ID.

        Args:
            tool_id: The ID of the tool to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    # ==================== Knowledge Source Operations ====================

    @abstractmethod
    def save_knowledge_source(self, source: KnowledgeSourceConfig) -> str:
        """Save a knowledge source configuration.

        Args:
            source: The knowledge source configuration to save

        Returns:
            The ID of the saved knowledge source
        """
        pass

    @abstractmethod
    def get_knowledge_source(self, source_id: str) -> KnowledgeSourceConfig | None:
        """Get a knowledge source by ID.

        Args:
            source_id: The ID of the knowledge source to retrieve

        Returns:
            The knowledge source configuration, or None if not found
        """
        pass

    @abstractmethod
    def get_knowledge_source_by_name(self, name: str) -> KnowledgeSourceConfig | None:
        """Get a knowledge source by name.

        Args:
            name: The name of the knowledge source to find

        Returns:
            The first knowledge source with matching name, or None if not found
        """
        pass

    @abstractmethod
    def list_knowledge_sources(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[KnowledgeSourceConfig], int]:
        """List knowledge sources with pagination.

        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
            tags: Optional filter by tags

        Returns:
            Tuple of (list of knowledge sources, total count)
        """
        pass

    @abstractmethod
    def update_knowledge_source(
        self, source_id: str, updates: dict[str, Any]
    ) -> KnowledgeSourceConfig | None:
        """Update a knowledge source configuration.

        Args:
            source_id: The ID of the knowledge source to update
            updates: Dictionary of fields to update

        Returns:
            The updated knowledge source configuration, or None if not found
        """
        pass

    @abstractmethod
    def delete_knowledge_source(self, source_id: str) -> bool:
        """Delete a knowledge source by ID.

        Args:
            source_id: The ID of the knowledge source to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    # ==================== MCP Server Operations ====================

    @abstractmethod
    def save_mcp_server(self, server: MCPServerConfig) -> str:
        """Save an MCP server configuration.

        Args:
            server: The MCP server configuration to save

        Returns:
            The ID of the saved MCP server
        """
        pass

    @abstractmethod
    def get_mcp_server(self, server_id: str) -> MCPServerConfig | None:
        """Get an MCP server by ID.

        Args:
            server_id: The ID of the MCP server to retrieve

        Returns:
            The MCP server configuration, or None if not found
        """
        pass

    @abstractmethod
    def get_mcp_server_by_name(self, name: str) -> MCPServerConfig | None:
        """Get an MCP server by name.

        Args:
            name: The name of the MCP server to find

        Returns:
            The first MCP server with matching name, or None if not found
        """
        pass

    @abstractmethod
    def list_mcp_servers(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[MCPServerConfig], int]:
        """List MCP servers with pagination.

        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
            tags: Optional filter by tags

        Returns:
            Tuple of (list of MCP servers, total count)
        """
        pass

    @abstractmethod
    def update_mcp_server(
        self, server_id: str, updates: dict[str, Any]
    ) -> MCPServerConfig | None:
        """Update an MCP server configuration.

        Args:
            server_id: The ID of the MCP server to update
            updates: Dictionary of fields to update

        Returns:
            The updated MCP server configuration, or None if not found
        """
        pass

    @abstractmethod
    def delete_mcp_server(self, server_id: str) -> bool:
        """Delete an MCP server by ID.

        Args:
            server_id: The ID of the MCP server to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    # ==================== Utility Operations ====================

    @abstractmethod
    def export_all(self) -> dict[str, Any]:
        """Export all data from the storage.

        Returns:
            Dictionary containing all agents, tasks, crews, and tools
        """
        pass

    @abstractmethod
    def import_all(self, data: dict[str, Any]) -> None:
        """Import data into the storage.

        Args:
            data: Dictionary containing agents, tasks, crews, and tools
        """
        pass

    @abstractmethod
    def clear_all(self) -> None:
        """Clear all data from the storage."""
        pass
