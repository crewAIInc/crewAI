"""In-memory storage implementation for CrewManager."""

from typing import Any

from crewai.manager.models.agent_config import AgentConfig
from crewai.manager.models.crew_config import CrewConfig
from crewai.manager.models.knowledge_config import KnowledgeSourceConfig
from crewai.manager.models.mcp_config import MCPServerConfig
from crewai.manager.models.task_config import TaskConfig
from crewai.manager.models.tool_config import ToolConfig
from crewai.manager.storage.base import BaseStorage


class InMemoryStorage(BaseStorage):
    """In-memory storage backend for development and testing.

    Data is stored in dictionaries and is lost when the process ends.
    This is useful for development, testing, and temporary operations.
    """

    def __init__(self) -> None:
        """Initialize the in-memory storage."""
        self._agents: dict[str, AgentConfig] = {}
        self._tasks: dict[str, TaskConfig] = {}
        self._crews: dict[str, CrewConfig] = {}
        self._tools: dict[str, ToolConfig] = {}
        self._knowledge_sources: dict[str, KnowledgeSourceConfig] = {}
        self._mcp_servers: dict[str, MCPServerConfig] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the storage backend."""
        self._initialized = True

    def _filter_by_tags(
        self, items: list[Any], tags: list[str] | None
    ) -> list[Any]:
        """Filter items by tags."""
        if not tags:
            return items
        return [
            item for item in items
            if hasattr(item, "tags") and any(tag in item.tags for tag in tags)
        ]

    def _paginate(
        self, items: list[Any], offset: int, limit: int
    ) -> tuple[list[Any], int]:
        """Apply pagination to a list."""
        total = len(items)
        return items[offset:offset + limit], total

    # ==================== Agent Operations ====================

    def save_agent(self, agent: AgentConfig) -> str:
        """Save an agent configuration."""
        self._agents[agent.id] = agent
        return agent.id

    def get_agent(self, agent_id: str) -> AgentConfig | None:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_agent_by_role(self, role: str) -> AgentConfig | None:
        """Get an agent by role."""
        for agent in self._agents.values():
            if agent.role == role:
                return agent
        return None

    def list_agents(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[AgentConfig], int]:
        """List agents with pagination."""
        items = list(self._agents.values())
        items = self._filter_by_tags(items, tags)
        # Sort by created_at descending (newest first)
        items.sort(key=lambda x: x.created_at, reverse=True)
        return self._paginate(items, offset, limit)

    def update_agent(
        self, agent_id: str, updates: dict[str, Any]
    ) -> AgentConfig | None:
        """Update an agent configuration."""
        agent = self._agents.get(agent_id)
        if not agent:
            return None

        # Update fields
        for key, value in updates.items():
            if hasattr(agent, key):
                setattr(agent, key, value)

        agent.update_timestamp()
        self._agents[agent_id] = agent
        return agent

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent by ID."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    # ==================== Task Operations ====================

    def save_task(self, task: TaskConfig) -> str:
        """Save a task configuration."""
        self._tasks[task.id] = task
        return task.id

    def get_task(self, task_id: str) -> TaskConfig | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[TaskConfig], int]:
        """List tasks with pagination."""
        items = list(self._tasks.values())
        items = self._filter_by_tags(items, tags)
        items.sort(key=lambda x: x.created_at, reverse=True)
        return self._paginate(items, offset, limit)

    def update_task(
        self, task_id: str, updates: dict[str, Any]
    ) -> TaskConfig | None:
        """Update a task configuration."""
        task = self._tasks.get(task_id)
        if not task:
            return None

        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)

        task.update_timestamp()
        self._tasks[task_id] = task
        return task

    def delete_task(self, task_id: str) -> bool:
        """Delete a task by ID."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    # ==================== Crew Operations ====================

    def save_crew(self, crew: CrewConfig) -> str:
        """Save a crew configuration."""
        self._crews[crew.id] = crew
        return crew.id

    def get_crew(self, crew_id: str) -> CrewConfig | None:
        """Get a crew by ID."""
        return self._crews.get(crew_id)

    def get_crew_by_name(self, name: str) -> CrewConfig | None:
        """Get a crew by name."""
        for crew in self._crews.values():
            if crew.name == name:
                return crew
        return None

    def list_crews(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[CrewConfig], int]:
        """List crews with pagination."""
        items = list(self._crews.values())
        items = self._filter_by_tags(items, tags)
        items.sort(key=lambda x: x.created_at, reverse=True)
        return self._paginate(items, offset, limit)

    def update_crew(
        self, crew_id: str, updates: dict[str, Any]
    ) -> CrewConfig | None:
        """Update a crew configuration."""
        crew = self._crews.get(crew_id)
        if not crew:
            return None

        for key, value in updates.items():
            if hasattr(crew, key):
                setattr(crew, key, value)

        crew.update_timestamp()
        self._crews[crew_id] = crew
        return crew

    def delete_crew(self, crew_id: str) -> bool:
        """Delete a crew by ID."""
        if crew_id in self._crews:
            del self._crews[crew_id]
            return True
        return False

    # ==================== Tool Operations ====================

    def save_tool(self, tool: ToolConfig) -> str:
        """Save a tool configuration."""
        self._tools[tool.id] = tool
        return tool.id

    def get_tool(self, tool_id: str) -> ToolConfig | None:
        """Get a tool by ID."""
        return self._tools.get(tool_id)

    def get_tool_by_name(self, name: str) -> ToolConfig | None:
        """Get a tool by name."""
        for tool in self._tools.values():
            if tool.name == name:
                return tool
        return None

    def list_tools(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[ToolConfig], int]:
        """List tools with pagination."""
        items = list(self._tools.values())
        items = self._filter_by_tags(items, tags)
        items.sort(key=lambda x: x.created_at, reverse=True)
        return self._paginate(items, offset, limit)

    def update_tool(
        self, tool_id: str, updates: dict[str, Any]
    ) -> ToolConfig | None:
        """Update a tool configuration."""
        tool = self._tools.get(tool_id)
        if not tool:
            return None

        for key, value in updates.items():
            if hasattr(tool, key):
                setattr(tool, key, value)

        tool.update_timestamp()
        self._tools[tool_id] = tool
        return tool

    def delete_tool(self, tool_id: str) -> bool:
        """Delete a tool by ID."""
        if tool_id in self._tools:
            del self._tools[tool_id]
            return True
        return False

    # ==================== Knowledge Source Operations ====================

    def save_knowledge_source(self, source: KnowledgeSourceConfig) -> str:
        """Save a knowledge source configuration."""
        self._knowledge_sources[source.id] = source
        return source.id

    def get_knowledge_source(self, source_id: str) -> KnowledgeSourceConfig | None:
        """Get a knowledge source by ID."""
        return self._knowledge_sources.get(source_id)

    def get_knowledge_source_by_name(self, name: str) -> KnowledgeSourceConfig | None:
        """Get a knowledge source by name."""
        for source in self._knowledge_sources.values():
            if source.name == name:
                return source
        return None

    def list_knowledge_sources(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[KnowledgeSourceConfig], int]:
        """List knowledge sources with pagination."""
        items = list(self._knowledge_sources.values())
        items = self._filter_by_tags(items, tags)
        items.sort(key=lambda x: x.created_at, reverse=True)
        return self._paginate(items, offset, limit)

    def update_knowledge_source(
        self, source_id: str, updates: dict[str, Any]
    ) -> KnowledgeSourceConfig | None:
        """Update a knowledge source configuration."""
        source = self._knowledge_sources.get(source_id)
        if not source:
            return None

        for key, value in updates.items():
            if hasattr(source, key):
                setattr(source, key, value)

        source.update_timestamp()
        self._knowledge_sources[source_id] = source
        return source

    def delete_knowledge_source(self, source_id: str) -> bool:
        """Delete a knowledge source by ID."""
        if source_id in self._knowledge_sources:
            del self._knowledge_sources[source_id]
            return True
        return False

    # ==================== MCP Server Operations ====================

    def save_mcp_server(self, server: MCPServerConfig) -> str:
        """Save an MCP server configuration."""
        self._mcp_servers[server.id] = server
        return server.id

    def get_mcp_server(self, server_id: str) -> MCPServerConfig | None:
        """Get an MCP server by ID."""
        return self._mcp_servers.get(server_id)

    def get_mcp_server_by_name(self, name: str) -> MCPServerConfig | None:
        """Get an MCP server by name."""
        for server in self._mcp_servers.values():
            if server.name == name:
                return server
        return None

    def list_mcp_servers(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[MCPServerConfig], int]:
        """List MCP servers with pagination."""
        items = list(self._mcp_servers.values())
        items = self._filter_by_tags(items, tags)
        items.sort(key=lambda x: x.created_at, reverse=True)
        return self._paginate(items, offset, limit)

    def update_mcp_server(
        self, server_id: str, updates: dict[str, Any]
    ) -> MCPServerConfig | None:
        """Update an MCP server configuration."""
        server = self._mcp_servers.get(server_id)
        if not server:
            return None

        for key, value in updates.items():
            if hasattr(server, key):
                setattr(server, key, value)

        server.update_timestamp()
        self._mcp_servers[server_id] = server
        return server

    def delete_mcp_server(self, server_id: str) -> bool:
        """Delete an MCP server by ID."""
        if server_id in self._mcp_servers:
            del self._mcp_servers[server_id]
            return True
        return False

    # ==================== Utility Operations ====================

    def export_all(self) -> dict[str, Any]:
        """Export all data from the storage."""
        return {
            "agents": [agent.model_dump() for agent in self._agents.values()],
            "tasks": [task.model_dump() for task in self._tasks.values()],
            "crews": [crew.model_dump() for crew in self._crews.values()],
            "tools": [tool.model_dump() for tool in self._tools.values()],
            "knowledge_sources": [
                source.model_dump() for source in self._knowledge_sources.values()
            ],
            "mcp_servers": [
                server.model_dump() for server in self._mcp_servers.values()
            ],
        }

    def import_all(self, data: dict[str, Any]) -> None:
        """Import data into the storage."""
        if "agents" in data:
            for agent_data in data["agents"]:
                agent = AgentConfig(**agent_data)
                self._agents[agent.id] = agent

        if "tasks" in data:
            for task_data in data["tasks"]:
                task = TaskConfig(**task_data)
                self._tasks[task.id] = task

        if "crews" in data:
            for crew_data in data["crews"]:
                crew = CrewConfig(**crew_data)
                self._crews[crew.id] = crew

        if "tools" in data:
            for tool_data in data["tools"]:
                tool = ToolConfig(**tool_data)
                self._tools[tool.id] = tool

        if "knowledge_sources" in data:
            for source_data in data["knowledge_sources"]:
                source = KnowledgeSourceConfig(**source_data)
                self._knowledge_sources[source.id] = source

        if "mcp_servers" in data:
            for server_data in data["mcp_servers"]:
                server = MCPServerConfig(**server_data)
                self._mcp_servers[server.id] = server

    def clear_all(self) -> None:
        """Clear all data from the storage."""
        self._agents.clear()
        self._tasks.clear()
        self._crews.clear()
        self._tools.clear()
        self._knowledge_sources.clear()
        self._mcp_servers.clear()
