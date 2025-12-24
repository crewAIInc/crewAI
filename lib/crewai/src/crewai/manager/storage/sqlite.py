"""SQLite storage implementation for CrewManager."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from crewai.manager.models.agent_config import AgentConfig
from crewai.manager.models.crew_config import CrewConfig
from crewai.manager.models.knowledge_config import KnowledgeSourceConfig
from crewai.manager.models.mcp_config import MCPServerConfig
from crewai.manager.models.task_config import TaskConfig
from crewai.manager.models.tool_config import ToolConfig
from crewai.manager.storage.base import BaseStorage


class SQLiteStorage(BaseStorage):
    """SQLite storage backend for persistent database storage.

    Uses SQLite for robust, file-based persistence with support for
    concurrent access and transactions.

    Args:
        db_path: Path to the SQLite database file
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the SQLite storage."""
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def initialize(self) -> None:
        """Initialize the database schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_connection()

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS crews (
                id TEXT PRIMARY KEY,
                name TEXT,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tools (
                id TEXT PRIMARY KEY,
                name TEXT,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_sources (
                id TEXT PRIMARY KEY,
                name TEXT,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS mcp_servers (
                id TEXT PRIMARY KEY,
                name TEXT,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_agents_created ON agents(created_at);
            CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);
            CREATE INDEX IF NOT EXISTS idx_crews_created ON crews(created_at);
            CREATE INDEX IF NOT EXISTS idx_crews_name ON crews(name);
            CREATE INDEX IF NOT EXISTS idx_tools_created ON tools(created_at);
            CREATE INDEX IF NOT EXISTS idx_tools_name ON tools(name);
            CREATE INDEX IF NOT EXISTS idx_knowledge_sources_created ON knowledge_sources(created_at);
            CREATE INDEX IF NOT EXISTS idx_knowledge_sources_name ON knowledge_sources(name);
            CREATE INDEX IF NOT EXISTS idx_mcp_servers_created ON mcp_servers(created_at);
            CREATE INDEX IF NOT EXISTS idx_mcp_servers_name ON mcp_servers(name);
        """)
        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _serialize(self, obj: Any) -> str:
        """Serialize an object to JSON."""
        return json.dumps(obj.model_dump(), default=str)

    def _deserialize_agent(self, data: str) -> AgentConfig:
        """Deserialize JSON to AgentConfig."""
        return AgentConfig(**json.loads(data))

    def _deserialize_task(self, data: str) -> TaskConfig:
        """Deserialize JSON to TaskConfig."""
        return TaskConfig(**json.loads(data))

    def _deserialize_crew(self, data: str) -> CrewConfig:
        """Deserialize JSON to CrewConfig."""
        return CrewConfig(**json.loads(data))

    def _deserialize_tool(self, data: str) -> ToolConfig:
        """Deserialize JSON to ToolConfig."""
        return ToolConfig(**json.loads(data))

    def _deserialize_knowledge_source(self, data: str) -> KnowledgeSourceConfig:
        """Deserialize JSON to KnowledgeSourceConfig."""
        return KnowledgeSourceConfig(**json.loads(data))

    def _deserialize_mcp_server(self, data: str) -> MCPServerConfig:
        """Deserialize JSON to MCPServerConfig."""
        return MCPServerConfig(**json.loads(data))

    # ==================== Agent Operations ====================

    def save_agent(self, agent: AgentConfig) -> str:
        """Save an agent configuration."""
        conn = self._get_connection()
        now = datetime.now().isoformat()

        conn.execute(
            """
            INSERT OR REPLACE INTO agents (id, data, created_at, updated_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM agents WHERE id = ?), ?), ?)
            """,
            (agent.id, self._serialize(agent), agent.id, now, now),
        )
        conn.commit()
        return agent.id

    def get_agent(self, agent_id: str) -> AgentConfig | None:
        """Get an agent by ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT data FROM agents WHERE id = ?", (agent_id,)
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_agent(row["data"])
        return None

    def get_agent_by_role(self, role: str) -> AgentConfig | None:
        """Get an agent by role."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT data FROM agents")
        for row in cursor:
            agent = self._deserialize_agent(row["data"])
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
        conn = self._get_connection()

        # Get total count
        cursor = conn.execute("SELECT COUNT(*) FROM agents")
        total = cursor.fetchone()[0]

        # Get paginated results
        cursor = conn.execute(
            "SELECT data FROM agents ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )

        agents = []
        for row in cursor:
            agent = self._deserialize_agent(row["data"])
            if tags:
                if any(tag in agent.tags for tag in tags):
                    agents.append(agent)
            else:
                agents.append(agent)

        return agents, total

    def update_agent(
        self, agent_id: str, updates: dict[str, Any]
    ) -> AgentConfig | None:
        """Update an agent configuration."""
        agent = self.get_agent(agent_id)
        if not agent:
            return None

        for key, value in updates.items():
            if hasattr(agent, key):
                setattr(agent, key, value)

        agent.update_timestamp()
        self.save_agent(agent)
        return agent

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent by ID."""
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        conn.commit()
        return cursor.rowcount > 0

    # ==================== Task Operations ====================

    def save_task(self, task: TaskConfig) -> str:
        """Save a task configuration."""
        conn = self._get_connection()
        now = datetime.now().isoformat()

        conn.execute(
            """
            INSERT OR REPLACE INTO tasks (id, data, created_at, updated_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM tasks WHERE id = ?), ?), ?)
            """,
            (task.id, self._serialize(task), task.id, now, now),
        )
        conn.commit()
        return task.id

    def get_task(self, task_id: str) -> TaskConfig | None:
        """Get a task by ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT data FROM tasks WHERE id = ?", (task_id,)
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_task(row["data"])
        return None

    def list_tasks(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[TaskConfig], int]:
        """List tasks with pagination."""
        conn = self._get_connection()

        cursor = conn.execute("SELECT COUNT(*) FROM tasks")
        total = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT data FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )

        tasks = []
        for row in cursor:
            task = self._deserialize_task(row["data"])
            if tags:
                if any(tag in task.tags for tag in tags):
                    tasks.append(task)
            else:
                tasks.append(task)

        return tasks, total

    def update_task(
        self, task_id: str, updates: dict[str, Any]
    ) -> TaskConfig | None:
        """Update a task configuration."""
        task = self.get_task(task_id)
        if not task:
            return None

        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)

        task.update_timestamp()
        self.save_task(task)
        return task

    def delete_task(self, task_id: str) -> bool:
        """Delete a task by ID."""
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        return cursor.rowcount > 0

    # ==================== Crew Operations ====================

    def save_crew(self, crew: CrewConfig) -> str:
        """Save a crew configuration."""
        conn = self._get_connection()
        now = datetime.now().isoformat()

        conn.execute(
            """
            INSERT OR REPLACE INTO crews (id, name, data, created_at, updated_at)
            VALUES (?, ?, ?, COALESCE((SELECT created_at FROM crews WHERE id = ?), ?), ?)
            """,
            (crew.id, crew.name, self._serialize(crew), crew.id, now, now),
        )
        conn.commit()
        return crew.id

    def get_crew(self, crew_id: str) -> CrewConfig | None:
        """Get a crew by ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT data FROM crews WHERE id = ?", (crew_id,)
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_crew(row["data"])
        return None

    def get_crew_by_name(self, name: str) -> CrewConfig | None:
        """Get a crew by name."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT data FROM crews WHERE name = ?", (name,)
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_crew(row["data"])
        return None

    def list_crews(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[CrewConfig], int]:
        """List crews with pagination."""
        conn = self._get_connection()

        cursor = conn.execute("SELECT COUNT(*) FROM crews")
        total = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT data FROM crews ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )

        crews = []
        for row in cursor:
            crew = self._deserialize_crew(row["data"])
            if tags:
                if any(tag in crew.tags for tag in tags):
                    crews.append(crew)
            else:
                crews.append(crew)

        return crews, total

    def update_crew(
        self, crew_id: str, updates: dict[str, Any]
    ) -> CrewConfig | None:
        """Update a crew configuration."""
        crew = self.get_crew(crew_id)
        if not crew:
            return None

        for key, value in updates.items():
            if hasattr(crew, key):
                setattr(crew, key, value)

        crew.update_timestamp()
        self.save_crew(crew)
        return crew

    def delete_crew(self, crew_id: str) -> bool:
        """Delete a crew by ID."""
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM crews WHERE id = ?", (crew_id,))
        conn.commit()
        return cursor.rowcount > 0

    # ==================== Tool Operations ====================

    def save_tool(self, tool: ToolConfig) -> str:
        """Save a tool configuration."""
        conn = self._get_connection()
        now = datetime.now().isoformat()

        conn.execute(
            """
            INSERT OR REPLACE INTO tools (id, name, data, created_at, updated_at)
            VALUES (?, ?, ?, COALESCE((SELECT created_at FROM tools WHERE id = ?), ?), ?)
            """,
            (tool.id, tool.name, self._serialize(tool), tool.id, now, now),
        )
        conn.commit()
        return tool.id

    def get_tool(self, tool_id: str) -> ToolConfig | None:
        """Get a tool by ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT data FROM tools WHERE id = ?", (tool_id,)
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_tool(row["data"])
        return None

    def get_tool_by_name(self, name: str) -> ToolConfig | None:
        """Get a tool by name."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT data FROM tools WHERE name = ?", (name,)
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_tool(row["data"])
        return None

    def list_tools(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[ToolConfig], int]:
        """List tools with pagination."""
        conn = self._get_connection()

        cursor = conn.execute("SELECT COUNT(*) FROM tools")
        total = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT data FROM tools ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )

        tools = []
        for row in cursor:
            tool = self._deserialize_tool(row["data"])
            if tags:
                if any(tag in tool.tags for tag in tags):
                    tools.append(tool)
            else:
                tools.append(tool)

        return tools, total

    def update_tool(
        self, tool_id: str, updates: dict[str, Any]
    ) -> ToolConfig | None:
        """Update a tool configuration."""
        tool = self.get_tool(tool_id)
        if not tool:
            return None

        for key, value in updates.items():
            if hasattr(tool, key):
                setattr(tool, key, value)

        tool.update_timestamp()
        self.save_tool(tool)
        return tool

    def delete_tool(self, tool_id: str) -> bool:
        """Delete a tool by ID."""
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM tools WHERE id = ?", (tool_id,))
        conn.commit()
        return cursor.rowcount > 0

    # ==================== Knowledge Source Operations ====================

    def save_knowledge_source(self, source: KnowledgeSourceConfig) -> str:
        """Save a knowledge source configuration."""
        conn = self._get_connection()
        now = datetime.now().isoformat()

        conn.execute(
            """
            INSERT OR REPLACE INTO knowledge_sources (id, name, data, created_at, updated_at)
            VALUES (?, ?, ?, COALESCE((SELECT created_at FROM knowledge_sources WHERE id = ?), ?), ?)
            """,
            (source.id, source.name, self._serialize(source), source.id, now, now),
        )
        conn.commit()
        return source.id

    def get_knowledge_source(self, source_id: str) -> KnowledgeSourceConfig | None:
        """Get a knowledge source by ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT data FROM knowledge_sources WHERE id = ?", (source_id,)
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_knowledge_source(row["data"])
        return None

    def get_knowledge_source_by_name(self, name: str) -> KnowledgeSourceConfig | None:
        """Get a knowledge source by name."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT data FROM knowledge_sources WHERE name = ?", (name,)
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_knowledge_source(row["data"])
        return None

    def list_knowledge_sources(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[KnowledgeSourceConfig], int]:
        """List knowledge sources with pagination."""
        conn = self._get_connection()

        cursor = conn.execute("SELECT COUNT(*) FROM knowledge_sources")
        total = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT data FROM knowledge_sources ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )

        sources = []
        for row in cursor:
            source = self._deserialize_knowledge_source(row["data"])
            if tags:
                if any(tag in source.tags for tag in tags):
                    sources.append(source)
            else:
                sources.append(source)

        return sources, total

    def update_knowledge_source(
        self, source_id: str, updates: dict[str, Any]
    ) -> KnowledgeSourceConfig | None:
        """Update a knowledge source configuration."""
        source = self.get_knowledge_source(source_id)
        if not source:
            return None

        for key, value in updates.items():
            if hasattr(source, key):
                setattr(source, key, value)

        source.update_timestamp()
        self.save_knowledge_source(source)
        return source

    def delete_knowledge_source(self, source_id: str) -> bool:
        """Delete a knowledge source by ID."""
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM knowledge_sources WHERE id = ?", (source_id,))
        conn.commit()
        return cursor.rowcount > 0

    # ==================== MCP Server Operations ====================

    def save_mcp_server(self, server: MCPServerConfig) -> str:
        """Save an MCP server configuration."""
        conn = self._get_connection()
        now = datetime.now().isoformat()

        conn.execute(
            """
            INSERT OR REPLACE INTO mcp_servers (id, name, data, created_at, updated_at)
            VALUES (?, ?, ?, COALESCE((SELECT created_at FROM mcp_servers WHERE id = ?), ?), ?)
            """,
            (server.id, server.name, self._serialize(server), server.id, now, now),
        )
        conn.commit()
        return server.id

    def get_mcp_server(self, server_id: str) -> MCPServerConfig | None:
        """Get an MCP server by ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT data FROM mcp_servers WHERE id = ?", (server_id,)
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_mcp_server(row["data"])
        return None

    def get_mcp_server_by_name(self, name: str) -> MCPServerConfig | None:
        """Get an MCP server by name."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT data FROM mcp_servers WHERE name = ?", (name,)
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_mcp_server(row["data"])
        return None

    def list_mcp_servers(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> tuple[list[MCPServerConfig], int]:
        """List MCP servers with pagination."""
        conn = self._get_connection()

        cursor = conn.execute("SELECT COUNT(*) FROM mcp_servers")
        total = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT data FROM mcp_servers ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )

        servers = []
        for row in cursor:
            server = self._deserialize_mcp_server(row["data"])
            if tags:
                if any(tag in server.tags for tag in tags):
                    servers.append(server)
            else:
                servers.append(server)

        return servers, total

    def update_mcp_server(
        self, server_id: str, updates: dict[str, Any]
    ) -> MCPServerConfig | None:
        """Update an MCP server configuration."""
        server = self.get_mcp_server(server_id)
        if not server:
            return None

        for key, value in updates.items():
            if hasattr(server, key):
                setattr(server, key, value)

        server.update_timestamp()
        self.save_mcp_server(server)
        return server

    def delete_mcp_server(self, server_id: str) -> bool:
        """Delete an MCP server by ID."""
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM mcp_servers WHERE id = ?", (server_id,))
        conn.commit()
        return cursor.rowcount > 0

    # ==================== Utility Operations ====================

    def export_all(self) -> dict[str, Any]:
        """Export all data from the storage."""
        conn = self._get_connection()

        agents = []
        for row in conn.execute("SELECT data FROM agents"):
            agents.append(json.loads(row["data"]))

        tasks = []
        for row in conn.execute("SELECT data FROM tasks"):
            tasks.append(json.loads(row["data"]))

        crews = []
        for row in conn.execute("SELECT data FROM crews"):
            crews.append(json.loads(row["data"]))

        tools = []
        for row in conn.execute("SELECT data FROM tools"):
            tools.append(json.loads(row["data"]))

        knowledge_sources = []
        for row in conn.execute("SELECT data FROM knowledge_sources"):
            knowledge_sources.append(json.loads(row["data"]))

        mcp_servers = []
        for row in conn.execute("SELECT data FROM mcp_servers"):
            mcp_servers.append(json.loads(row["data"]))

        return {
            "agents": agents,
            "tasks": tasks,
            "crews": crews,
            "tools": tools,
            "knowledge_sources": knowledge_sources,
            "mcp_servers": mcp_servers,
        }

    def import_all(self, data: dict[str, Any]) -> None:
        """Import data into the storage."""
        if "agents" in data:
            for agent_data in data["agents"]:
                agent = AgentConfig(**agent_data)
                self.save_agent(agent)

        if "tasks" in data:
            for task_data in data["tasks"]:
                task = TaskConfig(**task_data)
                self.save_task(task)

        if "crews" in data:
            for crew_data in data["crews"]:
                crew = CrewConfig(**crew_data)
                self.save_crew(crew)

        if "tools" in data:
            for tool_data in data["tools"]:
                tool = ToolConfig(**tool_data)
                self.save_tool(tool)

        if "knowledge_sources" in data:
            for source_data in data["knowledge_sources"]:
                source = KnowledgeSourceConfig(**source_data)
                self.save_knowledge_source(source)

        if "mcp_servers" in data:
            for server_data in data["mcp_servers"]:
                server = MCPServerConfig(**server_data)
                self.save_mcp_server(server)

    def clear_all(self) -> None:
        """Clear all data from the storage."""
        conn = self._get_connection()
        conn.executescript("""
            DELETE FROM agents;
            DELETE FROM tasks;
            DELETE FROM crews;
            DELETE FROM tools;
            DELETE FROM knowledge_sources;
            DELETE FROM mcp_servers;
        """)
        conn.commit()
