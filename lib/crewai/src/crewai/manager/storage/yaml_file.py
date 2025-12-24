"""YAML file storage implementation for CrewManager."""

from pathlib import Path
from typing import Any

from crewai.manager.models.agent_config import AgentConfig
from crewai.manager.models.crew_config import CrewConfig
from crewai.manager.models.knowledge_config import KnowledgeSourceConfig
from crewai.manager.models.mcp_config import MCPServerConfig
from crewai.manager.models.task_config import TaskConfig
from crewai.manager.models.tool_config import ToolConfig
from crewai.manager.storage.memory import InMemoryStorage


class YAMLFileStorage(InMemoryStorage):
    """YAML file storage backend for persistent storage.

    Extends InMemoryStorage and adds file persistence.
    Data is saved to a YAML file on disk, which is more human-readable
    than JSON.

    Args:
        file_path: Path to the YAML file
        auto_save: Whether to save after each modification (default: True)

    Requires:
        PyYAML package (pip install pyyaml)
    """

    def __init__(self, file_path: str | Path, auto_save: bool = True) -> None:
        """Initialize the YAML file storage."""
        super().__init__()
        self._file_path = Path(file_path)
        self._auto_save = auto_save
        self._yaml: Any = None

    def _get_yaml(self) -> Any:
        """Get the yaml module, importing it lazily."""
        if self._yaml is None:
            try:
                import yaml
                self._yaml = yaml
            except ImportError as e:
                raise ImportError(
                    "PyYAML is required for YAMLFileStorage. "
                    "Install it with: pip install pyyaml"
                ) from e
        return self._yaml

    def initialize(self) -> None:
        """Initialize the storage by loading existing data."""
        super().initialize()
        if self._file_path.exists():
            self._load()

    def _load(self) -> None:
        """Load data from the YAML file."""
        yaml = self._get_yaml()
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.import_all(data)
        except (yaml.YAMLError, OSError) as e:
            raise RuntimeError(f"Failed to load YAML file: {e}") from e

    def _save(self) -> None:
        """Save data to the YAML file."""
        yaml = self._get_yaml()
        try:
            # Ensure parent directory exists
            self._file_path.parent.mkdir(parents=True, exist_ok=True)

            data = self.export_all()

            # Convert datetime objects to strings
            data = self._serialize_dates(data)

            # Write to temp file first for atomic save
            temp_path = self._file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

            # Rename temp file to target
            temp_path.rename(self._file_path)
        except OSError as e:
            raise RuntimeError(f"Failed to save YAML file: {e}") from e

    def _serialize_dates(self, data: Any) -> Any:
        """Recursively convert datetime objects to ISO strings."""
        from datetime import datetime

        if isinstance(data, dict):
            return {k: self._serialize_dates(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_dates(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        return data

    def _maybe_save(self) -> None:
        """Save if auto_save is enabled."""
        if self._auto_save:
            self._save()

    def save(self) -> None:
        """Manually save data to file."""
        self._save()

    # Override methods to add auto-save

    def save_agent(self, agent: AgentConfig) -> str:
        """Save an agent configuration."""
        result = super().save_agent(agent)
        self._maybe_save()
        return result

    def update_agent(
        self, agent_id: str, updates: dict[str, Any]
    ) -> AgentConfig | None:
        """Update an agent configuration."""
        result = super().update_agent(agent_id, updates)
        if result:
            self._maybe_save()
        return result

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent by ID."""
        result = super().delete_agent(agent_id)
        if result:
            self._maybe_save()
        return result

    def save_task(self, task: TaskConfig) -> str:
        """Save a task configuration."""
        result = super().save_task(task)
        self._maybe_save()
        return result

    def update_task(
        self, task_id: str, updates: dict[str, Any]
    ) -> TaskConfig | None:
        """Update a task configuration."""
        result = super().update_task(task_id, updates)
        if result:
            self._maybe_save()
        return result

    def delete_task(self, task_id: str) -> bool:
        """Delete a task by ID."""
        result = super().delete_task(task_id)
        if result:
            self._maybe_save()
        return result

    def save_crew(self, crew: CrewConfig) -> str:
        """Save a crew configuration."""
        result = super().save_crew(crew)
        self._maybe_save()
        return result

    def update_crew(
        self, crew_id: str, updates: dict[str, Any]
    ) -> CrewConfig | None:
        """Update a crew configuration."""
        result = super().update_crew(crew_id, updates)
        if result:
            self._maybe_save()
        return result

    def delete_crew(self, crew_id: str) -> bool:
        """Delete a crew by ID."""
        result = super().delete_crew(crew_id)
        if result:
            self._maybe_save()
        return result

    def save_tool(self, tool: ToolConfig) -> str:
        """Save a tool configuration."""
        result = super().save_tool(tool)
        self._maybe_save()
        return result

    def update_tool(
        self, tool_id: str, updates: dict[str, Any]
    ) -> ToolConfig | None:
        """Update a tool configuration."""
        result = super().update_tool(tool_id, updates)
        if result:
            self._maybe_save()
        return result

    def delete_tool(self, tool_id: str) -> bool:
        """Delete a tool by ID."""
        result = super().delete_tool(tool_id)
        if result:
            self._maybe_save()
        return result

    # Knowledge Source operations with auto-save

    def save_knowledge_source(self, source: KnowledgeSourceConfig) -> str:
        """Save a knowledge source configuration."""
        result = super().save_knowledge_source(source)
        self._maybe_save()
        return result

    def update_knowledge_source(
        self, source_id: str, updates: dict[str, Any]
    ) -> KnowledgeSourceConfig | None:
        """Update a knowledge source configuration."""
        result = super().update_knowledge_source(source_id, updates)
        if result:
            self._maybe_save()
        return result

    def delete_knowledge_source(self, source_id: str) -> bool:
        """Delete a knowledge source by ID."""
        result = super().delete_knowledge_source(source_id)
        if result:
            self._maybe_save()
        return result

    # MCP Server operations with auto-save

    def save_mcp_server(self, server: MCPServerConfig) -> str:
        """Save an MCP server configuration."""
        result = super().save_mcp_server(server)
        self._maybe_save()
        return result

    def update_mcp_server(
        self, server_id: str, updates: dict[str, Any]
    ) -> MCPServerConfig | None:
        """Update an MCP server configuration."""
        result = super().update_mcp_server(server_id, updates)
        if result:
            self._maybe_save()
        return result

    def delete_mcp_server(self, server_id: str) -> bool:
        """Delete an MCP server by ID."""
        result = super().delete_mcp_server(server_id)
        if result:
            self._maybe_save()
        return result

    def import_all(self, data: dict[str, Any]) -> None:
        """Import data into the storage."""
        super().import_all(data)
        self._maybe_save()

    def clear_all(self) -> None:
        """Clear all data from the storage."""
        super().clear_all()
        self._maybe_save()
