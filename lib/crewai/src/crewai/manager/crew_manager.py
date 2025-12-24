"""CrewManager - Programmatic CRUD API for CrewAI entities."""

from __future__ import annotations

import importlib
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Callable

from crewai.manager.models.agent_config import AgentConfig
from crewai.manager.models.crew_config import CrewConfig
from crewai.manager.models.knowledge_config import KnowledgeSourceConfig
from crewai.manager.models.mcp_config import MCPServerConfig
from crewai.manager.models.responses import (
    ExecutionResult,
    ListResult,
    OperationResult,
)
from crewai.manager.models.task_config import TaskConfig
from crewai.manager.models.tool_config import ToolConfig
from crewai.manager.storage.base import BaseStorage
from crewai.manager.storage.memory import InMemoryStorage


class CrewManager:
    """Programmatic CRUD API for managing CrewAI agents, tasks, crews, and tools.

    CrewManager provides a Python library interface for creating, reading,
    updating, and deleting CrewAI configurations. It supports multiple
    storage backends and can convert configurations to runtime objects.

    Example:
        ```python
        from crewai.manager import CrewManager, AgentConfig, TaskConfig, CrewConfig
        from crewai.manager.storage import SQLiteStorage

        # Initialize with SQLite storage
        manager = CrewManager(storage=SQLiteStorage("my_crews.db"))

        # Create an agent
        agent = AgentConfig(
            role="Researcher",
            goal="Find information",
            backstory="Expert researcher",
            llm="gpt-4",
        )
        manager.create_agent(agent)

        # Create a task
        task = TaskConfig(
            description="Research {topic}",
            agent_id=agent.id,
        )
        manager.create_task(task)

        # Create a crew
        crew = CrewConfig(
            name="Research Crew",
            agent_ids=[agent.id],
            task_ids=[task.id],
        )
        manager.create_crew(crew)

        # Execute the crew
        result = manager.execute_crew(crew.id, inputs={"topic": "AI"})
        print(result.raw_output)
        ```

    Args:
        storage: Storage backend to use (defaults to InMemoryStorage)
    """

    def __init__(self, storage: BaseStorage | None = None) -> None:
        """Initialize the CrewManager.

        Args:
            storage: Storage backend to use. If None, uses InMemoryStorage.
        """
        self._storage = storage or InMemoryStorage()
        self._storage.initialize()

        # Runtime tool instances (for custom tools)
        self._tool_instances: dict[str, Any] = {}

        # Callback registries
        self._guardrail_functions: dict[str, Callable[..., Any]] = {}
        self._callback_functions: dict[str, Callable[..., Any]] = {}

    # ==================== Agent Operations ====================

    def create_agent(self, config: AgentConfig) -> OperationResult[AgentConfig]:
        """Create a new agent configuration.

        Args:
            config: The agent configuration to create

        Returns:
            OperationResult containing the created agent or error
        """
        try:
            self._storage.save_agent(config)
            return OperationResult.ok(config)
        except Exception as e:
            return OperationResult.fail(str(e), "AGENT_CREATE_ERROR")

    def get_agent(self, agent_id: str) -> OperationResult[AgentConfig]:
        """Get an agent by ID.

        Args:
            agent_id: The ID of the agent to retrieve

        Returns:
            OperationResult containing the agent or error
        """
        agent = self._storage.get_agent(agent_id)
        if agent:
            return OperationResult.ok(agent)
        return OperationResult.fail(f"Agent not found: {agent_id}", "AGENT_NOT_FOUND")

    def get_agent_by_role(self, role: str) -> OperationResult[AgentConfig]:
        """Get an agent by role.

        Args:
            role: The role of the agent to find

        Returns:
            OperationResult containing the agent or error
        """
        agent = self._storage.get_agent_by_role(role)
        if agent:
            return OperationResult.ok(agent)
        return OperationResult.fail(
            f"Agent with role not found: {role}", "AGENT_NOT_FOUND"
        )

    def list_agents(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> ListResult[AgentConfig]:
        """List agents with pagination.

        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
            tags: Optional filter by tags

        Returns:
            ListResult containing the agents
        """
        agents, total = self._storage.list_agents(offset, limit, tags)
        return ListResult.from_list(agents, total, offset, limit)

    def update_agent(
        self, agent_id: str, updates: dict[str, Any]
    ) -> OperationResult[AgentConfig]:
        """Update an agent configuration.

        Args:
            agent_id: The ID of the agent to update
            updates: Dictionary of fields to update

        Returns:
            OperationResult containing the updated agent or error
        """
        agent = self._storage.update_agent(agent_id, updates)
        if agent:
            return OperationResult.ok(agent)
        return OperationResult.fail(f"Agent not found: {agent_id}", "AGENT_NOT_FOUND")

    def delete_agent(self, agent_id: str) -> OperationResult[bool]:
        """Delete an agent by ID.

        Args:
            agent_id: The ID of the agent to delete

        Returns:
            OperationResult indicating success or failure
        """
        if self._storage.delete_agent(agent_id):
            return OperationResult.ok(True)
        return OperationResult.fail(f"Agent not found: {agent_id}", "AGENT_NOT_FOUND")

    # ==================== Task Operations ====================

    def create_task(self, config: TaskConfig) -> OperationResult[TaskConfig]:
        """Create a new task configuration.

        Args:
            config: The task configuration to create

        Returns:
            OperationResult containing the created task or error
        """
        try:
            self._storage.save_task(config)
            return OperationResult.ok(config)
        except Exception as e:
            return OperationResult.fail(str(e), "TASK_CREATE_ERROR")

    def get_task(self, task_id: str) -> OperationResult[TaskConfig]:
        """Get a task by ID.

        Args:
            task_id: The ID of the task to retrieve

        Returns:
            OperationResult containing the task or error
        """
        task = self._storage.get_task(task_id)
        if task:
            return OperationResult.ok(task)
        return OperationResult.fail(f"Task not found: {task_id}", "TASK_NOT_FOUND")

    def list_tasks(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> ListResult[TaskConfig]:
        """List tasks with pagination.

        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
            tags: Optional filter by tags

        Returns:
            ListResult containing the tasks
        """
        tasks, total = self._storage.list_tasks(offset, limit, tags)
        return ListResult.from_list(tasks, total, offset, limit)

    def update_task(
        self, task_id: str, updates: dict[str, Any]
    ) -> OperationResult[TaskConfig]:
        """Update a task configuration.

        Args:
            task_id: The ID of the task to update
            updates: Dictionary of fields to update

        Returns:
            OperationResult containing the updated task or error
        """
        task = self._storage.update_task(task_id, updates)
        if task:
            return OperationResult.ok(task)
        return OperationResult.fail(f"Task not found: {task_id}", "TASK_NOT_FOUND")

    def delete_task(self, task_id: str) -> OperationResult[bool]:
        """Delete a task by ID.

        Args:
            task_id: The ID of the task to delete

        Returns:
            OperationResult indicating success or failure
        """
        if self._storage.delete_task(task_id):
            return OperationResult.ok(True)
        return OperationResult.fail(f"Task not found: {task_id}", "TASK_NOT_FOUND")

    # ==================== Crew Operations ====================

    def create_crew(self, config: CrewConfig) -> OperationResult[CrewConfig]:
        """Create a new crew configuration.

        Args:
            config: The crew configuration to create

        Returns:
            OperationResult containing the created crew or error
        """
        try:
            self._storage.save_crew(config)
            return OperationResult.ok(config)
        except Exception as e:
            return OperationResult.fail(str(e), "CREW_CREATE_ERROR")

    def get_crew(self, crew_id: str) -> OperationResult[CrewConfig]:
        """Get a crew by ID.

        Args:
            crew_id: The ID of the crew to retrieve

        Returns:
            OperationResult containing the crew or error
        """
        crew = self._storage.get_crew(crew_id)
        if crew:
            return OperationResult.ok(crew)
        return OperationResult.fail(f"Crew not found: {crew_id}", "CREW_NOT_FOUND")

    def get_crew_by_name(self, name: str) -> OperationResult[CrewConfig]:
        """Get a crew by name.

        Args:
            name: The name of the crew to find

        Returns:
            OperationResult containing the crew or error
        """
        crew = self._storage.get_crew_by_name(name)
        if crew:
            return OperationResult.ok(crew)
        return OperationResult.fail(f"Crew not found: {name}", "CREW_NOT_FOUND")

    def list_crews(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> ListResult[CrewConfig]:
        """List crews with pagination.

        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
            tags: Optional filter by tags

        Returns:
            ListResult containing the crews
        """
        crews, total = self._storage.list_crews(offset, limit, tags)
        return ListResult.from_list(crews, total, offset, limit)

    def update_crew(
        self, crew_id: str, updates: dict[str, Any]
    ) -> OperationResult[CrewConfig]:
        """Update a crew configuration.

        Args:
            crew_id: The ID of the crew to update
            updates: Dictionary of fields to update

        Returns:
            OperationResult containing the updated crew or error
        """
        crew = self._storage.update_crew(crew_id, updates)
        if crew:
            return OperationResult.ok(crew)
        return OperationResult.fail(f"Crew not found: {crew_id}", "CREW_NOT_FOUND")

    def delete_crew(self, crew_id: str) -> OperationResult[bool]:
        """Delete a crew by ID.

        Args:
            crew_id: The ID of the crew to delete

        Returns:
            OperationResult indicating success or failure
        """
        if self._storage.delete_crew(crew_id):
            return OperationResult.ok(True)
        return OperationResult.fail(f"Crew not found: {crew_id}", "CREW_NOT_FOUND")

    # ==================== Tool Operations ====================

    def create_tool(self, config: ToolConfig) -> OperationResult[ToolConfig]:
        """Create a new tool configuration.

        Args:
            config: The tool configuration to create

        Returns:
            OperationResult containing the created tool or error
        """
        try:
            self._storage.save_tool(config)
            return OperationResult.ok(config)
        except Exception as e:
            return OperationResult.fail(str(e), "TOOL_CREATE_ERROR")

    def register_tool_instance(self, tool_id: str, tool: Any) -> None:
        """Register a runtime tool instance.

        For custom tools that can't be serialized, register the actual
        tool instance which will be used during crew execution.

        Args:
            tool_id: The ID of the tool config
            tool: The actual tool instance
        """
        self._tool_instances[tool_id] = tool

    def get_tool(self, tool_id: str) -> OperationResult[ToolConfig]:
        """Get a tool by ID.

        Args:
            tool_id: The ID of the tool to retrieve

        Returns:
            OperationResult containing the tool or error
        """
        tool = self._storage.get_tool(tool_id)
        if tool:
            return OperationResult.ok(tool)
        return OperationResult.fail(f"Tool not found: {tool_id}", "TOOL_NOT_FOUND")

    def list_tools(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> ListResult[ToolConfig]:
        """List tools with pagination.

        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
            tags: Optional filter by tags

        Returns:
            ListResult containing the tools
        """
        tools, total = self._storage.list_tools(offset, limit, tags)
        return ListResult.from_list(tools, total, offset, limit)

    def delete_tool(self, tool_id: str) -> OperationResult[bool]:
        """Delete a tool by ID.

        Args:
            tool_id: The ID of the tool to delete

        Returns:
            OperationResult indicating success or failure
        """
        if self._storage.delete_tool(tool_id):
            self._tool_instances.pop(tool_id, None)
            return OperationResult.ok(True)
        return OperationResult.fail(f"Tool not found: {tool_id}", "TOOL_NOT_FOUND")

    def update_tool(
        self, tool_id: str, updates: dict[str, Any]
    ) -> OperationResult[ToolConfig]:
        """Update a tool configuration.

        Args:
            tool_id: The ID of the tool to update
            updates: Dictionary of fields to update

        Returns:
            OperationResult containing the updated tool or error
        """
        tool = self._storage.update_tool(tool_id, updates)
        if tool:
            return OperationResult.ok(tool)
        return OperationResult.fail(f"Tool not found: {tool_id}", "TOOL_NOT_FOUND")

    # ==================== Knowledge Source Operations ====================

    def create_knowledge_source(
        self, config: KnowledgeSourceConfig
    ) -> OperationResult[KnowledgeSourceConfig]:
        """Create a new knowledge source configuration.

        Args:
            config: The knowledge source configuration to create

        Returns:
            OperationResult containing the created knowledge source or error
        """
        try:
            self._storage.save_knowledge_source(config)
            return OperationResult.ok(config)
        except Exception as e:
            return OperationResult.fail(str(e), "KNOWLEDGE_SOURCE_CREATE_ERROR")

    def get_knowledge_source(
        self, source_id: str
    ) -> OperationResult[KnowledgeSourceConfig]:
        """Get a knowledge source by ID.

        Args:
            source_id: The ID of the knowledge source to retrieve

        Returns:
            OperationResult containing the knowledge source or error
        """
        source = self._storage.get_knowledge_source(source_id)
        if source:
            return OperationResult.ok(source)
        return OperationResult.fail(
            f"Knowledge source not found: {source_id}", "KNOWLEDGE_SOURCE_NOT_FOUND"
        )

    def get_knowledge_source_by_name(
        self, name: str
    ) -> OperationResult[KnowledgeSourceConfig]:
        """Get a knowledge source by name.

        Args:
            name: The name of the knowledge source to find

        Returns:
            OperationResult containing the knowledge source or error
        """
        source = self._storage.get_knowledge_source_by_name(name)
        if source:
            return OperationResult.ok(source)
        return OperationResult.fail(
            f"Knowledge source not found: {name}", "KNOWLEDGE_SOURCE_NOT_FOUND"
        )

    def list_knowledge_sources(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> ListResult[KnowledgeSourceConfig]:
        """List knowledge sources with pagination.

        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
            tags: Optional filter by tags

        Returns:
            ListResult containing the knowledge sources
        """
        sources, total = self._storage.list_knowledge_sources(offset, limit, tags)
        return ListResult.from_list(sources, total, offset, limit)

    def update_knowledge_source(
        self, source_id: str, updates: dict[str, Any]
    ) -> OperationResult[KnowledgeSourceConfig]:
        """Update a knowledge source configuration.

        Args:
            source_id: The ID of the knowledge source to update
            updates: Dictionary of fields to update

        Returns:
            OperationResult containing the updated knowledge source or error
        """
        source = self._storage.update_knowledge_source(source_id, updates)
        if source:
            return OperationResult.ok(source)
        return OperationResult.fail(
            f"Knowledge source not found: {source_id}", "KNOWLEDGE_SOURCE_NOT_FOUND"
        )

    def delete_knowledge_source(self, source_id: str) -> OperationResult[bool]:
        """Delete a knowledge source by ID.

        Args:
            source_id: The ID of the knowledge source to delete

        Returns:
            OperationResult indicating success or failure
        """
        if self._storage.delete_knowledge_source(source_id):
            return OperationResult.ok(True)
        return OperationResult.fail(
            f"Knowledge source not found: {source_id}", "KNOWLEDGE_SOURCE_NOT_FOUND"
        )

    # ==================== MCP Server Operations ====================

    def create_mcp_server(
        self, config: MCPServerConfig
    ) -> OperationResult[MCPServerConfig]:
        """Create a new MCP server configuration.

        Args:
            config: The MCP server configuration to create

        Returns:
            OperationResult containing the created MCP server or error
        """
        try:
            self._storage.save_mcp_server(config)
            return OperationResult.ok(config)
        except Exception as e:
            return OperationResult.fail(str(e), "MCP_SERVER_CREATE_ERROR")

    def get_mcp_server(self, server_id: str) -> OperationResult[MCPServerConfig]:
        """Get an MCP server by ID.

        Args:
            server_id: The ID of the MCP server to retrieve

        Returns:
            OperationResult containing the MCP server or error
        """
        server = self._storage.get_mcp_server(server_id)
        if server:
            return OperationResult.ok(server)
        return OperationResult.fail(
            f"MCP server not found: {server_id}", "MCP_SERVER_NOT_FOUND"
        )

    def get_mcp_server_by_name(self, name: str) -> OperationResult[MCPServerConfig]:
        """Get an MCP server by name.

        Args:
            name: The name of the MCP server to find

        Returns:
            OperationResult containing the MCP server or error
        """
        server = self._storage.get_mcp_server_by_name(name)
        if server:
            return OperationResult.ok(server)
        return OperationResult.fail(
            f"MCP server not found: {name}", "MCP_SERVER_NOT_FOUND"
        )

    def list_mcp_servers(
        self,
        offset: int = 0,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> ListResult[MCPServerConfig]:
        """List MCP servers with pagination.

        Args:
            offset: Number of items to skip
            limit: Maximum number of items to return
            tags: Optional filter by tags

        Returns:
            ListResult containing the MCP servers
        """
        servers, total = self._storage.list_mcp_servers(offset, limit, tags)
        return ListResult.from_list(servers, total, offset, limit)

    def update_mcp_server(
        self, server_id: str, updates: dict[str, Any]
    ) -> OperationResult[MCPServerConfig]:
        """Update an MCP server configuration.

        Args:
            server_id: The ID of the MCP server to update
            updates: Dictionary of fields to update

        Returns:
            OperationResult containing the updated MCP server or error
        """
        server = self._storage.update_mcp_server(server_id, updates)
        if server:
            return OperationResult.ok(server)
        return OperationResult.fail(
            f"MCP server not found: {server_id}", "MCP_SERVER_NOT_FOUND"
        )

    def delete_mcp_server(self, server_id: str) -> OperationResult[bool]:
        """Delete an MCP server by ID.

        Args:
            server_id: The ID of the MCP server to delete

        Returns:
            OperationResult indicating success or failure
        """
        if self._storage.delete_mcp_server(server_id):
            return OperationResult.ok(True)
        return OperationResult.fail(
            f"MCP server not found: {server_id}", "MCP_SERVER_NOT_FOUND"
        )

    # ==================== Callback Registration ====================

    def register_guardrail(self, name: str, func: Callable[..., Any]) -> None:
        """Register a guardrail function.

        Args:
            name: Name to reference this guardrail
            func: The guardrail function
        """
        self._guardrail_functions[name] = func

    def register_callback(self, name: str, func: Callable[..., Any]) -> None:
        """Register a callback function.

        Args:
            name: Name to reference this callback
            func: The callback function
        """
        self._callback_functions[name] = func

    # ==================== Build Methods ====================

    def build_tool(self, config: ToolConfig) -> Any:
        """Build a runtime tool from configuration.

        Args:
            config: The tool configuration

        Returns:
            The runtime tool instance
        """
        # Check for registered instance first
        if config.id in self._tool_instances:
            return self._tool_instances[config.id]

        # Try to import and instantiate the tool
        if config.tool_type == "crewai_tools" and config.class_name:
            try:
                module_path = config.module_path or "crewai_tools"
                module = importlib.import_module(module_path)
                tool_class = getattr(module, config.class_name)
                return tool_class(**config.init_kwargs)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to import tool {config.class_name}: {e}")

        raise ValueError(f"Cannot build tool: {config.name}")

    def build_agent(self, config: AgentConfig) -> Any:
        """Build a runtime Agent from configuration.

        Args:
            config: The agent configuration

        Returns:
            A CrewAI Agent instance
        """
        from crewai.agent import Agent

        # Build tools
        tools = []
        for tool_id in config.tool_ids:
            tool_config = self._storage.get_tool(tool_id)
            if tool_config:
                try:
                    tools.append(self.build_tool(tool_config))
                except Exception:
                    pass  # Skip tools that fail to build

        # Build agent kwargs
        kwargs: dict[str, Any] = {
            "role": config.role,
            "goal": config.goal,
            "backstory": config.backstory,
            "verbose": config.verbose,
            "cache": config.cache,
            "allow_delegation": config.allow_delegation,
            "max_iter": config.max_iter,
        }

        # Add optional fields
        if config.llm:
            kwargs["llm"] = config.llm
        if config.function_calling_llm:
            kwargs["function_calling_llm"] = config.function_calling_llm
        if config.max_rpm:
            kwargs["max_rpm"] = config.max_rpm
        if config.max_execution_time:
            kwargs["max_execution_time"] = config.max_execution_time
        if config.max_retry_limit:
            kwargs["max_retry_limit"] = config.max_retry_limit
        if config.allow_code_execution:
            kwargs["allow_code_execution"] = config.allow_code_execution
            kwargs["code_execution_mode"] = config.code_execution_mode
        if config.multimodal:
            kwargs["multimodal"] = config.multimodal
        if config.reasoning:
            kwargs["reasoning"] = config.reasoning
            if config.max_reasoning_attempts:
                kwargs["max_reasoning_attempts"] = config.max_reasoning_attempts
        if config.inject_date:
            kwargs["inject_date"] = config.inject_date
            kwargs["date_format"] = config.date_format
        if config.respect_context_window:
            kwargs["respect_context_window"] = config.respect_context_window
        if config.use_system_prompt is not None:
            kwargs["use_system_prompt"] = config.use_system_prompt
        if config.system_template:
            kwargs["system_template"] = config.system_template
        if config.prompt_template:
            kwargs["prompt_template"] = config.prompt_template
        if config.response_template:
            kwargs["response_template"] = config.response_template
        if tools:
            kwargs["tools"] = tools

        # Handle guardrail
        if config.guardrail and config.guardrail in self._guardrail_functions:
            kwargs["guardrail"] = self._guardrail_functions[config.guardrail]
            kwargs["guardrail_max_retries"] = config.guardrail_max_retries

        return Agent(**kwargs)

    def build_task(
        self, config: TaskConfig, agent_map: dict[str, Any]
    ) -> Any:
        """Build a runtime Task from configuration.

        Args:
            config: The task configuration
            agent_map: Mapping of agent IDs to Agent instances

        Returns:
            A CrewAI Task instance
        """
        from crewai.task import Task
        from crewai.tasks.conditional_task import ConditionalTask

        # Determine agent
        agent = None
        if config.agent_id and config.agent_id in agent_map:
            agent = agent_map[config.agent_id]
        elif config.agent_role:
            for a in agent_map.values():
                if hasattr(a, "role") and a.role == config.agent_role:
                    agent = a
                    break

        # Build tools
        tools = []
        for tool_id in config.tool_ids:
            tool_config = self._storage.get_tool(tool_id)
            if tool_config:
                try:
                    tools.append(self.build_tool(tool_config))
                except Exception:
                    pass

        # Build task kwargs
        kwargs: dict[str, Any] = {
            "description": config.description,
            "action_based": config.action_based,
            "async_execution": config.async_execution,
            "human_input": config.human_input,
            "markdown": config.markdown,
        }

        if config.name:
            kwargs["name"] = config.name
        if config.expected_output:
            kwargs["expected_output"] = config.expected_output
        if agent:
            kwargs["agent"] = agent
        if config.output_file:
            kwargs["output_file"] = config.output_file
        if tools:
            kwargs["tools"] = tools

        # Handle guardrail
        if config.guardrail and config.guardrail in self._guardrail_functions:
            kwargs["guardrail"] = self._guardrail_functions[config.guardrail]
            kwargs["guardrail_max_retries"] = config.guardrail_max_retries

        # Handle callback
        if config.callback and config.callback in self._callback_functions:
            kwargs["callback"] = self._callback_functions[config.callback]

        # Create conditional or regular task
        if config.is_conditional:
            if config.condition and config.condition in self._callback_functions:
                kwargs["condition"] = self._callback_functions[config.condition]
            return ConditionalTask(**kwargs)

        return Task(**kwargs)

    def build_crew(self, config: CrewConfig) -> Any:
        """Build a runtime Crew from configuration.

        Args:
            config: The crew configuration

        Returns:
            A CrewAI Crew instance
        """
        from crewai.crew import Crew
        from crewai.process import Process

        # Build agents
        agent_map: dict[str, Any] = {}
        agents = []
        for agent_id in config.agent_ids:
            agent_config = self._storage.get_agent(agent_id)
            if agent_config:
                agent = self.build_agent(agent_config)
                agent_map[agent_id] = agent
                agents.append(agent)

        # Build tasks (in order)
        task_map: dict[str, Any] = {}
        tasks = []
        for task_id in config.task_ids:
            task_config = self._storage.get_task(task_id)
            if task_config:
                task = self.build_task(task_config, agent_map)
                task_map[task_id] = task
                tasks.append(task)

        # Set task context
        for task_id in config.task_ids:
            task_config = self._storage.get_task(task_id)
            if task_config and task_config.context_task_ids:
                task = task_map.get(task_id)
                if task:
                    context = [
                        task_map[ctx_id]
                        for ctx_id in task_config.context_task_ids
                        if ctx_id in task_map
                    ]
                    if context:
                        task.context = context

        # Build crew kwargs
        process = (
            Process.hierarchical
            if config.process == "hierarchical"
            else Process.sequential
        )

        kwargs: dict[str, Any] = {
            "name": config.name,
            "agents": agents,
            "tasks": tasks,
            "process": process,
            "verbose": config.verbose,
            "cache": config.cache,
            "memory": config.memory,
            "planning": config.planning,
            "stream": config.stream,
        }

        if config.max_rpm:
            kwargs["max_rpm"] = config.max_rpm
        if config.manager_llm:
            kwargs["manager_llm"] = config.manager_llm
        if config.function_calling_llm:
            kwargs["function_calling_llm"] = config.function_calling_llm
        if config.planning_llm:
            kwargs["planning_llm"] = config.planning_llm
        if config.chat_llm:
            kwargs["chat_llm"] = config.chat_llm
        if config.output_log_file:
            kwargs["output_log_file"] = config.output_log_file
        if config.prompt_file:
            kwargs["prompt_file"] = config.prompt_file
        if config.tracing is not None:
            kwargs["tracing"] = config.tracing

        # Handle callbacks
        if config.task_callback and config.task_callback in self._callback_functions:
            kwargs["task_callback"] = self._callback_functions[config.task_callback]
        if config.step_callback and config.step_callback in self._callback_functions:
            kwargs["step_callback"] = self._callback_functions[config.step_callback]

        return Crew(**kwargs)

    # ==================== Execution Methods ====================

    def execute_crew(
        self, crew_id: str, inputs: dict[str, Any] | None = None
    ) -> ExecutionResult:
        """Execute a crew by ID.

        Args:
            crew_id: The ID of the crew to execute
            inputs: Input variables for the crew

        Returns:
            ExecutionResult with the crew output
        """
        crew_config = self._storage.get_crew(crew_id)
        if not crew_config:
            return ExecutionResult.from_error(crew_id, f"Crew not found: {crew_id}")

        try:
            crew = self.build_crew(crew_config)
            start_time = time.time()
            output = crew.kickoff(inputs=inputs or {})
            execution_time = time.time() - start_time

            return ExecutionResult.from_crew_output(crew_id, output, execution_time)
        except Exception as e:
            return ExecutionResult.from_error(crew_id, str(e))

    def execute_crew_streaming(
        self, crew_id: str, inputs: dict[str, Any] | None = None
    ) -> Iterator[Any]:
        """Execute a crew with streaming output.

        Args:
            crew_id: The ID of the crew to execute
            inputs: Input variables for the crew

        Yields:
            StreamChunk objects during execution
        """
        crew_config = self._storage.get_crew(crew_id)
        if not crew_config:
            return

        # Enable streaming in the config
        crew_config.stream = True

        try:
            crew = self.build_crew(crew_config)
            output = crew.kickoff(inputs=inputs or {})

            # If streaming output, iterate over it
            if hasattr(output, "__iter__"):
                yield from output
        except Exception as e:
            # Yield error as final chunk
            from crewai.types.streaming import StreamChunk, ChunkType
            yield StreamChunk(
                type=ChunkType.ERROR,
                content=str(e),
            )

    async def execute_crew_async(
        self, crew_id: str, inputs: dict[str, Any] | None = None
    ) -> ExecutionResult:
        """Execute a crew asynchronously.

        Args:
            crew_id: The ID of the crew to execute
            inputs: Input variables for the crew

        Returns:
            ExecutionResult with the crew output
        """
        crew_config = self._storage.get_crew(crew_id)
        if not crew_config:
            return ExecutionResult.from_error(crew_id, f"Crew not found: {crew_id}")

        try:
            crew = self.build_crew(crew_config)
            start_time = time.time()
            output = await crew.kickoff_async(inputs=inputs or {})
            execution_time = time.time() - start_time

            return ExecutionResult.from_crew_output(crew_id, output, execution_time)
        except Exception as e:
            return ExecutionResult.from_error(crew_id, str(e))

    async def execute_crew_streaming_async(
        self, crew_id: str, inputs: dict[str, Any] | None = None
    ) -> AsyncIterator[Any]:
        """Execute a crew with async streaming output.

        Args:
            crew_id: The ID of the crew to execute
            inputs: Input variables for the crew

        Yields:
            StreamChunk objects during execution
        """
        crew_config = self._storage.get_crew(crew_id)
        if not crew_config:
            return

        crew_config.stream = True

        try:
            crew = self.build_crew(crew_config)
            output = await crew.kickoff_async(inputs=inputs or {})

            # If streaming output, iterate over it
            if hasattr(output, "__aiter__"):
                async for chunk in output:
                    yield chunk
        except Exception as e:
            from crewai.types.streaming import StreamChunk, ChunkType
            yield StreamChunk(
                type=ChunkType.ERROR,
                content=str(e),
            )

    # ==================== Import/Export ====================

    def export_to_file(
        self, file_path: str | Path, format: str = "json"
    ) -> OperationResult[str]:
        """Export all data to a file.

        Args:
            file_path: Path to the output file
            format: Format to use ("json" or "yaml")

        Returns:
            OperationResult with the file path
        """
        try:
            data = self._storage.export_all()
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "yaml":
                import yaml
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                import json
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)

            return OperationResult.ok(str(file_path))
        except Exception as e:
            return OperationResult.fail(str(e), "EXPORT_ERROR")

    def import_from_file(
        self, file_path: str | Path, format: str | None = None
    ) -> OperationResult[bool]:
        """Import data from a file.

        Args:
            file_path: Path to the input file
            format: Format to use ("json" or "yaml"). Auto-detected if None.

        Returns:
            OperationResult indicating success or failure
        """
        try:
            file_path = Path(file_path)

            # Auto-detect format
            if format is None:
                if file_path.suffix.lower() in (".yaml", ".yml"):
                    format = "yaml"
                else:
                    format = "json"

            if format == "yaml":
                import yaml
                with open(file_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            else:
                import json
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            self._storage.import_all(data)
            return OperationResult.ok(True)
        except Exception as e:
            return OperationResult.fail(str(e), "IMPORT_ERROR")

    def clear_all(self) -> OperationResult[bool]:
        """Clear all data from storage.

        Returns:
            OperationResult indicating success
        """
        try:
            self._storage.clear_all()
            self._tool_instances.clear()
            return OperationResult.ok(True)
        except Exception as e:
            return OperationResult.fail(str(e), "CLEAR_ERROR")
