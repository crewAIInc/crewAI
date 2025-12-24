"""Tests for CrewManager CRUD operations."""

import tempfile
from pathlib import Path

import pytest

from crewai.manager import (
    AgentConfig,
    CrewConfig,
    CrewManager,
    KnowledgeSourceConfig,
    MCPServerConfig,
    TaskConfig,
    ToolConfig,
)
from crewai.manager.storage import (
    InMemoryStorage,
    JSONFileStorage,
    SQLiteStorage,
    YAMLFileStorage,
)


class TestAgentCRUD:
    """Test agent CRUD operations."""

    def test_create_agent(self):
        """Test creating an agent."""
        manager = CrewManager()
        agent = AgentConfig(
            role="Researcher",
            goal="Find information",
            backstory="Expert researcher",
        )

        result = manager.create_agent(agent)

        assert result.success
        assert result.data.role == "Researcher"
        assert result.data.id == agent.id

    def test_get_agent(self):
        """Test getting an agent by ID."""
        manager = CrewManager()
        agent = AgentConfig(
            role="Researcher",
            goal="Find information",
            backstory="Expert researcher",
        )
        manager.create_agent(agent)

        result = manager.get_agent(agent.id)

        assert result.success
        assert result.data.role == "Researcher"

    def test_get_agent_not_found(self):
        """Test getting a non-existent agent."""
        manager = CrewManager()

        result = manager.get_agent("non-existent-id")

        assert not result.success
        assert result.error_code == "AGENT_NOT_FOUND"

    def test_get_agent_by_role(self):
        """Test getting an agent by role."""
        manager = CrewManager()
        agent = AgentConfig(
            role="Researcher",
            goal="Find information",
            backstory="Expert researcher",
        )
        manager.create_agent(agent)

        result = manager.get_agent_by_role("Researcher")

        assert result.success
        assert result.data.id == agent.id

    def test_list_agents(self):
        """Test listing agents."""
        manager = CrewManager()
        for i in range(5):
            manager.create_agent(
                AgentConfig(
                    role=f"Agent {i}",
                    goal="Goal",
                    backstory="Backstory",
                )
            )

        result = manager.list_agents()

        assert len(result.items) == 5
        assert result.total == 5

    def test_list_agents_pagination(self):
        """Test listing agents with pagination."""
        manager = CrewManager()
        for i in range(10):
            manager.create_agent(
                AgentConfig(
                    role=f"Agent {i}",
                    goal="Goal",
                    backstory="Backstory",
                )
            )

        result = manager.list_agents(offset=0, limit=3)

        assert len(result.items) == 3
        assert result.total == 10
        assert result.has_more

    def test_update_agent(self):
        """Test updating an agent."""
        manager = CrewManager()
        agent = AgentConfig(
            role="Researcher",
            goal="Find information",
            backstory="Expert researcher",
        )
        manager.create_agent(agent)

        result = manager.update_agent(agent.id, {"goal": "New goal"})

        assert result.success
        assert result.data.goal == "New goal"

    def test_delete_agent(self):
        """Test deleting an agent."""
        manager = CrewManager()
        agent = AgentConfig(
            role="Researcher",
            goal="Find information",
            backstory="Expert researcher",
        )
        manager.create_agent(agent)

        result = manager.delete_agent(agent.id)

        assert result.success
        assert manager.get_agent(agent.id).success is False


class TestTaskCRUD:
    """Test task CRUD operations."""

    def test_create_task(self):
        """Test creating a task."""
        manager = CrewManager()
        task = TaskConfig(
            description="Research the topic",
            expected_output="A detailed report",
        )

        result = manager.create_task(task)

        assert result.success
        assert result.data.description == "Research the topic"

    def test_get_task(self):
        """Test getting a task by ID."""
        manager = CrewManager()
        task = TaskConfig(
            description="Research the topic",
            expected_output="A detailed report",
        )
        manager.create_task(task)

        result = manager.get_task(task.id)

        assert result.success
        assert result.data.description == "Research the topic"

    def test_list_tasks(self):
        """Test listing tasks."""
        manager = CrewManager()
        for i in range(3):
            manager.create_task(
                TaskConfig(
                    description=f"Task {i}",
                    expected_output="Output",
                )
            )

        result = manager.list_tasks()

        assert len(result.items) == 3

    def test_update_task(self):
        """Test updating a task."""
        manager = CrewManager()
        task = TaskConfig(
            description="Research the topic",
            expected_output="A detailed report",
        )
        manager.create_task(task)

        result = manager.update_task(task.id, {"expected_output": "New output"})

        assert result.success
        assert result.data.expected_output == "New output"

    def test_delete_task(self):
        """Test deleting a task."""
        manager = CrewManager()
        task = TaskConfig(
            description="Research the topic",
            expected_output="A detailed report",
        )
        manager.create_task(task)

        result = manager.delete_task(task.id)

        assert result.success


class TestCrewCRUD:
    """Test crew CRUD operations."""

    def test_create_crew(self):
        """Test creating a crew."""
        manager = CrewManager()
        crew = CrewConfig(
            name="Research Crew",
            agent_ids=[],
            task_ids=[],
        )

        result = manager.create_crew(crew)

        assert result.success
        assert result.data.name == "Research Crew"

    def test_get_crew(self):
        """Test getting a crew by ID."""
        manager = CrewManager()
        crew = CrewConfig(
            name="Research Crew",
            agent_ids=[],
            task_ids=[],
        )
        manager.create_crew(crew)

        result = manager.get_crew(crew.id)

        assert result.success
        assert result.data.name == "Research Crew"

    def test_get_crew_by_name(self):
        """Test getting a crew by name."""
        manager = CrewManager()
        crew = CrewConfig(
            name="Research Crew",
            agent_ids=[],
            task_ids=[],
        )
        manager.create_crew(crew)

        result = manager.get_crew_by_name("Research Crew")

        assert result.success
        assert result.data.id == crew.id

    def test_list_crews(self):
        """Test listing crews."""
        manager = CrewManager()
        for i in range(3):
            manager.create_crew(
                CrewConfig(
                    name=f"Crew {i}",
                    agent_ids=[],
                    task_ids=[],
                )
            )

        result = manager.list_crews()

        assert len(result.items) == 3

    def test_update_crew(self):
        """Test updating a crew."""
        manager = CrewManager()
        crew = CrewConfig(
            name="Research Crew",
            agent_ids=[],
            task_ids=[],
        )
        manager.create_crew(crew)

        result = manager.update_crew(crew.id, {"verbose": True})

        assert result.success
        assert result.data.verbose is True

    def test_delete_crew(self):
        """Test deleting a crew."""
        manager = CrewManager()
        crew = CrewConfig(
            name="Research Crew",
            agent_ids=[],
            task_ids=[],
        )
        manager.create_crew(crew)

        result = manager.delete_crew(crew.id)

        assert result.success


class TestToolCRUD:
    """Test tool CRUD operations."""

    def test_create_tool(self):
        """Test creating a tool."""
        manager = CrewManager()
        tool = ToolConfig(
            name="search_tool",
            description="A tool for searching the web",
            tool_type="crewai_tools",
            class_name="SerperDevTool",
        )

        result = manager.create_tool(tool)

        assert result.success
        assert result.data.name == "search_tool"

    def test_get_tool(self):
        """Test getting a tool by ID."""
        manager = CrewManager()
        tool = ToolConfig(
            name="search_tool",
            description="A tool for searching the web",
            tool_type="crewai_tools",
            class_name="SerperDevTool",
        )
        manager.create_tool(tool)

        result = manager.get_tool(tool.id)

        assert result.success
        assert result.data.name == "search_tool"

    def test_list_tools(self):
        """Test listing tools."""
        manager = CrewManager()
        for i in range(3):
            manager.create_tool(
                ToolConfig(
                    name=f"tool_{i}",
                    description=f"Tool {i} description",
                    tool_type="crewai_tools",
                )
            )

        result = manager.list_tools()

        assert len(result.items) == 3

    def test_update_tool(self):
        """Test updating a tool."""
        manager = CrewManager()
        tool = ToolConfig(
            name="search_tool",
            description="A tool for searching the web",
            tool_type="crewai_tools",
            class_name="SerperDevTool",
        )
        manager.create_tool(tool)

        result = manager.update_tool(tool.id, {"class_name": "ScrapeWebsiteTool"})

        assert result.success
        assert result.data.class_name == "ScrapeWebsiteTool"

    def test_delete_tool(self):
        """Test deleting a tool."""
        manager = CrewManager()
        tool = ToolConfig(
            name="search_tool",
            description="A tool for searching",
            tool_type="crewai_tools",
        )
        manager.create_tool(tool)

        result = manager.delete_tool(tool.id)

        assert result.success


class TestKnowledgeSourceCRUD:
    """Test knowledge source CRUD operations."""

    def test_create_knowledge_source(self):
        """Test creating a knowledge source."""
        manager = CrewManager()
        source = KnowledgeSourceConfig(
            name="Company Docs",
            source_type="pdf",
            file_paths=["/path/to/doc.pdf"],
        )

        result = manager.create_knowledge_source(source)

        assert result.success
        assert result.data.name == "Company Docs"

    def test_get_knowledge_source(self):
        """Test getting a knowledge source by ID."""
        manager = CrewManager()
        source = KnowledgeSourceConfig(
            name="Company Docs",
            source_type="pdf",
            file_paths=["/path/to/doc.pdf"],
        )
        manager.create_knowledge_source(source)

        result = manager.get_knowledge_source(source.id)

        assert result.success
        assert result.data.name == "Company Docs"

    def test_get_knowledge_source_by_name(self):
        """Test getting a knowledge source by name."""
        manager = CrewManager()
        source = KnowledgeSourceConfig(
            name="Company Docs",
            source_type="pdf",
        )
        manager.create_knowledge_source(source)

        result = manager.get_knowledge_source_by_name("Company Docs")

        assert result.success
        assert result.data.id == source.id

    def test_list_knowledge_sources(self):
        """Test listing knowledge sources."""
        manager = CrewManager()
        for i in range(3):
            manager.create_knowledge_source(
                KnowledgeSourceConfig(
                    name=f"Source {i}",
                    source_type="text",
                )
            )

        result = manager.list_knowledge_sources()

        assert len(result.items) == 3

    def test_update_knowledge_source(self):
        """Test updating a knowledge source."""
        manager = CrewManager()
        source = KnowledgeSourceConfig(
            name="Company Docs",
            source_type="pdf",
            chunk_size=4000,
        )
        manager.create_knowledge_source(source)

        result = manager.update_knowledge_source(source.id, {"chunk_size": 2000})

        assert result.success
        assert result.data.chunk_size == 2000

    def test_delete_knowledge_source(self):
        """Test deleting a knowledge source."""
        manager = CrewManager()
        source = KnowledgeSourceConfig(
            name="Company Docs",
            source_type="pdf",
        )
        manager.create_knowledge_source(source)

        result = manager.delete_knowledge_source(source.id)

        assert result.success


class TestMCPServerCRUD:
    """Test MCP server CRUD operations."""

    def test_create_mcp_server(self):
        """Test creating an MCP server."""
        manager = CrewManager()
        server = MCPServerConfig(
            name="file_server",
            transport="stdio",
            command="npx",
            args=["@modelcontextprotocol/server-filesystem"],
        )

        result = manager.create_mcp_server(server)

        assert result.success
        assert result.data.name == "file_server"

    def test_get_mcp_server(self):
        """Test getting an MCP server by ID."""
        manager = CrewManager()
        server = MCPServerConfig(
            name="file_server",
            transport="stdio",
        )
        manager.create_mcp_server(server)

        result = manager.get_mcp_server(server.id)

        assert result.success
        assert result.data.name == "file_server"

    def test_get_mcp_server_by_name(self):
        """Test getting an MCP server by name."""
        manager = CrewManager()
        server = MCPServerConfig(
            name="file_server",
            transport="stdio",
        )
        manager.create_mcp_server(server)

        result = manager.get_mcp_server_by_name("file_server")

        assert result.success
        assert result.data.id == server.id

    def test_list_mcp_servers(self):
        """Test listing MCP servers."""
        manager = CrewManager()
        for i in range(3):
            manager.create_mcp_server(
                MCPServerConfig(
                    name=f"server_{i}",
                    transport="stdio",
                )
            )

        result = manager.list_mcp_servers()

        assert len(result.items) == 3

    def test_update_mcp_server(self):
        """Test updating an MCP server."""
        manager = CrewManager()
        server = MCPServerConfig(
            name="file_server",
            transport="stdio",
            connection_timeout=30,
        )
        manager.create_mcp_server(server)

        result = manager.update_mcp_server(server.id, {"connection_timeout": 60})

        assert result.success
        assert result.data.connection_timeout == 60

    def test_delete_mcp_server(self):
        """Test deleting an MCP server."""
        manager = CrewManager()
        server = MCPServerConfig(
            name="file_server",
            transport="stdio",
        )
        manager.create_mcp_server(server)

        result = manager.delete_mcp_server(server.id)

        assert result.success


class TestStorageBackends:
    """Test different storage backends."""

    def test_in_memory_storage(self):
        """Test InMemoryStorage."""
        storage = InMemoryStorage()
        manager = CrewManager(storage=storage)

        agent = AgentConfig(
            role="Test",
            goal="Test",
            backstory="Test",
        )
        manager.create_agent(agent)

        assert manager.get_agent(agent.id).success

    def test_json_file_storage(self):
        """Test JSONFileStorage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.json"
            storage = JSONFileStorage(file_path)
            manager = CrewManager(storage=storage)

            agent = AgentConfig(
                role="Test",
                goal="Test",
                backstory="Test",
            )
            manager.create_agent(agent)

            assert file_path.exists()
            assert manager.get_agent(agent.id).success

    def test_yaml_file_storage(self):
        """Test YAMLFileStorage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.yaml"
            storage = YAMLFileStorage(file_path)
            manager = CrewManager(storage=storage)

            agent = AgentConfig(
                role="Test",
                goal="Test",
                backstory="Test",
            )
            manager.create_agent(agent)

            assert file_path.exists()
            assert manager.get_agent(agent.id).success

    def test_sqlite_storage(self):
        """Test SQLiteStorage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = SQLiteStorage(db_path)
            manager = CrewManager(storage=storage)

            agent = AgentConfig(
                role="Test",
                goal="Test",
                backstory="Test",
            )
            manager.create_agent(agent)

            assert db_path.exists()
            assert manager.get_agent(agent.id).success


class TestImportExport:
    """Test import/export functionality."""

    def test_export_import_json(self):
        """Test exporting and importing JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CrewManager()

            # Create some data
            agent = AgentConfig(
                role="Test",
                goal="Test",
                backstory="Test",
            )
            manager.create_agent(agent)

            task = TaskConfig(
                description="Test task",
                expected_output="Output",
            )
            manager.create_task(task)

            # Export
            export_path = Path(tmpdir) / "export.json"
            result = manager.export_to_file(export_path)
            assert result.success

            # Create new manager and import
            manager2 = CrewManager()
            result = manager2.import_from_file(export_path)
            assert result.success

            # Verify data
            assert manager2.get_agent(agent.id).success
            assert manager2.get_task(task.id).success

    def test_export_import_yaml(self):
        """Test exporting and importing YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CrewManager()

            # Create some data
            crew = CrewConfig(
                name="Test Crew",
                agent_ids=[],
                task_ids=[],
            )
            manager.create_crew(crew)

            # Export
            export_path = Path(tmpdir) / "export.yaml"
            result = manager.export_to_file(export_path, format="yaml")
            assert result.success

            # Create new manager and import
            manager2 = CrewManager()
            result = manager2.import_from_file(export_path)
            assert result.success

            # Verify data
            assert manager2.get_crew(crew.id).success

    def test_clear_all(self):
        """Test clearing all data."""
        manager = CrewManager()

        # Create some data
        agent = AgentConfig(
            role="Test",
            goal="Test",
            backstory="Test",
        )
        manager.create_agent(agent)

        # Clear
        result = manager.clear_all()
        assert result.success

        # Verify empty
        assert manager.list_agents().total == 0


class TestCallbackRegistration:
    """Test callback and guardrail registration."""

    def test_register_guardrail(self):
        """Test registering a guardrail function."""
        manager = CrewManager()

        def my_guardrail(output):
            return output

        manager.register_guardrail("my_guardrail", my_guardrail)

        assert "my_guardrail" in manager._guardrail_functions

    def test_register_callback(self):
        """Test registering a callback function."""
        manager = CrewManager()

        def my_callback(output):
            pass

        manager.register_callback("my_callback", my_callback)

        assert "my_callback" in manager._callback_functions

    def test_register_tool_instance(self):
        """Test registering a tool instance."""
        manager = CrewManager()

        class MockTool:
            pass

        tool = ToolConfig(
            name="mock_tool",
            description="A mock tool for testing",
            tool_type="custom",
        )
        manager.create_tool(tool)
        manager.register_tool_instance(tool.id, MockTool())

        assert tool.id in manager._tool_instances


class TestTagFiltering:
    """Test tag-based filtering."""

    def test_list_agents_with_tags(self):
        """Test listing agents filtered by tags."""
        manager = CrewManager()

        # Create agents with different tags
        agent1 = AgentConfig(
            role="Research",
            goal="Research",
            backstory="Research",
            tags=["research", "ai"],
        )
        agent2 = AgentConfig(
            role="Writer",
            goal="Write",
            backstory="Writer",
            tags=["writing"],
        )
        manager.create_agent(agent1)
        manager.create_agent(agent2)

        # Filter by tag
        result = manager.list_agents(tags=["research"])

        assert len(result.items) == 1
        assert result.items[0].role == "Research"
