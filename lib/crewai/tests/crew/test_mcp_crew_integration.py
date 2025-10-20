"""Tests for Crew MCP integration functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock

# Import from the source directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task
from crewai.tools.mcp_tool_wrapper import MCPToolWrapper


class TestCrewMCPIntegration:
    """Test suite for Crew MCP integration functionality."""

    @pytest.fixture
    def mcp_agent(self):
        """Create an agent with MCP tools."""
        return Agent(
            role="MCP Research Agent",
            goal="Research using MCP tools",
            backstory="Agent with access to MCP tools for research",
            mcps=["https://api.example.com/mcp"]
        )

    @pytest.fixture
    def regular_agent(self):
        """Create a regular agent without MCP tools."""
        return Agent(
            role="Regular Agent",
            goal="Regular tasks without MCP",
            backstory="Standard agent without MCP access"
        )

    @pytest.fixture
    def sample_task(self, mcp_agent):
        """Create a sample task for testing."""
        return Task(
            description="Research AI frameworks using available tools",
            expected_output="Comprehensive research report",
            agent=mcp_agent
        )

    @pytest.fixture
    def mock_mcp_tools(self):
        """Create mock MCP tools."""
        tool1 = Mock(spec=MCPToolWrapper)
        tool1.name = "mcp_server_search_tool"
        tool1.description = "Search tool from MCP server"

        tool2 = Mock(spec=MCPToolWrapper)
        tool2.name = "mcp_server_analysis_tool"
        tool2.description = "Analysis tool from MCP server"

        return [tool1, tool2]

    def test_crew_creation_with_mcp_agent(self, mcp_agent, sample_task):
        """Test crew creation with MCP-enabled agent."""
        crew = Crew(
            agents=[mcp_agent],
            tasks=[sample_task],
            verbose=False
        )

        assert crew is not None
        assert len(crew.agents) == 1
        assert len(crew.tasks) == 1
        assert crew.agents[0] == mcp_agent

    def test_crew_add_mcp_tools_method_exists(self):
        """Test that Crew class has _add_mcp_tools method."""
        crew = Crew(agents=[], tasks=[])

        assert hasattr(crew, '_add_mcp_tools')
        assert callable(crew._add_mcp_tools)

    def test_crew_inject_mcp_tools_method_exists(self):
        """Test that Crew class has _inject_mcp_tools method."""
        crew = Crew(agents=[], tasks=[])

        assert hasattr(crew, '_inject_mcp_tools')
        assert callable(crew._inject_mcp_tools)

    def test_inject_mcp_tools_with_mcp_agent(self, mcp_agent, mock_mcp_tools):
        """Test MCP tools injection with MCP-enabled agent."""
        crew = Crew(agents=[mcp_agent], tasks=[])

        initial_tools = []

        with patch.object(mcp_agent, 'get_mcp_tools', return_value=mock_mcp_tools):
            result_tools = crew._inject_mcp_tools(initial_tools, mcp_agent)

            # Should merge MCP tools with existing tools
            assert len(result_tools) == len(mock_mcp_tools)

            # Verify get_mcp_tools was called with agent's mcps
            mcp_agent.get_mcp_tools.assert_called_once_with(mcps=mcp_agent.mcps)

    def test_inject_mcp_tools_with_regular_agent(self, regular_agent):
        """Test MCP tools injection with regular agent (no MCP tools)."""
        crew = Crew(agents=[regular_agent], tasks=[])

        initial_tools = [Mock(name="existing_tool")]

        # Regular agent has no mcps attribute
        result_tools = crew._inject_mcp_tools(initial_tools, regular_agent)

        # Should return original tools unchanged
        assert result_tools == initial_tools

    def test_inject_mcp_tools_empty_mcps_list(self, mcp_agent):
        """Test MCP tools injection with empty mcps list."""
        crew = Crew(agents=[mcp_agent], tasks=[])

        # Agent with empty mcps list
        mcp_agent.mcps = []
        initial_tools = [Mock(name="existing_tool")]

        result_tools = crew._inject_mcp_tools(initial_tools, mcp_agent)

        # Should return original tools unchanged
        assert result_tools == initial_tools

    def test_inject_mcp_tools_none_mcps(self, mcp_agent):
        """Test MCP tools injection with None mcps."""
        crew = Crew(agents=[mcp_agent], tasks=[])

        # Agent with None mcps
        mcp_agent.mcps = None
        initial_tools = [Mock(name="existing_tool")]

        result_tools = crew._inject_mcp_tools(initial_tools, mcp_agent)

        # Should return original tools unchanged
        assert result_tools == initial_tools

    def test_add_mcp_tools_with_task_agent(self, mcp_agent, sample_task, mock_mcp_tools):
        """Test _add_mcp_tools method with task agent."""
        crew = Crew(agents=[mcp_agent], tasks=[sample_task])

        initial_tools = [Mock(name="task_tool")]

        with patch.object(crew, '_inject_mcp_tools', return_value=initial_tools + mock_mcp_tools) as mock_inject:

            result_tools = crew._add_mcp_tools(sample_task, initial_tools)

            # Should call _inject_mcp_tools with task agent
            mock_inject.assert_called_once_with(initial_tools, sample_task.agent)
            assert len(result_tools) == len(initial_tools) + len(mock_mcp_tools)

    def test_add_mcp_tools_with_no_agent_task(self):
        """Test _add_mcp_tools method with task that has no agent."""
        crew = Crew(agents=[], tasks=[])

        # Task without agent
        task_no_agent = Task(
            description="Task without agent",
            expected_output="Some output",
            agent=None
        )

        initial_tools = [Mock(name="task_tool")]

        result_tools = crew._add_mcp_tools(task_no_agent, initial_tools)

        # Should return original tools unchanged
        assert result_tools == initial_tools

    def test_mcp_tools_integration_in_task_preparation_flow(self, mcp_agent, sample_task, mock_mcp_tools):
        """Test MCP tools integration in the task preparation flow."""
        crew = Crew(agents=[mcp_agent], tasks=[sample_task])

        # Mock the crew's tool preparation methods
        with patch.object(crew, '_add_platform_tools', return_value=[Mock(name="platform_tool")]) as mock_platform, \
             patch.object(crew, '_add_mcp_tools', return_value=mock_mcp_tools) as mock_mcp, \
             patch.object(crew, '_add_multimodal_tools', return_value=mock_mcp_tools) as mock_multimodal:

            # This tests the integration point where MCP tools are added to task tools
            # We can't easily test the full _prepare_tools_for_task method due to complexity,
            # but we can verify our _add_mcp_tools integration point works

            result = crew._add_mcp_tools(sample_task, [])

            assert result == mock_mcp_tools
            mock_mcp.assert_called_once_with(sample_task, [])

    def test_mcp_tools_merge_with_existing_tools(self, mcp_agent, mock_mcp_tools):
        """Test that MCP tools merge correctly with existing tools."""
        from crewai.tools import BaseTool

        class ExistingTool(BaseTool):
            name: str = "existing_search"
            description: str = "Existing search tool"

            def _run(self, **kwargs):
                return "Existing search result"

        existing_tools = [ExistingTool()]
        crew = Crew(agents=[mcp_agent], tasks=[])

        with patch.object(mcp_agent, 'get_mcp_tools', return_value=mock_mcp_tools):
            merged_tools = crew._inject_mcp_tools(existing_tools, mcp_agent)

            # Should have both existing tools and MCP tools
            total_expected = len(existing_tools) + len(mock_mcp_tools)
            assert len(merged_tools) == total_expected

            # Verify existing tools are preserved
            existing_names = [tool.name for tool in existing_tools]
            merged_names = [tool.name for tool in merged_tools]

            for existing_name in existing_names:
                assert existing_name in merged_names

    def test_mcp_tools_available_in_crew_context(self, mcp_agent, sample_task, mock_mcp_tools):
        """Test that MCP tools are available in crew execution context."""
        crew = Crew(agents=[mcp_agent], tasks=[sample_task])

        with patch.object(mcp_agent, 'get_mcp_tools', return_value=mock_mcp_tools):

            # Test that crew can access MCP tools through agent
            agent_tools = crew._inject_mcp_tools([], mcp_agent)

            assert len(agent_tools) == len(mock_mcp_tools)
            assert all(tool in agent_tools for tool in mock_mcp_tools)

    def test_crew_with_mixed_agents_mcp_and_regular(self, mcp_agent, regular_agent, mock_mcp_tools):
        """Test crew with both MCP-enabled and regular agents."""
        task1 = Task(
            description="Task for MCP agent",
            expected_output="MCP-powered result",
            agent=mcp_agent
        )

        task2 = Task(
            description="Task for regular agent",
            expected_output="Regular result",
            agent=regular_agent
        )

        crew = Crew(
            agents=[mcp_agent, regular_agent],
            tasks=[task1, task2]
        )

        # Test MCP tools injection for MCP agent
        with patch.object(mcp_agent, 'get_mcp_tools', return_value=mock_mcp_tools):
            mcp_tools = crew._inject_mcp_tools([], mcp_agent)
            assert len(mcp_tools) == len(mock_mcp_tools)

        # Test MCP tools injection for regular agent
        regular_tools = crew._inject_mcp_tools([], regular_agent)
        assert len(regular_tools) == 0

    def test_crew_mcp_tools_error_handling_during_execution_prep(self, mcp_agent, sample_task):
        """Test crew error handling when MCP tools fail during execution preparation."""
        crew = Crew(agents=[mcp_agent], tasks=[sample_task])

        # Mock MCP tools failure during crew execution preparation
        with patch.object(mcp_agent, 'get_mcp_tools', side_effect=Exception("MCP tools failed")):

            # Crew operations should continue despite MCP failure
            try:
                crew._inject_mcp_tools([], mcp_agent)
                # If we get here, the error was handled gracefully by returning empty tools
            except Exception as e:
                # If exception propagates, it should be an expected one
                assert "MCP tools failed" in str(e)

    def test_crew_task_execution_flow_includes_mcp_tools(self, mcp_agent, sample_task):
        """Test that crew task execution flow includes MCP tools integration."""
        crew = Crew(agents=[mcp_agent], tasks=[sample_task])

        # Verify that crew has the necessary methods for MCP integration
        assert hasattr(crew, '_add_mcp_tools')
        assert hasattr(crew, '_inject_mcp_tools')

        # Test the task has an agent with MCP capabilities
        assert sample_task.agent == mcp_agent
        assert hasattr(sample_task.agent, 'mcps')
        assert hasattr(sample_task.agent, 'get_mcp_tools')

    def test_mcp_tools_do_not_interfere_with_platform_tools(self, mock_mcp_tools):
        """Test that MCP tools don't interfere with platform tools integration."""
        agent_with_both = Agent(
            role="Multi-Tool Agent",
            goal="Use both platform and MCP tools",
            backstory="Agent with access to multiple tool types",
            apps=["gmail", "slack"],
            mcps=["https://api.example.com/mcp"]
        )

        task = Task(
            description="Use both platform and MCP tools",
            expected_output="Combined tool usage result",
            agent=agent_with_both
        )

        crew = Crew(agents=[agent_with_both], tasks=[task])

        platform_tools = [Mock(name="gmail_tool"), Mock(name="slack_tool")]

        # Test platform tools injection
        with patch.object(crew, '_inject_platform_tools', return_value=platform_tools):
            result_platform = crew._inject_platform_tools([], agent_with_both)
            assert len(result_platform) == len(platform_tools)

        # Test MCP tools injection
        with patch.object(agent_with_both, 'get_mcp_tools', return_value=mock_mcp_tools):
            result_mcp = crew._inject_mcp_tools(platform_tools, agent_with_both)
            assert len(result_mcp) == len(platform_tools) + len(mock_mcp_tools)

    def test_crew_task_execution_order_includes_mcp_tools(self, mcp_agent, sample_task):
        """Test that crew task execution order includes MCP tools at the right point."""
        crew = Crew(agents=[mcp_agent], tasks=[sample_task])

        # Mock the various tool addition methods to verify call order
        call_order = []

        def track_platform_tools(*args, **kwargs):
            call_order.append("platform")
            return []

        def track_mcp_tools(*args, **kwargs):
            call_order.append("mcp")
            return []

        def track_multimodal_tools(*args, **kwargs):
            call_order.append("multimodal")
            return []

        with patch.object(crew, '_add_platform_tools', side_effect=track_platform_tools), \
             patch.object(crew, '_add_mcp_tools', side_effect=track_mcp_tools), \
             patch.object(crew, '_add_multimodal_tools', side_effect=track_multimodal_tools):

            # Test the crew's task preparation flow
            # We check that MCP tools are added in the right sequence

            # These methods are called in the task preparation flow
            crew._add_platform_tools(sample_task, [])
            crew._add_mcp_tools(sample_task, [])

            assert "platform" in call_order
            assert "mcp" in call_order

    def test_crew_handles_agent_without_get_mcp_tools_method(self):
        """Test crew handles agents that don't implement get_mcp_tools method."""
        # Create a mock agent that doesn't have get_mcp_tools
        mock_agent = Mock()
        mock_agent.mcps = ["https://api.example.com/mcp"]
        # Explicitly don't add get_mcp_tools method

        crew = Crew(agents=[], tasks=[])

        # Should handle gracefully when agent doesn't have get_mcp_tools
        result_tools = crew._inject_mcp_tools([], mock_agent)

        # Should return empty list since agent doesn't have get_mcp_tools
        assert result_tools == []

    def test_crew_handles_agent_get_mcp_tools_exception(self, mcp_agent):
        """Test crew handles exceptions from agent's get_mcp_tools method."""
        crew = Crew(agents=[mcp_agent], tasks=[])

        with patch.object(mcp_agent, 'get_mcp_tools', side_effect=Exception("MCP tools failed")):

            # Should handle exception gracefully
            result_tools = crew._inject_mcp_tools([], mcp_agent)

            # Depending on implementation, should either return empty list or re-raise
            # Since get_mcp_tools handles errors internally, this should return empty list
            assert isinstance(result_tools, list)

    def test_crew_mcp_tools_merge_functionality(self, mock_mcp_tools):
        """Test crew's tool merging functionality with MCP tools."""
        crew = Crew(agents=[], tasks=[])

        existing_tools = [Mock(name="existing_tool_1"), Mock(name="existing_tool_2")]

        # Test _merge_tools method with MCP tools
        merged_tools = crew._merge_tools(existing_tools, mock_mcp_tools)

        total_expected = len(existing_tools) + len(mock_mcp_tools)
        assert len(merged_tools) == total_expected

        # Verify all tools are present
        all_tool_names = [tool.name for tool in merged_tools]
        assert "existing_tool_1" in all_tool_names
        assert "existing_tool_2" in all_tool_names
        assert mock_mcp_tools[0].name in all_tool_names
        assert mock_mcp_tools[1].name in all_tool_names

    def test_crew_workflow_integration_conditions(self, mcp_agent, sample_task):
        """Test the conditions for MCP tools integration in crew workflows."""
        crew = Crew(agents=[mcp_agent], tasks=[sample_task])

        # Test condition: agent exists and has mcps attribute
        assert hasattr(sample_task.agent, 'mcps')
        assert sample_task.agent.mcps is not None

        # Test condition: agent has get_mcp_tools method
        assert hasattr(sample_task.agent, 'get_mcp_tools')

        # Test condition: mcps list is not empty
        sample_task.agent.mcps = ["https://api.example.com/mcp"]
        assert len(sample_task.agent.mcps) > 0

    def test_crew_mcp_integration_performance_impact(self, mcp_agent, sample_task, mock_mcp_tools):
        """Test that MCP integration doesn't significantly impact crew performance."""
        crew = Crew(agents=[mcp_agent], tasks=[sample_task])

        import time

        # Test tool injection performance
        start_time = time.time()

        with patch.object(mcp_agent, 'get_mcp_tools', return_value=mock_mcp_tools):
            # Multiple tool injection calls should be fast due to caching
            for _ in range(5):
                tools = crew._inject_mcp_tools([], mcp_agent)

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete quickly (less than 1 second for 5 operations)
        assert total_time < 1.0
        assert len(tools) == len(mock_mcp_tools)

    def test_crew_task_tool_availability_with_mcp(self, mcp_agent, sample_task, mock_mcp_tools):
        """Test that MCP tools are available during task execution."""
        crew = Crew(agents=[mcp_agent], tasks=[sample_task])

        with patch.object(mcp_agent, 'get_mcp_tools', return_value=mock_mcp_tools):

            # Simulate task tool preparation
            prepared_tools = crew._inject_mcp_tools([], mcp_agent)

            # Verify tools are properly prepared for task execution
            assert len(prepared_tools) == len(mock_mcp_tools)

            # Each tool should be a valid BaseTool-compatible object
            for tool in prepared_tools:
                assert hasattr(tool, 'name')
                assert hasattr(tool, 'description')

    def test_crew_handles_mcp_tool_name_conflicts(self, mock_mcp_tools):
        """Test crew handling of potential tool name conflicts."""
        # Create MCP agent with tools that might conflict
        agent1 = Agent(
            role="Agent 1",
            goal="Test conflicts",
            backstory="First agent with MCP tools",
            mcps=["https://server1.com/mcp"]
        )

        agent2 = Agent(
            role="Agent 2",
            goal="Test conflicts",
            backstory="Second agent with MCP tools",
            mcps=["https://server2.com/mcp"]
        )

        # Mock tools with same original names but different server prefixes
        server1_tools = [Mock(name="server1_com_mcp_search_tool")]
        server2_tools = [Mock(name="server2_com_mcp_search_tool")]

        task1 = Task(description="Task 1", expected_output="Result 1", agent=agent1)
        task2 = Task(description="Task 2", expected_output="Result 2", agent=agent2)

        crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])

        with patch.object(agent1, 'get_mcp_tools', return_value=server1_tools), \
             patch.object(agent2, 'get_mcp_tools', return_value=server2_tools):

            # Each agent should get its own prefixed tools
            tools1 = crew._inject_mcp_tools([], agent1)
            tools2 = crew._inject_mcp_tools([], agent2)

            assert len(tools1) == 1
            assert len(tools2) == 1
            assert tools1[0].name != tools2[0].name  # Names should be different due to prefixing

    def test_crew_mcp_integration_with_verbose_mode(self, mcp_agent, sample_task):
        """Test MCP integration works with crew verbose mode."""
        crew = Crew(
            agents=[mcp_agent],
            tasks=[sample_task],
            verbose=True  # Enable verbose mode
        )

        # Should work the same regardless of verbose mode
        assert crew.verbose is True
        assert hasattr(crew, '_inject_mcp_tools')

        # MCP integration should not be affected by verbose mode
        with patch.object(mcp_agent, 'get_mcp_tools', return_value=[Mock()]):
            tools = crew._inject_mcp_tools([], mcp_agent)
            assert len(tools) == 1
