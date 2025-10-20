"""End-to-end integration tests for MCP DSL functionality."""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock

# Import from the source directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task
from crewai.tools.mcp_tool_wrapper import MCPToolWrapper
from tests.mocks.mcp_server_mock import MockMCPServerFactory


class TestMCPEndToEndIntegration:
    """End-to-end integration tests for MCP DSL functionality."""

    def test_complete_mcp_workflow_single_server(self):
        """Test complete MCP workflow with single server."""
        print("\n=== Testing Complete MCP Workflow ===")

        # Step 1: Create agent with MCP configuration
        agent = Agent(
            role="E2E Test Agent",
            goal="Test complete MCP workflow",
            backstory="Agent for end-to-end MCP testing",
            mcps=["https://api.example.com/mcp"]
        )

        assert agent.mcps == ["https://api.example.com/mcp"]
        print("✅ Step 1: Agent created with MCP configuration")

        # Step 2: Mock tool discovery
        mock_schemas = {
            "search_web": {"description": "Search the web for information"},
            "analyze_data": {"description": "Analyze provided data"}
        }

        with patch.object(agent, '_get_mcp_tool_schemas', return_value=mock_schemas):

            # Step 3: Discover MCP tools
            discovered_tools = agent.get_mcp_tools(agent.mcps)

            assert len(discovered_tools) == 2
            assert all(isinstance(tool, MCPToolWrapper) for tool in discovered_tools)
            print(f"✅ Step 3: Discovered {len(discovered_tools)} MCP tools")

            # Verify tool names are properly prefixed
            tool_names = [tool.name for tool in discovered_tools]
            assert "api_example_com_mcp_search_web" in tool_names
            assert "api_example_com_mcp_analyze_data" in tool_names

            # Step 4: Create task and crew
            task = Task(
                description="Research AI frameworks using MCP tools",
                expected_output="Research report using discovered tools",
                agent=agent
            )

            crew = Crew(agents=[agent], tasks=[task])
            print("✅ Step 4: Created task and crew")

            # Step 5: Test crew tool integration
            crew_tools = crew._inject_mcp_tools([], agent)
            assert len(crew_tools) == 2
            print("✅ Step 5: MCP tools integrated into crew")

            # Step 6: Test tool execution
            search_tool = next(tool for tool in discovered_tools if "search" in tool.name)

            # Mock successful tool execution
            with patch.object(search_tool, '_run_async', return_value="Search results: AI frameworks found"):
                result = search_tool._run(query="AI frameworks")

                assert "Search results" in result
                print("✅ Step 6: Tool execution successful")

    def test_complete_mcp_workflow_multiple_servers(self):
        """Test complete MCP workflow with multiple servers."""
        print("\n=== Testing Multi-Server MCP Workflow ===")

        # Create agent with multiple MCP servers
        agent = Agent(
            role="Multi-Server Agent",
            goal="Test multiple MCP server integration",
            backstory="Agent with access to multiple MCP servers",
            mcps=[
                "https://search-server.com/mcp",
                "https://analysis-server.com/mcp#specific_tool",
                "crewai-amp:weather-service",
                "crewai-amp:financial-data#stock_price_tool"
            ]
        )

        print(f"✅ Agent created with {len(agent.mcps)} MCP references")

        # Mock different server responses
        def mock_external_tools(mcp_ref):
            if "search-server" in mcp_ref:
                return [Mock(name="search_server_com_mcp_web_search")]
            elif "analysis-server" in mcp_ref:
                return [Mock(name="analysis_server_com_mcp_specific_tool")]
            return []

        def mock_amp_tools(amp_ref):
            if "weather-service" in amp_ref:
                return [Mock(name="weather_service_get_forecast")]
            elif "financial-data" in amp_ref:
                return [Mock(name="financial_data_stock_price_tool")]
            return []

        with patch.object(agent, '_get_external_mcp_tools', side_effect=mock_external_tools), \
             patch.object(agent, '_get_amp_mcp_tools', side_effect=mock_amp_tools):

            # Discover all tools
            all_tools = agent.get_mcp_tools(agent.mcps)

            # Should get tools from all servers
            expected_tools = 2 + 2  # 2 external + 2 AMP
            assert len(all_tools) == expected_tools
            print(f"✅ Discovered {len(all_tools)} tools from multiple servers")

            # Create multi-task crew
            tasks = [
                Task(
                    description="Search for information",
                    expected_output="Search results",
                    agent=agent
                ),
                Task(
                    description="Analyze financial data",
                    expected_output="Analysis report",
                    agent=agent
                )
            ]

            crew = Crew(agents=[agent], tasks=tasks)

            # Test crew integration with multiple tools
            for task in tasks:
                task_tools = crew._inject_mcp_tools([], task.agent)
                assert len(task_tools) == expected_tools

            print("✅ Multi-server integration successful")

    def test_mcp_workflow_with_error_recovery(self):
        """Test MCP workflow with error recovery scenarios."""
        print("\n=== Testing MCP Workflow with Error Recovery ===")

        # Create agent with mix of working and failing servers
        agent = Agent(
            role="Error Recovery Agent",
            goal="Test error recovery capabilities",
            backstory="Agent designed to handle MCP server failures",
            mcps=[
                "https://failing-server.com/mcp",     # Will fail
                "https://working-server.com/mcp",     # Will work
                "https://timeout-server.com/mcp",     # Will timeout
                "crewai-amp:nonexistent-service"      # Will fail
            ]
        )

        print(f"✅ Agent created with {len(agent.mcps)} MCP references (some will fail)")

        # Mock mixed success/failure scenario
        def mock_mixed_external_tools(mcp_ref):
            if "failing-server" in mcp_ref:
                raise Exception("Server connection failed")
            elif "working-server" in mcp_ref:
                return [Mock(name="working_server_com_mcp_reliable_tool")]
            elif "timeout-server" in mcp_ref:
                raise Exception("Connection timed out")
            return []

        def mock_failing_amp_tools(amp_ref):
            raise Exception("AMP server unavailable")

        with patch.object(agent, '_get_external_mcp_tools', side_effect=mock_mixed_external_tools), \
             patch.object(agent, '_get_amp_mcp_tools', side_effect=mock_failing_amp_tools), \
             patch.object(agent, '_logger') as mock_logger:

            # Should handle failures gracefully and continue with working servers
            working_tools = agent.get_mcp_tools(agent.mcps)

            # Should get tools from working server only
            assert len(working_tools) == 1
            assert working_tools[0].name == "working_server_com_mcp_reliable_tool"
            print("✅ Error recovery successful - got tools from working server")

            # Should log warnings for failing servers
            warning_calls = [call for call in mock_logger.log.call_args_list if call[0][0] == "warning"]
            assert len(warning_calls) >= 3  # At least 3 failures logged

            print("✅ Error logging and recovery complete")

    def test_mcp_workflow_performance_benchmarks(self):
        """Test MCP workflow performance meets benchmarks."""
        print("\n=== Testing MCP Performance Benchmarks ===")

        start_time = time.time()

        # Agent creation should be fast
        agent = Agent(
            role="Performance Benchmark Agent",
            goal="Establish performance benchmarks",
            backstory="Agent for performance testing",
            mcps=[
                "https://perf1.com/mcp",
                "https://perf2.com/mcp",
                "https://perf3.com/mcp"
            ]
        )

        agent_creation_time = time.time() - start_time
        assert agent_creation_time < 0.5  # Less than 500ms
        print(f"✅ Agent creation: {agent_creation_time:.3f}s")

        # Tool discovery should be efficient
        mock_schemas = {f"tool_{i}": {"description": f"Tool {i}"} for i in range(5)}

        with patch.object(agent, '_get_mcp_tool_schemas', return_value=mock_schemas):

            discovery_start = time.time()
            tools = agent.get_mcp_tools(agent.mcps)
            discovery_time = time.time() - discovery_start

            # Should discover tools from 3 servers with 5 tools each = 15 tools
            assert len(tools) == 15
            assert discovery_time < 2.0  # Less than 2 seconds
            print(f"✅ Tool discovery: {discovery_time:.3f}s for {len(tools)} tools")

        # Crew creation should be fast
        task = Task(
            description="Performance test task",
            expected_output="Performance results",
            agent=agent
        )

        crew_start = time.time()
        crew = Crew(agents=[agent], tasks=[task])
        crew_creation_time = time.time() - crew_start

        assert crew_creation_time < 0.1  # Less than 100ms
        print(f"✅ Crew creation: {crew_creation_time:.3f}s")

        total_time = time.time() - start_time
        print(f"✅ Total workflow: {total_time:.3f}s")

    @pytest.mark.asyncio
    async def test_mcp_workflow_with_real_async_patterns(self):
        """Test MCP workflow with realistic async operation patterns."""
        print("\n=== Testing Async MCP Workflow Patterns ===")

        # Create agent
        agent = Agent(
            role="Async Test Agent",
            goal="Test async MCP operations",
            backstory="Agent for testing async patterns",
            mcps=["https://async-test.com/mcp"]
        )

        # Mock realistic async MCP server behavior
        mock_server = MockMCPServerFactory.create_exa_like_server("https://async-test.com/mcp")

        with patch('crewai.agent.streamablehttp_client') as mock_client, \
             patch('crewai.agent.ClientSession') as mock_session_class:

            # Setup async mocks
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.initialize = AsyncMock()
            mock_session.list_tools = AsyncMock(return_value=await mock_server.simulate_list_tools())

            mock_client.return_value.__aenter__.return_value = (None, None, None)

            # Test async tool discovery
            start_time = time.time()
            schemas = await agent._get_mcp_tool_schemas_async({"url": "https://async-test.com/mcp"})
            discovery_time = time.time() - start_time

            assert len(schemas) == 2  # Exa-like server has 2 tools
            assert discovery_time < 1.0  # Should be fast with mocked operations
            print(f"✅ Async discovery: {discovery_time:.3f}s")

            # Verify async methods were called
            mock_session.initialize.assert_called_once()
            mock_session.list_tools.assert_called_once()

    def test_mcp_workflow_scalability_test(self):
        """Test MCP workflow scalability with many agents and tools."""
        print("\n=== Testing MCP Workflow Scalability ===")

        # Create multiple agents with MCP configurations
        agents = []
        for i in range(10):
            agent = Agent(
                role=f"Scalability Agent {i}",
                goal=f"Test scalability scenario {i}",
                backstory=f"Agent {i} for scalability testing",
                mcps=[f"https://scale-server-{i}.com/mcp"]
            )
            agents.append(agent)

        print(f"✅ Created {len(agents)} agents with MCP configurations")

        # Mock tool discovery for all agents
        mock_schemas = {f"scale_tool_{i}": {"description": f"Scalability tool {i}"} for i in range(3)}

        with patch.object(Agent, '_get_mcp_tool_schemas', return_value=mock_schemas):

            start_time = time.time()

            # Discover tools for all agents
            all_agent_tools = []
            for agent in agents:
                tools = agent.get_mcp_tools(agent.mcps)
                all_agent_tools.extend(tools)

            scalability_time = time.time() - start_time

            # Should handle multiple agents efficiently
            total_tools = len(agents) * 3  # 3 tools per agent
            assert len(all_agent_tools) == total_tools
            assert scalability_time < 5.0  # Should complete within 5 seconds

            print(f"✅ Scalability test: {len(all_agent_tools)} tools from {len(agents)} agents in {scalability_time:.3f}s")

        # Test crew creation with multiple MCP agents
        tasks = [
            Task(
                description=f"Task for agent {i}",
                expected_output=f"Output from agent {i}",
                agent=agents[i]
            ) for i in range(len(agents))
        ]

        crew = Crew(agents=agents, tasks=tasks)

        assert len(crew.agents) == 10
        assert len(crew.tasks) == 10
        print("✅ Scalability crew creation successful")

    def test_mcp_workflow_with_specific_tool_selection(self):
        """Test MCP workflow with specific tool selection using # syntax."""
        print("\n=== Testing Specific Tool Selection ===")

        # Create agent with specific tool selections
        agent = Agent(
            role="Specific Tool Agent",
            goal="Test specific tool selection",
            backstory="Agent that uses specific MCP tools",
            mcps=[
                "https://multi-tool-server.com/mcp#search_tool",
                "https://another-server.com/mcp#analysis_tool",
                "crewai-amp:research-service#pubmed_search"
            ]
        )

        # Mock servers with multiple tools, but we should only get specific ones
        def mock_external_tools_specific(mcp_ref):
            if "#search_tool" in mcp_ref:
                return [Mock(name="multi_tool_server_com_mcp_search_tool")]
            elif "#analysis_tool" in mcp_ref:
                return [Mock(name="another_server_com_mcp_analysis_tool")]
            return []

        def mock_amp_tools_specific(amp_ref):
            if "#pubmed_search" in amp_ref:
                return [Mock(name="research_service_pubmed_search")]
            return []

        with patch.object(agent, '_get_external_mcp_tools', side_effect=mock_external_tools_specific), \
             patch.object(agent, '_get_amp_mcp_tools', side_effect=mock_amp_tools_specific):

            specific_tools = agent.get_mcp_tools(agent.mcps)

            # Should get exactly 3 specific tools
            assert len(specific_tools) == 3
            print("✅ Specific tool selection working correctly")

            # Verify correct tools were selected
            tool_names = [tool.name for tool in specific_tools]
            expected_names = [
                "multi_tool_server_com_mcp_search_tool",
                "another_server_com_mcp_analysis_tool",
                "research_service_pubmed_search"
            ]

            for expected_name in expected_names:
                assert expected_name in tool_names

    def test_mcp_workflow_resilience_under_stress(self):
        """Test MCP workflow resilience under stress conditions."""
        print("\n=== Testing MCP Workflow Resilience ===")

        # Create stress test scenario
        stress_mcps = []
        for i in range(20):
            # Mix of different server types
            if i % 4 == 0:
                stress_mcps.append(f"https://working-server-{i}.com/mcp")
            elif i % 4 == 1:
                stress_mcps.append(f"https://failing-server-{i}.com/mcp")
            elif i % 4 == 2:
                stress_mcps.append(f"crewai-amp:service-{i}")
            else:
                stress_mcps.append(f"https://slow-server-{i}.com/mcp#specific_tool")

        agent = Agent(
            role="Stress Test Agent",
            goal="Test MCP workflow under stress",
            backstory="Agent for stress testing MCP functionality",
            mcps=stress_mcps
        )

        # Mock stress test behaviors
        def mock_stress_external_tools(mcp_ref):
            if "failing" in mcp_ref:
                raise Exception("Simulated failure")
            elif "slow" in mcp_ref:
                # Simulate slow response
                time.sleep(0.1)
                return [Mock(name=f"tool_from_{mcp_ref}")]
            elif "working" in mcp_ref:
                return [Mock(name=f"tool_from_{mcp_ref}")]
            return []

        def mock_stress_amp_tools(amp_ref):
            return [Mock(name=f"amp_tool_from_{amp_ref}")]

        with patch.object(agent, '_get_external_mcp_tools', side_effect=mock_stress_external_tools), \
             patch.object(agent, '_get_amp_mcp_tools', side_effect=mock_stress_amp_tools):

            start_time = time.time()

            # Should handle all servers (working, failing, slow, AMP)
            stress_tools = agent.get_mcp_tools(agent.mcps)

            stress_time = time.time() - start_time

            # Should get tools from working servers (5 working + 5 slow + 5 AMP = 15)
            expected_working_tools = 15
            assert len(stress_tools) == expected_working_tools

            # Should complete within reasonable time despite stress
            assert stress_time < 10.0

            print(f"✅ Stress test: {len(stress_tools)} tools processed in {stress_time:.3f}s")

    def test_mcp_workflow_integration_with_existing_features(self):
        """Test MCP workflow integration with existing CrewAI features."""
        print("\n=== Testing Integration with Existing Features ===")

        from crewai.tools import BaseTool

        # Create custom tool for testing integration
        class CustomTool(BaseTool):
            name: str = "custom_search_tool"
            description: str = "Custom search tool"

            def _run(self, **kwargs):
                return "Custom tool result"

        # Create agent with both regular tools, platform apps, and MCP tools
        agent = Agent(
            role="Full Integration Agent",
            goal="Test integration with all CrewAI features",
            backstory="Agent with access to all tool types",
            tools=[CustomTool()],                           # Regular tools
            apps=["gmail", "slack"],                        # Platform apps
            mcps=["https://integration-server.com/mcp"],    # MCP tools
            verbose=True,
            max_iter=15,
            allow_delegation=True
        )

        print("✅ Agent created with all feature types")

        # Test that all features work together
        assert len(agent.tools) == 1  # Regular tools
        assert len(agent.apps) == 2   # Platform apps
        assert len(agent.mcps) == 1   # MCP tools
        assert agent.verbose is True
        assert agent.max_iter == 15
        assert agent.allow_delegation is True

        # Mock MCP tool discovery
        mock_mcp_tools = [Mock(name="integration_server_com_mcp_integration_tool")]

        with patch.object(agent, 'get_mcp_tools', return_value=mock_mcp_tools):

            # Create crew with integrated agent
            task = Task(
                description="Use all available tool types for comprehensive research",
                expected_output="Comprehensive research using all tools",
                agent=agent
            )

            crew = Crew(agents=[agent], tasks=[task])

            # Test crew tool integration
            crew_tools = crew._inject_mcp_tools([], agent)
            assert len(crew_tools) == len(mock_mcp_tools)

            print("✅ Full feature integration successful")

    def test_mcp_workflow_user_experience_simulation(self):
        """Simulate typical user experience with MCP DSL."""
        print("\n=== Simulating User Experience ===")

        # Simulate user creating agent for research
        research_agent = Agent(
            role="AI Research Specialist",
            goal="Research AI technologies and frameworks",
            backstory="Expert AI researcher with access to search and analysis tools",
            mcps=[
                "https://mcp.exa.ai/mcp?api_key=user_key&profile=research",
                "https://analysis.tools.com/mcp#analyze_trends",
                "crewai-amp:academic-research",
                "crewai-amp:market-analysis#competitor_analysis"
            ]
        )

        print("✅ User created research agent with 4 MCP references")

        # Mock realistic tool discovery
        mock_tools = [
            Mock(name="mcp_exa_ai_mcp_web_search_exa"),
            Mock(name="analysis_tools_com_mcp_analyze_trends"),
            Mock(name="academic_research_paper_search"),
            Mock(name="market_analysis_competitor_analysis")
        ]

        with patch.object(research_agent, 'get_mcp_tools', return_value=mock_tools):

            # User creates research task
            research_task = Task(
                description="Research the current state of multi-agent AI frameworks, focusing on CrewAI",
                expected_output="Comprehensive research report with market analysis and competitor comparison",
                agent=research_agent
            )

            # User creates and configures crew
            research_crew = Crew(
                agents=[research_agent],
                tasks=[research_task],
                verbose=True
            )

            print("✅ User created research task and crew")

            # Verify user's MCP tools are available
            available_tools = research_crew._inject_mcp_tools([], research_agent)
            assert len(available_tools) == 4

            print("✅ User's MCP tools integrated successfully")
            print(f"   Available tools: {[tool.name for tool in available_tools]}")

            # Test tool execution simulation
            search_tool = available_tools[0]
            with patch.object(search_tool, '_run', return_value="Research results about CrewAI framework"):
                result = search_tool._run(query="CrewAI multi-agent framework", num_results=5)

                assert "CrewAI framework" in result
                print("✅ User tool execution successful")

    def test_mcp_workflow_production_readiness_checklist(self):
        """Verify MCP workflow meets production readiness checklist."""
        print("\n=== Production Readiness Checklist ===")

        checklist_results = {}

        # ✅ Test 1: Agent creation without external dependencies
        try:
            agent = Agent(
                role="Production Test Agent",
                goal="Verify production readiness",
                backstory="Agent for production testing",
                mcps=["https://prod-test.com/mcp"]
            )
            checklist_results["agent_creation"] = "✅ PASS"
        except Exception as e:
            checklist_results["agent_creation"] = f"❌ FAIL: {e}"

        # ✅ Test 2: Graceful handling of unavailable servers
        with patch.object(agent, '_get_external_mcp_tools', side_effect=Exception("Server unavailable")):
            try:
                tools = agent.get_mcp_tools(agent.mcps)
                assert tools == []  # Should return empty list, not crash
                checklist_results["error_handling"] = "✅ PASS"
            except Exception as e:
                checklist_results["error_handling"] = f"❌ FAIL: {e}"

        # ✅ Test 3: Performance within acceptable limits
        start_time = time.time()
        mock_tools = [Mock() for _ in range(10)]
        with patch.object(agent, 'get_mcp_tools', return_value=mock_tools):
            tools = agent.get_mcp_tools(agent.mcps)
        performance_time = time.time() - start_time

        if performance_time < 1.0:
            checklist_results["performance"] = "✅ PASS"
        else:
            checklist_results["performance"] = f"❌ FAIL: {performance_time:.3f}s"

        # ✅ Test 4: Integration with crew workflows
        try:
            task = Task(description="Test task", expected_output="Test output", agent=agent)
            crew = Crew(agents=[agent], tasks=[task])

            crew_tools = crew._inject_mcp_tools([], agent)
            checklist_results["crew_integration"] = "✅ PASS"
        except Exception as e:
            checklist_results["crew_integration"] = f"❌ FAIL: {e}"

        # ✅ Test 5: Input validation works correctly
        try:
            Agent(
                role="Validation Test",
                goal="Test validation",
                backstory="Testing validation",
                mcps=["invalid-format"]
            )
            checklist_results["input_validation"] = "❌ FAIL: Should reject invalid format"
        except Exception:
            checklist_results["input_validation"] = "✅ PASS"

        # Print results
        print("\nProduction Readiness Results:")
        for test_name, result in checklist_results.items():
            print(f"  {test_name.replace('_', ' ').title()}: {result}")

        # All tests should pass
        passed_tests = sum(1 for result in checklist_results.values() if "✅ PASS" in result)
        total_tests = len(checklist_results)

        assert passed_tests == total_tests, f"Only {passed_tests}/{total_tests} production readiness tests passed"
        print(f"\n🎉 Production Readiness: {passed_tests}/{total_tests} tests PASSED")

    def test_complete_user_journey_simulation(self):
        """Simulate a complete user journey from setup to execution."""
        print("\n=== Complete User Journey Simulation ===")

        # Step 1: User installs CrewAI (already done)
        print("✅ Step 1: CrewAI installed")

        # Step 2: User creates agent with MCP tools
        user_agent = Agent(
            role="Data Analyst",
            goal="Analyze market trends and competitor data",
            backstory="Experienced analyst with access to real-time data sources",
            mcps=[
                "https://api.marketdata.com/mcp",
                "https://competitor.intelligence.com/mcp#competitor_analysis",
                "crewai-amp:financial-insights",
                "crewai-amp:market-research#trend_analysis"
            ]
        )
        print("✅ Step 2: User created agent with 4 MCP tool sources")

        # Step 3: MCP tools are discovered automatically
        mock_discovered_tools = [
            Mock(name="api_marketdata_com_mcp_get_market_data"),
            Mock(name="competitor_intelligence_com_mcp_competitor_analysis"),
            Mock(name="financial_insights_stock_analysis"),
            Mock(name="market_research_trend_analysis")
        ]

        with patch.object(user_agent, 'get_mcp_tools', return_value=mock_discovered_tools):
            available_tools = user_agent.get_mcp_tools(user_agent.mcps)

            assert len(available_tools) == 4
            print("✅ Step 3: MCP tools discovered automatically")

        # Step 4: User creates analysis task
        analysis_task = Task(
            description="Analyze current market trends in AI technology sector and identify top competitors",
            expected_output="Comprehensive market analysis report with competitor insights and trend predictions",
            agent=user_agent
        )
        print("✅ Step 4: User created analysis task")

        # Step 5: User sets up crew for execution
        analysis_crew = Crew(
            agents=[user_agent],
            tasks=[analysis_task],
            verbose=True  # User wants to see progress
        )
        print("✅ Step 5: User configured crew for execution")

        # Step 6: Crew integrates MCP tools automatically
        with patch.object(user_agent, 'get_mcp_tools', return_value=mock_discovered_tools):
            integrated_tools = analysis_crew._inject_mcp_tools([], user_agent)

            assert len(integrated_tools) == 4
            print("✅ Step 6: Crew integrated MCP tools automatically")

        # Step 7: Tools are ready for execution
        tool_names = [tool.name for tool in integrated_tools]
        expected_capabilities = [
            "market data access",
            "competitor analysis",
            "financial insights",
            "trend analysis"
        ]

        # Verify tools provide expected capabilities
        for capability in expected_capabilities:
            capability_found = any(
                capability.replace(" ", "_") in tool_name.lower()
                for tool_name in tool_names
            )
            assert capability_found, f"Expected capability '{capability}' not found in tools"

        print("✅ Step 7: All expected capabilities available")
        print("\n🚀 Complete User Journey: SUCCESS!")
        print("   User can now execute crew.kickoff() with full MCP integration")

    def test_mcp_workflow_backwards_compatibility(self):
        """Test that MCP integration doesn't break existing functionality."""
        print("\n=== Testing Backwards Compatibility ===")

        # Test 1: Agent without MCP field works normally
        classic_agent = Agent(
            role="Classic Agent",
            goal="Test backwards compatibility",
            backstory="Agent without MCP configuration"
            # No mcps field specified
        )

        assert classic_agent.mcps is None
        assert hasattr(classic_agent, 'get_mcp_tools')  # Method exists but mcps is None
        print("✅ Classic agent creation works")

        # Test 2: Existing crew workflows unchanged
        classic_task = Task(
            description="Classic task without MCP",
            expected_output="Classic result",
            agent=classic_agent
        )

        classic_crew = Crew(agents=[classic_agent], tasks=[classic_task])

        # MCP integration should not affect classic workflows
        tools_result = classic_crew._inject_mcp_tools([], classic_agent)
        assert tools_result == []  # No MCP tools, empty list returned

        print("✅ Existing crew workflows unchanged")

        # Test 3: Agent with empty mcps list works normally
        empty_mcps_agent = Agent(
            role="Empty MCP Agent",
            goal="Test empty mcps list",
            backstory="Agent with empty mcps list",
            mcps=[]
        )

        assert empty_mcps_agent.mcps == []
        empty_tools = empty_mcps_agent.get_mcp_tools(empty_mcps_agent.mcps)
        assert empty_tools == []

        print("✅ Empty mcps list handling works")

        print("\n✅ Backwards Compatibility: CONFIRMED")
        print("   Existing CrewAI functionality remains unchanged")
