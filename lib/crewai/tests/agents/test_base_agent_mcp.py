"""Tests for BaseAgent MCP field validation and functionality."""

import pytest
from unittest.mock import Mock, patch
from pydantic import ValidationError

# Import from the source directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agent import Agent


class TestMCPAgent(BaseAgent):
    """Test implementation of BaseAgent for MCP testing."""

    def execute_task(self, task, context=None, tools=None):
        return "Test execution"

    def create_agent_executor(self, tools=None):
        pass

    def get_delegation_tools(self, agents):
        return []

    def get_platform_tools(self, apps):
        return []

    def get_mcp_tools(self, mcps):
        return []


class TestBaseAgentMCPField:
    """Test suite for BaseAgent MCP field validation and functionality."""

    def test_mcp_field_exists(self):
        """Test that mcps field exists on BaseAgent."""
        agent = TestMCPAgent(
            role="Test Agent",
            goal="Test MCP field",
            backstory="Testing BaseAgent MCP field"
        )

        assert hasattr(agent, 'mcps')
        assert agent.mcps is None  # Default value

    def test_mcp_field_accepts_none(self):
        """Test that mcps field accepts None value."""
        agent = TestMCPAgent(
            role="Test Agent",
            goal="Test MCP field",
            backstory="Testing BaseAgent MCP field",
            mcps=None
        )

        assert agent.mcps is None

    def test_mcp_field_accepts_empty_list(self):
        """Test that mcps field accepts empty list."""
        agent = TestMCPAgent(
            role="Test Agent",
            goal="Test MCP field",
            backstory="Testing BaseAgent MCP field",
            mcps=[]
        )

        assert agent.mcps == []

    def test_mcp_field_accepts_valid_https_urls(self):
        """Test that mcps field accepts valid HTTPS URLs."""
        valid_urls = [
            "https://api.example.com/mcp",
            "https://mcp.server.org/endpoint",
            "https://localhost:8080/mcp"
        ]

        agent = TestMCPAgent(
            role="Test Agent",
            goal="Test MCP field",
            backstory="Testing BaseAgent MCP field",
            mcps=valid_urls
        )

        # Field validator may reorder items due to set() deduplication
        assert len(agent.mcps) == len(valid_urls)
        assert all(url in agent.mcps for url in valid_urls)

    def test_mcp_field_accepts_valid_crewai_amp_references(self):
        """Test that mcps field accepts valid CrewAI AMP references."""
        valid_amp_refs = [
            "crewai-amp:weather-service",
            "crewai-amp:financial-data",
            "crewai-amp:research-tools"
        ]

        agent = TestMCPAgent(
            role="Test Agent",
            goal="Test MCP field",
            backstory="Testing BaseAgent MCP field",
            mcps=valid_amp_refs
        )

        # Field validator may reorder items due to set() deduplication
        assert len(agent.mcps) == len(valid_amp_refs)
        assert all(ref in agent.mcps for ref in valid_amp_refs)

    def test_mcp_field_accepts_mixed_valid_references(self):
        """Test that mcps field accepts mixed valid references."""
        mixed_refs = [
            "https://api.example.com/mcp",
            "crewai-amp:weather-service",
            "https://mcp.exa.ai/mcp?api_key=test",
            "crewai-amp:financial-data"
        ]

        agent = TestMCPAgent(
            role="Test Agent",
            goal="Test MCP field",
            backstory="Testing BaseAgent MCP field",
            mcps=mixed_refs
        )

        # Field validator may reorder items due to set() deduplication
        assert len(agent.mcps) == len(mixed_refs)
        assert all(ref in agent.mcps for ref in mixed_refs)

    def test_mcp_field_rejects_invalid_formats(self):
        """Test that mcps field rejects invalid URL formats."""
        invalid_refs = [
            "http://insecure.com/mcp",  # HTTP not allowed
            "invalid-format",           # No protocol
            "ftp://example.com/mcp",    # Wrong protocol
            "crewai:invalid",           # Wrong AMP format
            "",                         # Empty string
        ]

        for invalid_ref in invalid_refs:
            with pytest.raises(ValidationError, match="Invalid MCP reference"):
                TestMCPAgent(
                    role="Test Agent",
                    goal="Test MCP field",
                    backstory="Testing BaseAgent MCP field",
                    mcps=[invalid_ref]
                )

    def test_mcp_field_removes_duplicates(self):
        """Test that mcps field removes duplicate references."""
        mcps_with_duplicates = [
            "https://api.example.com/mcp",
            "crewai-amp:weather-service",
            "https://api.example.com/mcp",  # Duplicate
            "crewai-amp:weather-service"    # Duplicate
        ]

        agent = TestMCPAgent(
            role="Test Agent",
            goal="Test MCP field",
            backstory="Testing BaseAgent MCP field",
            mcps=mcps_with_duplicates
        )

        # Should contain only unique references
        assert len(agent.mcps) == 2
        assert "https://api.example.com/mcp" in agent.mcps
        assert "crewai-amp:weather-service" in agent.mcps

    def test_mcp_field_validates_list_type(self):
        """Test that mcps field validates list type."""
        with pytest.raises(ValidationError):
            TestMCPAgent(
                role="Test Agent",
                goal="Test MCP field",
                backstory="Testing BaseAgent MCP field",
                mcps="not-a-list"  # Should be list[str]
            )

    def test_abstract_get_mcp_tools_method_exists(self):
        """Test that get_mcp_tools abstract method exists."""
        assert hasattr(BaseAgent, 'get_mcp_tools')

        # Verify it's abstract by checking it's in __abstractmethods__
        assert 'get_mcp_tools' in BaseAgent.__abstractmethods__

    def test_concrete_implementation_must_implement_get_mcp_tools(self):
        """Test that concrete implementations must implement get_mcp_tools."""
        # This should work - TestMCPAgent implements get_mcp_tools
        agent = TestMCPAgent(
            role="Test Agent",
            goal="Test MCP field",
            backstory="Testing BaseAgent MCP field"
        )

        assert hasattr(agent, 'get_mcp_tools')
        assert callable(agent.get_mcp_tools)

    def test_copy_method_excludes_mcps_field(self):
        """Test that copy method excludes mcps field from being copied."""
        agent = TestMCPAgent(
            role="Test Agent",
            goal="Test MCP field",
            backstory="Testing BaseAgent MCP field",
            mcps=["https://api.example.com/mcp"]
        )

        copied_agent = agent.copy()

        # MCP field should be excluded from copy
        assert copied_agent.mcps is None or copied_agent.mcps == []

    def test_model_validation_pipeline_with_mcps(self):
        """Test model validation pipeline with mcps field."""
        # Test validation runs correctly through entire pipeline
        agent = TestMCPAgent(
            role="Test Agent",
            goal="Test MCP field",
            backstory="Testing BaseAgent MCP field",
            mcps=["https://api.example.com/mcp", "crewai-amp:test-service"]
        )

        # Verify all required fields are set
        assert agent.role == "Test Agent"
        assert agent.goal == "Test MCP field"
        assert agent.backstory == "Testing BaseAgent MCP field"
        assert len(agent.mcps) == 2

    def test_mcp_field_description_is_correct(self):
        """Test that mcps field has correct description."""
        # Get field info from model
        fields = BaseAgent.model_fields
        mcps_field = fields.get('mcps')

        assert mcps_field is not None
        assert "MCP server references" in mcps_field.description
        assert "https://" in mcps_field.description
        assert "crewai-amp:" in mcps_field.description
        assert "#tool_name" in mcps_field.description


class TestAgentMCPFieldIntegration:
    """Test MCP field integration with concrete Agent class."""

    def test_agent_class_has_mcp_field(self):
        """Test that concrete Agent class inherits MCP field."""
        agent = Agent(
            role="Test Agent",
            goal="Test MCP integration",
            backstory="Testing Agent MCP field",
            mcps=["https://api.example.com/mcp"]
        )

        assert hasattr(agent, 'mcps')
        assert agent.mcps == ["https://api.example.com/mcp"]

    def test_agent_class_implements_get_mcp_tools(self):
        """Test that concrete Agent class implements get_mcp_tools."""
        agent = Agent(
            role="Test Agent",
            goal="Test MCP integration",
            backstory="Testing Agent MCP field"
        )

        assert hasattr(agent, 'get_mcp_tools')
        assert callable(agent.get_mcp_tools)

        # Test it can be called
        result = agent.get_mcp_tools([])
        assert isinstance(result, list)

    def test_agent_mcp_field_validation_integration(self):
        """Test MCP field validation works with concrete Agent class."""
        # Valid case
        agent = Agent(
            role="Test Agent",
            goal="Test MCP integration",
            backstory="Testing Agent MCP field",
            mcps=["https://mcp.exa.ai/mcp", "crewai-amp:research-tools"]
        )

        assert len(agent.mcps) == 2

        # Invalid case
        with pytest.raises(ValidationError, match="Invalid MCP reference"):
            Agent(
                role="Test Agent",
                goal="Test MCP integration",
                backstory="Testing Agent MCP field",
                mcps=["invalid-format"]
            )

    def test_agent_docstring_mentions_mcps(self):
        """Test that Agent class docstring mentions mcps field."""
        docstring = Agent.__doc__

        assert docstring is not None
        assert "mcps" in docstring.lower()

    @patch('crewai.agent.create_llm')
    def test_agent_initialization_with_mcps_field(self, mock_create_llm):
        """Test complete Agent initialization with mcps field."""
        mock_create_llm.return_value = Mock()

        agent = Agent(
            role="MCP Test Agent",
            goal="Test complete MCP integration",
            backstory="Agent for testing MCP functionality",
            mcps=[
                "https://mcp.exa.ai/mcp?api_key=test",
                "crewai-amp:financial-data#get_stock_price"
            ],
            verbose=True
        )

        # Verify agent is properly initialized
        assert agent.role == "MCP Test Agent"
        assert len(agent.mcps) == 2
        assert agent.verbose is True

        # Verify MCP-specific functionality is available
        assert hasattr(agent, 'get_mcp_tools')
        assert hasattr(agent, '_get_external_mcp_tools')
        assert hasattr(agent, '_get_amp_mcp_tools')
