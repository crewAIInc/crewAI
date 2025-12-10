from typing import Any

import pytest
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.agent_adapters.base_agent_adapter import BaseAgentAdapter
from crewai.tools.base_tool import BaseTool
from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from pydantic import BaseModel


# Concrete implementation for testing
class ConcreteAgentAdapter(BaseAgentAdapter):
    def configure_tools(
        self, tools: list[BaseTool] | None = None, **kwargs: Any
    ) -> None:
        # Simple implementation for testing
        self.tools = tools or []

    def execute_task(
        self,
        task: Any,
        context: str | None = None,
        tools: list[Any] | None = None,
    ) -> str:
        # Dummy implementation needed due to BaseAgent inheritance
        return "Task executed"

    def create_agent_executor(self, tools: list[BaseTool] | None = None) -> Any:
        # Dummy implementation
        return None

    def get_delegation_tools(
        self, tools: list[BaseTool], tool_map: dict[str, BaseTool] | None
    ) -> list[BaseTool]:
        # Dummy implementation
        return []

    def _parse_output(self, agent_output: Any, token_process: TokenProcess):
        # Dummy implementation
        pass

    def get_output_converter(self, tools: list[BaseTool] | None = None) -> Any:
        # Dummy implementation
        return None

    def get_platform_tools(self, apps: Any) -> list[BaseTool]:
        # Dummy implementation
        return []

    def get_mcp_tools(self, mcps: list[str]) -> list[BaseTool]:
        # Dummy implementation for MCP tools
        return []

    async def aexecute_task(
        self,
        task: Any,
        context: str | None = None,
        tools: list[Any] | None = None,
    ) -> str:
        # Dummy async implementation
        return "Task executed"


def test_base_agent_adapter_initialization():
    """Test initialization of the concrete agent adapter."""
    adapter = ConcreteAgentAdapter(
        role="test role", goal="test goal", backstory="test backstory"
    )
    assert isinstance(adapter, BaseAgent)
    assert isinstance(adapter, BaseAgentAdapter)
    assert adapter.role == "test role"
    assert adapter._agent_config is None
    assert adapter.adapted_structured_output is False


def test_base_agent_adapter_initialization_with_config():
    """Test initialization with agent_config."""
    config = {"model": "gpt-4"}
    adapter = ConcreteAgentAdapter(
        agent_config=config,
        role="test role",
        goal="test goal",
        backstory="test backstory",
    )
    assert adapter._agent_config == config


def test_configure_tools_method_exists():
    """Test that configure_tools method exists and can be called."""
    adapter = ConcreteAgentAdapter(
        role="test role", goal="test goal", backstory="test backstory"
    )
    # Create dummy tools if needed, or pass None
    tools = []
    adapter.configure_tools(tools)
    assert hasattr(adapter, "tools")
    assert adapter.tools == tools


def test_configure_structured_output_method_exists():
    """Test that configure_structured_output method exists and can be called."""
    adapter = ConcreteAgentAdapter(
        role="test role", goal="test goal", backstory="test backstory"
    )

    # Define a dummy structure or pass None/Any
    class DummyOutput(BaseModel):
        data: str

    structured_output = DummyOutput
    adapter.configure_structured_output(structured_output)
    # Add assertions here if configure_structured_output modifies state
    # For now, just ensuring it runs without error is sufficient


def test_base_agent_adapter_inherits_base_agent():
    """Test that BaseAgentAdapter inherits from BaseAgent."""
    assert issubclass(BaseAgentAdapter, BaseAgent)


class ConcreteAgentAdapterWithoutRequiredMethods(BaseAgentAdapter):
    pass


def test_base_agent_adapter_fails_without_required_methods():
    """Test that BaseAgentAdapter fails without required methods."""
    with pytest.raises(TypeError):
        ConcreteAgentAdapterWithoutRequiredMethods()  # type: ignore
