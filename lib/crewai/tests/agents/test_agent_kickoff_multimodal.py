from __future__ import annotations

from typing import Any

import pytest

from crewai.agent.core import Agent
from crewai.tools.agent_tools.add_image_tool import AddImageTool
from crewai.tools.base_tool import BaseTool


class _SpyLiteAgent:
    """LiteAgent stand-in that just records the tools passed in."""

    def __init__(
        self,
        *args: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.tools = list(tools or [])

    def kickoff(self, messages: Any) -> dict[str, Any]:
        # We don't care about LLM behaviour here, only tool wiring.
        return {"messages": messages, "tools": self.tools}

    async def kickoff_async(self, messages: Any) -> dict[str, Any]:
        # Mirror the sync API for async tests.
        return {"messages": messages, "tools": self.tools}


class _DummyTool(BaseTool):
    name: str = "DummyTool"
    description: str = "A dummy tool for testing"

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        # Minimal implementation; we only care that the tool is instantiable.
        return "dummy output"

    name: str = "dummy"
    description: str = "A dummy tool for testing"


def _patch_lite_agent(monkeypatch, spy_cls=_SpyLiteAgent) -> None:
    import crewai.agent.core as agent_core

    monkeypatch.setattr(agent_core, "LiteAgent", spy_cls)


def test_agent_kickoff_multimodal_adds_add_image_tool_once(monkeypatch) -> None:
    _patch_lite_agent(monkeypatch)

    agent = Agent(
        role="Image agent",
        goal="Handle image inputs",
        backstory="Test agent for multimodal kickoff wiring.",
        tools=[],  # start with no tools
        multimodal=True,
    )

    result1 = agent.kickoff("Describe https://example.com/image.png")
    tools1 = result1["tools"]

    # We should get exactly one AddImageTool
    add_image_tools_1 = [t for t in tools1 if isinstance(t, AddImageTool)]
    assert add_image_tools_1, "Expected AddImageTool to be injected for multimodal agent kickoff"
    assert len(add_image_tools_1) == 1, "Expected exactly one AddImageTool on first kickoff run"

    # A second kickoff should not accumulate extra AddImageTool instances.
    result2 = agent.kickoff("Describe https://example.com/another.png")
    tools2 = result2["tools"]
    add_image_tools_2 = [t for t in tools2 if isinstance(t, AddImageTool)]
    assert len(add_image_tools_2) == 1, "Expected no duplicate AddImageTool instances across multiple kickoffs"


def test_agent_kickoff_multimodal_does_not_duplicate_existing_add_image_tool(
    monkeypatch,
) -> None:
    _patch_lite_agent(monkeypatch)

    # Agent already has an AddImageTool in its tools list.
    agent = Agent(
        role="Image agent",
        goal="Handle image inputs",
        backstory="Test agent with existing AddImageTool.",
        tools=[AddImageTool()],
        multimodal=True,
    )

    result = agent.kickoff("Describe https://example.com/existing.png")
    tools = result["tools"]
    add_image_tools = [t for t in tools if isinstance(t, AddImageTool)]
    assert (
        len(add_image_tools) == 1
    ), "Agent.kickoff should not duplicate an existing AddImageTool"


def test_agent_kickoff_does_not_mutate_agent_tools(monkeypatch) -> None:
    _patch_lite_agent(monkeypatch)

    # Start with a single dummy tool in the agent's tools list.
    dummy = _DummyTool()
    agent = Agent(
        role="Image agent",
        goal="Handle image inputs",
        backstory="Agent with existing non-image tool.",
        tools=[dummy],
        multimodal=True,
    )

    original_tools_id = id(agent.tools)
    original_len = len(agent.tools)

    result = agent.kickoff("Describe https://example.com/with-dummy.png")
    tools = result["tools"]

    # LiteAgent should see both the original tool and AddImageTool.
    assert any(isinstance(t, _DummyTool) for t in tools), "Expected DummyTool to be preserved for kickoff"
    assert any(isinstance(t, AddImageTool) for t in tools), "Expected AddImageTool to be injected for kickoff"

    # Agent.tools should not be mutated in-place or grow with extra tools.
    assert len(agent.tools) == original_len, "Agent.tools length should not change during kickoff"
    assert id(agent.tools) == original_tools_id, "Agent.tools reference should not be replaced during kickoff"


@pytest.mark.asyncio
async def test_agent_kickoff_async_multimodal_adds_add_image_tool_once(monkeypatch) -> None:
    _patch_lite_agent(monkeypatch)

    agent = Agent(
        role="Image agent",
        goal="Handle image inputs async",
        backstory="Test agent for async multimodal kickoff wiring.",
        tools=[],
        multimodal=True,
    )

    result1 = await agent.kickoff_async("Describe https://example.com/image.png")
    tools1 = result1["tools"]
    add_image_tools_1 = [t for t in tools1 if isinstance(t, AddImageTool)]
    assert add_image_tools_1, "Expected AddImageTool to be injected for multimodal agent kickoff_async"
    assert len(add_image_tools_1) == 1, "Expected exactly one AddImageTool on first kickoff_async run"

    result2 = await agent.kickoff_async("Describe https://example.com/another.png")
    tools2 = result2["tools"]
    add_image_tools_2 = [t for t in tools2 if isinstance(t, AddImageTool)]
    assert len(add_image_tools_2) == 1, "Expected no duplicate AddImageTool instances across multiple kickoff_async runs"


@pytest.mark.asyncio
async def test_agent_kickoff_async_multimodal_does_not_duplicate_existing_add_image_tool(
    monkeypatch,
) -> None:
    _patch_lite_agent(monkeypatch)

    agent = Agent(
        role="Image agent",
        goal="Handle image inputs async",
        backstory="Async agent with existing AddImageTool.",
        tools=[AddImageTool()],
        multimodal=True,
    )

    result = await agent.kickoff_async("Describe https://example.com/existing.png")
    tools = result["tools"]
    add_image_tools = [t for t in tools if isinstance(t, AddImageTool)]
    assert (
        len(add_image_tools) == 1
    ), "Agent.kickoff_async should not duplicate an existing AddImageTool"
