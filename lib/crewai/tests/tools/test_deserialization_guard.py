"""Tests for the CREWAI_DESERIALIZE_CALLBACKS guard on serialized tool dicts."""

from __future__ import annotations

import json
import sys

from pydantic import BaseModel, ValidationError
import pytest

from crewai.tools.agent_tools.add_image_tool import AddImageTool
from crewai.tools.base_tool import (
    _TOOL_TYPE_REGISTRY,
    BaseTool,
    _default_cache_function,
)


ADD_IMAGE_MODULE = "crewai.tools.agent_tools.add_image_tool"
ADD_IMAGE_TYPE = f"{ADD_IMAGE_MODULE}.AddImageTool"

class _ToolHolder(BaseModel):
    """BaseTool hydrates ``tool_type`` dicts when it is a field type, as in Agent.tools."""

    tool: BaseTool


def restore_tool(data: dict[str, object]) -> BaseTool:
    return _ToolHolder.model_validate({"tool": data}).tool


@pytest.fixture(autouse=True)
def _no_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CREWAI_DESERIALIZE_CALLBACKS", raising=False)


def _serialized_add_image_tool() -> dict[str, object]:
    return json.loads(AddImageTool().model_dump_json())


@pytest.mark.parametrize("tool_type", ["os.system", "subprocess.Popen"])
def test_blocks_third_party_tool_type_without_opt_in(tool_type: str) -> None:
    with pytest.raises(
        ValidationError, match=f"Refusing to resolve tool type '{tool_type}'"
    ):
        restore_tool({"tool_type": tool_type, "name": "evil", "description": "x"})


def test_rejects_non_tool_target_even_with_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CREWAI_DESERIALIZE_CALLBACKS", "1")

    with pytest.raises(
        ValidationError, match="Tool type 'os.system' is not a BaseTool subclass"
    ):
        restore_tool({"tool_type": "os.system", "name": "evil", "description": "x"})


def test_first_party_tool_type_restores_in_a_fresh_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resuming a checkpoint must not need the opt-in for crewai's own tools."""
    monkeypatch.delitem(sys.modules, ADD_IMAGE_MODULE)
    monkeypatch.delitem(_TOOL_TYPE_REGISTRY, ADD_IMAGE_TYPE)

    tool = restore_tool(_serialized_add_image_tool())

    assert type(tool).__qualname__ == "AddImageTool"
    assert tool.name == AddImageTool().name


def test_default_cache_function_restores_without_opt_in() -> None:
    data = _serialized_add_image_tool()
    assert data["cache_function"] == "crewai.tools.base_tool._default_cache_function"

    tool = restore_tool(data)

    assert tool.cache_function is _default_cache_function


def test_custom_cache_function_requires_opt_in() -> None:
    data = _serialized_add_image_tool()
    data["cache_function"] = "os.system"

    with pytest.raises(
        ValidationError, match="Refusing to resolve callback path 'os.system'"
    ):
        restore_tool(data)
