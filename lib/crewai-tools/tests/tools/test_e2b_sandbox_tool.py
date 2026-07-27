from importlib.metadata import version
import sys
from types import ModuleType
from unittest.mock import MagicMock, call, patch

from crewai_tools.tools.e2b_sandbox_tool.e2b_base_tool import (
    E2B_INTEGRATION,
    E2BBaseTool,
)


class ConcreteE2BTool(E2BBaseTool):
    name: str = "Test E2B tool"
    description: str = "Test tool for the E2B base lifecycle."

    def _run(self) -> None:
        return None


def test_sets_versioned_integration_before_creating_sandbox() -> None:
    calls = MagicMock()
    connection_config = MagicMock()
    connection_config.set_integration.side_effect = calls.set_integration
    sandbox_class = MagicMock()
    sandbox_class.create.side_effect = calls.create
    e2b_module = ModuleType("e2b")
    e2b_module.ConnectionConfig = connection_config
    tool = ConcreteE2BTool()

    with (
        patch.dict(sys.modules, {"e2b": e2b_module}),
        patch.object(tool, "_import_sandbox_class", return_value=sandbox_class),
    ):
        tool._acquire_sandbox()

    expected_integration = f"crewai-tools/{version('crewai-tools')}"
    assert E2B_INTEGRATION == expected_integration
    assert calls.mock_calls == [
        call.set_integration(expected_integration),
        call.create(timeout=300),
    ]


def test_sets_versioned_integration_before_connecting_to_sandbox() -> None:
    calls = MagicMock()
    connection_config = MagicMock()
    connection_config.set_integration.side_effect = calls.set_integration
    sandbox_class = MagicMock()
    sandbox_class.connect.side_effect = calls.connect
    e2b_module = ModuleType("e2b")
    e2b_module.ConnectionConfig = connection_config
    tool = ConcreteE2BTool(sandbox_id="sandbox-id")

    with (
        patch.dict(sys.modules, {"e2b": e2b_module}),
        patch.object(tool, "_import_sandbox_class", return_value=sandbox_class),
    ):
        tool._acquire_sandbox()

    assert calls.mock_calls == [
        call.set_integration(E2B_INTEGRATION),
        call.connect("sandbox-id", timeout=300),
    ]
