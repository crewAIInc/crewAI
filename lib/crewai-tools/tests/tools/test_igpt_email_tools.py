from datetime import date
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

from crewai_tools import IgptEmailAskTool, IgptEmailSearchTool
from crewai_tools.tools import IgptEmailAskTool as IgptEmailAskToolFromTools
from crewai_tools.tools import IgptEmailSearchTool as IgptEmailSearchToolFromTools
import pytest


def _mock_igpt_module(client: MagicMock) -> tuple[dict[str, ModuleType], MagicMock]:
    module = ModuleType("igptai")
    igpt_cls = MagicMock(return_value=client)
    module.IGPT = igpt_cls  # type: ignore[attr-defined]
    return {"igptai": module}, igpt_cls


def test_tools_are_exported():
    assert IgptEmailAskTool is IgptEmailAskToolFromTools
    assert IgptEmailSearchTool is IgptEmailSearchToolFromTools


def test_ask_tool_initializes_with_explicit_credentials():
    client = MagicMock()
    module_patch, igpt_cls = _mock_igpt_module(client)

    with patch.dict(sys.modules, module_patch):
        tool = IgptEmailAskTool(api_key="api-key", user="user-id")

    igpt_cls.assert_called_once_with(api_key="api-key", user="user-id")
    assert isinstance(tool, IgptEmailAskTool)


def test_ask_tool_initializes_with_env_credentials():
    client = MagicMock()
    module_patch, igpt_cls = _mock_igpt_module(client)

    with (
        patch.dict(sys.modules, module_patch),
        patch.dict(
            os.environ,
            {"IGPT_API_KEY": "env-api-key", "IGPT_API_USER": "env-user-id"},
            clear=True,
        ),
    ):
        IgptEmailAskTool()

    igpt_cls.assert_called_once_with(api_key="env-api-key", user="env-user-id")


def test_ask_tool_requires_api_key():
    with patch.dict(os.environ, {"IGPT_API_USER": "env-user-id"}, clear=True):
        with pytest.raises(ValueError, match="IGPT_API_KEY"):
            IgptEmailAskTool()


def test_ask_tool_requires_user():
    with patch.dict(os.environ, {"IGPT_API_KEY": "env-api-key"}, clear=True):
        with pytest.raises(ValueError, match="IGPT_API_USER"):
            IgptEmailAskTool()


def test_ask_tool_run_json_output():
    client = MagicMock()
    client.recall.ask.return_value = {"answer": "done", "citations": ["msg-1"]}
    module_patch, _ = _mock_igpt_module(client)

    with patch.dict(sys.modules, module_patch):
        tool = IgptEmailAskTool(api_key="api-key", user="user-id")
        result = tool._run(question="What did we decide on pricing?", output_format="json")

    client.recall.ask.assert_called_once_with(
        input="What did we decide on pricing?",
        quality="cef-1-normal",
        output_format="json",
    )
    assert result == '{"answer": "done", "citations": ["msg-1"]}'


def test_ask_tool_run_schema_output():
    client = MagicMock()
    client.recall.ask.return_value = {"status": "ok"}
    module_patch, _ = _mock_igpt_module(client)
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
    }

    with patch.dict(sys.modules, module_patch):
        tool = IgptEmailAskTool(api_key="api-key", user="user-id")
        tool._run(
            question="Summarize open action items.",
            output_format="schema",
            output_schema=schema,
        )

    client.recall.ask.assert_called_once_with(
        input="Summarize open action items.",
        quality="cef-1-normal",
        output_format={"strict": True, "schema": schema},
    )


def test_ask_tool_schema_requires_output_schema():
    client = MagicMock()
    module_patch, _ = _mock_igpt_module(client)

    with patch.dict(sys.modules, module_patch):
        tool = IgptEmailAskTool(api_key="api-key", user="user-id")
        with pytest.raises(ValueError, match="output_schema"):
            tool._run(
                question="Summarize open action items.",
                output_format="schema",
            )


def test_search_tool_run_with_dates():
    client = MagicMock()
    client.recall.search.return_value = [{"id": "thread-1"}]
    module_patch, igpt_cls = _mock_igpt_module(client)

    with patch.dict(sys.modules, module_patch):
        tool = IgptEmailSearchTool(api_key="api-key", user="user-id")
        result = tool._run(
            query="pricing discussion",
            date_from=date(2026, 2, 1),
            date_to=date(2026, 2, 28),
            max_results=5,
        )

    igpt_cls.assert_called_once_with(api_key="api-key", user="user-id")
    client.recall.search.assert_called_once_with(
        query="pricing discussion",
        quality="cef-1-normal",
        max_results=5,
        date_from="2026-02-01",
        date_to="2026-02-28",
    )
    assert result == '[{"id": "thread-1"}]'


def test_search_tool_rejects_invalid_date_range():
    client = MagicMock()
    module_patch, _ = _mock_igpt_module(client)

    with patch.dict(sys.modules, module_patch):
        tool = IgptEmailSearchTool(api_key="api-key", user="user-id")
        with pytest.raises(ValueError, match="date_from"):
            tool._run(
                query="handoff notes",
                date_from=date(2026, 3, 10),
                date_to=date(2026, 3, 1),
            )
