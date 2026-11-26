"""JevDecisionTool and its optional SDK. These run whether or not it is installed."""

from importlib.metadata import requires
import json
import subprocess
import sys

from crewai.utilities.pydantic_schema_utils import (
    sanitize_tool_params_for_openai_strict,
)
import crewai_tools
from crewai_tools import JevDecisionTool
from crewai_tools.generate_tool_specs import ToolSpecExtractor
import pytest


def test_typesafe_sdk_is_an_optional_extra() -> None:
    sdk = [r for r in requires("crewai-tools") or [] if r.startswith("typesafe-sdk")]

    assert sdk
    assert all("extra == 'typesafe-sdk'" in r.replace('"', "'") for r in sdk)


def test_missing_sdk_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)

    with pytest.raises(ImportError, match=r"crewai-tools\[typesafe-sdk\]"):
        JevDecisionTool()


@pytest.mark.timeout(180)  # a fresh interpreter importing every tool is slow under load
def test_crewai_tools_imports_without_the_sdk() -> None:
    code = (
        "import sys; sys.modules['typesafe_sdk'] = sys.modules['httpx2'] = None; "
        "import crewai_tools; "
        "crewai_tools.JevDecisionTool; crewai_tools.SerperDevTool"
    )

    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr


def test_tool_is_exported_and_described_for_tool_specs() -> None:
    assert crewai_tools.tools.JevDecisionTool is JevDecisionTool
    assert "JevDecisionTool" in crewai_tools.__all__
    assert "JevDecisionTool" in crewai_tools.tools.__all__

    extractor = ToolSpecExtractor()
    extractor.extract_tool_info(JevDecisionTool)
    (spec,) = extractor.tools_spec

    assert spec["package_dependencies"] == ["typesafe-sdk"]
    (env_var,) = spec["env_vars"]
    assert (env_var["name"], env_var["required"]) == ("TYPESAFE_API_KEY", True)
    assert spec["run_params_schema"]["required"] == ["state", "question", "choices"]


def test_function_calling_schema_has_no_untyped_array_items() -> None:
    # Strict function calling rejects an array whose items have no type.
    schema = JevDecisionTool.model_fields["args_schema"].default.model_json_schema()

    assert '"items": {}' not in json.dumps(
        sanitize_tool_params_for_openai_strict(schema)
    )
