import subprocess
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.plasmate_website_tool.plasmate_website_tool import (
    FixedPlasmateWebsiteToolSchema,
    PlasmateWebsiteTool,
    PlasmateWebsiteToolSchema,
)


def _completed_process(stdout: str = "# Page\n\nContent.", returncode: int = 0) -> MagicMock:
    m = MagicMock()
    m.stdout = stdout
    m.returncode = returncode
    m.stderr = ""
    return m


@pytest.fixture
def tool():
    with patch("shutil.which", return_value="/usr/local/bin/plasmate"):
        return PlasmateWebsiteTool()


@pytest.fixture
def fixed_tool():
    with patch("shutil.which", return_value="/usr/local/bin/plasmate"):
        return PlasmateWebsiteTool(website_url="https://example.com")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestPlasmateWebsiteToolInit:
    def test_defaults(self, tool):
        assert tool.output_format == "markdown"
        assert tool.timeout == 30
        assert tool.selector is None
        assert tool.extra_headers == {}
        assert tool.website_url is None
        assert tool.args_schema is PlasmateWebsiteToolSchema

    def test_fixed_url_switches_schema(self, fixed_tool):
        assert fixed_tool.website_url == "https://example.com"
        assert fixed_tool.args_schema is FixedPlasmateWebsiteToolSchema

    def test_fixed_url_updates_description(self, fixed_tool):
        assert "example.com" in fixed_tool.description

    def test_custom_format(self):
        with patch("shutil.which", return_value="/usr/local/bin/plasmate"):
            t = PlasmateWebsiteTool(output_format="text")
        assert t.output_format == "text"

    def test_all_valid_formats(self):
        for fmt in ("markdown", "text", "som", "links"):
            with patch("shutil.which", return_value="/usr/local/bin/plasmate"):
                t = PlasmateWebsiteTool(output_format=fmt)
            assert t.output_format == fmt

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="output_format must be one of"):
            PlasmateWebsiteTool(output_format="html")

    def test_package_dependencies(self, tool):
        assert "plasmate" in tool.package_dependencies


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


class TestBuildCmd:
    def test_basic_cmd(self, tool):
        with patch("shutil.which", return_value="/usr/local/bin/plasmate"):
            cmd = tool._build_cmd("https://example.com")
        assert cmd[0] == "/usr/local/bin/plasmate"
        assert "fetch" in cmd
        assert "https://example.com" in cmd
        assert "--format" in cmd
        assert "markdown" in cmd
        assert "--timeout" in cmd
        assert "30000" in cmd  # converted to ms

    def test_selector_added(self):
        with patch("shutil.which", return_value="/usr/local/bin/plasmate"):
            t = PlasmateWebsiteTool(selector="main")
            cmd = t._build_cmd("https://example.com")
        assert "--selector" in cmd
        assert cmd[cmd.index("--selector") + 1] == "main"

    def test_extra_headers_added(self):
        with patch("shutil.which", return_value="/usr/local/bin/plasmate"):
            t = PlasmateWebsiteTool(extra_headers={"X-Custom": "val"})
            cmd = t._build_cmd("https://example.com")
        assert "--header" in cmd
        assert "X-Custom: val" in cmd[cmd.index("--header") + 1]

    def test_timeout_in_ms(self):
        with patch("shutil.which", return_value="/usr/local/bin/plasmate"):
            t = PlasmateWebsiteTool(timeout=60)
            cmd = t._build_cmd("https://example.com")
        assert "60000" in cmd

    def test_missing_binary_raises_import_error(self):
        with patch("shutil.which", return_value=None):
            t = PlasmateWebsiteTool()
        with pytest.raises(ImportError, match="plasmate is required"):
            t._build_cmd("https://example.com")


# ---------------------------------------------------------------------------
# _run
# ---------------------------------------------------------------------------


class TestRun:
    def test_successful_fetch(self, tool):
        with patch("shutil.which", return_value="/usr/local/bin/plasmate"), \
             patch("subprocess.run", return_value=_completed_process("# Heading\n\nBody")):
            result = tool._run(website_url="https://example.com")
        assert "# Heading" in result
        assert "Body" in result

    def test_uses_fixed_url_when_no_kwarg(self, fixed_tool):
        with patch("shutil.which", return_value="/usr/local/bin/plasmate"), \
             patch("subprocess.run", return_value=_completed_process("fixed content")) as mock_run:
            result = fixed_tool._run()
        cmd = mock_run.call_args[0][0]
        assert "https://example.com" in cmd
        assert "fixed content" in result

    def test_no_url_returns_error(self, tool):
        result = tool._run()
        assert "Error" in result
        assert "no URL provided" in result

    def test_nonzero_returncode_returns_error(self, tool):
        with patch("shutil.which", return_value="/usr/local/bin/plasmate"), \
             patch("subprocess.run", return_value=_completed_process("", returncode=1)):
            result = tool._run(website_url="https://example.com")
        assert "Error" in result
        assert "exited 1" in result

    def test_timeout_returns_error(self, tool):
        with patch("shutil.which", return_value="/usr/local/bin/plasmate"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="plasmate", timeout=30)):
            result = tool._run(website_url="https://example.com")
        assert "timed out" in result

    def test_missing_binary_returns_error(self, tool):
        with patch("shutil.which", return_value=None), \
             patch("subprocess.run", side_effect=FileNotFoundError()):
            result = tool._run(website_url="https://example.com")
        assert "not found" in result.lower() or "plasmate" in result.lower()

    def test_empty_content_returns_warning(self, tool):
        with patch("shutil.which", return_value="/usr/local/bin/plasmate"), \
             patch("subprocess.run", return_value=_completed_process("")):
            result = tool._run(website_url="https://example.com")
        assert "Warning" in result or "empty" in result.lower()

    def test_text_format(self):
        with patch("shutil.which", return_value="/usr/local/bin/plasmate"):
            t = PlasmateWebsiteTool(output_format="text")
        with patch("shutil.which", return_value="/usr/local/bin/plasmate"), \
             patch("subprocess.run", return_value=_completed_process("plain text")) as mock_run:
            result = t._run(website_url="https://example.com")
        cmd = mock_run.call_args[0][0]
        assert "text" in cmd
        assert "plain text" in result

    def test_som_format(self):
        som = '{"role":"document","children":[{"role":"heading","name":"Title"}]}'
        with patch("shutil.which", return_value="/usr/local/bin/plasmate"):
            t = PlasmateWebsiteTool(output_format="som")
        with patch("shutil.which", return_value="/usr/local/bin/plasmate"), \
             patch("subprocess.run", return_value=_completed_process(som)):
            result = t._run(website_url="https://example.com")
        assert "heading" in result


# ---------------------------------------------------------------------------
# Integration: exported from package
# ---------------------------------------------------------------------------


class TestPackageExport:
    def test_importable_from_crewai_tools(self):
        from crewai_tools import PlasmateWebsiteTool as Exported
        assert Exported is PlasmateWebsiteTool

    def test_tool_has_name(self, tool):
        assert "Plasmate" in tool.name

    def test_tool_has_description(self, tool):
        assert len(tool.description) > 20
