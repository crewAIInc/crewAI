"""Tests for MultimodalToolResult and the multimodal pipeline in tool_usage."""

from unittest.mock import MagicMock

import pytest
from crewai_files import ImageFile
from crewai_files.core.types import BaseFile

from crewai.tools.tool_types import MultimodalToolResult, ToolResult
from crewai.tools.tool_usage import ToolUsage

# Minimal 1×1 white PNG bytes used across tests.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x11\x00\x01\xd9\x7f\xd3\r\x00\x00\x00\x00IEND\xaeB`\x82"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_tool_usage() -> ToolUsage:
    return ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )


# ---------------------------------------------------------------------------
# MultimodalToolResult dataclass
# ---------------------------------------------------------------------------


class TestMultimodalToolResult:
    def test_text_only_defaults(self):
        r = MultimodalToolResult(text="hello")
        assert r.text == "hello"
        assert r.files == {}
        assert r.result_as_answer is False

    def test_files_stored(self):
        img = ImageFile(source=_PNG_BYTES)
        r = MultimodalToolResult(text="captured", files={"img": img})
        assert r.files["img"] is img

    def test_result_as_answer_flag(self):
        r = MultimodalToolResult(text="final", result_as_answer=True)
        assert r.result_as_answer is True


# ---------------------------------------------------------------------------
# ToolResult.files field
# ---------------------------------------------------------------------------


class TestToolResultFiles:
    def test_default_files_empty(self):
        r = ToolResult(result="ok")
        assert r.files == {}

    def test_files_stored(self):
        img = ImageFile(source=_PNG_BYTES)
        r = ToolResult(result="ok", files={"img": img})
        assert r.files["img"] is img


# ---------------------------------------------------------------------------
# ToolUsage._extract_multimodal_files
# ---------------------------------------------------------------------------


class TestExtractMultimodalFiles:
    def test_plain_string_returns_none(self):
        tu = _make_tool_usage()
        assert tu._extract_multimodal_files("plain text") is None
        assert tu._result_files == {}

    def test_plain_int_returns_none(self):
        tu = _make_tool_usage()
        assert tu._extract_multimodal_files(42) is None

    def test_multimodal_tool_result_extracts_text(self):
        tu = _make_tool_usage()
        img = ImageFile(source=_PNG_BYTES)
        mtr = MultimodalToolResult(text="captured", files={"screenshot": img})

        result = tu._extract_multimodal_files(mtr)

        assert result == "captured"
        assert tu._result_files == {"screenshot": img}

    def test_multimodal_tool_result_empty_files(self):
        tu = _make_tool_usage()
        mtr = MultimodalToolResult(text="text only")

        result = tu._extract_multimodal_files(mtr)

        assert result == "text only"
        assert tu._result_files == {}

    def test_base_file_bytes_source(self):
        tu = _make_tool_usage()
        img = ImageFile(source=_PNG_BYTES)

        result = tu._extract_multimodal_files(img)

        # bytes source has no filename → key falls back to "file"
        assert result is not None
        assert "ImageFile" in result
        assert "file" in tu._result_files
        assert tu._result_files["file"] is img

    def test_base_file_named_source(self):
        import tempfile, pathlib

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_PNG_BYTES)
            tmp = pathlib.Path(f.name)

        try:
            tu = _make_tool_usage()
            img = ImageFile(source=tmp)
            result = tu._extract_multimodal_files(img)

            stem = tmp.stem
            assert result == f"[ImageFile: {tmp.name}]"
            assert stem in tu._result_files
            assert tu._result_files[stem] is img
        finally:
            tmp.unlink(missing_ok=True)

    def test_result_files_reset_between_calls(self):
        tu = _make_tool_usage()
        img = ImageFile(source=_PNG_BYTES)
        mtr = MultimodalToolResult(text="first", files={"img": img})
        tu._extract_multimodal_files(mtr)
        assert tu._result_files != {}

        # plain call should NOT reset _result_files (caller is _format_result)
        tu._extract_multimodal_files("plain")
        # returns None, _result_files unchanged
        assert tu._result_files != {}


# ---------------------------------------------------------------------------
# ToolUsage._format_result integration
# ---------------------------------------------------------------------------


class TestFormatResult:
    def test_plain_string_passthrough(self):
        tu = _make_tool_usage()
        tu.task.used_tools = 0
        assert tu._format_result("hello world") == "hello world"
        assert tu._result_files == {}

    def test_multimodal_result_returns_text(self):
        tu = _make_tool_usage()
        tu.task.used_tools = 0
        img = ImageFile(source=_PNG_BYTES)
        mtr = MultimodalToolResult(text="image ready", files={"img": img})

        text = tu._format_result(mtr)

        assert text == "image ready"
        assert tu._result_files == {"img": img}

    def test_integer_converted_to_str(self):
        tu = _make_tool_usage()
        tu.task.used_tools = 0
        assert tu._format_result(42) == "42"
        assert tu._result_files == {}

    def test_result_files_cleared_on_plain_result(self):
        """A plain _format_result call must clear files set by a prior multimodal call."""
        tu = _make_tool_usage()
        tu.task.used_tools = 0
        img = ImageFile(source=_PNG_BYTES)
        mtr = MultimodalToolResult(text="with file", files={"img": img})
        tu._format_result(mtr)
        assert tu._result_files != {}

        tu.task.used_tools = 1
        tu._format_result("plain follow-up")
        assert tu._result_files == {}


# ---------------------------------------------------------------------------
# AddImageTool
# ---------------------------------------------------------------------------


class TestAddImageTool:
    def test_run_returns_multimodal_tool_result(self):
        from crewai.tools.agent_tools.add_image_tool import AddImageTool

        tool = AddImageTool()
        result = tool._run(image_url="https://example.com/photo.jpg")

        assert isinstance(result, MultimodalToolResult)
        assert "image" in result.files
        assert isinstance(result.files["image"], BaseFile)

    def test_run_with_action_text(self):
        from crewai.tools.agent_tools.add_image_tool import AddImageTool

        tool = AddImageTool()
        result = tool._run(
            image_url="https://example.com/photo.jpg",
            action="Describe what you see",
        )

        assert isinstance(result, MultimodalToolResult)
        assert result.text == "Describe what you see"

    def test_run_default_action_text_not_empty(self):
        from crewai.tools.agent_tools.add_image_tool import AddImageTool

        tool = AddImageTool()
        result = tool._run(image_url="https://example.com/photo.jpg")

        assert isinstance(result, MultimodalToolResult)
        assert result.text  # non-empty default action
