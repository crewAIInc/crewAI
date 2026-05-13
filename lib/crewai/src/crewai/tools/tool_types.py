from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crewai_files import FileInput


@dataclass
class MultimodalToolResult:
    """Multimodal result returned by a tool.

    Allows tool authors to return both a text observation and file attachments
    (images, audio, video, PDFs) in a single value. The files are injected into
    the LLM message via the existing ``files`` pipeline so every vision-capable
    provider receives them in the correct format.

    Example::

        from crewai.tools import BaseTool, MultimodalToolResult
        from crewai_files import ImageFile

        class ScreenshotTool(BaseTool):
            name = "take_screenshot"
            description = "Capture a screenshot of a URL"

            def _run(self, url: str) -> MultimodalToolResult:
                data = capture(url)
                return MultimodalToolResult(
                    text="Screenshot captured",
                    files={"screenshot": ImageFile(data)},
                )
    """

    text: str
    files: dict[str, FileInput] = field(default_factory=dict)
    result_as_answer: bool = False


@dataclass
class ToolResult:
    """Internal result of tool execution propagated through the pipeline."""

    result: str
    result_as_answer: bool = False
    files: dict[str, FileInput] = field(default_factory=dict)
