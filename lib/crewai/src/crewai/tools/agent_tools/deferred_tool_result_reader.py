"""Bounded reader for temporary native-tool result artifacts."""

from __future__ import annotations

from itertools import islice
from os import path

from pydantic import BaseModel, Field, PrivateAttr

from crewai.tools.base_tool import BaseTool


class DeferredToolResultReaderSchema(BaseModel):
    """Input accepted by the deferred tool-result reader."""

    file_path: str = Field(
        ..., description="The temporary artifact path supplied in the tool result."
    )
    start_line: int | None = Field(
        default=1, description="Line number to start reading from (1-indexed)."
    )
    line_count: int | None = Field(default=None, description="Number of lines to read.")


class DeferredToolResultReader(BaseTool):
    """Read framework-managed deferred tool results through exact aliases only."""

    name: str = "Read a deferred tool result"
    description: str = (
        "Read a bounded section of a temporary tool-result artifact. "
        "Only artifact paths supplied by a previous tool result are available."
    )
    args_schema: type[BaseModel] = DeferredToolResultReaderSchema

    _allowed_file_paths: dict[str, str] = PrivateAttr(default_factory=dict)

    def allow_file(self, file_path: str, alias: str) -> None:
        """Allow an exact temporary artifact path for this execution."""
        if not alias:
            raise ValueError("Allowed file aliases must not be empty.")
        self._allowed_file_paths[alias] = path.realpath(file_path)

    def disallow_file(self, alias: str) -> None:
        """Revoke an artifact path. Missing aliases are ignored."""
        self._allowed_file_paths.pop(alias, None)

    def _run(
        self,
        file_path: str,
        start_line: int | None = 1,
        line_count: int | None = None,
    ) -> str:
        """Read a complete artifact or a bounded line range."""
        resolved_path = self._allowed_file_paths.get(file_path)
        if resolved_path is None:
            return "Error: This temporary tool-result artifact is not available."

        start_index = max((start_line or 1) - 1, 0)
        try:
            with open(resolved_path, encoding="utf-8") as artifact_file:
                if start_index == 0 and line_count is None:
                    return artifact_file.read()

                stop_index = (
                    None if line_count is None else start_index + max(line_count, 0)
                )
                lines = list(islice(artifact_file, start_index, stop_index))
        except FileNotFoundError:
            return "Error: Temporary tool-result artifact is no longer available."

        if not lines and start_index > 0:
            return f"Error: Start line {start_line} exceeds the number of lines in the artifact."
        return "".join(lines)
