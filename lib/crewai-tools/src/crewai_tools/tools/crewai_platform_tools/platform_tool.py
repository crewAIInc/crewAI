from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from crewai.utilities.string_utils import sanitize_tool_name


@dataclass(frozen=True)
class PlatformTool:
    """A platform tool before or after remote resolution."""

    application: str
    tool: str | None = None
    connection_id: UUID | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None

    @classmethod
    def from_selector(cls, value: str) -> PlatformTool:
        """Parse an ``application[/tool][@connection_uuid]`` selector."""
        if not value:
            raise ValueError(f"Invalid application selector {value!r}: cannot be empty")
        if "@" in value and "/" in value and value.index("@") < value.index("/"):
            raise ValueError(
                f"Invalid application selector {value!r}: "
                "connection ID must be the last segment"
            )

        application_and_tool, connection_separator, connection_id = value.partition("@")
        application, tool_separator, tool = application_and_tool.partition("/")

        if not application:
            raise ValueError(
                f"Invalid application selector {value!r}: application cannot be empty"
            )
        if tool_separator and not tool:
            raise ValueError(
                f"Invalid application selector {value!r}: action cannot be empty"
            )
        if connection_separator and not connection_id:
            raise ValueError(
                f"Invalid application selector {value!r}: connection ID cannot be empty"
            )

        parsed_connection_id = None
        if connection_id:
            try:
                parsed_connection_id = UUID(connection_id)
            except ValueError as error:
                raise ValueError(
                    f"Invalid application selector {value!r}: "
                    "connection ID must be a valid UUID"
                ) from error

        return cls(
            application=application,
            tool=tool if tool_separator else None,
            connection_id=parsed_connection_id,
        )

    @property
    def python_identifier(self) -> str:
        """Return the tool's Python identifier."""
        if self.tool is None:
            raise ValueError("Platform tool must include a tool to compute its name")
        if self.connection_id is not None:
            return sanitize_tool_name(f"{self.application}__{self.tool}")
        return sanitize_tool_name(self.tool)
