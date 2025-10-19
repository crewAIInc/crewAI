"""Tools handler for managing tool execution and caching."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from crewai.tools.cache_tools.cache_tools import CacheTools


if TYPE_CHECKING:
    from crewai.agents.cache.cache_handler import CacheHandler
    from crewai.tools.tool_calling import InstructorToolCalling, ToolCalling


class ToolsHandler:
    """Callback handler for tool usage.

    Attributes:
        last_used_tool: The most recently used tool calling instance.
        cache: Optional cache handler for storing tool outputs.
    """

    def __init__(self, cache: CacheHandler | None = None) -> None:
        """Initialize the callback handler.

        Args:
            cache: Optional cache handler for storing tool outputs.
        """
        self.cache: CacheHandler | None = cache
        self.last_used_tool: ToolCalling | InstructorToolCalling | None = None

    def on_tool_use(
        self,
        calling: ToolCalling | InstructorToolCalling,
        output: str,
        should_cache: bool = True,
    ) -> None:
        """Run when tool ends running.

        Args:
            calling: The tool calling instance.
            output: The output from the tool execution.
            should_cache: Whether to cache the tool output.
        """
        self.last_used_tool = calling
        if self.cache and should_cache and calling.tool_name != CacheTools().name:
            # Convert arguments to string for cache
            input_str = ""
            if calling.arguments:
                if isinstance(calling.arguments, dict):
                    input_str = json.dumps(calling.arguments)
                else:
                    input_str = str(calling.arguments)

            self.cache.add(
                tool=calling.tool_name,
                input=input_str,
                output=output,
            )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for BaseClient Protocol.

        This allows the Protocol to be used in Pydantic models without
        requiring arbitrary_types_allowed=True.
        """
        return core_schema.any_schema()
