import logging
import time

from typing import Any, Optional, Union

from ..tools.cache_tools import CacheTools
from ..tools.tool_calling import InstructorToolCalling, ToolCalling
from .cache.cache_handler import CacheHandler

logger = logging.getLogger(__name__)


class ToolsHandler:
    """Callback handler for tool usage."""

    last_used_tool: ToolCalling = {}  # type: ignore # BUG?: Incompatible types in assignment (expression has type "Dict[...]", variable has type "ToolCalling")
    cache: Optional[CacheHandler]

    def __init__(self, cache: Optional[CacheHandler] = None):
        """Initialize the callback handler."""
        self.cache = cache
        self.last_used_tool = {}  # type: ignore # BUG?: same as above
        logger.info(f"ToolsHandler initialized with cache: {cache is not None}")

    def on_tool_use(
        self,
        calling: Union[ToolCalling, InstructorToolCalling],
        output: str,
        should_cache: bool = True,
    ) -> Any:
        """Run when tool ends running."""
        start_time = time.time()

        logger.info(f"Tool used: {calling.tool_name}")
        logger.debug(f"Tool arguments: {calling.arguments}")
        logger.debug(f"Tool output length: {len(output)} characters")

        self.last_used_tool = calling  # type: ignore # BUG?: Incompatible types in assignment (expression has type "Union[ToolCalling, InstructorToolCalling]", variable has type "ToolCalling")
        if self.cache and should_cache and calling.tool_name != CacheTools().name:
            logger.debug(f"Caching result for tool: {calling.tool_name}")
            try:
                self.cache.add(
                    tool=calling.tool_name,
                    input=calling.arguments,
                    output=output,
                )
                logger.info(f"Successfully cached result for tool: {calling.tool_name}")
            except Exception as e:
                logger.error(
                    f"Error caching result for tool {calling.tool_name}: {str(e)}",
                    exc_info=True,
                )
        elif not should_cache:
            logger.info(f"Skipped caching for tool: {calling.tool_name}")
        elif calling.tool_name == CacheTools().name:
            logger.info("Skipped caching for CacheTools")

        duration = time.time() - start_time
        logger.info(f"Tool handling completed in {duration:.4f} seconds")
