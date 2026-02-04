import logging

from crewai.tools import BaseTool

from crewai_tools.adapters.tool_collection import ToolCollection
from crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder import (
    CrewaiPlatformToolBuilder,
)
from crewai_tools.tools.crewai_platform_tools.file_hook import (
    register_file_processing_hook,
)


logger = logging.getLogger(__name__)


def CrewaiPlatformTools(  # noqa: N802
    apps: list[str],
) -> ToolCollection[BaseTool]:
    """Factory function that returns crewai platform tools.

    Args:
        apps: List of platform apps to get tools that are available on the platform.

    Returns:
        A list of BaseTool instances for platform actions
    """
    register_file_processing_hook()

    builder = CrewaiPlatformToolBuilder(apps=apps)

    return builder.tools()  # type: ignore
