import re
import os
import typing as t
from typing import Literal
import logging
import json
from crewai.tools import BaseTool
from crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder import CrewaiPlatformToolBuilder
from crewai_tools.adapters.tool_collection import ToolCollection

logger = logging.getLogger(__name__)



def CrewaiPlatformTools(
    apps: list[str],
) -> ToolCollection[BaseTool]:
    """Factory function that returns crewai platform tools.
    Args:
        apps: List of platform apps to get tools that are available on the platform.

    Returns:
        A list of BaseTool instances for platform actions
    """

    builder = CrewaiPlatformToolBuilder(apps=apps)

    return builder.tools()
