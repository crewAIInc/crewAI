from crewai.tools import BaseTool

from crewai_tools.adapters.tool_collection import ToolCollection
from crewai_tools.tools.crewai_platform_tools._client import (
    _PlatformToolSelector,
    _PlatformToolsClient,
)
from crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool import (
    CrewAIPlatformActionTool,
)


def CrewaiPlatformTools(  # noqa: N802
    apps: list[str],
) -> ToolCollection[BaseTool]:
    """Factory function that returns crewai platform tools.

    Args:
        apps: List of platform apps to get tools that are available on the platform.

    Returns:
        A list of BaseTool instances for platform actions
    """
    client = _PlatformToolsClient()
    selectors = [_PlatformToolSelector.from_string(app) for app in apps]

    tool_infos = client.get_tools(selectors)

    return ToolCollection(
        [
            CrewAIPlatformActionTool(tool_info=tool_info, client=client)
            for tool_info in tool_infos
        ]
    )
