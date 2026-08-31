from crewai.tools import BaseTool

from crewai_tools.adapters.tool_collection import ToolCollection
from crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool import (
    ClipperClient,
    CrewAIPlatformActionTool,
    LegacyIntegrationsClient,
)
from crewai_tools.tools.crewai_platform_tools.platform_tool import PlatformTool


def CrewaiPlatformTools(  # noqa: N802
    apps: list[str],
) -> ToolCollection[BaseTool]:
    """Factory function that returns crewai platform tools.

    Args:
        apps: Platform application selectors.

    Returns:
        A list of platform tools.
    """
    tools: list[BaseTool] = []
    for app in apps:
        selector = PlatformTool.from_selector(app)
        client = (
            ClipperClient()
            if selector.connection_id is not None
            else LegacyIntegrationsClient()
        )
        tools.extend(
            CrewAIPlatformActionTool(platform_tool=tool, client=client)
            for tool in client.list_tools(selector)
        )

    return ToolCollection(tools)
