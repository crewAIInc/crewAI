from crewai.tools import BaseTool
from crewai_core.platform_apps import PlatformAppsUnavailableError

from crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool import (
    CrewAIPlatformActionTool,
)
from crewai_tools.tools.crewai_platform_tools.integrations_client import (
    ApplicationSelector,
    client_for_selector,
)


def CrewaiPlatformTools(  # noqa: N802
    apps: list[str],
) -> list[BaseTool]:
    """Factory function that returns crewai platform tools.

    Args:
        apps: List of platform apps to get tools that are available on the platform.

    Returns:
        A list of BaseTool instances for platform actions

    Raises:
        PlatformAppsUnavailableError: CrewAI AMP could not resolve one or more
            of the apps; it names every one.
    """
    selectors = [ApplicationSelector.from_string(app) for app in apps]
    tools: list[BaseTool] = []
    missing_apps: list[str] = []

    for selector in selectors:
        client = client_for_selector(selector)
        try:
            tool_infos = client.get_actions([selector])
        except PlatformAppsUnavailableError as error:
            missing_apps.extend(error.missing_apps)
            continue
        tools.extend(
            CrewAIPlatformActionTool(tool_info, client=client)
            for tool_info in tool_infos
        )

    if missing_apps:
        raise PlatformAppsUnavailableError(list(dict.fromkeys(missing_apps)))
    return tools
