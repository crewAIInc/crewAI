import logging

from crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool import (
    CrewAIPlatformActionTool,
)
from crewai_tools.tools.crewai_platform_tools.integrations_client import (
    ApplicationSelector,
    LegacyClient,
)


logger = logging.getLogger(__name__)


def CrewaiPlatformTools(  # noqa: N802
    apps: list[str],
) -> list[CrewAIPlatformActionTool]:
    """Factory function that returns crewai platform tools.

    Args:
        apps: List of platform apps to get tools that are available on the platform.

    Returns:
        A list of BaseTool instances for platform actions
    """
    selectors = [ApplicationSelector.from_string(app) for app in apps]
    client = LegacyClient()

    try:
        tool_infos = client.get_actions(selectors)
    except ValueError:
        raise
    except Exception as error:
        logger.error(f"Failed to fetch platform tools for apps {apps}: {error}")
        return []

    return [CrewAIPlatformActionTool(tool_info) for tool_info in tool_infos]
