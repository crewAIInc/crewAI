import logging

from crewai.tools import BaseTool

from crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool import (
    CrewAIPlatformActionTool,
)
from crewai_tools.tools.crewai_platform_tools.integrations_client import (
    ApplicationSelector,
    client_for_selector,
)


logger = logging.getLogger(__name__)


def CrewaiPlatformTools(  # noqa: N802
    apps: list[str],
) -> list[BaseTool]:
    """Factory function that returns crewai platform tools.

    Args:
        apps: List of platform apps to get tools that are available on the platform.

    Returns:
        A list of BaseTool instances for platform actions
    """
    selectors = [ApplicationSelector.from_string(app) for app in apps]
    tools: list[BaseTool] = []

    try:
        for selector in selectors:
            client = client_for_selector(selector)
            tools.extend(
                CrewAIPlatformActionTool(tool_info, client=client)
                for tool_info in client.get_actions([selector])
            )
    except ValueError:
        raise
    except Exception as error:
        logger.error(f"Failed to fetch platform tools for apps {apps}: {error}")
        return []

    return tools
