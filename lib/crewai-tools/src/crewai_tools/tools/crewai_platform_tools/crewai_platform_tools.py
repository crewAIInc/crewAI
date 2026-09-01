import logging

from crewai.tools import BaseTool

from crewai_tools.adapters.tool_collection import ToolCollection
from crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder import (
    CrewaiPlatformToolBuilder,
)
from crewai_tools.tools.crewai_platform_tools.integrations_client import (
    IntegrationsClient,
)


logger = logging.getLogger(__name__)


def CrewaiPlatformTools(  # noqa: N802
    apps: list[str],
    integrations_client: IntegrationsClient | None = None,
) -> ToolCollection[BaseTool]:
    """Factory function that returns crewai platform tools.

    Args:
        apps: List of platform apps to get tools that are available on the platform.
        integrations_client: Client used to get and execute actions.

    Returns:
        A list of BaseTool instances for platform actions
    """
    builder = CrewaiPlatformToolBuilder(
        apps=apps, integrations_client=integrations_client
    )

    return builder.tools()  # type: ignore
