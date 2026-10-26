import json
from typing import Any

from crewai.tools import (
    BaseTool,
    ToolExecutionFailedError,
    ToolFailure,
    ToolFailureRecord,
)

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
    """
    selectors = [ApplicationSelector.from_string(app) for app in apps]
    tools: list[BaseTool] = []

    for selector in selectors:
        client = client_for_selector(selector)
        tools.extend(
            CrewAIPlatformActionTool(tool_info, client=client)
            for tool_info in client.get_actions([selector])
        )

    return tools


def run_platform_action(selector: str, /, **arguments: Any) -> dict[str, Any]:
    """Run one integration app action and return its output, raising on failure.

    Unlike calling a platform tool's ``run()``, which returns a ``ToolFailure``
    for the agent to read, this raises, so code can check an integration
    without building an agent or crew. Requires the
    ``CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN`` environment variable.

    Args:
        selector: One action, as ``<app>/<action>[@<connection>]``.
        **arguments: The action's arguments.

    Returns:
        The action's output.

    Raises:
        ValueError: If the token is missing, the selector is malformed or does
            not resolve to exactly one action, or the arguments do not match
            the action's schema.
        requests.HTTPError: If looking up the action fails, for example because
            the token is rejected.
        ToolExecutionFailedError: If the action reports a failure. The
            exception's ``record.failure`` carries the ``code`` and ``retryable``.
        TypeError: If the action's output is not a JSON object.
    """
    if ApplicationSelector.from_string(selector).action is None:
        raise ValueError(
            f"selector {selector!r} names an app, not one action; "
            "expected '<app>/<action>[@<connection>]'"
        )
    tools = CrewaiPlatformTools(apps=[selector])
    if len(tools) != 1:
        raise ValueError(
            f"selector {selector!r} resolved to {len(tools)} actions; the action "
            "does not exist or the app is not connected"
        )
    tool = tools[0]
    result = tool.run(**arguments)
    if isinstance(result, ToolFailure):
        raise ToolExecutionFailedError(
            ToolFailureRecord(tool_name=tool.name, failure=result, tool_args=arguments)
        )
    output = json.loads(result)
    if not isinstance(output, dict):
        raise TypeError(
            f"action {selector!r} returned {type(output).__name__}, not a JSON object"
        )
    return output
