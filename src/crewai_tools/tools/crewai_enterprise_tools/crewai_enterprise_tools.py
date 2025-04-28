"""
Crewai Enterprise Tools
"""

import os
import typing as t
from crewai.tools import BaseTool
from crewai_tools.adapters.enterprise_adapter import EnterpriseActionKitToolAdapter


def CrewaiEnterpriseTools(
    enterprise_token: t.Optional[str] = None,
    actions_list: t.Optional[t.List[str]] = None
) -> t.List[BaseTool]:
    """Factory function that returns crewai enterprise tools.

    Args:
        enterprise_token: The token for accessing enterprise actions.
                         If not provided, will try to use CREWAI_ENTEPRISE_TOOLS_TOKEN env var.
        actions_list: Optional list of specific tool names to include.
                   If provided, only tools with these names will be returned.

    Returns:
        A list of BaseTool instances for enterprise actions
    """
    if enterprise_token is None:
        enterprise_token = os.environ.get("CREWAI_ENTEPRISE_TOOLS_TOKEN")
        if enterprise_token is None:
            raise ValueError(
                "No enterprise token provided. Please provide a token or set the CREWAI_ENTEPRISE_TOOLS_TOKEN environment variable."
            )

    adapter = EnterpriseActionKitToolAdapter(enterprise_token)
    all_tools = adapter.tools()

    if actions_list is None:
        return all_tools

    # Filter tools based on the provided list
    return [tool for tool in all_tools if tool.name in actions_list]
