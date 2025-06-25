"""
Crewai Enterprise Tools
"""

import os
import typing as t
import logging
import json
from crewai.tools import BaseTool
from crewai_tools.adapters.enterprise_adapter import EnterpriseActionKitToolAdapter
from crewai_tools.adapters.tool_collection import ToolCollection

logger = logging.getLogger(__name__)


def CrewaiEnterpriseTools(
    enterprise_token: t.Optional[str] = None,
    actions_list: t.Optional[t.List[str]] = None,
    enterprise_action_kit_project_id: t.Optional[str] = None,
    enterprise_action_kit_project_url: t.Optional[str] = None,
) -> ToolCollection[BaseTool]:
    """Factory function that returns crewai enterprise tools.

    Args:
        enterprise_token: The token for accessing enterprise actions.
                         If not provided, will try to use CREWAI_ENTERPRISE_TOOLS_TOKEN env var.
        actions_list: Optional list of specific tool names to include.
                   If provided, only tools with these names will be returned.
        enterprise_action_kit_project_id: Optional ID of the Enterprise Action Kit project.
        enterprise_action_kit_project_url: Optional URL of the Enterprise Action Kit project.

    Returns:
        A ToolCollection of BaseTool instances for enterprise actions
    """

    if enterprise_token is None or enterprise_token == "":
        enterprise_token = os.environ.get("CREWAI_ENTERPRISE_TOOLS_TOKEN")
        if not enterprise_token:
            logger.warning("No enterprise token provided")

    adapter_kwargs = {"enterprise_action_token": enterprise_token}

    if enterprise_action_kit_project_id is not None:
        adapter_kwargs["enterprise_action_kit_project_id"] = (
            enterprise_action_kit_project_id
        )
    if enterprise_action_kit_project_url is not None:
        adapter_kwargs["enterprise_action_kit_project_url"] = (
            enterprise_action_kit_project_url
        )

    adapter = EnterpriseActionKitToolAdapter(**adapter_kwargs)
    all_tools = adapter.tools()
    parsed_actions_list = _parse_actions_list(actions_list)

    # Filter tools based on the provided list
    return ToolCollection(all_tools).filter_by_names(parsed_actions_list)


# ENTERPRISE INJECTION ONLY
def _parse_actions_list(actions_list: t.Optional[t.List[str]]) -> t.List[str] | None:
    """Parse a string representation of a list of tool names to a list of tool names.

    Args:
        actions_list: A string representation of a list of tool names.

    Returns:
        A list of tool names.
    """
    if actions_list is not None:
        return actions_list

    actions_list_from_env = os.environ.get("CREWAI_ENTERPRISE_TOOLS_ACTIONS_LIST")
    if actions_list_from_env is None:
        return None

    try:
        return json.loads(actions_list_from_env)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse actions_list as JSON: {actions_list_from_env}")
        return None
