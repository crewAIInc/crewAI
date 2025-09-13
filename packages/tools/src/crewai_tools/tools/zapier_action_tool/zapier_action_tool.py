import os
import logging
from typing import List, Optional
from crewai.tools import BaseTool
from crewai_tools.adapters.zapier_adapter import ZapierActionsAdapter

logger = logging.getLogger(__name__)


def ZapierActionTools(
    zapier_api_key: Optional[str] = None, action_list: Optional[List[str]] = None
) -> List[BaseTool]:
    """Factory function that returns Zapier action tools.

    Args:
        zapier_api_key: The API key for Zapier.
        action_list: Optional list of specific tool names to include.

    Returns:
        A list of Zapier action tools.
    """
    if zapier_api_key is None:
        zapier_api_key = os.getenv("ZAPIER_API_KEY")
        if zapier_api_key is None:
            logger.error("ZAPIER_API_KEY is not set")
            raise ValueError("ZAPIER_API_KEY is not set")
    adapter = ZapierActionsAdapter(zapier_api_key)
    all_tools = adapter.tools()

    if action_list is None:
        return all_tools

    return [tool for tool in all_tools if tool.name in action_list]
