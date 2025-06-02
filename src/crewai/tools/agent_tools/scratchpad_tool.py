"""Tool for accessing data stored in the agent's scratchpad during reasoning."""

from typing import Any, Dict, Optional, Type, Union, Callable
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class ScratchpadToolSchema(BaseModel):
    """Input schema for ScratchpadTool."""
    key: str = Field(
        ...,
        description=(
            "The key name to retrieve data from the scratchpad. "
            "Must be one of the available keys shown in the tool description. "
            "Example: if 'email_data' is listed as available, use {\"key\": \"email_data\"}"
        )
    )


class ScratchpadTool(BaseTool):
    """Tool that allows agents to access data stored in their scratchpad during task execution.

    This tool's description is dynamically updated to show all available keys,
    making it easy for agents to know what data they can retrieve.
    """

    name: str = "Access Scratchpad Memory"
    description: str = "Access data stored in your scratchpad memory during task execution."
    args_schema: Type[BaseModel] = ScratchpadToolSchema
    scratchpad_data: Dict[str, Any] = Field(default_factory=dict)

    # Allow repeated usage of this tool - scratchpad access should not be limited
    cache_function: Callable = lambda _args, _result: False  # Don't cache scratchpad access
    allow_repeated_usage: bool = True  # Allow accessing the same key multiple times

    def __init__(self, scratchpad_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the scratchpad tool with optional initial data.

        Args:
            scratchpad_data: Initial scratchpad data (usually from agent state)
        """
        super().__init__(**kwargs)
        if scratchpad_data:
            self.scratchpad_data = scratchpad_data
        self._update_description()

    def _run(
        self,
        key: str,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any], Any]:
        """Retrieve data from the scratchpad using the specified key.

        Args:
            key: The key to look up in the scratchpad

        Returns:
            The value associated with the key, or an error message if not found
        """
        print(f"[DEBUG] ScratchpadTool._run called with key: '{key}'")
        print(f"[DEBUG] Current scratchpad keys: {list(self.scratchpad_data.keys())}")
        print(f"[DEBUG] Scratchpad data size: {len(self.scratchpad_data)}")

        if not self.scratchpad_data:
            return (
                "❌ SCRATCHPAD IS EMPTY\n\n"
                "The scratchpad does not contain any data yet.\n"
                "Data will be automatically stored here as you use other tools.\n"
                "Try executing other tools first to gather information.\n\n"
                "💡 TIP: Tools like search, read, or fetch operations will automatically store their results in the scratchpad."
            )

        if key not in self.scratchpad_data:
            available_keys = list(self.scratchpad_data.keys())
            keys_formatted = "\n".join(f"  - '{k}'" for k in available_keys)

            # Create more helpful examples based on actual keys
            example_key = available_keys[0] if available_keys else 'example_key'

            # Check if the user tried a similar key (case-insensitive or partial match)
            similar_keys = [k for k in available_keys if key.lower() in k.lower() or k.lower() in key.lower()]
            similarity_hint = ""
            if similar_keys:
                similarity_hint = f"\n\n🔍 Did you mean one of these?\n" + "\n".join(f"  - '{k}'" for k in similar_keys)

            return (
                f"❌ KEY NOT FOUND: '{key}'\n"
                f"{'='*50}\n\n"
                f"The key '{key}' does not exist in the scratchpad.\n\n"
                f"📦 AVAILABLE KEYS IN SCRATCHPAD:\n{keys_formatted}\n"
                f"{similarity_hint}\n\n"
                f"✅ CORRECT USAGE EXAMPLE:\n"
                f"Action: Access Scratchpad Memory\n"
                f"Action Input: {{\"key\": \"{example_key}\"}}\n\n"
                f"⚠️ IMPORTANT:\n"
                f"- Keys are case-sensitive and must match EXACTLY\n"
                f"- Use the exact key name from the list above\n"
                f"- Do NOT modify or guess key names\n\n"
                f"{'='*50}"
            )

        value = self.scratchpad_data[key]

        # Format the output nicely based on the type
        if isinstance(value, dict):
            import json
            formatted_output = f"✅ Successfully retrieved data for key '{key}':\n\n"
            formatted_output += json.dumps(value, indent=2)
            return formatted_output
        elif isinstance(value, list):
            import json
            formatted_output = f"✅ Successfully retrieved data for key '{key}':\n\n"
            formatted_output += json.dumps(value, indent=2)
            return formatted_output
        else:
            return f"✅ Successfully retrieved data for key '{key}':\n\n{str(value)}"

    def update_scratchpad(self, new_data: Dict[str, Any]) -> None:
        """Update the scratchpad data and refresh the tool description.

        Args:
            new_data: The new complete scratchpad data
        """
        self.scratchpad_data = new_data
        self._update_description()

    def _update_description(self) -> None:
        """Update the tool description to include all available keys."""
        base_description = (
            "Access data stored in your scratchpad memory during task execution.\n\n"
            "HOW TO USE THIS TOOL:\n"
            "Provide a JSON object with a 'key' field containing the exact name of the data you want to retrieve.\n"
            "Example: {\"key\": \"email_data\"}"
        )

        if not self.scratchpad_data:
            self.description = (
                f"{base_description}\n\n"
                "📝 STATUS: Scratchpad is currently empty.\n"
                "Data will be automatically stored here as you use other tools."
            )
            return

        # Build a description of available keys with a preview of their contents
        key_descriptions = []
        example_key = None

        for key, value in self.scratchpad_data.items():
            if not example_key:
                example_key = key

            # Create a brief description of what's stored
            if isinstance(value, dict):
                preview = f"dict with {len(value)} items"
                if 'data' in value and isinstance(value['data'], list):
                    preview = f"list of {len(value['data'])} items"
            elif isinstance(value, list):
                preview = f"list of {len(value)} items"
            elif isinstance(value, str):
                preview = f"string ({len(value)} chars)"
            else:
                preview = type(value).__name__

            key_descriptions.append(f"  📌 '{key}': {preview}")

        available_keys_text = "\n".join(key_descriptions)

        self.description = (
            f"{base_description}\n\n"
            f"📦 AVAILABLE DATA IN SCRATCHPAD:\n{available_keys_text}\n\n"
            f"💡 EXAMPLE USAGE:\n"
            f"To retrieve the '{example_key}' data, use:\n"
            f"Action Input: {{\"key\": \"{example_key}\"}}"
        )