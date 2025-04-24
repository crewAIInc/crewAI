import requests
from pydantic import Field, create_model
from typing import List, Any, Dict, Optional
import json
from crewai.tools import BaseTool


ENTERPRISE_ACTION_KIT_PROJECT_ID = "dd525517-df22-49d2-a69e-6a0eed211166"


class EnterpriseActionTool(BaseTool):
    """A tool that executes a specific enterprise action."""

    enterprise_action_token: str = Field(
        default="", description="The enterprise action token"
    )
    action_name: str = Field(default="", description="The name of the action")
    action_schema: Dict[str, Any] = Field(
        default={}, description="The schema of the action"
    )

    def __init__(
        self,
        name: str,
        description: str,
        enterprise_action_token: str,
        action_name: str,
        action_schema: Dict[str, Any],
    ):
        schema_props = (
            action_schema.get("function", {})
            .get("parameters", {})
            .get("properties", {})
        )
        required = (
            action_schema.get("function", {}).get("parameters", {}).get("required", [])
        )

        # Define field definitions for the model
        field_definitions = {}
        for param_name, param_details in schema_props.items():
            param_type = str  # Default to string type
            param_desc = param_details.get("description", "")
            is_required = param_name in required

            # Basic type mapping (can be extended)
            if param_details.get("type") == "integer":
                param_type = int
            elif param_details.get("type") == "number":
                param_type = float
            elif param_details.get("type") == "boolean":
                param_type = bool

            # Create field with appropriate type and config
            field_definitions[param_name] = (
                param_type if is_required else Optional[param_type],
                Field(description=param_desc),
            )

        # Create the model
        if field_definitions:
            args_schema = create_model(
                f"{name.capitalize()}Schema", **field_definitions
            )
        else:
            # Fallback for empty schema
            args_schema = create_model(
                f"{name.capitalize()}Schema",
                input_text=(str, Field(description="Input for the action")),
            )

        super().__init__(name=name, description=description, args_schema=args_schema)
        self.enterprise_action_token = enterprise_action_token
        self.action_name = action_name
        self.action_schema = action_schema

    def _run(self, **kwargs) -> str:
        """Execute the specific enterprise action with validated parameters."""
        try:
            params = {k: v for k, v in kwargs.items() if v is not None}

            api_url = f"https://worker-actionkit.tools.crewai.com/projects/{ENTERPRISE_ACTION_KIT_PROJECT_ID}/actions"
            headers = {
                "Authorization": f"Bearer {self.enterprise_action_token}",
                "Content-Type": "application/json",
            }
            payload = {"action": self.action_name, "parameters": params}

            response = requests.post(
                url=api_url, headers=headers, json=payload, timeout=60
            )

            data = response.json()
            if not response.ok:
                error_message = data.get("error", {}).get("message", json.dumps(data))
                return f"API request failed: {error_message}"

            return json.dumps(data, indent=2)

        except Exception as e:
            return f"Error executing action {self.action_name}: {str(e)}"


class EnterpriseActionKitToolAdapter:
    """Adapter that creates BaseTool instances for enterprise actions."""

    def __init__(self, enterprise_action_token: str):
        """Initialize the adapter with an enterprise action token."""
        if not enterprise_action_token:
            raise ValueError("enterprise_action_token is required")

        self.enterprise_action_token = enterprise_action_token
        self._actions_schema = {}
        self._tools = None

    def tools(self) -> List[BaseTool]:
        """Get the list of tools created from enterprise actions.

        Returns:
            List of BaseTool instances, one for each enterprise action.
        """
        if self._tools is None:
            self._fetch_actions()
            self._create_tools()
        return self._tools

    def _fetch_actions(self):
        """Fetch available actions from the API."""
        try:
            actions_url = f"https://worker-actionkit.tools.crewai.com/projects/{ENTERPRISE_ACTION_KIT_PROJECT_ID}/actions"
            headers = {"Authorization": f"Bearer {self.enterprise_action_token}"}
            params = {"format": "json_schema"}

            response = requests.get(
                actions_url, headers=headers, params=params, timeout=30
            )
            response.raise_for_status()

            raw_data = response.json()
            if "actions" not in raw_data:
                print(f"Unexpected API response structure: {raw_data}")
                return

            # Parse the actions schema
            parsed_schema = {}
            action_categories = raw_data["actions"]

            for category, action_list in action_categories.items():
                if isinstance(action_list, list):
                    for action in action_list:
                        func_details = action.get("function")
                        if func_details and "name" in func_details:
                            action_name = func_details["name"]
                            parsed_schema[action_name] = action

            self._actions_schema = parsed_schema

        except Exception as e:
            print(f"Error fetching actions: {e}")
            import traceback

            traceback.print_exc()

    def _create_tools(self):
        """Create BaseTool instances for each action."""
        tools = []

        for action_name, action_schema in self._actions_schema.items():
            function_details = action_schema.get("function", {})
            description = function_details.get("description", f"Execute {action_name}")

            # Get parameter info for a better description
            parameters = function_details.get("parameters", {}).get("properties", {})
            param_info = []
            for param_name, param_details in parameters.items():
                param_desc = param_details.get("description", "")
                required = param_name in function_details.get("parameters", {}).get(
                    "required", []
                )
                param_info.append(
                    f"- {param_name}: {param_desc} {'(required)' if required else '(optional)'}"
                )

            full_description = f"{description}\n\nParameters:\n" + "\n".join(param_info)

            tool = EnterpriseActionTool(
                name=action_name.lower().replace(" ", "_"),
                description=full_description,
                action_name=action_name,
                action_schema=action_schema,
                enterprise_action_token=self.enterprise_action_token,
            )

            tools.append(tool)

        self._tools = tools

    # Adding context manager support for convenience, but direct usage is also supported
    def __enter__(self):
        return self.tools()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
