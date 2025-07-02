import os
import json
import requests
from typing import List, Any, Dict, Optional
from pydantic import Field, create_model
from crewai.tools import BaseTool

# DEFAULTS
ENTERPRISE_ACTION_KIT_PROJECT_ID = "dd525517-df22-49d2-a69e-6a0eed211166"
ENTERPRISE_ACTION_KIT_PROJECT_URL = "https://worker-actionkit.tools.crewai.com/projects"


class EnterpriseActionTool(BaseTool):
    """A tool that executes a specific enterprise action."""

    enterprise_action_token: str = Field(
        default="", description="The enterprise action token"
    )
    action_name: str = Field(default="", description="The name of the action")
    action_schema: Dict[str, Any] = Field(
        default={}, description="The schema of the action"
    )
    enterprise_action_kit_project_id: str = Field(
        default=ENTERPRISE_ACTION_KIT_PROJECT_ID, description="The project id"
    )
    enterprise_action_kit_project_url: str = Field(
        default=ENTERPRISE_ACTION_KIT_PROJECT_URL, description="The project url"
    )

    def __init__(
        self,
        name: str,
        description: str,
        enterprise_action_token: str,
        action_name: str,
        action_schema: Dict[str, Any],
        enterprise_action_kit_project_url: str = ENTERPRISE_ACTION_KIT_PROJECT_URL,
        enterprise_action_kit_project_id: str = ENTERPRISE_ACTION_KIT_PROJECT_ID,
    ):
        schema_props, required = self._extract_schema_info(action_schema)

        # Define field definitions for the model
        field_definitions = {}
        for param_name, param_details in schema_props.items():
            param_desc = param_details.get("description", "")
            is_required = param_name in required
            is_nullable, param_type = self._analyze_field_type(param_details)

            # Create field definition based on nullable and required status
            field_definitions[param_name] = self._create_field_definition(
                param_type, is_required, is_nullable, param_desc
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

        if enterprise_action_kit_project_id is not None:
            self.enterprise_action_kit_project_id = enterprise_action_kit_project_id
        if enterprise_action_kit_project_url is not None:
            self.enterprise_action_kit_project_url = enterprise_action_kit_project_url

    def _extract_schema_info(
        self, action_schema: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[str]]:
        """Extract schema properties and required fields from action schema."""
        schema_props = (
            action_schema.get("function", {})
            .get("parameters", {})
            .get("properties", {})
        )
        required = (
            action_schema.get("function", {}).get("parameters", {}).get("required", [])
        )
        return schema_props, required

    def _analyze_field_type(self, param_details: Dict[str, Any]) -> tuple[bool, type]:
        """Analyze field type and nullability from parameter details."""
        is_nullable = False
        param_type = str  # Default type

        if "anyOf" in param_details:
            any_of_types = param_details["anyOf"]
            is_nullable = any(t.get("type") == "null" for t in any_of_types)
            non_null_types = [t for t in any_of_types if t.get("type") != "null"]
            if non_null_types:
                first_type = non_null_types[0].get("type", "string")
                param_type = self._map_json_type_to_python(
                    first_type, non_null_types[0]
                )
        else:
            json_type = param_details.get("type", "string")
            param_type = self._map_json_type_to_python(json_type, param_details)
            is_nullable = json_type == "null"

        return is_nullable, param_type

    def _create_field_definition(
        self, param_type: type, is_required: bool, is_nullable: bool, param_desc: str
    ) -> tuple:
        """Create Pydantic field definition based on type, requirement, and nullability."""
        if is_nullable:
            return (
                Optional[param_type],
                Field(default=None, description=param_desc),
            )
        elif is_required:
            return (
                param_type,
                Field(description=param_desc),
            )
        else:
            return (
                Optional[param_type],
                Field(default=None, description=param_desc),
            )

    def _map_json_type_to_python(
        self, json_type: str, param_details: Dict[str, Any]
    ) -> type:
        """Map JSON schema types to Python types."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_mapping.get(json_type, str)

    def _get_required_nullable_fields(self) -> List[str]:
        """Get a list of required nullable fields from the action schema."""
        schema_props, required = self._extract_schema_info(self.action_schema)

        required_nullable_fields = []
        for param_name in required:
            param_details = schema_props.get(param_name, {})
            is_nullable, _ = self._analyze_field_type(param_details)
            if is_nullable:
                required_nullable_fields.append(param_name)

        return required_nullable_fields

    def _run(self, **kwargs) -> str:
        """Execute the specific enterprise action with validated parameters."""
        try:
            required_nullable_fields = self._get_required_nullable_fields()

            for field_name in required_nullable_fields:
                if field_name not in kwargs:
                    kwargs[field_name] = None

            params = {k: v for k, v in kwargs.items() if v is not None}

            api_url = f"{self.enterprise_action_kit_project_url}/{self.enterprise_action_kit_project_id}/actions"
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

    def __init__(
        self,
        enterprise_action_token: str,
        enterprise_action_kit_project_url: str = ENTERPRISE_ACTION_KIT_PROJECT_URL,
        enterprise_action_kit_project_id: str = ENTERPRISE_ACTION_KIT_PROJECT_ID,
    ):
        """Initialize the adapter with an enterprise action token."""

        self.enterprise_action_token = enterprise_action_token
        self._actions_schema = {}
        self._tools = None
        self.enterprise_action_kit_project_id = enterprise_action_kit_project_id
        self.enterprise_action_kit_project_url = enterprise_action_kit_project_url

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
            if (
                self.enterprise_action_token is None
                or self.enterprise_action_token == ""
            ):
                self.enterprise_action_token = os.environ.get(
                    "CREWAI_ENTERPRISE_TOOLS_TOKEN"
                )

            actions_url = f"{self.enterprise_action_kit_project_url}/{self.enterprise_action_kit_project_id}/actions"
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
                enterprise_action_kit_project_id=self.enterprise_action_kit_project_id,
                enterprise_action_kit_project_url=self.enterprise_action_kit_project_url,
            )

            tools.append(tool)

        self._tools = tools

    # Adding context manager support for convenience, but direct usage is also supported
    def __enter__(self):
        return self.tools()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
