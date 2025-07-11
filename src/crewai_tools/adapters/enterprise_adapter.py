import os
import json
import requests
from typing import List, Any, Dict, Literal, Optional, Union, get_origin
from pydantic import Field, create_model
from crewai.tools import BaseTool
import re


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
        self._model_registry = {}
        self._base_name = self._sanitize_name(name)

        schema_props, required = self._extract_schema_info(action_schema)

        # Define field definitions for the model
        field_definitions = {}
        for param_name, param_details in schema_props.items():
            param_desc = param_details.get("description", "")
            is_required = param_name in required

            try:
                field_type = self._process_schema_type(
                    param_details, self._sanitize_name(param_name).title()
                )
            except Exception as e:
                print(f"Warning: Could not process schema for {param_name}: {e}")
                field_type = str

            # Create field definition based on requirement
            field_definitions[param_name] = self._create_field_definition(
                field_type, is_required, param_desc
            )

        # Create the model
        if field_definitions:
            try:
                args_schema = create_model(
                    f"{self._base_name}Schema", **field_definitions
                )
            except Exception as e:
                print(f"Warning: Could not create main schema model: {e}")
                args_schema = create_model(
                    f"{self._base_name}Schema",
                    input_text=(str, Field(description="Input for the action")),
                )
        else:
            # Fallback for empty schema
            args_schema = create_model(
                f"{self._base_name}Schema",
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

    def _sanitize_name(self, name: str) -> str:
        """Sanitize names to create proper Python class names."""
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "", name)
        parts = sanitized.split("_")
        return "".join(word.capitalize() for word in parts if word)

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

    def _process_schema_type(self, schema: Dict[str, Any], type_name: str) -> type:
        """Process a JSON schema and return appropriate Python type."""
        if "anyOf" in schema:
            any_of_types = schema["anyOf"]
            is_nullable = any(t.get("type") == "null" for t in any_of_types)
            non_null_types = [t for t in any_of_types if t.get("type") != "null"]

            if non_null_types:
                base_type = self._process_schema_type(non_null_types[0], type_name)
                return Optional[base_type] if is_nullable else base_type
            return Optional[str]

        if "oneOf" in schema:
            return self._process_schema_type(schema["oneOf"][0], type_name)

        if "allOf" in schema:
            return self._process_schema_type(schema["allOf"][0], type_name)

        json_type = schema.get("type", "string")

        if "enum" in schema:
            enum_values = schema["enum"]
            if not enum_values:
                return self._map_json_type_to_python(json_type)
            return Literal[tuple(enum_values)]  # type: ignore

        if json_type == "array":
            items_schema = schema.get("items", {"type": "string"})
            item_type = self._process_schema_type(items_schema, f"{type_name}Item")
            return List[item_type]

        if json_type == "object":
            return self._create_nested_model(schema, type_name)

        return self._map_json_type_to_python(json_type)

    def _create_nested_model(self, schema: Dict[str, Any], model_name: str) -> type:
        """Create a nested Pydantic model for complex objects."""
        full_model_name = f"{self._base_name}{model_name}"

        if full_model_name in self._model_registry:
            return self._model_registry[full_model_name]

        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        if not properties:
            return dict

        field_definitions = {}
        for prop_name, prop_schema in properties.items():
            prop_desc = prop_schema.get("description", "")
            is_required = prop_name in required_fields

            try:
                prop_type = self._process_schema_type(
                    prop_schema, f"{model_name}{self._sanitize_name(prop_name).title()}"
                )
            except Exception as e:
                print(f"Warning: Could not process schema for {prop_name}: {e}")
                prop_type = str

            field_definitions[prop_name] = self._create_field_definition(
                prop_type, is_required, prop_desc
            )

        try:
            nested_model = create_model(full_model_name, **field_definitions)
            self._model_registry[full_model_name] = nested_model
            return nested_model
        except Exception as e:
            print(f"Warning: Could not create nested model {full_model_name}: {e}")
            return dict

    def _create_field_definition(
        self, field_type: type, is_required: bool, description: str
    ) -> tuple:
        """Create Pydantic field definition based on type and requirement."""
        if is_required:
            return (field_type, Field(description=description))
        else:
            if get_origin(field_type) is Union:
                return (field_type, Field(default=None, description=description))
            else:
                return (
                    Optional[field_type],
                    Field(default=None, description=description),
                )

    def _map_json_type_to_python(self, json_type: str) -> type:
        """Map basic JSON schema types to Python types."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        return type_mapping.get(json_type, str)

    def _get_required_nullable_fields(self) -> List[str]:
        """Get a list of required nullable fields from the action schema."""
        schema_props, required = self._extract_schema_info(self.action_schema)

        required_nullable_fields = []
        for param_name in required:
            param_details = schema_props.get(param_name, {})
            if self._is_nullable_type(param_details):
                required_nullable_fields.append(param_name)

        return required_nullable_fields

    def _is_nullable_type(self, schema: Dict[str, Any]) -> bool:
        """Check if a schema represents a nullable type."""
        if "anyOf" in schema:
            return any(t.get("type") == "null" for t in schema["anyOf"])
        return schema.get("type") == "null"

    def _run(self, **kwargs) -> str:
        """Execute the specific enterprise action with validated parameters."""
        try:
            cleaned_kwargs = {}
            for key, value in kwargs.items():
                if value is not None:
                    cleaned_kwargs[key] = value

            required_nullable_fields = self._get_required_nullable_fields()

            for field_name in required_nullable_fields:
                if field_name not in cleaned_kwargs:
                    cleaned_kwargs[field_name] = None

            api_url = f"{self.enterprise_action_kit_project_url}/{self.enterprise_action_kit_project_id}/actions"
            headers = {
                "Authorization": f"Bearer {self.enterprise_action_token}",
                "Content-Type": "application/json",
            }
            payload = {"action": self.action_name, "parameters": cleaned_kwargs}

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
        """Get the list of tools created from enterprise actions."""
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

    def _generate_detailed_description(
        self, schema: Dict[str, Any], indent: int = 0
    ) -> List[str]:
        """Generate detailed description for nested schema structures."""
        descriptions = []
        indent_str = "  " * indent

        schema_type = schema.get("type", "string")

        if schema_type == "object":
            properties = schema.get("properties", {})
            required_fields = schema.get("required", [])

            if properties:
                descriptions.append(f"{indent_str}Object with properties:")
                for prop_name, prop_schema in properties.items():
                    prop_desc = prop_schema.get("description", "")
                    is_required = prop_name in required_fields
                    req_str = " (required)" if is_required else " (optional)"
                    descriptions.append(
                        f"{indent_str}  - {prop_name}: {prop_desc}{req_str}"
                    )

                    if prop_schema.get("type") == "object":
                        descriptions.extend(
                            self._generate_detailed_description(prop_schema, indent + 2)
                        )
                    elif prop_schema.get("type") == "array":
                        items_schema = prop_schema.get("items", {})
                        if items_schema.get("type") == "object":
                            descriptions.append(f"{indent_str}    Array of objects:")
                            descriptions.extend(
                                self._generate_detailed_description(
                                    items_schema, indent + 3
                                )
                            )
                        elif "enum" in items_schema:
                            descriptions.append(
                                f"{indent_str}    Array of enum values: {items_schema['enum']}"
                            )
                    elif "enum" in prop_schema:
                        descriptions.append(
                            f"{indent_str}    Enum values: {prop_schema['enum']}"
                        )

        return descriptions

    def _create_tools(self):
        """Create BaseTool instances for each action."""
        tools = []

        for action_name, action_schema in self._actions_schema.items():
            function_details = action_schema.get("function", {})
            description = function_details.get("description", f"Execute {action_name}")

            parameters = function_details.get("parameters", {})
            param_descriptions = []

            if parameters.get("properties"):
                param_descriptions.append("\nDetailed Parameter Structure:")
                param_descriptions.extend(
                    self._generate_detailed_description(parameters)
                )

            full_description = description + "\n".join(param_descriptions)

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

    def __enter__(self):
        return self.tools()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
