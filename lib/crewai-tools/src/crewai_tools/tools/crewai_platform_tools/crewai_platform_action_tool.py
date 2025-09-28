"""
Crewai Enterprise Tools
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional, Type, Union, cast, get_origin

from crewai.tools import BaseTool
from pydantic import Field, create_model
import requests

from crewai_tools.tools.crewai_platform_tools.misc import (
    get_platform_api_base_url,
    get_platform_integration_token,
)


class CrewAIPlatformActionTool(BaseTool):
    action_name: str = Field(default="", description="The name of the action")
    action_schema: Dict[str, Any] = Field(
        default_factory=dict, description="The schema of the action"
    )

    def __init__(
        self,
        description: str,
        action_name: str,
        action_schema: Dict[str, Any],
    ):
        self._model_registry = {}
        self._base_name = self._sanitize_name(action_name)

        schema_props, required = self._extract_schema_info(action_schema)

        field_definitions = {}
        for param_name, param_details in schema_props.items():
            param_desc = param_details.get("description", "")
            is_required = param_name in required

            try:
                field_type = self._process_schema_type(
                    param_details, self._sanitize_name(param_name).title()
                )
            except Exception:
                field_type = str

            field_definitions[param_name] = self._create_field_definition(
                field_type, is_required, param_desc
            )

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
            args_schema = create_model(
                f"{self._base_name}Schema",
                input_text=(str, Field(description="Input for the action")),
            )

        super().__init__(
            name=action_name.lower().replace(" ", "_"),
            description=description,
            args_schema=args_schema,
        )
        self.action_name = action_name
        self.action_schema = action_schema

    def _sanitize_name(self, name: str) -> str:
        name = name.lower().replace(" ", "_")
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "", name)
        parts = sanitized.split("_")
        return "".join(word.capitalize() for word in parts if word)

    def _extract_schema_info(
        self, action_schema: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[str]]:
        schema_props = (
            action_schema.get("function", {})
            .get("parameters", {})
            .get("properties", {})
        )
        required = (
            action_schema.get("function", {}).get("parameters", {}).get("required", [])
        )
        return schema_props, required

    def _process_schema_type(self, schema: Dict[str, Any], type_name: str) -> Type[Any]:
        if "anyOf" in schema:
            any_of_types = schema["anyOf"]
            is_nullable = any(t.get("type") == "null" for t in any_of_types)
            non_null_types = [t for t in any_of_types if t.get("type") != "null"]

            if non_null_types:
                base_type = self._process_schema_type(non_null_types[0], type_name)
                return Optional[base_type] if is_nullable else base_type
            return cast(Type[Any], Optional[str])

        if "oneOf" in schema:
            return self._process_schema_type(schema["oneOf"][0], type_name)

        if "allOf" in schema:
            return self._process_schema_type(schema["allOf"][0], type_name)

        json_type = schema.get("type", "string")

        if "enum" in schema:
            enum_values = schema["enum"]
            if not enum_values:
                return self._map_json_type_to_python(json_type)
            return Literal[tuple(enum_values)]

        if json_type == "array":
            items_schema = schema.get("items", {"type": "string"})
            item_type = self._process_schema_type(items_schema, f"{type_name}Item")
            return List[item_type]

        if json_type == "object":
            return self._create_nested_model(schema, type_name)

        return self._map_json_type_to_python(json_type)

    def _create_nested_model(
        self, schema: Dict[str, Any], model_name: str
    ) -> Type[Any]:
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
            except Exception:
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
        self, field_type: Type[Any], is_required: bool, description: str
    ) -> tuple:
        if is_required:
            return (field_type, Field(description=description))
        if get_origin(field_type) is Union:
            return (field_type, Field(default=None, description=description))
        return (
            Optional[field_type],
            Field(default=None, description=description),
        )

    def _map_json_type_to_python(self, json_type: str) -> Type[Any]:
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
        schema_props, required = self._extract_schema_info(self.action_schema)

        required_nullable_fields = []
        for param_name in required:
            param_details = schema_props.get(param_name, {})
            if self._is_nullable_type(param_details):
                required_nullable_fields.append(param_name)

        return required_nullable_fields

    def _is_nullable_type(self, schema: Dict[str, Any]) -> bool:
        if "anyOf" in schema:
            return any(t.get("type") == "null" for t in schema["anyOf"])
        return schema.get("type") == "null"

    def _run(self, **kwargs) -> str:
        try:
            cleaned_kwargs = {}
            for key, value in kwargs.items():
                if value is not None:
                    cleaned_kwargs[key] = value

            required_nullable_fields = self._get_required_nullable_fields()

            for field_name in required_nullable_fields:
                if field_name not in cleaned_kwargs:
                    cleaned_kwargs[field_name] = None

            api_url = (
                f"{get_platform_api_base_url()}/actions/{self.action_name}/execute"
            )
            token = get_platform_integration_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            payload = cleaned_kwargs

            response = requests.post(
                url=api_url, headers=headers, json=payload, timeout=60
            )

            data = response.json()
            if not response.ok:
                error_message = data.get("error", {}).get("message", json.dumps(data))
                return f"API request failed: {error_message}"

            return json.dumps(data, indent=2)

        except Exception as e:
            return f"Error executing action {self.action_name}: {e!s}"
