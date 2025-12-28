"""Crewai Enterprise Tools."""
import os
import json
import re
from typing import Any, Optional, Union, cast, get_origin

from crewai.tools import BaseTool
from pydantic import Field, create_model
import requests

from crewai_tools.tools.crewai_platform_tools.misc import (
    get_platform_api_base_url,
    get_platform_integration_token,
)


class AllOfSchemaAnalyzer:
    """Helper class to analyze and merge allOf schemas."""

    def __init__(self, schemas: list[dict[str, Any]]):
        self.schemas = schemas
        self._explicit_types: list[str] = []
        self._merged_properties: dict[str, Any] = {}
        self._merged_required: list[str] = []
        self._analyze_schemas()

    def _analyze_schemas(self) -> None:
        """Analyze all schemas and extract relevant information."""
        for schema in self.schemas:
            if "type" in schema:
                self._explicit_types.append(schema["type"])

            # Merge object properties
            if schema.get("type") == "object" and "properties" in schema:
                self._merged_properties.update(schema["properties"])
                if "required" in schema:
                    self._merged_required.extend(schema["required"])

    def has_consistent_type(self) -> bool:
        """Check if all schemas have the same explicit type."""
        return len(set(self._explicit_types)) == 1 if self._explicit_types else False

    def get_consistent_type(self) -> type[Any]:
        """Get the consistent type if all schemas agree."""
        if not self.has_consistent_type():
            raise ValueError("No consistent type found")

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        return type_mapping.get(self._explicit_types[0], str)

    def has_object_schemas(self) -> bool:
        """Check if any schemas are object types with properties."""
        return bool(self._merged_properties)

    def get_merged_properties(self) -> dict[str, Any]:
        """Get merged properties from all object schemas."""
        return self._merged_properties

    def get_merged_required_fields(self) -> list[str]:
        """Get merged required fields from all object schemas."""
        return list(set(self._merged_required))  # Remove duplicates

    def get_fallback_type(self) -> type[Any]:
        """Get a fallback type when merging fails."""
        if self._explicit_types:
            # Use the first explicit type
            type_mapping = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": list,
                "object": dict,
                "null": type(None),
            }
            return type_mapping.get(self._explicit_types[0], str)
        return str


class CrewAIPlatformActionTool(BaseTool):
    action_name: str = Field(default="", description="The name of the action")
    action_schema: dict[str, Any] = Field(
        default_factory=dict, description="The schema of the action"
    )

    def __init__(
        self,
        description: str,
        action_name: str,
        action_schema: dict[str, Any],
    ):
        self._model_registry: dict[str, type[Any]] = {}
        self._base_name = self._sanitize_name(action_name)

        schema_props, required = self._extract_schema_info(action_schema)

        field_definitions: dict[str, Any] = {}
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
            except Exception:
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

    @staticmethod
    def _sanitize_name(name: str) -> str:
        name = name.lower().replace(" ", "_")
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "", name)
        parts = sanitized.split("_")
        return "".join(word.capitalize() for word in parts if word)

    @staticmethod
    def _extract_schema_info(
        action_schema: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        schema_props = (
            action_schema.get("function", {})
            .get("parameters", {})
            .get("properties", {})
        )
        required = (
            action_schema.get("function", {}).get("parameters", {}).get("required", [])
        )
        return schema_props, required

    def _process_schema_type(self, schema: dict[str, Any], type_name: str) -> type[Any]:
        """
        Process a JSON Schema type definition into a Python type.

        Handles complex schema constructs like anyOf, oneOf, allOf, enums, arrays, and objects.
        """
        # Handle composite schema types (anyOf, oneOf, allOf)
        if composite_type := self._process_composite_schema(schema, type_name):
            return composite_type

        # Handle primitive types and simple constructs
        return self._process_primitive_schema(schema, type_name)

    def _process_composite_schema(
        self, schema: dict[str, Any], type_name: str
    ) -> type[Any] | None:
        """Process composite schema types: anyOf, oneOf, allOf."""
        if "anyOf" in schema:
            return self._process_any_of_schema(schema["anyOf"], type_name)
        if "oneOf" in schema:
            return self._process_one_of_schema(schema["oneOf"], type_name)
        if "allOf" in schema:
            return self._process_all_of_schema(schema["allOf"], type_name)
        return None

    def _process_any_of_schema(
        self, any_of_types: list[dict[str, Any]], type_name: str
    ) -> type[Any]:
        """Process anyOf schema - creates Union of possible types."""
        is_nullable = any(t.get("type") == "null" for t in any_of_types)
        non_null_types = [t for t in any_of_types if t.get("type") != "null"]

        if not non_null_types:
            return cast(
                type[Any], cast(object, str | None)
            )  # fallback for only-null case

        base_type = (
            self._process_schema_type(non_null_types[0], type_name)
            if len(non_null_types) == 1
            else self._create_union_type(non_null_types, type_name, "AnyOf")
        )
        return base_type | None if is_nullable else base_type  # type: ignore[return-value]

    def _process_one_of_schema(
        self, one_of_types: list[dict[str, Any]], type_name: str
    ) -> type[Any]:
        """Process oneOf schema - creates Union of mutually exclusive types."""
        return (
            self._process_schema_type(one_of_types[0], type_name)
            if len(one_of_types) == 1
            else self._create_union_type(one_of_types, type_name, "OneOf")
        )

    def _process_all_of_schema(
        self, all_of_schemas: list[dict[str, Any]], type_name: str
    ) -> type[Any]:
        """Process allOf schema - merges schemas that must all be satisfied."""
        if len(all_of_schemas) == 1:
            return self._process_schema_type(all_of_schemas[0], type_name)
        return self._merge_all_of_schemas(all_of_schemas, type_name)

    def _create_union_type(
        self, schemas: list[dict[str, Any]], type_name: str, prefix: str
    ) -> type[Any]:
        """Create a Union type from multiple schemas."""
        return Union[  # type: ignore  # noqa: UP007
            tuple(
                self._process_schema_type(schema, f"{type_name}{prefix}{i}")
                for i, schema in enumerate(schemas)
            )
        ]

    def _process_primitive_schema(
        self, schema: dict[str, Any], type_name: str
    ) -> type[Any]:
        """Process primitive schema types: string, number, array, object, etc."""
        json_type = schema.get("type", "string")

        if "enum" in schema:
            return self._process_enum_schema(schema, json_type)

        if json_type == "array":
            return self._process_array_schema(schema, type_name)

        if json_type == "object":
            return self._create_nested_model(schema, type_name)

        return self._map_json_type_to_python(json_type)

    def _process_enum_schema(self, schema: dict[str, Any], json_type: str) -> type[Any]:
        """Process enum schema - currently falls back to base type."""
        enum_values = schema["enum"]
        if not enum_values:
            return self._map_json_type_to_python(json_type)

        # For Literal types, we need to pass the values directly, not as a tuple
        # This is a workaround since we can't dynamically create Literal types easily
        # Fall back to the base JSON type for now
        return self._map_json_type_to_python(json_type)

    def _process_array_schema(
        self, schema: dict[str, Any], type_name: str
    ) -> type[Any]:
        items_schema = schema.get("items", {"type": "string"})
        item_type = self._process_schema_type(items_schema, f"{type_name}Item")
        return list[item_type]  # type: ignore

    def _merge_all_of_schemas(
        self, schemas: list[dict[str, Any]], type_name: str
    ) -> type[Any]:
        schema_analyzer = AllOfSchemaAnalyzer(schemas)

        if schema_analyzer.has_consistent_type():
            return schema_analyzer.get_consistent_type()

        if schema_analyzer.has_object_schemas():
            return self._create_merged_object_model(
                schema_analyzer.get_merged_properties(),
                schema_analyzer.get_merged_required_fields(),
                type_name,
            )

        return schema_analyzer.get_fallback_type()

    def _create_merged_object_model(
        self, properties: dict[str, Any], required: list[str], model_name: str
    ) -> type[Any]:
        full_model_name = f"{self._base_name}{model_name}AllOf"

        if full_model_name in self._model_registry:
            return self._model_registry[full_model_name]

        if not properties:
            return dict

        field_definitions = self._build_field_definitions(
            properties, required, model_name
        )

        try:
            merged_model = create_model(full_model_name, **field_definitions)
            self._model_registry[full_model_name] = merged_model
            return merged_model
        except Exception:
            return dict

    def _build_field_definitions(
        self, properties: dict[str, Any], required: list[str], model_name: str
    ) -> dict[str, Any]:
        field_definitions = {}

        for prop_name, prop_schema in properties.items():
            prop_desc = prop_schema.get("description", "")
            is_required = prop_name in required

            try:
                prop_type = self._process_schema_type(
                    prop_schema, f"{model_name}{self._sanitize_name(prop_name).title()}"
                )
            except Exception:
                prop_type = str

            field_definitions[prop_name] = self._create_field_definition(
                prop_type, is_required, prop_desc
            )

        return field_definitions

    def _create_nested_model(
        self, schema: dict[str, Any], model_name: str
    ) -> type[Any]:
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
            nested_model = create_model(full_model_name, **field_definitions)  # type: ignore
            self._model_registry[full_model_name] = nested_model
            return nested_model
        except Exception:
            return dict

    def _create_field_definition(
        self, field_type: type[Any], is_required: bool, description: str
    ) -> tuple:
        if is_required:
            return (field_type, Field(description=description))
        if get_origin(field_type) is Union:
            return (field_type, Field(default=None, description=description))
        return (
            Optional[field_type],  # noqa: UP045
            Field(default=None, description=description),
        )

    def _map_json_type_to_python(self, json_type: str) -> type[Any]:
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

    def _get_required_nullable_fields(self) -> list[str]:
        schema_props, required = self._extract_schema_info(self.action_schema)

        required_nullable_fields = []
        for param_name in required:
            param_details = schema_props.get(param_name, {})
            if self._is_nullable_type(param_details):
                required_nullable_fields.append(param_name)

        return required_nullable_fields

    def _is_nullable_type(self, schema: dict[str, Any]) -> bool:
        if "anyOf" in schema:
            return any(t.get("type") == "null" for t in schema["anyOf"])
        return schema.get("type") == "null"

    def _run(self, **kwargs) -> str:
        try:
            cleaned_kwargs = {
                key: value for key, value in kwargs.items() if value is not None
            }

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
                url=api_url,
                headers=headers,
                json=payload,
                timeout=60,
                verify=os.environ.get("CREWAI_FACTORY", "false").lower() != "true",
            )

            data = response.json()
            if not response.ok:
                error_message = data.get("error", {}).get("message", json.dumps(data))
                return f"API request failed: {error_message}"

            return json.dumps(data, indent=2)

        except Exception as e:
            return f"Error executing action {self.action_name}: {e!s}"
