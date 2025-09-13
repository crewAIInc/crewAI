#!/usr/bin/env python3

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from crewai_tools import tools
from crewai.tools.base_tool import BaseTool, EnvVar

from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import PydanticOmit


class SchemaGenerator(GenerateJsonSchema):
    def handle_invalid_for_json_schema(self, schema, error_info):
        raise PydanticOmit


class ToolSpecExtractor:
    def __init__(self) -> None:
        self.tools_spec: List[Dict[str, Any]] = []
        self.processed_tools: set[str] = set()

    def extract_all_tools(self) -> List[Dict[str, Any]]:
        for name in dir(tools):
            if name.endswith("Tool") and name not in self.processed_tools:
                obj = getattr(tools, name, None)
                if inspect.isclass(obj):
                    self.extract_tool_info(obj)
                    self.processed_tools.add(name)
        return self.tools_spec

    def extract_tool_info(self, tool_class: BaseTool) -> None:
        try:
            core_schema = tool_class.__pydantic_core_schema__
            if not core_schema:
                return

            schema = self._unwrap_schema(core_schema)
            fields = schema.get("schema", {}).get("fields", {})

            tool_info = {
                "name": tool_class.__name__,
                "humanized_name": self._extract_field_default(
                    fields.get("name"), fallback=tool_class.__name__
                ),
                "description": self._extract_field_default(
                    fields.get("description")
                ).strip(),
                "run_params_schema": self._extract_params(fields.get("args_schema")),
                "init_params_schema": self._extract_init_params(tool_class),
                "env_vars": self._extract_env_vars(fields.get("env_vars")),
                "package_dependencies": self._extract_field_default(
                    fields.get("package_dependencies"), fallback=[]
                ),
            }

            self.tools_spec.append(tool_info)

        except Exception as e:
            print(f"Error extracting {tool_class.__name__}: {e}")

    def _unwrap_schema(self, schema: Dict) -> Dict:
        while (
            schema.get("type") in {"function-after", "default"} and "schema" in schema
        ):
            schema = schema["schema"]
        return schema

    def _extract_field_default(self, field: Optional[Dict], fallback: str = "") -> str:
        if not field:
            return fallback

        schema = field.get("schema", {})
        default = schema.get("default")
        return default if isinstance(default, (list, str, int)) else fallback

    def _extract_params(
        self, args_schema_field: Optional[Dict]
    ) -> List[Dict[str, str]]:
        if not args_schema_field:
            return {}

        args_schema_class = args_schema_field.get("schema", {}).get("default")
        if not (
            inspect.isclass(args_schema_class)
            and hasattr(args_schema_class, "__pydantic_core_schema__")
        ):
            return {}

        try:
            return args_schema_class.model_json_schema(
                schema_generator=SchemaGenerator, mode="validation"
            )
        except Exception as e:
            print(f"Error extracting params from {args_schema_class}: {e}")
            return {}

    def _extract_env_vars(self, env_vars_field: Optional[Dict]) -> List[Dict[str, str]]:
        if not env_vars_field:
            return []

        env_vars = []
        for env_var in env_vars_field.get("schema", {}).get("default", []):
            if isinstance(env_var, EnvVar):
                env_vars.append(
                    {
                        "name": env_var.name,
                        "description": env_var.description,
                        "required": env_var.required,
                        "default": env_var.default,
                    }
                )
        return env_vars

    def _extract_init_params(self, tool_class: BaseTool) -> dict:
        ignored_init_params = [
            "name",
            "description",
            "env_vars",
            "args_schema",
            "description_updated",
            "cache_function",
            "result_as_answer",
            "max_usage_count",
            "current_usage_count",
            "package_dependencies",
        ]

        json_schema = tool_class.model_json_schema(
            schema_generator=SchemaGenerator, mode="serialization"
        )

        properties = {}
        for key, value in json_schema["properties"].items():
            if key not in ignored_init_params:
                properties[key] = value

        json_schema["properties"] = properties
        return json_schema

    def save_to_json(self, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"tools": self.tools_spec}, f, indent=2, sort_keys=True)
        print(f"Saved tool specs to {output_path}")


if __name__ == "__main__":
    output_file = Path(__file__).parent / "tool.specs.json"
    extractor = ToolSpecExtractor()

    specs = extractor.extract_all_tools()
    extractor.save_to_json(str(output_file))

    print(f"Extracted {len(specs)} tool classes.")
