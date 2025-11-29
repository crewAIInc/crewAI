#!/usr/bin/env python3

from collections.abc import Mapping
import inspect
import json
from pathlib import Path
from typing import Any

from crewai.tools.base_tool import BaseTool, EnvVar
from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import PydanticOmit

from crewai_tools import tools


class SchemaGenerator(GenerateJsonSchema):
    def handle_invalid_for_json_schema(
        self, schema: Any, error_info: Any
    ) -> dict[str, Any]:
        raise PydanticOmit


class ToolSpecExtractor:
    def __init__(self) -> None:
        self.tools_spec: list[dict[str, Any]] = []
        self.processed_tools: set[str] = set()

    def extract_all_tools(self) -> list[dict[str, Any]]:
        for name in dir(tools):
            if name.endswith("Tool") and name not in self.processed_tools:
                obj = getattr(tools, name, None)
                if inspect.isclass(obj) and issubclass(obj, BaseTool):
                    self.extract_tool_info(obj)
                    self.processed_tools.add(name)
        return self.tools_spec

    def extract_tool_info(self, tool_class: type[BaseTool]) -> None:
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
                "description": str(
                    self._extract_field_default(fields.get("description"))
                ).strip(),
                "run_params_schema": self._extract_params(fields.get("args_schema")),
                "init_params_schema": self._extract_init_params(tool_class),
                "env_vars": self._extract_env_vars(fields.get("env_vars")),
                "package_dependencies": self._extract_field_default(
                    fields.get("package_dependencies"), fallback=[]
                ),
            }

            self.tools_spec.append(tool_info)

        except Exception:  # noqa: S110
            pass

    @staticmethod
    def _unwrap_schema(schema: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = dict(schema)
        while (
            result.get("type") in {"function-after", "default"} and "schema" in result
        ):
            result = dict(result["schema"])
        return result

    @staticmethod
    def _extract_field_default(
        field: dict[str, Any] | None, fallback: str | list[Any] = ""
    ) -> str | list[Any] | int:
        if not field:
            return fallback

        schema = field.get("schema", {})
        default = schema.get("default")
        return default if isinstance(default, (list, str, int)) else fallback

    @staticmethod
    def _extract_params(args_schema_field: dict[str, Any] | None) -> dict[str, Any]:
        if not args_schema_field:
            return {}

        args_schema_class = args_schema_field.get("schema", {}).get("default")
        if not (
            inspect.isclass(args_schema_class)
            and issubclass(args_schema_class, BaseModel)
        ):
            return {}

        try:
            return args_schema_class.model_json_schema(schema_generator=SchemaGenerator)
        except Exception:
            return {}

    @staticmethod
    def _extract_env_vars(
        env_vars_field: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not env_vars_field:
            return []

        return [
            {
                "name": env_var.name,
                "description": env_var.description,
                "required": env_var.required,
                "default": env_var.default,
            }
            for env_var in env_vars_field.get("schema", {}).get("default", [])
            if isinstance(env_var, EnvVar)
        ]

    @staticmethod
    def _extract_init_params(tool_class: type[BaseTool]) -> dict[str, Any]:
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

        json_schema["properties"] = {
            key: value
            for key, value in json_schema["properties"].items()
            if key not in ignored_init_params
        }
        return json_schema

    def save_to_json(self, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"tools": self.tools_spec}, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    output_file = Path(__file__).parent / "tool.specs.json"
    extractor = ToolSpecExtractor()

    extractor.extract_all_tools()
    extractor.save_to_json(str(output_file))
