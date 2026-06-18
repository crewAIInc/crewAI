"""Definition models for inline CrewAI crew configurations."""

from __future__ import annotations

from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


__all__ = [
    "CrewAgentDefinition",
    "CrewDefinition",
    "CrewTaskDefinition",
    "PythonReferenceDefinition",
]


class PythonReferenceDefinition(BaseModel):
    """Dotted Python reference used by crew definitions."""

    python: str

    @field_validator("python")
    @classmethod
    def _validate_python_ref(cls, value: str) -> str:
        path = value.strip()
        if not path:
            raise ValueError("Python reference 'python' must be a string")
        if "." not in path:
            raise ValueError(
                f"Python reference '{path}' must be a dotted import path "
                "like 'module.attribute'"
            )
        return path


class CrewAgentDefinition(BaseModel):
    """Inline agent definition used by a crew definition."""

    model_config = ConfigDict(extra="allow")

    role: str
    goal: str
    backstory: str
    type: str | PythonReferenceDefinition | None = None
    settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("settings", mode="before")
    @classmethod
    def _validate_settings(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, dict):
            raise ValueError("agent.settings must be a mapping")
        return value or {}


class CrewTaskDefinition(BaseModel):
    """Task definition used by a crew definition."""

    model_config = ConfigDict(extra="allow")

    description: str
    expected_output: str
    name: str | None = None
    agent: str | None = None
    context: list[str] | None = None
    type: str | PythonReferenceDefinition | None = None


_CrewAgentsInput: TypeAlias = dict[str, CrewAgentDefinition] | list[dict[str, Any]]


class CrewDefinition(BaseModel):
    """In-memory JSON/YAML crew definition with inline agents and tasks."""

    model_config = ConfigDict(extra="allow")

    agents: dict[str, CrewAgentDefinition]
    tasks: list[CrewTaskDefinition]
    inputs: dict[str, Any] = Field(default_factory=dict)
    manager_agent: str | PythonReferenceDefinition | None = None

    @field_validator("inputs", mode="before")
    @classmethod
    def _validate_inputs(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, dict):
            raise ValueError("crew.inputs must be a mapping")
        return value or {}

    @field_validator(
        "agents",
        mode="before",
        json_schema_input_type=_CrewAgentsInput,
    )
    @classmethod
    def _validate_inline_agents(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return value
        if not isinstance(value, list):
            return value

        agents: dict[str, Any] = {}
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"agents[{index}] must be an inline agent mapping")

            if "name" in item:
                name = item["name"]
                if not isinstance(name, str) or not name:
                    raise ValueError(f"agents[{index}].name must be a non-empty string")
                agents[name] = {key: val for key, val in item.items() if key != "name"}
                continue

            if len(item) != 1:
                raise ValueError(
                    f"agents[{index}] must include a name field or be a one-key mapping"
                )
            name, definition = next(iter(item.items()))
            agents[str(name)] = definition

        return agents

    @model_validator(mode="after")
    def _validate_inline_shape(self) -> CrewDefinition:
        if not self.agents:
            raise ValueError("crew action requires inline agent definitions")

        if not self.tasks:
            raise ValueError("crew action requires a non-empty tasks list")
        return self
