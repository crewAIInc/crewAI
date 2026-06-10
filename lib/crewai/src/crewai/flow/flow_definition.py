"""Flow Structure: the serializable, language-agnostic Flow contract.

Defines :class:`FlowDefinition` and its sub-models — a static, textual
(JSON/YAML) representation of a Flow: its methods, trigger conditions,
state, and configuration. It is independent of the Python authoring
layer that may have produced it and of the engine that runs it (see
``runtime``).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal as TypingLiteral

from pydantic import BaseModel, ConfigDict, Field
import yaml

from crewai.flow.conversational_definition import (
    FlowConversationalDefinition,
    FlowConversationalRouterDefinition,
)


logger = logging.getLogger(__name__)

FlowDefinitionCondition = str | dict[str, Any]

__all__ = [
    "FlowConfigDefinition",
    "FlowConversationalDefinition",
    "FlowConversationalRouterDefinition",
    "FlowDefinition",
    "FlowDefinitionCondition",
    "FlowDefinitionDiagnostic",
    "FlowHumanFeedbackDefinition",
    "FlowMethodDefinition",
    "FlowPersistenceDefinition",
    "FlowStateDefinition",
]


class FlowDefinitionDiagnostic(BaseModel):
    """A non-fatal Flow Definition build or validation diagnostic."""

    code: str
    message: str
    severity: TypingLiteral["warning", "error"] = "warning"
    path: str | None = None


class FlowStateDefinition(BaseModel):
    """Static description of a Flow state contract."""

    type: TypingLiteral["dict", "pydantic", "unknown"] = "dict"
    ref: str | None = None
    default: Any = None


class FlowConfigDefinition(BaseModel):
    """Serializable Flow-level configuration."""

    tracing: bool | None = None
    stream: bool = False
    memory: Any = None
    input_provider: Any = None
    suppress_flow_events: bool = False
    max_method_calls: int = 100


class FlowPersistenceDefinition(BaseModel):
    """Static persistence configuration."""

    enabled: bool = False
    verbose: bool = False
    persistence: Any = None


class FlowHumanFeedbackDefinition(BaseModel):
    """Static human feedback configuration."""

    message: str
    emit: list[str] | None = None
    llm: Any = "gpt-4o-mini"
    default_outcome: str | None = None
    metadata: dict[str, Any] | None = None
    provider: Any = None
    learn: bool = False
    learn_source: str = "hitl"
    learn_strict: bool = False


class FlowMethodDefinition(BaseModel):
    """Static definition of one Flow method and its execution roles."""

    start: bool | FlowDefinitionCondition | None = None
    listen: FlowDefinitionCondition | None = None
    router: bool = False
    emit: list[str] | None = None
    human_feedback: FlowHumanFeedbackDefinition | None = None
    persist: FlowPersistenceDefinition | None = None

    @property
    def is_start(self) -> bool:
        """Whether this method is a start method.

        A loaded contract may carry ``start: false`` to mark a non-start
        method explicitly, so falsy values (``False``/``None``/empty string)
        are treated as "not a start method".
        """
        return bool(self.start)


class FlowDefinition(BaseModel):
    """Static, serializable definition of a Flow."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    schema_: str = Field(default="crewai.flow/v1", alias="schema")
    name: str
    description: str | None = None
    state: FlowStateDefinition | None = None
    config: FlowConfigDefinition = Field(default_factory=FlowConfigDefinition)
    persist: FlowPersistenceDefinition | None = None
    conversational: FlowConversationalDefinition | None = None
    methods: dict[str, FlowMethodDefinition] = Field(default_factory=dict)
    diagnostics: list[FlowDefinitionDiagnostic] = Field(default_factory=list)

    def to_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        """Serialize the definition to a JSON/YAML-ready dictionary."""
        return self.model_dump(by_alias=True, exclude_none=exclude_none, mode="json")

    def to_json(self, *, indent: int | None = 2, exclude_none: bool = True) -> str:
        """Serialize the definition to JSON."""
        data = self.to_dict(exclude_none=exclude_none)
        return json.dumps(data, indent=indent)

    def to_yaml(self, *, exclude_none: bool = True) -> str:
        """Serialize the definition to YAML."""
        return yaml.safe_dump(
            self.to_dict(exclude_none=exclude_none),
            sort_keys=False,
            allow_unicode=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowDefinition:
        """Load a definition from a dictionary and attach diagnostics."""
        serialized_diagnostics = _deserialize_diagnostics(data.get("diagnostics", []))
        definition = cls.model_validate(data)
        definition.diagnostics = _merge_diagnostics(
            serialized_diagnostics, definition.validate_contract()
        )
        definition.log_diagnostics()
        return definition

    @classmethod
    def from_json(cls, data: str) -> FlowDefinition:
        """Load a definition from JSON."""
        return cls.from_dict(json.loads(data))

    @classmethod
    def from_yaml(cls, data: str) -> FlowDefinition:
        """Load a definition from YAML."""
        loaded = yaml.safe_load(data) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Flow definition YAML must contain a mapping")
        return cls.from_dict(loaded)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema for the Flow Definition contract."""
        return cls.model_json_schema(by_alias=True)

    def validate_contract(self) -> list[FlowDefinitionDiagnostic]:
        """Validate the static contract without rejecting dynamic routing."""
        diagnostics: list[FlowDefinitionDiagnostic] = []
        for method_name, method in self.methods.items():
            path = f"methods.{method_name}"
            if method.router and not method.is_start and method.listen is None:
                diagnostics.append(
                    FlowDefinitionDiagnostic(
                        code="router_without_trigger",
                        severity="error",
                        path=path,
                        message="router: true requires either start or listen",
                    )
                )
            if method.emit and not method.router:
                diagnostics.append(
                    FlowDefinitionDiagnostic(
                        code="emit_without_router",
                        path=f"{path}.emit",
                        message="emit is only used by routers to declare downstream events",
                    )
                )
            if method.human_feedback:
                human_feedback_config = method.human_feedback
                if human_feedback_config.emit and not human_feedback_config.llm:
                    diagnostics.append(
                        FlowDefinitionDiagnostic(
                            code="human_feedback_llm_required",
                            severity="error",
                            path=f"{path}.human_feedback.llm",
                            message="llm is required when human_feedback.emit is set",
                        )
                    )
                if (
                    human_feedback_config.default_outcome is not None
                    and not human_feedback_config.emit
                ):
                    diagnostics.append(
                        FlowDefinitionDiagnostic(
                            code="human_feedback_default_requires_emit",
                            severity="error",
                            path=f"{path}.human_feedback.default_outcome",
                            message="default_outcome requires human_feedback.emit",
                        )
                    )
                elif (
                    human_feedback_config.default_outcome is not None
                    and human_feedback_config.emit
                ):
                    if (
                        human_feedback_config.default_outcome
                        not in human_feedback_config.emit
                    ):
                        diagnostics.append(
                            FlowDefinitionDiagnostic(
                                code="human_feedback_default_not_in_emit",
                                severity="error",
                                path=f"{path}.human_feedback.default_outcome",
                                message="default_outcome must be one of human_feedback.emit",
                            )
                        )

        return diagnostics

    def with_diagnostics(self) -> FlowDefinition:
        """Attach fresh diagnostics and return this definition."""
        self.diagnostics = self.validate_contract()
        self.log_diagnostics()
        return self

    def log_diagnostics(self) -> None:
        """Emit all attached diagnostics through the flow definition logger."""
        _log_flow_definition_diagnostics(self.name, self.diagnostics)


def _log_flow_definition_diagnostics(
    definition_name: str,
    diagnostics: list[FlowDefinitionDiagnostic],
) -> None:
    for diagnostic in diagnostics:
        level = logging.ERROR if diagnostic.severity == "error" else logging.WARNING
        path = f" at {diagnostic.path}" if diagnostic.path else ""
        logger.log(
            level,
            "Flow definition diagnostic for %s%s [%s]: %s",
            definition_name,
            path,
            diagnostic.code,
            diagnostic.message,
        )


def _deserialize_diagnostics(value: Any) -> list[FlowDefinitionDiagnostic]:
    return [FlowDefinitionDiagnostic.model_validate(item) for item in value or []]


def _merge_diagnostics(
    *diagnostic_groups: list[FlowDefinitionDiagnostic],
) -> list[FlowDefinitionDiagnostic]:
    diagnostics: list[FlowDefinitionDiagnostic] = []
    seen: set[tuple[str, str, str | None, str]] = set()
    for group in diagnostic_groups:
        for diagnostic in group:
            key = (
                diagnostic.code,
                diagnostic.severity,
                diagnostic.path,
                diagnostic.message,
            )
            if key in seen:
                continue
            seen.add(key)
            diagnostics.append(diagnostic)
    return diagnostics
