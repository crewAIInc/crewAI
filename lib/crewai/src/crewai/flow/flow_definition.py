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
import re
from typing import Any, Literal as TypingLiteral

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_serializer,
    model_validator,
)
import yaml

from crewai.flow.conversational_definition import (
    FlowConversationalDefinition,
    FlowConversationalRouterDefinition,
)
from crewai.project.crew_definition import CrewDefinition


logger = logging.getLogger(__name__)

FlowDefinitionCondition = str | dict[str, Any]
_STEP_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

__all__ = [
    "FlowActionDefinition",
    "FlowCodeActionDefinition",
    "FlowConfigDefinition",
    "FlowConversationalDefinition",
    "FlowConversationalRouterDefinition",
    "FlowCrewActionDefinition",
    "FlowDefinition",
    "FlowDefinitionCondition",
    "FlowDefinitionDiagnostic",
    "FlowEachActionDefinition",
    "FlowEachInnerActionDefinition",
    "FlowExpressionActionDefinition",
    "FlowHumanFeedbackDefinition",
    "FlowMethodDefinition",
    "FlowPersistenceDefinition",
    "FlowStateDefinition",
    "FlowToolActionDefinition",
]


def _object_ref(value: Any) -> str:
    """Format a class or instance as the canonical ``module:qualname`` ref."""
    target = value if isinstance(value, type) else type(value)
    module = getattr(target, "__module__", "")
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", ""))
    return f"{module}:{qualname}" if module and qualname else repr(value)


class FlowDefinitionDiagnostic(BaseModel):
    """A non-fatal Flow Definition build or validation diagnostic."""

    code: str
    message: str
    severity: TypingLiteral["warning", "error"] = "warning"
    path: str | None = None


class FlowStateDefinition(BaseModel):
    """Static description of a Flow state contract."""

    type: TypingLiteral["dict", "pydantic", "json_schema", "unknown"] = "dict"
    ref: str | None = None
    json_schema: dict[str, Any] | None = None
    default: dict[str, Any] | None = None


class FlowConfigDefinition(BaseModel):
    """Serializable Flow-level configuration."""

    tracing: bool | None = None
    stream: bool = False
    memory: dict[str, Any] | None = None
    input_provider: str | None = None
    suppress_flow_events: bool = False
    max_method_calls: int = 100
    defer_trace_finalization: bool = False
    checkpoint: bool | dict[str, Any] | None = None


class FlowPersistenceDefinition(BaseModel):
    """Static persistence configuration.

    ``persistence`` may hold a live backend when the definition is built from
    a decorated class — the engine then persists through the exact instance
    the user configured; the JSON/YAML projection degrades it to its
    serialized config.
    """

    enabled: bool = False
    verbose: bool = False
    persistence: Any = None

    @field_serializer("persistence", when_used="json")
    def _serialize_persistence(self, value: Any) -> Any:
        if value is None or isinstance(value, dict):
            return value
        if isinstance(value, BaseModel):
            try:
                return value.model_dump(mode="json")
            except Exception:
                logger.warning(
                    "Persistence backend %s is not fully serializable; "
                    "preserved import reference only.",
                    _object_ref(value),
                )
        return {"ref": _object_ref(value)}


class FlowHumanFeedbackDefinition(BaseModel):
    """Static human feedback configuration.

    ``llm`` and ``provider`` may hold live Python objects when the definition
    is built from a decorated class; the JSON/YAML projection degrades them to
    a serialized config (``llm``) or a ``module:qualname`` ref (``provider``).
    """

    message: str
    emit: list[str] | None = None
    llm: Any = "gpt-4o-mini"
    default_outcome: str | None = None
    metadata: dict[str, Any] | None = None
    provider: Any = None
    learn: bool = False
    learn_source: str = "hitl"
    learn_strict: bool = False

    @field_serializer("llm", when_used="json")
    def _serialize_llm(self, value: Any) -> dict[str, Any] | str | None:
        if value is None or isinstance(value, (str, dict)):
            return value
        from crewai.flow.human_feedback import _serialize_llm_for_context

        return _serialize_llm_for_context(value)

    @field_serializer("provider", when_used="json")
    def _serialize_provider(self, value: Any) -> str | None:
        if value is None or isinstance(value, str):
            return value
        return _object_ref(value)


class FlowCodeActionDefinition(BaseModel):
    """A Flow method action that executes importable Python code."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    call: TypingLiteral["code"] = "code"
    ref: str
    with_: dict[str, Any] | None = Field(default=None, alias="with")


class FlowToolActionDefinition(BaseModel):
    """A Flow method action that invokes a CrewAI tool."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    call: TypingLiteral["tool"]
    ref: str
    with_: dict[str, Any] | None = Field(default=None, alias="with")


class FlowCrewActionDefinition(BaseModel):
    """A Flow method action that builds and kicks off a CrewAI crew."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    call: TypingLiteral["crew"]
    with_: CrewDefinition = Field(alias="with")


class FlowExpressionActionDefinition(BaseModel):
    """A Flow method action that evaluates a CEL expression."""

    model_config = ConfigDict(extra="forbid")

    call: TypingLiteral["expression"]
    expr: str


FlowInnerActionDefinition = (
    FlowCodeActionDefinition
    | FlowToolActionDefinition
    | FlowCrewActionDefinition
    | FlowExpressionActionDefinition
)


class FlowEachInnerActionDefinition(RootModel[dict[str, FlowInnerActionDefinition]]):
    """One named action inside an ``each`` composite action."""

    @model_validator(mode="after")
    def _validate_action_mapping(self) -> FlowEachInnerActionDefinition:
        if len(self.root) != 1:
            raise ValueError("each.do entries must be one-key mappings")
        _validate_step_name(self.name, field="each.do action names")
        return self

    @property
    def name(self) -> str:
        return next(iter(self.root))

    @property
    def action(self) -> FlowInnerActionDefinition:
        return next(iter(self.root.values()))


class FlowEachActionDefinition(BaseModel):
    """A composite action that runs a sequential mini-pipeline for each item."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    call: TypingLiteral["each"]
    in_: str = Field(alias="in")
    do: list[FlowEachInnerActionDefinition]

    @model_validator(mode="after")
    def _validate_inner_action_list(self) -> FlowEachActionDefinition:
        if not self.do:
            raise ValueError("each.do must contain at least one action")

        seen: set[str] = set()
        for inner_action in self.do:
            name = inner_action.name
            if name in seen:
                raise ValueError(f"each.do action names must be unique: {name!r}")
            seen.add(name)

        return self


FlowActionDefinition = (
    FlowCodeActionDefinition
    | FlowToolActionDefinition
    | FlowCrewActionDefinition
    | FlowExpressionActionDefinition
    | FlowEachActionDefinition
)


class FlowMethodDefinition(BaseModel):
    """Static definition of one Flow method and its execution roles."""

    description: str | None = None
    do: FlowActionDefinition
    start: bool | FlowDefinitionCondition | None = None
    listen: FlowDefinitionCondition | None = None
    router: bool = False
    emit: list[str] | None = None
    human_feedback: FlowHumanFeedbackDefinition | None = None
    persist: FlowPersistenceDefinition | None = None

    @model_validator(mode="after")
    def _canonicalize_human_feedback_routing(self) -> FlowMethodDefinition:
        # Canonical shape: a method whose human_feedback declares emit
        # outcomes routes like a router, regardless of how the definition
        # was authored.
        if self.human_feedback is not None and self.human_feedback.emit:
            self.router = True
            self.emit = None
        return self

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

    schema_: TypingLiteral["crewai.flow/v1"] = Field(
        default="crewai.flow/v1", alias="schema"
    )
    name: str
    description: str | None = None
    state: FlowStateDefinition | None = None
    config: FlowConfigDefinition = Field(default_factory=FlowConfigDefinition)
    persist: FlowPersistenceDefinition | None = None
    conversational: FlowConversationalDefinition | None = None
    methods: dict[str, FlowMethodDefinition] = Field(default_factory=dict)
    diagnostics: list[FlowDefinitionDiagnostic] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_method_names(self) -> FlowDefinition:
        for method_name in self.methods:
            _validate_step_name(method_name, field="Flow method names")
        return self

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


def _validate_step_name(name: str, *, field: str) -> None:
    if not isinstance(name, str) or not _STEP_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"{field} must match {_STEP_NAME_PATTERN.pattern}")


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
