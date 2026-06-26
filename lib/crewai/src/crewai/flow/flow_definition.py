"""Flow Definition: the serializable, declarative Flow contract.

Defines :class:`FlowDefinition` and its sub-models — a static declarative
representation of a Flow: its methods, trigger conditions,
state, and configuration. It is independent of the Python authoring
layer that may have produced it and of the engine that runs it (see
``runtime``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Annotated, Any, Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    model_validator,
)
import yaml

from crewai.flow.conversational_definition import (
    FlowConversationalDefinition,
    FlowConversationalRouterDefinition,
)
from crewai.flow.expressions import ExpressionData
from crewai.project.crew_definition import AgentDefinition, CrewDefinition


logger = logging.getLogger(__name__)

FlowDefinitionCondition = str | dict[str, Any]
_STEP_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BASE_CEL_ROOTS = frozenset({"outputs", "state"})
_EACH_STEP_CEL_ROOTS = frozenset({"item", "outputs", "state"})

__all__ = [
    "FlowActionDefinition",
    "FlowAgentActionDefinition",
    "FlowAtomicActionDefinition",
    "FlowCodeActionDefinition",
    "FlowConfigDefinition",
    "FlowConversationalDefinition",
    "FlowConversationalRouterDefinition",
    "FlowCrewActionDefinition",
    "FlowDefinition",
    "FlowDefinitionCondition",
    "FlowDictStateDefinition",
    "FlowEachActionDefinition",
    "FlowEachStepDefinition",
    "FlowExpressionActionDefinition",
    "FlowHumanFeedbackDefinition",
    "FlowJsonSchemaStateDefinition",
    "FlowMethodDefinition",
    "FlowPersistenceDefinition",
    "FlowPydanticStateDefinition",
    "FlowScriptActionDefinition",
    "FlowStateDefinition",
    "FlowToolActionDefinition",
    "FlowUnknownStateDefinition",
]


def _object_ref(value: Any) -> str:
    """Format a class or instance as the canonical ``module:qualname`` ref."""
    target = value if isinstance(value, type) else type(value)
    module = getattr(target, "__module__", "")
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", ""))
    return f"{module}:{qualname}" if module and qualname else repr(value)


class FlowDictStateDefinition(BaseModel):
    """Static description of a plain dictionary Flow state contract."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["dict"] = Field(
        default="dict",
        description="Plain dictionary state with optional default values.",
        examples=["dict"],
    )
    default: dict[str, Any] | None = Field(
        default=None,
        description="Default state values applied before kickoff inputs.",
        examples=[{"topic": "AI agents", "limit": 3}],
    )


class FlowPydanticStateDefinition(BaseModel):
    """Static description of an importable Pydantic Flow state contract."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pydantic"] = Field(
        default="pydantic",
        description="Importable Pydantic model used as the Flow state type.",
        examples=["pydantic"],
    )
    ref: str | None = Field(
        default=None,
        description="Import reference for the state model, formatted as module:qualname.",
        examples=["my_project.flows:ResearchState"],
    )
    json_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Fallback JSON Schema used when the Pydantic state ref is unavailable."
        ),
        examples=[
            {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            }
        ],
    )
    default: dict[str, Any] | None = Field(
        default=None,
        description="Default state values applied before kickoff inputs.",
        examples=[{"topic": "AI agents", "limit": 3}],
    )


class FlowJsonSchemaStateDefinition(BaseModel):
    """Static description of an inline JSON Schema Flow state contract."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"] = Field(
        default="json_schema",
        description="Inline JSON Schema used as the Flow state contract.",
        examples=["json_schema"],
    )
    json_schema: dict[str, Any] = Field(
        description="JSON Schema used to validate and document flow state.",
        examples=[
            {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            }
        ],
    )
    default: dict[str, Any] | None = Field(
        default=None,
        description="Default state values applied before kickoff inputs.",
        examples=[{"topic": "AI agents", "limit": 3}],
    )


class FlowUnknownStateDefinition(BaseModel):
    """Static description of a state contract that could not be serialized."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["unknown"] = Field(
        default="unknown",
        description="Unknown state representation; runtime falls back to dictionary state.",
        examples=["unknown"],
    )
    ref: str | None = Field(
        default=None,
        description="Best-effort import reference for the unknown state type.",
        examples=["my_project.flows:CustomState"],
    )
    default: dict[str, Any] | None = Field(
        default=None,
        description="Default state values applied before kickoff inputs.",
        examples=[{"topic": "AI agents", "limit": 3}],
    )


FlowStateDefinition: TypeAlias = Annotated[
    FlowDictStateDefinition
    | FlowPydanticStateDefinition
    | FlowJsonSchemaStateDefinition
    | FlowUnknownStateDefinition,
    Field(discriminator="type"),
]


class FlowConfigDefinition(BaseModel):
    """Serializable Flow-level configuration."""

    tracing: bool | None = Field(
        default=None,
        description="Override for flow tracing; when omitted, runtime defaults apply.",
        examples=[True],
    )
    stream: bool = Field(
        default=False,
        description="Whether the flow should emit streaming events when supported.",
        examples=[True],
    )
    memory: dict[str, Any] | None = Field(
        default=None,
        description="Serializable memory configuration passed to flow execution.",
        examples=[{"enabled": True}],
    )
    input_provider: str | None = Field(
        default=None,
        description="Import reference or provider key used to supply flow inputs.",
        examples=["my_project.inputs:load_inputs"],
    )
    suppress_flow_events: bool = Field(
        default=False,
        description="Disable flow event emission for this definition.",
        examples=[False],
    )
    max_method_calls: int = Field(
        default=100,
        description="Maximum number of method executions allowed during one kickoff.",
        examples=[20],
    )
    defer_trace_finalization: bool = Field(
        default=False,
        description="Defer trace finalization so callers can complete tracing later.",
        examples=[False],
    )
    checkpoint: bool | dict[str, Any] | None = Field(
        default=None,
        description="Checkpointing configuration, or true to use default checkpointing.",
        examples=[True, {"enabled": True}],
    )


class FlowPersistenceDefinition(BaseModel):
    """Static persistence configuration.

    ``persistence`` may hold a live backend when the definition is built from
    a decorated class — the engine then persists through the exact instance
    the user configured; the declarative projection degrades it to its
    serialized config.
    """

    enabled: bool = Field(
        default=False,
        description="Whether persistence is enabled for this flow or method.",
        examples=[True],
    )
    verbose: bool = Field(
        default=False,
        description="Whether persistence should emit verbose diagnostic output.",
        examples=[False],
    )
    persistence: Any = Field(
        default=None,
        description="Persistence backend configuration or import reference.",
        examples=[{"ref": "my_project.persistence:FlowStore"}],
    )

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
    is built from a decorated class; the declarative projection degrades them to
    a serialized config (``llm``) or a ``module:qualname`` ref (``provider``).
    """

    message: str = Field(
        description="Prompt shown to the human reviewer when feedback is requested.",
        examples=["Review the research summary before publishing."],
    )
    emit: list[str] | None = Field(
        default=None,
        description=(
            "Allowed feedback outcomes. When set, the method routes like a router "
            "using the selected outcome."
        ),
        examples=[["approved", "revise"]],
    )
    llm: Any = Field(
        default="gpt-4o-mini",
        description="LLM configuration used to assist or process human feedback.",
        examples=["gpt-4o-mini"],
    )
    default_outcome: str | None = Field(
        default=None,
        description="Outcome to use when feedback cannot be collected.",
        examples=["revise"],
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Serializable metadata attached to the feedback request.",
        examples=[{"team": "research"}],
    )
    provider: Any = Field(
        default=None,
        description="Feedback provider configuration or import reference.",
        examples=["my_project.feedback:provider"],
    )
    learn: bool = Field(
        default=False,
        description="Whether feedback should be recorded for later learning workflows.",
        examples=[True],
    )
    learn_source: str = Field(
        default="hitl",
        description="Source label attached to learned feedback records.",
        examples=["hitl"],
    )
    learn_strict: bool = Field(
        default=False,
        description="Whether learning should enforce strict validation of feedback data.",
        examples=[False],
    )

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

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    call: Literal["code"] = Field(
        default="code",
        description="Action discriminator. Use code to call importable Python.",
        examples=["code"],
    )
    ref: str = Field(
        description="Import reference for the callable, formatted as module:qualname.",
        examples=["my_project.flows:normalize_topic"],
    )
    with_: dict[str, ExpressionData] | None = Field(
        default=None,
        alias="with",
        description=(
            "Keyword arguments passed to the callable. String values are evaluated "
            "as CEL only when the trimmed value starts with ${ and ends with }; "
            "all other values are literal."
        ),
        examples=[{"topic": "${state.topic}"}],
    )


class FlowToolActionDefinition(BaseModel):
    """A Flow method action that invokes a CrewAI tool."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    call: Literal["tool"] = Field(
        description="Action discriminator. Use tool to instantiate and run a CrewAI tool.",
        examples=["tool"],
    )
    ref: str = Field(
        description="Import reference for a BaseTool class, formatted as module:qualname.",
        examples=["my_project.tools:SearchTool"],
    )
    with_: dict[str, ExpressionData] | None = Field(
        default=None,
        alias="with",
        description=(
            "Tool input arguments. String values are evaluated as CEL only when "
            "the trimmed value starts with ${ and ends with }; all other values "
            "are literal."
        ),
        examples=[{"query": "${outputs.normalize_topic}", "limit": 5}],
    )


class FlowCrewActionDefinition(BaseModel):
    """A Flow method action that builds and kicks off a CrewAI crew."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    call: Literal["crew"] = Field(
        description=(
            "Action discriminator. Use crew to run an inline or referenced Crew "
            "definition."
        ),
        examples=["crew"],
    )
    from_declaration: str | None = Field(
        default=None,
        description="Path to a JSON/JSONC Crew declaration file or folder.",
        examples=["crews/research_crew"],
    )
    with_: CrewDefinition | None = Field(
        default=None,
        alias="with",
        description="Inline Crew definition to load and execute for this action.",
        examples=[
            {
                "name": "inline_research",
                "agents": {
                    "researcher": {
                        "role": "Researcher",
                        "goal": "Research {topic}",
                        "backstory": "Knows the domain.",
                    }
                },
                "tasks": [
                    {
                        "name": "research_task",
                        "description": "Research {topic}",
                        "expected_output": "Findings about {topic}",
                        "agent": "researcher",
                    }
                ],
            }
        ],
    )
    inputs: dict[str, ExpressionData] | None = Field(
        default=None,
        description=(
            "Input overrides passed to the Crew. String values are evaluated as CEL "
            "only when the trimmed value starts with ${ and ends with }; all other "
            "values are literal."
        ),
        examples=[{"topic": "${state.topic}"}],
    )

    @model_validator(mode="after")
    def _validate_crew_source(self) -> FlowCrewActionDefinition:
        if bool(self.from_declaration) == (self.with_ is not None):
            raise ValueError(
                "crew action requires exactly one of from_declaration or with"
            )
        return self


class FlowAgentActionDefinition(BaseModel):
    """A Flow method action that builds and kicks off a CrewAI agent."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    call: Literal["agent"] = Field(
        description="Action discriminator. Use agent to run an inline Agent definition.",
        examples=["agent"],
    )
    with_: AgentDefinition = Field(
        alias="with",
        description="Inline Agent definition to load and execute for this action.",
        examples=[
            {
                "role": "Analyst",
                "goal": "Answer user questions",
                "backstory": "Precise and concise.",
                "settings": {"llm": "openai/gpt-4o-mini"},
                "input": "${state.question}",
            }
        ],
    )


class FlowExpressionActionDefinition(BaseModel):
    """A Flow method action that evaluates a CEL expression."""

    model_config = ConfigDict(extra="forbid")

    call: Literal["expression"] = Field(
        description="Action discriminator. Use expression to evaluate a CEL expression.",
        examples=["expression"],
    )
    expr: str = Field(
        description="CEL expression evaluated against state, outputs, and local context.",
        examples=["state.topic", "outputs.normalize_topic"],
    )


class FlowScriptActionDefinition(BaseModel):
    """A Flow method action that executes trusted inline Python."""

    model_config = ConfigDict(extra="forbid")

    call: Literal["script"] = Field(
        description="Action discriminator. Use script to execute trusted inline Python.",
        examples=["script"],
    )
    code: str = Field(
        description=(
            "Trusted Python source executed as a generated function. Runtime values are "
            "passed as state, outputs, input, and item; they are not interpolated into "
            "the source. This is not sandboxed."
        ),
        examples=[
            "state['normalized_topic'] = input.strip()\n"
            "return state['normalized_topic']"
        ],
    )
    language: Literal["python"] = Field(
        default="python",
        description="Script language. Only python is currently supported.",
        examples=["python"],
    )


FlowAtomicActionDefinition: TypeAlias = Annotated[
    FlowCodeActionDefinition
    | FlowToolActionDefinition
    | FlowCrewActionDefinition
    | FlowAgentActionDefinition
    | FlowExpressionActionDefinition
    | FlowScriptActionDefinition,
    Field(discriminator="call"),
]


class FlowEachStepDefinition(BaseModel):
    """One named step inside an ``each`` composite action."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    name: str = Field(
        description="Step name used to reference this step's output.",
        examples=["clean"],
    )
    if_: str | None = Field(
        default=None,
        alias="if",
        description=(
            "Optional CEL expression evaluated against state, outputs, and local "
            "context. When present, the step runs only if the expression evaluates "
            "to true."
        ),
        examples=["item.kind == 'invoice'"],
    )
    action: FlowAtomicActionDefinition = Field(
        description="Atomic action to run for this step.",
        examples=[{"call": "script", "code": "return item.strip()"}],
    )

    @model_validator(mode="after")
    def _validate_step_name(self) -> FlowEachStepDefinition:
        _validate_step_name(self.name, field="each.do step names")
        return self


class FlowEachActionDefinition(BaseModel):
    """A composite action that runs a sequential mini-pipeline for each item."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    call: Literal["each"] = Field(
        description=(
            "Action discriminator. Use each to run a sequence of actions for every "
            "item in an input list."
        ),
        examples=["each"],
    )
    in_: str = Field(
        alias="in",
        description="CEL expression that must evaluate to the list to iterate.",
        examples=["state.rows"],
    )
    do: list[FlowEachStepDefinition] = Field(
        description=(
            "Ordered steps to run for each item. Each step has a name, optional "
            "if expression, and atomic action."
        ),
        examples=[
            [
                {
                    "name": "clean",
                    "action": {"call": "script", "code": "return item.strip()"},
                },
                {
                    "name": "tag",
                    "if": "outputs.clean != ''",
                    "action": {"call": "expression", "expr": "outputs.clean"},
                },
            ]
        ],
    )

    @model_validator(mode="after")
    def _validate_step_list(self) -> FlowEachActionDefinition:
        if not self.do:
            raise ValueError("each.do must contain at least one step")

        _validate_step_list(self.do, field="each.do")
        return self


FlowActionDefinition: TypeAlias = (
    FlowCodeActionDefinition
    | FlowToolActionDefinition
    | FlowCrewActionDefinition
    | FlowAgentActionDefinition
    | FlowExpressionActionDefinition
    | FlowScriptActionDefinition
    | FlowEachActionDefinition
)


class FlowMethodDefinition(BaseModel):
    """Static definition of one Flow method and its execution roles."""

    description: str | None = Field(
        default=None,
        description="Human-readable summary of what this method does.",
        examples=["Normalize the incoming topic."],
    )
    do: FlowActionDefinition = Field(
        description="Action executed when this method runs.",
        examples=[{"call": "script", "code": "return input.strip()"}],
    )
    start: bool | FlowDefinitionCondition | None = Field(
        default=None,
        description=(
            "Marks a start method. True starts unconditionally; a condition starts "
            "when the kickoff inputs or events satisfy it."
        ),
        examples=[True],
    )
    listen: FlowDefinitionCondition | None = Field(
        default=None,
        description="Trigger condition that runs this method after upstream events.",
        examples=["seed", {"or": ["approved", "revise"]}],
    )
    router: bool = Field(
        default=False,
        description="Whether the method output should be treated as the next event name.",
        examples=[True],
    )
    emit: list[str] | None = Field(
        default=None,
        description="Declared router events this method may emit.",
        examples=[["approved", "revise"]],
    )
    human_feedback: FlowHumanFeedbackDefinition | None = Field(
        default=None,
        description="Optional human feedback step applied after the method action.",
        examples=[{"message": "Review the research summary before publishing."}],
    )
    persist: FlowPersistenceDefinition | None = Field(
        default=None,
        description="Method-level persistence override.",
        examples=[{"enabled": True}],
    )

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

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    _source_path: Path | None = PrivateAttr(default=None)

    schema_: Literal["crewai.flow/v1"] = Field(
        default="crewai.flow/v1",
        alias="schema",
        description="Declarative Flow schema identifier and version.",
        examples=["crewai.flow/v1"],
    )
    name: str = Field(
        description="Unique flow name used in logs, events, and traces.",
        examples=["ResearchFlow"],
    )
    description: str | None = Field(
        default=None,
        description="Human-readable summary of the flow.",
        examples=["Normalize a topic and prepare it for research."],
    )
    state: FlowStateDefinition | None = Field(
        default=None,
        description="State contract for kickoff inputs and runtime state.",
        examples=[{"type": "dict", "default": {"topic": "AI agents"}}],
    )
    config: FlowConfigDefinition = Field(
        default_factory=FlowConfigDefinition,
        description="Serializable flow-level runtime configuration.",
        examples=[{"stream": True, "max_method_calls": 20}],
    )
    persist: FlowPersistenceDefinition | None = Field(
        default=None,
        description="Flow-level persistence configuration.",
        examples=[{"enabled": True}],
    )
    conversational: FlowConversationalDefinition | None = Field(
        default=None,
        description="Conversational flow configuration, when the flow supports chat.",
    )
    methods: dict[str, FlowMethodDefinition] = Field(
        default_factory=dict,
        description="Mapping of method names to method definitions.",
        examples=[
            {
                "seed": {
                    "start": True,
                    "do": {"call": "expression", "expr": "state.topic"},
                }
            }
        ],
    )

    @model_validator(mode="after")
    def _validate_method_names(self) -> FlowDefinition:
        for method_name in self.methods:
            _validate_step_name(method_name, field="Flow method names")
        return self

    @model_validator(mode="after")
    def _validate_cel_expressions(self) -> FlowDefinition:
        for method_name, method in self.methods.items():
            _validate_action_cel(
                method.do,
                path=f"methods.{method_name}.do",
                allowed_roots=_BASE_CEL_ROOTS,
            )
        return self

    def to_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        """Serialize the definition to a declaration-ready dictionary."""
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

    @property
    def source_path(self) -> Path | None:
        """Original definition file path, when loaded from a file."""
        return self._source_path

    @property
    def source_dir(self) -> Path | None:
        """Directory used to resolve relative paths in the definition."""
        if self._source_path is None:
            return None
        return self._source_path.parent

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, source_path: Path | None = None
    ) -> FlowDefinition:
        """Load a definition from a dictionary."""
        definition = cls.model_validate(data)
        if source_path is not None:
            definition._source_path = source_path.expanduser().resolve()
        log_flow_definition_issues(definition)
        return definition

    @classmethod
    def from_declaration(
        cls,
        *,
        contents: FlowDefinition | str | dict[str, Any] | None = None,
        path: Path | str | None = None,
    ) -> FlowDefinition:
        """Load a declarative flow from contents or a file path."""
        if isinstance(contents, cls):
            return contents

        source_path: Path | None = None
        if contents is None:
            if path is None:
                raise ValueError("Provide contents or path")
            source_path = Path(path)
            contents = source_path.expanduser().read_text(encoding="utf-8")

        if isinstance(contents, dict):
            return cls.from_dict(contents)

        if not isinstance(contents, str):
            raise TypeError("Flow declaration contents must be a string or dictionary")

        if not contents.strip():
            if source_path is not None:
                raise ValueError(f"Flow declaration file is empty: {source_path}")
            raise ValueError("Flow declaration contents are empty")

        loaded = yaml.safe_load(contents)
        if not isinstance(loaded, dict):
            raise ValueError("Flow declaration must contain a mapping")
        return cls.from_dict(loaded, source_path=source_path)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema for the declarative Flow contract."""
        return cls.model_json_schema(by_alias=True)


def _validate_step_name(name: str, *, field: str) -> None:
    if not isinstance(name, str) or not _STEP_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"{field} must match {_STEP_NAME_PATTERN.pattern}")


def _validate_step_list(steps: list[FlowEachStepDefinition], *, field: str) -> None:
    seen: set[str] = set()
    for step in steps:
        name = step.name
        if name in seen:
            raise ValueError(f"{field} step names must be unique: {name!r}")
        seen.add(name)


def _validate_action_cel(
    action: FlowActionDefinition,
    *,
    path: str,
    allowed_roots: frozenset[str],
) -> None:
    from crewai.flow.expressions import Expression

    if isinstance(action, FlowExpressionActionDefinition):
        Expression(action.expr).validate_expression(
            allowed_roots=allowed_roots, source=f"{path}.expr"
        )
        return

    if isinstance(action, (FlowCodeActionDefinition, FlowToolActionDefinition)):
        if action.with_ is not None:
            Expression(action.with_).validate_template(
                allowed_roots=allowed_roots, source=f"{path}.with"
            )
        return

    if isinstance(action, FlowCrewActionDefinition):
        if action.with_ is not None:
            Expression(cast(ExpressionData, action.with_.inputs)).validate_template(
                allowed_roots=allowed_roots,
                source=f"{path}.with.inputs",
            )
        if action.inputs is not None:
            Expression(cast(ExpressionData, action.inputs)).validate_template(
                allowed_roots=allowed_roots,
                source=f"{path}.inputs",
            )
        return

    if isinstance(action, FlowAgentActionDefinition):
        Expression(cast(ExpressionData, action.with_.input)).validate_template(
            allowed_roots=allowed_roots,
            source=f"{path}.with.input",
        )
        return

    if isinstance(action, FlowEachActionDefinition):
        Expression(action.in_).validate_expression(
            allowed_roots=_BASE_CEL_ROOTS,
            source=f"{path}.in",
        )
        for index, step in enumerate(action.do):
            step_path = f"{path}.do[{index}]"
            if step.if_ is not None:
                Expression(step.if_).validate_expression(
                    allowed_roots=_EACH_STEP_CEL_ROOTS,
                    source=f"{step_path}.if",
                )
            _validate_action_cel(
                step.action,
                path=f"{step_path}.action",
                allowed_roots=_EACH_STEP_CEL_ROOTS,
            )
        return

    if isinstance(action, FlowScriptActionDefinition):
        return

    raise TypeError(
        f"no CEL validation defined for action type {type(action).__name__} at "
        f"{path}; add a branch to _validate_action_cel for it."
    )


def log_flow_definition_issues(definition: FlowDefinition) -> None:
    for method_name, method in definition.methods.items():
        path = f"methods.{method_name}"
        if method.emit and not method.router:
            _log_flow_definition_issue(
                definition.name,
                code="emit_without_router",
                path=f"{path}.emit",
                message="emit is only used by routers to declare downstream events",
            )
        if method.human_feedback:
            human_feedback_config = method.human_feedback
            if human_feedback_config.emit and not human_feedback_config.llm:
                _log_flow_definition_issue(
                    definition.name,
                    code="human_feedback_llm_required",
                    severity="error",
                    path=f"{path}.human_feedback.llm",
                    message="llm is required when human_feedback.emit is set",
                )
            if (
                human_feedback_config.default_outcome is not None
                and not human_feedback_config.emit
            ):
                _log_flow_definition_issue(
                    definition.name,
                    code="human_feedback_default_requires_emit",
                    severity="error",
                    path=f"{path}.human_feedback.default_outcome",
                    message="default_outcome requires human_feedback.emit",
                )
            elif (
                human_feedback_config.default_outcome is not None
                and human_feedback_config.emit
                and human_feedback_config.default_outcome
                not in human_feedback_config.emit
            ):
                _log_flow_definition_issue(
                    definition.name,
                    code="human_feedback_default_not_in_emit",
                    severity="error",
                    path=f"{path}.human_feedback.default_outcome",
                    message="default_outcome must be one of human_feedback.emit",
                )


def _log_flow_definition_issue(
    definition_name: str,
    *,
    code: str,
    message: str,
    severity: Literal["warning", "error"] = "warning",
    path: str | None = None,
) -> None:
    level = logging.ERROR if severity == "error" else logging.WARNING
    location = f" at {path}" if path else ""
    logger.log(
        level,
        "Flow definition issue for %s%s [%s]: %s",
        definition_name,
        location,
        code,
        message,
    )
