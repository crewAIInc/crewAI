from __future__ import annotations

import json
import logging
from typing import Any, ParamSpec, TypeVar

from pydantic import BaseModel
from typing_extensions import TypeIs

from crewai.flow.flow_definition import (
    FlowConfigDefinition,
    FlowDefinition,
    FlowDefinitionDiagnostic,
    FlowHumanFeedbackDefinition,
    FlowMethodDefinition,
    FlowPersistenceDefinition,
    FlowStateDefinition,
)
from crewai.flow.flow_wrappers import (
    FlowMethod,
)


P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)

_FLOW_METHOD_DEFINITION_ATTR = "__flow_method_definition__"


def is_flow_method(obj: Any) -> TypeIs[FlowMethod[Any, Any]]:
    """Check if the object carries Flow method wrapper metadata."""
    return hasattr(obj, "__is_flow_method__") or hasattr(
        obj, _FLOW_METHOD_DEFINITION_ATTR
    )


def _should_include_flow_method(flow_class: type, method: Any) -> bool:
    if getattr(method, "__conversational_only__", False):
        return bool(getattr(flow_class, "conversational", False))
    return True


def _set_flow_method_definition(
    wrapper: FlowMethod[P, R],
    definition: FlowMethodDefinition,
) -> None:
    setattr(wrapper, _FLOW_METHOD_DEFINITION_ATTR, definition)


def _get_flow_method_definition(method: Any) -> FlowMethodDefinition | None:
    definition = getattr(method, _FLOW_METHOD_DEFINITION_ATTR, None)
    if isinstance(definition, FlowMethodDefinition):
        return definition
    if definition is not None:
        return FlowMethodDefinition.model_validate(definition)
    return None


def _object_ref(value: Any) -> str:
    target = value if isinstance(value, type) else type(value)
    module = getattr(target, "__module__", "")
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", ""))
    return f"{module}:{qualname}" if module and qualname else repr(value)


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _serialize_static_value(
    value: Any,
    diagnostics: list[FlowDefinitionDiagnostic],
    path: str,
) -> Any:
    if value is None or _is_json_serializable(value):
        return value

    to_config = getattr(value, "to_config_dict", None)
    if callable(to_config):
        try:
            config = to_config()
            if _is_json_serializable(config):
                return config
        except Exception:
            logger.debug(
                "Failed to serialize %s via to_config_dict().",
                path,
                exc_info=True,
            )

    if isinstance(value, BaseModel):
        try:
            data = value.model_dump(mode="json")
            if _is_json_serializable(data):
                return data
        except Exception:
            logger.debug(
                "Failed to serialize %s via Pydantic model_dump().",
                path,
                exc_info=True,
            )

    ref = _object_ref(value)
    diagnostics.append(
        FlowDefinitionDiagnostic(
            code="non_serializable_value",
            path=path,
            message=f"value is not fully serializable; preserved import reference {ref}",
        )
    )
    return {"ref": ref}


def _state_ref(value: Any) -> str | None:
    if value is None:
        return None
    target = value if isinstance(value, type) else type(value)
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if module and qualname:
        return f"{module}:{qualname}"
    return None


def _build_state_definition(
    flow_class: type,
    diagnostics: list[FlowDefinitionDiagnostic],
) -> FlowStateDefinition | None:
    from pydantic import BaseModel as PydanticBaseModel

    state_value = getattr(flow_class, "_initial_state_t", None)
    initial_state = getattr(flow_class, "initial_state", None)
    if initial_state is not None:
        state_value = initial_state

    if state_value is None:
        return None
    if state_value is dict or isinstance(state_value, dict):
        default = None
        if isinstance(state_value, dict):
            default = _serialize_static_value(state_value, diagnostics, "state.default")
        return FlowStateDefinition(type="dict", default=default)
    if isinstance(state_value, type) and issubclass(state_value, PydanticBaseModel):
        return FlowStateDefinition(type="pydantic", ref=_state_ref(state_value))
    if isinstance(state_value, PydanticBaseModel):
        return FlowStateDefinition(
            type="pydantic",
            ref=_state_ref(state_value),
            default=_serialize_static_value(state_value, diagnostics, "state.default"),
        )
    diagnostics.append(
        FlowDefinitionDiagnostic(
            code="unknown_state_type",
            path="state",
            message=f"could not serialize state type {_object_ref(state_value)}",
        )
    )
    return FlowStateDefinition(type="unknown", ref=_state_ref(state_value))


def _build_config_definition(
    flow_class: type,
    diagnostics: list[FlowDefinitionDiagnostic],
) -> FlowConfigDefinition:
    config_field_names = set(FlowConfigDefinition.model_fields)
    field_defaults = {
        name: field.default
        for name, field in getattr(flow_class, "model_fields", {}).items()
        if name in config_field_names
    }
    values: dict[str, Any] = {}
    for field_name, default in field_defaults.items():
        value = getattr(flow_class, field_name, default)
        values[field_name] = _serialize_static_value(
            value, diagnostics, f"config.{field_name}"
        )
    return FlowConfigDefinition(**values)


def _build_human_feedback_definition(
    method: Any,
    diagnostics: list[FlowDefinitionDiagnostic],
    path: str,
) -> FlowHumanFeedbackDefinition | None:
    config = getattr(method, "__human_feedback_config__", None)
    if config is None:
        return None
    emit = getattr(config, "emit", None)
    return FlowHumanFeedbackDefinition(
        message=str(config.message),
        emit=[str(value) for value in emit] if emit is not None else None,
        llm=_serialize_static_value(
            getattr(config, "llm", None), diagnostics, f"{path}.llm"
        ),
        default_outcome=getattr(config, "default_outcome", None),
        metadata=_serialize_static_value(
            getattr(config, "metadata", None), diagnostics, f"{path}.metadata"
        ),
        provider=_serialize_static_value(
            getattr(config, "provider", None), diagnostics, f"{path}.provider"
        ),
        learn=bool(getattr(config, "learn", False)),
        learn_source=str(getattr(config, "learn_source", "hitl")),
        learn_strict=bool(getattr(config, "learn_strict", False)),
    )


def _build_persistence_definition(
    value: Any,
    diagnostics: list[FlowDefinitionDiagnostic],
    path: str,
) -> FlowPersistenceDefinition | None:
    config = getattr(value, "__flow_persistence_config__", None)
    if config is None:
        return None
    persistence = getattr(config, "persistence", None)
    verbose = bool(getattr(config, "verbose", False))
    return FlowPersistenceDefinition(
        enabled=True,
        verbose=verbose,
        persistence=_serialize_static_value(
            persistence, diagnostics, f"{path}.persistence"
        ),
    )


def _build_method_definition(
    method: Any,
    diagnostics: list[FlowDefinitionDiagnostic],
    path: str,
) -> FlowMethodDefinition:
    fragment = _get_flow_method_definition(method)
    if fragment is None:
        method_definition = FlowMethodDefinition()
    else:
        method_definition = fragment.model_copy(deep=True)

    human_feedback = _build_human_feedback_definition(
        method, diagnostics, f"{path}.human_feedback"
    )
    if human_feedback is not None:
        method_definition.human_feedback = human_feedback
        if human_feedback.emit:
            method_definition.router = True
            method_definition.emit = None

    method_definition.persist = _build_persistence_definition(
        method, diagnostics, f"{path}.persist"
    )

    return method_definition


def _iter_flow_methods(flow_class: type) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for attr_name in flow_class.__dict__:
        if attr_name.startswith("_"):
            continue
        try:
            attr_value = getattr(flow_class, attr_name)
        except AttributeError:
            continue
        if is_flow_method(attr_value) and _should_include_flow_method(
            flow_class, attr_value
        ):
            methods[attr_name] = attr_value

    # A wrapped method whose name collides with a base Flow model field
    # (e.g. ``checkpoint``) is absorbed by Pydantic as a field; the underlying
    # function is preserved as the field default. Recover those so the
    # definition still reflects every method once the class is built.
    for field_name, field in getattr(flow_class, "model_fields", {}).items():
        if field_name in methods or field_name.startswith("_"):
            continue
        default = getattr(field, "default", None)
        if is_flow_method(default) and _should_include_flow_method(flow_class, default):
            methods[field_name] = default
    return methods


def _build_flow_definition_from_class(
    flow_class: type,
    namespace: dict[str, Any] | None = None,
) -> FlowDefinition:
    diagnostics: list[FlowDefinitionDiagnostic] = []
    methods: dict[str, FlowMethodDefinition] = {}
    flow_methods = _iter_flow_methods(flow_class)
    if namespace is not None:
        for attr_name, attr_value in namespace.items():
            if is_flow_method(attr_value) and _should_include_flow_method(
                flow_class, attr_value
            ):
                flow_methods[attr_name] = attr_value

    for method_name, method in flow_methods.items():
        methods[method_name] = _build_method_definition(
            method, diagnostics, f"methods.{method_name}"
        )

    description = None
    docstring = flow_class.__doc__
    if docstring:
        description = docstring.strip()

    definition = FlowDefinition(
        name=getattr(flow_class, "__name__", "Flow"),
        description=description,
        state=_build_state_definition(flow_class, diagnostics),
        config=_build_config_definition(flow_class, diagnostics),
        persist=_build_persistence_definition(flow_class, diagnostics, "persist"),
        methods=methods,
        diagnostics=diagnostics,
    )
    definition.diagnostics.extend(definition.validate_contract())
    definition.log_diagnostics()
    return definition


def build_flow_definition(
    flow_class: type,
    namespace: dict[str, Any] | None = None,
) -> FlowDefinition:
    """Build a FlowDefinition from a Python Flow class."""
    return _build_flow_definition_from_class(flow_class, namespace)
