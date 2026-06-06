from __future__ import annotations

from collections.abc import Sequence
import json
import logging
from typing import Any, ParamSpec, TypeVar

from pydantic import BaseModel
from typing_extensions import TypeIs

from crewai.flow.constants import AND_CONDITION, OR_CONDITION
from crewai.flow.dsl._conditions import (
    _definition_condition_from_runtime,
    _extract_all_methods,
    _method_reference_name,
    _runtime_listener_condition_from_definition,
    is_flow_condition_dict,
)
from crewai.flow.dsl._types import FlowTrigger
from crewai.flow.flow_definition import (
    FlowConfigDefinition,
    FlowDefinition,
    FlowDefinitionCondition,
    FlowDefinitionDiagnostic,
    FlowHumanFeedbackDefinition,
    FlowMethodDefinition,
    FlowPersistenceDefinition,
    FlowStateDefinition,
)
from crewai.flow.flow_wrappers import (
    FlowMethod,
    ListenMethod,
    RouterMethod,
    StartMethod,
)
from crewai.flow.types import FlowMethodName


P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)

_FLOW_METHOD_DEFINITION_ATTR = "__flow_method_definition__"


def is_flow_method(obj: Any) -> TypeIs[FlowMethod[Any, Any]]:
    """Check if the object carries Flow method wrapper metadata."""
    return (
        hasattr(obj, "__is_flow_method__")
        or hasattr(obj, "__is_start_method__")
        or hasattr(obj, "__trigger_methods__")
        or hasattr(obj, "__is_router__")
        or hasattr(obj, _FLOW_METHOD_DEFINITION_ATTR)
    )


def _should_include_flow_method(flow_class: type, method: Any) -> bool:
    if getattr(method, "__conversational_only__", False):
        return bool(getattr(flow_class, "conversational", False))
    return True


def _flow_method_names(values: Sequence[Any]) -> list[FlowMethodName]:
    return [FlowMethodName(str(value)) for value in values]


def _set_trigger_metadata(
    wrapper: StartMethod[P, R] | ListenMethod[P, R] | RouterMethod[P, R],
    condition: FlowTrigger,
) -> None:
    if isinstance(condition, str):
        wrapper.__trigger_methods__ = [FlowMethodName(condition)]
        wrapper.__condition_type__ = OR_CONDITION
        return

    if is_flow_condition_dict(condition):
        if "conditions" in condition:
            wrapper.__trigger_condition__ = condition
            wrapper.__trigger_methods__ = _extract_all_methods(condition)
            wrapper.__condition_type__ = condition["type"]
            return
        if "methods" in condition:
            wrapper.__trigger_methods__ = _flow_method_names(condition["methods"])
            wrapper.__condition_type__ = condition["type"]
            return
        raise ValueError("Condition dict must contain 'conditions' or 'methods'")

    method_name = _method_reference_name(condition)
    if method_name is not None:
        wrapper.__trigger_methods__ = [method_name]
        wrapper.__condition_type__ = OR_CONDITION
        return

    raise ValueError(
        "Condition must be a method, string, or a result of or_() or and_()"
    )


def _set_flow_method_definition(
    wrapper: StartMethod[P, R] | ListenMethod[P, R] | RouterMethod[P, R],
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


def _condition_from_method_metadata(method: Any) -> FlowDefinitionCondition | None:
    trigger_condition = getattr(method, "__trigger_condition__", None)
    if trigger_condition is not None:
        return _definition_condition_from_runtime(trigger_condition)

    trigger_methods = getattr(method, "__trigger_methods__", None)
    if trigger_methods is None:
        return None
    condition_type = getattr(method, "__condition_type__", OR_CONDITION)
    method_names = [str(method_name) for method_name in trigger_methods]
    if condition_type == AND_CONDITION:
        return {"and": method_names}
    if len(method_names) == 1:
        return method_names[0]
    return {"or": method_names}


def _flow_method_definition_from_legacy_metadata(method: Any) -> FlowMethodDefinition:
    is_start = bool(getattr(method, "__is_start_method__", False))
    is_router = bool(getattr(method, "__is_router__", False))
    condition = _condition_from_method_metadata(method)

    if not is_start:
        start_value: bool | FlowDefinitionCondition | None = None
    elif condition is not None:
        start_value = condition
    else:
        start_value = True

    definition = FlowMethodDefinition(
        start=start_value,
        listen=condition if not is_start else None,
        router=is_router,
    )

    router_emit = getattr(method, "__router_emit__", None)
    if router_emit:
        definition.emit = [str(value) for value in router_emit]
    return definition


def _definition_trigger_condition(
    method_definition: FlowMethodDefinition,
) -> FlowDefinitionCondition | None:
    if method_definition.listen is not None:
        return method_definition.listen
    if isinstance(method_definition.start, (str, dict)):
        return method_definition.start
    return None


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
        method_definition = _flow_method_definition_from_legacy_metadata(method)
    else:
        method_definition = fragment.model_copy(deep=True)

    if bool(getattr(method, "__is_router__", False)):
        method_definition.router = True

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

    router_emit = getattr(method, "__router_emit__", None)
    if router_emit and not (human_feedback and human_feedback.emit):
        if not method_definition.emit:
            method_definition.emit = [str(value) for value in router_emit]

    return method_definition


def _iter_flow_methods(flow_class: type) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for attr_name in dir(flow_class):
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


def extract_flow_definition(
    namespace: dict[str, Any],
) -> tuple[list[str], dict[str, Any], set[str], dict[str, Any]]:
    """Extract the structural flow registries from a Python class namespace."""
    start_methods = []
    listeners = {}
    router_emit = {}
    routers = set()

    for attr_name, attr_value in namespace.items():
        if is_flow_method(attr_value):
            method_definition = _get_flow_method_definition(attr_value)
            if method_definition is not None:
                if method_definition.is_start:
                    start_methods.append(attr_name)

                condition = _definition_trigger_condition(method_definition)
                if condition is not None:
                    listeners[attr_name] = _runtime_listener_condition_from_definition(
                        condition
                    )

                is_router = method_definition.router or bool(
                    getattr(attr_value, "__is_router__", False)
                )
                if is_router:
                    routers.add(attr_name)
                    if method_definition.emit:
                        router_emit[attr_name] = [
                            str(value) for value in method_definition.emit
                        ]
                    elif (
                        hasattr(attr_value, "__router_emit__")
                        and attr_value.__router_emit__
                    ):
                        router_emit[attr_name] = attr_value.__router_emit__
                    else:
                        router_emit[attr_name] = []
                continue

            if hasattr(attr_value, "__is_start_method__"):
                start_methods.append(attr_name)

            if (
                hasattr(attr_value, "__trigger_methods__")
                and attr_value.__trigger_methods__ is not None
            ):
                methods = attr_value.__trigger_methods__
                condition_type = getattr(attr_value, "__condition_type__", OR_CONDITION)

                if (
                    hasattr(attr_value, "__trigger_condition__")
                    and attr_value.__trigger_condition__ is not None
                ):
                    listeners[attr_name] = attr_value.__trigger_condition__
                else:
                    listeners[attr_name] = (condition_type, methods)

                if hasattr(attr_value, "__is_router__") and attr_value.__is_router__:
                    routers.add(attr_name)
                    if (
                        hasattr(attr_value, "__router_emit__")
                        and attr_value.__router_emit__
                    ):
                        router_emit[attr_name] = attr_value.__router_emit__
                    else:
                        router_emit[attr_name] = []

            if (
                hasattr(attr_value, "__is_start_method__")
                and hasattr(attr_value, "__is_router__")
                and attr_value.__is_router__
            ):
                routers.add(attr_name)
                if (
                    hasattr(attr_value, "__router_emit__")
                    and attr_value.__router_emit__
                ):
                    router_emit[attr_name] = attr_value.__router_emit__
                else:
                    router_emit[attr_name] = []

    return start_methods, listeners, routers, router_emit
