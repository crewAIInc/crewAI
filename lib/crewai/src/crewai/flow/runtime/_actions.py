"""Build FlowDefinition actions into live runtime callables."""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Callable
import contextvars
import inspect
import os
from typing import TYPE_CHECKING, Any, Protocol, cast

from crewai.flow.flow_definition import (
    FlowActionDefinition,
    FlowCodeActionDefinition,
    FlowCrewActionDefinition,
    FlowEachActionDefinition,
    FlowEachInnerActionDefinition,
    FlowExpressionActionDefinition,
    FlowScriptActionDefinition,
    FlowToolActionDefinition,
)
from crewai.flow.runtime._expressions import evaluate_expression, render_with_block
from crewai.flow.runtime._outputs import outputs_by_name
from crewai.flow.runtime._refs import InvalidRefError, resolve_ref


if TYPE_CHECKING:
    from crewai.flow.runtime import Flow


__all__ = ["build_action"]

LocalContext = dict[str, Any]
_LOCAL_CONTEXT_KWARG = "__flow_definition_local_context"
_ALLOW_SCRIPT_EXECUTION_ENV_VAR = "CREWAI_ALLOW_FLOW_SCRIPT_EXECUTION"
_TRUSTED_SCRIPT_EXECUTION_VALUES = frozenset({"1", "true", "yes"})


class _BuiltAction(Protocol):
    def run(self, *args: Any, **kwargs: Any) -> Any: ...


class _ActionType(Protocol):
    definition_type: type[Any]

    def __call__(self, flow: Flow[Any], definition: Any) -> _BuiltAction: ...


class CodeAction:
    definition_type = FlowCodeActionDefinition

    def __init__(self, flow: Flow[Any], definition: FlowCodeActionDefinition) -> None:
        self.flow = flow
        self.definition = definition
        self.handler = self._resolve_handler()
        self.signature = inspect.signature(self.handler)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        local_context = _pop_local_context(kwargs)
        if self.definition.with_ is None:
            return self.handler(*args, **kwargs)
        return self.handler(
            **render_with_block(
                self.flow, self.definition.with_, local_context=local_context
            )
        )

    def _resolve_handler(self) -> Callable[..., Any]:
        ref = self.definition.ref
        target = resolve_ref(ref, field="do")
        if not callable(target):
            raise InvalidRefError(f"invalid do ref {ref!r}; object is not callable")
        handler = cast(Callable[..., Any], target)
        if getattr(handler, "__self__", None) is None and hasattr(handler, "__get__"):
            handler = handler.__get__(self.flow, type(self.flow))
        return handler


class ToolAction:
    definition_type = FlowToolActionDefinition

    def __init__(self, flow: Flow[Any], definition: FlowToolActionDefinition) -> None:
        self.flow = flow
        self.definition = definition
        self.tool = self._build_tool()
        self.kwargs = definition.with_ or {}

    def run(self, *_args: Any, **kwargs: Any) -> Any:
        local_context = _pop_local_context(kwargs)
        return self.tool.run(
            **render_with_block(self.flow, self.kwargs, local_context=local_context)
        )

    def _build_tool(self) -> Any:
        target = resolve_ref(self.definition.ref, field="do")
        from crewai.tools import BaseTool

        if not (inspect.isclass(target) and issubclass(target, BaseTool)):
            raise InvalidRefError(
                f"invalid tool ref {self.definition.ref!r}; expected a BaseTool class"
            )

        try:
            tool_cls = cast(Callable[[], BaseTool], target)
            return tool_cls()
        except Exception as e:
            raise InvalidRefError(
                f"cannot instantiate tool ref {self.definition.ref!r} "
                f"without arguments: {e}"
            ) from e


class CrewAction:
    definition_type = FlowCrewActionDefinition

    def __init__(self, flow: Flow[Any], definition: FlowCrewActionDefinition) -> None:
        self.flow = flow
        self.definition = definition

    async def run(self, *_args: Any, **kwargs: Any) -> Any:
        from crewai.project.crew_loader import load_crew_from_definition

        local_context = _pop_local_context(kwargs)
        crew_definition = self.definition.with_
        inputs = render_with_block(
            self.flow, crew_definition.inputs, local_context=local_context
        )
        crew, _ = load_crew_from_definition(crew_definition, source="crew action")
        return await crew.kickoff_async(inputs=inputs)


class ExpressionAction:
    definition_type = FlowExpressionActionDefinition

    def __init__(
        self, flow: Flow[Any], definition: FlowExpressionActionDefinition
    ) -> None:
        self.flow = flow
        self.definition = definition

    def run(self, *_args: Any, **kwargs: Any) -> Any:
        local_context = _pop_local_context(kwargs)
        return evaluate_expression(
            self.flow, self.definition.expr, local_context=local_context
        )


class ScriptAction:
    definition_type = FlowScriptActionDefinition

    def __init__(self, flow: Flow[Any], definition: FlowScriptActionDefinition) -> None:
        self.flow = flow
        self.definition = definition
        self.handler = self._compile_handler()

    def run(self, *args: Any, **kwargs: Any) -> Any:
        local_context = _pop_local_context(kwargs)
        return self.handler(
            state=self.flow.state,
            outputs=outputs_by_name(
                self.flow._method_outputs,
                local_outputs=local_context.get("outputs") if local_context else None,
            ),
            input=args[0] if args else None,
            item=local_context.get("item") if local_context else None,
        )

    def _compile_handler(self) -> Callable[..., Any]:
        raw = os.environ.get(_ALLOW_SCRIPT_EXECUTION_ENV_VAR, "")
        if raw.strip().lower() not in _TRUSTED_SCRIPT_EXECUTION_VALUES:
            raise RuntimeError(
                "Flow script execution is disabled by default. "
                f"Set {_ALLOW_SCRIPT_EXECUTION_ENV_VAR}=1 to enable it only for "
                "trusted flow definitions."
            )

        filename = f"crewai.flow.script.{self.flow._definition.name}"
        module = ast.parse(self.definition.code, filename=filename)
        function = ast.FunctionDef(
            name="_flow_script",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg) for arg in ("state", "outputs", "input", "item")],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=module.body or [ast.Pass()],
            decorator_list=[],
            returns=None,
            type_comment=None,
            type_params=[],
        )
        module.body = [function]
        ast.fix_missing_locations(module)

        namespace: dict[str, Any] = {"__name__": filename}
        exec(compile(module, filename, "exec"), namespace)  # nosec B102 # noqa: S102
        return cast(Callable[..., Any], namespace["_flow_script"])


class EachAction:
    definition_type = FlowEachActionDefinition

    def __init__(self, flow: Flow[Any], definition: FlowEachActionDefinition) -> None:
        self.flow = flow
        self.definition = definition
        self.inner_actions = [
            (inner_action.name, self._build_inner_action(inner_action))
            for inner_action in definition.do
        ]

    async def run(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        items = evaluate_expression(self.flow, self.definition.in_)
        if not isinstance(items, list):
            raise ValueError("each.in must evaluate to an array")

        results: list[Any] = []

        for item in items:
            local_outputs: dict[str, Any] = {}
            last_output: Any = None
            for name, run_inner_action in self.inner_actions:
                last_output = await run_inner_action(
                    {"item": item, "outputs": local_outputs}
                )
                local_outputs[name] = last_output
            results.append(last_output)

        return results

    def _build_inner_action(
        self, inner_action: FlowEachInnerActionDefinition
    ) -> Callable[[LocalContext], Any]:
        run_action = build_action(self.flow, inner_action.action)

        async def run_inner_action(local_context: LocalContext) -> Any:
            kwargs = {_LOCAL_CONTEXT_KWARG: local_context}
            if inspect.iscoroutinefunction(run_action):
                result = run_action(**kwargs)
            else:
                ctx = contextvars.copy_context()

                def run_with_context() -> Any:
                    return run_action(**kwargs)

                result = await asyncio.to_thread(ctx.run, run_with_context)
            if inspect.isawaitable(result):
                result = await result
            return result

        return run_inner_action


_ACTION_TYPES: tuple[_ActionType, ...] = (
    EachAction,
    CodeAction,
    ToolAction,
    CrewAction,
    ExpressionAction,
    ScriptAction,
)


def build_action(
    flow: Flow[Any], definition: FlowActionDefinition
) -> Callable[..., Any]:
    """Turn one `do:` action into the callable the flow runs for that node."""
    for action_type in _ACTION_TYPES:
        if isinstance(definition, action_type.definition_type):
            return _as_flow_method(action_type(flow, definition))
    raise ValueError(f"unknown call type {getattr(definition, 'call', None)!r}")


def _as_flow_method(action: _BuiltAction) -> Callable[..., Any]:
    run: Callable[..., Any]
    if inspect.iscoroutinefunction(action.run):

        async def run_async(*args: Any, **kwargs: Any) -> Any:
            return await action.run(*args, **kwargs)

        run = run_async
    else:

        def run_sync(*args: Any, **kwargs: Any) -> Any:
            return action.run(*args, **kwargs)

        run = run_sync

    signature = getattr(action, "signature", None)
    if signature is not None:
        object.__setattr__(run, "__signature__", signature)
    return run


def _pop_local_context(kwargs: dict[str, Any]) -> LocalContext | None:
    local_context = kwargs.pop(_LOCAL_CONTEXT_KWARG, None)
    if local_context is None:
        return None
    if not isinstance(local_context, dict):
        raise TypeError("flow definition local context must be a mapping")
    return cast(LocalContext, local_context)
