import contextvars
import threading
from typing import Any
import urllib.request
import warnings

from pydantic import PydanticUserError

from crewai.agent.core import Agent
from crewai.agent.planning_config import PlanningConfig
from crewai.context import ExecutionContext
from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.flow.flow import Flow
from crewai.knowledge.knowledge import Knowledge
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.process import Process
from crewai.state.checkpoint_config import CheckpointConfig  # noqa: F401
from crewai.task import Task
from crewai.tasks.llm_guardrail import LLMGuardrail
from crewai.tasks.task_output import TaskOutput
from crewai.telemetry.telemetry import Telemetry


def _suppress_pydantic_deprecation_warnings() -> None:
    """Suppress Pydantic deprecation warnings using targeted monkey patch."""
    original_warn = warnings.warn

    def filtered_warn(
        message: Any,
        category: type | None = None,
        stacklevel: int = 1,
        source: Any = None,
    ) -> Any:
        if (
            category
            and hasattr(category, "__module__")
            and category.__module__ == "pydantic.warnings"
        ):
            return None
        return original_warn(message, category, stacklevel + 1, source)

    warnings.warn = filtered_warn  # type: ignore[assignment]


_suppress_pydantic_deprecation_warnings()

__version__ = "1.14.2rc1"
_telemetry_submitted = False


def _track_install() -> None:
    """Track package installation/first-use via Scarf analytics."""
    global _telemetry_submitted

    if _telemetry_submitted or Telemetry._is_telemetry_disabled():
        return

    try:
        pixel_url = "https://api.scarf.sh/v2/packages/CrewAI/crewai/docs/00f2dad1-8334-4a39-934e-003b2e1146db"

        req = urllib.request.Request(pixel_url)  # noqa: S310
        req.add_header("User-Agent", f"CrewAI-Python/{__version__}")

        with urllib.request.urlopen(req, timeout=2):  # noqa: S310
            _telemetry_submitted = True
    except Exception:  # noqa: S110
        pass


def _track_install_async() -> None:
    """Track installation in background thread to avoid blocking imports."""
    if not Telemetry._is_telemetry_disabled():
        ctx = contextvars.copy_context()
        thread = threading.Thread(target=ctx.run, args=(_track_install,), daemon=True)
        thread.start()


_track_install_async()

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Memory": ("crewai.memory.unified_memory", "Memory"),
}


def __getattr__(name: str) -> Any:
    """Lazily import heavy modules (e.g. Memory → lancedb) on first access."""
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'crewai' has no attribute {name!r}")


try:
    from crewai.agents.agent_builder.base_agent import BaseAgent as _BaseAgent
    from crewai.agents.agent_builder.base_agent_executor import (
        BaseAgentExecutor as _BaseAgentExecutor,
    )
    from crewai.agents.tools_handler import ToolsHandler as _ToolsHandler
    from crewai.experimental.agent_executor import AgentExecutor as _AgentExecutor
    from crewai.hooks.llm_hooks import LLMCallHookContext as _LLMCallHookContext
    from crewai.tools.tool_types import ToolResult as _ToolResult
    from crewai.utilities.prompts import (
        StandardPromptResult as _StandardPromptResult,
        SystemPromptResult as _SystemPromptResult,
    )

    _base_namespace: dict[str, type] = {
        "Agent": Agent,
        "BaseAgent": _BaseAgent,
        "Crew": Crew,
        "Flow": Flow,
        "BaseLLM": BaseLLM,
        "Task": Task,
        "BaseAgentExecutor": _BaseAgentExecutor,
        "ExecutionContext": ExecutionContext,
        "StandardPromptResult": _StandardPromptResult,
        "SystemPromptResult": _SystemPromptResult,
    }

    from crewai.tools.base_tool import BaseTool as _BaseTool
    from crewai.tools.structured_tool import CrewStructuredTool as _CrewStructuredTool

    _base_namespace["BaseTool"] = _BaseTool
    _base_namespace["CrewStructuredTool"] = _CrewStructuredTool

    try:
        from crewai.a2a.config import (
            A2AClientConfig as _A2AClientConfig,
            A2AConfig as _A2AConfig,
            A2AServerConfig as _A2AServerConfig,
        )

        _base_namespace.update(
            {
                "A2AConfig": _A2AConfig,
                "A2AClientConfig": _A2AClientConfig,
                "A2AServerConfig": _A2AServerConfig,
            }
        )
    except ImportError:
        pass

    import sys

    _full_namespace = {
        **_base_namespace,
        "ToolsHandler": _ToolsHandler,
        "StandardPromptResult": _StandardPromptResult,
        "SystemPromptResult": _SystemPromptResult,
        "LLMCallHookContext": _LLMCallHookContext,
        "ToolResult": _ToolResult,
    }

    _resolve_namespace = {
        **_full_namespace,
        **sys.modules[_BaseAgent.__module__].__dict__,
    }

    import crewai.state.runtime as _runtime_state_mod

    for _mod_name in (
        _BaseAgent.__module__,
        Agent.__module__,
        Crew.__module__,
        Flow.__module__,
        Task.__module__,
        "crewai.agents.crew_agent_executor",
        _runtime_state_mod.__name__,
        _AgentExecutor.__module__,
    ):
        sys.modules[_mod_name].__dict__.update(_resolve_namespace)

    from crewai.agents.crew_agent_executor import (
        CrewAgentExecutor as _CrewAgentExecutor,
    )
    from crewai.tasks.conditional_task import ConditionalTask as _ConditionalTask

    _BaseAgentExecutor.model_rebuild(force=True, _types_namespace=_full_namespace)
    _BaseAgent.model_rebuild(force=True, _types_namespace=_full_namespace)
    Task.model_rebuild(force=True, _types_namespace=_full_namespace)
    _ConditionalTask.model_rebuild(force=True, _types_namespace=_full_namespace)
    _CrewAgentExecutor.model_rebuild(force=True, _types_namespace=_full_namespace)
    Crew.model_rebuild(force=True, _types_namespace=_full_namespace)
    Flow.model_rebuild(force=True, _types_namespace=_full_namespace)
    _AgentExecutor.model_rebuild(force=True, _types_namespace=_full_namespace)

    from typing import Annotated

    from pydantic import Field

    from crewai.state.runtime import RuntimeState

    Entity = Annotated[
        Flow | Crew | Agent,  # type: ignore[type-arg]
        Field(discriminator="entity_type"),
    ]

    RuntimeState.model_rebuild(
        force=True,
        _types_namespace={**_full_namespace, "Entity": Entity},
    )

    try:
        Agent.model_rebuild(force=True, _types_namespace=_full_namespace)
    except PydanticUserError:
        pass

except (ImportError, PydanticUserError):
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "model_rebuild() failed; forward refs may be unresolved.",
        exc_info=True,
    )
    RuntimeState = None  # type: ignore[assignment,misc]

__all__ = [
    "LLM",
    "Agent",
    "BaseLLM",
    "Crew",
    "CrewOutput",
    "Entity",
    "ExecutionContext",
    "Flow",
    "Knowledge",
    "LLMGuardrail",
    "Memory",
    "PlanningConfig",
    "Process",
    "RuntimeState",
    "Task",
    "TaskOutput",
    "__version__",
]
