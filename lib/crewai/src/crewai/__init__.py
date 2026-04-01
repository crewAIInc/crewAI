import contextvars
import threading
from typing import Any
import urllib.request
import warnings

from pydantic import PydanticUserError

from crewai.agent.core import Agent
from crewai.agent.planning_config import PlanningConfig
from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.flow.flow import Flow
from crewai.knowledge.knowledge import Knowledge
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.process import Process
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

__version__ = "1.13.0a6"
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
    from crewai.agents.tools_handler import ToolsHandler as _ToolsHandler
    from crewai.experimental.agent_executor import AgentExecutor as _AgentExecutor
    from crewai.hooks.llm_hooks import LLMCallHookContext as _LLMCallHookContext
    from crewai.tools.tool_types import ToolResult as _ToolResult
    from crewai.utilities.prompts import (
        StandardPromptResult as _StandardPromptResult,
        SystemPromptResult as _SystemPromptResult,
    )

    _AgentExecutor.model_rebuild(
        force=True,
        _types_namespace={
            "Agent": Agent,
            "ToolsHandler": _ToolsHandler,
            "Crew": Crew,
            "BaseLLM": BaseLLM,
            "Task": Task,
            "StandardPromptResult": _StandardPromptResult,
            "SystemPromptResult": _SystemPromptResult,
            "LLMCallHookContext": _LLMCallHookContext,
            "ToolResult": _ToolResult,
        },
    )
except (ImportError, PydanticUserError):
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "AgentExecutor.model_rebuild() failed; forward refs may be unresolved.",
        exc_info=True,
    )

__all__ = [
    "LLM",
    "Agent",
    "BaseLLM",
    "Crew",
    "CrewOutput",
    "Flow",
    "Knowledge",
    "LLMGuardrail",
    "Memory",
    "PlanningConfig",
    "Process",
    "Task",
    "TaskOutput",
    "__version__",
]
