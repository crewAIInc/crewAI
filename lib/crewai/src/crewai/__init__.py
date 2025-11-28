import threading
from typing import Any
import urllib.request
import warnings


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

__version__ = "1.6.0"
_telemetry_submitted = False


def _track_install() -> None:
    """Track package installation/first-use via Scarf analytics."""
    global _telemetry_submitted

    from crewai.telemetry.telemetry import Telemetry

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
    from crewai.telemetry.telemetry import Telemetry

    if not Telemetry._is_telemetry_disabled():
        thread = threading.Thread(target=_track_install, daemon=True)
        thread.start()


_track_install_async()

__all__ = [
    "LLM",
    "Agent",
    "BaseLLM",
    "Crew",
    "CrewOutput",
    "Flow",
    "Knowledge",
    "LLMGuardrail",
    "Process",
    "Task",
    "TaskOutput",
    "__version__",
]


def __getattr__(name: str) -> Any:
    if name == "Agent":
        from crewai.agent.core import Agent

        return Agent
    if name == "Crew":
        from crewai.crew import Crew

        return Crew
    if name == "CrewOutput":
        from crewai.crews.crew_output import CrewOutput

        return CrewOutput
    if name == "Flow":
        from crewai.flow.flow import Flow

        return Flow
    if name == "Knowledge":
        from crewai.knowledge.knowledge import Knowledge

        return Knowledge
    if name == "LLM":
        from crewai.llm import LLM

        return LLM
    if name == "BaseLLM":
        from crewai.llms.base_llm import BaseLLM

        return BaseLLM
    if name == "Process":
        from crewai.process import Process

        return Process
    if name == "Task":
        from crewai.task import Task

        return Task
    if name == "LLMGuardrail":
        from crewai.tasks.llm_guardrail import LLMGuardrail

        return LLMGuardrail
    if name == "TaskOutput":
        from crewai.tasks.task_output import TaskOutput

        return TaskOutput
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
