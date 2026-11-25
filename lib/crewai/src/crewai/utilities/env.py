import contextvars
import os

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.env_events import (
    CCEnvEvent,
    CodexEnvEvent,
    CursorEnvEvent,
    DefaultEnvEvent,
)
from crewai.utilities.constants import CODING_AGENT_ENV_MARKERS


_env_context_emitted: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_env_context_emitted", default=False
)

# Assistants with an event of their own. Anything else in the shared table is
# detected with the same precedence but reported as DefaultEnvEvent, so the two
# detection paths can never disagree about which assistant is present.
_AGENT_EVENTS = {
    "claude_code": CCEnvEvent,
    "codex": CodexEnvEvent,
    "cursor": CursorEnvEvent,
}


def get_env_context() -> None:
    if _env_context_emitted.get():
        return
    _env_context_emitted.set(True)

    # Walks the shared table rather than restating precedence: hard-coding the
    # order here let the two paths drift, so a marker added for telemetry was
    # invisible to this one.
    for agent_name, env_vars in CODING_AGENT_ENV_MARKERS:
        if any(os.environ.get(var) for var in env_vars):
            event = _AGENT_EVENTS.get(agent_name, DefaultEnvEvent)
            crewai_event_bus.emit(None, event())
            return

    crewai_event_bus.emit(None, DefaultEnvEvent())


# Env var that opts out of the automatic `load_dotenv()` calls in
# `crewai.llm` and `crewai.project.crew_base` (see #9744). Hosts that manage
# env vars themselves set it to a truthy value before the first import.
AUTO_DOTENV_OPT_OUT_ENV_VAR = "CREWAI_NO_AUTO_DOTENV"


def auto_dotenv_enabled() -> bool:
    """Whether import-time `.env` autoloading is enabled.

    Returns False only when `CREWAI_NO_AUTO_DOTENV` is set to a truthy value
    ("1", "true", "yes", "on", case-insensitive). Anything else, including
    unset, keeps the historical default of loading the nearest `.env` on
    import.
    """
    return os.getenv(AUTO_DOTENV_OPT_OUT_ENV_VAR, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }
