from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from crewai.events.base_events import BaseEvent


class CCEnvEvent(BaseEvent):
    type: Literal["cc_env"] = "cc_env"


class CodexEnvEvent(BaseEvent):
    type: Literal["codex_env"] = "codex_env"


class CursorEnvEvent(BaseEvent):
    type: Literal["cursor_env"] = "cursor_env"


class DefaultEnvEvent(BaseEvent):
    type: Literal["default_env"] = "default_env"


EnvContextEvent = Annotated[
    CCEnvEvent | CodexEnvEvent | CursorEnvEvent | DefaultEnvEvent,
    Field(discriminator="type"),
]

env_context_event_adapter: TypeAdapter[EnvContextEvent] = TypeAdapter(EnvContextEvent)

ENV_CONTEXT_EVENT_TYPES: tuple[type[BaseEvent], ...] = (
    CCEnvEvent,
    CodexEnvEvent,
    CursorEnvEvent,
    DefaultEnvEvent,
)
