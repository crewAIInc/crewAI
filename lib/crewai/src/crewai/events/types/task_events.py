import builtins
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, field_serializer

from crewai.events.base_events import BaseEvent
from crewai.tasks.task_output import TaskOutput


def _resolve_exception_class(value: Any) -> Any:
    """Turn a serialized class name back into the class, leaving anything else alone.

    Needed because the JSON dump writes the class *name*. Without this, restoring a
    checkpoint fails ``TaskFailedEvent`` validation and ``_resolve_event`` silently
    degrades the whole event to a bare ``BaseEvent`` -- losing ``error`` as well, which
    is a plain string that would otherwise have survived.

    Resolution is against real exception classes only, so this does not reopen the hole
    the class-typed field closes: a message such as ``"secret_token"`` resolves to
    nothing, is returned unchanged, and is then rejected by the field's own type.

    A name whose class is not imported in this process resolves to nothing and degrades
    as before -- accepted, because inventing a class from an arbitrary string would be
    exactly the injection risk this field exists to avoid.
    """
    if not isinstance(value, str):
        return value

    candidate = getattr(builtins, value, None)
    if isinstance(candidate, type) and issubclass(candidate, BaseException):
        return candidate

    seen: set[type[BaseException]] = set()
    stack: list[type[BaseException]] = [BaseException]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        if cls.__name__ == value:
            return cls
        stack.extend(cls.__subclasses__())
    return value


_ExceptionClass = Annotated[
    type[BaseException], BeforeValidator(_resolve_exception_class)
]
"""An exception class, e.g. ``ValueError``, or the class name a JSON dump produced.

Declared here rather than inline because ``TaskFailedEvent`` has a field named ``type``,
which shadows the builtin for the rest of that class body: an inline
``type[BaseException]`` after that field is bound raises ``TypeError`` at import, and mypy
rejects it as "Variable ... is not valid as a type".
"""


def _set_task_fingerprint(event: BaseEvent, task: Any) -> None:
    """Set task identity and fingerprint data on an event."""
    if task is None:
        return
    task_id = getattr(task, "id", None)
    if task_id is not None:
        event.task_id = str(task_id)
    task_name = getattr(task, "name", None) or getattr(task, "description", None)
    if task_name:
        event.task_name = task_name
    if task.fingerprint:
        event.source_fingerprint = task.fingerprint.uuid_str
        event.source_type = "task"
        if task.fingerprint.metadata:
            event.fingerprint_metadata = task.fingerprint.metadata


class TaskStartedEvent(BaseEvent):
    """Event emitted when a task starts"""

    type: Literal["task_started"] = "task_started"
    context: str | None
    task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        _set_task_fingerprint(self, self.task)


class TaskCompletedEvent(BaseEvent):
    """Event emitted when a task completes"""

    output: TaskOutput
    type: Literal["task_completed"] = "task_completed"
    task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        _set_task_fingerprint(self, self.task)


class TaskFailedEvent(BaseEvent):
    """Event emitted when a task fails"""

    error: str
    error_type: _ExceptionClass | None = None
    """The exception's class, e.g. ``ValidationError``.

    The class and not its name, matching ``Telemetry._safe_error_type``: a message
    is never a type, so ``str(error)`` cannot be passed here at all. A name would
    not be safe on its own, because a single-word message such as
    ``"secret_token"`` is itself a valid identifier.

    Kept separate from ``error`` so telemetry can record what kind of failure
    occurred without ever touching the message, which routinely contains prompts,
    model output, file paths or credentials.
    """
    type: Literal["task_failed"] = "task_failed"
    task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        _set_task_fingerprint(self, self.task)

    @field_serializer("error_type", when_used="json")
    def _serialize_error_type(self, error_type: _ExceptionClass | None) -> str | None:
        """The class name, so the event stays JSON-serializable.

        A class is not a JSON type, so without this ``model_dump(mode="json")``
        raises ``PydanticSerializationError`` for the whole event -- which breaks
        checkpointing after a task failure and sends a ``repr`` to AMP.

        ``when_used="json"`` is load-bearing: ``event_listener`` hands the live class
        to ``Telemetry.task_failed``, which needs it for ``_safe_error_type``, and
        python-mode dumps must keep it too.
        """
        return error_type.__name__ if error_type else None


class TaskEvaluationEvent(BaseEvent):
    """Event emitted when a task evaluation is completed"""

    type: Literal["task_evaluation"] = "task_evaluation"
    evaluation_type: str
    task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        _set_task_fingerprint(self, self.task)
