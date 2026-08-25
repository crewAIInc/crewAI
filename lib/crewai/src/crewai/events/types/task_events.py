import sys
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, field_serializer

from crewai.events.base_events import BaseEvent
from crewai.tasks.task_output import TaskOutput


def _resolve_exception_class(value: Any) -> Any:
    """Turn a serialized ``module:qualname`` back into the class it names.

    Needed because the JSON dump cannot write a class. Without this, restoring a
    checkpoint fails ``TaskFailedEvent`` validation and ``_resolve_event`` silently
    degrades the whole event to a bare ``BaseEvent`` -- losing ``error`` as well, which
    is a plain string that would otherwise have survived.

    **Resolution is exact, and by module rather than by name.** An earlier revision
    matched on the bare ``__name__`` by walking ``BaseException.__subclasses__()``, which
    could bind to a *different* class of the same name -- two modules, two scopes, or two
    dynamically created classes can all share one. Reproduced by review with two distinct
    ``type("DupErr", (Exception,), {})`` classes: the event restored as the wrong one.

    **Nothing is imported here, and the lookup is by dict rather than by attribute.**
    The module has to be in ``sys.modules`` already. Importing a module named in
    serialized data would execute its top-level code -- a worse defect than the collision
    it would fix, in a field that exists to keep untrusted strings out of telemetry.

    ``vars()`` rather than ``getattr()`` is what makes that true, and the difference is
    not theoretical: a module with a PEP 562 ``__getattr__`` runs it on any attribute
    miss, and ``crewai.events`` itself is such a module -- its hook calls
    ``importlib.import_module``. So ``getattr`` here would import a submodule named in
    the serialized string, and any non-``AttributeError`` the hook raised would escape
    validation and degrade the whole event, dropping ``error`` with it. A ``__dict__``
    lookup consults no hook and runs no user code.

    A serialized identity that does not resolve becomes ``None`` -- an unloaded module, a
    ``<locals>`` qualname unreachable by attribute lookup, or a target that is not a
    ``BaseException`` subclass. ``None`` rather than the original string, so the field
    still validates and ``error`` survives the restore; the previous behaviour dropped
    the entire event.

    **A string with no ``":"`` is handled the other way, on purpose.** It cannot be
    something this serializer wrote, so it is a caller passing a string where a class
    belongs -- most likely ``str(error)``. That is returned unchanged and rejected by the
    field's own type, which keeps the loud failure a silent ``None`` would hide. This is
    the check that stops a message being recorded, and it is why the field is typed as a
    class: a one-word message such as ``"secret_token"`` is itself a valid identifier, so
    an ``isidentifier()`` gate on a plain string would let it through.
    """
    if not isinstance(value, str):
        return value

    module_name, separator, qualname = value.partition(":")
    if not separator:
        return value
    if not qualname:
        return None

    resolved: Any = sys.modules.get(module_name)
    if resolved is None:
        return None
    for attribute in qualname.split("."):
        try:
            namespace = vars(resolved)
        except TypeError:
            # No __dict__ to read: the qualname points at something that cannot
            # hold a nested class, so there is nothing to resolve.
            return None
        resolved = namespace.get(attribute)
        if resolved is None:
            return None

    if isinstance(resolved, type) and issubclass(resolved, BaseException):
        return resolved
    return None


_ExceptionClass = Annotated[
    type[BaseException] | None, BeforeValidator(_resolve_exception_class)
]
"""An exception class, e.g. ``ValueError``, or the ``module:qualname`` a JSON dump wrote.

Declared here rather than inline because ``TaskFailedEvent`` has a field named ``type``,
which shadows the builtin for the rest of that class body: an inline
``type[BaseException]`` after that field is bound raises ``TypeError`` at import, and mypy
rejects it as "Variable ... is not valid as a type".

``| None`` is inside the alias rather than on the field so the validator runs once, on
the whole annotation. Spelling the field ``_ExceptionClass | None`` would make it a union
whose members are each tried in turn, so a ``None`` returned by the validator would
depend on union resolution order.
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
    error_type: _ExceptionClass = None
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
    def _serialize_error_type(self, error_type: _ExceptionClass) -> str | None:
        """``module:qualname``, so the event stays JSON-serializable and restores exactly.

        A class is not a JSON type, so without this ``model_dump(mode="json")``
        raises ``PydanticSerializationError`` for the whole event -- which breaks
        checkpointing after a task failure and sends a ``repr`` to AMP.

        Qualified rather than the bare ``__name__``: a name alone cannot distinguish two
        exception classes that share one, so restoring from it could pick the wrong
        class. See ``_resolve_exception_class``.

        ``when_used="json"`` is load-bearing: ``event_listener`` hands the live class
        to ``Telemetry.task_failed``, which needs it for ``_safe_error_type``, and
        python-mode dumps must keep it too.
        """
        if error_type is None:
            return None
        return f"{error_type.__module__}:{error_type.__qualname__}"


class TaskEvaluationEvent(BaseEvent):
    """Event emitted when a task evaluation is completed"""

    type: Literal["task_evaluation"] = "task_evaluation"
    evaluation_type: str
    task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        _set_task_fingerprint(self, self.task)
