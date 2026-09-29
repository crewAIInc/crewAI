"""Process-context ``role -> model`` and ``model -> model`` overlay.

``llm_overlay`` lets a per-run caller make specific agents, or specific
models, use a different model without editing the code that builds them.
While the block is active, an ``Agent`` or ``LiteAgent`` whose ``role`` is a
key of the mapping is built with the mapped model instead of its declared
``llm``. Roles that are not keys, and every agent built outside the block,
keep their own model.

A key that starts with :data:`MODEL_KEY_PREFIX` names a model instead of a
role: ``"model:openai/gpt-4o"`` maps every LLM built from that model string
inside the block, and ``"model:*"`` every LLM built from any model string.
That is what reaches an LLM no role names — a flow step's own
``LLM(model="openai/gpt-4o").call(...)`` — and the simplest way to say "this
whole run on model X". A model key is read wherever an LLM is built from a
model string: ``LLM(model=...)`` (and so ``create_llm`` and an agent's
``llm="..."``, which go through it), and an agent's declared ``llm`` instance
when it was built outside the block. A model is compared as the caller wrote
it and without the provider prefix native providers strip, so
``"model:openai/gpt-4o"`` matches ``LLM(model="gpt-4o")`` and an instance
whose ``model`` reads ``"gpt-4o"``, and ``"model:gpt-4o"`` matches
``LLM(model="openai/gpt-4o")``; an exact key wins over a stripped one, and
both over ``"model:*"``. The mapped model is built once, never looked up
again: ``{"model:a": "b", "model:b": "c"}`` puts an ``a`` on ``b``. For an
agent a role key wins over every model key, so ``{"Researcher": x,
"model:*": y}`` runs the Researcher on ``x`` and everything else on ``y``.
The settings the caller passed to ``LLM(...)`` are carried to the mapped model
by the same rule as a declared ``llm``'s (below). Only ``LLM`` itself is
mapped: a subclass of it, or a provider class built directly, is the caller's
own choice of class and keeps its model.

Roles are matched exactly, by the text they have when the overlay is read;
only whitespace around a role or a key is ignored, so a role a YAML file
leaves with a trailing newline matches a key written for the clean text.
An ``Agent`` is read when it is built, with its declared role, and read
again when ``crew.kickoff(inputs=...)`` interpolates the inputs into that
role and the text changes. The second read is what lets a role declared as
a template (``"Researcher for {repo}"``, the YAML form of a ``CrewBase``
crew) match a key written for the interpolated text, which is the role
every trace records; the template itself is not a key at construction. A
role the interpolation leaves unchanged is not read again. Inside a block,
an agent runs on the model its current role maps to, and on its declared
``llm`` when that role is not a key: a kickoff that interpolates to a role
outside the mapping puts the declared ``llm`` instance back, so a ``Crew``
reused across kickoffs with different inputs never bills a previous role's
provider. Outside any block an interpolation changes nothing.

A mapped model is built with the settings a caller configured on the
declared ``llm`` (``crewai.utilities.llm_utils.create_llm_like``): generation
and runtime settings — temperature, timeouts, token limits, stop sequences —
when the new model's class has the field and accepts the value; credentials,
endpoints and provider-specific settings only when the mapped model is on the
same provider — another provider gets its own defaults and environment. What a
provider derived from the model rather than took from the caller is derived
again for the new model.

The overlay is a :class:`contextvars.ContextVar`, so it follows the calling
context, not the process. A plain ``threading.Thread`` started inside the
block does not see it: callers that run agents in their own threads must
propagate the context themselves, e.g.
``contextvars.copy_context().run(build_and_run_agents)``. With
``Crew(stream=True)``, ``kickoff`` returns a stream and runs the crew — and the
kickoff-time read — when that stream is first iterated, in a copy of the
context taken then: iterate it inside the block.

Each ``Agent`` reads the overlay once when built and once more only if a
kickoff rewrites its role; a re-validation of an existing agent (the event bus
registers an agent when it first emits) does not read it again, so an agent
built outside a block keeps its model through a kickoff inside one.

Example:
    >>> with llm_overlay({"model:*": "openai/gpt-4o-mini"}):
    ...     flow.kickoff()  # every LLM built in the run is on gpt-4o-mini

    >>> with llm_overlay({"Researcher": "openai/gpt-4o"}):
    ...     crew = build_crew()  # the Researcher agent is built on gpt-4o
    ...     crew.kickoff()

    >>> with llm_overlay({"Researcher for crewAIInc/x": "openai/gpt-4o"}):
    ...     crew = build_crew()  # role "Researcher for {repo}": declared llm
    ...     crew.kickoff(inputs={"repo": "crewAIInc/x"})  # now runs on gpt-4o
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Final, TypeVar
import weakref


MODEL_KEY_PREFIX: Final[str] = "model:"
"""A key starting with this names a model (``"model:openai/gpt-4o"``), not a role."""

_T = TypeVar("_T")

ANY_MODEL: Final[str] = "*"
"""The model part of the key that maps every LLM built from a model string."""


active: ContextVar[dict[str, str] | None] = ContextVar(
    "llm_overlay_active", default=None
)

# Set while the overlay builds the model it mapped to, so that build is not
# mapped again (a chain of model keys, or ``model:*`` under a role's model).
_building_mapped: ContextVar[bool] = ContextVar(
    "llm_overlay_building_mapped", default=False
)

# The LLMs a model key already mapped, by id. An agent reading its declared
# ``llm`` skips one of these: its model is the mapped one, never a key again.
_mapped_llms: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()


@contextmanager
def llm_overlay(mapping: dict[str, str] | None) -> Iterator[None]:
    """Route agent roles and models to models for the duration of the block.

    Args:
        mapping: ``{role: model}`` and ``{"model:<model>": model}`` (or
            ``{"model:*": model}``) for the block; ``None`` clears any active
            overlay for the block. Whitespace around each role, and around the
            model a model key names, is dropped from the copy the block uses,
            so a key matches a role that differs from it only by its
            surrounding whitespace; the mapping passed in is left as it is. The
            previous value is always restored on exit, including when the
            block raises.

    Raises:
        ValueError: Two keys name one role or one model with different
            models, or a model key names no model.
    """
    token = active.set(_stripped(mapping))
    try:
        yield
    finally:
        active.reset(token)


def overlay_model_for(role: str | None) -> str | None:
    """The model the active overlay assigns to ``role``.

    Whitespace around ``role`` is ignored, as it was around the keys when the
    overlay was set. An empty or ``None`` role matches nothing, and a role is
    never matched against a model key.

    Returns:
        The mapped model string, or ``None`` when no overlay is active or
        ``role`` is not one of its keys.
    """
    mapping = active.get()
    if not mapping or role is None:
        return None
    role = role.strip()
    if not role or role.startswith(MODEL_KEY_PREFIX):
        return None
    return mapping.get(role)


def overlay_model_for_model(
    model: str | None, provider: str | None = None
) -> str | None:
    """The model the active overlay's model keys assign to an LLM on ``model``.

    ``model`` is compared as written, then with its provider prefix put back
    (``provider``, when ``model`` has none) or taken off (when it has one), so
    a key matches whichever form a caller or a native provider left on it;
    ``model:*`` matches every model. Nothing matches while the overlay is
    building a model it mapped.

    Args:
        model: The model string an LLM is being built from, or the ``model``
            of an existing instance.
        provider: The provider the LLM routes to, when ``model`` does not say.

    Returns:
        The mapped model string, or ``None`` when no overlay is active, it has
        no model key for ``model``, or a mapped model is being built.
    """
    mapping = active.get()
    if not mapping or not model or _building_mapped.get():
        return None
    model = model.strip()
    _, separator, bare = model.partition("/")
    forms = [model]
    if separator:
        forms.append(bare)
    elif provider:
        forms.append(f"{provider}/{model}")
    for form in forms:
        mapped = mapping.get(MODEL_KEY_PREFIX + form)
        if mapped is not None:
            return mapped
    return mapping.get(MODEL_KEY_PREFIX + ANY_MODEL)


def overlay_model_for_llm(llm: Any) -> str | None:
    """The model the active overlay's model keys assign to an existing ``llm``.

    An agent's declared ``llm`` built outside the block is looked up by its
    ``model`` and ``provider``; one a model key already mapped when it was
    built is not looked up again.
    """
    if llm is None or _mapped_llms.get(id(llm)) is llm:
        return None
    return overlay_model_for_model(
        getattr(llm, "model", None), getattr(llm, "provider", None)
    )


@contextmanager
def building_mapped_model() -> Iterator[None]:
    """Build the model the overlay mapped to, without mapping it again."""
    token = _building_mapped.set(True)
    try:
        yield
    finally:
        _building_mapped.reset(token)


def mark_mapped(llm: _T) -> _T:
    """Record ``llm`` as built by a model key, so no model key maps it again."""
    try:
        _mapped_llms[id(llm)] = llm
    except TypeError:
        pass
    return llm


def _stripped(mapping: dict[str, str] | None) -> dict[str, str] | None:
    """A copy of ``mapping`` with the whitespace around each role dropped.

    Two keys that differ only by whitespace name one role (or, after the
    :data:`MODEL_KEY_PREFIX`, one model). When they name the same model the
    copy holds it once; when they name different models the mapping is
    ambiguous and is refused, so the model an agent runs on never depends on
    dictionary order.
    """
    if mapping is None:
        return None
    stripped: dict[str, str] = {}
    for role, model in mapping.items():
        key = role.strip()
        if key.startswith(MODEL_KEY_PREFIX):
            named = key[len(MODEL_KEY_PREFIX) :].strip()
            if not named:
                raise ValueError(
                    f"llm_overlay: the key {role!r} names no model; write "
                    f"'{MODEL_KEY_PREFIX}<provider/model>' or "
                    f"'{MODEL_KEY_PREFIX}{ANY_MODEL}'"
                )
            key = MODEL_KEY_PREFIX + named
        if key in stripped and stripped[key] != model:
            raise ValueError(
                f"llm_overlay: role {key!r} is mapped twice with different models "
                f"({stripped[key]!r} and {model!r}); give each role one model"
            )
        stripped[key] = model
    return stripped
