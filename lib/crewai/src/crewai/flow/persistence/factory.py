"""Pluggable default persistence backend for flows.

By default, ``@persist`` and the flow runtime persist state with
:class:`~crewai.flow.persistence.sqlite.SQLiteFlowPersistence` when no explicit
``persistence=`` is given. Registering a factory via
:func:`set_flow_persistence_factory` lets an application back flow state with a
custom :class:`~crewai.flow.persistence.base.FlowPersistence` -- a database, a
remote service, an in-memory fake for tests -- without passing a
``persistence=`` instance at every ``@persist`` / kickoff site.

This mirrors :func:`crewai_core.lock_store.set_lock_backend`: a one-time,
process-wide setter intended for application startup. Pass ``None`` to restore
the built-in SQLite default. Call :func:`default_flow_persistence` to build the
default backend (the registered factory if any, else SQLite).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from crewai.flow.persistence.base import FlowPersistence

FlowPersistenceFactory = Callable[[], "FlowPersistence"]

_factory: FlowPersistenceFactory | None = None


def set_flow_persistence_factory(factory: FlowPersistenceFactory | None) -> None:
    """Replace the process-wide default flow persistence factory.

    Intended for one-time setup at startup. Pass ``None`` to restore the
    built-in ``SQLiteFlowPersistence``. Only affects flows that fall back to
    the default; an explicit ``persistence=`` instance always wins.

    The default is resolved at each fall-back site (``@persist`` and the
    runtime's pause/resume paths), so the factory may be called more than once
    for a single flow. Return instances backed by shared durable state (or a
    singleton) so state saved on one call is visible to the next -- the
    built-in SQLite default satisfies this by sharing one on-disk file.
    """
    global _factory
    _factory = factory


def default_flow_persistence() -> FlowPersistence:
    """Build the default flow persistence backend.

    Returns the result of the registered factory if one is set, otherwise a
    built-in :class:`~crewai.flow.persistence.sqlite.SQLiteFlowPersistence`.
    """
    factory = _factory
    if factory is not None:
        return factory()

    from crewai.flow.persistence.sqlite import SQLiteFlowPersistence

    return SQLiteFlowPersistence()
