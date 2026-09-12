"""MongoDB (Atlas) implementation of flow state persistence.

Mirrors :class:`crewai.flow.persistence.sqlite.SQLiteFlowPersistence`, using
collections instead of SQLite tables:

- ``flow_states``: append-only history, one document per saved state (mirrors
  SQLite's ``INSERT`` into an ``AUTOINCREMENT`` table). SQLite's autoincrement
  ``id`` is replaced by a server-assigned monotonic ``seq`` (see ``counters``),
  and the latest state is read back with ``WHERE flow_uuid=? ORDER BY seq DESC``
  — not by sorting on the client-generated ObjectId ``_id``. A compound
  ``{flow_uuid: 1, seq: -1}`` index serves this latest-state query with an index
  scan and no blocking SORT (the Mongo-idiomatic equivalent of SQLite's
  single-column ``idx_flow_states_uuid``).
- ``pending_feedback``: one document per flow (unique on ``flow_uuid``), upserted
  to mirror SQLite's ``INSERT OR REPLACE``.
- ``counters``: internal bookkeeping. MongoDB has no autoincrement, so a single
  document holds an atomically ``$inc``-ed sequence that stands in for SQLite's
  ``AUTOINCREMENT id``, giving server-assigned ordering of appended states.

State is stored as a JSON string (``state_json``) via ``json.dumps`` exactly like
the SQLite backend, guaranteeing identical round-trips and avoiding BSON edge
cases.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr

from crewai.flow.persistence.base import FlowPersistence


if TYPE_CHECKING:
    from crewai.flow.async_feedback.types import PendingFeedbackContext


class MongoDbFlowPersistence(FlowPersistence):
    """MongoDB Atlas-based implementation of flow state persistence.

    The connection string comes from the ``MONGODB_CONNECTION_STRING`` env var
    (or the positional ``connection_string`` argument). The database defaults to
    ``flow_persistence`` and can be overridden with the optional
    ``MONGODB_DATABASE`` env var.

    Connection is lazy: constructing this object performs no I/O and requires
    neither a live database nor the ``MONGODB_CONNECTION_STRING`` env var. The
    Mongo client is created (and indexes ensured) on the first ``save_state`` /
    ``load_state`` call, and both the connection string and the
    ``MONGODB_DATABASE`` env var are resolved at that point. This keeps importing
    modules that merely *reference* the persistence (e.g. via
    ``@persist(MongoDbFlowPersistence())`` at class definition time) free of side
    effects, so tooling like ``crewai plot``, tests, and deploy builds do not
    need Atlas access — and env vars set after construction are still honored.

    Example:
        ```python
        persistence = MongoDbFlowPersistence()


        class MyFlow(Flow[MyState]):
            @start()
            @persist(persistence)
            def begin(self): ...
        ```
    """

    persistence_type: str = Field(default="MongoDbFlowPersistence")
    connection_string: str | None = Field(default=None)
    database_name: str | None = Field(default=None)
    states_collection: str = Field(default="flow_states")
    pending_collection: str = Field(default="pending_feedback")
    counters_collection: str = Field(default="counters")

    _client: Any = PrivateAttr(default=None)
    _db: Any = PrivateAttr(default=None)
    _indexes_ready: bool = PrivateAttr(default=False)

    def __init__(self, connection_string: str | None = None, /, **kwargs: Any) -> None:
        if connection_string is not None:
            kwargs["connection_string"] = connection_string
        super().__init__(**kwargs)

    def _resolve_connection_string(self) -> str:
        """Resolve the connection string, reading the env var at call time."""
        connection_string = self.connection_string or os.getenv(
            "MONGODB_CONNECTION_STRING"
        )
        if not connection_string:
            raise ValueError(
                "MongoDbFlowPersistence requires MONGODB_CONNECTION_STRING or a "
                "connection_string argument."
            )
        return connection_string

    def _resolve_database_name(self) -> str:
        """Resolve the database name, reading the env var at call time."""
        return self.database_name or os.getenv("MONGODB_DATABASE") or "flow_persistence"

    def _ensure_client(self) -> Any:
        """Create the Mongo client/database on first use and return the db."""
        if self._db is None:
            try:
                from pymongo import MongoClient
            except ImportError as exc:
                raise ImportError(
                    "MongoDbFlowPersistence requires the optional 'pymongo' package.\n"
                    "Install it with: uv add 'crewai[mongodb]'"
                ) from exc

            self._client = MongoClient(self._resolve_connection_string())
            self._db = self._client[self._resolve_database_name()]
        return self._db

    def _db_ready(self) -> Any:
        """Return the database, connecting and ensuring indexes on first use."""
        db = self._ensure_client()
        if not self._indexes_ready:
            self.init_db()
            self._indexes_ready = True
        return db

    def init_db(self) -> None:
        """Create the collections' indexes if they don't exist.

        Mirrors SQLite: ``flow_states`` keeps an append-only history (many states
        per flow), while ``pending_feedback`` holds at most one document per flow
        (unique).

        The ``flow_states`` index is compound ``{flow_uuid: 1, seq: -1}`` so
        ``load_state``'s latest-state query (``find({flow_uuid}).sort(seq desc)``)
        is served entirely from the index (IXSCAN, no blocking SORT); its
        ``flow_uuid`` prefix also covers plain ``flow_uuid`` lookups. This is the
        Mongo-idiomatic equivalent of SQLite's single-column
        ``idx_flow_states_uuid`` (SQLite gets latest-state ordering for free from
        rowid/``id``, which Mongo lacks), so behavior parity holds even though the
        index DDL differs.
        """
        db = self._ensure_client()
        db[self.states_collection].create_index([("flow_uuid", 1), ("seq", -1)])
        db[self.pending_collection].create_index("flow_uuid", unique=True)

    @staticmethod
    def _to_state_dict(state_data: dict[str, Any] | BaseModel) -> dict[str, Any]:
        """Convert state_data to a plain dict."""
        if isinstance(state_data, BaseModel):
            return state_data.model_dump(mode="json")
        if isinstance(state_data, dict):
            return state_data
        raise ValueError(
            f"state_data must be either a Pydantic BaseModel or dict, got {type(state_data)}"
        )

    def _next_sequence(self, name: str) -> int:
        """Return the next value of a server-assigned monotonic counter.

        MongoDB has no autoincrement, so this atomically ``$inc`` a per-name
        sequence document server-side, standing in for SQLite's
        ``AUTOINCREMENT id`` to order appended states reliably (independent of
        client-generated ObjectId monotonicity).
        """
        from pymongo import ReturnDocument

        doc = self._db_ready()[self.counters_collection].find_one_and_update(
            {"_id": name},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return int(doc["seq"])

    def save_state(
        self,
        flow_uuid: str,
        method_name: str,
        state_data: dict[str, Any] | BaseModel,
    ) -> None:
        """Persist the flow state after method completion.

        Appends a new state document (mirrors SQLite's ``INSERT``), tagged with
        a server-assigned ``seq`` so the most recent state can be selected by
        ordering on ``seq`` rather than the client-generated ObjectId ``_id``.
        """
        state_dict = self._to_state_dict(state_data)
        self._db_ready()[self.states_collection].insert_one(
            {
                "flow_uuid": flow_uuid,
                "method_name": method_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "state_json": json.dumps(state_dict),
                "seq": self._next_sequence(self.states_collection),
            }
        )

    def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
        """Load the most recent state for a given flow UUID."""
        doc = self._db_ready()[self.states_collection].find_one(
            {"flow_uuid": flow_uuid}, sort=[("seq", -1)]
        )
        if doc:
            result = json.loads(doc["state_json"])
            return result if isinstance(result, dict) else None
        return None

    def save_pending_feedback(
        self,
        flow_uuid: str,
        context: PendingFeedbackContext,
        state_data: dict[str, Any] | BaseModel,
    ) -> None:
        """Save state with a pending feedback marker (upsert per flow)."""
        state_dict = self._to_state_dict(state_data)

        # Mirror SQLite: record the state snapshot, then upsert the pending row.
        self.save_state(flow_uuid, context.method_name, state_data)

        self._db_ready()[self.pending_collection].replace_one(
            {"flow_uuid": flow_uuid},
            {
                "flow_uuid": flow_uuid,
                "context_json": json.dumps(context.to_dict()),
                "state_json": json.dumps(state_dict),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            upsert=True,
        )

    def load_pending_feedback(
        self,
        flow_uuid: str,
    ) -> tuple[dict[str, Any], PendingFeedbackContext] | None:
        """Load state and pending feedback context, if any."""
        # Import here to avoid circular imports.
        from crewai.flow.async_feedback.types import PendingFeedbackContext

        doc = self._db_ready()[self.pending_collection].find_one(
            {"flow_uuid": flow_uuid}
        )
        if doc:
            state_dict = json.loads(doc["state_json"])
            context = PendingFeedbackContext.from_dict(json.loads(doc["context_json"]))
            return (state_dict, context)
        return None

    def clear_pending_feedback(self, flow_uuid: str) -> None:
        """Clear the pending feedback marker after successful resume."""
        self._db_ready()[self.pending_collection].delete_one({"flow_uuid": flow_uuid})
