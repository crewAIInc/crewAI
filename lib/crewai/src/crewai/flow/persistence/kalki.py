"""Kalki-based implementation of flow state persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

from crewai.flow.persistence.base import FlowPersistence


if TYPE_CHECKING:
    from crewai.flow.async_feedback.types import PendingFeedbackContext


_STATE_KIND = "flow_state"
_PENDING_FEEDBACK_KIND = "pending_feedback"
_PENDING_FEEDBACK_CLEARED_KIND = "pending_feedback_cleared"


def _utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_timestamp(value: Any) -> datetime:
    """Parse timestamp values from payloads/log records."""
    if isinstance(value, datetime):
        dt = value
    elif value is None:
        dt = datetime.fromtimestamp(0, tz=timezone.utc)
    else:
        raw = str(value)
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            dt = datetime.fromtimestamp(0, tz=timezone.utc)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class KalkiClient(Protocol):
    """Client protocol used by KalkiFlowPersistence."""

    def store_log(
        self,
        *,
        agent_id: str,
        session_id: str,
        conversation_log: str,
        summary: str,
    ) -> None:
        """Store a single log entry."""

    def query_logs(
        self,
        *,
        caller_agent_id: str,
        query: str,
        session_id: str,
        agent_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Query log entries."""


class _GrpcKalkiClient:
    """gRPC adapter around Kalki's `KalkiService` API."""

    _PROTO_IMPORT_CANDIDATES: tuple[tuple[str, str], ...] = (
        ("kalki_pb2", "kalki_pb2_grpc"),
        ("kalki.proto.kalki_pb2", "kalki.proto.kalki_pb2_grpc"),
        ("kalki.kalki_pb2", "kalki.kalki_pb2_grpc"),
    )

    def __init__(self, target: str, timeout_seconds: float = 5.0) -> None:
        self._target = target
        self._timeout_seconds = timeout_seconds
        self._pb2: Any = None
        self._stub: Any = None

    def _ensure_stub(self) -> None:
        if self._stub is not None and self._pb2 is not None:
            return

        try:
            import grpc
        except ImportError as exc:  # pragma: no cover - tested by behavior
            raise ImportError(
                "KalkiFlowPersistence requires gRPC support. Install `grpcio` and "
                "Kalki protobuf stubs (kalki_pb2 / kalki_pb2_grpc)."
            ) from exc

        pb2_module: Any = None
        pb2_grpc_module: Any = None
        for pb2_path, pb2_grpc_path in self._PROTO_IMPORT_CANDIDATES:
            try:
                pb2_module = importlib.import_module(pb2_path)
                pb2_grpc_module = importlib.import_module(pb2_grpc_path)
                break
            except ImportError:
                continue

        if pb2_module is None or pb2_grpc_module is None:
            raise ImportError(
                "Unable to import Kalki protobuf stubs. Expected one of: "
                "kalki_pb2/kalki_pb2_grpc, "
                "kalki.proto.kalki_pb2/kalki.proto.kalki_pb2_grpc, "
                "or kalki.kalki_pb2/kalki.kalki_pb2_grpc."
            )

        channel = grpc.insecure_channel(self._target)
        self._pb2 = pb2_module
        self._stub = pb2_grpc_module.KalkiServiceStub(channel)

    def store_log(
        self,
        *,
        agent_id: str,
        session_id: str,
        conversation_log: str,
        summary: str,
    ) -> None:
        self._ensure_stub()
        request = self._pb2.StoreLogRequest(
            agent_id=agent_id,
            session_id=session_id,
            conversation_log=conversation_log,
            summary=summary,
        )
        self._stub.StoreLog(request, timeout=self._timeout_seconds)

    def query_logs(
        self,
        *,
        caller_agent_id: str,
        query: str,
        session_id: str,
        agent_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        self._ensure_stub()
        request = self._pb2.QueryLogsRequest(
            caller_agent_id=caller_agent_id,
            query=query,
            session_id=session_id,
            agent_id=agent_id,
            limit=limit,
        )
        response = self._stub.QueryLogs(request, timeout=self._timeout_seconds)
        logs = getattr(response, "logs", [])
        return [
            {
                "agent_id": str(getattr(log, "agent_id", "")),
                "session_id": str(getattr(log, "session_id", "")),
                "conversation_log": str(getattr(log, "conversation_log", "")),
                "summary": str(getattr(log, "summary", "")),
                "timestamp": str(getattr(log, "timestamp", "")),
            }
            for log in logs
        ]


class KalkiFlowPersistence(FlowPersistence):
    """Kalki-backed flow state persistence using Kalki's gRPC API."""

    def __init__(
        self,
        target: str = "127.0.0.1:50051",
        *,
        agent_id: str = "crewai-flow",
        query_limit: int = 200,
        query_text: str = "crewai flow checkpoint",
        client: KalkiClient | None = None,
    ) -> None:
        """Initialize Kalki persistence.

        Args:
            target: Kalki gRPC endpoint, e.g. ``127.0.0.1:50051``.
            agent_id: Agent identifier used for all CrewAI checkpoint writes.
            query_limit: Maximum logs to fetch per recovery query.
            query_text: Query text passed to Kalki QueryLogs.
            client: Optional custom client implementing the KalkiClient protocol.
        """
        if query_limit < 1:
            raise ValueError("query_limit must be >= 1")

        self._target = target
        self._agent_id = agent_id
        self._query_limit = query_limit
        self._query_text = query_text
        self._client: KalkiClient = client or _GrpcKalkiClient(target=target)
        self.init_db()

    def init_db(self) -> None:
        """Initialize persistence backend.

        Kalki manages storage internally, so no schema setup is required.
        """

    @staticmethod
    def _coerce_state_dict(state_data: dict[str, Any] | BaseModel) -> dict[str, Any]:
        if isinstance(state_data, BaseModel):
            return state_data.model_dump()
        if isinstance(state_data, dict):
            return state_data
        raise ValueError(
            f"state_data must be either a Pydantic BaseModel or dict, got {type(state_data)}"
        )

    def _write_payload(
        self,
        *,
        flow_uuid: str,
        method_name: str,
        kind: str,
        state: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema": "crewai.flow.persistence.v1",
            "kind": kind,
            "flow_uuid": flow_uuid,
            "method_name": method_name,
            "timestamp": _utc_now_iso(),
            "state": state,
        }
        if context is not None:
            payload["context"] = context

        self._client.store_log(
            agent_id=self._agent_id,
            session_id=flow_uuid,
            conversation_log=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            summary=f"crewai.{kind}.{method_name}",
        )

    def _query_payloads(self, flow_uuid: str) -> list[dict[str, Any]]:
        logs = self._client.query_logs(
            caller_agent_id=self._agent_id,
            query=self._query_text,
            session_id=flow_uuid,
            agent_id=self._agent_id,
            limit=self._query_limit,
        )

        payloads: list[dict[str, Any]] = []
        for log in logs:
            raw_payload = log.get("conversation_log")
            if not isinstance(raw_payload, str):
                continue
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("flow_uuid") != flow_uuid:
                continue

            enriched = dict(payload)
            enriched["_resolved_timestamp"] = _parse_iso_timestamp(
                payload.get("timestamp") or log.get("timestamp")
            )
            payloads.append(enriched)

        payloads.sort(key=lambda item: item["_resolved_timestamp"])
        return payloads

    @staticmethod
    def _latest_by_kind(
        payloads: list[dict[str, Any]],
        *,
        kinds: set[str],
    ) -> dict[str, Any] | None:
        filtered = [p for p in payloads if p.get("kind") in kinds]
        if not filtered:
            return None
        return max(filtered, key=lambda item: item["_resolved_timestamp"])

    def save_state(
        self,
        flow_uuid: str,
        method_name: str,
        state_data: dict[str, Any] | BaseModel,
    ) -> None:
        state = self._coerce_state_dict(state_data)
        self._write_payload(
            flow_uuid=flow_uuid,
            method_name=method_name,
            kind=_STATE_KIND,
            state=state,
        )

    def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
        payloads = self._query_payloads(flow_uuid)
        latest = self._latest_by_kind(payloads, kinds={_STATE_KIND})
        if latest is None:
            return None
        state = latest.get("state")
        if not isinstance(state, dict):
            return None
        return state

    def save_pending_feedback(
        self,
        flow_uuid: str,
        context: PendingFeedbackContext,
        state_data: dict[str, Any] | BaseModel,
    ) -> None:
        state = self._coerce_state_dict(state_data)

        # Keep parity with SQLite behavior: persist latest regular state too.
        self.save_state(flow_uuid, context.method_name, state)

        self._write_payload(
            flow_uuid=flow_uuid,
            method_name=context.method_name,
            kind=_PENDING_FEEDBACK_KIND,
            state=state,
            context=context.to_dict(),
        )

    def load_pending_feedback(
        self,
        flow_uuid: str,
    ) -> tuple[dict[str, Any], PendingFeedbackContext] | None:
        from crewai.flow.async_feedback.types import PendingFeedbackContext

        payloads = self._query_payloads(flow_uuid)
        latest = self._latest_by_kind(
            payloads,
            kinds={_PENDING_FEEDBACK_KIND, _PENDING_FEEDBACK_CLEARED_KIND},
        )
        if latest is None or latest.get("kind") != _PENDING_FEEDBACK_KIND:
            return None

        state = latest.get("state")
        context_dict = latest.get("context")
        if not isinstance(state, dict) or not isinstance(context_dict, dict):
            return None

        context = PendingFeedbackContext.from_dict(context_dict)
        return (state, context)

    def clear_pending_feedback(self, flow_uuid: str) -> None:
        # Kalki's public API currently does not expose delete-by-key. We model
        # clearing as an append-only tombstone marker.
        self._write_payload(
            flow_uuid=flow_uuid,
            method_name="clear_pending_feedback",
            kind=_PENDING_FEEDBACK_CLEARED_KIND,
            state={},
        )
