"""Exchange AMP credentials for short-lived, execution-bound collector grants."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
from threading import Lock
from urllib.parse import urlsplit
from uuid import UUID

from crewai_core.plus_api import PlusAPI
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult, SpanExporter

from crewai.auth.token import AuthError, get_auth_token
from crewai.context import get_platform_integration_token
from crewai.telemetry.tracing.session import MAX_EXPORT_BATCH_SIZE, otlp_exporter


logger = logging.getLogger(__name__)
# Wharf's default encoded OTLP request limit for both OSS grant tiers.
MAX_EXPORT_BODY_BYTES = 3_072_000


class TraceGrantError(Exception):
    """AMP could not authorize tracing; never downgrade a supplied credential."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def tracing_credential() -> str | None:
    """Resolve an explicit PAT, integration credential, or saved CLI login."""
    if token := os.getenv("CREWAI_USER_PAT"):
        return token
    if token := get_platform_integration_token():
        return token
    try:
        return get_auth_token()
    except AuthError:
        return None


@dataclass(frozen=True)
class TraceGrant:
    token: str = field(repr=False)
    collector_url: str
    execution_uuid: str
    expires_at: datetime


class TraceGrantClient:
    """Use credentials only with AMP; trace payloads never pass through here."""

    def __init__(self, amp_credential: str | None, *, base_url: str | None = None):
        if amp_credential is not None and not amp_credential.strip():
            raise TraceGrantError(
                "Authenticated tracing requires a nonblank credential", 401
            )
        self._tier = "authenticated" if amp_credential is not None else "ephemeral"
        self._api = PlusAPI(api_key=amp_credential, base_url=base_url)
        if amp_credential is None:
            # Anonymous grants must not carry a saved organization identifier.
            self._api.headers.pop("X-Crewai-Organization-Id", None)

    def create(self, execution_uuid: str) -> TraceGrant:
        execution_uuid = str(UUID(execution_uuid))
        try:
            response = self._api._make_request(
                "POST",
                f"{PlusAPI.TRACING_RESOURCE}/grants",
                json={"execution_uuid": execution_uuid},
                timeout=5,
            )
        except Exception as error:
            raise TraceGrantError(
                f"AMP trace grant request failed ({type(error).__name__})"
            ) from None
        if response.status_code != 200:
            raise TraceGrantError(
                f"AMP trace grant request failed (HTTP {response.status_code})",
                response.status_code,
            )
        try:
            data = response.json()
            endpoint = urlsplit(data["collector_url"])
            expiry = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
            if (
                data["tier"] != self._tier
                or str(UUID(data["execution_uuid"])) != execution_uuid
                or not isinstance(data["token"], str)
                or not data["token"].strip()
                or endpoint.scheme not in {"http", "https"}
                or not endpoint.hostname
                or endpoint.username is not None
                or endpoint.password is not None
                or expiry.tzinfo is None
                or expiry <= datetime.now(timezone.utc)
            ):
                raise ValueError("Invalid grant")
            return TraceGrant(
                data["token"], data["collector_url"], execution_uuid, expiry
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            raise TraceGrantError(
                f"AMP returned an invalid {self._tier} trace grant"
            ) from None


class GrantSpanExporter(SpanExporter):
    """Refresh a grant before exporting; delegate OTLP transport to the shared path."""

    def __init__(self, client: TraceGrantClient, grant: TraceGrant):
        self._client = client
        self._grant = grant
        self._lock = Lock()
        self._delegate = self._exporter(grant)

    @staticmethod
    def _exporter(grant: TraceGrant) -> SpanExporter:
        return otlp_exporter(
            grant.collector_url, {"Authorization": f"Bearer {grant.token}"}
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Bound each request; oversized single spans are skipped with FAILURE."""
        with self._lock:
            result = SpanExportResult.SUCCESS
            pending = [
                spans[offset : offset + MAX_EXPORT_BATCH_SIZE]
                for offset in reversed(range(0, len(spans), MAX_EXPORT_BATCH_SIZE))
            ]
            while pending:
                batch = pending.pop()
                size = encode_spans(batch).ByteSize()
                if size > MAX_EXPORT_BODY_BYTES:
                    if len(batch) == 1:
                        logger.warning(
                            "Skipping execution trace span: encoded size %d exceeds "
                            "Wharf's %d-byte request limit",
                            size,
                            MAX_EXPORT_BODY_BYTES,
                        )
                        result = SpanExportResult.FAILURE
                    else:
                        midpoint = len(batch) // 2
                        # Process the left half first to retain span order.
                        pending.extend((batch[midpoint:], batch[:midpoint]))
                    continue
                if (
                    self._grant.expires_at - datetime.now(timezone.utc)
                ).total_seconds() <= 30:
                    try:
                        grant = self._client.create(self._grant.execution_uuid)
                    except TraceGrantError as error:
                        logger.warning(
                            "Could not renew execution trace grant (HTTP %s)",
                            error.status_code,
                        )
                        return SpanExportResult.FAILURE
                    exporter = self._exporter(grant)
                    self._delegate.shutdown()
                    self._delegate, self._grant = exporter, grant
                if self._delegate.export(batch) != SpanExportResult.SUCCESS:
                    return SpanExportResult.FAILURE
                logger.info(
                    "Exported %d spans to Wharf for execution %s",
                    len(batch),
                    self._grant.execution_uuid,
                )
            return result

    def shutdown(self) -> None:
        with self._lock:
            self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)
