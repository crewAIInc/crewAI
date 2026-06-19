"""Pytest configuration for crewAI workspace."""

import base64
from collections.abc import Generator
import gzip
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from dotenv import load_dotenv
import pytest


def _patch_vcrpy_aiohttp_compat() -> None:
    """Keep vcrpy's aiohttp stub working under aiohttp 3.14.0.

    aiohttp 3.14.0 (pulled in to fix GHSA-jg22-mg44-37j8 and GHSA-hg6j-4rv6-33pg):
      * removed ``aiohttp.streams.AsyncStreamReaderMixin`` (folded into ``StreamReader``),
        which vcrpy's ``MockStream`` still subclasses -- vcr's patch machinery then raises
        ``AttributeError`` at collection time; and
      * added a required ``stream_writer`` keyword-only arg to ``ClientResponse.__init__``,
        which vcrpy's ``MockClientResponse`` does not pass -- raising ``TypeError`` at
        cassette playback.

    Restore the mixin, then rebuild ``MockClientResponse``'s ``super().__init__`` call from
    the live ``ClientResponse`` signature (defaulting every required keyword-only arg to
    ``None``, mirroring vcrpy's original call) so it also survives future aiohttp additions.
    """
    import asyncio
    import inspect

    from aiohttp import streams
    from aiohttp.client_reqrep import ClientResponse

    if not hasattr(streams, "AsyncStreamReaderMixin"):

        class AsyncStreamReaderMixin:
            __slots__ = ()

            def __aiter__(self) -> streams.AsyncStreamIterator[bytes]:
                return streams.AsyncStreamIterator(self.readline)  # type: ignore[attr-defined]

            def iter_chunked(self, n: int) -> streams.AsyncStreamIterator[bytes]:
                return streams.AsyncStreamIterator(lambda: self.read(n))  # type: ignore[attr-defined]

            def iter_any(self) -> streams.AsyncStreamIterator[bytes]:
                return streams.AsyncStreamIterator(self.readany)  # type: ignore[attr-defined]

            def iter_chunks(self) -> streams.ChunkTupleAsyncStreamIterator:
                return streams.ChunkTupleAsyncStreamIterator(self)  # type: ignore[arg-type]

        streams.AsyncStreamReaderMixin = AsyncStreamReaderMixin  # type: ignore[attr-defined]

    # Importing the stub builds MockStream/MockClientResponse, so it must run after the
    # mixin is restored above.
    import vcr.stubs.aiohttp_stubs as aiohttp_stubs  # type: ignore[import-untyped]

    if getattr(aiohttp_stubs.MockClientResponse, "_crewai_aiohttp_patched", False):
        return

    keyword_only = [
        name
        for name, param in inspect.signature(ClientResponse.__init__).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    ]

    class _NullStreamWriter:
        # aiohttp 3.14.0 reads stream_writer.output_size in the "request already
        # sent" branch (writer is None), so None is not enough -- supply a stub.
        output_size = 0

    fallback_loop: list[asyncio.AbstractEventLoop] = []

    def _resolve_loop() -> asyncio.AbstractEventLoop:
        # MockClientResponse is normally built inside aiohttp's running loop, so
        # prefer that. In a sync context there is no running loop; avoid
        # asyncio.get_event_loop(), which on 3.12+ emits a DeprecationWarning
        # (and can RuntimeError) when no current loop is set. Use one cached
        # loop instead -- the mock only stores it and calls loop.get_debug().
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if not fallback_loop:
                fallback_loop.append(asyncio.new_event_loop())
            return fallback_loop[0]

    def _mock_client_response_init(
        self: Any, method: str, url: Any, request_info: Any = None
    ) -> None:
        kwargs: dict[str, Any] = dict.fromkeys(keyword_only)
        kwargs["request_info"] = request_info
        if "loop" in kwargs:
            kwargs["loop"] = _resolve_loop()
        if "stream_writer" in kwargs:
            kwargs["stream_writer"] = _NullStreamWriter()
        ClientResponse.__init__(self, method, url, **kwargs)

    aiohttp_stubs.MockClientResponse.__init__ = _mock_client_response_init
    aiohttp_stubs.MockClientResponse._crewai_aiohttp_patched = True


_patch_vcrpy_aiohttp_compat()

from vcr.request import Request  # type: ignore[import-untyped]  # noqa: E402


try:
    import vcr.stubs.httpx_stubs as httpx_stubs  # type: ignore[import-untyped]
except ModuleNotFoundError:
    import vcr.stubs.httpcore_stubs as httpx_stubs  # type: ignore[import-untyped]


env_test_path = Path(__file__).parent / ".env.test"

load_dotenv(env_test_path, override=False)
load_dotenv(override=False)

BEDROCK_HOST_PLACEHOLDER = "bedrock-runtime.vcr.amazonaws.com"
_BEDROCK_HOST_RE = re.compile(r"^bedrock-runtime\.[a-z0-9-]+\.amazonaws\.com$")


def _normalize_bedrock_host(host: str) -> str:
    if _BEDROCK_HOST_RE.match(host):
        return BEDROCK_HOST_PLACEHOLDER
    return host


def bedrock_host_matcher(r1: Request, r2: Request) -> bool:  # type: ignore[no-any-unimported]
    """Match Bedrock requests across AWS regions (CI uses us-east-1, local may use us-west-2)."""
    return _normalize_bedrock_host(r1.host or "") == _normalize_bedrock_host(
        r2.host or ""
    )


def _patched_make_vcr_request(httpx_request: Any, **kwargs: Any) -> Any:
    """Patched version of VCR's _make_vcr_request that handles binary content.

    The original implementation fails on binary request bodies (like file uploads)
    because it assumes all content can be decoded as UTF-8.
    """
    raw_body = httpx_request.read()
    try:
        body = raw_body.decode("utf-8")
    except UnicodeDecodeError:
        body = base64.b64encode(raw_body).decode("ascii")
    uri = str(httpx_request.url)
    headers = dict(httpx_request.headers)
    return Request(httpx_request.method, uri, body, headers)


httpx_stubs._make_vcr_request = _patched_make_vcr_request


# Patch the response-side of VCR to fix httpx.ResponseNotRead errors.
# VCR's _from_serialized_response mocks httpx.Response.read(), which prevents
# the response's internal _content attribute from being properly initialized.
# When OpenAI's client (using with_raw_response) accesses response.content,
# httpx raises ResponseNotRead because read() was never actually called.
# This patch ensures _content is explicitly set after response creation.
_original_from_serialized_response = getattr(
    httpx_stubs, "_from_serialized_response", None
)

if _original_from_serialized_response is not None:
    _from_serialized: Any = _original_from_serialized_response

    def _patched_from_serialized_response(
        request: Any, serialized_response: Any, history: Any = None
    ) -> Any:
        """Patched version that ensures response._content is properly set."""
        response = _from_serialized(request, serialized_response, history)
        # Explicitly set _content to avoid ResponseNotRead errors
        # The content was passed to the constructor but the mocked read() prevents
        # proper initialization of the internal state
        body_content = serialized_response.get("body", {}).get("string", b"")
        if isinstance(body_content, str):
            body_content = body_content.encode("utf-8")
        response._content = body_content
        return response

    httpx_stubs._from_serialized_response = _patched_from_serialized_response


@pytest.fixture(autouse=True, scope="function")
def cleanup_event_handlers() -> Generator[None, Any, None]:
    """Clean up event bus handlers after each test to prevent test pollution."""
    yield

    try:
        from crewai.events.event_bus import crewai_event_bus

        with crewai_event_bus._rwlock.w_locked():
            crewai_event_bus._sync_handlers.clear()
            crewai_event_bus._async_handlers.clear()
    except Exception:  # noqa: S110
        pass


@pytest.fixture(autouse=True, scope="function")
def reset_event_state() -> None:
    """Reset event system state before each test for isolation."""
    from crewai.events.base_events import reset_emission_counter
    from crewai.events.event_context import (
        EventContextConfig,
        _event_context_config,
        _event_id_stack,
    )

    reset_emission_counter()
    _event_id_stack.set(())
    _event_context_config.set(EventContextConfig())


@pytest.fixture(autouse=True, scope="function")
def setup_test_environment() -> Generator[None, Any, None]:
    """Setup test environment for crewAI workspace."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_dir = Path(temp_dir) / "crewai_test_storage"
        storage_dir.mkdir(parents=True, exist_ok=True)

        if not storage_dir.exists() or not storage_dir.is_dir():
            raise RuntimeError(
                f"Failed to create test storage directory: {storage_dir}"
            )

        try:
            test_file = storage_dir / ".permissions_test"
            test_file.touch()
            test_file.unlink()
        except (OSError, IOError) as e:
            raise RuntimeError(
                f"Test storage directory {storage_dir} is not writable: {e}"
            ) from e

        os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)
        os.environ["CREWAI_TESTING"] = "true"

        try:
            yield
        finally:
            os.environ.pop("CREWAI_TESTING", "true")
            os.environ.pop("CREWAI_STORAGE_DIR", None)
            os.environ.pop("CREWAI_DISABLE_TELEMETRY", "true")
            os.environ.pop("OTEL_SDK_DISABLED", "true")
            os.environ.pop("OPENAI_BASE_URL", "https://api.openai.com/v1")
            os.environ.pop("OPENAI_API_BASE", "https://api.openai.com/v1")


HEADERS_TO_FILTER = {
    "authorization": "AUTHORIZATION-XXX",
    "content-security-policy": "CSP-FILTERED",
    "cookie": "COOKIE-XXX",
    "set-cookie": "SET-COOKIE-XXX",
    "permissions-policy": "PERMISSIONS-POLICY-XXX",
    "referrer-policy": "REFERRER-POLICY-XXX",
    "strict-transport-security": "STS-XXX",
    "x-content-type-options": "X-CONTENT-TYPE-XXX",
    "x-frame-options": "X-FRAME-OPTIONS-XXX",
    "x-permitted-cross-domain-policies": "X-PERMITTED-XXX",
    "x-request-id": "X-REQUEST-ID-XXX",
    "x-runtime": "X-RUNTIME-XXX",
    "x-xss-protection": "X-XSS-PROTECTION-XXX",
    "x-stainless-arch": "X-STAINLESS-ARCH-XXX",
    "x-stainless-os": "X-STAINLESS-OS-XXX",
    "x-stainless-read-timeout": "X-STAINLESS-READ-TIMEOUT-XXX",
    "cf-ray": "CF-RAY-XXX",
    "etag": "ETAG-XXX",
    "Strict-Transport-Security": "STS-XXX",
    "access-control-expose-headers": "ACCESS-CONTROL-XXX",
    "openai-organization": "OPENAI-ORG-XXX",
    "openai-project": "OPENAI-PROJECT-XXX",
    "x-ratelimit-limit-requests": "X-RATELIMIT-LIMIT-REQUESTS-XXX",
    "x-ratelimit-limit-tokens": "X-RATELIMIT-LIMIT-TOKENS-XXX",
    "x-ratelimit-remaining-requests": "X-RATELIMIT-REMAINING-REQUESTS-XXX",
    "x-ratelimit-remaining-tokens": "X-RATELIMIT-REMAINING-TOKENS-XXX",
    "x-ratelimit-reset-requests": "X-RATELIMIT-RESET-REQUESTS-XXX",
    "x-ratelimit-reset-tokens": "X-RATELIMIT-RESET-TOKENS-XXX",
    "x-goog-api-key": "X-GOOG-API-KEY-XXX",
    "api-key": "X-API-KEY-XXX",
    "User-Agent": "X-USER-AGENT-XXX",
    "apim-request-id:": "X-API-CLIENT-REQUEST-ID-XXX",
    "azureml-model-session": "AZUREML-MODEL-SESSION-XXX",
    "x-ms-client-request-id": "X-MS-CLIENT-REQUEST-ID-XXX",
    "x-ms-region": "X-MS-REGION-XXX",
    "apim-request-id": "APIM-REQUEST-ID-XXX",
    "x-api-key": "X-API-KEY-XXX",
    "anthropic-organization-id": "ANTHROPIC-ORGANIZATION-ID-XXX",
    "request-id": "REQUEST-ID-XXX",
    "anthropic-ratelimit-input-tokens-limit": "ANTHROPIC-RATELIMIT-INPUT-TOKENS-LIMIT-XXX",
    "anthropic-ratelimit-input-tokens-remaining": "ANTHROPIC-RATELIMIT-INPUT-TOKENS-REMAINING-XXX",
    "anthropic-ratelimit-input-tokens-reset": "ANTHROPIC-RATELIMIT-INPUT-TOKENS-RESET-XXX",
    "anthropic-ratelimit-output-tokens-limit": "ANTHROPIC-RATELIMIT-OUTPUT-TOKENS-LIMIT-XXX",
    "anthropic-ratelimit-output-tokens-remaining": "ANTHROPIC-RATELIMIT-OUTPUT-TOKENS-REMAINING-XXX",
    "anthropic-ratelimit-output-tokens-reset": "ANTHROPIC-RATELIMIT-OUTPUT-TOKENS-RESET-XXX",
    "anthropic-ratelimit-tokens-limit": "ANTHROPIC-RATELIMIT-TOKENS-LIMIT-XXX",
    "anthropic-ratelimit-tokens-remaining": "ANTHROPIC-RATELIMIT-TOKENS-REMAINING-XXX",
    "anthropic-ratelimit-tokens-reset": "ANTHROPIC-RATELIMIT-TOKENS-RESET-XXX",
    "x-amz-date": "X-AMZ-DATE-XXX",
    "x-amz-security-token": "X-AMZ-SECURITY-TOKEN-XXX",
    "amz-sdk-invocation-id": "AMZ-SDK-INVOCATION-ID-XXX",
    "accept-encoding": "ACCEPT-ENCODING-XXX",
    "x-amzn-requestid": "X-AMZN-REQUESTID-XXX",
    "x-amzn-RequestId": "X-AMZN-REQUESTID-XXX",
    "x-a2a-notification-token": "X-A2A-NOTIFICATION-TOKEN-XXX",
    "x-a2a-version": "X-A2A-VERSION-XXX",
}


def _filter_request_headers(request: Request) -> Request:  # type: ignore[no-any-unimported]
    """Filter sensitive headers from request before recording."""
    for header_name, replacement in HEADERS_TO_FILTER.items():
        for variant in [header_name, header_name.upper(), header_name.title()]:
            if variant in request.headers:
                request.headers[variant] = [replacement]

    request.method = request.method.upper()

    # Normalize Azure OpenAI endpoints to a consistent placeholder for cassette matching.
    if request.host and request.host.endswith(".openai.azure.com"):
        original_host = request.host
        placeholder_host = "fake-azure-endpoint.openai.azure.com"
        request.uri = request.uri.replace(original_host, placeholder_host)

    # Normalize Bedrock regional endpoints so cassettes work in any AWS region.
    if request.host and _BEDROCK_HOST_RE.match(request.host):
        request.uri = request.uri.replace(request.host, BEDROCK_HOST_PLACEHOLDER)

    return request


def _filter_response_headers(response: dict[str, Any]) -> dict[str, Any] | None:
    """Filter sensitive headers from response before recording.

    Returns None to skip recording responses with empty bodies. This handles
    duplicate recordings caused by OpenAI's stainless client using
    with_raw_response which triggers httpx to re-read the consumed stream.
    """
    body = response.get("body", {}).get("string", "")
    headers = response.get("headers", {})
    content_length = headers.get("content-length", headers.get("Content-Length", []))

    if body == "" or body == b"" or content_length == ["0"]:
        return None

    status_code = response.get("status", {}).get("code")
    if isinstance(status_code, int) and status_code >= 400:
        # Avoid persisting auth/model errors when re-recording without valid AWS creds.
        return None

    for encoding_header in ["Content-Encoding", "content-encoding"]:
        if encoding_header in headers:
            encoding = headers.pop(encoding_header)
            if encoding and encoding[0] == "gzip":
                body = response.get("body", {}).get("string", b"")
                if isinstance(body, bytes) and body.startswith(b"\x1f\x8b"):
                    response["body"]["string"] = gzip.decompress(body).decode("utf-8")

    for header_name, replacement in HEADERS_TO_FILTER.items():
        for variant in [header_name, header_name.upper(), header_name.title()]:
            if variant in headers:
                headers[variant] = [replacement]
    return response


@pytest.fixture(scope="module")
def vcr_cassette_dir(request: Any) -> str:
    """Generate cassette directory path based on test module location.

    Organizes cassettes to mirror test directory structure within each package:
    lib/crewai/tests/llms/google/test_google.py -> lib/crewai/tests/cassettes/llms/google/
    lib/crewai-tools/tests/tools/test_search.py -> lib/crewai-tools/tests/cassettes/tools/
    """
    test_file = Path(request.fspath)

    for parent in test_file.parents:
        if (
            parent.name
            in ("crewai", "crewai-tools", "crewai-files", "cli", "crewai-core")
            and parent.parent.name == "lib"
        ):
            package_root = parent
            break
    else:
        package_root = test_file.parent

    tests_root = package_root / "tests"
    test_dir = test_file.parent

    if test_dir != tests_root:
        relative_path = test_dir.relative_to(tests_root)
        cassette_dir = tests_root / "cassettes" / relative_path
    else:
        cassette_dir = tests_root / "cassettes"

    cassette_dir.mkdir(parents=True, exist_ok=True)

    return str(cassette_dir)


def pytest_recording_configure(vcr: Any, config: Any) -> None:
    """Register custom VCR matchers for each test cassette session."""
    vcr.register_matcher("bedrock_host", bedrock_host_matcher)


@pytest.fixture(scope="module")
def vcr_config(vcr_cassette_dir: str) -> dict[str, Any]:
    """Configure VCR with organized cassette storage."""
    config = {
        "cassette_library_dir": vcr_cassette_dir,
        "record_mode": os.getenv("PYTEST_VCR_RECORD_MODE", "once"),
        "filter_headers": [(k, v) for k, v in HEADERS_TO_FILTER.items()],
        "before_record_request": _filter_request_headers,
        "before_record_response": _filter_response_headers,
        "filter_query_parameters": ["key"],
        "match_on": ["method", "scheme", "host", "port", "path"],
    }

    if os.getenv("GITHUB_ACTIONS") == "true":
        config["record_mode"] = "none"

    return config
