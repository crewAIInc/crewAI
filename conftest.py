"""Pytest configuration for crewAI workspace."""

from collections.abc import Generator
import os
from pathlib import Path
import tempfile
from typing import Any

from dotenv import load_dotenv
import pytest
from vcr.request import Request  # type: ignore[import-untyped]


env_test_path = Path(__file__).parent / ".env.test"
load_dotenv(env_test_path, override=True)
load_dotenv(override=True)


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
    "amz-sdk-invocation-id": "AMZ-SDK-INVOCATION-ID-XXX",
    "accept-encoding": "ACCEPT-ENCODING-XXX",
    "x-amzn-requestid": "X-AMZN-REQUESTID-XXX",
    "x-amzn-RequestId": "X-AMZN-REQUESTID-XXX",
}


def _filter_request_headers(request: Request) -> Request:  # type: ignore[no-any-unimported]
    """Filter sensitive headers from request before recording."""
    for header_name, replacement in HEADERS_TO_FILTER.items():
        for variant in [header_name, header_name.upper(), header_name.title()]:
            if variant in request.headers:
                request.headers[variant] = [replacement]

    request.method = request.method.upper()
    return request


def _filter_response_headers(response: dict[str, Any]) -> dict[str, Any]:
    """Filter sensitive headers from response before recording."""
    for header_name, replacement in HEADERS_TO_FILTER.items():
        for variant in [header_name, header_name.upper(), header_name.title()]:
            if variant in response["headers"]:
                response["headers"][variant] = [replacement]
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
        if parent.name in ("crewai", "crewai-tools") and parent.parent.name == "lib":
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
