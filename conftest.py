"""Pytest configuration for crewAI workspace."""

import os
from pathlib import Path
import tempfile
from typing import Any

import pytest


# Load .env.test for consistent test environment
# env_test_path = Path(__file__).parent / ".env.test"
# load_dotenv(env_test_path, override=True)
# load_dotenv(override=True)


@pytest.fixture(autouse=True, scope="function")
def setup_test_environment() -> None:  # type: ignore
    """Set up test environment with a temporary directory for SQLite storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the directory with proper permissions
        storage_dir = Path(temp_dir) / "crewai_test_storage"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Validate that the directory was created successfully
        if not storage_dir.exists() or not storage_dir.is_dir():
            raise RuntimeError(
                f"Failed to create test storage directory: {storage_dir}"
            )

        # Verify directory permissions
        try:
            # Try to create a test file to verify write permissions
            test_file = storage_dir / ".permissions_test"
            test_file.touch()
            test_file.unlink()
        except (OSError, IOError) as e:
            raise RuntimeError(
                f"Test storage directory {storage_dir} is not writable: {e}"
            ) from e

        os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)
        os.environ["CREWAI_TESTING"] = "true"
        yield

        os.environ.pop("CREWAI_TESTING", None)
        # Cleanup is handled automatically when tempfile context exits


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
}


def _filter_response_headers(response) -> dict[str, Any]:  # type: ignore
    """Filter sensitive headers from response before recording."""
    for header_name, replacement in HEADERS_TO_FILTER.items():
        for variant in [header_name, header_name.upper(), header_name.title()]:
            if variant in response["headers"]:
                response["headers"][variant] = [replacement]
    return response  # type: ignore


@pytest.fixture(scope="module")
def vcr_cassette_dir(request: Any) -> str:
    """Generate cassette directory path based on test module location.

    Organizes cassettes to mirror test directory structure within each package:
    lib/crewai/tests/llms/google/test_google.py -> lib/crewai/tests/cassettes/llms/google/
    lib/crewai-tools/tests/tools/test_search.py -> lib/crewai-tools/tests/cassettes/tools/
    """
    test_file = Path(request.fspath)

    # Find the package root (lib/crewai or lib/crewai-tools)
    for parent in test_file.parents:
        if parent.name in ("crewai", "crewai-tools") and parent.parent.name == "lib":
            package_root = parent
            break
    else:
        # Fallback to old behavior if we can't find package root
        package_root = test_file.parent

    tests_root = package_root / "tests"
    test_dir = test_file.parent

    # Get relative path from tests root
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
        "record_mode": os.getenv("PYTEST_VCR_RECORD_MODE") or "once",
        "filter_headers": [(k, v) for k, v in HEADERS_TO_FILTER.items()],
        "before_record_response": _filter_response_headers,
    }

    if os.getenv("GITHUB_ACTIONS") == "true":
        config["record_mode"] = "none"

    return config
