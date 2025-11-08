# conftest.py
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

# Load .env.test for consistent test environment (mimics CI)
env_test_path = Path(__file__).parent.parent.parent.parent / ".env.test"
load_dotenv(env_test_path, override=True)


@pytest.fixture(autouse=True, scope="function")
def setup_test_environment():
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


def _filter_response_headers(response):
    """Filter sensitive headers from response before recording."""
    for header_name, replacement in HEADERS_TO_FILTER.items():
      for variant in [header_name, header_name.upper(), header_name.title()]:
        if variant in response["headers"]:
          response["headers"][variant] = [replacement]
    return response


@pytest.fixture(scope="session", autouse=True)
def vcr_config(request) -> dict[str, Any]:
    config = {
        "cassette_library_dir": os.path.join(os.path.dirname(__file__), "cassettes"),
        "record_mode": os.getenv("PYTEST_VCR_RECORD_MODE") or "once",
        "filter_headers": [(k, v) for k, v in HEADERS_TO_FILTER.items()],
        "before_record_response": _filter_response_headers,
    }

    if os.getenv("GITHUB_ACTIONS") == "true":
        config["record_mode"] = "none"

    return config
