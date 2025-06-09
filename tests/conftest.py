# conftest.py
import os
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_result = load_dotenv(override=True)


@pytest.fixture(autouse=True)
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
            )

        # Set environment variable to point to the test storage directory
        os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)

        yield

        # Cleanup is handled automatically when tempfile context exits


def filter_stream_parameter(request):
    """Filter out stream parameter from request body to maintain VCR compatibility."""
    if hasattr(request, 'body') and request.body:
        try:
            import json
            # Check if it's likely JSON content
            if hasattr(request, 'headers') and request.headers:
                content_type = request.headers.get('content-type', '').lower()
                if 'application/json' not in content_type:
                    return request

            # Try to decode and parse the body
            if isinstance(request.body, bytes):
                try:
                    body_str = request.body.decode('utf-8')
                except UnicodeDecodeError:
                    # If we can't decode it, it's probably binary data, leave it as is
                    return request
            else:
                body_str = request.body

            body = json.loads(body_str)
            # Remove stream parameter to match original recordings
            if 'stream' in body:
                body.pop('stream')
            request.body = json.dumps(body).encode() if isinstance(request.body, bytes) else json.dumps(body)
        except (json.JSONDecodeError, AttributeError, TypeError):
            # If we can't parse the body, leave it as is
            pass
    return request


@pytest.fixture(autouse=True)
def configure_litellm_for_testing():
    """Configure litellm to work better with VCR cassettes."""
    import litellm

    # Disable litellm's internal streaming optimizations that might conflict with VCR
    original_drop_params = litellm.drop_params
    litellm.drop_params = True

    yield

    # Restore original setting
    litellm.drop_params = original_drop_params


@pytest.fixture(scope="module")
def vcr_config(request) -> dict:
    return {
        "cassette_library_dir": "tests/cassettes",
        "record_mode": "new_episodes",
        "filter_headers": [("authorization", "AUTHORIZATION-XXX")],
        "before_record_request": filter_stream_parameter,
    }
