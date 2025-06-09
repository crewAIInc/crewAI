# conftest.py
import os
import tempfile
import threading
import time
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
def reset_litellm_state():
    """Reset LiteLLM global state before each test to ensure isolation."""
    import litellm

    # Save original state
    original_drop_params = getattr(litellm, 'drop_params', None)
    original_success_callback = getattr(litellm, 'success_callback', [])
    original_async_success_callback = getattr(litellm, '_async_success_callback', [])
    original_callbacks = getattr(litellm, 'callbacks', [])
    original_failure_callback = getattr(litellm, 'failure_callback', [])

    # Reset to clean state for the test
    litellm.drop_params = True
    litellm.success_callback = []
    litellm._async_success_callback = []
    litellm.callbacks = []
    litellm.failure_callback = []

    yield

    # Restore original state after test
    if original_drop_params is not None:
        litellm.drop_params = original_drop_params
    litellm.success_callback = original_success_callback
    litellm._async_success_callback = original_async_success_callback
    litellm.callbacks = original_callbacks
    litellm.failure_callback = original_failure_callback


@pytest.fixture(autouse=True)
def cleanup_background_threads():
    """Cleanup all background threads and resources after each test."""
    original_threads = set(threading.enumerate())

    yield

    # Force cleanup of various background resources
    try:
        # Cleanup RPM controllers first - they create Timer threads
        try:
            from crewai.utilities.rpm_controller import RPMController
            # Find any RPMController instances and stop their timers
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, RPMController):
                    if hasattr(obj, 'stop_rpm_counter'):
                        obj.stop_rpm_counter()
                    if hasattr(obj, '_timer') and obj._timer:
                        obj._timer.cancel()
                        obj._timer = None
                    if hasattr(obj, '_shutdown_flag'):
                        obj._shutdown_flag = True
        except Exception:
            pass

        # Cleanup all Timer threads explicitly
        try:
            for thread in threading.enumerate():
                if isinstance(thread, threading.Timer):
                    thread.cancel()
        except Exception:
            pass

        # Cleanup CrewAI telemetry
        try:
            from crewai.telemetry.telemetry import Telemetry
            telemetry = Telemetry()
            if hasattr(telemetry, 'provider') and telemetry.provider:
                if hasattr(telemetry.provider, 'shutdown'):
                    telemetry.provider.shutdown()
                if hasattr(telemetry.provider, 'force_flush'):
                    telemetry.provider.force_flush(timeout_millis=1000)
        except Exception:
            pass

        # Cleanup console formatters and spinners
        try:
            from crewai.utilities.events.utils.console_formatter import ConsoleFormatter, SimpleSpinner
            # Force stop any running spinners and formatters
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, ConsoleFormatter):
                    if hasattr(obj, 'cleanup'):
                        obj.cleanup()
                elif isinstance(obj, SimpleSpinner):
                    if hasattr(obj, 'stop'):
                        obj.stop()

            # Also check threads for spinners
            for thread in threading.enumerate():
                if hasattr(thread, '_target') and thread._target:
                    target_name = getattr(thread._target, '__name__', '')
                    if 'spin' in target_name.lower():
                        if hasattr(thread, '_stop_event'):
                            thread._stop_event.set()
        except Exception:
            pass

        # Cleanup HTTP clients and connection pools
        try:
            import httpx
            import gc
            # Force close any httpx clients
            for obj in gc.get_objects():
                if isinstance(obj, httpx.Client):
                    try:
                        obj.close()
                    except Exception:
                        pass
                elif isinstance(obj, httpx.AsyncClient):
                    try:
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                loop.create_task(obj.aclose())
                            else:
                                asyncio.run(obj.aclose())
                        except Exception:
                            pass
                    except Exception:
                        pass
        except Exception:
            pass

        # Cleanup any ThreadPoolExecutor instances
        try:
            import concurrent.futures
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, concurrent.futures.ThreadPoolExecutor):
                    try:
                        obj.shutdown(wait=False)
                    except Exception:
                        pass
        except Exception:
            pass

        # Force cleanup of any remaining daemon threads
        new_threads = set(threading.enumerate()) - original_threads
        for thread in new_threads:
            if thread.daemon and thread.is_alive():
                try:
                    # Try to stop the thread gracefully
                    if hasattr(thread, '_stop_event'):
                        thread._stop_event.set()
                    elif hasattr(thread, 'stop'):
                        thread.stop()

                    # Give it a moment to stop
                    thread.join(timeout=0.1)
                except Exception:
                    pass

        # Final aggressive cleanup - wait a bit for threads to finish
        time.sleep(0.1)

        # Force garbage collection to clean up any remaining references
        import gc
        gc.collect()

    except Exception:
        # If cleanup fails, don't fail the test
        pass


@pytest.fixture(autouse=True)
def patch_httpx_for_vcr():
    """Patch httpx Response to handle VCR replay issues."""
    try:
        import httpx
        from unittest.mock import patch

        original_content = httpx.Response.content

        def patched_content(self):
            try:
                return original_content.fget(self)
            except httpx.ResponseNotRead:
                # If content hasn't been read, try to read it from _content or return empty
                if hasattr(self, '_content') and self._content is not None:
                    return self._content
                # For VCR-replayed responses, try to get content from text if available
                try:
                    return self.text.encode(self.encoding or 'utf-8')
                except Exception:
                    return b''

        with patch.object(httpx.Response, 'content', property(patched_content)):
            yield

    except ImportError:
        # httpx not available, skip patching
        yield


@pytest.fixture(autouse=True)
def disable_telemetry():
    """Disable telemetry during tests to prevent background threads."""
    original_telemetry = os.environ.get("CREWAI_TELEMETRY", None)
    os.environ["CREWAI_TELEMETRY"] = "false"

    yield

    if original_telemetry is not None:
        os.environ["CREWAI_TELEMETRY"] = original_telemetry
    else:
        os.environ.pop("CREWAI_TELEMETRY", None)


@pytest.fixture(scope="module")
def vcr_config(request) -> dict:
    return {
        "cassette_library_dir": "tests/cassettes",
        "record_mode": "new_episodes",
        "filter_headers": [("authorization", "AUTHORIZATION-XXX")],
        "before_record_request": filter_stream_parameter,
    }


@pytest.fixture(autouse=True)
def patch_rpm_controller():
    """Patch RPMController to prevent recurring timers during tests."""
    try:
        from crewai.utilities.rpm_controller import RPMController
        from unittest.mock import patch

        original_reset_request_count = RPMController._reset_request_count

        def mock_reset_request_count(self):
            """Mock that prevents the recurring timer from being set up."""
            if self._lock:
                with self._lock:
                    self._current_rpm = 0
                    # Don't start a new timer during tests
            else:
                self._current_rpm = 0

        with patch.object(RPMController, '_reset_request_count', mock_reset_request_count):
            yield

    except ImportError:
        # If RPMController is not available, just yield
        yield
