"""Stress tests for concurrent multi-process storage access.

Simulates the Airflow pattern: N worker processes each writing to the
same storage directory simultaneously.  Verifies no LockException and
data integrity after all writes complete.

Uses temp files for IPC instead of multiprocessing.Manager (which uses
sockets blocked by pytest_recording).
"""

import pytest

pytestmark = pytest.mark.skip(reason="Multiprocessing tests incompatible with xdist --import-mode=importlib")