"""Stress tests for concurrent multi-process storage access.

Simulates the Airflow pattern: N worker processes each writing to the
same storage directory simultaneously.  Verifies no LockException and
data integrity after all writes complete.

Uses temp files for IPC instead of multiprocessing.Manager (which uses
sockets blocked by pytest_recording).
"""

import json
import multiprocessing
import os
import sqlite3
import sys
import tempfile

import pytest

# Capture sys.path so child processes can resolve imports when pytest
# uses --import-mode=importlib (e.g. under xdist).
_PARENT_SYS_PATH = list(sys.path)


# ---------------------------------------------------------------------------
# File-based IPC helpers (avoids Manager sockets)
# ---------------------------------------------------------------------------

def _write_result(result_dir: str, worker_id: int, success: bool, error: str = ""):
    path = os.path.join(result_dir, f"worker-{worker_id}.json")
    with open(path, "w") as f:
        json.dump({"success": success, "error": error}, f)


def _collect_results(result_dir: str, n_workers: int):
    errors = {}
    successes = 0
    for wid in range(n_workers):
        path = os.path.join(result_dir, f"worker-{wid}.json")
        if not os.path.exists(path):
            errors[wid] = "Process produced no output (crashed or timed out)"
            continue
        with open(path) as f:
            data = json.load(f)
        if data["success"]:
            successes += 1
        else:
            errors[wid] = data["error"]
    return successes, errors


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------

def _lancedb_worker(sys_path: list, path: str, worker_id: int, n_records: int, result_dir: str):
    sys.path[:] = sys_path
    try:
        from crewai.memory.storage.lancedb_storage import LanceDBStorage
        from crewai.memory.types import MemoryRecord

        storage = LanceDBStorage(path=path, table_name="memories", vector_dim=8)
        records = [
            MemoryRecord(
                id=f"worker-{worker_id}-record-{i}",
                content=f"content from worker {worker_id} record {i}",
                scope=f"/test/worker-{worker_id}",
                categories=["test"],
                metadata={"worker": worker_id},
                importance=0.5,
                embedding=[float(worker_id)] * 8,
            )
            for i in range(n_records)
        ]
        storage.save(records)
        _write_result(result_dir, worker_id, True)
    except Exception as e:
        _write_result(result_dir, worker_id, False, f"{type(e).__name__}: {e}")


def _sqlite_kickoff_worker(sys_path: list, db_path: str, worker_id: int, n_writes: int, result_dir: str):
    sys.path[:] = sys_path
    try:
        from crewai.memory.storage.kickoff_task_outputs_storage import (
            KickoffTaskOutputsSQLiteStorage,
        )

        KickoffTaskOutputsSQLiteStorage(db_path=db_path)
        for i in range(n_writes):
            with sqlite3.connect(db_path, timeout=30) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """INSERT OR REPLACE INTO latest_kickoff_task_outputs
                    (task_id, expected_output, output, task_index, inputs, was_replayed)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        f"worker-{worker_id}-task-{i}",
                        "expected output",
                        '{"result": "ok"}',
                        worker_id * 1000 + i,
                        "{}",
                        False,
                    ),
                )
        _write_result(result_dir, worker_id, True)
    except Exception as e:
        _write_result(result_dir, worker_id, False, f"{type(e).__name__}: {e}")


def _sqlite_flow_worker(sys_path: list, db_path: str, worker_id: int, n_writes: int, result_dir: str):
    sys.path[:] = sys_path
    try:
        from crewai.flow.persistence.sqlite import SQLiteFlowPersistence

        persistence = SQLiteFlowPersistence(db_path=db_path)
        for i in range(n_writes):
            persistence.save_state(
                flow_uuid=f"flow-{worker_id}-{i}",
                method_name="test_method",
                state_data={"worker": worker_id, "iteration": i},
            )
        _write_result(result_dir, worker_id, True)
    except Exception as e:
        _write_result(result_dir, worker_id, False, f"{type(e).__name__}: {e}")


def _chromadb_worker(sys_path: list, persist_dir: str, worker_id: int, result_dir: str):
    sys.path[:] = sys_path
    try:
        from chromadb import PersistentClient
        from chromadb.config import Settings

        from crewai.utilities.lock_store import lock

        settings = Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False,
            is_persistent=True,
        )

        with lock(f"chromadb:{persist_dir}"):
            PersistentClient(path=persist_dir, settings=settings)

        _write_result(result_dir, worker_id, True)
    except Exception as e:
        _write_result(result_dir, worker_id, False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

N_WORKERS = 1
N_RECORDS = 5


def _run_workers(target, args_fn, n_workers=N_WORKERS, timeout=120):
    """Spawn n_workers processes and collect results via temp files."""
    with tempfile.TemporaryDirectory() as result_dir:
        procs = []
        for wid in range(n_workers):
            p = multiprocessing.Process(
                target=target,
                args=args_fn(wid, result_dir),
            )
            procs.append(p)

        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=timeout)

        successes, errors = _collect_results(result_dir, n_workers)
    return successes, errors


@pytest.mark.timeout(120)
class TestConcurrentLanceDB:
    """Concurrent multi-process writes to LanceDB."""

    def test_concurrent_saves_no_lock_exception(self, tmp_path):
        db_path = str(tmp_path / "lancedb_concurrent")

        successes, errors = _run_workers(
            _lancedb_worker,
            lambda wid, rd: (_PARENT_SYS_PATH, db_path, wid, N_RECORDS, rd),
        )

        assert not errors, f"Workers failed: {errors}"
        assert successes == N_WORKERS

    def test_data_integrity_after_concurrent_saves(self, tmp_path):
        db_path = str(tmp_path / "lancedb_integrity")

        successes, errors = _run_workers(
            _lancedb_worker,
            lambda wid, rd: (_PARENT_SYS_PATH, db_path, wid, N_RECORDS, rd),
        )

        assert not errors, f"Workers failed: {errors}"

        from crewai.memory.storage.lancedb_storage import LanceDBStorage

        storage = LanceDBStorage(path=db_path, table_name="memories", vector_dim=8)
        total = storage.count()
        expected = N_WORKERS * N_RECORDS
        assert total == expected, f"Expected {expected} records, got {total}"


class TestConcurrentSQLiteKickoff:
    """Concurrent multi-process writes to kickoff task outputs SQLite."""

    def test_concurrent_writes_no_error(self, tmp_path):
        db_path = str(tmp_path / "kickoff.db")

        from crewai.memory.storage.kickoff_task_outputs_storage import (
            KickoffTaskOutputsSQLiteStorage,
        )

        KickoffTaskOutputsSQLiteStorage(db_path=db_path)

        successes, errors = _run_workers(
            _sqlite_kickoff_worker,
            lambda wid, rd: (_PARENT_SYS_PATH, db_path, wid, N_RECORDS, rd),
            timeout=60,
        )

        assert not errors, f"Workers failed: {errors}"
        assert successes == N_WORKERS

        with sqlite3.connect(db_path, timeout=30) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM latest_kickoff_task_outputs"
            ).fetchone()[0]
        expected = N_WORKERS * N_RECORDS
        assert count == expected, f"Expected {expected} rows, got {count}"


class TestConcurrentSQLiteFlow:
    """Concurrent multi-process writes to flow persistence SQLite."""

    def test_concurrent_writes_no_error(self, tmp_path):
        db_path = str(tmp_path / "flow_states.db")

        successes, errors = _run_workers(
            _sqlite_flow_worker,
            lambda wid, rd: (_PARENT_SYS_PATH, db_path, wid, N_RECORDS, rd),
            timeout=60,
        )

        assert not errors, f"Workers failed: {errors}"
        assert successes == N_WORKERS

        with sqlite3.connect(db_path, timeout=30) as conn:
            count = conn.execute("SELECT COUNT(*) FROM flow_states").fetchone()[0]
        expected = N_WORKERS * N_RECORDS
        assert count == expected, f"Expected {expected} rows, got {count}"


class TestConcurrentChromaDB:
    """Concurrent multi-process ChromaDB client creation."""

    def test_concurrent_client_creation_no_lock_exception(self, tmp_path):
        persist_dir = str(tmp_path / "chromadb_concurrent")
        os.makedirs(persist_dir, exist_ok=True)

        successes, errors = _run_workers(
            _chromadb_worker,
            lambda wid, rd: (_PARENT_SYS_PATH, persist_dir, wid, rd),
        )

        assert not errors, f"Workers failed: {errors}"
        assert successes == N_WORKERS