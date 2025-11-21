"""Tests for read-write lock implementation.

This module tests the RWLock class for correct concurrent read and write behavior.
"""

import threading
import time

from crewai.utilities.rw_lock import RWLock


def test_multiple_readers_concurrent():
    lock = RWLock()
    active_readers = [0]
    max_concurrent_readers = [0]
    lock_for_counters = threading.Lock()

    def reader(reader_id: int) -> None:
        with lock.r_locked():
            with lock_for_counters:
                active_readers[0] += 1
                max_concurrent_readers[0] = max(
                    max_concurrent_readers[0], active_readers[0]
                )

            time.sleep(0.1)

            with lock_for_counters:
                active_readers[0] -= 1

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert max_concurrent_readers[0] == 5


def test_writer_blocks_readers():
    lock = RWLock()
    writer_holding_lock = [False]
    reader_accessed_during_write = [False]

    def writer() -> None:
        with lock.w_locked():
            writer_holding_lock[0] = True
            time.sleep(0.2)
            writer_holding_lock[0] = False

    def reader() -> None:
        time.sleep(0.05)
        with lock.r_locked():
            if writer_holding_lock[0]:
                reader_accessed_during_write[0] = True

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)

    writer_thread.start()
    reader_thread.start()

    writer_thread.join()
    reader_thread.join()

    assert not reader_accessed_during_write[0]


def test_writer_blocks_other_writers():
    lock = RWLock()
    execution_order: list[int] = []
    lock_for_order = threading.Lock()

    def writer(writer_id: int) -> None:
        with lock.w_locked():
            with lock_for_order:
                execution_order.append(writer_id)
            time.sleep(0.1)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert len(execution_order) == 3
    assert len(set(execution_order)) == 3


def test_readers_block_writers():
    lock = RWLock()
    reader_count = [0]
    writer_accessed_during_read = [False]
    lock_for_counters = threading.Lock()

    def reader() -> None:
        with lock.r_locked():
            with lock_for_counters:
                reader_count[0] += 1
            time.sleep(0.2)
            with lock_for_counters:
                reader_count[0] -= 1

    def writer() -> None:
        time.sleep(0.05)
        with lock.w_locked():
            with lock_for_counters:
                if reader_count[0] > 0:
                    writer_accessed_during_read[0] = True

    reader_thread = threading.Thread(target=reader)
    writer_thread = threading.Thread(target=writer)

    reader_thread.start()
    writer_thread.start()

    reader_thread.join()
    writer_thread.join()

    assert not writer_accessed_during_read[0]


def test_alternating_readers_and_writers():
    lock = RWLock()
    operations: list[str] = []
    lock_for_operations = threading.Lock()

    def reader(reader_id: int) -> None:
        with lock.r_locked():
            with lock_for_operations:
                operations.append(f"r{reader_id}_start")
            time.sleep(0.05)
            with lock_for_operations:
                operations.append(f"r{reader_id}_end")

    def writer(writer_id: int) -> None:
        with lock.w_locked():
            with lock_for_operations:
                operations.append(f"w{writer_id}_start")
            time.sleep(0.05)
            with lock_for_operations:
                operations.append(f"w{writer_id}_end")

    threads = [
        threading.Thread(target=reader, args=(0,)),
        threading.Thread(target=writer, args=(0,)),
        threading.Thread(target=reader, args=(1,)),
        threading.Thread(target=writer, args=(1,)),
        threading.Thread(target=reader, args=(2,)),
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert len(operations) == 10

    start_ops = [op for op in operations if "_start" in op]
    end_ops = [op for op in operations if "_end" in op]
    assert len(start_ops) == 5
    assert len(end_ops) == 5


def test_context_manager_releases_on_exception():
    lock = RWLock()
    exception_raised = False

    try:
        with lock.r_locked():
            raise ValueError("Test exception")
    except ValueError:
        exception_raised = True

    assert exception_raised

    acquired = False
    with lock.w_locked():
        acquired = True

    assert acquired


def test_write_lock_releases_on_exception():
    lock = RWLock()
    exception_raised = False

    try:
        with lock.w_locked():
            raise ValueError("Test exception")
    except ValueError:
        exception_raised = True

    assert exception_raised

    acquired = False
    with lock.r_locked():
        acquired = True

    assert acquired


def test_stress_many_readers_few_writers():
    lock = RWLock()
    read_count = [0]
    write_count = [0]
    lock_for_counters = threading.Lock()

    def reader() -> None:
        for _ in range(10):
            with lock.r_locked():
                with lock_for_counters:
                    read_count[0] += 1
                time.sleep(0.001)

    def writer() -> None:
        for _ in range(5):
            with lock.w_locked():
                with lock_for_counters:
                    write_count[0] += 1
                time.sleep(0.01)

    reader_threads = [threading.Thread(target=reader) for _ in range(10)]
    writer_threads = [threading.Thread(target=writer) for _ in range(2)]

    all_threads = reader_threads + writer_threads

    for thread in all_threads:
        thread.start()

    for thread in all_threads:
        thread.join()

    assert read_count[0] == 100
    assert write_count[0] == 10


def test_nested_read_locks_same_thread():
    lock = RWLock()
    nested_acquired = False

    with lock.r_locked():
        with lock.r_locked():
            nested_acquired = True

    assert nested_acquired


def test_manual_acquire_release():
    lock = RWLock()

    lock.r_acquire()
    lock.r_release()

    lock.w_acquire()
    lock.w_release()

    with lock.r_locked():
        pass
