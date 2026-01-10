import os
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from crewai.utilities.file_handler import PickleHandler


class TestPickleHandler(unittest.TestCase):
    def setUp(self):
        unique_id = str(uuid.uuid4())
        self.file_name = f"test_data_{unique_id}.pkl"
        self.file_path = os.path.join(os.getcwd(), self.file_name)
        self.handler = PickleHandler(self.file_name)

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_initialize_file(self):
        assert os.path.exists(self.file_path) is False

        self.handler.initialize_file()

        assert os.path.exists(self.file_path) is True
        assert os.path.getsize(self.file_path) >= 0

    def test_save_and_load(self):
        data = {"key": "value"}
        self.handler.save(data)
        loaded_data = self.handler.load()
        assert loaded_data == data

    def test_load_empty_file(self):
        loaded_data = self.handler.load()
        assert loaded_data == {}

    def test_load_corrupted_file(self):
        with open(self.file_path, "wb") as file:
            file.write(b"corrupted data")
            file.flush()
            os.fsync(file.fileno())  # Ensure data is written to disk

        with pytest.raises(Exception) as exc:
            self.handler.load()

        assert str(exc.value) == "pickle data was truncated"
        assert "<class '_pickle.UnpicklingError'>" == str(exc.type)


class TestPickleHandlerThreadSafety(unittest.TestCase):
    """Tests for thread-safety of PickleHandler operations."""

    def setUp(self):
        unique_id = str(uuid.uuid4())
        self.file_name = f"test_thread_safe_{unique_id}.pkl"
        self.file_path = os.path.join(os.getcwd(), self.file_name)
        self.handler = PickleHandler(self.file_name)

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_concurrent_writes_same_handler(self):
        """Test that concurrent writes from multiple threads using the same handler don't corrupt data."""
        num_threads = 10
        num_writes_per_thread = 20
        errors: list[Exception] = []
        write_count = 0
        count_lock = threading.Lock()

        def write_data(thread_id: int) -> None:
            nonlocal write_count
            for i in range(num_writes_per_thread):
                try:
                    data = {"thread": thread_id, "iteration": i, "data": f"value_{thread_id}_{i}"}
                    self.handler.save(data)
                    with count_lock:
                        write_count += 1
                except Exception as e:
                    errors.append(e)

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=write_data, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred during concurrent writes: {errors}"
        assert write_count == num_threads * num_writes_per_thread
        loaded_data = self.handler.load()
        assert isinstance(loaded_data, dict)
        assert "thread" in loaded_data
        assert "iteration" in loaded_data

    def test_concurrent_reads_same_handler(self):
        """Test that concurrent reads from multiple threads don't cause issues."""
        test_data = {"key": "value", "nested": {"a": 1, "b": 2}}
        self.handler.save(test_data)

        num_threads = 20
        results: list[dict] = []
        errors: list[Exception] = []
        results_lock = threading.Lock()

        def read_data() -> None:
            try:
                data = self.handler.load()
                with results_lock:
                    results.append(data)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=read_data)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred during concurrent reads: {errors}"
        assert len(results) == num_threads
        for result in results:
            assert result == test_data

    def test_concurrent_read_write_same_handler(self):
        """Test that concurrent reads and writes don't corrupt data or cause errors."""
        initial_data = {"counter": 0}
        self.handler.save(initial_data)

        num_writers = 5
        num_readers = 10
        writes_per_thread = 10
        reads_per_thread = 20
        write_errors: list[Exception] = []
        read_errors: list[Exception] = []
        read_results: list[dict] = []
        results_lock = threading.Lock()

        def writer(thread_id: int) -> None:
            for i in range(writes_per_thread):
                try:
                    data = {"writer": thread_id, "write_num": i}
                    self.handler.save(data)
                except Exception as e:
                    write_errors.append(e)

        def reader() -> None:
            for _ in range(reads_per_thread):
                try:
                    data = self.handler.load()
                    with results_lock:
                        read_results.append(data)
                except Exception as e:
                    read_errors.append(e)

        threads = []
        for i in range(num_writers):
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)

        for _ in range(num_readers):
            t = threading.Thread(target=reader)
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(write_errors) == 0, f"Write errors: {write_errors}"
        assert len(read_errors) == 0, f"Read errors: {read_errors}"
        for result in read_results:
            assert isinstance(result, dict)

    def test_atomic_write_no_partial_data(self):
        """Test that atomic writes prevent partial/corrupted data from being read."""
        large_data = {"key": "x" * 100000, "numbers": list(range(10000))}
        num_iterations = 50
        errors: list[Exception] = []
        corruption_detected = False
        corruption_lock = threading.Lock()

        def writer() -> None:
            for _ in range(num_iterations):
                try:
                    self.handler.save(large_data)
                except Exception as e:
                    errors.append(e)

        def reader() -> None:
            nonlocal corruption_detected
            for _ in range(num_iterations * 2):
                try:
                    data = self.handler.load()
                    if data and data != {} and data != large_data:
                        with corruption_lock:
                            corruption_detected = True
                except Exception as e:
                    errors.append(e)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert not corruption_detected, "Partial/corrupted data was read"

    def test_thread_pool_concurrent_operations(self):
        """Test thread safety using ThreadPoolExecutor for more realistic concurrent access."""
        num_operations = 100
        errors: list[Exception] = []

        def operation(op_id: int) -> str:
            try:
                if op_id % 3 == 0:
                    self.handler.save({"op_id": op_id, "type": "write"})
                    return f"write_{op_id}"
                else:
                    data = self.handler.load()
                    return f"read_{op_id}_{type(data).__name__}"
            except Exception as e:
                errors.append(e)
                return f"error_{op_id}"

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(operation, i) for i in range(num_operations)]
            results = [f.result() for f in as_completed(futures)]

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_operations

    def test_multiple_handlers_same_file(self):
        """Test that multiple PickleHandler instances for the same file work correctly."""
        handler1 = PickleHandler(self.file_name)
        handler2 = PickleHandler(self.file_name)

        num_operations = 50
        errors: list[Exception] = []

        def use_handler1() -> None:
            for i in range(num_operations):
                try:
                    handler1.save({"handler": 1, "iteration": i})
                except Exception as e:
                    errors.append(e)

        def use_handler2() -> None:
            for i in range(num_operations):
                try:
                    handler2.save({"handler": 2, "iteration": i})
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=use_handler1)
        t2 = threading.Thread(target=use_handler2)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        final_data = self.handler.load()
        assert isinstance(final_data, dict)
        assert "handler" in final_data
        assert "iteration" in final_data
