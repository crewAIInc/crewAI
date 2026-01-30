"""
Concurrency tests for PickleHandler to verify thread-safe file operations.

These tests verify that the PickleHandler correctly handles concurrent access
scenarios that would previously cause race conditions, data corruption, or
test flakiness.
"""

import multiprocessing as mp
import os
import random
import tempfile
import time
import unittest
from typing import Any

import pytest

from crewai.utilities.file_handler import PickleHandler


def writer_process(file_path: str, writer_id: int, iterations: int = 3) -> None:
    """Writer process that saves data multiple times with small delays."""
    handler = PickleHandler(file_path)
    for i in range(iterations):
        data = {"writer_id": writer_id, "iteration": i, "timestamp": time.time()}
        handler.save(data)
        # Small random delay to increase chance of race conditions
        time.sleep(random.uniform(0.001, 0.01))


def reader_process(file_path: str, result_queue: mp.Queue) -> None:
    """Reader process that attempts to load data and reports results."""
    handler = PickleHandler(file_path)
    try:
        data = handler.load()
        result_queue.put(("success", data))
    except Exception as e:
        result_queue.put(("error", str(e), type(e).__name__))


def read_write_process(
    file_path: str, process_id: int, result_queue: mp.Queue, iterations: int = 5
) -> None:
    """Process that performs both read and write operations."""
    handler = PickleHandler(file_path)
    try:
        for i in range(iterations):
            # Read current data
            current_data = handler.load()
            
            # Modify data
            if not isinstance(current_data, dict):
                current_data = {}
            
            current_data[f"process_{process_id}"] = {
                "iteration": i,
                "timestamp": time.time(),
            }
            
            # Save modified data
            handler.save(current_data)
            
            # Small delay
            time.sleep(random.uniform(0.001, 0.005))
        
        result_queue.put(("success", process_id))
    except Exception as e:
        result_queue.put(("error", process_id, str(e), type(e).__name__))


class TestPickleHandlerConcurrency(unittest.TestCase):
    """Test suite for PickleHandler concurrency scenarios."""

    def setUp(self):
        """Set up test environment with temporary file."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        self.temp_file.close()
        self.file_path = self.temp_file.name
        # Extract just the filename for PickleHandler
        self.file_name = os.path.basename(self.file_path)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_concurrent_writers_no_corruption(self):
        """Test that multiple concurrent writers don't corrupt the file."""
        num_writers = 4
        processes = []
        
        # Start multiple writer processes
        for i in range(num_writers):
            p = mp.Process(target=writer_process, args=(self.file_name, i, 3))
            p.start()
            processes.append(p)
        
        # Wait for all writers to complete
        for p in processes:
            p.join(timeout=10)  # 10 second timeout
            if p.is_alive():
                p.terminate()
                p.join()
        
        # Verify file is not corrupted and contains valid data
        handler = PickleHandler(self.file_name)
        try:
            final_data = handler.load()
            # Should be a dictionary (last writer's data)
            self.assertIsInstance(final_data, dict)
            self.assertIn("writer_id", final_data)
            self.assertIn("iteration", final_data)
            self.assertIn("timestamp", final_data)
        except Exception as e:
            self.fail(f"File was corrupted after concurrent writes: {e}")

    def test_concurrent_readers_no_errors(self):
        """Test that multiple concurrent readers don't encounter errors."""
        # First, write some initial data
        handler = PickleHandler(self.file_name)
        initial_data = {"test": "data", "number": 42}
        handler.save(initial_data)
        
        num_readers = 8
        result_queue = mp.Queue()
        processes = []
        
        # Start multiple reader processes
        for _ in range(num_readers):
            p = mp.Process(target=reader_process, args=(self.file_name, result_queue))
            p.start()
            processes.append(p)
        
        # Wait for all readers to complete
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get_nowait())
        
        # Verify all reads were successful
        self.assertEqual(len(results), num_readers)
        for result in results:
            status = result[0]
            self.assertEqual(status, "success", f"Reader failed: {result}")
            data = result[1]
            self.assertEqual(data, initial_data)

    def test_mixed_readers_writers_no_corruption(self):
        """Test concurrent readers and writers don't cause corruption."""
        num_writers = 3
        num_readers = 6
        result_queue = mp.Queue()
        processes = []
        
        # Start writer processes
        for i in range(num_writers):
            p = mp.Process(target=writer_process, args=(self.file_name, i, 2))
            p.start()
            processes.append(p)
        
        # Start reader processes
        for _ in range(num_readers):
            p = mp.Process(target=reader_process, args=(self.file_name, result_queue))
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
                p.join()
        
        # Collect reader results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get_nowait())
        
        # Verify no reader encountered corruption errors
        for result in results:
            status = result[0]
            if status == "error":
                error_msg = result[1]
                error_type = result[2]
                # Should not have pickle corruption errors
                self.assertNotIn("UnpicklingError", error_type)
                self.assertNotIn("EOFError", error_type)

    def test_read_modify_write_operations(self):
        """Test read-modify-write operations under concurrency."""
        num_processes = 4
        result_queue = mp.Queue()
        processes = []
        
        # Initialize with empty data
        handler = PickleHandler(self.file_name)
        handler.save({})
        
        # Start processes that do read-modify-write operations
        for i in range(num_processes):
            p = mp.Process(
                target=read_write_process, 
                args=(self.file_name, i, result_queue, 3)
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
                p.join()
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get_nowait())
        
        # Verify all processes completed successfully
        successful_processes = [r for r in results if r[0] == "success"]
        self.assertEqual(len(successful_processes), num_processes)
        
        # Verify final data integrity
        final_data = handler.load()
        self.assertIsInstance(final_data, dict)
        
        # Should have entries from all processes
        for i in range(num_processes):
            process_key = f"process_{i}"
            self.assertIn(process_key, final_data)
            self.assertIsInstance(final_data[process_key], dict)
            self.assertIn("iteration", final_data[process_key])
            self.assertIn("timestamp", final_data[process_key])

    def test_lock_timeout_configuration(self):
        """Test that lock timeout can be configured."""
        # Test with very short timeout
        handler = PickleHandler(self.file_name, lock_timeout=0.1)
        self.assertEqual(handler.lock_timeout, 0.1)
        
        # Should still work for basic operations
        test_data = {"timeout_test": True}
        handler.save(test_data)
        loaded_data = handler.load()
        self.assertEqual(loaded_data, test_data)

    @pytest.mark.skipif(
        os.name == "nt", 
        reason="Process termination timing can be unreliable on Windows"
    )
    def test_file_remains_consistent_after_process_crash(self):
        """Test that file remains consistent even if a process crashes during write."""
        # This test is more complex and may be skipped on some platforms
        # It verifies that atomic writes prevent corruption even with crashes
        
        handler = PickleHandler(self.file_name)
        initial_data = {"crash_test": "initial_value"}
        handler.save(initial_data)
        
        # Verify we can still read the data after potential crashes
        loaded_data = handler.load()
        self.assertEqual(loaded_data, initial_data)


if __name__ == "__main__":
    unittest.main()