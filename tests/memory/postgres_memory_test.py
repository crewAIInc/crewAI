import json
import os
import unittest
from unittest.mock import MagicMock, patch

import pytest
from datetime import datetime

from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.memory.storage.ltm_postgres_storage import LTMPostgresStorage
from crewai.memory.storage.ltm_storage_factory import LTMStorageFactory
from crewai.utilities.postgres_config import get_postgres_config, get_postgres_connection_string


class TestLTMPostgresStorage(unittest.TestCase):
    """Test the LTMPostgresStorage class."""

    @patch("psycopg.connect")
    def test_initialization(self, mock_connect):
        """Test that the storage initializes the database correctly."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Create storage instance
        storage = LTMPostgresStorage(
            connection_string="postgresql://user:pass@localhost:5432/testdb"
        )

        # Verify schema creation
        mock_cursor.execute.assert_any_call("CREATE SCHEMA IF NOT EXISTS public")

        # Verify table creation
        create_table_call_found = False
        for call in mock_cursor.execute.call_args_list:
            arg = call[0][0]
            if "CREATE TABLE IF NOT EXISTS" in arg and "long_term_memories" in arg:
                create_table_call_found = True
                break
        self.assertTrue(create_table_call_found, "Table creation SQL not called")

        # Verify index creation
        index_call_found = False
        for call in mock_cursor.execute.call_args_list:
            arg = call[0][0]
            if "CREATE INDEX IF NOT EXISTS" in arg and "task_description" in arg:
                index_call_found = True
                break
        self.assertTrue(index_call_found, "Index creation SQL not called")

    @patch("psycopg.connect")
    def test_save(self, mock_connect):
        """Test saving data to Postgres."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Create storage instance
        storage = LTMPostgresStorage(
            connection_string="postgresql://user:pass@localhost:5432/testdb"
        )

        # Clear mock calls from initialization
        mock_cursor.execute.reset_mock()

        # Test data
        task = "Test task"
        metadata = {"key": "value", "quality": 0.8}
        dt = datetime.now().isoformat()
        score = 0.9

        # Save data
        storage.save(task, metadata, dt, score)

        # Verify SQL execution
        insert_call_found = False
        for call in mock_cursor.execute.call_args_list:
            arg = call[0][0]
            if "INSERT INTO" in arg and "long_term_memories" in arg:
                insert_call_found = True
                # Verify parameters were included
                params = call[0][1]
                self.assertEqual(params[0], task)
                self.assertEqual(json.loads(params[1]), metadata)
                self.assertEqual(params[2], dt)
                self.assertEqual(params[3], score)
                break
        self.assertTrue(insert_call_found, "Insert SQL not called")

    @patch("psycopg.connect")
    def test_load(self, mock_connect):
        """Test loading data from Postgres."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Mock return value for cursor.fetchall
        mock_metadata = json.dumps({"key": "value", "quality": 0.8})
        mock_dt = datetime.now().isoformat()
        mock_score = 0.9
        mock_cursor.fetchall.return_value = [
            {"metadata": mock_metadata, "datetime": mock_dt, "score": mock_score}
        ]

        # Create storage instance
        storage = LTMPostgresStorage(
            connection_string="postgresql://user:pass@localhost:5432/testdb"
        )

        # Clear mock calls from initialization
        mock_cursor.execute.reset_mock()

        # Load data
        result = storage.load("Test task", 3)

        # Verify SQL execution
        select_call_found = False
        for call in mock_cursor.execute.call_args_list:
            arg = call[0][0]
            if "SELECT" in arg and "FROM" in arg and "long_term_memories" in arg:
                select_call_found = True
                # Verify parameters
                params = call[0][1]
                self.assertEqual(params[0], "Test task")
                self.assertEqual(params[1], 3)
                break
        self.assertTrue(select_call_found, "Select SQL not called")

        # Verify result format
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["datetime"], mock_dt)
        self.assertEqual(result[0]["score"], mock_score)
        self.assertIsInstance(result[0]["metadata"], dict)

    @patch("psycopg.connect")
    def test_reset(self, mock_connect):
        """Test resetting the database."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Create storage instance
        storage = LTMPostgresStorage(
            connection_string="postgresql://user:pass@localhost:5432/testdb"
        )

        # Clear mock calls from initialization
        mock_cursor.execute.reset_mock()

        # Reset database
        storage.reset()

        # Verify SQL execution
        delete_call_found = False
        for call in mock_cursor.execute.call_args_list:
            arg = call[0][0]
            if "DELETE FROM" in arg and "long_term_memories" in arg:
                delete_call_found = True
                break
        self.assertTrue(delete_call_found, "Delete SQL not called")


@pytest.mark.skipif(
    "POSTGRES_CONNECTION_STRING" not in os.environ,
    reason="Postgres connection string not provided in environment variables",
)
class TestLongTermMemoryWithPostgres(unittest.TestCase):
    """Integration tests for LongTermMemory with Postgres backend."""

    def setUp(self):
        """Set up test environment."""
        connection_string = os.environ.get("POSTGRES_CONNECTION_STRING")
        self.memory = LongTermMemory(
            storage_type="postgres",
            postgres_connection_string=connection_string,
            postgres_schema="test_schema",
            postgres_table_name="test_memories",
        )

    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, "memory"):
            self.memory.reset()

    def test_save_and_search(self):
        """Test saving and searching items with Postgres backend."""
        # Create current datetime
        current_time = datetime.now().isoformat()
        
        # Create and save memory item
        item = LongTermMemoryItem(
            task="Test Postgres Task",
            agent="test_agent",
            expected_output="Expected output",
            datetime=current_time,
            quality=0.95,
            metadata={"test_key": "test_value", "quality": 0.95},  # Added quality to metadata
        )
        self.memory.save(item)

        # Search for the item
        results = self.memory.search("Test Postgres Task")

        # Verify results
        self.assertIsNotNone(results)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["metadata"]["test_key"], "test_value")


class TestStorageFactory(unittest.TestCase):
    """Test the storage factory."""

    def test_create_sqlite_storage(self):
        """Test creating SQLite storage."""
        storage = LTMStorageFactory.create_storage(storage_type="sqlite")
        self.assertIsNotNone(storage)
        self.assertEqual(storage.__class__.__name__, "LTMSQLiteStorage")

    @pytest.mark.skipif(
        "POSTGRES_CONNECTION_STRING" not in os.environ,
        reason="Postgres connection string not provided in environment variables",
    )
    def test_create_postgres_storage(self):
        """Test creating Postgres storage."""
        connection_string = os.environ.get("POSTGRES_CONNECTION_STRING")
        storage = LTMStorageFactory.create_storage(
            storage_type="postgres",
            connection_string=connection_string,
            schema="test_schema",
            table_name="test_factory",
        )
        self.assertIsNotNone(storage)
        self.assertEqual(storage.__class__.__name__, "LTMPostgresStorage")

    def test_invalid_storage_type(self):
        """Test providing an invalid storage type."""
        with self.assertRaises(ValueError):
            LTMStorageFactory.create_storage(storage_type="invalid_type")

    def test_postgres_without_connection_string(self):
        """Test creating Postgres storage without connection string."""
        # Save the current environment variables related to Postgres
        postgres_env_vars = {}
        postgres_conn = os.environ.pop("POSTGRES_CONNECTION_STRING", None)
        for key in list(os.environ.keys()):
            if key.startswith("CREWAI_PG_"):
                postgres_env_vars[key] = os.environ.pop(key)
        
        try:
            # Patch the utility functions to ensure they don't use the environment
            with patch("crewai.memory.storage.ltm_storage_factory.get_postgres_connection_string", return_value=None):
                with patch("crewai.memory.storage.ltm_storage_factory.get_postgres_config", return_value={
                    "host": "localhost", "port": "5432", "user": "postgres", 
                    "password": "", "db": "crewai", "schema": "public", 
                    "table": "long_term_memories", "min_pool": 1, 
                    "max_pool": 5, "enable_pool": True
                }):
                    with self.assertRaises(ValueError):
                        LTMStorageFactory.create_storage(storage_type="postgres")
        finally:
            # Restore environment variables
            if postgres_conn:
                os.environ["POSTGRES_CONNECTION_STRING"] = postgres_conn
            for key, value in postgres_env_vars.items():
                os.environ[key] = value


if __name__ == "__main__":
    unittest.main()