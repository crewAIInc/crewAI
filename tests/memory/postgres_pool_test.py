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


class TestConnectionPooling(unittest.TestCase):
    """Tests for PostgreSQL connection pooling."""
    
    @patch("crewai.memory.storage.ltm_postgres_storage.HAS_CONNECTION_POOL", True)
    @patch("crewai.memory.storage.ltm_postgres_storage.ConnectionPool")
    @patch("psycopg.connect")
    def test_pool_creation(self, mock_connect, mock_pool):
        """Test that connection pool is created with the correct parameters."""
        # Setup mock pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance
        
        # Setup mock connection for initialization
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Create storage with connection pooling enabled
        storage = LTMPostgresStorage(
            connection_string="postgresql://user:pass@localhost:5432/testdb",
            min_pool_size=2,
            max_pool_size=10,
            use_connection_pool=True
        )
        
        # Verify that the pool was created with the correct parameters
        mock_pool.assert_called_once()
        args, kwargs = mock_pool.call_args
        self.assertEqual(args[0], "postgresql://user:pass@localhost:5432/testdb")
        self.assertEqual(kwargs["min_size"], 2)
        self.assertEqual(kwargs["max_size"], 10)
        
    @patch("crewai.memory.storage.ltm_postgres_storage.HAS_CONNECTION_POOL", False)
    @patch("psycopg.connect")
    def test_pool_disabled_when_unavailable(self, mock_connect):
        """Test that connection pool is not created when not available."""
        # Setup mock connection for initialization
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Create storage with connection pooling enabled but unavailable
        storage = LTMPostgresStorage(
            connection_string="postgresql://user:pass@localhost:5432/testdb",
            use_connection_pool=True
        )
        
        # Verify that use_connection_pool was set to False due to unavailability
        self.assertFalse(storage.use_connection_pool)
        self.assertIsNone(storage.pool)
        
    @patch("crewai.memory.storage.ltm_postgres_storage.HAS_CONNECTION_POOL", True)
    @patch("psycopg.connect")
    def test_pool_disabled_explicitly(self, mock_connect):
        """Test that connection pool is not created when explicitly disabled."""
        # Setup mock connection for initialization
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Create storage with connection pooling disabled
        storage = LTMPostgresStorage(
            connection_string="postgresql://user:pass@localhost:5432/testdb",
            use_connection_pool=False
        )
        
        # Verify that the pool was not created
        self.assertFalse(storage.use_connection_pool)
        self.assertIsNone(storage.pool)
        
    @patch("crewai.memory.storage.ltm_postgres_storage.HAS_CONNECTION_POOL", True)
    @patch("crewai.memory.storage.ltm_postgres_storage.ConnectionPool")
    @patch("psycopg.connect")
    def test_save_uses_pool(self, mock_connect, mock_pool):
        """Test that save operation uses the connection pool."""
        # Setup mock pool
        mock_pool_instance = MagicMock()
        mock_pool_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_pool_instance.connection.return_value.__enter__.return_value = mock_pool_conn
        mock_pool_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool.return_value = mock_pool_instance
        
        # Setup mock connection for initialization
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Create storage with connection pooling enabled
        storage = LTMPostgresStorage(
            connection_string="postgresql://user:pass@localhost:5432/testdb",
            use_connection_pool=True
        )
        
        # Clear mock calls from initialization
        mock_pool_instance.connection.reset_mock()
        mock_connect.reset_mock()
        
        # Call save operation
        storage.save(
            task_description="Test task",
            metadata={"key": "value"},
            datetime=datetime.now().isoformat(),
            score=0.9
        )
        
        # Verify that the pool connection was used
        self.assertTrue(mock_pool_instance.connection.called)
        self.assertFalse(mock_connect.called)  # Direct connection should not be used
        
    @patch("crewai.memory.storage.ltm_postgres_storage.HAS_CONNECTION_POOL", True)
    @patch("crewai.memory.storage.ltm_postgres_storage.ConnectionPool")
    @patch("psycopg.connect")
    def test_save_many(self, mock_connect, mock_pool):
        """Test batch save operation with connection pool."""
        # Setup mock pool
        mock_pool_instance = MagicMock()
        mock_pool_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_pool_instance.connection.return_value.__enter__.return_value = mock_pool_conn
        mock_pool_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool.return_value = mock_pool_instance
        
        # Setup mock connection for initialization
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Create storage with connection pooling enabled
        storage = LTMPostgresStorage(
            connection_string="postgresql://user:pass@localhost:5432/testdb",
            use_connection_pool=True
        )
        
        # Clear mock calls from initialization
        mock_pool_instance.connection.reset_mock()
        mock_connect.reset_mock()
        
        # Prepare test data
        test_items = [
            {
                "task_description": "Task 1",
                "metadata": {"key": "value1"},
                "datetime": datetime.now().isoformat(),
                "score": 0.9
            },
            {
                "task_description": "Task 2",
                "metadata": {"key": "value2"},
                "datetime": datetime.now().isoformat(),
                "score": 0.8
            }
        ]
        
        # Call batch save operation
        storage.save_many(test_items)
        
        # Verify that the pool connection was used
        self.assertTrue(mock_pool_instance.connection.called)
        self.assertFalse(mock_connect.called)  # Direct connection should not be used
        
        # Verify that executemany was called
        mock_cursor.executemany.assert_called_once()
        
    @patch("crewai.memory.storage.ltm_postgres_storage.HAS_CONNECTION_POOL", True)
    @patch("crewai.memory.storage.ltm_postgres_storage.ConnectionPool")
    def test_close_pool(self, mock_pool):
        """Test that close method closes the connection pool."""
        # Setup mock pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance
        
        # Mock _initialize_db to avoid the actual call
        with patch.object(LTMPostgresStorage, '_initialize_db'):
            # Create storage with connection pooling enabled
            storage = LTMPostgresStorage(
                connection_string="postgresql://user:pass@localhost:5432/testdb",
                use_connection_pool=True
            )
            
            # Call close method
            storage.close()
            
            # Verify that pool close was called
            mock_pool_instance.close.assert_called_once()


class TestEnvironmentConfig(unittest.TestCase):
    """Tests for PostgreSQL environment variable configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_postgres_config()
            
            self.assertEqual(config["host"], "localhost")
            self.assertEqual(config["port"], "5432")
            self.assertEqual(config["user"], "postgres")
            self.assertEqual(config["password"], "")
            self.assertEqual(config["db"], "crewai")
            self.assertEqual(config["schema"], "public")
            self.assertEqual(config["table"], "long_term_memories")
            self.assertEqual(config["min_pool"], 1)
            self.assertEqual(config["max_pool"], 5)
            self.assertTrue(config["enable_pool"])
    
    def test_custom_config(self):
        """Test configuration with custom environment variables."""
        with patch.dict(os.environ, {
            "CREWAI_PG_HOST": "pgserver.example.com",
            "CREWAI_PG_PORT": "5433",
            "CREWAI_PG_USER": "testuser",
            "CREWAI_PG_PASSWORD": "testpass",
            "CREWAI_PG_DB": "testdb",
            "CREWAI_PG_SCHEMA": "testschema",
            "CREWAI_PG_TABLE": "test_memories",
            "CREWAI_PG_MIN_POOL": "3",
            "CREWAI_PG_MAX_POOL": "15",
            "CREWAI_PG_ENABLE_POOL": "false"
        }):
            config = get_postgres_config()
            
            self.assertEqual(config["host"], "pgserver.example.com")
            self.assertEqual(config["port"], "5433")
            self.assertEqual(config["user"], "testuser")
            self.assertEqual(config["password"], "testpass")
            self.assertEqual(config["db"], "testdb")
            self.assertEqual(config["schema"], "testschema")
            self.assertEqual(config["table"], "test_memories")
            self.assertEqual(config["min_pool"], 3)
            self.assertEqual(config["max_pool"], 15)
            self.assertFalse(config["enable_pool"])
    
    def test_connection_string_direct(self):
        """Test connection string from direct environment variable."""
        with patch.dict(os.environ, {
            "CREWAI_PG_CONNECTION_STRING": "postgresql://direct:auth@example.com:5432/directdb"
        }):
            conn_string = get_postgres_connection_string()
            
            self.assertEqual(conn_string, "postgresql://direct:auth@example.com:5432/directdb")
    
    def test_connection_string_components(self):
        """Test connection string built from component environment variables."""
        with patch.dict(os.environ, {
            "CREWAI_PG_HOST": "pgserver.example.com",
            "CREWAI_PG_USER": "testuser",
            "CREWAI_PG_PASSWORD": "testpass",
            "CREWAI_PG_DB": "testdb"
        }):
            conn_string = get_postgres_connection_string()
            
            self.assertEqual(conn_string, "postgresql://testuser:testpass@pgserver.example.com:5432/testdb")
    
    def test_connection_string_no_password(self):
        """Test connection string built without password."""
        with patch.dict(os.environ, {
            "CREWAI_PG_HOST": "pgserver.example.com",
            "CREWAI_PG_USER": "testuser",
            "CREWAI_PG_DB": "testdb"
        }):
            conn_string = get_postgres_connection_string()
            
            self.assertEqual(conn_string, "postgresql://testuser@pgserver.example.com:5432/testdb")


class TestStorageFactoryWithEnv(unittest.TestCase):
    """Tests for StorageFactory with environment variables."""
    
    @patch("crewai.memory.storage.ltm_postgres_storage.ConnectionPool")
    @patch("crewai.memory.storage.ltm_postgres_storage.HAS_CONNECTION_POOL", True)
    @patch("crewai.memory.storage.ltm_postgres_storage.LTMPostgresStorage._initialize_db")
    def test_create_postgres_from_env(self, mock_init_db, mock_pool):
        """Test creating PostgreSQL storage from environment variables."""
        # Setup mock pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance
        
        with patch.dict(os.environ, {
            "CREWAI_PG_HOST": "envhost.example.com",
            "CREWAI_PG_USER": "envuser",
            "CREWAI_PG_PASSWORD": "envpass",
            "CREWAI_PG_DB": "envdb",
            "CREWAI_PG_SCHEMA": "envschema",
            "CREWAI_PG_TABLE": "env_memories",
            "CREWAI_PG_MIN_POOL": "4",
            "CREWAI_PG_MAX_POOL": "20",
            "CREWAI_PG_ENABLE_POOL": "true"
        }):
            # Create storage with just the type
            storage = LTMStorageFactory.create_storage(storage_type="postgres")
            
            # Verify that the storage was created with the correct parameters
            self.assertEqual(storage.connection_string, 
                             "postgresql://envuser:envpass@envhost.example.com:5432/envdb")
            self.assertEqual(storage.schema, "envschema")
            self.assertEqual(storage.table_name, "env_memories")
            self.assertTrue(storage.use_connection_pool)
    
    @patch("crewai.memory.storage.ltm_postgres_storage.ConnectionPool")
    @patch("crewai.memory.storage.ltm_postgres_storage.HAS_CONNECTION_POOL", True)
    @patch("crewai.memory.storage.ltm_postgres_storage.LTMPostgresStorage._initialize_db")
    def test_override_env_with_params(self, mock_init_db, mock_pool, mock_connect=None):
        """Test overriding environment variables with parameters."""
        # Setup mock pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value = mock_pool_instance
        
        with patch.dict(os.environ, {
            "CREWAI_PG_HOST": "envhost.example.com",
            "CREWAI_PG_USER": "envuser",
            "CREWAI_PG_PASSWORD": "envpass",
            "CREWAI_PG_DB": "envdb",
            "CREWAI_PG_SCHEMA": "envschema",
            "CREWAI_PG_TABLE": "env_memories",
            "CREWAI_PG_MIN_POOL": "4",
            "CREWAI_PG_MAX_POOL": "20",
            "CREWAI_PG_ENABLE_POOL": "true"
        }):
            # Create storage with explicit parameters
            storage = LTMStorageFactory.create_storage(
                storage_type="postgres",
                connection_string="postgresql://explicit:pass@explicit.com:5432/explicitdb",
                schema="explicitschema",
                table_name="explicit_memories",
                use_connection_pool=False,
                min_pool_size=2,
                max_pool_size=10
            )
            
            # Verify that the explicit parameters took precedence
            self.assertEqual(storage.connection_string, 
                             "postgresql://explicit:pass@explicit.com:5432/explicitdb")
            self.assertEqual(storage.schema, "explicitschema")
            self.assertEqual(storage.table_name, "explicit_memories")
            self.assertFalse(storage.use_connection_pool)


@pytest.mark.skipif(
    "POSTGRES_CONNECTION_STRING" not in os.environ,
    reason="Postgres connection string not provided in environment variables",
)
class TestPoolingIntegration(unittest.TestCase):
    """Integration tests for PostgreSQL connection pooling."""
    
    def setUp(self):
        """Set up test environment."""
        self.connection_string = os.environ.get("POSTGRES_CONNECTION_STRING")
        
        # Create storage with connection pooling enabled
        self.storage = LTMPostgresStorage(
            connection_string=self.connection_string,
            schema="test_pool_schema",
            table_name="test_pool_memories",
            use_connection_pool=True,
            min_pool_size=1,
            max_pool_size=5
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, "storage"):
            self.storage.reset()
            self.storage.close()
    
    def test_pooled_operations(self):
        """Test operations with connection pooling."""
        # Skip if connection pooling is not available
        if not hasattr(self.storage, 'pool') or self.storage.pool is None:
            self.skipTest("Connection pooling not available")
            
        # Prepare test data
        task_description = "Pooled Task Test"
        metadata = {"test_key": "pool_value", "quality": 0.95}
        dt = datetime.now().isoformat()
        
        # Test save operation
        self.storage.save(task_description, metadata, dt, 0.95)
        
        # Test load operation
        results = self.storage.load(task_description, 1)
        
        # Verify that data was saved and retrieved correctly
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["test_key"], "pool_value")
        
        # Test batch save
        batch_items = [
            {
                "task_description": "Batch Task 1",
                "metadata": {"test_key": "batch1", "quality": 0.91},
                "datetime": datetime.now().isoformat(),
                "score": 0.91
            },
            {
                "task_description": "Batch Task 2",
                "metadata": {"test_key": "batch2", "quality": 0.92},
                "datetime": datetime.now().isoformat(),
                "score": 0.92
            }
        ]
        
        self.storage.save_many(batch_items)
        
        # Test loading batch items
        results1 = self.storage.load("Batch Task 1", 1)
        results2 = self.storage.load("Batch Task 2", 1)
        
        # Verify batch data was saved correctly
        self.assertIsNotNone(results1)
        self.assertEqual(results1[0]["metadata"]["test_key"], "batch1")
        
        self.assertIsNotNone(results2)
        self.assertEqual(results2[0]["metadata"]["test_key"], "batch2")
        
        # Test reset operation
        self.storage.reset()
        
        # Verify data was deleted
        results_after_reset = self.storage.load(task_description, 1)
        self.assertIsNone(results_after_reset)


if __name__ == "__main__":
    unittest.main()