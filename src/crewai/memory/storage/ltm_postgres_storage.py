import json
import os
from typing import Any, Dict, List, Optional, Union

import psycopg
from psycopg.rows import dict_row

try:
    from psycopg.pool import ConnectionPool
    HAS_CONNECTION_POOL = True
except ImportError:
    # For older versions of psycopg or when pool extras aren't installed
    HAS_CONNECTION_POOL = False
    ConnectionPool = None

from crewai.utilities import Printer


class LTMPostgresStorage:
    """
    Postgres storage implementation for Long Term Memory data storage.
    Compatible with Postgres 16 and later versions.
    
    Supports connection pooling for improved performance in high-volume environments.
    """

    def __init__(
        self, 
        connection_string: str,
        schema: str = "public",
        table_name: str = "long_term_memories",
        min_pool_size: int = 1,
        max_pool_size: int = 5,
        use_connection_pool: bool = True,
    ) -> None:
        """
        Initialize the Postgres storage.
        
        Args:
            connection_string: Postgres connection string (e.g., "postgresql://user:password@localhost:5432/dbname")
            schema: Database schema (defaults to "public")
            table_name: Table name for storing memories (defaults to "long_term_memories")
            min_pool_size: Minimum size of the connection pool (default 1)
            max_pool_size: Maximum size of the connection pool (default 5)
            use_connection_pool: Whether to use connection pooling (default True)
        """
        self.connection_string = connection_string
        self.schema = schema
        self.table_name = table_name
        self.full_table_name = f"{self.schema}.{self.table_name}"
        self._printer: Printer = Printer()
        
        # Only use connection pooling if explicitly requested AND available
        self.use_connection_pool = use_connection_pool and HAS_CONNECTION_POOL
        
        if not HAS_CONNECTION_POOL and use_connection_pool:
            self._printer.print(
                content="Connection pooling requested but psycopg pool module not available. "
                        "Install with 'pip install \"psycopg[pool]\"'. "
                        "Falling back to direct connections.",
                color="yellow",
            )
        
        # Create connection pool if enabled and available
        if self.use_connection_pool:
            self.pool = ConnectionPool(
                self.connection_string,
                min_size=min_pool_size,
                max_size=max_pool_size,
                # Configure pool behavior
                kwargs={"row_factory": dict_row}
            )
        else:
            self.pool = None
            
        self._initialize_db()

    def _initialize_db(self):
        """
        Initializes the Postgres database and creates LTM table if it doesn't exist.
        """
        try:
            # Use the connection pool if available, otherwise create a new connection
            if self.use_connection_pool:
                with self.pool.connection() as conn:
                    self._create_schema_and_table(conn)
            else:
                with psycopg.connect(self.connection_string) as conn:
                    self._create_schema_and_table(conn)
                    
        except Exception as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred during database initialization: {e}",
                color="red",
            )
            
    def _create_schema_and_table(self, conn):
        """
        Create schema and table with the given connection.
        
        Args:
            conn: The database connection
        """
        with conn.cursor() as cursor:
            # Create schema if it doesn't exist
            cursor.execute(
                f"CREATE SCHEMA IF NOT EXISTS {self.schema}"
            )
            
            # Create the table if it doesn't exist
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.full_table_name} (
                    id SERIAL PRIMARY KEY,
                    task_description TEXT,
                    metadata JSONB,
                    datetime TEXT,
                    score REAL
                )
                """
            )
            
            # Create index on task_description for faster lookups
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_task_description 
                ON {self.full_table_name} (task_description)
                """
            )
            
            conn.commit()

    def save(
        self,
        task_description: str,
        metadata: Dict[str, Any],
        datetime: str,
        score: Union[int, float],
    ) -> None:
        """
        Saves data to the LTM table with error handling.
        
        Args:
            task_description: The description of the task
            metadata: Dictionary containing metadata about the memory
            datetime: ISO formatted datetime string
            score: Numerical score for the memory
        """
        try:
            # Use the connection pool if available, otherwise create a new connection
            if self.use_connection_pool:
                with self.pool.connection() as conn:
                    self._perform_save(conn, task_description, metadata, datetime, score)
            else:
                with psycopg.connect(self.connection_string) as conn:
                    self._perform_save(conn, task_description, metadata, datetime, score)
                    
        except Exception as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while saving to LTM: {e}",
                color="red",
            )
            
    def _perform_save(self, conn, task_description, metadata, datetime, score):
        """
        Execute the save operation with the given connection.
        
        Args:
            conn: The database connection
            task_description: The description of the task
            metadata: Dictionary containing metadata about the memory
            datetime: ISO formatted datetime string
            score: Numerical score for the memory
        """
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self.full_table_name} 
                (task_description, metadata, datetime, score)
                VALUES (%s, %s, %s, %s)
                """,
                (task_description, json.dumps(metadata), datetime, score),
            )
            conn.commit()

    def save_many(
        self,
        items: List[Dict[str, Any]],
    ) -> None:
        """
        Saves multiple data items to the LTM table in a single transaction.
        
        Args:
            items: List of dictionaries with keys: task_description, metadata, datetime, score
        """
        if not items:
            return
            
        try:
            # Use the connection pool if available, otherwise create a new connection
            if self.use_connection_pool:
                with self.pool.connection() as conn:
                    self._perform_save_many(conn, items)
            else:
                with psycopg.connect(self.connection_string) as conn:
                    self._perform_save_many(conn, items)
                    
        except Exception as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while batch saving to LTM: {e}",
                color="red",
            )
            
    def _perform_save_many(self, conn, items):
        """
        Execute the batch save operation with the given connection.
        
        Args:
            conn: The database connection
            items: List of dictionaries with keys: task_description, metadata, datetime, score
        """
        with conn.cursor() as cursor:
            values = [
                (
                    item["task_description"],
                    json.dumps(item["metadata"]),
                    item["datetime"],
                    item["score"],
                )
                for item in items
            ]
            
            cursor.executemany(
                f"""
                INSERT INTO {self.full_table_name} 
                (task_description, metadata, datetime, score)
                VALUES (%s, %s, %s, %s)
                """,
                values,
            )
            conn.commit()

    def load(
        self, task_description: str, latest_n: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Queries the LTM table by task description with error handling.
        
        Args:
            task_description: The description of the task to query for
            latest_n: Maximum number of results to return
            
        Returns:
            List of memory items or None if not found/error
        """
        try:
            # Use the connection pool if available, otherwise create a new connection
            if self.use_connection_pool:
                with self.pool.connection() as conn:
                    return self._perform_load(conn, task_description, latest_n)
            else:
                with psycopg.connect(self.connection_string, row_factory=dict_row) as conn:
                    return self._perform_load(conn, task_description, latest_n)

        except Exception as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while querying LTM: {e}",
                color="red",
            )
        return None
        
    def _perform_load(self, conn, task_description, latest_n):
        """
        Execute the load operation with the given connection.
        
        Args:
            conn: The database connection
            task_description: The description of the task to query for
            latest_n: Maximum number of results to return
            
        Returns:
            List of memory items or None if not found
        """
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT metadata, datetime, score
                FROM {self.full_table_name}
                WHERE task_description = %s
                ORDER BY datetime DESC, score ASC
                LIMIT %s
                """,
                (task_description, latest_n),
            )
            rows = cursor.fetchall()
            
            if rows:
                return [
                    {
                        "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
                        "datetime": row["datetime"],
                        "score": row["score"],
                    }
                    for row in rows
                ]
        return None

    def reset(
        self,
    ) -> None:
        """
        Resets the LTM table by deleting all rows (with error handling).
        """
        try:
            # Use the connection pool if available, otherwise create a new connection
            if self.use_connection_pool:
                with self.pool.connection() as conn:
                    self._perform_reset(conn)
            else:
                with psycopg.connect(self.connection_string) as conn:
                    self._perform_reset(conn)

        except Exception as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while deleting all rows in LTM: {e}",
                color="red",
            )
        return None
        
    def _perform_reset(self, conn):
        """
        Execute the reset operation with the given connection.
        
        Args:
            conn: The database connection
        """
        with conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM {self.full_table_name}")
            conn.commit()
            
    def close(self):
        """
        Close the connection pool if it exists.
        """
        if self.pool:
            self.pool.close()