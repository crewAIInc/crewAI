import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from crewai.utilities import Printer
from crewai.utilities.paths import db_storage_path


class LTMSQLiteStorage:
    """
    An updated SQLite storage class for LTM data storage.
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        """Initialize LTM SQLite storage.
        
        Args:
            storage_path: Optional custom path for storage location
            
        Raises:
            PermissionError: If storage path is not writable
            OSError: If storage path cannot be created
        """
        self.storage_path = storage_path if storage_path else Path(f"{db_storage_path()}/latest_long_term_memories.db")
        
        # Validate storage path
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            if not os.access(self.storage_path.parent, os.W_OK):
                raise PermissionError(f"No write permission for storage path: {self.storage_path}")
        except OSError as e:
            raise OSError(f"Failed to initialize storage path: {str(e)}")
            
        self._printer: Printer = Printer()
        self._initialize_db()

    def _initialize_db(self):
        """
        Initializes the SQLite database and creates LTM table
        """
        try:
            with sqlite3.connect(str(self.storage_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS long_term_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_description TEXT,
                        metadata TEXT,
                        datetime TEXT,
                        score REAL
                    )
                """
                )

                conn.commit()
        except sqlite3.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred during database initialization: {e}",
                color="red",
            )

    def save(
        self,
        task_description: str,
        metadata: Dict[str, Any],
        datetime: str,
        score: Union[int, float],
    ) -> None:
        """Save a memory entry to long-term memory.
        
        Args:
            task_description: Description of the task this memory relates to
            metadata: Additional data to store with the memory
            datetime: Timestamp for when this memory was created
            score: Relevance score for this memory (higher is more relevant)
            
        Raises:
            sqlite3.Error: If there is an error saving to the database
        """
        """Saves data to the LTM table with error handling."""
        try:
            with sqlite3.connect(str(self.storage_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                INSERT INTO long_term_memories (task_description, metadata, datetime, score)
                VALUES (?, ?, ?, ?)
            """,
                    (task_description, json.dumps(metadata), datetime, score),
                )
                conn.commit()
        except sqlite3.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while saving to LTM: {e}",
                color="red",
            )

    def load(
        self, task_description: str, latest_n: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Queries the LTM table by task description with error handling."""
        try:
            with sqlite3.connect(str(self.storage_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT metadata, datetime, score
                    FROM long_term_memories
                    WHERE task_description = ?
                    ORDER BY datetime DESC, score ASC
                    LIMIT {latest_n}
                """,  # nosec
                    (task_description,),
                )
                rows = cursor.fetchall()
                if rows:
                    return [
                        {
                            "metadata": json.loads(row[0]),
                            "datetime": row[1],
                            "score": row[2],
                        }
                        for row in rows
                    ]

        except sqlite3.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while querying LTM: {e}",
                color="red",
            )
        return None

    def reset(
        self,
    ) -> None:
        """Resets the LTM table with error handling."""
        try:
            with sqlite3.connect(str(self.storage_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM long_term_memories")
                conn.commit()

        except sqlite3.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while deleting all rows in LTM: {e}",
                color="red",
            )
        return None
