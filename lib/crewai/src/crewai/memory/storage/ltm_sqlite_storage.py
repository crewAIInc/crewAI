import json
from pathlib import Path
import sqlite3
from typing import Any

import aiosqlite

from crewai.utilities import Printer
from crewai.utilities.paths import db_storage_path


class LTMSQLiteStorage:
    """SQLite storage class for long-term memory data."""

    def __init__(self, db_path: str | None = None, verbose: bool = True) -> None:
        """Initialize the SQLite storage.

        Args:
            db_path: Optional path to the database file.
            verbose: Whether to print error messages.
        """
        if db_path is None:
            db_path = str(Path(db_storage_path()) / "long_term_memory_storage.db")
        self.db_path = db_path
        self._verbose = verbose
        self._printer: Printer = Printer()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the SQLite database and create LTM table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: An error occurred during database initialization: {e}",
                    color="red",
                )

    def save(
        self,
        task_description: str,
        metadata: dict[str, Any],
        datetime: str,
        score: int | float,
    ) -> None:
        """Saves data to the LTM table with error handling."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: An error occurred while saving to LTM: {e}",
                    color="red",
                )

    def load(self, task_description: str, latest_n: int) -> list[dict[str, Any]] | None:
        """Queries the LTM table by task description with error handling."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT metadata, datetime, score
                    FROM long_term_memories
                    WHERE task_description = ?
                    ORDER BY datetime DESC, score ASC
                    LIMIT {latest_n}
                """,  # nosec # noqa: S608
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
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: An error occurred while querying LTM: {e}",
                    color="red",
                )
        return None

    def reset(self) -> None:
        """Resets the LTM table with error handling."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM long_term_memories")
                conn.commit()

        except sqlite3.Error as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: An error occurred while deleting all rows in LTM: {e}",
                    color="red",
                )

    async def asave(
        self,
        task_description: str,
        metadata: dict[str, Any],
        datetime: str,
        score: int | float,
    ) -> None:
        """Save data to the LTM table asynchronously.

        Args:
            task_description: Description of the task.
            metadata: Metadata associated with the memory.
            datetime: Timestamp of the memory.
            score: Quality score of the memory.
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute(
                    """
                    INSERT INTO long_term_memories (task_description, metadata, datetime, score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (task_description, json.dumps(metadata), datetime, score),
                )
                await conn.commit()
        except aiosqlite.Error as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: An error occurred while saving to LTM: {e}",
                    color="red",
                )

    async def aload(
        self, task_description: str, latest_n: int
    ) -> list[dict[str, Any]] | None:
        """Query the LTM table by task description asynchronously.

        Args:
            task_description: Description of the task to search for.
            latest_n: Maximum number of results to return.

        Returns:
            List of matching memory entries or None if error occurs.
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    f"""
                    SELECT metadata, datetime, score
                    FROM long_term_memories
                    WHERE task_description = ?
                    ORDER BY datetime DESC, score ASC
                    LIMIT {latest_n}
                    """,  # nosec # noqa: S608
                    (task_description,),
                )
                rows = await cursor.fetchall()
                if rows:
                    return [
                        {
                            "metadata": json.loads(row[0]),
                            "datetime": row[1],
                            "score": row[2],
                        }
                        for row in rows
                    ]
        except aiosqlite.Error as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: An error occurred while querying LTM: {e}",
                    color="red",
                )
        return None

    async def areset(self) -> None:
        """Reset the LTM table asynchronously."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("DELETE FROM long_term_memories")
                await conn.commit()
        except aiosqlite.Error as e:
            if self._verbose:
                self._printer.print(
                    content=f"MEMORY ERROR: An error occurred while deleting all rows in LTM: {e}",
                    color="red",
                )
