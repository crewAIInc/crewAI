"""
SQLite-based implementation of flow state persistence.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel

from crewai.flow.persistence.base import FlowPersistence


class SQLiteFlowPersistence(FlowPersistence):
    """SQLite-based implementation of flow state persistence.

    This class provides a simple, file-based persistence implementation using SQLite.
    It's suitable for development and testing, or for production use cases with
    moderate performance requirements.
    """

    db_path: str  # Type annotation for instance variable

    def __init__(self, db_path: Optional[str] = None):
        """Initialize SQLite persistence.

        Args:
            db_path: Path to the SQLite database file. If not provided, uses
                    db_storage_path() from utilities.paths.

        Raises:
            ValueError: If db_path is invalid
        """
        from crewai.utilities.paths import db_storage_path

        # Get path from argument or default location
        path = db_path or str(Path(db_storage_path()) / "flow_states.db")

        if not path:
            raise ValueError("Database path must be provided")

        self.db_path = path  # Now mypy knows this is str
        self.init_db()

    def init_db(self) -> None:
        """Create the necessary tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS flow_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_uuid TEXT NOT NULL,
                method_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                state_json TEXT NOT NULL
            )
            """
            )
            # Add index for faster UUID lookups
            conn.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_flow_states_uuid
            ON flow_states(flow_uuid)
            """
            )

    def save_state(
        self,
        flow_uuid: str,
        method_name: str,
        state_data: Union[Dict[str, Any], BaseModel],
    ) -> None:
        """Save the current flow state to SQLite.

        Args:
            flow_uuid: Unique identifier for the flow instance
            method_name: Name of the method that just completed
            state_data: Current state data (either dict or Pydantic model)

        Raises:
            ValueError: If state_data is neither a dict nor a BaseModel
            RuntimeError: If database operations fail
            TypeError: If JSON serialization fails
        """
        try:
            # Convert state_data to a JSON-serializable dict using the helper method
            state_dict = self._convert_to_dict(state_data)

            # Try to serialize to JSON to catch any serialization issues early
            try:
                state_json = json.dumps(state_dict)
            except (TypeError, ValueError, OverflowError) as json_err:
                raise TypeError(
                    f"Failed to serialize state to JSON: {json_err}"
                ) from json_err

            # Perform database operation with error handling
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                    INSERT INTO flow_states (
                        flow_uuid,
                        method_name,
                        timestamp,
                        state_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                        (
                            flow_uuid,
                            method_name,
                            datetime.now(timezone.utc).isoformat(),
                            state_json,
                        ),
                    )
            except sqlite3.Error as db_err:
                raise RuntimeError(f"Database operation failed: {db_err}") from db_err

        except Exception as e:
            # Log the error but don't crash the application
            import logging

            logging.error(f"Failed to save flow state: {e}")
            # Re-raise to allow caller to handle or ignore
            raise

    def load_state(self, flow_uuid: str) -> Optional[Dict[str, Any]]:
        """Load the most recent state for a given flow UUID.

        Args:
            flow_uuid: Unique identifier for the flow instance

        Returns:
            The most recent state as a dictionary, or None if no state exists
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
            SELECT state_json
            FROM flow_states
            WHERE flow_uuid = ?
            ORDER BY id DESC
            LIMIT 1
            """,
                (flow_uuid,),
            )
            row = cursor.fetchone()

        if row:
            return json.loads(row[0])
        return None
