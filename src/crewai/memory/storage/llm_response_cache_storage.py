import json
import sqlite3
import hashlib
from typing import Any, Dict, List, Optional

from crewai.utilities import Printer
from crewai.utilities.crew_json_encoder import CrewJSONEncoder
from crewai.utilities.paths import db_storage_path

class LLMResponseCacheStorage:
    """
    SQLite storage for caching LLM responses.
    Used for offline record/replay functionality.
    """

    def __init__(
        self, db_path: str = f"{db_storage_path()}/llm_response_cache.db"
    ) -> None:
        self.db_path = db_path
        self._printer: Printer = Printer()
        self._initialize_db()

    def _initialize_db(self):
        """
        Initializes the SQLite database and creates the llm_response_cache table
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS llm_response_cache (
                        request_hash TEXT PRIMARY KEY,
                        model TEXT,
                        messages TEXT,
                        response TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )
                conn.commit()
        except sqlite3.Error as e:
            self._printer.print(
                content=f"LLM RESPONSE CACHE ERROR: An error occurred during database initialization: {e}",
                color="red",
            )

    def _compute_request_hash(self, model: str, messages: List[Dict[str, str]]) -> str:
        """
        Computes a hash for the request based on the model and messages.
        This hash is used as the key for caching.
        
        Sensitive information like API keys should not be included in the hash.
        """
        message_str = json.dumps(messages, sort_keys=True)
        request_hash = hashlib.sha256(f"{model}:{message_str}".encode()).hexdigest()
        return request_hash

    def add(self, model: str, messages: List[Dict[str, str]], response: str) -> None:
        """
        Adds a response to the cache.
        """
        try:
            request_hash = self._compute_request_hash(model, messages)
            messages_json = json.dumps(messages, cls=CrewJSONEncoder)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO llm_response_cache
                    (request_hash, model, messages, response)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        request_hash,
                        model,
                        messages_json,
                        response,
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            self._printer.print(
                content=f"LLM RESPONSE CACHE ERROR: Failed to add response: {e}",
                color="red",
            )

    def get(self, model: str, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Retrieves a response from the cache based on the model and messages.
        Returns None if not found.
        """
        try:
            request_hash = self._compute_request_hash(model, messages)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT response
                    FROM llm_response_cache
                    WHERE request_hash = ?
                    """,
                    (request_hash,),
                )
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except sqlite3.Error as e:
            self._printer.print(
                content=f"LLM RESPONSE CACHE ERROR: Failed to retrieve response: {e}",
                color="red",
            )
            return None

    def delete_all(self) -> None:
        """
        Deletes all records from the llm_response_cache table.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM llm_response_cache")
                conn.commit()
        except sqlite3.Error as e:
            self._printer.print(
                content=f"LLM RESPONSE CACHE ERROR: Failed to clear cache: {e}",
                color="red",
            )
