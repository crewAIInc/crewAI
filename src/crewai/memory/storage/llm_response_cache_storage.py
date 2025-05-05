import hashlib
import json
import logging
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from crewai.utilities import Printer
from crewai.utilities.crew_json_encoder import CrewJSONEncoder
from crewai.utilities.paths import db_storage_path

logger = logging.getLogger(__name__)


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
        self._connection_pool: Dict[int, sqlite3.Connection] = {}
        self._initialize_db()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Gets a connection from the connection pool or creates a new one.
        Uses thread-local storage to ensure thread safety.
        """
        thread_id = threading.get_ident()
        if thread_id not in self._connection_pool:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                self._connection_pool[thread_id] = conn
            except sqlite3.Error as e:
                error_msg = f"Failed to create SQLite connection: {e}"
                self._printer.print(
                    content=f"LLM RESPONSE CACHE ERROR: {error_msg}",
                    color="red",
                )
                logger.error(error_msg)
                raise
        return self._connection_pool[thread_id]

    def _close_connections(self) -> None:
        """
        Closes all connections in the connection pool.
        """
        for thread_id, conn in list(self._connection_pool.items()):
            try:
                conn.close()
                del self._connection_pool[thread_id]
            except sqlite3.Error as e:
                error_msg = f"Failed to close SQLite connection: {e}"
                self._printer.print(
                    content=f"LLM RESPONSE CACHE ERROR: {error_msg}",
                    color="red",
                )
                logger.error(error_msg)

    def _initialize_db(self) -> None:
        """
        Initializes the SQLite database and creates the llm_response_cache table
        """
        try:
            conn = self._get_connection()
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
            error_msg = f"Failed to initialize database: {e}"
            self._printer.print(
                content=f"LLM RESPONSE CACHE ERROR: {error_msg}",
                color="red",
            )
            logger.error(error_msg)
            raise

    def _compute_request_hash(self, model: str, messages: List[Dict[str, str]]) -> str:
        """
        Computes a hash for the request based on the model and messages.
        This hash is used as the key for caching.
        
        Sensitive information like API keys should not be included in the hash.
        """
        try:
            message_str = json.dumps(messages, sort_keys=True)
            request_hash = hashlib.sha256(f"{model}:{message_str}".encode()).hexdigest()
            return request_hash
        except Exception as e:
            error_msg = f"Failed to compute request hash: {e}"
            self._printer.print(
                content=f"LLM RESPONSE CACHE ERROR: {error_msg}",
                color="red",
            )
            logger.error(error_msg)
            raise

    def add(self, model: str, messages: List[Dict[str, str]], response: str) -> None:
        """
        Adds a response to the cache.
        """
        try:
            request_hash = self._compute_request_hash(model, messages)
            messages_json = json.dumps(messages, cls=CrewJSONEncoder)
            
            conn = self._get_connection()
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
            error_msg = f"Failed to add response to cache: {e}"
            self._printer.print(
                content=f"LLM RESPONSE CACHE ERROR: {error_msg}",
                color="red",
            )
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Unexpected error when adding response: {e}"
            self._printer.print(
                content=f"LLM RESPONSE CACHE ERROR: {error_msg}",
                color="red",
            )
            logger.error(error_msg)
            raise

    def get(self, model: str, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Retrieves a response from the cache based on the model and messages.
        Returns None if not found.
        """
        try:
            request_hash = self._compute_request_hash(model, messages)
            
            conn = self._get_connection()
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
            error_msg = f"Failed to retrieve response from cache: {e}"
            self._printer.print(
                content=f"LLM RESPONSE CACHE ERROR: {error_msg}",
                color="red",
            )
            logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Unexpected error when retrieving response: {e}"
            self._printer.print(
                content=f"LLM RESPONSE CACHE ERROR: {error_msg}",
                color="red",
            )
            logger.error(error_msg)
            return None

    def delete_all(self) -> None:
        """
        Deletes all records from the llm_response_cache table.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM llm_response_cache")
            conn.commit()
        except sqlite3.Error as e:
            error_msg = f"Failed to clear cache: {e}"
            self._printer.print(
                content=f"LLM RESPONSE CACHE ERROR: {error_msg}",
                color="red",
            )
            logger.error(error_msg)
            raise

    def cleanup_expired_cache(self, max_age_days: int = 7) -> None:
        """
        Removes cache entries older than the specified number of days.
        
        Args:
            max_age_days: Maximum age of cache entries in days. Defaults to 7.
                          If set to 0, all entries will be deleted.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if max_age_days <= 0:
                cursor.execute("DELETE FROM llm_response_cache")
                deleted_count = cursor.rowcount
                logger.info("Deleting all cache entries (max_age_days <= 0)")
            else:
                cursor.execute(
                    """
                    DELETE FROM llm_response_cache
                    WHERE timestamp < datetime('now', ? || ' days')
                    """,
                    (f"-{max_age_days}",)
                )
                deleted_count = cursor.rowcount
                
            conn.commit()
            
            if deleted_count > 0:
                self._printer.print(
                    content=f"LLM RESPONSE CACHE: Removed {deleted_count} expired cache entries",
                    color="green",
                )
                logger.info(f"Removed {deleted_count} expired cache entries")
                
        except sqlite3.Error as e:
            error_msg = f"Failed to cleanup expired cache: {e}"
            self._printer.print(
                content=f"LLM RESPONSE CACHE ERROR: {error_msg}",
                color="red",
            )
            logger.error(error_msg)
            raise

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the cache.
        
        Returns:
            A dictionary containing cache statistics.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM llm_response_cache")
            total_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT model, COUNT(*) FROM llm_response_cache GROUP BY model")
            model_counts = {model: count for model, count in cursor.fetchall()}
            
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM llm_response_cache")
            oldest, newest = cursor.fetchone()
            
            return {
                "total_entries": total_count,
                "entries_by_model": model_counts,
                "oldest_entry": oldest,
                "newest_entry": newest,
            }
            
        except sqlite3.Error as e:
            error_msg = f"Failed to get cache stats: {e}"
            self._printer.print(
                content=f"LLM RESPONSE CACHE ERROR: {error_msg}",
                color="red",
            )
            logger.error(error_msg)
            return {"error": str(e)}

    def __del__(self) -> None:
        """
        Closes all connections when the object is garbage collected.
        """
        self._close_connections()
