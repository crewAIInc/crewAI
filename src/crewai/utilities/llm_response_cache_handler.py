import logging
from typing import Any, Dict, List, Optional

from crewai.memory.storage.llm_response_cache_storage import LLMResponseCacheStorage

logger = logging.getLogger(__name__)


class LLMResponseCacheHandler:
    """
    Handler for the LLM response cache storage.
    Used for record/replay functionality.
    """

    def __init__(self, max_cache_age_days: int = 7) -> None:
        """
        Initializes the LLM response cache handler.
        
        Args:
            max_cache_age_days: Maximum age of cache entries in days. Defaults to 7.
        """
        self.storage = LLMResponseCacheStorage()
        self._recording = False
        self._replaying = False
        self.max_cache_age_days = max_cache_age_days
        
        try:
            self.storage.cleanup_expired_cache(self.max_cache_age_days)
        except Exception as e:
            logger.warning(f"Failed to cleanup expired cache on initialization: {e}")

    def start_recording(self) -> None:
        """
        Starts recording LLM responses.
        """
        self._recording = True
        self._replaying = False
        logger.info("Started recording LLM responses")

    def start_replaying(self) -> None:
        """
        Starts replaying LLM responses from the cache.
        """
        self._recording = False
        self._replaying = True
        logger.info("Started replaying LLM responses from cache")
        
        try:
            stats = self.storage.get_cache_stats()
            logger.info(f"Cache statistics: {stats}")
        except Exception as e:
            logger.warning(f"Failed to get cache statistics: {e}")

    def stop(self) -> None:
        """
        Stops recording or replaying.
        """
        was_recording = self._recording
        was_replaying = self._replaying
        
        self._recording = False
        self._replaying = False
        
        if was_recording:
            logger.info("Stopped recording LLM responses")
        if was_replaying:
            logger.info("Stopped replaying LLM responses")

    def is_recording(self) -> bool:
        """
        Returns whether recording is active.
        """
        return self._recording

    def is_replaying(self) -> bool:
        """
        Returns whether replaying is active.
        """
        return self._replaying

    def cache_response(self, model: str, messages: List[Dict[str, str]], response: str) -> None:
        """
        Caches an LLM response if recording is active.
        
        Args:
            model: The model used for the LLM call.
            messages: The messages sent to the LLM.
            response: The response from the LLM.
        """
        if not self._recording:
            return
            
        try:
            self.storage.add(model, messages, response)
            logger.debug(f"Cached response for model {model}")
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")

    def get_cached_response(self, model: str, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Retrieves a cached LLM response if replaying is active.
        Returns None if not found or if replaying is not active.
        
        Args:
            model: The model used for the LLM call.
            messages: The messages sent to the LLM.
            
        Returns:
            The cached response, or None if not found or if replaying is not active.
        """
        if not self._replaying:
            return None
            
        try:
            response = self.storage.get(model, messages)
            if response:
                logger.debug(f"Retrieved cached response for model {model}")
            else:
                logger.debug(f"No cached response found for model {model}")
            return response
        except Exception as e:
            logger.error(f"Failed to retrieve cached response: {e}")
            return None

    def clear_cache(self) -> None:
        """
        Clears the LLM response cache.
        """
        try:
            self.storage.delete_all()
            logger.info("Cleared LLM response cache")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            
    def cleanup_expired_cache(self) -> None:
        """
        Removes cache entries older than the maximum age.
        """
        try:
            self.storage.cleanup_expired_cache(self.max_cache_age_days)
            logger.info(f"Cleaned up expired cache entries (older than {self.max_cache_age_days} days)")
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the cache.
        
        Returns:
            A dictionary containing cache statistics.
        """
        try:
            return self.storage.get_cache_stats()
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
