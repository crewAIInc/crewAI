from typing import Any, Dict, List, Optional

from crewai.memory.storage.llm_response_cache_storage import LLMResponseCacheStorage

class LLMResponseCacheHandler:
    """
    Handler for the LLM response cache storage.
    Used for record/replay functionality.
    """

    def __init__(self) -> None:
        self.storage = LLMResponseCacheStorage()
        self._recording = False
        self._replaying = False

    def start_recording(self) -> None:
        """
        Starts recording LLM responses.
        """
        self._recording = True
        self._replaying = False

    def start_replaying(self) -> None:
        """
        Starts replaying LLM responses from the cache.
        """
        self._recording = False
        self._replaying = True

    def stop(self) -> None:
        """
        Stops recording or replaying.
        """
        self._recording = False
        self._replaying = False

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
        """
        if self._recording:
            self.storage.add(model, messages, response)

    def get_cached_response(self, model: str, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Retrieves a cached LLM response if replaying is active.
        Returns None if not found or if replaying is not active.
        """
        if self._replaying:
            return self.storage.get(model, messages)
        return None

    def clear_cache(self) -> None:
        """
        Clears the LLM response cache.
        """
        self.storage.delete_all()
