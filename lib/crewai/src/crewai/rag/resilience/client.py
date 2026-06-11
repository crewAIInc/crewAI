import asyncio
import logging
import time
import functools
from typing import Any, List, Protocol, runtime_checkable, Mapping, Union
from crewai.rag.core.base_client import BaseClient
from crewai.rag.types import BaseRecord, SearchResult

logger = logging.getLogger(__name__)

def with_resilience(method):
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        max_retries = 3
        delay = 1.0
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return method(*args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {method.__name__} due to {type(e).__name__}. Sleeping {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    break
        
        if method.__name__ in ("search", "asearch"):
            logger.info("Triggering graceful fallback: returning empty search results.")
            return []
            
        raise last_exception if last_exception else RuntimeError("Resilience wrapper failed")
    return wrapper

def with_resilience_async(method):
    @functools.wraps(method)
    async def wrapper(*args, **kwargs):
        max_retries = 3
        delay = 1.0
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await method(*args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {method.__name__} due to {type(e).__name__}. Sleeping {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    break
        
        if method.__name__ in ("search", "asearch"):
            logger.info("Triggering graceful fallback: returning empty search results.")
            return []
            
        raise last_exception if last_exception else RuntimeError("Resilience wrapper failed")
    return wrapper

class ResilientRAGClient(BaseClient):
    def __init__(self, _client: BaseClient):
        self._client = _client

    @with_resilience
    def search(self, **kwargs) -> List[SearchResult]:
        return self._client.search(**kwargs)

    @with_resilience
    def add_documents(self, **kwargs) -> None:
        return self._client.add_documents(**kwargs)

    @with_resilience_async
    async def asearch(self, **kwargs) -> List[SearchResult]:
        return await self._client.asearch(**kwargs)

    @with_resilience_async
    async def aadd_documents(self, **kwargs) -> None:
        return await self._client.aadd_documents(**kwargs)

    # Explicitly implement remaining BaseClient methods to ensure type safety
    def create_collection(self, **kwargs) -> None:
        return self._client.create_collection(**kwargs)

    async def acreate_collection(self, **kwargs) -> None:
        return await self._client.acreate_collection(**kwargs)

    def get_or_create_collection(self, **kwargs) -> None:
        return self._client.get_or_create_collection(**kwargs)

    async def aget_or_create_collection(self, **kwargs) -> None:
        return await self._client.aget_or_create_collection(**kwargs)

    def delete_collection(self, **kwargs) -> None:
        return self._client.delete_collection(**kwargs)

    async def adelete_collection(self, **kwargs) -> None:
        return await self._client.adelete_collection(**kwargs)

    def reset(self, **kwargs) -> None:
        return self._client.reset(**kwargs)

    async def areset(self, **kwargs) -> None:
        return await self._client.areset(**kwargs)
