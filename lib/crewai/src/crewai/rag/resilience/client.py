import asyncio
import logging
import time
import functools
from typing import Any, List, Protocol, runtime_checkable
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
        
        if "search" == method.__name__:
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
        
        if "search" == method.__name__:
            logger.info("Triggering graceful fallback: returning empty search results.")
            return []
            
        raise last_exception if last_exception else RuntimeError("Resilience wrapper failed")
    return wrapper

@runtime_checkable
class ResilientRAGClient(BaseClient):
    def __init__(self, _client: BaseClient):
        self._client = _client

    @with_resilience
    def search(self, query: str, *args, **kwargs) -> List[SearchResult]:
        return self._client.search(query, *args, **kwargs)

    @with_resilience
    def add_documents(self, documents: List[BaseRecord], *args, **kwargs) -> None:
        return self._client.add_documents(documents, *args, **kwargs)

    @with_resilience_async
    async def asearch(self, query: str, *args, **kwargs) -> List[SearchResult]:
        return await self._client.asearch(query, *args, **kwargs)

    @with_resilience_async
    async def aadd_documents(self, documents: List[BaseRecord], *args, **kwargs) -> None:
        return await self._client.aadd_documents(documents, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Forward only a limited set of attributes to avoid bypassing type safety
        # In a real implementation, this would be an allowlist.
        return getattr(self._client, name)
