import time
import logging
from typing import Any, List, Protocol, runtime_checkable, TypeVar, cast, Annotated
from functools import wraps
from typing_extensions import Required, TypedDict, Unpack

# Types from crewAI
class BaseRecord(TypedDict, total=False):
    doc_id: str
    content: Required[str]
    metadata: dict[str, Any]

class SearchResult(TypedDict):
    id: str
    content: str
    metadata: dict[str, Any]
    score: float

class BaseCollectionParams(TypedDict):
    collection_name: Required[str]

class BaseCollectionAddParams(BaseCollectionParams, total=False):
    documents: Required[list[BaseRecord]]
    batch_size: int

class BaseCollectionSearchParams(BaseCollectionParams, total=False):
    query: Required[str]
    limit: int
    metadata_filter: dict[str, Any] | None
    score_threshold: float

@runtime_checkable
class BaseClient(Protocol):
    client: Any
    embedding_function: Any
    def create_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None: ...
    async def acreate_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None: ...
    def get_or_create_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> Any: ...
    async def aget_or_create_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> Any: ...
    def add_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None: ...
    async def aadd_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None: ...
    def search(self, **kwargs: Unpack[BaseCollectionSearchParams]) -> List[SearchResult]: ...
    async def asearch(self, **kwargs: Unpack[BaseCollectionSearchParams]) -> List[SearchResult]: ...
    def delete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None: ...
    async def adelete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None: ...
    def reset(self) -> None: ...
    async def areset(self) -> None: ...

logger = logging.getLogger("ResilientRAG")

def with_resilience(method):
    def wrapper(self, *args, **kwargs):
        max_retries = 3
        delay = 1.0
        last_exception = None
        for attempt in range(max_retries):
            try:
                return method(self, *args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                # Only sleep if there are attempts remaining
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
        if "search" in method.__name__:
            logger.info("Triggering graceful fallback: returning empty search results.")
            return []
        if last_exception:
            raise last_exception
        raise RuntimeError("Resilience wrapper failed")
    return wrapper

class ResilientRAGClient:
    def __init__(self, client: BaseClient):
        self._client = client
        self.client = getattr(client, "client", None)
        self.embedding_function = getattr(client, "embedding_function", None)

    @with_resilience
    def search(self, **kwargs) -> List[SearchResult]:
        return self._client.search(**kwargs)

    @with_resilience
    def add_documents(self, **kwargs) -> None:
        self._client.add_documents(**kwargs)

    def create_collection(self, **kwargs): return self._client.create_collection(**kwargs)
    async def acreate_collection(self, **kwargs): return await self._client.acreate_collection(**kwargs)
    def get_or_create_collection(self, **kwargs): return self._client.get_or_create_collection(**kwargs)
    async def aget_or_create_collection(self, **kwargs): return await self._client.aget_or_create_collection(**kwargs)
    async def aadd_documents(self, **kwargs): return await self._client.aadd_documents(**kwargs)
    async def asearch(self, **kwargs): return await self._client.asearch(**kwargs)
    def delete_collection(self, **kwargs): return self._client.delete_collection(**kwargs)
    async def adelete_collection(self, **kwargs): return await self._client.adelete_collection(**kwargs)
    def reset(self): return self._client.reset()
    async def areset(self): return await self._client.areset()

    def __getattr__(self, name):
        return getattr(self._client, name)
