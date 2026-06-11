import time
import logging
from typing import Any, List, Protocol, runtime_checkable, TypeVar, cast
from functools import wraps

# Mocking the types from crewAI for the prototype
class SearchResult(dict): pass
class BaseRecord(dict): pass

T = TypeVar("T", bound="BaseClient")

@runtime_checkable
class BaseClient(Protocol):
    def search(self, **kwargs) -> List[SearchResult]: ...
    def add_documents(self, **kwargs) -> None: ...

logger = logging.getLogger("ResilientRAG")

def with_resilience(method):
    """Decorator to add retry logic and fallback to ResilientRAGClient methods."""
    def wrapper(self, *args, **kwargs):
        max_retries = 3
        delay = 1.0
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return method(self, *args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"All {max_retries} attempts failed.")
        
        # Fallback logic: return empty list for searches or raise last exception for others
        if "search" in method.__name__:
            logger.info("Triggering graceful fallback: returning empty search results.")
            return []
        
        if last_exception:
            raise last_exception
        raise RuntimeError("Resilience wrapper failed without a caught exception")
    return wrapper

class ResilientRAGClient:
    """
    A wrapper for BaseClient that provides resilience:
    - Exponential backoff for connection errors.
    - Graceful fallback to prevent agent crashes.
    """
    def __init__(self, client: BaseClient):
        self._client = client

    @with_resilience
    def search(self, **kwargs) -> List[SearchResult]:
        return self._client.search(**kwargs)

    @with_resilience
    def add_documents(self, **kwargs) -> None:
        self._client.add_documents(**kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)

# --- Verification ---
class MockFailingClient:
    def __init__(self):
        self.calls = 0
    
    def search(self, **kwargs):
        self.calls += 1
        if self.calls <= 2:
            raise ConnectionError("DB is down")
        return [{"id": "1", "content": "Resilient result", "score": 0.9}]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ResilientRAGClient with recovering DB...")
    fail_client = MockFailingClient()
    resilient = ResilientRAGClient(fail_client)
    print(f"Final Results: {resilient.search(query='test')}")
    
    print("\nTesting ResilientRAGClient with permanent failure...")
    class PermanentFailClient:
        def search(self, **kwargs):
            raise ConnectionError("DB permanently gone")
    
    resilient_fail = ResilientRAGClient(PermanentFailClient())
    print(f"Final Results (Fallback): {resilient_fail.search(query='test')}")
