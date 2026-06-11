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
                logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
        
        logger.error(f"All {max_retries} attempts failed. Triggering fallback.")
        # Fallback logic: return empty list for searches or log error for additions
        if "search" in method.__name__:
            return []
        raise last_exception
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
        # Delegate other calls to the underlying client
        return getattr(self._client, name)

# --- Verification ---
class MockFailingClient:
    """Simulates a vector DB that fails twice then succeeds."""
    def __init__(self):
        self.calls = 0
    
    def search(self, **kwargs):
        self.calls += 1
        if self.calls <= 2:
            print(f"  [Mock] Call {self.calls}: Simulating ConnectionError...")
            raise ConnectionError("DB is down")
        print(f"  [Mock] Call {self.calls}: Success!")
        return [{"id": "1", "content": "Resilient result", "score": 0.9}]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ResilientRAGClient with recovering DB...")
    fail_client = MockFailingClient()
    resilient = ResilientRAGClient(fail_client)
    
    results = resilient.search(query="test")
    print(f"Final Results: {results}")
    
    print("\nTesting ResilientRAGClient with permanent failure...")
    class PermanentFailClient:
        def search(self, **kwargs):
            raise ConnectionError("DB permanently gone")
    
    resilient_fail = ResilientRAGClient(PermanentFailClient())
    results_fail = resilient_fail.search(query="test")
    print(f"Final Results (Fallback): {results_fail}")
