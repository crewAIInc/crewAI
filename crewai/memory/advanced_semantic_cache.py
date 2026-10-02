
import time
from typing import Dict, Any, Optional

class AdvancedSemanticCache:
    """
    Enterprise-grade semantic cache for CrewAI agents to reduce token usage and latency.
    """
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds

    def get(self, query: str) -> Optional[str]:
        if query in self.cache:
            entry = self.cache[query]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['response']
            else:
                del self.cache[query]
        return None

    def set(self, query: str, response: str) -> None:
        self.cache[query] = {
            'response': response,
            'timestamp': time.time()
        }
        
    def clear(self) -> None:
        self.cache.clear()
