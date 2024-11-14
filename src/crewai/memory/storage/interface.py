from typing import Any, Dict, List


class Storage:
    """Abstract base class defining the storage interface"""

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        pass

    def search(
        self, query: str, limit: int, score_threshold: float
    ) -> Dict[str, Any] | List[Any]:
        return {}

    def reset(self) -> None:
        pass
