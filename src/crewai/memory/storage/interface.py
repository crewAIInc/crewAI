from typing import Any, Dict


class Storage:
    """Abstract base class defining the storage interface"""

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        pass

    def search(
        self, key: str, query: str, limit: int, filters: Dict, score_threshold: float
    ) -> Dict[str, Any]:  # type: ignore
        return {}

    def reset(self) -> None:
        pass
