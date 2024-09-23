from typing import Any, Dict


class Storage:
    """Abstract base class defining the storage interface"""

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        pass

    def search(
        self, query: str, limit: int, filters: Dict, score_threshold: float
    ) -> Dict[str, Any]:  # type: ignore
        pass

    def reset(self) -> None:
        pass
