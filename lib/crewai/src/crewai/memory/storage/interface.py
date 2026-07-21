from typing import Any


class Storage:
    """Abstract base class defining the storage interface"""

    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        pass

    def search(
        self, query: str, limit: int, score_threshold: float
    ) -> dict[str, Any] | list[Any]:
        return {}

    def reset(self) -> None:
        pass
