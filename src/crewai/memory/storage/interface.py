from typing import Any, Dict


class Storage:
    """Abstract base class defining the storage interface"""

    def save(self, key: str, value: Any, metadata: Dict[str, Any]) -> None:
        pass

    def search(self, key: str) -> Dict[str, Any]:  # type: ignore
        pass
