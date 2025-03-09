from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Generic, List, Protocol, TypeVar, TypedDict, runtime_checkable

from pydantic import BaseModel, ConfigDict

class SearchResult(TypedDict, total=False):
    """Type definition for search results"""
    context: str
    metadata: Dict[str, Any]
    score: float
    memory: str  # For Mem0Storage compatibility

T = TypeVar('T')

@runtime_checkable
class StorageProtocol(Protocol):
    """Protocol defining the storage interface"""
    def save(self, value: Any, metadata: Dict[str, Any]) -> None: ...
    def search(self, query: str, limit: int, score_threshold: float) -> List[Any]: ...
    def reset(self) -> None: ...

class Storage(ABC, Generic[T]):
    """Abstract base class defining the storage interface"""
    
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def search(
        self, query: str, limit: int, score_threshold: float
    ) -> List[SearchResult]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
