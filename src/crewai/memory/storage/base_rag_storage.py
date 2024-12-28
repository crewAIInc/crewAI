from abc import ABC, abstractmethod
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, TypeVar
from abc import ABC, abstractmethod
from pathlib import Path

from crewai.utilities.paths import get_default_storage_path


class BaseRAGStorage(ABC):
    """
    Base class for RAG-based Storage implementations.
    """

    app: Any | None = None

    def __init__(
        self,
        type: str,
        storage_path: Optional[Path] = None,
        allow_reset: bool = True,
        embedder_config: Optional[Any] = None,
        crew: Any = None,
    ) -> None:
        """Initialize the BaseRAGStorage.

        Args:
            type: Type of storage being used
            storage_path: Optional custom path for storage location
            allow_reset: Whether storage can be reset
            embedder_config: Optional configuration for the embedder
            crew: Optional crew instance this storage belongs to
        
        Raises:
            PermissionError: If storage path is not writable
            OSError: If storage path cannot be created
        """
        self.type = type
        self.storage_path = storage_path if storage_path else get_default_storage_path('rag')
        
        # Validate storage path
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            if not os.access(self.storage_path.parent, os.W_OK):
                raise PermissionError(f"No write permission for storage path: {self.storage_path}")
        except OSError as e:
            raise OSError(f"Failed to initialize storage path: {str(e)}")
            
        self.allow_reset = allow_reset
        self.embedder_config = embedder_config
        self.crew = crew
        self.agents = self._initialize_agents()

    def _initialize_agents(self) -> str:
        """Initialize agent identifiers for storage.
        
        Returns:
            str: Underscore-joined string of sanitized agent role names
        """
        if self.crew:
            return "_".join(
                [self._sanitize_role(agent.role) for agent in self.crew.agents]
            )
        return ""

    @abstractmethod
    def _sanitize_role(self, role: str) -> str:
        """Sanitizes agent roles to ensure valid directory names.
        
        Args:
            role: The agent role name to sanitize
            
        Returns:
            str: Sanitized role name safe for use in paths
        """
        pass

    @abstractmethod
    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        """Save a value with metadata to the storage.
        
        Args:
            value: The value to store
            metadata: Additional metadata to store with the value
            
        Raises:
            OSError: If there is an error writing to storage
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """Search for entries in the storage.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            filter: Optional filter criteria
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List[Dict[str, Any]]: List of matching entries with their metadata
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the storage.
        
        Raises:
            OSError: If there is an error clearing storage
            PermissionError: If reset is not allowed
        """
        pass

    @abstractmethod
    def _generate_embedding(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """Generate an embedding for the given text and metadata.
        
        Args:
            text: Text to generate embedding for
            metadata: Optional metadata to include in embedding
            
        Returns:
            List[float]: Vector embedding of the text
            
        Raises:
            ValueError: If text is empty or invalid
        """
        pass

    @abstractmethod
    def _initialize_app(self) -> None:
        """Initialize the vector db.
        
        Raises:
            OSError: If vector db initialization fails
        """
        pass

    def setup_config(self, config: Dict[str, Any]):
        """Setup the config of the storage."""
        pass

    def initialize_client(self):
        """Initialize the client of the storage. This should setup the app and the db collection"""
        pass
