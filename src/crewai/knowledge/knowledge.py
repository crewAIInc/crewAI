import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.utilities.logger import Logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # removes logging from fastembed


class Knowledge(BaseModel):
    """
    Knowledge is a collection of sources and setup for the vector store to save and query relevant context.
    
    This class manages knowledge sources and provides methods to query them for relevant information.
    It automatically detects and reloads file-based knowledge sources when their underlying files change.
    
    Args:
        sources: List[BaseKnowledgeSource] = Field(default_factory=list)
            The knowledge sources to use for querying.
        storage: Optional[KnowledgeStorage] = Field(default=None)
            The storage backend for knowledge embeddings.
        embedder: Optional[Dict[str, Any]] = None
            Configuration for the embedding model.
        collection_name: Optional[str] = None
            Name of the collection to use for storage.
    """

    sources: List[BaseKnowledgeSource] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: Optional[KnowledgeStorage] = Field(default=None)
    embedder: Optional[Dict[str, Any]] = None
    collection_name: Optional[str] = None
    _logger: Logger = Logger(verbose=True)

    def __init__(
        self,
        collection_name: str,
        sources: List[BaseKnowledgeSource],
        embedder: Optional[Dict[str, Any]] = None,
        storage: Optional[KnowledgeStorage] = None,
        **data,
    ):
        super().__init__(**data)
        if storage:
            self.storage = storage
        else:
            self.storage = KnowledgeStorage(
                embedder=embedder, collection_name=collection_name
            )
        self.sources = sources
        self.storage.initialize_knowledge_storage()

    def query(
        self, query: List[str], results_limit: int = 3, score_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        """
        Query across all knowledge sources to find the most relevant information.
        Returns the top_k most relevant chunks.

        Raises:
            ValueError: If storage is not initialized.
        """
        if self.storage is None:
            raise ValueError("Storage is not initialized.")
            
        self._check_and_reload_sources()

        results = self.storage.search(
            query,
            limit=results_limit,
            score_threshold=score_threshold,
        )
        return results
        
    def _check_and_reload_sources(self):
        """
        Check if any file-based knowledge sources have changed and reload them if necessary.
        
        This method detects modifications to source files by comparing their modification timestamps
        with previously recorded values. When changes are detected, the source is reloaded and
        the storage is updated with the new content.
        
        Handles specific exceptions for file operations to provide better error reporting.
        """
        for source in self.sources:
            try:
                if hasattr(source, 'files_have_changed') and source.files_have_changed():
                    self._logger.log("info", f"Reloading modified source: {source.__class__.__name__}")
                    source._record_file_mtimes()  # Update timestamps
                    source.content = source.load_content()
                    source.add()  # Reload and update storage
            except FileNotFoundError as e:
                self._logger.log("error", f"File not found when checking for updates: {str(e)}")
            except PermissionError as e:
                self._logger.log("error", f"Permission error when checking for updates: {str(e)}")
            except IOError as e:
                self._logger.log("error", f"IO error when checking for updates: {str(e)}")
            except Exception as e:
                self._logger.log("error", f"Unexpected error when checking for updates: {str(e)}")

    def add_sources(self):
        try:
            for source in self.sources:
                source.storage = self.storage
                source.add()
        except Exception as e:
            raise e

    def reset(self) -> None:
        if self.storage:
            self.storage.reset()
        else:
            raise ValueError("Storage is not initialized.")
