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
        
        The method handles various file-related exceptions with specific error messages:
        - FileNotFoundError: When a source file no longer exists
        - PermissionError: When there are permission issues accessing a file
        - IOError: When there are I/O errors reading a file
        - ValueError: When there are issues with file content format
        - Other unexpected exceptions are also caught and logged
        
        Each exception is logged with appropriate context to aid in troubleshooting.
        """
        for source in self.sources:
            source_name = source.__class__.__name__
            try:
                if hasattr(source, 'files_have_changed') and source.files_have_changed():
                    self._logger.log("info", f"Reloading modified source: {source_name}")
                    
                    # Update file modification timestamps
                    try:
                        source._record_file_mtimes()
                    except (PermissionError, IOError) as e:
                        self._logger.log("warning", f"Could not record file timestamps for {source_name}: {str(e)}")
                    
                    try:
                        source.content = source.load_content()
                    except FileNotFoundError as e:
                        self._logger.log("error", f"File not found when loading content for {source_name}: {str(e)}")
                        continue
                    except PermissionError as e:
                        self._logger.log("error", f"Permission error when loading content for {source_name}: {str(e)}")
                        continue
                    except IOError as e:
                        self._logger.log("error", f"IO error when loading content for {source_name}: {str(e)}")
                        continue
                    except ValueError as e:
                        self._logger.log("error", f"Invalid content format in {source_name}: {str(e)}")
                        continue
                    
                    try:
                        source.add()
                        self._logger.log("info", f"Successfully reloaded and updated {source_name}")
                    except Exception as e:
                        self._logger.log("error", f"Failed to update storage for {source_name}: {str(e)}")
                        
            except FileNotFoundError as e:
                self._logger.log("error", f"File not found when checking for updates in {source_name}: {str(e)}")
            except PermissionError as e:
                self._logger.log("error", f"Permission error when checking for updates in {source_name}: {str(e)}")
            except IOError as e:
                self._logger.log("error", f"IO error when checking for updates in {source_name}: {str(e)}")
            except Exception as e:
                self._logger.log("error", f"Unexpected error when checking for updates in {source_name}: {str(e)}")

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
