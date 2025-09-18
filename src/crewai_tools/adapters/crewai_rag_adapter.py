"""Adapter for CrewAI's native RAG system."""

from typing import Any, TypedDict, TypeAlias
from typing_extensions import Unpack
from pathlib import Path
import hashlib

from pydantic import Field, PrivateAttr
from crewai.rag.config.utils import get_rag_client
from crewai.rag.config.types import RagConfigType
from crewai.rag.types import BaseRecord, SearchResult
from crewai.rag.core.base_client import BaseClient
from crewai.rag.factory import create_client

from crewai_tools.tools.rag.rag_tool import Adapter
from crewai_tools.rag.data_types import DataType
from crewai_tools.rag.misc import sanitize_metadata_for_chromadb
from crewai_tools.rag.chunkers.base_chunker import BaseChunker

ContentItem: TypeAlias = str | Path | dict[str, Any]

class AddDocumentParams(TypedDict, total=False):
    """Parameters for adding documents to the RAG system."""
    data_type: DataType
    metadata: dict[str, Any]
    website: str
    url: str
    file_path: str | Path
    github_url: str
    youtube_url: str
    directory_path: str | Path


class CrewAIRagAdapter(Adapter):
    """Adapter that uses CrewAI's native RAG system.
    
    Supports custom vector database configuration through the config parameter.
    """
    
    collection_name: str = "default"
    summarize: bool = False
    similarity_threshold: float = 0.6
    limit: int = 5
    config: RagConfigType | None = None
    _client: BaseClient | None = PrivateAttr(default=None)
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize the CrewAI RAG client after model initialization."""
        if self.config is not None:
            self._client = create_client(self.config)
        else:
            self._client = get_rag_client()
        self._client.get_or_create_collection(collection_name=self.collection_name)
    
    def query(self, question: str, similarity_threshold: float | None = None, limit: int | None = None) -> str:
        """Query the knowledge base with a question.

        Args:
            question: The question to ask
            similarity_threshold: Minimum similarity score for results (default: 0.6)
            limit: Maximum number of results to return (default: 5)

        Returns:
            Relevant content from the knowledge base
        """
        search_limit = limit if limit is not None else self.limit
        search_threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold

        results: list[SearchResult] = self._client.search(
            collection_name=self.collection_name,
            query=question,
            limit=search_limit,
            score_threshold=search_threshold
        )
        
        if not results:
            return "No relevant content found."
        
        contents: list[str] = []
        for result in results:
            content: str = result.get("content", "")
            if content:
                contents.append(content)
        
        return "\n\n".join(contents)
    
    def add(self, *args: ContentItem, **kwargs: Unpack[AddDocumentParams]) -> None:
        """Add content to the knowledge base.
        
        This method handles various input types and converts them to documents
        for the vector database. It supports the data_type parameter for 
        compatibility with existing tools.
        
        Args:
            *args: Content items to add (strings, paths, or document dicts)
            **kwargs: Additional parameters including data_type, metadata, etc.
        """
        from crewai_tools.rag.data_types import DataTypes, DataType
        from crewai_tools.rag.source_content import SourceContent
        from crewai_tools.rag.base_loader import LoaderResult
        import os
        
        documents: list[BaseRecord] = []
        data_type: DataType | None = kwargs.get("data_type")
        base_metadata: dict[str, Any] = kwargs.get("metadata", {})
        
        for arg in args:
            source_ref: str
            if isinstance(arg, dict):
                source_ref = str(arg.get("source", arg.get("content", "")))
            else:
                source_ref = str(arg)
            
            if not data_type:
                data_type = DataTypes.from_content(source_ref)
            
            if data_type == DataType.DIRECTORY:
                if not os.path.isdir(source_ref):
                    raise ValueError(f"Directory does not exist: {source_ref}")
                
                # Define binary and non-text file extensions to skip
                binary_extensions = {'.pyc', '.pyo', '.png', '.jpg', '.jpeg', '.gif', 
                                    '.bmp', '.ico', '.svg', '.webp', '.pdf', '.zip', 
                                    '.tar', '.gz', '.bz2', '.7z', '.rar', '.exe', 
                                    '.dll', '.so', '.dylib', '.bin', '.dat', '.db',
                                    '.sqlite', '.class', '.jar', '.war', '.ear'}
                
                for root, dirs, files in os.walk(source_ref):
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    
                    for filename in files:
                        if filename.startswith('.'):
                            continue
                        
                        # Skip binary files based on extension
                        file_ext = os.path.splitext(filename)[1].lower()
                        if file_ext in binary_extensions:
                            continue
                        
                        # Skip __pycache__ directories
                        if '__pycache__' in root:
                            continue
                        
                        file_path: str = os.path.join(root, filename)
                        try:
                            file_data_type: DataType = DataTypes.from_content(file_path)
                            file_loader = file_data_type.get_loader()
                            file_chunker = file_data_type.get_chunker()
                            
                            file_source = SourceContent(file_path)
                            file_result: LoaderResult = file_loader.load(file_source)
                            
                            file_chunks = file_chunker.chunk(file_result.content)
                            
                            for chunk_idx, file_chunk in enumerate(file_chunks):
                                file_metadata: dict[str, Any] = base_metadata.copy()
                                file_metadata.update(file_result.metadata)
                                file_metadata["data_type"] = str(file_data_type)
                                file_metadata["file_path"] = file_path
                                file_metadata["chunk_index"] = chunk_idx
                                file_metadata["total_chunks"] = len(file_chunks)
                                
                                if isinstance(arg, dict):
                                    file_metadata.update(arg.get("metadata", {}))
                                
                                chunk_id = hashlib.sha256(f"{file_result.doc_id}_{chunk_idx}_{file_chunk}".encode()).hexdigest()
                                
                                documents.append({
                                    "doc_id": chunk_id,
                                    "content": file_chunk,
                                    "metadata": sanitize_metadata_for_chromadb(file_metadata)
                                })
                        except Exception:
                            # Silently skip files that can't be processed
                            continue
            else:
                metadata: dict[str, Any] = base_metadata.copy()
                
                if data_type in [DataType.PDF_FILE, DataType.TEXT_FILE, DataType.DOCX, 
                                  DataType.CSV, DataType.JSON, DataType.XML, DataType.MDX]:
                    if not os.path.isfile(source_ref):
                        raise FileNotFoundError(f"File does not exist: {source_ref}")
                
                loader = data_type.get_loader()
                chunker = data_type.get_chunker()
                
                source_content = SourceContent(source_ref)
                loader_result: LoaderResult = loader.load(source_content)
                
                chunks = chunker.chunk(loader_result.content)
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata: dict[str, Any] = metadata.copy()
                    chunk_metadata.update(loader_result.metadata)
                    chunk_metadata["data_type"] = str(data_type)
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(chunks)
                    chunk_metadata["source"] = source_ref
                    
                    if isinstance(arg, dict):
                        chunk_metadata.update(arg.get("metadata", {}))
                    
                    chunk_id = hashlib.sha256(f"{loader_result.doc_id}_{i}_{chunk}".encode()).hexdigest()
                    
                    documents.append({
                        "doc_id": chunk_id,
                        "content": chunk,
                        "metadata": sanitize_metadata_for_chromadb(chunk_metadata)
                    })
        
        if documents:
            self._client.add_documents(
                collection_name=self.collection_name,
                documents=documents
            )