import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import chromadb
import litellm
from pydantic import BaseModel, Field, PrivateAttr

from crewai_tools.tools.rag.rag_tool import Adapter
from crewai_tools.rag.data_types import  DataType
from crewai_tools.rag.base_loader import BaseLoader
from crewai_tools.rag.chunkers.base_chunker import BaseChunker
from crewai_tools.rag.source_content import SourceContent
from crewai_tools.rag.misc import compute_sha256

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model: str = "text-embedding-3-small", **kwargs):
        self.model = model
        self.kwargs = kwargs

    def embed_text(self, text: str) -> List[float]:
        try:
            response = litellm.embedding(
                model=self.model,
                input=[text],
                **self.kwargs
            )
            return response.data[0]['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        try:
            response = litellm.embedding(
                model=self.model,
                input=texts,
                **self.kwargs
            )
            return [data['embedding'] for data in response.data]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    data_type: DataType = DataType.TEXT
    source: Optional[str] = None


class RAG(Adapter):
    collection_name: str = "crewai_knowledge_base"
    persist_directory: Optional[str] = None
    embedding_model: str = "text-embedding-3-large"
    summarize: bool = False
    top_k: int = 5
    embedding_config: Dict[str, Any] = Field(default_factory=dict)

    _client: Any = PrivateAttr()
    _collection: Any = PrivateAttr()
    _embedding_service: EmbeddingService = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        try:
            if self.persist_directory:
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "description": "CrewAI Knowledge Base"}
            )

            self._embedding_service = EmbeddingService(model=self.embedding_model, **self.embedding_config)
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

        super().model_post_init(__context)

    def add(
        self,
        content: str | Path,
        data_type: Optional[Union[str, DataType]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        loader: Optional[BaseLoader] = None,
        chunker: Optional[BaseChunker] = None,
        **kwargs: Any
    ) -> None:
        source_content = SourceContent(content)

        data_type = self._get_data_type(data_type=data_type, content=source_content)

        if not loader:
            loader = data_type.get_loader()

        if not chunker:
            chunker = data_type.get_chunker()

        loader_result = loader.load(source_content)
        doc_id = loader_result.doc_id

        existing_doc = self._collection.get(where={"source": source_content.source_ref}, limit=1)
        existing_doc_id = existing_doc and existing_doc['metadatas'][0]['doc_id'] if existing_doc['metadatas'] else None

        if existing_doc_id == doc_id:
            logger.warning(f"Document with source {loader_result.source} already exists")
            return

        # Document with same source ref does exists but the content has changed, deleting the oldest reference
        if existing_doc_id and existing_doc_id != loader_result.doc_id:
            logger.warning(f"Deleting old document with doc_id {existing_doc_id}")
            self._collection.delete(where={"doc_id": existing_doc_id})

        documents = []

        chunks = chunker.chunk(loader_result.content)
        for i, chunk in enumerate(chunks):
            doc_metadata = (metadata or {}).copy()
            doc_metadata['chunk_index'] = i
            documents.append(Document(
                id=compute_sha256(chunk),
                content=chunk,
                metadata=doc_metadata,
                data_type=data_type,
                source=loader_result.source
            ))

        if not documents:
            logger.warning("No documents to add")
            return

        contents = [doc.content for doc in documents]
        try:
            embeddings = self._embedding_service.embed_batch(contents)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return

        ids = [doc.id for doc in documents]
        metadatas = []

        for doc in documents:
            doc_metadata = doc.metadata.copy()
            doc_metadata.update({
                "data_type": doc.data_type.value,
                "source": doc.source,
                "doc_id": doc_id
            })
            metadatas.append(doc_metadata)

        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
            )
            logger.info(f"Added {len(documents)} documents to knowledge base")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")

    def query(self, question: str, where: Optional[Dict[str, Any]] = None) -> str:
        try:
            question_embedding = self._embedding_service.embed_text(question)

            results = self._collection.query(
                query_embeddings=[question_embedding],
                n_results=self.top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            if not results or not results.get("documents") or not results["documents"][0]:
                return "No relevant content found."

            documents = results["documents"][0]
            metadatas = results.get("metadatas", [None])[0] or []
            distances = results.get("distances", [None])[0] or []

            # Return sources with relevance scores
            formatted_results = []
            for i, doc in enumerate(documents):
                metadata = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else 1.0
                source = metadata.get("source", "unknown") if metadata else "unknown"
                score = 1 - distance if distance is not None else 0  # Convert distance to similarity
                formatted_results.append(f"[Source: {source}, Relevance: {score:.3f}]\n{doc}")

            return "\n\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Error querying knowledge base: {e}"

    def delete_collection(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")

    def get_collection_info(self) -> Dict[str, Any]:
        try:
            count = self._collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "embedding_model": self.embedding_model
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}

    def _get_data_type(self, content: SourceContent, data_type: str | DataType | None = None) -> DataType:
        try:
            if isinstance(data_type, str):
                return DataType(data_type)
        except Exception as e:
            pass

        return content.data_type
