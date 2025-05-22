from typing import Any, Dict, List, Optional
import hashlib
import logging
import os
from sqlalchemy import create_engine, Column, String, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector

from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from crewai.utilities import EmbeddingConfigurator

Base = declarative_base()

class Document(Base):
    """SQLAlchemy model for document storage with pgvector."""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)
    content = Column(Text)
    metadata = Column(Text)  # JSON serialized metadata
    embedding = Column(Vector(1536))  # Adjust dimension based on embedding model

class PGVectorKnowledgeStorage(BaseKnowledgeStorage):
    """
    Knowledge storage implementation using pgvector.
    
    This class provides an implementation of BaseKnowledgeStorage using PostgreSQL
    with the pgvector extension for vector similarity search.
    """
    
    def __init__(
        self,
        connection_string: str,
        embedder: Optional[Dict[str, Any]] = None,
        table_name: str = "documents",
        embedding_dimension: int = 1536,
    ):
        """
        Initialize the pgvector knowledge storage.
        
        Args:
            connection_string: PostgreSQL connection string
            embedder: Configuration dictionary for the embedder
            table_name: Name of the table to store documents
            embedding_dimension: Dimension of the embedding vectors
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        
        self._set_embedder_config(embedder)
        
        Base.metadata.create_all(self.engine)
    
    def _set_embedder_config(self, embedder: Optional[Dict[str, Any]] = None) -> None:
        """
        Set the embedding configuration for the knowledge storage.
        
        Args:
            embedder_config: Configuration dictionary for the embedder.
                If None or empty, defaults to the default embedding function.
        """
        self.embedder = (
            EmbeddingConfigurator().configure_embedder(embedder)
            if embedder
            else self._create_default_embedding_function()
        )
    
    def search(
        self,
        query: List[str],
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents in the knowledge base.
        
        Args:
            query: List of query strings
            limit: Maximum number of results to return
            filter: Optional metadata filter
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with id, metadata, context, and score
        """
        session = self.Session()
        
        try:
            query_embedding = self.embedder([query[0]])[0]
            
            sql_query = f"""
            SELECT id, content, metadata, 1 - (embedding <=> '{query_embedding}') as similarity
            FROM {self.table_name}
            ORDER BY embedding <=> '{query_embedding}'
            LIMIT {limit}
            """
            
            results = session.execute(sql_query).fetchall()
            
            formatted_results = []
            for row in results:
                similarity = float(row[3])
                if similarity >= score_threshold:
                    formatted_results.append({
                        "id": row[0],
                        "context": row[1],
                        "metadata": row[2],
                        "score": similarity,
                    })
            
            return formatted_results
        finally:
            session.close()
    
    def save(
        self,
        documents: List[str],
        metadata: Optional[Dict[str, Any] | List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Save documents to the knowledge base.
        
        Args:
            documents: List of document strings
            metadata: Optional metadata for the documents
        """
        session = self.Session()
        
        try:
            unique_docs = {}
            
            for idx, doc in enumerate(documents):
                doc_id = hashlib.sha256(doc.encode("utf-8")).hexdigest()
                doc_metadata = None
                if metadata is not None:
                    if isinstance(metadata, list):
                        doc_metadata = metadata[idx]
                    else:
                        doc_metadata = metadata
                unique_docs[doc_id] = (doc, doc_metadata)
            
            docs_list = [doc for doc, _ in unique_docs.values()]
            embeddings = self.embedder(docs_list)
            
            for i, (doc_id, (doc, meta)) in enumerate(unique_docs.items()):
                embedding = embeddings[i]
                
                existing = session.query(Document).filter(Document.id == doc_id).first()
                
                if existing:
                    existing.content = doc
                    existing.metadata = str(meta) if meta else None
                    existing.embedding = embedding
                else:
                    new_doc = Document(
                        id=doc_id,
                        content=doc,
                        metadata=str(meta) if meta else None,
                        embedding=embedding,
                    )
                    session.add(new_doc)
            
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"Failed to save documents: {e}")
            raise
        finally:
            session.close()
    
    def reset(self) -> None:
        """Reset the knowledge base by dropping and recreating the table."""
        session = self.Session()
        try:
            session.query(Document).delete()
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"Failed to reset knowledge base: {e}")
            raise
        finally:
            session.close()
    
    def _create_default_embedding_function(self):
        """Create a default embedding function for the knowledge storage."""
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )
        
        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )
