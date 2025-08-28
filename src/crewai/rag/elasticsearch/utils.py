"""Utility functions for Elasticsearch RAG implementation."""

import hashlib
from typing import Any, TypeGuard

from crewai.rag.elasticsearch.types import (
    AsyncEmbeddingFunction,
    EmbeddingFunction,
    ElasticsearchClientType,
)
from crewai.rag.types import BaseRecord, SearchResult

try:
    from elasticsearch import Elasticsearch, AsyncElasticsearch
except ImportError:
    Elasticsearch = None
    AsyncElasticsearch = None


def _is_sync_client(client: ElasticsearchClientType) -> TypeGuard[Any]:
    """Type guard to check if the client is a sync Elasticsearch client."""
    if Elasticsearch is None:
        return False
    return isinstance(client, Elasticsearch)


def _is_async_client(client: ElasticsearchClientType) -> TypeGuard[Any]:
    """Type guard to check if the client is an async Elasticsearch client."""
    if AsyncElasticsearch is None:
        return False
    return isinstance(client, AsyncElasticsearch)


def _is_async_embedding_function(
    func: EmbeddingFunction | AsyncEmbeddingFunction,
) -> TypeGuard[AsyncEmbeddingFunction]:
    """Type guard to check if the embedding function is async."""
    import inspect
    return inspect.iscoroutinefunction(func)


def _generate_doc_id(content: str) -> str:
    """Generate a document ID from content using SHA256 hash."""
    return hashlib.sha256(content.encode()).hexdigest()


def _prepare_document_for_elasticsearch(
    doc: BaseRecord, embedding: list[float]
) -> dict[str, Any]:
    """Prepare a document for Elasticsearch indexing.
    
    Args:
        doc: Document record to prepare.
        embedding: Embedding vector for the document.
        
    Returns:
        Document formatted for Elasticsearch.
    """
    doc_id = doc.get("doc_id") or _generate_doc_id(doc["content"])
    
    es_doc = {
        "content": doc["content"],
        "content_vector": embedding,
        "metadata": doc.get("metadata", {}),
    }
    
    return {"id": doc_id, "body": es_doc}


def _process_search_results(
    response: dict[str, Any], score_threshold: float | None = None
) -> list[SearchResult]:
    """Process Elasticsearch search response into SearchResult format.
    
    Args:
        response: Raw Elasticsearch search response.
        score_threshold: Optional minimum score threshold.
        
    Returns:
        List of SearchResult dictionaries.
    """
    results = []
    
    hits = response.get("hits", {}).get("hits", [])
    
    for hit in hits:
        score = hit.get("_score", 0.0)
        
        if score_threshold is not None and score < score_threshold:
            continue
            
        source = hit.get("_source", {})
        
        result = SearchResult(
            id=hit.get("_id", ""),
            content=source.get("content", ""),
            metadata=source.get("metadata", {}),
            score=score,
        )
        results.append(result)
    
    return results


def _build_vector_search_query(
    query_vector: list[float],
    limit: int = 10,
    metadata_filter: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> dict[str, Any]:
    """Build Elasticsearch query for vector similarity search.
    
    Args:
        query_vector: Query embedding vector.
        limit: Maximum number of results.
        metadata_filter: Optional metadata filter.
        score_threshold: Optional minimum score threshold.
        
    Returns:
        Elasticsearch query dictionary.
    """
    query = {
        "size": limit,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }
    
    if metadata_filter:
        bool_query = {
            "bool": {
                "must": [
                    query["query"]
                ],
                "filter": []
            }
        }
        
        for key, value in metadata_filter.items():
            bool_query["bool"]["filter"].append({
                "term": {f"metadata.{key}": value}
            })
        
        query["query"] = bool_query
    
    if score_threshold is not None:
        query["min_score"] = score_threshold
    
    return query


def _get_index_mapping(vector_dimension: int, similarity: str = "cosine") -> dict[str, Any]:
    """Get Elasticsearch index mapping for vector search.
    
    Args:
        vector_dimension: Dimension of the embedding vectors.
        similarity: Similarity function to use.
        
    Returns:
        Elasticsearch mapping dictionary.
    """
    return {
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "content_vector": {
                    "type": "dense_vector",
                    "dims": vector_dimension,
                    "similarity": similarity
                },
                "metadata": {
                    "type": "object",
                    "dynamic": True
                }
            }
        }
    }
