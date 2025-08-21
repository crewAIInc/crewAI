"""Type definitions specific to Qdrant implementation."""

from qdrant_client import AsyncQdrantClient, QdrantClient as SyncQdrantClient

QdrantClientType = SyncQdrantClient | AsyncQdrantClient
