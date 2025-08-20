"""Type definitions specific to ChromaDB implementation."""

from chromadb.api import ClientAPI, AsyncClientAPI

ChromaDBClientType = ClientAPI | AsyncClientAPI
