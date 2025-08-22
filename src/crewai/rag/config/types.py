"""Type definitions for RAG configuration."""

from typing import Literal
from crewai.rag.chromadb.config import ChromaDBConfig

RagProvider = Literal["chromadb"]
RagConfigType = ChromaDBConfig
