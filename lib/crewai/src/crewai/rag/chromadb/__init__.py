"""ChromaDB RAG provider. Requires the optional ``chromadb`` extra."""

try:
    import chromadb  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The 'chromadb' provider requires chromadb. "
        "Install it with: pip install crewai[chromadb]"
    ) from e
