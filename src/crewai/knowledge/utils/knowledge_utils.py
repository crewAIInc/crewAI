from crewai.rag.types import SearchResult


def extract_knowledge_context(knowledge_snippets: list[SearchResult]) -> str:
    """Extract knowledge from the task prompt."""
    valid_snippets = [
        result["content"]
        for result in knowledge_snippets
        if result and result.get("content")
    ]
    snippet = "\n".join(valid_snippets)
    return f"Additional Information: {snippet}" if valid_snippets else ""
