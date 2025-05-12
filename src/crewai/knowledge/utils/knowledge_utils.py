from typing import Any


def extract_knowledge_context(knowledge_snippets: list[dict[str, Any]]) -> str:
    """Extract knowledge from the task prompt."""
    valid_snippets = [
        result["context"]
        for result in knowledge_snippets
        if result and result.get("context")
    ]
    snippet = "\n".join(valid_snippets)
    return f"Additional Information: {snippet}" if valid_snippets else ""
