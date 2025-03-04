from typing import Any, Dict, List


def extract_knowledge_context(knowledge_snippets: List[Dict[str, Any]]) -> str:
    """Extract knowledge from the task prompt and format it for effective LLM usage."""
    valid_snippets = [
        result["context"]
        for result in knowledge_snippets
        if result and result.get("context")
    ]
    if not valid_snippets:
        return ""
    
    snippet = "\n".join(valid_snippets)
    return (
        "Important Context (You MUST use this information to complete your task "
        "accurately and effectively):\n"
        f"{snippet}\n\n"
        "Make sure to incorporate the above context into your response."
    )
