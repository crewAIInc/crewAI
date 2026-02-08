"""LLM-powered analysis for memory save and recall."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from crewai.memory.types import ScopeInfo
from crewai.utilities.i18n import get_i18n

_logger = logging.getLogger(__name__)


class ExtractedMetadata(BaseModel):
    """Fixed schema for LLM-extracted metadata (OpenAI requires additionalProperties: false)."""

    model_config = ConfigDict(extra="forbid")

    entities: list[str] = Field(
        default_factory=list,
        description="Entities (people, orgs, places) mentioned in the content.",
    )
    dates: list[str] = Field(
        default_factory=list,
        description="Dates or time references in the content.",
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Topics or themes in the content.",
    )


class MemoryAnalysis(BaseModel):
    """LLM output for analyzing content before saving to memory."""

    suggested_scope: str = Field(
        description="Best matching existing scope or new path (e.g. /company/decisions).",
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Categories for the memory (prefer existing, add new if needed).",
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score from 0.0 to 1.0.",
    )
    extracted_metadata: ExtractedMetadata = Field(
        default_factory=ExtractedMetadata,
        description="Entities, dates, topics extracted from the content.",
    )


class QueryAnalysis(BaseModel):
    """LLM output for analyzing a recall query."""

    keywords: list[str] = Field(
        default_factory=list,
        description="Key entities or keywords for filtering.",
    )
    time_hints: list[str] = Field(
        default_factory=list,
        description="Any time or recency hints in the query.",
    )
    suggested_scopes: list[str] = Field(
        default_factory=list,
        description="Scope paths to search (subset of available scopes).",
    )
    complexity: str = Field(
        default="simple",
        description="One of 'simple' (single fact) or 'complex' (aggregation/reasoning).",
    )


class ExtractedMemories(BaseModel):
    """LLM output for extracting discrete memories from raw content."""

    memories: list[str] = Field(
        default_factory=list,
        description="List of discrete, self-contained memory statements extracted from the content.",
    )


def _get_prompt(key: str) -> str:
    """Retrieve a memory prompt from the i18n translations.

    Args:
        key: The prompt key under the "memory" section.

    Returns:
        The prompt string.
    """
    return get_i18n().memory(key)


def extract_memories_from_content(content: str, llm: Any) -> list[str]:
    """Use the LLM to extract discrete memory statements from raw content.

    This is a pure helper: it does NOT store anything. Callers should call
    memory.remember() on each returned string to persist them.

    On LLM failure, returns the full content as a single memory so callers
    still persist something rather than dropping the output.

    Args:
        content: Raw text (e.g. task description + result dump).
        llm: The LLM instance to use.

    Returns:
        List of short, self-contained memory statements (or [content] on failure).
    """
    if not (content or "").strip():
        return []
    user = f"Content:\n{content}\n\nExtract memory statements as described. Return structured output."
    messages = [
        {"role": "system", "content": _get_prompt("extract_memories_system")},
        {"role": "user", "content": user},
    ]
    try:
        if getattr(llm, "supports_function_calling", lambda: False)():
            response = llm.call(messages, response_model=ExtractedMemories)
            if isinstance(response, ExtractedMemories):
                return response.memories
            return ExtractedMemories.model_validate(response).memories
        response = llm.call(messages)
        if isinstance(response, ExtractedMemories):
            return response.memories
        if isinstance(response, str):
            data = json.loads(response)
            return ExtractedMemories.model_validate(data).memories
        return ExtractedMemories.model_validate(response).memories
    except Exception as e:
        _logger.warning(
            "Memory extraction failed, storing full content as single memory: %s",
            e,
            exc_info=False,
        )
        return [content]


async def aextract_memories_from_content(content: str, llm: Any) -> list[str]:
    """Async variant of extract_memories_from_content."""
    return extract_memories_from_content(content, llm)


def analyze_for_save(
    content: str,
    existing_scopes: list[str],
    existing_categories: list[str],
    llm: Any,
) -> MemoryAnalysis:
    """Use the LLM to infer scope, categories, importance, and metadata for a memory.

    On LLM failure, returns safe defaults so remember() still persists the content.

    Args:
        content: The memory content to analyze.
        existing_scopes: Current scope paths in the memory store.
        existing_categories: Current categories in use.
        llm: The LLM instance to use.

    Returns:
        MemoryAnalysis with suggested_scope, categories, importance, extracted_metadata.
    """
    user = (
        f"Content to store:\n{content}\n\n"
        f"Existing scopes: {existing_scopes or ['/']}\n"
        f"Existing categories: {existing_categories or []}\n\n"
        "Return the analysis as structured output."
    )
    messages = [
        {"role": "system", "content": _get_prompt("save_system")},
        {"role": "user", "content": user},
    ]
    try:
        if getattr(llm, "supports_function_calling", lambda: False)():
            response = llm.call(messages, response_model=MemoryAnalysis)
            if isinstance(response, MemoryAnalysis):
                return response
            return MemoryAnalysis.model_validate(response)
        response = llm.call(messages)
        if isinstance(response, MemoryAnalysis):
            return response
        if isinstance(response, str):
            data = json.loads(response)
            return MemoryAnalysis.model_validate(data)
        return MemoryAnalysis.model_validate(response)
    except Exception as e:
        _logger.warning(
            "Memory save analysis failed, using defaults (scope=/, importance=0.5): %s",
            e,
            exc_info=False,
        )
        return MemoryAnalysis(
            suggested_scope="/",
            categories=[],
            importance=0.5,
            extracted_metadata=ExtractedMetadata(),
        )


async def aanalyze_for_save(
    content: str,
    existing_scopes: list[str],
    existing_categories: list[str],
    llm: Any,
) -> MemoryAnalysis:
    """Async variant of analyze_for_save."""
    # Fallback to sync if no acall
    return analyze_for_save(content, existing_scopes, existing_categories, llm)


def analyze_query(
    query: str,
    available_scopes: list[str],
    scope_info: ScopeInfo | None,
    llm: Any,
) -> QueryAnalysis:
    """Use the LLM to analyze a recall query.

    On LLM failure, returns safe defaults so recall degrades to plain vector search.

    Args:
        query: The user's recall query.
        available_scopes: Scope paths that exist in the store.
        scope_info: Optional info about the current scope.
        llm: The LLM instance to use.

    Returns:
        QueryAnalysis with keywords, time_hints, suggested_scopes, complexity.
    """
    scope_desc = ""
    if scope_info:
        scope_desc = f"Current scope has {scope_info.record_count} records, categories: {scope_info.categories}"
    user = (
        f"Query: {query}\n\n"
        f"Available scopes: {available_scopes or ['/']}\n"
        f"{scope_desc}\n\n"
        "Return the analysis as structured output."
    )
    messages = [
        {"role": "system", "content": _get_prompt("query_system")},
        {"role": "user", "content": user},
    ]
    try:
        if getattr(llm, "supports_function_calling", lambda: False)():
            response = llm.call(messages, response_model=QueryAnalysis)
            if isinstance(response, QueryAnalysis):
                return response
            return QueryAnalysis.model_validate(response)
        response = llm.call(messages)
        if isinstance(response, QueryAnalysis):
            return response
        if isinstance(response, str):
            data = json.loads(response)
            return QueryAnalysis.model_validate(data)
        return QueryAnalysis.model_validate(response)
    except Exception as e:
        _logger.warning(
            "Query analysis failed, using defaults (complexity=simple): %s",
            e,
            exc_info=False,
        )
        scopes = (available_scopes or ["/"])[:5]
        return QueryAnalysis(
            keywords=[],
            time_hints=[],
            suggested_scopes=scopes,
            complexity="simple",
        )


async def aanalyze_query(
    query: str,
    available_scopes: list[str],
    scope_info: ScopeInfo | None,
    llm: Any,
) -> QueryAnalysis:
    """Async variant of analyze_query."""
    return analyze_query(query, available_scopes, scope_info, llm)
