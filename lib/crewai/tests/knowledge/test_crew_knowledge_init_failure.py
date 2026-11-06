"""Crew knowledge_sources must fail loudly when initialization fails."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from crewai import Agent, Crew, Process, Task
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.llms.base_llm import BaseLLM


class _StubLLM(BaseLLM):
    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        return "Thought: ok.\nFinal Answer: stub"

    def supports_function_calling(self) -> bool:
        return False


def test_crew_raises_when_knowledge_sources_fail_without_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit knowledge_sources must not silently degrade to empty knowledge."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    knowledge = StringKnowledgeSource(content="Secret fact: the vault code is 42.")
    agent = Agent(
        role="tester",
        goal="answer from knowledge",
        backstory="bg",
        llm=_StubLLM(model="stub"),
        verbose=False,
    )
    task = Task(
        description="What is the vault code?",
        expected_output="the code",
        agent=agent,
    )

    with pytest.raises(ValueError, match="Failed to initialize crew knowledge_sources"):
        Crew(
            agents=[agent],
            tasks=[task],
            knowledge_sources=[knowledge],
            process=Process.sequential,
            verbose=False,
        )


def test_chromadb_default_embedder_docstring_mentions_openai() -> None:
    from crewai.rag.chromadb import config as chroma_config

    doc = chroma_config._default_embedding_function.__doc__ or ""
    assert "OpenAI" in doc
    assert "all-MiniLM-L6-v2" not in doc
    # Construction still requires an API key when env is empty.
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            chroma_config._default_embedding_function()
