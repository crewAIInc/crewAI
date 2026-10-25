"""Tests that the recall flow degrades gracefully when storage calls fail."""

import json

from crewai.memory.recall_flow import RecallFlow
from crewai.memory.types import MemoryConfig, ScopeInfo

LONG_QUERY = "Summarize everything known about the quarterly rollout plan. " * 6


class _FakeStorage:
    def __init__(self, scopes=None):
        self._scopes = scopes or []

    def list_scopes(self, parent="/"):
        return list(self._scopes)

    def get_scope_info(self, scope):
        return ScopeInfo(path=scope)

    def search(self, embedding, **kwargs):
        return []


class _FailingListScopesStorage(_FakeStorage):
    def list_scopes(self, parent="/"):
        raise RuntimeError("storage temporarily unavailable")


class _FailingScopeInfoStorage(_FakeStorage):
    def get_scope_info(self, scope):
        raise RuntimeError("storage temporarily unavailable")


class _FakeLLM:
    def __init__(self):
        self.user_messages: list[str] = []

    def supports_function_calling(self):
        return False

    def call(self, messages, **kwargs):
        self.user_messages.append(messages[-1]["content"])
        return json.dumps(
            {
                "keywords": [],
                "suggested_scopes": [],
                "complexity": "simple",
                "recall_queries": ["rollout plan"],
                "time_filter": None,
            }
        )


def _fake_embedder(texts):
    return [[0.1, 0.2, 0.3] for _ in texts]


def _kickoff(flow, scope=None):
    flow.kickoff(
        inputs={
            "query": LONG_QUERY,
            "scope": scope,
            "categories": [],
            "limit": 5,
            "source": None,
            "include_private": False,
        }
    )
    return flow.state.final_results


def test_list_scopes_failure_degrades_gracefully():
    flow = RecallFlow(
        storage=_FailingListScopesStorage(),
        llm=_FakeLLM(),
        embedder=_fake_embedder,
        config=MemoryConfig(),
    )
    results = _kickoff(flow)
    assert results == []


def test_get_scope_info_failure_degrades_gracefully():
    flow = RecallFlow(
        storage=_FailingScopeInfoStorage(),
        llm=_FakeLLM(),
        embedder=_fake_embedder,
        config=MemoryConfig(),
    )
    results = _kickoff(flow, scope="/prod")
    assert results == []


def test_available_scopes_are_used_when_listable():
    llm = _FakeLLM()
    flow = RecallFlow(
        storage=_FakeStorage(scopes=["/prod", "/marketing"]),
        llm=llm,
        embedder=_fake_embedder,
        config=MemoryConfig(),
    )
    results = _kickoff(flow)
    assert results == []
    assert "/prod" in llm.user_messages[0]
    assert "/marketing" in llm.user_messages[0]
