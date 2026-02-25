"""Unit tests for MengramMemory (Mengram memory backend for CrewAI).

All tests are fully mocked -- no real API calls are made.
"""

from __future__ import annotations

import os
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from crewai.memory.storage.mengram_storage import (
    MengramConfig,
    MengramMemory,
    _MengramClient,
    _chunk_to_match,
    _episodic_to_match,
    _procedural_to_match,
    _semantic_to_match,
)
from crewai.memory.types import MemoryMatch, MemoryRecord, ScopeInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides: object) -> MengramConfig:
    defaults: dict[str, object] = {
        "api_key": "om-test-key-123",
        "base_url": "https://mengram.io",
        "user_id": "test-user",
    }
    defaults.update(overrides)
    return MengramConfig(**defaults)


def _make_memory(**config_overrides: object) -> tuple[MengramMemory, MagicMock]:
    """Return a ``MengramMemory`` whose ``_client`` is a ``MagicMock``."""
    cfg = _make_config(**config_overrides)
    memory = MengramMemory(config=cfg)
    mock_client = MagicMock(spec=_MengramClient)
    memory._client = mock_client
    return memory, mock_client


# -- Sample API responses ---------------------------------------------------

SEMANTIC_RESULT = {
    "entity": "PostgreSQL",
    "type": "technology",
    "score": 0.85,
    "facts": ["uses port 5432", "supports JSONB"],
    "knowledge": [{"title": "Config", "content": "max_connections=100"}],
}

EPISODIC_RESULT = {
    "summary": "Deployed v2.0 to production",
    "outcome": "success",
    "when": "2026-02-20",
    "score": 0.7,
}

PROCEDURAL_RESULT = {
    "id": "proc-123",
    "name": "Deploy to prod",
    "steps": [{"action": "run tests"}, {"action": "push to main"}],
    "success_count": 5,
    "fail_count": 1,
    "score": 0.65,
}

CHUNK_RESULT = {
    "content": "We fixed the OOM with Redis cache.",
    "score": 0.4,
}

SEARCH_ALL_RESPONSE = {
    "semantic": [SEMANTIC_RESULT],
    "episodic": [EPISODIC_RESULT],
    "procedural": [PROCEDURAL_RESULT],
    "chunks": [CHUNK_RESULT],
}


# ===========================================================================
# TestMengramConfig
# ===========================================================================

class TestMengramConfig:
    def test_valid_config(self) -> None:
        cfg = _make_config()
        assert cfg.api_key == "om-test-key-123"
        assert cfg.base_url == "https://mengram.io"
        assert cfg.user_id == "test-user"

    def test_defaults(self) -> None:
        cfg = _make_config()
        assert cfg.graph_depth == 2
        assert cfg.search_limit == 5

    def test_api_key_required(self) -> None:
        with pytest.raises(ValidationError, match="api_key is required"):
            MengramConfig(api_key=None)

    @patch.dict(os.environ, {"MENGRAM_API_KEY": "om-from-env"}, clear=False)
    def test_api_key_from_env(self) -> None:
        cfg = MengramConfig()
        assert cfg.api_key == "om-from-env"

    @patch.dict(os.environ, {"MENGRAM_API_KEY": "om-from-env"}, clear=False)
    def test_explicit_api_key_overrides_env(self) -> None:
        cfg = MengramConfig(api_key="om-explicit")
        assert cfg.api_key == "om-explicit"

    @patch.dict(os.environ, {"MENGRAM_BASE_URL": "https://custom.example.com"}, clear=False)
    def test_base_url_from_env(self) -> None:
        cfg = _make_config()
        assert cfg.base_url == "https://custom.example.com"

    def test_empty_base_url_rejected(self) -> None:
        with pytest.raises(ValidationError, match="base_url must not be empty"):
            _make_config(base_url="")

    def test_empty_user_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="user_id must not be empty"):
            _make_config(user_id="")

    def test_whitespace_user_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="user_id must not be empty"):
            _make_config(user_id="   ")

    def test_base_url_trailing_slash_stripped(self) -> None:
        cfg = _make_config(base_url="https://mengram.io/")
        assert cfg.base_url == "https://mengram.io"

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            _make_config(nonexistent_field="oops")

    def test_all_fields_set(self) -> None:
        cfg = MengramConfig(
            api_key="om-key",
            base_url="https://custom.io",
            user_id="alice",
            graph_depth=3,
            search_limit=10,
        )
        assert cfg.api_key == "om-key"
        assert cfg.base_url == "https://custom.io"
        assert cfg.user_id == "alice"
        assert cfg.graph_depth == 3
        assert cfg.search_limit == 10


# ===========================================================================
# TestMengramMemoryRemember
# ===========================================================================

class TestMengramMemoryRemember:
    def test_calls_add_text(self) -> None:
        mem, client = _make_memory()
        client.add_text.return_value = {"status": "accepted", "job_id": "job-1"}
        mem.remember("hello world")
        client.add_text.assert_called_once_with(text="hello world", user_id="test-user")

    def test_returns_memory_record(self) -> None:
        mem, client = _make_memory()
        client.add_text.return_value = {"status": "accepted", "job_id": "job-2"}
        record = mem.remember("facts about AI", scope="/research", categories=["tech"])
        assert isinstance(record, MemoryRecord)
        assert record.content == "facts about AI"
        assert record.scope == "/research"
        assert record.categories == ["tech"]
        assert record.metadata["mengram_job_id"] == "job-2"

    def test_with_agent_role_enriches_text(self) -> None:
        mem, client = _make_memory()
        client.add_text.return_value = {"status": "accepted", "job_id": "job-3"}
        mem.remember("task result", agent_role="Researcher")
        call_args = client.add_text.call_args
        assert "[Agent: Researcher]" in call_args.kwargs["text"]

    def test_with_metadata(self) -> None:
        mem, client = _make_memory()
        client.add_text.return_value = {"status": "accepted", "job_id": "job-4"}
        record = mem.remember("data", metadata={"project": "alpha"})
        assert record.metadata["project"] == "alpha"
        assert record.metadata["mengram_job_id"] == "job-4"

    def test_api_error_returns_record_with_empty_job_id(self) -> None:
        mem, client = _make_memory()
        client.add_text.side_effect = RuntimeError("API down")
        record = mem.remember("test content")
        assert isinstance(record, MemoryRecord)
        assert record.content == "test content"
        assert record.metadata["mengram_job_id"] == ""

    def test_with_source(self) -> None:
        mem, client = _make_memory()
        client.add_text.return_value = {"status": "accepted", "job_id": "j"}
        record = mem.remember("data", source="session-42")
        assert record.source == "session-42"
        assert record.metadata["source"] == "session-42"


# ===========================================================================
# TestMengramMemoryRememberMany
# ===========================================================================

class TestMengramMemoryRememberMany:
    def test_returns_empty_list(self) -> None:
        mem, client = _make_memory()
        client.add_text.return_value = {"status": "accepted", "job_id": "j"}
        result = mem.remember_many(["a", "b", "c"])
        assert result == []

    def test_empty_list_is_noop(self) -> None:
        mem, client = _make_memory()
        result = mem.remember_many([])
        assert result == []
        client.add_text.assert_not_called()

    def test_combines_content(self) -> None:
        mem, client = _make_memory()
        client.add_text.return_value = {"status": "accepted", "job_id": "j"}
        mem.remember_many(["fact 1", "fact 2"])
        mem.drain_writes()
        call_args = client.add_text.call_args
        assert "fact 1" in call_args.kwargs["text"]
        assert "fact 2" in call_args.kwargs["text"]

    def test_with_agent_role(self) -> None:
        mem, client = _make_memory()
        client.add_text.return_value = {"status": "accepted", "job_id": "j"}
        mem.remember_many(["x"], agent_role="Coder")
        mem.drain_writes()
        call_args = client.add_text.call_args
        assert "[Agent: Coder]" in call_args.kwargs["text"]


# ===========================================================================
# TestMengramMemoryRecall
# ===========================================================================

class TestMengramMemoryRecall:
    def test_deep_returns_all_types(self) -> None:
        mem, client = _make_memory()
        client.search_all.return_value = SEARCH_ALL_RESPONSE
        matches = mem.recall("database", depth="deep")
        types = {m.match_reasons[0] for m in matches}
        assert "semantic" in types
        assert "episodic" in types
        assert "procedural" in types

    def test_shallow_semantic_only(self) -> None:
        mem, client = _make_memory()
        client.search.return_value = [SEMANTIC_RESULT]
        matches = mem.recall("database", depth="shallow")
        assert len(matches) == 1
        assert matches[0].match_reasons == ["semantic"]
        client.search_all.assert_not_called()

    def test_sorted_by_score(self) -> None:
        mem, client = _make_memory()
        client.search_all.return_value = SEARCH_ALL_RESPONSE
        matches = mem.recall("test", depth="deep")
        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_respects_limit(self) -> None:
        mem, client = _make_memory()
        client.search_all.return_value = SEARCH_ALL_RESPONSE
        matches = mem.recall("test", depth="deep", limit=2)
        assert len(matches) <= 2

    def test_empty_results(self) -> None:
        mem, client = _make_memory()
        client.search_all.return_value = {"semantic": [], "episodic": [], "procedural": []}
        matches = mem.recall("nothing", depth="deep")
        assert matches == []

    def test_api_error_returns_empty(self) -> None:
        mem, client = _make_memory()
        client.search_all.side_effect = RuntimeError("timeout")
        matches = mem.recall("query", depth="deep")
        assert matches == []

    def test_drains_writes_before_search(self) -> None:
        mem, client = _make_memory()
        client.search_all.return_value = {"semantic": [], "episodic": [], "procedural": []}

        drain_called = False
        original_drain = mem.drain_writes

        def mock_drain() -> None:
            nonlocal drain_called
            drain_called = True
            original_drain()

        mem.drain_writes = mock_drain
        mem.recall("test")
        assert drain_called

    def test_episodic_formatting(self) -> None:
        mem, client = _make_memory()
        client.search_all.return_value = {
            "semantic": [],
            "episodic": [EPISODIC_RESULT],
            "procedural": [],
        }
        matches = mem.recall("deploy", depth="deep")
        assert len(matches) == 1
        assert "Deployed v2.0 to production" in matches[0].record.content
        assert "success" in matches[0].record.content
        assert "2026-02-20" in matches[0].record.content


# ===========================================================================
# TestMengramMemoryForget
# ===========================================================================

class TestMengramMemoryForget:
    def test_with_entity_name(self) -> None:
        mem, client = _make_memory()
        client.delete_entity.return_value = {"status": "deleted"}
        count = mem.forget(metadata_filter={"entity": "PostgreSQL"})
        assert count == 1
        client.delete_entity.assert_called_once_with("PostgreSQL", user_id="test-user")

    def test_without_entity_calls_reset(self) -> None:
        mem, client = _make_memory()
        client.delete_all.return_value = {"status": "deleted", "count": 5}
        count = mem.forget()
        assert count == 0
        client.delete_all.assert_called_once()

    def test_api_error(self) -> None:
        mem, client = _make_memory()
        client.delete_entity.side_effect = RuntimeError("fail")
        count = mem.forget(metadata_filter={"entity": "X"})
        assert count == 0


# ===========================================================================
# TestMengramMemoryReset
# ===========================================================================

class TestMengramMemoryReset:
    def test_calls_delete_all(self) -> None:
        mem, client = _make_memory()
        client.delete_all.return_value = {"status": "deleted", "count": 3}
        mem.reset()
        client.delete_all.assert_called_once_with(user_id="test-user")

    def test_api_error_no_exception(self) -> None:
        mem, client = _make_memory()
        client.delete_all.side_effect = RuntimeError("fail")
        mem.reset()  # should not raise


# ===========================================================================
# TestMengramMemoryDrainWrites
# ===========================================================================

class TestMengramMemoryDrainWrites:
    def test_waits_for_futures(self) -> None:
        mem, _client = _make_memory()
        future = Future()
        future.set_result(None)
        with mem._pending_lock:
            mem._pending_saves.append(future)
        mem.drain_writes()
        # should complete without error

    def test_handles_failed_future(self) -> None:
        mem, _client = _make_memory()
        future = Future()
        future.set_exception(RuntimeError("boom"))
        with mem._pending_lock:
            mem._pending_saves.append(future)
        mem.drain_writes()  # should not raise


# ===========================================================================
# TestMengramMemoryClose
# ===========================================================================

class TestMengramMemoryClose:
    def test_drains_and_shuts_down(self) -> None:
        mem, _client = _make_memory()
        mem.close()
        assert mem._save_pool._shutdown

    def test_close_twice_safe(self) -> None:
        mem, _client = _make_memory()
        mem.close()
        mem.close()  # should not raise


# ===========================================================================
# TestMengramMemoryScope
# ===========================================================================

class TestMengramMemoryScope:
    def test_returns_self(self) -> None:
        mem, _client = _make_memory()
        assert mem.scope("/test") is mem

    def test_list_scopes(self) -> None:
        mem, _client = _make_memory()
        assert mem.list_scopes() == ["/"]


# ===========================================================================
# TestMengramMemoryInfo
# ===========================================================================

class TestMengramMemoryInfo:
    def test_returns_scope_info(self) -> None:
        mem, client = _make_memory()
        client.stats.return_value = {"entities": 10, "facts": 50, "episodes": 5}
        info = mem.info()
        assert isinstance(info, ScopeInfo)
        assert info.record_count == 60  # entities + facts
        assert info.path == "/"

    def test_maps_entity_types(self) -> None:
        mem, client = _make_memory()
        client.stats.return_value = {
            "entities": 3,
            "facts": 10,
            "by_type": {"person": 2, "technology": 1},
        }
        info = mem.info()
        assert "person" in info.categories
        assert "technology" in info.categories

    def test_api_error_returns_empty(self) -> None:
        mem, client = _make_memory()
        client.stats.side_effect = RuntimeError("fail")
        info = mem.info()
        assert isinstance(info, ScopeInfo)
        assert info.record_count == 0


# ===========================================================================
# TestMengramMemoryTree
# ===========================================================================

class TestMengramMemoryTree:
    def test_tree_format(self) -> None:
        mem, client = _make_memory()
        client.stats.return_value = {"entities": 5, "facts": 20}
        result = mem.tree()
        assert "/ (25 records)" in result


# ===========================================================================
# TestResultConversion
# ===========================================================================

class TestResultConversion:
    def test_semantic_to_match(self) -> None:
        match = _semantic_to_match(SEMANTIC_RESULT)
        assert isinstance(match, MemoryMatch)
        assert "PostgreSQL" in match.record.content
        assert "port 5432" in match.record.content
        assert match.score == 0.85
        assert match.match_reasons == ["semantic"]
        assert match.record.metadata["entity"] == "PostgreSQL"

    def test_episodic_to_match(self) -> None:
        match = _episodic_to_match(EPISODIC_RESULT)
        assert "Deployed v2.0" in match.record.content
        assert "success" in match.record.content
        assert match.match_reasons == ["episodic"]

    def test_procedural_to_match(self) -> None:
        match = _procedural_to_match(PROCEDURAL_RESULT)
        assert "Deploy to prod" in match.record.content
        assert "run tests" in match.record.content
        assert match.record.metadata["reliability"] == 5 / 6
        assert match.match_reasons == ["procedural"]

    def test_chunk_to_match(self) -> None:
        match = _chunk_to_match(CHUNK_RESULT)
        assert "Redis cache" in match.record.content
        assert match.score == 0.4
        assert match.match_reasons == ["chunk"]

    def test_semantic_empty_facts(self) -> None:
        match = _semantic_to_match({"entity": "X", "score": 0.5, "facts": []})
        assert match.record.content == "X"

    def test_procedural_zero_runs(self) -> None:
        match = _procedural_to_match({
            "name": "New Proc",
            "steps": [],
            "success_count": 0,
            "fail_count": 0,
        })
        assert match.record.metadata["reliability"] == 0.5


# ===========================================================================
# TestAsyncMethods
# ===========================================================================

class TestAsyncMethods:
    @pytest.mark.asyncio
    async def test_aremember(self) -> None:
        mem, client = _make_memory()
        client.add_text.return_value = {"status": "accepted", "job_id": "j"}
        record = await mem.aremember("hello")
        assert isinstance(record, MemoryRecord)

    @pytest.mark.asyncio
    async def test_arecall(self) -> None:
        mem, client = _make_memory()
        client.search_all.return_value = {"semantic": [], "episodic": [], "procedural": []}
        matches = await mem.arecall("query")
        assert matches == []

    @pytest.mark.asyncio
    async def test_aextract_memories(self) -> None:
        mem, _client = _make_memory()
        result = await mem.aextract_memories("some content")
        assert result == ["some content"]


# ===========================================================================
# TestMengramClient
# ===========================================================================

class TestMengramClient:
    @patch("crewai.memory.storage.mengram_storage.urllib.request.urlopen")
    def test_auth_header(self, mock_urlopen: MagicMock) -> None:
        import io
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"ok": true}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = _MengramClient(api_key="om-secret", base_url="https://mengram.io")
        client.stats()

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer om-secret"

    @patch("crewai.memory.storage.mengram_storage.urllib.request.urlopen")
    def test_retry_on_429(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        error_resp = MagicMock()
        error_resp.read.return_value = b'{"detail": "rate limited"}'

        error = urllib.error.HTTPError(
            url="https://mengram.io/v1/stats",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=error_resp,
        )

        ok_resp = MagicMock()
        ok_resp.read.return_value = b'{"entities": 5}'
        ok_resp.__enter__ = MagicMock(return_value=ok_resp)
        ok_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [error, ok_resp]

        client = _MengramClient(api_key="om-key", base_url="https://mengram.io")
        with patch("crewai.memory.storage.mengram_storage.time.sleep"):
            result = client.stats()
        assert result == {"entities": 5}

    @patch("crewai.memory.storage.mengram_storage.urllib.request.urlopen")
    def test_raises_on_permanent_error(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        error_resp = MagicMock()
        error_resp.read.return_value = b'{"detail": "not found"}'

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://mengram.io/v1/stats",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=error_resp,
        )

        client = _MengramClient(api_key="om-key", base_url="https://mengram.io")
        with pytest.raises(RuntimeError, match="Mengram API error 404"):
            client.stats()

    @patch("crewai.memory.storage.mengram_storage.urllib.request.urlopen")
    def test_search_all_request_format(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"semantic": [], "episodic": [], "procedural": []}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = _MengramClient(api_key="om-key", base_url="https://mengram.io")
        client.search_all(query="test", user_id="alice", limit=3, graph_depth=1)

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://mengram.io/v1/search/all"
        body = json.loads(req.data.decode())
        assert body["query"] == "test"
        assert body["user_id"] == "alice"
        assert body["limit"] == 3
        assert body["graph_depth"] == 1


# ---------------------------------------------------------------------------
# TestExtractMemories
# ---------------------------------------------------------------------------

class TestExtractMemories:
    def test_returns_content_as_list(self) -> None:
        mem, _client = _make_memory()
        assert mem.extract_memories("some text") == ["some text"]

    def test_empty_content(self) -> None:
        mem, _client = _make_memory()
        assert mem.extract_memories("") == []


# ---------------------------------------------------------------------------
# TestListCategories
# ---------------------------------------------------------------------------

class TestListCategories:
    def test_returns_by_type(self) -> None:
        mem, client = _make_memory()
        client.stats.return_value = {"by_type": {"person": 3, "tool": 1}}
        cats = mem.list_categories()
        assert cats == {"person": 3, "tool": 1}

    def test_api_error(self) -> None:
        mem, client = _make_memory()
        client.stats.side_effect = RuntimeError("fail")
        assert mem.list_categories() == {}


# ---------------------------------------------------------------------------
# TestListRecords
# ---------------------------------------------------------------------------

class TestListRecords:
    def test_returns_empty(self) -> None:
        mem, _client = _make_memory()
        assert mem.list_records() == []


# need json import for TestMengramClient
import json
