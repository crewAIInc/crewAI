"""Unit tests for HindsightMemory (Hindsight memory backend for CrewAI)."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from crewai.memory.storage.hindsight_storage import (
    HindsightConfig,
    HindsightMemory,
    _call_sync,
)
from crewai.memory.types import MemoryMatch, MemoryRecord, ScopeInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passthrough(fn, *args, **kwargs):
    """Replace _call_sync with a direct call for testing."""
    return fn(*args, **kwargs)


def _make_recall_result(text="Memory text", type_="world", **kwargs):
    """Create a mock RecallResult."""
    r = MagicMock()
    r.text = text
    r.type = type_
    r.context = kwargs.get("context")
    r.occurred_start = kwargs.get("occurred_start")
    r.document_id = kwargs.get("document_id")
    r.metadata = kwargs.get("metadata")
    r.tags = kwargs.get("tags")
    return r


def _make_config(**overrides):
    """Create a HindsightConfig with sensible defaults."""
    defaults = {
        "api_url": "http://localhost:8888",
        "bank_id": "test-bank",
    }
    defaults.update(overrides)
    return HindsightConfig(**defaults)


def _make_memory(config=None, **config_overrides):
    """Create a HindsightMemory with a mocked client."""
    cfg = config or _make_config(**config_overrides)
    memory = HindsightMemory(config=cfg)
    mock_client = MagicMock()
    memory._local.client = mock_client
    memory._created_banks.add(cfg.bank_id)
    return memory, mock_client


# ---------------------------------------------------------------------------
# HindsightConfig tests
# ---------------------------------------------------------------------------


class TestHindsightConfig:
    def test_valid_config(self):
        config = HindsightConfig(
            api_url="http://localhost:8888",
            bank_id="my-bank",
        )
        assert config.api_url == "http://localhost:8888"
        assert config.bank_id == "my-bank"
        assert config.budget == "mid"
        assert config.max_tokens == 4096
        assert config.tags is None
        assert config.mission is None

    def test_defaults(self):
        config = _make_config()
        assert config.budget == "mid"
        assert config.max_tokens == 4096

    def test_api_url_required(self):
        with pytest.raises(ValidationError):
            HindsightConfig(bank_id="test")

    def test_bank_id_required(self):
        with pytest.raises(ValidationError):
            HindsightConfig(api_url="http://localhost:8888")

    def test_empty_api_url_rejected(self):
        with pytest.raises(ValidationError, match="api_url must not be empty"):
            HindsightConfig(api_url="", bank_id="test")

    def test_empty_bank_id_rejected(self):
        with pytest.raises(ValidationError, match="bank_id must not be empty"):
            HindsightConfig(api_url="http://localhost:8888", bank_id="")

    def test_invalid_budget_rejected(self):
        with pytest.raises(ValidationError, match="budget must be one of"):
            HindsightConfig(
                api_url="http://localhost:8888",
                bank_id="test",
                budget="extreme",
            )

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            HindsightConfig(
                api_url="http://localhost:8888",
                bank_id="test",
                unknown_field="value",
            )

    def test_api_url_trailing_slash_stripped(self):
        config = HindsightConfig(
            api_url="http://localhost:8888/",
            bank_id="test",
        )
        assert config.api_url == "http://localhost:8888"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_KEY", "env-key-123")
        config = HindsightConfig(
            api_url="http://localhost:8888",
            bank_id="test",
        )
        assert config.api_key == "env-key-123"

    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_KEY", "env-key")
        config = HindsightConfig(
            api_url="http://localhost:8888",
            bank_id="test",
            api_key="explicit-key",
        )
        assert config.api_key == "explicit-key"

    def test_all_fields(self):
        config = HindsightConfig(
            api_url="http://localhost:8888",
            api_key="key-123",
            bank_id="my-bank",
            budget="high",
            max_tokens=8192,
            tags=["env:prod", "team:ml"],
            mission="Track user preferences",
        )
        assert config.budget == "high"
        assert config.max_tokens == 8192
        assert config.tags == ["env:prod", "team:ml"]
        assert config.mission == "Track user preferences"

    def test_whitespace_api_url_rejected(self):
        with pytest.raises(ValidationError, match="api_url must not be empty"):
            HindsightConfig(api_url="   ", bank_id="test")

    def test_whitespace_bank_id_rejected(self):
        with pytest.raises(ValidationError, match="bank_id must not be empty"):
            HindsightConfig(api_url="http://localhost:8888", bank_id="   ")


# ---------------------------------------------------------------------------
# HindsightMemory.remember() tests
# ---------------------------------------------------------------------------


class TestRemember:
    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_remember_calls_retain(self, _mock_cs):
        memory, mock_client = _make_memory()

        record = memory.remember("Task output text", metadata={"task": "research"})

        mock_client.retain.assert_called_once()
        call_kwargs = mock_client.retain.call_args[1]
        assert call_kwargs["bank_id"] == "test-bank"
        assert call_kwargs["content"] == "Task output text"
        assert call_kwargs["metadata"]["source"] == "crewai"
        assert call_kwargs["metadata"]["task"] == "research"

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_remember_returns_memory_record(self, _mock_cs):
        memory, mock_client = _make_memory()

        record = memory.remember("Some fact")

        assert isinstance(record, MemoryRecord)
        assert record.content == "Some fact"
        assert record.scope == "/"
        assert record.importance == 0.5

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_remember_stringifies_metadata_values(self, _mock_cs):
        memory, mock_client = _make_memory()

        memory.remember("text", metadata={"count": 42, "active": True})

        call_kwargs = mock_client.retain.call_args[1]
        assert call_kwargs["metadata"]["count"] == "42"
        assert call_kwargs["metadata"]["active"] == "True"

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_remember_without_metadata(self, _mock_cs):
        memory, mock_client = _make_memory()

        memory.remember("text")

        call_kwargs = mock_client.retain.call_args[1]
        assert call_kwargs["metadata"] == {"source": "crewai"}

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_remember_raises_on_failure(self, _mock_cs):
        memory, mock_client = _make_memory()
        mock_client.retain.side_effect = RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="Hindsight retain failed"):
            memory.remember("text")

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_remember_passes_tags(self, _mock_cs):
        memory, mock_client = _make_memory(tags=["env:prod"])

        memory.remember("text")

        call_kwargs = mock_client.retain.call_args[1]
        assert call_kwargs["tags"] == ["env:prod"]

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_remember_merges_categories_with_tags(self, _mock_cs):
        memory, mock_client = _make_memory(tags=["env:prod"])

        memory.remember("text", categories=["research", "ml"])

        call_kwargs = mock_client.retain.call_args[1]
        assert call_kwargs["tags"] == ["env:prod", "research", "ml"]

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_remember_includes_source_in_metadata(self, _mock_cs):
        memory, mock_client = _make_memory()

        memory.remember("text", source="user-123")

        call_kwargs = mock_client.retain.call_args[1]
        assert call_kwargs["metadata"]["provenance"] == "user-123"

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_remember_includes_agent_role_in_metadata(self, _mock_cs):
        memory, mock_client = _make_memory()

        memory.remember("text", agent_role="researcher")

        call_kwargs = mock_client.retain.call_args[1]
        assert call_kwargs["metadata"]["agent_role"] == "researcher"


# ---------------------------------------------------------------------------
# HindsightMemory.remember_many() tests
# ---------------------------------------------------------------------------


class TestRememberMany:
    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_remember_many_calls_retain_for_each(self, _mock_cs):
        memory, mock_client = _make_memory()

        records = memory.remember_many(["fact1", "fact2", "fact3"])

        assert mock_client.retain.call_count == 3
        assert len(records) == 3
        assert all(isinstance(r, MemoryRecord) for r in records)

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_remember_many_empty_list(self, _mock_cs):
        memory, mock_client = _make_memory()

        records = memory.remember_many([])

        assert records == []
        mock_client.retain.assert_not_called()


# ---------------------------------------------------------------------------
# HindsightMemory.recall() tests
# ---------------------------------------------------------------------------


class TestRecall:
    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_recall_calls_recall_api(self, _mock_cs):
        memory, mock_client = _make_memory()

        mock_response = MagicMock()
        mock_response.results = [_make_recall_result()]
        mock_client.recall.return_value = mock_response

        matches = memory.recall("programming preferences", limit=5)

        mock_client.recall.assert_called_once()
        call_kwargs = mock_client.recall.call_args[1]
        assert call_kwargs["bank_id"] == "test-bank"
        assert call_kwargs["query"] == "programming preferences"
        assert call_kwargs["budget"] == "mid"
        assert call_kwargs["max_tokens"] == 4096
        assert len(matches) == 1

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_recall_returns_memory_match_objects(self, _mock_cs):
        memory, mock_client = _make_memory()

        mock_response = MagicMock()
        mock_response.results = [_make_recall_result(text="Found fact")]
        mock_client.recall.return_value = mock_response

        matches = memory.recall("test")

        assert len(matches) == 1
        assert isinstance(matches[0], MemoryMatch)
        assert isinstance(matches[0].record, MemoryRecord)
        assert matches[0].record.content == "Found fact"
        assert matches[0].score > 0
        assert "semantic" in matches[0].match_reasons

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_recall_respects_limit(self, _mock_cs):
        memory, mock_client = _make_memory()

        mock_response = MagicMock()
        mock_response.results = [
            _make_recall_result(text=f"Memory {i}") for i in range(10)
        ]
        mock_client.recall.return_value = mock_response

        matches = memory.recall("test", limit=3)
        assert len(matches) == 3

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_recall_returns_empty_on_no_results(self, _mock_cs):
        memory, mock_client = _make_memory()

        mock_response = MagicMock()
        mock_response.results = []
        mock_client.recall.return_value = mock_response

        matches = memory.recall("nonexistent topic")
        assert matches == []

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_recall_includes_rich_metadata(self, _mock_cs):
        memory, mock_client = _make_memory()

        r = _make_recall_result(
            context="conversation",
            occurred_start="2024-01-01",
            document_id="doc-1",
            metadata={"key": "value"},
            tags=["tag1"],
        )
        mock_response = MagicMock()
        mock_response.results = [r]
        mock_client.recall.return_value = mock_response

        matches = memory.recall("test")
        meta = matches[0].record.metadata
        assert meta["source_context"] == "conversation"
        assert meta["occurred_start"] == "2024-01-01"
        assert meta["document_id"] == "doc-1"
        assert meta["key"] == "value"
        assert meta["tags"] == ["tag1"]

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_recall_raises_on_failure(self, _mock_cs):
        memory, mock_client = _make_memory()
        mock_client.recall.side_effect = RuntimeError("timeout")

        with pytest.raises(RuntimeError, match="Hindsight recall failed"):
            memory.recall("test")

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_recall_scores_descend(self, _mock_cs):
        memory, mock_client = _make_memory()

        mock_response = MagicMock()
        mock_response.results = [
            _make_recall_result(text=f"Memory {i}") for i in range(5)
        ]
        mock_client.recall.return_value = mock_response

        matches = memory.recall("test", limit=5)
        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 1.0

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_recall_uses_config_budget_and_max_tokens(self, _mock_cs):
        memory, mock_client = _make_memory(budget="high", max_tokens=8192)

        mock_response = MagicMock()
        mock_response.results = []
        mock_client.recall.return_value = mock_response

        memory.recall("test")

        call_kwargs = mock_client.recall.call_args[1]
        assert call_kwargs["budget"] == "high"
        assert call_kwargs["max_tokens"] == 8192


# ---------------------------------------------------------------------------
# HindsightMemory.reset() tests
# ---------------------------------------------------------------------------


class TestReset:
    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_reset_deletes_bank(self, _mock_cs):
        memory, mock_client = _make_memory()

        memory.reset()

        mock_client.delete_bank.assert_called_once_with("test-bank")

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_reset_recreates_bank(self, _mock_cs):
        memory, mock_client = _make_memory()

        memory.reset()

        # After delete, _ensure_bank should be called (which calls create_bank)
        assert mock_client.create_bank.called

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_reset_is_best_effort(self, _mock_cs):
        memory, mock_client = _make_memory()
        mock_client.delete_bank.side_effect = RuntimeError("not found")

        # Should not raise
        memory.reset()


# ---------------------------------------------------------------------------
# drain_writes / close tests
# ---------------------------------------------------------------------------


class TestDrainWritesAndClose:
    def test_drain_writes_no_op(self):
        memory, _ = _make_memory()
        # Should not raise
        memory.drain_writes()

    def test_close_no_op(self):
        memory, _ = _make_memory()
        # Should not raise
        memory.close()


# ---------------------------------------------------------------------------
# scope / info tests
# ---------------------------------------------------------------------------


class TestScopeAndInfo:
    def test_scope_returns_self(self):
        memory, _ = _make_memory()
        scoped = memory.scope("/project/alpha")
        assert scoped is memory

    def test_info_returns_scope_info(self):
        memory, _ = _make_memory()
        info = memory.info("/project")
        assert isinstance(info, ScopeInfo)
        assert info.path == "/project"
        assert info.record_count == 0


# ---------------------------------------------------------------------------
# extract_memories test
# ---------------------------------------------------------------------------


class TestExtractMemories:
    def test_extract_memories_returns_content_as_list(self):
        memory, _ = _make_memory()
        result = memory.extract_memories("Some raw content")
        assert result == ["Some raw content"]


# ---------------------------------------------------------------------------
# Bank lifecycle tests
# ---------------------------------------------------------------------------


class TestBankLifecycle:
    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_ensure_bank_on_first_remember(self, _mock_cs):
        """Bank should be created on first remember if not already cached."""
        cfg = _make_config()
        memory = HindsightMemory(config=cfg)
        mock_client = MagicMock()
        memory._local.client = mock_client
        # Don't pre-add the bank to _created_banks

        memory.remember("text")

        mock_client.create_bank.assert_called_once()
        mock_client.retain.assert_called_once()

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_409_cached_no_retry(self, _mock_cs):
        """409 conflict should be cached so subsequent calls don't retry."""
        cfg = _make_config()
        memory = HindsightMemory(config=cfg)
        mock_client = MagicMock()
        memory._local.client = mock_client
        mock_client.create_bank.side_effect = Exception("409 Conflict")

        memory.remember("text1")
        memory.remember("text2")

        # create_bank called once (first remember), 409 cached, not called again
        assert mock_client.create_bank.call_count == 1
        assert mock_client.retain.call_count == 2

    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_transient_error_retries(self, _mock_cs):
        """Non-409 errors should NOT be cached so creation is retried."""
        cfg = _make_config()
        memory = HindsightMemory(config=cfg)
        mock_client = MagicMock()
        memory._local.client = mock_client
        # First call fails transiently, second succeeds
        mock_client.create_bank.side_effect = [
            Exception("Connection reset"),
            None,
        ]

        memory.remember("text1")
        memory.remember("text2")

        # create_bank called twice (first failed, not cached, second succeeded)
        assert mock_client.create_bank.call_count == 2


# ---------------------------------------------------------------------------
# Import error test
# ---------------------------------------------------------------------------


class TestImportError:
    def test_import_error_when_client_not_installed(self):
        config = _make_config()
        memory = HindsightMemory(config=config)
        # Clear any cached client
        memory._local = MagicMock()
        memory._local.client = None

        with patch.dict("sys.modules", {"hindsight_client": None}):
            with pytest.raises(ModuleNotFoundError, match="hindsight-client"):
                memory._get_client()


# ---------------------------------------------------------------------------
# forget() test
# ---------------------------------------------------------------------------


class TestForget:
    @patch(
        "crewai.memory.storage.hindsight_storage._call_sync",
        side_effect=_passthrough,
    )
    def test_forget_calls_reset(self, _mock_cs):
        memory, mock_client = _make_memory()

        memory.forget()

        mock_client.delete_bank.assert_called_once_with("test-bank")
