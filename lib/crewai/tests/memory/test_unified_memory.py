"""Tests for unified memory: types, storage, Memory, MemoryScope, MemorySlice, Flow integration."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crewai.utilities.printer import Printer
from crewai.memory.types import (
    MemoryConfig,
    MemoryMatch,
    MemoryRecord,
    ScopeInfo,
    compute_composite_score,
)


# --- Types ---


def test_memory_record_defaults() -> None:
    r = MemoryRecord(content="hello")
    assert r.content == "hello"
    assert r.scope == "/"
    assert r.categories == []
    assert r.importance == 0.5
    assert r.embedding is None
    assert r.id is not None
    assert isinstance(r.created_at, datetime)


def test_memory_match() -> None:
    r = MemoryRecord(content="x", scope="/a")
    m = MemoryMatch(record=r, score=0.9, match_reasons=["semantic"])
    assert m.record.content == "x"
    assert m.score == 0.9
    assert m.match_reasons == ["semantic"]


def test_scope_info() -> None:
    i = ScopeInfo(path="/", record_count=5, categories=["c1"], child_scopes=["/a"])
    assert i.path == "/"
    assert i.record_count == 5
    assert i.categories == ["c1"]
    assert i.child_scopes == ["/a"]


def test_memory_config() -> None:
    c = MemoryConfig()
    assert c.recency_weight == 0.3
    assert c.semantic_weight == 0.5
    assert c.importance_weight == 0.2


# --- LanceDB storage ---


@pytest.fixture
def lancedb_path(tmp_path: Path) -> Path:
    return tmp_path / "mem"


def test_lancedb_save_search(lancedb_path: Path) -> None:
    from crewai.memory.storage.lancedb_storage import LanceDBStorage

    storage = LanceDBStorage(path=str(lancedb_path), vector_dim=4)
    r = MemoryRecord(
        content="test content",
        scope="/foo",
        categories=["cat1"],
        importance=0.8,
        embedding=[0.1, 0.2, 0.3, 0.4],
    )
    storage.save([r])
    results = storage.search(
        [0.1, 0.2, 0.3, 0.4],
        scope_prefix="/foo",
        limit=5,
    )
    assert len(results) == 1
    rec, score = results[0]
    assert rec.content == "test content"
    assert rec.scope == "/foo"
    assert score >= 0.0


def test_lancedb_delete_count(lancedb_path: Path) -> None:
    from crewai.memory.storage.lancedb_storage import LanceDBStorage

    storage = LanceDBStorage(path=str(lancedb_path), vector_dim=4)
    r = MemoryRecord(content="x", scope="/", embedding=[0.0] * 4)
    storage.save([r])
    assert storage.count() == 1
    n = storage.delete(scope_prefix="/")
    assert n >= 1
    assert storage.count() == 0


def test_lancedb_list_scopes_get_scope_info(lancedb_path: Path) -> None:
    from crewai.memory.storage.lancedb_storage import LanceDBStorage

    storage = LanceDBStorage(path=str(lancedb_path), vector_dim=4)
    storage.save([
        MemoryRecord(content="a", scope="/", embedding=[0.0] * 4),
        MemoryRecord(content="b", scope="/team", embedding=[0.0] * 4),
    ])
    scopes = storage.list_scopes("/")
    assert "/" in scopes
    info = storage.get_scope_info("/")
    assert info.record_count >= 1
    assert info.path == "/"


# --- Memory class (with mock embedder, no LLM for explicit remember) ---


@pytest.fixture
def mock_embedder() -> MagicMock:
    m = MagicMock()
    m.return_value = [[0.1] * 1536]  # Chroma-style returns list of lists
    return m


@pytest.fixture
def memory_with_storage(tmp_path: Path, mock_embedder: MagicMock) -> None:
    import os
    os.environ.pop("OPENAI_API_KEY", None)


def test_memory_remember_recall_shallow(tmp_path: Path, mock_embedder: MagicMock) -> None:
    from crewai.memory.unified_memory import Memory

    m = Memory(
        storage=str(tmp_path / "db"),
        llm=MagicMock(),
        embedder=mock_embedder,
    )
    # Explicit scope/categories/importance so no LLM analysis
    r = m.remember(
        "We decided to use Python.",
        scope="/project",
        categories=["decision"],
        importance=0.7,
    )
    assert r.content == "We decided to use Python."
    assert r.scope == "/project"

    matches = m.recall("Python decision", scope="/project", limit=5, depth="shallow")
    assert len(matches) >= 1
    assert "Python" in matches[0].record.content or "python" in matches[0].record.content.lower()


def test_memory_forget(tmp_path: Path, mock_embedder: MagicMock) -> None:
    from crewai.memory.unified_memory import Memory

    m = Memory(storage=str(tmp_path / "db2"), llm=MagicMock(), embedder=mock_embedder)
    m.remember("To forget", scope="/x", categories=[], importance=0.5, metadata={})
    assert m._storage.count("/x") >= 1
    n = m.forget(scope="/x")
    assert n >= 1
    assert m._storage.count("/x") == 0


def test_memory_scope_slice(tmp_path: Path, mock_embedder: MagicMock) -> None:
    from crewai.memory.unified_memory import Memory

    mem = Memory(storage=str(tmp_path / "db3"), llm=MagicMock(), embedder=mock_embedder)
    sc = mem.scope("/agent/1")
    assert sc._root in ("/agent/1", "/agent/1/")
    sl = mem.slice(["/a", "/b"], read_only=True)
    assert sl._read_only is True
    assert "/a" in sl._scopes and "/b" in sl._scopes


def test_memory_list_scopes_info_tree(tmp_path: Path, mock_embedder: MagicMock) -> None:
    from crewai.memory.unified_memory import Memory

    m = Memory(storage=str(tmp_path / "db4"), llm=MagicMock(), embedder=mock_embedder)
    m.remember("Root", scope="/", categories=[], importance=0.5, metadata={})
    scopes = m.list_scopes("/")
    assert "/" in scopes
    info = m.info("/")
    assert info.record_count >= 1
    tree = m.tree("/", max_depth=2)
    assert "/" in tree or "0 records" in tree or "1 records" in tree


# --- MemoryScope ---


def test_memory_scope_remember_recall(tmp_path: Path, mock_embedder: MagicMock) -> None:
    from crewai.memory.unified_memory import Memory
    from crewai.memory.memory_scope import MemoryScope

    mem = Memory(storage=str(tmp_path / "db5"), llm=MagicMock(), embedder=mock_embedder)
    scope = MemoryScope(mem, "/crew/1")
    scope.remember("Scoped note", scope="/", categories=[], importance=0.5, metadata={})
    results = scope.recall("note", limit=5, depth="shallow")
    assert len(results) >= 1


# --- MemorySlice recall (read-only) ---


def test_memory_slice_recall(tmp_path: Path, mock_embedder: MagicMock) -> None:
    from crewai.memory.unified_memory import Memory
    from crewai.memory.memory_scope import MemorySlice

    mem = Memory(storage=str(tmp_path / "db6"), llm=MagicMock(), embedder=mock_embedder)
    mem.remember("In scope A", scope="/a", categories=[], importance=0.5, metadata={})
    sl = MemorySlice(mem, ["/a"], read_only=True)
    matches = sl.recall("scope", limit=5, depth="shallow")
    assert isinstance(matches, list)


def test_memory_slice_remember_raises_when_read_only(tmp_path: Path, mock_embedder: MagicMock) -> None:
    from crewai.memory.unified_memory import Memory
    from crewai.memory.memory_scope import MemorySlice

    mem = Memory(storage=str(tmp_path / "db7"), llm=MagicMock(), embedder=mock_embedder)
    sl = MemorySlice(mem, ["/a"], read_only=True)
    with pytest.raises(PermissionError):
        sl.remember("x", scope="/a")


# --- Flow memory ---


def test_flow_has_default_memory() -> None:
    """Flow auto-creates a Memory instance when none is provided."""
    from crewai.flow.flow import Flow
    from crewai.memory.unified_memory import Memory

    class DefaultFlow(Flow):
        pass

    f = DefaultFlow()
    assert f.memory is not None
    assert isinstance(f.memory, Memory)


def test_flow_recall_remember_raise_when_memory_explicitly_none() -> None:
    """Flow raises ValueError when memory is explicitly set to None."""
    from crewai.flow.flow import Flow

    class NoMemoryFlow(Flow):
        memory = None

    f = NoMemoryFlow()
    # Explicitly set to None after __init__ auto-creates
    f.memory = None
    with pytest.raises(ValueError, match="No memory configured"):
        f.recall("query")
    with pytest.raises(ValueError, match="No memory configured"):
        f.remember("content")


def test_flow_recall_remember_with_memory(tmp_path: Path, mock_embedder: MagicMock) -> None:
    from crewai.flow.flow import Flow
    from crewai.memory.unified_memory import Memory

    mem = Memory(storage=str(tmp_path / "flow_db"), llm=MagicMock(), embedder=mock_embedder)

    class FlowWithMemory(Flow):
        memory = mem

    f = FlowWithMemory()
    f.remember("Flow remembered this", scope="/flow", categories=[], importance=0.6, metadata={})
    results = f.recall("remembered", limit=5, depth="shallow")
    assert len(results) >= 1


# --- extract_memories ---


def test_memory_extract_memories_returns_list_from_llm(tmp_path: Path) -> None:
    """Memory.extract_memories() delegates to LLM and returns list of strings."""
    from crewai.memory.analyze import ExtractedMemories
    from crewai.memory.unified_memory import Memory

    mock_llm = MagicMock()
    mock_llm.supports_function_calling.return_value = True
    mock_llm.call.return_value = ExtractedMemories(
        memories=["We use Python for the backend.", "API rate limit is 100/min."]
    )

    mem = Memory(
        storage=str(tmp_path / "extract_db"),
        llm=mock_llm,
        embedder=MagicMock(return_value=[[0.1] * 1536]),
    )
    result = mem.extract_memories("Task: Build API. Result: We used Python and set rate limit 100/min.")
    assert result == ["We use Python for the backend.", "API rate limit is 100/min."]
    mock_llm.call.assert_called_once()
    call_kw = mock_llm.call.call_args[1]
    assert call_kw.get("response_model") == ExtractedMemories


def test_memory_extract_memories_empty_content_returns_empty_list(tmp_path: Path) -> None:
    """Memory.extract_memories() with empty/whitespace content returns [] without calling LLM."""
    from crewai.memory.unified_memory import Memory

    mock_llm = MagicMock()
    mem = Memory(storage=str(tmp_path / "empty_db"), llm=mock_llm, embedder=MagicMock())
    assert mem.extract_memories("") == []
    assert mem.extract_memories("   \n  ") == []
    mock_llm.call.assert_not_called()


def test_executor_save_to_memory_calls_extract_then_remember_per_item() -> None:
    """_save_to_memory calls memory.extract_memories(raw) then memory.remember(m) for each."""
    from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
    from crewai.agents.parser import AgentFinish

    mock_memory = MagicMock()
    mock_memory.extract_memories.return_value = ["Fact A.", "Fact B."]

    mock_agent = MagicMock()
    mock_agent.memory = mock_memory
    mock_agent._logger = MagicMock()
    mock_agent.role = "Researcher"

    mock_task = MagicMock()
    mock_task.description = "Do research"
    mock_task.expected_output = "A report"

    class MinimalExecutor(CrewAgentExecutorMixin):
        crew = None
        agent = mock_agent
        task = mock_task
        iterations = 0
        max_iter = 1
        messages = []
        _i18n = MagicMock()
        _printer = Printer()

    executor = MinimalExecutor()
    executor._save_to_memory(
        AgentFinish(thought="", output="We found X and Y.", text="We found X and Y.")
    )

    raw_expected = "Task: Do research\nAgent: Researcher\nExpected result: A report\nResult: We found X and Y."
    mock_memory.extract_memories.assert_called_once_with(raw_expected)
    assert mock_memory.remember.call_count == 2
    calls = [mock_memory.remember.call_args_list[i][0][0] for i in range(2)]
    assert calls == ["Fact A.", "Fact B."]


def test_executor_save_to_memory_skips_delegation_output() -> None:
    """_save_to_memory does nothing when output contains delegate action."""
    from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
    from crewai.agents.parser import AgentFinish
    from crewai.utilities.string_utils import sanitize_tool_name

    mock_memory = MagicMock()
    mock_agent = MagicMock()
    mock_agent.memory = mock_memory
    mock_agent._logger = MagicMock()
    mock_task = MagicMock(description="Task", expected_output="Out")

    class MinimalExecutor(CrewAgentExecutorMixin):
        crew = None
        agent = mock_agent
        task = mock_task
        iterations = 0
        max_iter = 1
        messages = []
        _i18n = MagicMock()
        _printer = Printer()

    delegate_text = f"Action: {sanitize_tool_name('Delegate work to coworker')}"
    full_text = delegate_text + " rest"
    executor = MinimalExecutor()
    executor._save_to_memory(
        AgentFinish(thought="", output=full_text, text=full_text)
    )

    mock_memory.extract_memories.assert_not_called()
    mock_memory.remember.assert_not_called()


def test_memory_scope_extract_memories_delegates() -> None:
    """MemoryScope.extract_memories delegates to underlying Memory."""
    from crewai.memory.memory_scope import MemoryScope

    mock_memory = MagicMock()
    mock_memory.extract_memories.return_value = ["Scoped fact."]
    scope = MemoryScope(mock_memory, "/agent/1")
    result = scope.extract_memories("Some content")
    mock_memory.extract_memories.assert_called_once_with("Some content")
    assert result == ["Scoped fact."]


def test_memory_slice_extract_memories_delegates() -> None:
    """MemorySlice.extract_memories delegates to underlying Memory."""
    from crewai.memory.memory_scope import MemorySlice

    mock_memory = MagicMock()
    mock_memory.extract_memories.return_value = ["Sliced fact."]
    sl = MemorySlice(mock_memory, ["/a", "/b"], read_only=True)
    result = sl.extract_memories("Some content")
    mock_memory.extract_memories.assert_called_once_with("Some content")
    assert result == ["Sliced fact."]


def test_flow_extract_memories_raises_when_memory_explicitly_none() -> None:
    """Flow.extract_memories raises ValueError when memory is explicitly set to None."""
    from crewai.flow.flow import Flow

    f = Flow()
    f.memory = None
    with pytest.raises(ValueError, match="No memory configured"):
        f.extract_memories("some content")


def test_flow_extract_memories_delegates_when_memory_present() -> None:
    """Flow.extract_memories delegates to flow memory and returns list."""
    from crewai.flow.flow import Flow

    mock_memory = MagicMock()
    mock_memory.extract_memories.return_value = ["Flow fact 1.", "Flow fact 2."]

    class FlowWithMemory(Flow):
        memory = mock_memory

    f = FlowWithMemory()
    result = f.extract_memories("content here")
    mock_memory.extract_memories.assert_called_once_with("content here")
    assert result == ["Flow fact 1.", "Flow fact 2."]


# --- Composite scoring ---


def test_composite_score_brand_new_memory() -> None:
    """Brand-new memory has decay ~ 1.0; composite = 0.5*0.8 + 0.3*1.0 + 0.2*0.7 = 0.84."""
    config = MemoryConfig()
    record = MemoryRecord(
        content="test",
        scope="/",
        importance=0.7,
        created_at=datetime.utcnow(),
    )
    score, reasons = compute_composite_score(record, 0.8, config)
    assert 0.82 <= score <= 0.86
    assert "semantic" in reasons
    assert "recency" in reasons
    assert "importance" in reasons


def test_composite_score_old_memory_decayed() -> None:
    """Memory 60 days old (2 half-lives) has decay = 0.25; composite ~ 0.575."""
    config = MemoryConfig(recency_half_life_days=30)
    old_date = datetime.utcnow() - timedelta(days=60)
    record = MemoryRecord(
        content="old",
        scope="/",
        importance=0.5,
        created_at=old_date,
    )
    score, reasons = compute_composite_score(record, 0.8, config)
    assert 0.55 <= score <= 0.60
    assert "semantic" in reasons
    assert "recency" not in reasons  # decay 0.25 is not > 0.5


def test_composite_score_reranks_results(
    tmp_path: Path, mock_embedder: MagicMock
) -> None:
    """Same semantic score: high-importance recent memory ranks first."""
    from crewai.memory.unified_memory import Memory

    # Use same dim as default LanceDB (1536) so storage does not overwrite embedding
    emb = [0.1] * 1536
    mem = Memory(
        storage=str(tmp_path / "rerank_db"),
        llm=MagicMock(),
        embedder=MagicMock(return_value=[emb]),
    )
    # Same embedding so same semantic score from storage
    mem.remember(
        "Important decision",
        scope="/",
        categories=[],
        importance=1.0,
        metadata={},
    )
    old = datetime.utcnow() - timedelta(days=90)
    record_low = MemoryRecord(
        content="Old trivial note",
        scope="/",
        importance=0.1,
        created_at=old,
        embedding=emb,
    )
    mem._storage.save([record_low])

    matches = mem.recall("decision", scope="/", limit=5, depth="shallow")
    assert len(matches) >= 2
    # Top result should be the high-importance recent one (stored via remember)
    assert "Important" in matches[0].record.content or "important" in matches[0].record.content.lower()


def test_composite_score_match_reasons_populated() -> None:
    """match_reasons includes recency for fresh, importance for high-importance; omits for old/low."""
    config = MemoryConfig()
    fresh_high = MemoryRecord(
        content="x",
        importance=0.9,
        created_at=datetime.utcnow(),
    )
    score1, reasons1 = compute_composite_score(fresh_high, 0.5, config)
    assert "semantic" in reasons1
    assert "recency" in reasons1
    assert "importance" in reasons1

    old_low = MemoryRecord(
        content="y",
        importance=0.1,
        created_at=datetime.utcnow() - timedelta(days=60),
    )
    score2, reasons2 = compute_composite_score(old_low, 0.5, config)
    assert "semantic" in reasons2
    assert "recency" not in reasons2
    assert "importance" not in reasons2


def test_composite_score_custom_config() -> None:
    """Zero recency/importance weights => composite equals semantic score."""
    config = MemoryConfig(
        recency_weight=0.0,
        semantic_weight=1.0,
        importance_weight=0.0,
    )
    record = MemoryRecord(
        content="any",
        importance=0.9,
        created_at=datetime.utcnow(),
    )
    score, reasons = compute_composite_score(record, 0.73, config)
    assert score == pytest.approx(0.73, rel=1e-5)
    assert "semantic" in reasons


# --- LLM fallback ---


def test_analyze_for_save_llm_failure_returns_defaults() -> None:
    """When LLM raises, analyze_for_save returns safe defaults."""
    from crewai.memory.analyze import MemoryAnalysis, analyze_for_save

    llm = MagicMock()
    llm.call.side_effect = RuntimeError("API rate limit")
    result = analyze_for_save(
        "some content",
        existing_scopes=["/", "/project"],
        existing_categories=["cat1"],
        llm=llm,
    )
    assert isinstance(result, MemoryAnalysis)
    assert result.suggested_scope == "/"
    assert result.categories == []
    assert result.importance == 0.5
    assert result.extracted_metadata.entities == []
    assert result.extracted_metadata.dates == []
    assert result.extracted_metadata.topics == []


def test_extract_memories_llm_failure_returns_raw() -> None:
    """When LLM raises, extract_memories_from_content returns [content]."""
    from crewai.memory.analyze import extract_memories_from_content

    llm = MagicMock()
    llm.call.side_effect = RuntimeError("Network error")
    content = "Task result: We chose PostgreSQL."
    result = extract_memories_from_content(content, llm)
    assert result == [content]


def test_analyze_query_llm_failure_returns_defaults() -> None:
    """When LLM raises, analyze_query returns safe defaults with available scopes."""
    from crewai.memory.analyze import QueryAnalysis, analyze_query

    llm = MagicMock()
    llm.call.side_effect = RuntimeError("Timeout")
    result = analyze_query(
        "what did we decide?",
        available_scopes=["/", "/project", "/team", "/company", "/other", "/extra"],
        scope_info=None,
        llm=llm,
    )
    assert isinstance(result, QueryAnalysis)
    assert result.keywords == []
    assert result.time_hints == []
    assert result.complexity == "simple"
    assert result.suggested_scopes == ["/", "/project", "/team", "/company", "/other"]


def test_remember_survives_llm_failure(
    tmp_path: Path, mock_embedder: MagicMock
) -> None:
    """When analyze_for_save fails (LLM raises), remember() still saves with defaults."""
    from crewai.memory.unified_memory import Memory

    llm = MagicMock()
    llm.call.side_effect = RuntimeError("LLM unavailable")
    mem = Memory(
        storage=str(tmp_path / "fallback_db"),
        llm=llm,
        embedder=mock_embedder,
    )
    record = mem.remember("We decided to use PostgreSQL.")
    assert record.content == "We decided to use PostgreSQL."
    assert record.scope == "/"
    assert record.categories == []
    assert record.importance == 0.5
    assert record.id is not None
    assert mem._storage.count() == 1
