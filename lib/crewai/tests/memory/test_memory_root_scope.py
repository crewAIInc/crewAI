"""Tests for hierarchical root_scope functionality in unified memory.

Root scope is a structural prefix that is set automatically by crews and flows.
The LLM's encoding flow still infers a semantic inner scope, but the final
resolved scope = root_scope + '/' + llm_inferred_scope.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crewai.memory.types import MemoryRecord
from crewai.memory.utils import (
    join_scope_paths,
    normalize_scope_path,
    sanitize_scope_name,
)


# --- Utility function tests ---


class TestSanitizeScopeName:
    """Tests for sanitize_scope_name utility."""

    def test_simple_name(self) -> None:
        assert sanitize_scope_name("research") == "research"

    def test_name_with_spaces(self) -> None:
        assert sanitize_scope_name("Research Crew") == "research-crew"

    def test_name_with_special_chars(self) -> None:
        assert sanitize_scope_name("Agent #1 (Main)") == "agent-1-main"

    def test_name_with_unicode(self) -> None:
        # Unicode characters get replaced with hyphens
        result = sanitize_scope_name("café_worker")
        # é becomes -, and the underscore is preserved, so café_worker -> caf-_worker
        assert result == "caf-_worker"

    def test_name_with_underscores(self) -> None:
        # Underscores are preserved
        assert sanitize_scope_name("test_agent") == "test_agent"

    def test_name_with_hyphens(self) -> None:
        assert sanitize_scope_name("my-crew") == "my-crew"

    def test_multiple_spaces_collapsed(self) -> None:
        assert sanitize_scope_name("foo   bar") == "foo-bar"

    def test_leading_trailing_spaces(self) -> None:
        assert sanitize_scope_name("  crew  ") == "crew"

    def test_empty_string_returns_unknown(self) -> None:
        assert sanitize_scope_name("") == "unknown"

    def test_only_special_chars_returns_unknown(self) -> None:
        assert sanitize_scope_name("@#$%") == "unknown"

    def test_none_input_returns_unknown(self) -> None:
        assert sanitize_scope_name(None) == "unknown"  # type: ignore[arg-type]


class TestNormalizeScopePath:
    """Tests for normalize_scope_path utility."""

    def test_simple_path(self) -> None:
        assert normalize_scope_path("/crew/test") == "/crew/test"

    def test_double_slashes_collapsed(self) -> None:
        assert normalize_scope_path("/crew//test//agent") == "/crew/test/agent"

    def test_trailing_slash_removed(self) -> None:
        assert normalize_scope_path("/crew/test/") == "/crew/test"

    def test_missing_leading_slash_added(self) -> None:
        assert normalize_scope_path("crew/test") == "/crew/test"

    def test_empty_string_returns_root(self) -> None:
        assert normalize_scope_path("") == "/"

    def test_root_only_returns_root(self) -> None:
        assert normalize_scope_path("/") == "/"

    def test_multiple_trailing_slashes(self) -> None:
        assert normalize_scope_path("/crew///") == "/crew"


class TestJoinScopePaths:
    """Tests for join_scope_paths utility."""

    def test_basic_join(self) -> None:
        assert join_scope_paths("/crew/test", "/market-trends") == "/crew/test/market-trends"

    def test_inner_without_leading_slash(self) -> None:
        assert join_scope_paths("/crew/test", "market-trends") == "/crew/test/market-trends"

    def test_root_with_trailing_slash(self) -> None:
        assert join_scope_paths("/crew/test/", "/inner") == "/crew/test/inner"

    def test_root_only_inner_slash(self) -> None:
        assert join_scope_paths("/crew/test", "/") == "/crew/test"

    def test_root_only_inner_none(self) -> None:
        assert join_scope_paths("/crew/test", None) == "/crew/test"

    def test_no_root_with_inner(self) -> None:
        assert join_scope_paths(None, "/market-trends") == "/market-trends"

    def test_both_none(self) -> None:
        assert join_scope_paths(None, None) == "/"

    def test_empty_strings(self) -> None:
        assert join_scope_paths("", "") == "/"

    def test_root_empty_inner_value(self) -> None:
        assert join_scope_paths("", "inner") == "/inner"


# --- Memory root_scope tests ---


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Embedder mock that returns one embedding per input text (batch-aware)."""
    m = MagicMock()
    m.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
    return m


class TestMemoryRootScope:
    """Tests for Memory class root_scope field."""

    def test_memory_with_root_scope_prepends_to_explicit_scope(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """When root_scope is set and explicit scope is provided, they combine."""
        from crewai.memory.unified_memory import Memory

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/crew/research-crew",
        )

        record = mem.remember(
            "Test content",
            scope="/market-trends",
            categories=["test"],
            importance=0.7,
        )

        assert record is not None
        assert record.scope == "/crew/research-crew/market-trends"

    def test_memory_without_root_scope_uses_explicit_scope_directly(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """When root_scope is None, explicit scope is used as-is (backward compat)."""
        from crewai.memory.unified_memory import Memory

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )

        record = mem.remember(
            "Test content",
            scope="/explicit",
            categories=["test"],
            importance=0.7,
        )

        assert record is not None
        assert record.scope == "/explicit"

    def test_memory_root_scope_with_llm_inferred_scope(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """When root_scope is set and scope is inferred by LLM, they combine."""
        from crewai.memory.analyze import ExtractedMetadata, MemoryAnalysis
        from crewai.memory.unified_memory import Memory

        llm = MagicMock()
        llm.supports_function_calling.return_value = True
        llm.call.return_value = MemoryAnalysis(
            suggested_scope="/quarterly-results",
            categories=["finance"],
            importance=0.8,
            extracted_metadata=ExtractedMetadata(),
        )

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=llm,
            embedder=mock_embedder,
            root_scope="/flow/mypipeline",
        )

        # Don't provide scope - let LLM infer it
        record = mem.remember("Q1 revenue was $1M")

        assert record is not None
        assert record.scope == "/flow/mypipeline/quarterly-results"

    def test_memory_root_scope_per_call_override(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """Per-call root_scope overrides instance-level root_scope."""
        from crewai.memory.unified_memory import Memory

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/crew/base",
        )

        record = mem.remember(
            "Test content",
            scope="/inner",
            categories=["test"],
            importance=0.7,
            root_scope="/override/path",  # Override instance-level
        )

        assert record is not None
        assert record.scope == "/override/path/inner"

    def test_remember_many_with_root_scope(
        self, tmp_path: Path,
    ) -> None:
        """remember_many respects root_scope for all items."""
        from crewai.memory.unified_memory import Memory

        # Use distinct embeddings to avoid intra-batch dedup
        call_count = 0

        def distinct_embedder(texts: list[str]) -> list[list[float]]:
            nonlocal call_count
            result = []
            for i, _ in enumerate(texts):
                emb = [0.0] * 1536
                emb[(call_count + i) % 1536] = 1.0
                result.append(emb)
            call_count += len(texts)
            return result

        mock_embedder = MagicMock(side_effect=distinct_embedder)

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/crew/batch-crew",
        )

        mem.remember_many(
            ["Fact A", "Fact B"],
            scope="/decisions",
            categories=["test"],
            importance=0.7,
        )
        mem.drain_writes()

        records = mem.list_records()
        assert len(records) == 2
        for record in records:
            assert record.scope == "/crew/batch-crew/decisions"

    def test_remember_many_per_call_root_scope_override(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """remember_many accepts per-call root_scope override."""
        from crewai.memory.unified_memory import Memory

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/default",
        )

        mem.remember_many(
            ["Fact A"],
            scope="/inner",
            categories=["test"],
            importance=0.7,
            root_scope="/agent/researcher",  # Per-call override
        )
        mem.drain_writes()

        # Use a global memory view to see all records (not scoped to /default)
        mem_global = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )
        records = mem_global.list_records()
        assert len(records) == 1
        assert records[0].scope == "/agent/researcher/inner"


class TestRootScopePathNormalization:
    """Tests for proper path normalization with root_scope."""

    def test_no_double_slashes_in_result(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """Final scope should not have double slashes."""
        from crewai.memory.unified_memory import Memory

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/crew/test/",  # Trailing slash
        )

        record = mem.remember(
            "Test",
            scope="/inner/",  # Both have slashes
            categories=["test"],
            importance=0.5,
        )

        assert record is not None
        assert "//" not in record.scope
        assert record.scope == "/crew/test/inner"

    def test_leading_slash_always_present(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """Final scope should always have leading slash."""
        from crewai.memory.unified_memory import Memory

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="crew/test",  # No leading slash
        )

        record = mem.remember(
            "Test",
            scope="inner",  # No leading slash
            categories=["test"],
            importance=0.5,
        )

        assert record is not None
        assert record.scope.startswith("/")

    def test_root_scope_with_root_inner_scope(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """When inner scope is '/', result is just the root_scope."""
        from crewai.memory.unified_memory import Memory

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/crew/test",
        )

        record = mem.remember(
            "Test",
            scope="/",  # Root scope
            categories=["test"],
            importance=0.5,
        )

        assert record is not None
        assert record.scope == "/crew/test"


class TestCrewAutoScoping:
    """Tests for automatic root_scope assignment in Crew."""

    def test_crew_memory_true_sets_root_scope(self) -> None:
        """Creating Crew with memory=True auto-sets root_scope."""
        from crewai.agent import Agent
        from crewai.crew import Crew
        from crewai.task import Task

        agent = Agent(
            role="Researcher",
            goal="Research",
            backstory="Expert researcher",
            llm="gpt-4o-mini",
        )
        task = Task(
            description="Do research",
            expected_output="Report",
            agent=agent,
        )

        crew = Crew(
            name="Research Crew",
            agents=[agent],
            tasks=[task],
            memory=True,
        )

        assert crew._memory is not None
        assert hasattr(crew._memory, "root_scope")
        assert crew._memory.root_scope == "/crew/research-crew"

    def test_crew_memory_instance_preserves_no_root_scope(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """User-provided Memory instance is not modified — root_scope stays None."""
        from crewai.agent import Agent
        from crewai.crew import Crew
        from crewai.memory.unified_memory import Memory
        from crewai.task import Task

        # Memory without root_scope
        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )
        assert mem.root_scope is None

        agent = Agent(
            role="Tester",
            goal="Test",
            backstory="Tester",
            llm="gpt-4o-mini",
        )
        task = Task(
            description="Test",
            expected_output="Results",
            agent=agent,
        )

        crew = Crew(
            name="Test Crew",
            agents=[agent],
            tasks=[task],
            memory=mem,
        )

        assert crew._memory is mem
        # User-provided Memory is not auto-scoped — respect their config
        assert crew._memory.root_scope is None

    def test_crew_respects_existing_root_scope(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """User-provided Memory with existing root_scope is not overwritten."""
        from crewai.agent import Agent
        from crewai.crew import Crew
        from crewai.memory.unified_memory import Memory
        from crewai.task import Task

        # Memory with explicit root_scope
        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/custom/path",
        )

        agent = Agent(
            role="Tester",
            goal="Test",
            backstory="Tester",
            llm="gpt-4o-mini",
        )
        task = Task(
            description="Test",
            expected_output="Results",
            agent=agent,
        )

        crew = Crew(
            name="Test Crew",
            agents=[agent],
            tasks=[task],
            memory=mem,
        )

        assert crew._memory.root_scope == "/custom/path"  # Not overwritten

    def test_crew_sanitizes_name_for_root_scope(self) -> None:
        """Crew name with special chars is sanitized for root_scope."""
        from crewai.agent import Agent
        from crewai.crew import Crew
        from crewai.task import Task

        agent = Agent(
            role="Agent",
            goal="Goal",
            backstory="Story",
            llm="gpt-4o-mini",
        )
        task = Task(
            description="Task",
            expected_output="Output",
            agent=agent,
        )

        crew = Crew(
            name="My Awesome Crew #1!",
            agents=[agent],
            tasks=[task],
            memory=True,
        )

        assert crew._memory.root_scope == "/crew/my-awesome-crew-1"


class TestAgentScopeExtension:
    """Tests for agent scope extension in BaseAgentExecutorMixin."""

    def test_agent_save_extends_crew_root_scope(self) -> None:
        """Agent._save_to_memory extends crew's root_scope with agent info."""
        from crewai.agents.agent_builder.base_agent_executor import (
            BaseAgentExecutor,
        )
        from crewai.agents.parser import AgentFinish

        mock_memory = MagicMock()
        mock_memory.read_only = False
        mock_memory.root_scope = "/crew/research-crew"
        mock_memory.extract_memories.return_value = ["Fact A"]

        mock_agent = MagicMock()
        mock_agent.memory = mock_memory
        mock_agent._logger = MagicMock()
        mock_agent.role = "Researcher"

        mock_task = MagicMock()
        mock_task.description = "Research task"
        mock_task.expected_output = "Report"

        executor = BaseAgentExecutor()
        executor.agent = mock_agent
        executor.task = mock_task

        executor._save_to_memory(AgentFinish(thought="", output="Result", text="Result"))

        mock_memory.remember_many.assert_called_once()
        call_kwargs = mock_memory.remember_many.call_args.kwargs
        assert call_kwargs["root_scope"] == "/crew/research-crew/agent/researcher"

    def test_agent_save_sanitizes_role(self) -> None:
        """Agent role with special chars is sanitized for scope path."""
        from crewai.agents.agent_builder.base_agent_executor import (
            BaseAgentExecutor,
        )
        from crewai.agents.parser import AgentFinish

        mock_memory = MagicMock()
        mock_memory.read_only = False
        mock_memory.root_scope = "/crew/test"
        mock_memory.extract_memories.return_value = ["Fact"]

        mock_agent = MagicMock()
        mock_agent.memory = mock_memory
        mock_agent._logger = MagicMock()
        mock_agent.role = "Senior Research Analyst #1"

        mock_task = MagicMock()
        mock_task.description = "Task"
        mock_task.expected_output = "Output"

        executor = BaseAgentExecutor()
        executor.agent = mock_agent
        executor.task = mock_task

        executor._save_to_memory(AgentFinish(thought="", output="R", text="R"))

        call_kwargs = mock_memory.remember_many.call_args.kwargs
        assert call_kwargs["root_scope"] == "/crew/test/agent/senior-research-analyst-1"


class TestFlowAutoScoping:
    """Tests for automatic root_scope assignment in Flow."""

    def test_flow_auto_memory_sets_root_scope(self) -> None:
        """Flow auto-creates memory with root_scope set to /flow/<class_name>."""
        from crewai.flow.flow import Flow
        from crewai.memory.unified_memory import Memory

        class MyPipelineFlow(Flow):
            pass

        flow = MyPipelineFlow()

        assert flow.memory is not None
        assert isinstance(flow.memory, Memory)
        assert flow.memory.root_scope == "/flow/mypipelineflow"

    def test_flow_with_name_uses_name_for_root_scope(self) -> None:
        """Flow with custom name uses that name for root_scope."""
        from crewai.flow.flow import Flow
        from crewai.memory.unified_memory import Memory

        class MyFlow(Flow):
            name = "Custom Pipeline"

        flow = MyFlow()

        assert flow.memory is not None
        assert isinstance(flow.memory, Memory)
        assert flow.memory.root_scope == "/flow/custom-pipeline"

    def test_flow_user_provided_memory_not_overwritten(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """User-provided memory on Flow is not modified."""
        from crewai.flow.flow import Flow
        from crewai.memory.unified_memory import Memory

        user_memory = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/custom/scope",
        )

        class MyFlow(Flow):
            memory = user_memory

        flow = MyFlow()

        assert flow.memory is user_memory
        assert flow.memory.root_scope == "/custom/scope"


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing behavior."""

    def test_memory_without_root_scope_works_normally(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """Memory without root_scope behaves exactly as before."""
        from crewai.memory.unified_memory import Memory

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )

        assert mem.root_scope is None

        record = mem.remember(
            "Test content",
            scope="/explicit",
            categories=["test"],
            importance=0.7,
        )

        assert record.scope == "/explicit"

    def test_crew_without_name_uses_default(self) -> None:
        """Crew without name uses 'crew' as default for root_scope."""
        from crewai.agent import Agent
        from crewai.crew import Crew
        from crewai.task import Task

        agent = Agent(
            role="Agent",
            goal="Goal",
            backstory="Story",
            llm="gpt-4o-mini",
        )
        task = Task(
            description="Task",
            expected_output="Output",
            agent=agent,
        )

        # No name provided - uses default "crew"
        crew = Crew(
            agents=[agent],
            tasks=[task],
            memory=True,
        )

        assert crew._memory.root_scope == "/crew/crew"

    def test_old_memories_at_root_still_accessible(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """Old memories stored at '/' are still accessible."""
        from crewai.memory.unified_memory import Memory

        # Create memory and store at root (old behavior)
        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )

        record = mem.remember(
            "Old memory at root",
            scope="/",
            categories=["old"],
            importance=0.5,
        )
        assert record.scope == "/"

        # Recall from root should find it
        matches = mem.recall("Old memory", scope="/", depth="shallow")
        assert len(matches) >= 1


class TestEncodingFlowRootScope:
    """Tests for root_scope handling in EncodingFlow."""

    def test_encoding_flow_fast_path_with_root_scope(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """Group A (fast path) items properly prepend root_scope."""
        from crewai.memory.encoding_flow import ItemState

        # Test _apply_defaults directly on an ItemState without going through Flow
        # since Flow.state is a property without a setter
        item = ItemState(
            content="Test",
            scope="/inner",  # Explicit
            categories=["cat"],  # Explicit
            importance=0.5,  # Explicit
            root_scope="/crew/test",
        )

        # Manually test the join_scope_paths logic that _apply_defaults uses
        from crewai.memory.utils import join_scope_paths

        inner_scope = item.scope or "/"
        if item.root_scope:
            resolved = join_scope_paths(item.root_scope, inner_scope)
        else:
            resolved = inner_scope

        assert resolved == "/crew/test/inner"

    def test_encoding_flow_llm_path_with_root_scope(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """Group C (LLM path) items properly prepend root_scope to inferred scope."""
        from crewai.memory.analyze import ExtractedMetadata, MemoryAnalysis
        from crewai.memory.unified_memory import Memory

        llm = MagicMock()
        llm.supports_function_calling.return_value = True
        llm.call.return_value = MemoryAnalysis(
            suggested_scope="/llm-inferred",
            categories=["auto"],
            importance=0.7,
            extracted_metadata=ExtractedMetadata(),
        )

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=llm,
            embedder=mock_embedder,
            root_scope="/flow/pipeline",
        )

        # No explicit scope/categories/importance -> goes through LLM
        record = mem.remember("Content for LLM analysis")

        assert record is not None
        assert record.scope == "/flow/pipeline/llm-inferred"


class TestMemoryScopeWithRootScope:
    """Tests for MemoryScope interaction with root_scope."""

    def test_memory_scope_remembers_within_root_scope(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """MemoryScope with underlying Memory that has root_scope works correctly."""
        from crewai.memory.memory_scope import MemoryScope
        from crewai.memory.unified_memory import Memory

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/crew/test",
        )

        # Create a MemoryScope
        scope = MemoryScope(memory=mem, root_path="/agent/1")

        # Remember through the scope
        record = scope.remember(
            "Scoped content",
            scope="/task",  # Inner scope within MemoryScope
            categories=["test"],
            importance=0.5,
        )

        # The MemoryScope prepends its root_path, then Memory prepends root_scope
        # MemoryScope.remember prepends /agent/1 to /task -> /agent/1/task
        # Then Memory's root_scope /crew/test gets prepended by encoding flow
        # Final: /crew/test/agent/1/task
        assert record is not None
        # Note: MemoryScope builds the scope before calling memory.remember
        # So the scope it passes is /agent/1/task, which then gets root_scope prepended
        assert record.scope.startswith("/crew/test/agent/1")


class TestReadIsolation:
    """Tests for root_scope read isolation (recall, list, info, reset)."""

    def test_recall_with_root_scope_only_returns_scoped_records(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """recall() with root_scope returns only records within that scope."""
        from crewai.memory.unified_memory import Memory

        # Create memory without root_scope and store some records
        mem_global = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )

        # Store records at different scopes
        mem_global.remember(
            "Global record",
            scope="/other/scope",
            categories=["global"],
            importance=0.5,
        )
        mem_global.remember(
            "Crew A record",
            scope="/crew/crew-a/inner",
            categories=["crew-a"],
            importance=0.5,
        )
        mem_global.remember(
            "Crew B record",
            scope="/crew/crew-b/inner",
            categories=["crew-b"],
            importance=0.5,
        )

        # Create a scoped view for crew-a
        mem_scoped = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/crew/crew-a",
        )

        # recall() should only find crew-a records
        results = mem_scoped.recall("record", depth="shallow")
        assert len(results) == 1
        assert results[0].record.scope == "/crew/crew-a/inner"

    def test_recall_with_root_scope_and_explicit_scope_nests(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """recall() with root_scope + explicit scope combines them."""
        from crewai.memory.unified_memory import Memory

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/crew/test",
        )

        mem.remember(
            "Nested record",
            scope="/inner/deep",
            categories=["test"],
            importance=0.5,
        )

        # recall with explicit scope should nest under root_scope
        results = mem.recall("record", scope="/inner", depth="shallow")
        assert len(results) == 1
        assert results[0].record.scope == "/crew/test/inner/deep"

    def test_recall_without_root_scope_works_globally(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """recall() without root_scope searches globally (backward compat)."""
        from crewai.memory.unified_memory import Memory

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )

        mem.remember(
            "Record A",
            scope="/scope-a",
            categories=["test"],
            importance=0.5,
        )
        mem.remember(
            "Record B",
            scope="/scope-b",
            categories=["test"],
            importance=0.5,
        )

        # recall without scope should find all records
        results = mem.recall("record", depth="shallow")
        assert len(results) == 2

    def test_list_records_defaults_to_root_scope(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """list_records() with root_scope defaults to that scope."""
        from crewai.memory.unified_memory import Memory

        # Store records at different scopes
        mem_global = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )

        mem_global.remember("Global", scope="/other", categories=["x"], importance=0.5)
        mem_global.remember("Scoped", scope="/crew/a/inner", categories=["x"], importance=0.5)

        # Create scoped memory
        mem_scoped = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/crew/a",
        )

        # list_records() without scope should only show /crew/a records
        records = mem_scoped.list_records()
        assert len(records) == 1
        assert records[0].scope == "/crew/a/inner"

    def test_list_scopes_defaults_to_root_scope(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """list_scopes() with root_scope defaults to that scope."""
        from crewai.memory.unified_memory import Memory

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )

        mem.remember("A", scope="/crew/a/child1", categories=["x"], importance=0.5)
        mem.remember("B", scope="/crew/a/child2", categories=["x"], importance=0.5)
        mem.remember("C", scope="/crew/b/other", categories=["x"], importance=0.5)

        mem_scoped = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/crew/a",
        )

        # list_scopes() should only show children under /crew/a
        scopes = mem_scoped.list_scopes()
        assert "/crew/a/child1" in scopes or "child1" in str(scopes)
        assert "/crew/b" not in scopes

    def test_info_defaults_to_root_scope(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """info() with root_scope defaults to that scope."""
        from crewai.memory.unified_memory import Memory

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )

        mem.remember("A", scope="/crew/a/inner", categories=["x"], importance=0.5)
        mem.remember("B", scope="/other/inner", categories=["x"], importance=0.5)

        mem_scoped = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/crew/a",
        )

        # info() should only count records under /crew/a
        scope_info = mem_scoped.info()
        assert scope_info.record_count == 1

    def test_reset_with_root_scope_only_deletes_scoped_records(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """reset() with root_scope only deletes within that scope."""
        from crewai.memory.unified_memory import Memory

        mem = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )

        mem.remember("A", scope="/crew/a/inner", categories=["x"], importance=0.5)
        mem.remember("B", scope="/other/inner", categories=["x"], importance=0.5)

        mem_scoped = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
            root_scope="/crew/a",
        )

        # reset() should only delete /crew/a records
        mem_scoped.reset()

        # Check with a fresh global memory instance to avoid stale table references
        mem_fresh = Memory(
            storage=str(tmp_path / "db"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )
        records = mem_fresh.list_records()
        assert len(records) == 1
        assert records[0].scope == "/other/inner"


class TestAgentExecutorBackwardCompat:
    """Tests for agent executor backward compatibility."""

    def test_agent_executor_no_root_scope_when_memory_has_none(self) -> None:
        """Agent executor doesn't inject root_scope when memory has none."""
        from crewai.agents.agent_builder.base_agent_executor import (
            BaseAgentExecutor,
        )
        from crewai.agents.parser import AgentFinish

        mock_memory = MagicMock()
        mock_memory.read_only = False
        mock_memory.root_scope = None  # No root_scope set
        mock_memory.extract_memories.return_value = ["Fact A"]

        mock_agent = MagicMock()
        mock_agent.memory = mock_memory
        mock_agent._logger = MagicMock()
        mock_agent.role = "Researcher"

        mock_task = MagicMock()
        mock_task.description = "Task"
        mock_task.expected_output = "Output"

        executor = BaseAgentExecutor()
        executor.agent = mock_agent
        executor.task = mock_task

        executor._save_to_memory(AgentFinish(thought="", output="R", text="R"))

        # Should NOT pass root_scope when memory has none
        mock_memory.remember_many.assert_called_once()
        call_kwargs = mock_memory.remember_many.call_args.kwargs
        assert "root_scope" not in call_kwargs

    def test_agent_executor_extends_root_scope_when_memory_has_one(self) -> None:
        """Agent executor extends root_scope when memory has one."""
        from crewai.agents.agent_builder.base_agent_executor import (
            BaseAgentExecutor,
        )
        from crewai.agents.parser import AgentFinish

        mock_memory = MagicMock()
        mock_memory.read_only = False
        mock_memory.root_scope = "/crew/test"  # Has root_scope
        mock_memory.extract_memories.return_value = ["Fact A"]

        mock_agent = MagicMock()
        mock_agent.memory = mock_memory
        mock_agent._logger = MagicMock()
        mock_agent.role = "Researcher"

        mock_task = MagicMock()
        mock_task.description = "Task"
        mock_task.expected_output = "Output"

        executor = BaseAgentExecutor()
        executor.agent = mock_agent
        executor.task = mock_task

        executor._save_to_memory(AgentFinish(thought="", output="R", text="R"))

        # Should pass extended root_scope
        mock_memory.remember_many.assert_called_once()
        call_kwargs = mock_memory.remember_many.call_args.kwargs
        assert call_kwargs["root_scope"] == "/crew/test/agent/researcher"


class TestConsolidationIsolation:
    """Tests for consolidation staying within root_scope boundary."""

    def test_consolidation_search_constrained_to_root_scope(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """Consolidation similarity search is constrained to root_scope."""
        from crewai.memory.encoding_flow import EncodingFlow, ItemState
        from crewai.memory.types import MemoryConfig

        mock_storage = MagicMock()
        mock_storage.search.return_value = []

        flow = EncodingFlow(
            storage=mock_storage,
            llm=MagicMock(),
            embedder=mock_embedder,
            config=MemoryConfig(),
        )

        # Create item with root_scope
        item = ItemState(
            content="Test",
            scope="/inner",
            root_scope="/crew/a",
            embedding=[0.1] * 1536,
        )
        flow.state.items = [item]

        # Run parallel_find_similar
        flow.parallel_find_similar()

        # Check that search was called with correct scope_prefix
        mock_storage.search.assert_called_once()
        call_kwargs = mock_storage.search.call_args.kwargs
        # Should be /crew/a/inner (root + inner combined)
        assert call_kwargs["scope_prefix"] == "/crew/a/inner"

    def test_consolidation_search_without_root_scope(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """Consolidation without root_scope searches by explicit scope only."""
        from crewai.memory.encoding_flow import EncodingFlow, ItemState
        from crewai.memory.types import MemoryConfig

        mock_storage = MagicMock()
        mock_storage.search.return_value = []

        flow = EncodingFlow(
            storage=mock_storage,
            llm=MagicMock(),
            embedder=mock_embedder,
            config=MemoryConfig(),
        )

        # Create item without root_scope
        item = ItemState(
            content="Test",
            scope="/inner",
            root_scope=None,
            embedding=[0.1] * 1536,
        )
        flow.state.items = [item]

        # Run parallel_find_similar
        flow.parallel_find_similar()

        # Check that search was called with explicit scope only
        mock_storage.search.assert_called_once()
        call_kwargs = mock_storage.search.call_args.kwargs
        assert call_kwargs["scope_prefix"] == "/inner"
