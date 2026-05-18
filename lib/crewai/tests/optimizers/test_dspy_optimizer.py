"""Tests for DSPyOptimizer, OptimizationResult, AgentInstructions, and optimizer events.

Covers:
- Package import without dspy (AC-4)
- Types: OptimizationResult, AgentInstructions (spec 03)
- Events: optimizer lifecycle events (spec 05)
- DSPyOptimizer behavior with mocked dspy (AC-2, AC-5)
- Hook cleanup on normal + exception paths (AC-5)
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# TC-01: Package / type imports require no dspy
# ─────────────────────────────────────────────────────────────────────────────


def test_import_crewai_optimizers_module() -> None:
    """crewai.optimizers package imports without dspy present."""
    import crewai.optimizers  # noqa: F401 — import-only test


def test_import_optimization_result_no_dspy() -> None:
    """OptimizationResult is importable without dspy installed."""
    from crewai.optimizers import OptimizationResult  # noqa: F401


def test_import_agent_instructions_no_dspy() -> None:
    """AgentInstructions is importable without dspy installed."""
    from crewai.optimizers import AgentInstructions  # noqa: F401


def test_import_dspy_optimizer_class_no_dspy() -> None:
    """DSPyOptimizer class is importable even without dspy installed."""
    from crewai.optimizers import DSPyOptimizer  # noqa: F401


def test_dspy_not_imported_at_module_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """crewai.optimizers must not trigger a dspy import at module load time."""
    # Remove dspy from modules cache so we can detect a fresh import
    monkeypatch.delitem(sys.modules, "dspy", raising=False)
    monkeypatch.delitem(sys.modules, "crewai.optimizers", raising=False)
    monkeypatch.delitem(sys.modules, "crewai.optimizers.types", raising=False)
    monkeypatch.delitem(sys.modules, "crewai.optimizers.dspy_optimizer", raising=False)

    import crewai.optimizers  # noqa: F401

    assert "dspy" not in sys.modules, "dspy must not be imported at package load time"


# ─────────────────────────────────────────────────────────────────────────────
# TC-02: OptimizationResult and AgentInstructions construction + properties
# ─────────────────────────────────────────────────────────────────────────────


def test_agent_instructions_defaults() -> None:
    """AgentInstructions fields default to None when no values are supplied."""
    from crewai.optimizers import AgentInstructions

    instr = AgentInstructions()
    assert instr.role is None
    assert instr.goal is None
    assert instr.backstory is None


def test_agent_instructions_with_values() -> None:
    """AgentInstructions stores role, goal, and backstory when provided."""
    from crewai.optimizers import AgentInstructions

    instr = AgentInstructions(role="analyst", goal="find patterns", backstory="expert")
    assert instr.role == "analyst"
    assert instr.goal == "find patterns"
    assert instr.backstory == "expert"


def test_optimization_result_score_delta() -> None:
    """score_delta equals optimized_score minus baseline_score."""
    from crewai.optimizers import AgentInstructions, OptimizationResult

    result = OptimizationResult(
        crew=MagicMock(),
        baseline_score=0.50,
        optimized_score=0.68,
        optimized_instructions={"researcher": AgentInstructions(role="senior researcher")},
        num_trials=10,
    )
    assert result.score_delta == pytest.approx(0.18)


def test_optimization_result_version_id_is_uuid() -> None:
    """version_id is a valid UUID4 string."""
    import uuid

    from crewai.optimizers import OptimizationResult

    result = OptimizationResult(
        crew=MagicMock(),
        baseline_score=0.0,
        optimized_score=0.0,
        optimized_instructions={},
        num_trials=5,
    )
    assert isinstance(result.version_id, str)
    parsed = uuid.UUID(result.version_id)
    assert str(parsed) == result.version_id
    assert parsed.version == 4


def test_optimization_result_version_ids_are_unique() -> None:
    """Each OptimizationResult instance gets a distinct version_id."""
    from crewai.optimizers import OptimizationResult

    r1 = OptimizationResult(crew=MagicMock(), baseline_score=0.0, optimized_score=0.0, optimized_instructions={}, num_trials=1)
    r2 = OptimizationResult(crew=MagicMock(), baseline_score=0.0, optimized_score=0.0, optimized_instructions={}, num_trials=1)
    assert r1.version_id != r2.version_id


# ─────────────────────────────────────────────────────────────────────────────
# TC-03: Optimizer events instantiate with correct types
# ─────────────────────────────────────────────────────────────────────────────


def test_optimization_started_event_type() -> None:
    """OptimizationStartedEvent has type='optimization_started' and carries algorithm."""
    from crewai.events.types import OptimizationStartedEvent

    event = OptimizationStartedEvent(algorithm="MIPROv2", num_trials=20, trainset_size=5)
    assert event.type == "optimization_started"
    assert event.algorithm == "MIPROv2"


def test_optimization_completed_event_type() -> None:
    """OptimizationCompletedEvent has type='optimization_completed' and carries score_delta."""
    from crewai.events.types import OptimizationCompletedEvent

    event = OptimizationCompletedEvent(
        algorithm="MIPROv2",
        baseline_score=0.50,
        optimized_score=0.68,
        score_delta=0.18,
        num_trials=20,
        version_id="abc-123",
    )
    assert event.type == "optimization_completed"
    assert event.score_delta == pytest.approx(0.18)


def test_optimization_failed_event_type() -> None:
    """OptimizationFailedEvent has type='optimization_failed' and carries the error string."""
    from crewai.events.types import OptimizationFailedEvent

    event = OptimizationFailedEvent(error="something went wrong")
    assert event.type == "optimization_failed"
    assert event.error == "something went wrong"


def test_optimization_trial_completed_event_type() -> None:
    """OptimizationTrialCompletedEvent stores algorithm, trial_number, and optional crew_name."""
    # Import directly from optimizer_events — this class is intentionally excluded
    # from crewai.events.types.__all__ until DSPy exposes a per-trial callback.
    from crewai.events.types.optimizer_events import OptimizationTrialCompletedEvent

    event = OptimizationTrialCompletedEvent(
        algorithm="MIPROv2", trial_number=1, trial_score=0.75
    )
    assert event.type == "optimization_trial_completed"
    assert event.algorithm == "MIPROv2"
    assert event.crew_name is None


def test_all_optimizer_events_importable_from_types_package() -> None:
    """Started, completed, and failed events are importable from crewai.events.types."""
    # OptimizationTrialCompletedEvent is intentionally excluded from the public
    # package surface until DSPy adds per-trial callback support.
    from crewai.events.types import (
        OptimizationCompletedEvent,
        OptimizationFailedEvent,
        OptimizationStartedEvent,
    )

    assert OptimizationStartedEvent is not None
    assert OptimizationCompletedEvent is not None
    assert OptimizationFailedEvent is not None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: build a minimal mock crew + DSPyOptimizer with dspy stubbed out
# ─────────────────────────────────────────────────────────────────────────────


def _make_mock_agent(role: str = "researcher") -> MagicMock:
    """Return a MagicMock with role, goal, and backstory attributes set."""
    agent = MagicMock()
    agent.role = role
    agent.goal = "find facts"
    agent.backstory = "expert analyst"
    return agent


def _make_mock_crew(agents: list[Any] | None = None) -> MagicMock:
    """Return a MagicMock crew with a name, agents list, and a kickoff stub."""
    crew = MagicMock()
    crew.name = "test_crew"
    crew.agents = agents or [_make_mock_agent()]
    crew.kickoff.return_value = "mock crew output"
    return crew


def _make_mock_dspy() -> MagicMock:
    """Build a minimal dspy stub that satisfies DSPyOptimizer's usage."""
    mock = MagicMock()

    # dspy.Module base class
    class _FakeModule:
        """Minimal stand-in for dspy.Module."""

        def __init__(self) -> None:
            """No-op initializer."""

    mock.Module = _FakeModule

    # dspy.Prediction
    class _FakePrediction:
        """Minimal stand-in for dspy.Prediction."""

        def __init__(self, **kwargs: Any) -> None:
            """Store all keyword arguments as instance attributes."""
            for k, v in kwargs.items():
                setattr(self, k, v)

    mock.Prediction = _FakePrediction

    # dspy.Signature — return an object with .instructions attribute
    def _fake_signature(spec_str: str, instructions: str = "") -> MagicMock:
        """Return a MagicMock with an .instructions attribute set."""
        sig = MagicMock()
        sig.instructions = instructions
        return sig

    mock.Signature = _fake_signature

    # dspy.ChainOfThought — stores a .predict sub-predictor
    class _FakeChainOfThought(_FakeModule):
        """Minimal stand-in for dspy.ChainOfThought with a .predict sub-predictor."""

        def __init__(self, signature: Any) -> None:
            """Attach a MagicMock predictor holding the given signature."""
            super().__init__()
            self.predict = MagicMock()
            self.predict.signature = signature
            self.predict.demos = []

    mock.ChainOfThought = _FakeChainOfThought

    # dspy.context — acts as a no-op context manager
    import contextlib

    mock.context = MagicMock(return_value=contextlib.nullcontext())

    # dspy.MIPROv2 — teleprompter that returns the student module unchanged
    class _FakeMIPROv2:
        """Minimal MIPROv2 stub that returns the student module unchanged."""

        def __init__(self, metric: Any, num_candidates: int = 5, **kwargs: Any) -> None:
            """Store the metric; ignore other teleprompter config."""
            self.metric = metric

        def compile(self, student: Any, trainset: Any, **kwargs: Any) -> Any:
            """Return the student unchanged (no real optimization)."""
            return student

    mock.MIPROv2 = _FakeMIPROv2

    # dspy.BootstrapFewShot
    class _FakeBootstrapFewShot:
        """Minimal BootstrapFewShot stub that returns the student module unchanged."""

        def __init__(self, metric: Any, **kwargs: Any) -> None:
            """Store the metric; ignore other teleprompter config."""
            self.metric = metric

        def compile(self, student: Any, trainset: Any, **kwargs: Any) -> Any:
            """Return the student unchanged (no real optimization)."""
            return student

    mock.BootstrapFewShot = _FakeBootstrapFewShot

    return mock


@pytest.fixture()
def mock_dspy_env(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the _dspy and _HAS_DSPY globals inside dspy_optimizer module."""
    import crewai.optimizers.dspy_optimizer as opt_mod

    mock = _make_mock_dspy()
    monkeypatch.setattr(opt_mod, "_dspy", mock)
    monkeypatch.setattr(opt_mod, "_HAS_DSPY", True)
    # Re-derive _ModuleBase from the patched _dspy
    monkeypatch.setattr(opt_mod, "_ModuleBase", mock.Module)
    return mock


@pytest.fixture()
def simple_trainset() -> list[MagicMock]:
    """Three fake dspy.Example objects with .inputs() returning a dict."""
    examples = []
    for i in range(3):
        ex = MagicMock()
        ex.inputs.return_value = {"topic": f"topic_{i}"}
        examples.append(ex)
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# TC-04: DSPyOptimizer raises ImportError without dspy (AC-4)
# ─────────────────────────────────────────────────────────────────────────────


def test_dspy_optimizer_raises_import_error_without_dspy(monkeypatch: pytest.MonkeyPatch) -> None:
    """DSPyOptimizer.__init__ raises ImportError when _HAS_DSPY is False."""
    import crewai.optimizers.dspy_optimizer as opt_mod

    monkeypatch.setattr(opt_mod, "_HAS_DSPY", False)
    from crewai.optimizers import DSPyOptimizer

    with pytest.raises(ImportError, match="pip install 'crewai\\[dspy\\]'"):
        DSPyOptimizer(crew=MagicMock(), metric=lambda e, p: 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# TC-05: compile() returns OptimizationResult (AC-2)
# ─────────────────────────────────────────────────────────────────────────────


def test_compile_returns_optimization_result(
    mock_dspy_env: MagicMock, simple_trainset: list[Any]
) -> None:
    """compile() returns an OptimizationResult with the expected fields populated."""
    from crewai.optimizers import DSPyOptimizer, OptimizationResult

    crew = _make_mock_crew()
    optimizer = DSPyOptimizer(crew=crew, metric=lambda e, p: 1.0)
    result = optimizer.compile(trainset=simple_trainset, num_trials=3)

    assert isinstance(result, OptimizationResult)
    assert isinstance(result.score_delta, float)
    assert isinstance(result.version_id, str)
    assert isinstance(result.optimized_instructions, dict)
    assert result.num_trials == 3


def test_compile_raises_on_empty_trainset(
    mock_dspy_env: MagicMock,
) -> None:
    """compile() raises ValueError when trainset is an empty list."""
    from crewai.optimizers import DSPyOptimizer

    optimizer = DSPyOptimizer(crew=_make_mock_crew(), metric=lambda e, p: 1.0)
    with pytest.raises(ValueError, match="non-empty"):
        optimizer.compile(trainset=[], num_trials=3)


def test_compile_raises_on_non_callable_metric(
    mock_dspy_env: MagicMock, simple_trainset: list[Any]
) -> None:
    """compile() raises TypeError when metric is not callable."""
    from crewai.optimizers import DSPyOptimizer

    optimizer = DSPyOptimizer(crew=_make_mock_crew(), metric=lambda e, p: 1.0)
    optimizer.metric = "not_callable"  # type: ignore[assignment]
    with pytest.raises(TypeError, match="callable"):
        optimizer.compile(trainset=simple_trainset, num_trials=3)


# ─────────────────────────────────────────────────────────────────────────────
# TC-06: Hook cleanup — normal completion (AC-5)
# ─────────────────────────────────────────────────────────────────────────────


def test_hooks_cleaned_up_after_normal_compile(
    mock_dspy_env: MagicMock, simple_trainset: list[Any]
) -> None:
    """Before-LLM hooks registered during compile() are unregistered on success."""
    from crewai.hooks.llm_hooks import get_before_llm_call_hooks
    from crewai.optimizers import DSPyOptimizer

    initial_hooks = list(get_before_llm_call_hooks())
    optimizer = DSPyOptimizer(crew=_make_mock_crew(), metric=lambda e, p: 1.0)
    optimizer.compile(trainset=simple_trainset, num_trials=2)

    assert get_before_llm_call_hooks() == initial_hooks, "hooks must be restored after compile"


# ─────────────────────────────────────────────────────────────────────────────
# TC-07: Hook cleanup — exception path (AC-5b)
# ─────────────────────────────────────────────────────────────────────────────


def test_hooks_cleaned_up_after_exception(
    mock_dspy_env: MagicMock, simple_trainset: list[Any]
) -> None:
    """Before-LLM hooks are unregistered even when crew.kickoff() raises during baseline."""
    from crewai.hooks.llm_hooks import get_before_llm_call_hooks
    from crewai.optimizers import DSPyOptimizer

    initial_hooks = list(get_before_llm_call_hooks())
    crew = _make_mock_crew()
    # Make kickoff raise on every call (baseline measurement will fail)
    crew.kickoff.side_effect = RuntimeError("kickoff boom")

    optimizer = DSPyOptimizer(crew=crew, metric=lambda e, p: 1.0)
    with pytest.raises(RuntimeError, match="kickoff boom"):
        optimizer.compile(trainset=simple_trainset, num_trials=2)

    assert get_before_llm_call_hooks() == initial_hooks, "hooks must be restored even on exception"


def test_hooks_cleaned_up_when_teleprompter_raises(
    mock_dspy_env: MagicMock, simple_trainset: list[Any]
) -> None:
    """Before-LLM hooks are unregistered even when the teleprompter raises during compile."""
    import crewai.optimizers.dspy_optimizer as opt_mod
    from crewai.hooks.llm_hooks import get_before_llm_call_hooks
    from crewai.optimizers import DSPyOptimizer

    initial_hooks = list(get_before_llm_call_hooks())

    # Make MIPROv2.compile raise after hooks are registered
    class _BrokenMIPROv2:
        """MIPROv2 stub whose compile() always raises to test hook cleanup on failure."""

        def __init__(self, metric: Any, **kwargs: Any) -> None:
            """Accept but ignore all arguments."""

        def compile(self, student: Any, trainset: Any, **kwargs: Any) -> Any:
            """Always raise to simulate a teleprompter failure."""
            raise RuntimeError("teleprompter exploded")

    mock_dspy_env.MIPROv2 = _BrokenMIPROv2
    optimizer = DSPyOptimizer(crew=_make_mock_crew(), metric=lambda e, p: 1.0)
    with pytest.raises(RuntimeError, match="teleprompter exploded"):
        optimizer.compile(trainset=simple_trainset, num_trials=2)

    assert get_before_llm_call_hooks() == initial_hooks


# ─────────────────────────────────────────────────────────────────────────────
# TC-08: OptimizationStartedEvent emitted during compile() (spec 05 TC-03)
# ─────────────────────────────────────────────────────────────────────────────


def test_optimization_started_event_emitted(
    mock_dspy_env: MagicMock, simple_trainset: list[Any]
) -> None:
    """compile() emits an OptimizationStartedEvent with algorithm and trainset_size."""
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types import OptimizationStartedEvent
    from crewai.optimizers import DSPyOptimizer

    received: list[OptimizationStartedEvent] = []

    def _on_started(src: Any, event: OptimizationStartedEvent) -> None:
        """Collect received OptimizationStartedEvent for assertion."""
        received.append(event)

    crewai_event_bus.on(OptimizationStartedEvent)(_on_started)
    try:
        optimizer = DSPyOptimizer(crew=_make_mock_crew(), metric=lambda e, p: 1.0)
        optimizer.compile(trainset=simple_trainset, num_trials=3)
        crewai_event_bus.flush()  # wait for async event handlers
    finally:
        crewai_event_bus.off(OptimizationStartedEvent, _on_started)

    event = next((e for e in received if e.algorithm == "MIPROv2"), None)
    assert event is not None, "No OptimizationStartedEvent with algorithm='MIPROv2' received"
    assert event.trainset_size == len(simple_trainset)


def test_optimization_completed_event_emitted(
    mock_dspy_env: MagicMock, simple_trainset: list[Any]
) -> None:
    """compile() emits an OptimizationCompletedEvent on success."""
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types import OptimizationCompletedEvent
    from crewai.optimizers import DSPyOptimizer

    received: list[OptimizationCompletedEvent] = []

    def _on_completed(src: Any, event: OptimizationCompletedEvent) -> None:
        """Collect received OptimizationCompletedEvent for assertion."""
        received.append(event)

    crewai_event_bus.on(OptimizationCompletedEvent)(_on_completed)
    try:
        optimizer = DSPyOptimizer(crew=_make_mock_crew(), metric=lambda e, p: 1.0)
        optimizer.compile(trainset=simple_trainset, num_trials=3)
        crewai_event_bus.flush()
    finally:
        crewai_event_bus.off(OptimizationCompletedEvent, _on_completed)

    event = next(
        (e for e in received if e.algorithm == "MIPROv2" and e.num_trials == 3),
        None,
    )
    assert event is not None, "No OptimizationCompletedEvent with algorithm='MIPROv2' and num_trials=3 received"


def test_optimization_failed_event_emitted_on_exception(
    mock_dspy_env: MagicMock, simple_trainset: list[Any]
) -> None:
    """compile() emits an OptimizationFailedEvent when crew.kickoff() raises."""
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types import OptimizationFailedEvent
    from crewai.optimizers import DSPyOptimizer

    crew = _make_mock_crew()
    crew.kickoff.side_effect = RuntimeError("forced failure")

    received: list[OptimizationFailedEvent] = []

    def _on_failed(src: Any, event: OptimizationFailedEvent) -> None:
        """Collect received OptimizationFailedEvent for assertion."""
        received.append(event)

    crewai_event_bus.on(OptimizationFailedEvent)(_on_failed)
    try:
        optimizer = DSPyOptimizer(crew=crew, metric=lambda e, p: 1.0)
        with pytest.raises(RuntimeError):
            optimizer.compile(trainset=simple_trainset, num_trials=2)
        crewai_event_bus.flush()
    finally:
        crewai_event_bus.off(OptimizationFailedEvent, _on_failed)

    event = next((e for e in received if "forced failure" in e.error), None)
    assert event is not None, "No OptimizationFailedEvent with expected error received"


# ─────────────────────────────────────────────────────────────────────────────
# TC-09: Algorithm selection — BootstrapFewShot (no num_candidates in constructor)
# ─────────────────────────────────────────────────────────────────────────────


def test_compile_with_bootstrap_few_shot(
    mock_dspy_env: MagicMock, simple_trainset: list[Any]
) -> None:
    """compile() succeeds with algorithm='BootstrapFewShot' and returns OptimizationResult."""
    from crewai.optimizers import DSPyOptimizer, OptimizationResult

    optimizer = DSPyOptimizer(
        crew=_make_mock_crew(),
        metric=lambda e, p: 0.5,
        algorithm="BootstrapFewShot",
    )
    result = optimizer.compile(trainset=simple_trainset, num_trials=5)
    assert isinstance(result, OptimizationResult)


def test_unknown_algorithm_raises_value_error(
    mock_dspy_env: MagicMock, simple_trainset: list[Any]
) -> None:
    """compile() raises ValueError when an unrecognized algorithm name is used."""
    from crewai.optimizers import DSPyOptimizer

    optimizer = DSPyOptimizer(
        crew=_make_mock_crew(),
        metric=lambda e, p: 1.0,
        algorithm="MIPROv2",
    )
    # Patch algorithm after construction to bypass Literal check
    optimizer.algorithm = "UnknownAlgo"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown algorithm"):
        optimizer.compile(trainset=simple_trainset, num_trials=3)


# ─────────────────────────────────────────────────────────────────────────────
# TC-10: Agent field writeback when optimized instructions present
# ─────────────────────────────────────────────────────────────────────────────


def test_agent_backstory_written_back_when_instructions_found(
    mock_dspy_env: MagicMock, simple_trainset: list[Any]
) -> None:
    """Optimized signature instructions are written back to agent.backstory after compile."""
    import crewai.optimizers.dspy_optimizer as opt_mod
    from crewai.optimizers import DSPyOptimizer

    agent = _make_mock_agent(role="analyst")
    crew = _make_mock_crew(agents=[agent])

    # Override ChainOfThought so its signature returns a new instruction
    class _ChainWithInstructions:
        """ChainOfThought stub whose predictor carries a fixed optimized instruction string."""

        def __init__(self, signature: Any) -> None:
            """Ignore the input signature; always install the hardcoded optimized instruction."""
            self.predict = MagicMock()
            self.predict.signature = MagicMock()
            self.predict.signature.instructions = "Optimized: be concise and accurate."
            self.predict.demos = []

    mock_dspy_env.ChainOfThought = _ChainWithInstructions

    optimizer = DSPyOptimizer(crew=crew, metric=lambda e, p: 1.0)
    optimizer.compile(trainset=simple_trainset, num_trials=2)

    # The agent's backstory should have been updated
    assert agent.backstory == "Optimized: be concise and accurate."
