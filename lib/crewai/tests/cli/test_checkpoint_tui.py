from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import pytest

from crewai.cli import checkpoint_tui


class _FakeCheckpointConfig:
    def __init__(self, restore_from: str) -> None:
        self.restore_from = restore_from


class _FakeCrew:
    tasks: list[object] = []
    resume_calls = 0
    fork_calls = 0

    @classmethod
    def from_checkpoint(cls, config: _FakeCheckpointConfig) -> "_FakeCrew":
        cls.resume_calls += 1
        raise RuntimeError(f"Crew.from_checkpoint:{config.restore_from}")

    @classmethod
    def fork(cls, config: _FakeCheckpointConfig) -> "_FakeCrew":
        cls.fork_calls += 1
        raise RuntimeError(f"Crew.fork:{config.restore_from}")


class _FakeFlow:
    tasks: list[object] = []
    resume_calls = 0
    fork_calls = 0

    @classmethod
    def from_checkpoint(cls, config: _FakeCheckpointConfig) -> "_FakeFlow":
        cls.resume_calls += 1
        raise RuntimeError(f"Flow.from_checkpoint:{config.restore_from}")

    @classmethod
    def fork(cls, config: _FakeCheckpointConfig) -> "_FakeFlow":
        cls.fork_calls += 1
        raise RuntimeError(f"Flow.fork:{config.restore_from}")


class _ConcreteFakeFlow(_FakeFlow):
    pass


@pytest.fixture(autouse=True)
def _reset_fake_calls() -> None:
    _FakeCrew.resume_calls = 0
    _FakeCrew.fork_calls = 0
    _FakeFlow.resume_calls = 0
    _FakeFlow.fork_calls = 0
    _ConcreteFakeFlow.resume_calls = 0
    _ConcreteFakeFlow.fork_calls = 0


def _install_fake_runtime_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    crew_mod = ModuleType("crewai.crew")
    crew_mod.Crew = _FakeCrew
    flow_mod = ModuleType("crewai.flow.flow")
    flow_mod.Flow = _FakeFlow
    config_mod = ModuleType("crewai.state.checkpoint_config")
    config_mod.CheckpointConfig = _FakeCheckpointConfig

    monkeypatch.setitem(sys.modules, "crewai.crew", crew_mod)
    monkeypatch.setitem(sys.modules, "crewai.flow.flow", flow_mod)
    monkeypatch.setitem(sys.modules, "crewai.state.checkpoint_config", config_mod)

    utils_mod = ModuleType("crewai.cli.utils")
    utils_mod.get_flows = lambda: []
    monkeypatch.setitem(sys.modules, "crewai.cli.utils", utils_mod)


@pytest.mark.asyncio
async def test_run_checkpoint_tui_async_uses_flow_restore_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime_modules(monkeypatch)

    async def fake_run_async(self: checkpoint_tui.CheckpointTUI) -> tuple[str, str, None, None]:
        return ("/tmp/fake-flow.json", "resume", None, None)

    monkeypatch.setattr(checkpoint_tui.CheckpointTUI, "run_async", fake_run_async)
    monkeypatch.setattr(
        checkpoint_tui,
        "_load_selected_checkpoint_entry",
        lambda location: {"entities": [{"type": "flow", "name": "MyFlow"}]},
    )
    monkeypatch.setattr(checkpoint_tui, "_resolve_flow_runner", lambda location: _ConcreteFakeFlow)

    with pytest.raises(
        RuntimeError,
        match="Flow.from_checkpoint:/tmp/fake-flow.json",
    ):
        await checkpoint_tui._run_checkpoint_tui_async("/tmp/unused")

    assert _ConcreteFakeFlow.resume_calls == 1
    assert _FakeCrew.resume_calls == 0


@pytest.mark.asyncio
async def test_run_checkpoint_tui_async_uses_flow_fork_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime_modules(monkeypatch)

    async def fake_run_async(self: checkpoint_tui.CheckpointTUI) -> tuple[str, str, None, None]:
        return ("/tmp/fake-flow.json", "fork", None, None)

    monkeypatch.setattr(checkpoint_tui.CheckpointTUI, "run_async", fake_run_async)
    monkeypatch.setattr(
        checkpoint_tui,
        "_load_selected_checkpoint_entry",
        lambda location: {"entities": [{"type": "flow", "name": "MyFlow"}]},
    )
    monkeypatch.setattr(checkpoint_tui, "_resolve_flow_runner", lambda location: _ConcreteFakeFlow)

    with pytest.raises(RuntimeError, match="Flow.fork:/tmp/fake-flow.json"):
        await checkpoint_tui._run_checkpoint_tui_async("/tmp/unused")

    assert _ConcreteFakeFlow.fork_calls == 1
    assert _FakeCrew.fork_calls == 0


def test_selected_runner_type_defaults_to_crew_for_unknown_entries() -> None:
    assert checkpoint_tui._selected_runner_type("/tmp/unknown.json") == "crew"


def test_selected_runner_type_detects_flow_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        checkpoint_tui,
        "_load_selected_checkpoint_entry",
        lambda location: {"entities": [{"type": "flow"}]},
    )

    assert checkpoint_tui._selected_runner_type(str(Path("/tmp/fake-flow.json"))) == "flow"


def test_resolve_flow_runner_prefers_matching_concrete_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    utils_mod = ModuleType("crewai.cli.utils")

    class MyFlow(_ConcreteFakeFlow):
        name = "MyFlow"

    utils_mod.get_flows = lambda: [MyFlow()]
    monkeypatch.setitem(sys.modules, "crewai.cli.utils", utils_mod)
    monkeypatch.setattr(
        checkpoint_tui,
        "_load_selected_checkpoint_entry",
        lambda location: {"entities": [{"type": "flow", "name": "MyFlow"}]},
    )

    runner = checkpoint_tui._resolve_flow_runner("/tmp/fake-flow.json")

    assert runner is MyFlow
