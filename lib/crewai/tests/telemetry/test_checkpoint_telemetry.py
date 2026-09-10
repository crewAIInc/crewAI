"""Runtime checkpoint operations report feature usage, including Python-only calls."""

from collections import Counter
from unittest.mock import patch

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.event_listener import event_listener
from crewai.events.types.checkpoint_events import (
    CheckpointCompletedEvent,
    CheckpointFailedEvent,
    CheckpointForkCompletedEvent,
    CheckpointPrunedEvent,
    CheckpointRestoreCompletedEvent,
    CheckpointRestoreFailedEvent,
)
from crewai.state.checkpoint_config import CheckpointConfig
from crewai.state.checkpoint_listener import _do_checkpoint
from crewai.state.provider.json_provider import JsonProvider
from crewai.state.runtime import RuntimeState

from ..utils import wait_for_event_handlers


@pytest.fixture
def features(monkeypatch):
    recorded = []
    with crewai_event_bus.scoped_handlers():
        event_listener.setup_listeners(crewai_event_bus)
        monkeypatch.setattr(
            event_listener._telemetry, "feature_usage_span", recorded.append
        )
        yield recorded
        wait_for_event_handlers()


def test_manual_save_restore_and_fork(tmp_path, features):
    state = RuntimeState(root=[])
    location = state.checkpoint(str(tmp_path))
    restored = RuntimeState.from_checkpoint(CheckpointConfig(restore_from=location))
    restored.fork("private-branch")
    wait_for_event_handlers()

    assert Counter(features) == {
        "checkpoint:save": 1,
        "checkpoint:restore": 1,
        "checkpoint:fork": 1,
    }


@pytest.mark.asyncio
async def test_async_save_and_restore(tmp_path, features):
    state = RuntimeState(root=[])
    location = await state.acheckpoint(str(tmp_path))
    await RuntimeState.afrom_checkpoint(CheckpointConfig(restore_from=location))
    wait_for_event_handlers()

    assert Counter(features) == {"checkpoint:save": 1, "checkpoint:restore": 1}


def test_automatic_checkpoint_and_prune(tmp_path, features):
    _do_checkpoint(
        RuntimeState(root=[]),
        CheckpointConfig(location=str(tmp_path), max_checkpoints=1),
    )
    wait_for_event_handlers()

    assert Counter(features) == {"checkpoint:save": 1, "checkpoint:prune": 1}


def test_failed_operations_do_not_count_as_success(tmp_path, features):
    with patch.object(JsonProvider, "checkpoint", side_effect=OSError("private path")):
        with pytest.raises(OSError):
            RuntimeState(root=[]).checkpoint(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        RuntimeState.from_checkpoint(
            CheckpointConfig(restore_from=str(tmp_path / "missing.json"))
        )
    wait_for_event_handlers()

    assert Counter(features) == {
        "checkpoint:save_failed": 1,
        "checkpoint:restore_failed": 1,
    }


@pytest.mark.parametrize(
    "event",
    [
        CheckpointCompletedEvent(
            location="private",
            provider="JsonProvider",
            checkpoint_id="private-id",
            duration_ms=1,
        ),
        CheckpointFailedEvent(
            location="private", provider="JsonProvider", error="private error"
        ),
        CheckpointRestoreCompletedEvent(
            location="private", checkpoint_id="private-id", duration_ms=1
        ),
        CheckpointRestoreFailedEvent(location="private", error="private error"),
        CheckpointForkCompletedEvent(branch="private-branch"),
        CheckpointPrunedEvent(
            location="private",
            provider="JsonProvider",
            removed_count=1,
            max_checkpoints=1,
        ),
    ],
)
def test_replayed_checkpoint_events_are_not_counted(event, features):
    future = crewai_event_bus.replay(None, event)
    if future is not None:
        future.result(timeout=5)
    wait_for_event_handlers()

    assert features == []
