"""Regression tests for checkpoint resume listener inputs.

Issue: https://github.com/crewAIInc/crewAI/issues/8980
"""

from __future__ import annotations

import glob

import pytest
import json
import tempfile
import threading
from typing import Any

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.flow_events import (
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.flow.flow import Flow, listen, start
from crewai.state.checkpoint_config import CheckpointConfig


class TestResumePassesRecordedOutput:
    """Listeners of a method that completed before the failure must receive that
    method's recorded output on resume, not None."""

    def test_resumed_listener_receives_recorded_output(self, tmp_path) -> None:
        received: list[Any] = []
        release = threading.Event()
        failures = [1]

        class ChainFlow(Flow[dict]):
            @start()
            def method_a(self) -> str:
                return "A"

            @listen(method_a)
            def method_b(self, value: str) -> str:
                return "B"

            @listen(method_b)
            def method_c(self, value: str) -> str:
                received.append(value)
                if failures[0]:
                    failures[0] -= 1
                    raise RuntimeError("c failed")
                release.set()
                return "C"

        with pytest.raises(RuntimeError, match="c failed"):
            ChainFlow(
                checkpoint=CheckpointConfig(
                    location=str(tmp_path), on_events=["method_execution_finished"]
                )
            ).kickoff()

        assert received == ["B"]

        latest = next(
            path
            for path in glob.glob(f"{tmp_path}/**/*.json", recursive=True)
            if "method_b"
            in json.load(open(path))["entities"][0]["checkpoint_completed_methods"]
        )

        restored = ChainFlow.from_checkpoint(CheckpointConfig(restore_from=latest))
        restored.kickoff()

        # method_b is not re-run (it completed), but method_c must be fed its
        # recorded output instead of None.
        assert received == ["B", "B"]
