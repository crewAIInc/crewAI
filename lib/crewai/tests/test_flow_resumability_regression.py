"""Regression tests for flow listener resumability fix.

These tests ensure that:
1. HITL flows can resume properly without re-executing completed methods
2. Cyclic flows can re-execute methods on each iteration
"""

from typing import Dict

from crewai.flow.flow import Flow, listen, router, start
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence


def test_hitl_resumption_skips_completed_listeners(tmp_path):
    """Test that HITL resumption skips completed listener methods but continues chains."""
    db_path = tmp_path / "test_flows.db"
    persistence = SQLiteFlowPersistence(str(db_path))
    execution_log = []

    class HitlFlow(Flow[Dict[str, str]]):
        @start()
        def step_1(self):
            execution_log.append("step_1_executed")
            self.state["step1"] = "done"
            return "step1_result"

        @listen(step_1)
        def step_2(self):
            execution_log.append("step_2_executed")
            self.state["step2"] = "done"
            return "step2_result"

        @listen(step_2)
        def step_3(self):
            execution_log.append("step_3_executed")
            self.state["step3"] = "done"
            return "step3_result"

    flow1 = HitlFlow(persistence=persistence)
    flow1.kickoff()
    flow_id = flow1.state["id"]

    assert execution_log == ["step_1_executed", "step_2_executed", "step_3_executed"]

    flow2 = HitlFlow(persistence=persistence)
    flow2._completed_methods = {"step_1", "step_2"}  # Simulate partial completion
    execution_log.clear()

    flow2.kickoff(inputs={"id": flow_id})

    assert "step_1_executed" not in execution_log
    assert "step_2_executed" not in execution_log
    assert "step_3_executed" in execution_log


def test_cyclic_flow_re_executes_on_each_iteration():
    """Test that cyclic flows properly re-execute methods on each iteration."""
    execution_log = []

    class CyclicFlowTest(Flow[Dict[str, str]]):
        iteration = 0
        max_iterations = 3

        @start("loop")
        def step_1(self):
            if self.iteration >= self.max_iterations:
                return None
            execution_log.append(f"step_1_{self.iteration}")
            return f"result_{self.iteration}"

        @listen(step_1)
        def step_2(self):
            execution_log.append(f"step_2_{self.iteration}")

        @router(step_2)
        def step_3(self):
            execution_log.append(f"step_3_{self.iteration}")
            self.iteration += 1
            if self.iteration < self.max_iterations:
                return "loop"
            return "exit"

    flow = CyclicFlowTest()
    flow.kickoff()

    expected = []
    for i in range(3):
        expected.extend([f"step_1_{i}", f"step_2_{i}", f"step_3_{i}"])

    assert execution_log == expected


def test_conditional_start_with_resumption(tmp_path):
    """Test that conditional start methods work correctly with resumption."""
    db_path = tmp_path / "test_flows.db"
    persistence = SQLiteFlowPersistence(str(db_path))
    execution_log = []

    class ConditionalStartFlow(Flow[Dict[str, str]]):
        @start()
        def init(self):
            execution_log.append("init")
            return "initialized"

        @router(init)
        def route_to_branch(self):
            execution_log.append("router")
            return "branch_a"

        @start("branch_a")
        def branch_a_start(self):
            execution_log.append("branch_a_start")
            self.state["branch"] = "a"

        @listen(branch_a_start)
        def branch_a_process(self):
            execution_log.append("branch_a_process")
            self.state["processed"] = "yes"

    flow1 = ConditionalStartFlow(persistence=persistence)
    flow1.kickoff()
    flow_id = flow1.state["id"]

    assert execution_log == ["init", "router", "branch_a_start", "branch_a_process"]

    flow2 = ConditionalStartFlow(persistence=persistence)
    flow2._completed_methods = {"init", "route_to_branch", "branch_a_start"}
    execution_log.clear()

    flow2.kickoff(inputs={"id": flow_id})

    assert execution_log == ["branch_a_process"]


def test_cyclic_flow_with_conditional_start():
    """Test that cyclic flows work properly with conditional start methods."""
    execution_log = []

    class CyclicConditionalFlow(Flow[Dict[str, str]]):
        iteration = 0

        @start()
        def initial(self):
            execution_log.append("initial")
            return "init_done"

        @router(initial)
        def route_to_cycle(self):
            execution_log.append("router_initial")
            return "loop"

        @start("loop")
        def cycle_entry(self):
            execution_log.append(f"cycle_{self.iteration}")
            self.iteration += 1

        @router(cycle_entry)
        def cycle_router(self):
            execution_log.append(f"router_{self.iteration - 1}")
            if self.iteration < 3:
                return "loop"
            return "exit"

    flow = CyclicConditionalFlow()
    flow.kickoff()

    expected = [
        "initial",
        "router_initial",
        "cycle_0",
        "router_0",
        "cycle_1",
        "router_1",
        "cycle_2",
        "router_2",
    ]

    assert execution_log == expected


import pytest
from pydantic import BaseModel


class _CheckpointQuote(BaseModel):
    amount: int


def test_checkpoint_restore_preserves_pydantic_output_for_listener(tmp_path):
    """Regression test for #8974: Pydantic output survives checkpoint restore."""
    from crewai import CheckpointConfig

    failures = [1]

    class OrderFlow(Flow):
        @start()
        def price(self) -> _CheckpointQuote:
            return _CheckpointQuote(amount=42)

        @listen(price)
        def charge(self, quote):
            if failures[0]:
                failures[0] -= 1
                raise RuntimeError("charge failed")
            self.state["charged"] = quote.amount

    with pytest.raises(RuntimeError, match="charge failed"):
        OrderFlow(
            checkpoint=CheckpointConfig(
                location=str(tmp_path), on_events=["method_execution_finished"]
            )
        ).kickoff()

    checkpoints = list(tmp_path.rglob("*.json"))
    assert checkpoints
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

    restored = OrderFlow.from_checkpoint(CheckpointConfig(restore_from=str(latest)))
    restored.kickoff()
    assert restored.state.get("charged") == 42


def test_checkpoint_restore_preserves_human_feedback_result_dataclass(tmp_path):
    """Regression test for #8974: HumanFeedbackResult dataclass survives checkpoint restore."""
    from crewai import CheckpointConfig
    from crewai.flow.async_feedback.types import HumanFeedbackProvider
    from crewai.flow.human_feedback import HumanFeedbackResult, human_feedback

    class _StubProvider(HumanFeedbackProvider):
        def request_feedback(self, context, flow):
            return "approved"

    failures = [1]

    class ReviewFlow(Flow):
        @start()
        @human_feedback(message="Approve?", provider=_StubProvider())
        def ask(self):
            return "draft"

        @listen(ask)
        def record(self, result):
            if failures[0]:
                failures[0] -= 1
                raise RuntimeError("record failed")
            assert isinstance(result, HumanFeedbackResult)
            self.state["feedback"] = result.feedback

    with pytest.raises(RuntimeError, match="record failed"):
        ReviewFlow(
            checkpoint=CheckpointConfig(
                location=str(tmp_path), on_events=["method_execution_finished"]
            )
        ).kickoff()

    checkpoints = list(tmp_path.rglob("*.json"))
    assert checkpoints
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

    restored = ReviewFlow.from_checkpoint(CheckpointConfig(restore_from=str(latest)))
    restored.kickoff()
    assert restored.state.get("feedback") == "approved"


def test_checkpoint_restore_rejects_untrusted_arbitrary_output_type(monkeypatch):
    """Untrusted checkpoint output_type cannot instantiate arbitrary classes without opt-in."""
    from crewai.flow.runtime import _restore_method_output_entry

    class UnannotatedFlow(Flow):
        @start()
        def step(self):
            return {"amount": 1}

    flow = UnannotatedFlow()
    monkeypatch.delenv("CREWAI_DESERIALIZE_CALLBACKS", raising=False)

    tampered = {
        "method": "step",
        "output": {"amount": 42},
        "output_type": f"{_CheckpointQuote.__module__}:{_CheckpointQuote.__qualname__}",
    }
    restored = _restore_method_output_entry(tampered, flow=flow)
    assert restored["output"] == {"amount": 42}
    assert isinstance(restored["output"], dict)

    monkeypatch.setenv("CREWAI_DESERIALIZE_CALLBACKS", "1")
    local_tampered = {
        "method": "step",
        "output": {"amount": 42},
        "output_type": f"{_CheckpointQuote.__module__}:<locals>.Fake",
    }
    restored_local = _restore_method_output_entry(local_tampered, flow=flow)
    assert isinstance(restored_local["output"], dict)

