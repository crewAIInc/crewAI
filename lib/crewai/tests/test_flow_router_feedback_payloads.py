"""Router payloads belong to the method execution that produced them."""

from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from typing import Any, Literal

from crewai.flow import Flow, HumanFeedbackResult, human_feedback, listen, router, start
from crewai.flow.async_feedback.types import (
    HumanFeedbackPending,
    PendingFeedbackContext,
)
from crewai.flow.persistence import SQLiteFlowPersistence, persist
from pydantic import BaseModel
import pytest


Mode = Literal["sync", "async", "resume"]


class Reviewer:
    def __init__(self, pause: bool = False) -> None:
        self.pause = pause

    def request_feedback(self, context: PendingFeedbackContext, flow: Flow[Any]) -> str:
        if self.pause:
            raise HumanFeedbackPending(context=context)
        # An empty reply selects the explicit default without an LLM call.
        return ""


def run_flow(flow: Flow[Any], mode: Mode, inputs: dict[str, Any] | None = None) -> Any:
    if mode == "async":
        return asyncio.run(flow.kickoff_async(inputs=inputs))
    return flow.kickoff(inputs=inputs)


@pytest.mark.parametrize("mode", ["sync", "async", "resume"])
def test_ordinary_router_after_review_keeps_its_own_payload(
    tmp_path: Path, mode: Mode
) -> None:
    persistence = SQLiteFlowPersistence(str(tmp_path / "feedback.db"))

    class InvoiceFlow(Flow):
        @start()
        @human_feedback(message="Review draft", provider=Reviewer(mode == "resume"))
        def review(self) -> str:
            return "reviewed draft"

        @listen(review)
        def invoice(self, feedback: HumanFeedbackResult) -> dict[str, int]:
            assert feedback.method_name == "review"
            return {"amount": 42}

        @router(invoice)
        def choose(self, invoice: dict[str, int]) -> str:
            assert invoice == {"amount": 42}
            return "invoice_ready"

        @listen("invoice_ready")
        def dispatch(self, route: str) -> str:
            return route.upper()

    flow = InvoiceFlow(persistence=persistence)
    result = run_flow(flow, mode)
    if mode == "resume":
        assert isinstance(result, HumanFeedbackPending)
        flow = InvoiceFlow.from_pending(result.context.flow_id, persistence)
        result = flow.resume("")

    assert result == "INVOICE_READY"
    assert len(flow.human_feedback_history) == 1
    assert flow.last_human_feedback is flow.human_feedback_history[0]
    assert flow.last_human_feedback.output == "reviewed draft"


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("review_first", [False, True])
def test_start_router_does_not_reuse_previous_kickoff_feedback(
    mode: Mode, review_first: bool
) -> None:
    class TwoTurnFlow(Flow):
        @start()
        @router(emit=["needs_review", "work"])
        def choose(self) -> str:
            return "needs_review" if self.state.get("needs_review") else "work"

        @listen("needs_review")
        @human_feedback(message="Review draft", provider=Reviewer())
        def review(self, route: str) -> str:
            return "draft from the previous run"

        @listen("work")
        def process(self, route: str) -> str:
            return route.upper()

    flow = TwoTurnFlow()
    run_flow(flow, mode, {"needs_review": review_first})
    previous_feedback = flow.last_human_feedback
    assert run_flow(flow, mode, {"needs_review": False}) == "WORK"
    assert flow.last_human_feedback is previous_feedback
    assert len(flow.human_feedback_history) == int(review_first)


@pytest.mark.parametrize("mode", ["sync", "async", "resume"])
@pytest.mark.parametrize("start_review", [False, True])
def test_real_hitl_outcomes_keep_their_producing_feedback(
    tmp_path: Path, mode: Mode, start_review: bool
) -> None:
    persistence = SQLiteFlowPersistence(str(tmp_path / "approval.db"))
    seen: list[HumanFeedbackResult] = []

    def publish(feedback: HumanFeedbackResult) -> str:
        assert isinstance(feedback, HumanFeedbackResult)
        assert feedback.method_name == "review"
        assert feedback.output == "draft"
        seen.append(feedback)
        return "published"

    if start_review:

        class ReviewFlow(Flow):
            @start()
            @human_feedback(
                message="Review",
                emit=["approved"],
                default_outcome="approved",
                provider=Reviewer(mode == "resume"),
            )
            def review(self) -> str:
                return "draft"

            @listen("approved")
            def on_approved(self, feedback: HumanFeedbackResult) -> str:
                return publish(feedback)
    else:

        class ReviewFlow(Flow):
            @start()
            def draft(self) -> str:
                return "draft"

            @listen(draft)
            @human_feedback(
                message="Review",
                emit=["approved"],
                default_outcome="approved",
                provider=Reviewer(mode == "resume"),
            )
            def review(self, draft: str) -> str:
                return draft

            @listen("approved")
            def on_approved(self, feedback: HumanFeedbackResult) -> str:
                return publish(feedback)

    flow = ReviewFlow(persistence=persistence)
    result = run_flow(flow, mode)
    if mode == "resume":
        assert isinstance(result, HumanFeedbackPending)
        flow = ReviewFlow.from_pending(result.context.flow_id, persistence)
        result = flow.resume("")
    assert result == "published"
    assert seen == flow.human_feedback_history
    assert seen[0] is flow.last_human_feedback


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_chained_reviews_keep_distinct_feedback_objects(mode: Mode) -> None:
    class ChainedReviews(Flow):
        @start()
        @human_feedback(
            message="First",
            emit=["first_done"],
            default_outcome="first_done",
            provider=Reviewer(),
        )
        def first(self) -> str:
            return "first draft"

        @listen("first_done")
        @human_feedback(
            message="Second",
            emit=["second_done"],
            default_outcome="second_done",
            provider=Reviewer(),
        )
        def second(self, feedback: HumanFeedbackResult) -> str:
            assert feedback.method_name == "first"
            return "second draft"

        @listen("second_done")
        def publish(self, feedback: HumanFeedbackResult) -> str:
            assert feedback.method_name == "second"
            return feedback.output

    flow = ChainedReviews()
    assert run_flow(flow, mode) == "second draft"
    assert [feedback.method_name for feedback in flow.human_feedback_history] == [
        "first",
        "second",
    ]


def test_parallel_start_reviews_keep_feedback_across_persistence_waits(
    tmp_path: Path,
) -> None:
    left_saving = threading.Event()
    right_saved = threading.Event()
    seen: dict[str, HumanFeedbackResult] = {}

    class OrderedPersistence(SQLiteFlowPersistence):
        def save_state(
            self,
            flow_uuid: str,
            method_name: str,
            state_data: dict[str, Any] | BaseModel,
        ) -> None:
            if method_name == "left":
                left_saving.set()
                assert right_saved.wait(10), "right review did not persist"
            super().save_state(flow_uuid, method_name, state_data)
            if method_name == "right":
                right_saved.set()

    persistence = OrderedPersistence(str(tmp_path / "parallel.db"))

    @persist(persistence)
    class ParallelReviews(Flow):
        @start()
        @human_feedback(
            message="Left",
            emit=["left_done"],
            default_outcome="left_done",
            provider=Reviewer(),
        )
        async def left(self) -> str:
            return "left draft"

        @start()
        @human_feedback(
            message="Right",
            emit=["right_done"],
            default_outcome="right_done",
            provider=Reviewer(),
        )
        async def right(self) -> str:
            assert await asyncio.to_thread(left_saving.wait, 10)
            return "right draft"

        @listen("left_done")
        def on_left(self, feedback: HumanFeedbackResult) -> None:
            seen["left"] = feedback

        @listen("right_done")
        def on_right(self, feedback: HumanFeedbackResult) -> None:
            seen["right"] = feedback

    flow = ParallelReviews(persistence=persistence)
    asyncio.run(flow.kickoff_async())
    assert seen["left"].method_name == "left"
    assert seen["left"].output == "left draft"
    assert seen["right"].method_name == "right"
    assert seen["right"].output == "right draft"
    assert flow.last_human_feedback is seen["right"]
