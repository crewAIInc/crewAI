"""Persist and resume multiple feedback points without orphan notifications."""
import pytest
from pydantic import BaseModel

from crewai.flow import Flow, human_feedback, listen, router, start
from crewai.flow.async_feedback import HumanFeedbackPending, HumanFeedbackProvider
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence


@pytest.mark.parametrize("second_kind", ["listener", "start", "same_batch"])
def test_feedback_requests_wait_for_the_previous_reply(tmp_path, second_kind):
    requests = []

    class Provider(HumanFeedbackProvider):
        def request_feedback(self, context, flow):
            requests.append(context.method_name)
            raise HumanFeedbackPending(context=context)

    trigger = "begin" if second_kind == "same_batch" else "ready"
    second_decorator = start if second_kind == "start" else listen

    class ReviewFlow(Flow):
        _skip_auto_memory = True

        @start()
        def begin(self):
            return "draft"

        @router(begin)
        def route(self):
            return "ready"

        @listen(begin)
        @human_feedback(message="Review A", provider=Provider(), learn=False)
        def review_a(self):
            return "A"

        @listen(review_a)
        def record_a(self, result):
            self.state["reply_a"] = result.feedback

        @second_decorator(trigger)
        @human_feedback(message="Review B", provider=Provider(), learn=False)
        def review_b(self):
            return "B"

        @listen(review_b)
        def record_b(self, result):
            self.state["reply_b"] = result.feedback

        @listen("ready")
        def ship(self):
            self.state["ship_count"] = self.state.get("ship_count", 0) + 1

    persistence = SQLiteFlowPersistence(str(tmp_path / "flows.db"))
    pending = ReviewFlow(persistence=persistence).kickoff()
    assert isinstance(pending, HumanFeedbackPending)
    assert pending.context.method_name == "review_a"
    assert requests == ["review_a"]

    flow_id = pending.context.flow_id
    restored = ReviewFlow.from_pending(flow_id, persistence)
    next_pending = restored.resume("approved A")
    assert isinstance(next_pending, HumanFeedbackPending)
    assert next_pending.context.method_name == "review_b"
    assert requests == ["review_a", "review_b"]

    restored_again = ReviewFlow.from_pending(flow_id, persistence)
    restored_again.resume("approved B")
    assert restored_again.state["reply_a"] == "approved A"
    assert restored_again.state["reply_b"] == "approved B"
    assert restored_again.state["ship_count"] == 1
    assert requests == ["review_a", "review_b"]
    assert persistence.load_pending_feedback(flow_id) is None


class TypedPayload(BaseModel):
    value: int


def test_typed_listener_input_survives_sqlite_reload(tmp_path):
    class Provider(HumanFeedbackProvider):
        def request_feedback(self, context, flow):
            raise HumanFeedbackPending(context=context)

    class TypedFlow(Flow):
        _skip_auto_memory = True

        @start()
        def begin(self) -> TypedPayload:
            return TypedPayload(value=42)

        @listen(begin)
        @human_feedback(message="Review", provider=Provider(), learn=False)
        def review(self, payload: TypedPayload):
            return "draft"

        @listen(begin)
        def record(self, payload: TypedPayload):
            assert isinstance(payload, TypedPayload)
            self.state["received_value"] = payload.value

    persistence = SQLiteFlowPersistence(str(tmp_path / "typed.db"))
    pending = TypedFlow(persistence=persistence).kickoff()
    assert isinstance(pending, HumanFeedbackPending)
    restored = TypedFlow.from_pending(pending.context.flow_id, persistence)
    restored.resume("approved")
    assert restored.state["received_value"] == 42
    assert persistence.load_pending_feedback(pending.context.flow_id) is None



def test_typed_resume_in_a_new_python_process(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys

    worker = Path(__file__).with_name("feedback_process_worker.py")
    for mode in ("pause", "resume"):
        completed = subprocess.run(
            [sys.executable, str(worker), mode, str(tmp_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=35,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads((tmp_path / "process-result.json").read_text()) == {"received": 42}


def test_unsupported_tail_keeps_the_original_request_visible(tmp_path):
    requests = []

    class Opaque:
        pass

    class Provider(HumanFeedbackProvider):
        def request_feedback(self, context, flow):
            requests.append(context.method_name)
            raise HumanFeedbackPending(context=context)

    class UnsupportedFlow(Flow):
        _skip_auto_memory = True

        @start()
        def begin(self):
            return Opaque()

        @listen(begin)
        @human_feedback(message="Review", provider=Provider(), learn=False)
        def review(self, payload):
            return "draft"

        @listen(begin)
        def deferred(self, payload):
            raise AssertionError("Unsupported work must not silently run with a string")

    persistence = SQLiteFlowPersistence(str(tmp_path / "unsupported.db"))
    pending = UnsupportedFlow(persistence=persistence).kickoff()
    assert isinstance(pending, HumanFeedbackPending)
    assert requests == ["review"]
    assert pending.context.continuation_error
    restored = UnsupportedFlow.from_pending(pending.context.flow_id, persistence)
    with pytest.raises(ValueError, match="Cannot resume deferred work"):
        restored.resume("approved")
    assert persistence.load_pending_feedback(pending.context.flow_id) is not None
