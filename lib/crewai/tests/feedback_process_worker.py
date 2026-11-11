"""Subprocess fixture: pause in one interpreter, restore in a new interpreter."""
import json
from pathlib import Path
import sys

from pytest_recording.network import block

block(allowed_hosts=[r"^127[.]0[.]0[.]1$", r"^::1$", r"^localhost$"])

from pydantic import BaseModel

from crewai.flow import Flow, human_feedback, listen, start
from crewai.flow.async_feedback import HumanFeedbackPending, HumanFeedbackProvider
from crewai.flow.persistence import SQLiteFlowPersistence


class Payload(BaseModel):
    value: int


class Provider(HumanFeedbackProvider):
    def request_feedback(self, context, flow):
        raise HumanFeedbackPending(context=context)


class ReviewFlow(Flow):
    _skip_auto_memory = True

    @start()
    def begin(self) -> Payload:
        return Payload(value=42)

    @listen(begin)
    @human_feedback(message="review", provider=Provider(), learn=False)
    def review(self, payload: Payload):
        return "draft"

    @listen(begin)
    def record(self, payload: Payload):
        assert isinstance(payload, Payload)
        self.state["received"] = payload.value


mode, folder = sys.argv[1], Path(sys.argv[2])
persistence = SQLiteFlowPersistence(str(folder / "process.db"))
if mode == "pause":
    pending = ReviewFlow(persistence=persistence).kickoff()
    assert isinstance(pending, HumanFeedbackPending)
    (folder / "flow-id.txt").write_text(pending.context.flow_id, encoding="utf-8")
else:
    flow_id = (folder / "flow-id.txt").read_text(encoding="utf-8")
    flow = ReviewFlow.from_pending(flow_id, persistence)
    flow.resume("approved")
    assert persistence.load_pending_feedback(flow_id) is None
    (folder / "process-result.json").write_text(json.dumps({"received": flow.state["received"]}), encoding="utf-8")
