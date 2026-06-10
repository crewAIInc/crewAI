from __future__ import annotations

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.flow_events import (
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.flow import Flow, and_, listen, or_, router, start
from crewai.flow.flow_definition import FlowDefinition


class ChainFlow(Flow):
    @start()
    def begin(self):
        self.state["begin_ran"] = True
        return "hello"

    @listen(begin)
    def shout(self, result):
        return result.upper()

    @listen(shout)
    def confirm(self):
        self.state["confirmed"] = True
        return f"confirmed:{self.state['confirmed']}"


CHAIN_YAML = f"""
schema: crewai.flow/v1
name: ChainFlow
methods:
  begin:
    handler: {__name__}:ChainFlow.begin
    start: true
  shout:
    handler: {__name__}:ChainFlow.shout
    listen: begin
  confirm:
    handler: {__name__}:ChainFlow.confirm
    listen: shout
"""


class MergeFlow(Flow):
    @start()
    def begin(self):
        return "go"

    @listen(begin)
    def left(self):
        return "left"

    @listen(begin)
    def right(self):
        return "right"

    @listen(or_(left, right))
    def either(self):
        self.state["either_ran"] = True
        return "either"

    @listen(and_(left, right, either))
    def join(self):
        self.state["joined"] = True
        return "joined"


MERGE_YAML = f"""
schema: crewai.flow/v1
name: MergeFlow
methods:
  begin:
    handler: {__name__}:MergeFlow.begin
    start: true
  left:
    handler: {__name__}:MergeFlow.left
    listen: begin
  right:
    handler: {__name__}:MergeFlow.right
    listen: begin
  either:
    handler: {__name__}:MergeFlow.either
    listen:
      or: [left, right]
  join:
    handler: {__name__}:MergeFlow.join
    listen:
      and: [left, right, either]
"""


class RouteFlow(Flow):
    @start()
    def begin(self):
        return "go"

    @router(begin)
    def decide(self):
        return "left" if self.state.get("direction") == "left" else "right"

    @listen("left")
    def take_left(self):
        return "took-left"

    @listen("right")
    def take_right(self):
        return "took-right"


ROUTE_YAML = f"""
schema: crewai.flow/v1
name: RouteFlow
methods:
  begin:
    handler: {__name__}:RouteFlow.begin
    start: true
  decide:
    handler: {__name__}:RouteFlow.decide
    listen: begin
    router: true
  take_left:
    handler: {__name__}:RouteFlow.take_left
    listen: left
  take_right:
    handler: {__name__}:RouteFlow.take_right
    listen: right
"""


class LoopFlow(Flow):
    @start("retry")
    def step(self):
        self.state["count"] = self.state.get("count", 0) + 1
        return self.state["count"]

    @router(step)
    def decide(self):
        if self.state["count"] < 3:
            return "retry"
        return "done"

    @listen("done")
    def finish(self):
        return f"finished:{self.state['count']}"


LOOP_YAML = f"""
schema: crewai.flow/v1
name: LoopFlow
methods:
  step:
    handler: {__name__}:LoopFlow.step
    start: retry
  decide:
    handler: {__name__}:LoopFlow.decide
    listen: step
    router: true
  finish:
    handler: {__name__}:LoopFlow.finish
    listen: done
"""


def _run_with_events(flow, inputs=None):
    events = []
    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(MethodExecutionStartedEvent)
        def on_started(source, event):
            events.append(event)

        @crewai_event_bus.on(MethodExecutionFinishedEvent)
        def on_finished(source, event):
            events.append(event)

        result = flow.kickoff(inputs=inputs)
    events.sort(key=lambda e: e.timestamp)
    return result, [
        (type(e).__name__, str(e.method_name), e.flow_name) for e in events
    ]


def _state_without_id(flow):
    snapshot = dict(flow.state)
    snapshot.pop("id", None)
    return snapshot


def assert_parity(flow_cls, yaml_str, inputs=None, ordered=True):
    class_flow = flow_cls()
    class_result, class_events = _run_with_events(class_flow, inputs)

    definition = FlowDefinition.from_yaml(yaml_str)
    definition_flow = Flow.from_definition(definition)
    definition_result, definition_events = _run_with_events(definition_flow, inputs)

    assert definition_result == class_result
    assert _state_without_id(definition_flow) == _state_without_id(class_flow)
    if ordered:
        assert definition_flow.method_outputs == class_flow.method_outputs
        assert definition_events == class_events
    else:
        assert sorted(map(repr, definition_flow.method_outputs)) == sorted(
            map(repr, class_flow.method_outputs)
        )
        assert sorted(definition_events) == sorted(class_events)
    return definition_flow, definition_result


def test_simple_chain_parity():
    flow, result = assert_parity(ChainFlow, CHAIN_YAML)
    assert result == "confirmed:True"
    assert flow.method_outputs == ["hello", "HELLO", "confirmed:True"]


def test_and_or_merge_parity():
    flow, _ = assert_parity(MergeFlow, MERGE_YAML, ordered=False)
    assert flow.state["joined"] is True
    assert flow.state["either_ran"] is True


def test_router_label_parity_for_each_branch():
    left_flow, _ = assert_parity(RouteFlow, ROUTE_YAML, inputs={"direction": "left"})
    assert "took-left" in left_flow.method_outputs
    assert "took-right" not in left_flow.method_outputs

    right_flow, _ = assert_parity(RouteFlow, ROUTE_YAML, inputs={"direction": "right"})
    assert "took-right" in right_flow.method_outputs


def test_cyclic_flow_parity():
    flow, result = assert_parity(LoopFlow, LOOP_YAML)
    assert result == "finished:3"
    assert flow.state["count"] == 3


def test_definition_flow_events_use_definition_name():
    definition = FlowDefinition.from_yaml(CHAIN_YAML)
    flow = Flow.from_definition(definition)
    _, events = _run_with_events(flow)
    assert events
    assert all(flow_name == "ChainFlow" for _, _, flow_name in events)


def test_from_definition_missing_handler_raises():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "NoHandlers",
            "methods": {"begin": {"start": True}},
        }
    )

    with pytest.raises(ValueError, match="begin: no handler"):
        Flow.from_definition(definition)


def test_from_definition_unresolvable_handler_raises():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "BadHandlers",
            "methods": {
                "begin": {
                    "start": True,
                    "handler": "definitely_not_a_module_xyz:nope",
                }
            },
        }
    )

    with pytest.raises(ValueError, match="missing or unresolvable handlers.*begin"):
        Flow.from_definition(definition)


def test_from_definition_malformed_handler_raises():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "MalformedHandlers",
            "methods": {"begin": {"start": True, "handler": "no-colon-here"}},
        }
    )

    with pytest.raises(ValueError, match="expected 'module:qualname'"):
        Flow.from_definition(definition)


def test_flow_definition_stamps_handler_refs():
    definition = ChainFlow.flow_definition()

    assert definition.methods["begin"].handler == f"{__name__}:ChainFlow.begin"
    assert definition.methods["shout"].handler == f"{__name__}:ChainFlow.shout"
