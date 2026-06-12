from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.flow_events import (
    FlowCreatedEvent,
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.flow import Flow, and_, human_feedback, listen, or_, router, start
from crewai.flow.async_feedback import PendingFeedbackContext
from crewai.flow.flow import FlowState
from crewai.flow.flow_definition import FlowConfigDefinition, FlowDefinition
from crewai.flow.persistence import persist
from crewai.flow.persistence.base import FlowPersistence
from crewai.state.checkpoint_config import CheckpointConfig
from crewai.types.streaming import FlowStreamingOutput


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
    do:
      call: code
      ref: {__name__}:ChainFlow.begin
    start: true
  shout:
    do:
      ref: {__name__}:ChainFlow.shout
    listen: begin
  confirm:
    do:
      ref: {__name__}:ChainFlow.confirm
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
    do:
      ref: {__name__}:MergeFlow.begin
    start: true
  left:
    do:
      ref: {__name__}:MergeFlow.left
    listen: begin
  right:
    do:
      ref: {__name__}:MergeFlow.right
    listen: begin
  either:
    do:
      ref: {__name__}:MergeFlow.either
    listen:
      or: [left, right]
  join:
    do:
      ref: {__name__}:MergeFlow.join
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
    do:
      ref: {__name__}:RouteFlow.begin
    start: true
  decide:
    do:
      ref: {__name__}:RouteFlow.decide
    listen: begin
    router: true
  take_left:
    do:
      ref: {__name__}:RouteFlow.take_left
    listen: left
  take_right:
    do:
      ref: {__name__}:RouteFlow.take_right
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
    do:
      ref: {__name__}:LoopFlow.step
    start: retry
  decide:
    do:
      ref: {__name__}:LoopFlow.decide
    listen: step
    router: true
  finish:
    do:
      ref: {__name__}:LoopFlow.finish
    listen: done
"""


class CounterState(FlowState):
    count: int = 0
    label: str = "none"


class PydanticStateFlow(Flow[CounterState]):
    @start()
    def begin(self):
        self.state.count += 1
        return self.state.count

    @listen(begin)
    def finish(self):
        self.state.label = f"count={self.state.count}"
        return self.state.label


PYDANTIC_STATE_YAML = f"""
schema: crewai.flow/v1
name: PydanticStateFlow
state:
  type: pydantic
  ref: {__name__}:CounterState
methods:
  begin:
    do:
      ref: {__name__}:PydanticStateFlow.begin
    start: true
  finish:
    do:
      ref: {__name__}:PydanticStateFlow.finish
    listen: begin
"""

PYDANTIC_STATE_OVERLAY_YAML = f"""
schema: crewai.flow/v1
name: PydanticStateFlow
state:
  type: pydantic
  ref: {__name__}:CounterState
  default:
    count: 5
methods:
  begin:
    do:
      ref: {__name__}:PydanticStateFlow.begin
    start: true
  finish:
    do:
      ref: {__name__}:PydanticStateFlow.finish
    listen: begin
"""

JSON_SCHEMA_STATE_YAML = f"""
schema: crewai.flow/v1
name: JsonSchemaStateFlow
state:
  type: json_schema
  json_schema:
    title: CounterState
    type: object
    properties:
      count:
        type: integer
        default: 0
      label:
        type: string
        default: none
methods:
  begin:
    do:
      ref: {__name__}:PydanticStateFlow.begin
    start: true
  finish:
    do:
      ref: {__name__}:PydanticStateFlow.finish
    listen: begin
"""

PYDANTIC_REF_WITH_SCHEMA_FALLBACK_YAML = f"""
schema: crewai.flow/v1
name: SchemaFallbackFlow
state:
  type: pydantic
  ref: definitely_not_a_module_xyz:MissingState
  json_schema:
    title: CounterState
    type: object
    properties:
      count:
        type: integer
        default: 0
      label:
        type: string
        default: none
methods:
  begin:
    do:
      ref: {__name__}:PydanticStateFlow.begin
    start: true
  finish:
    do:
      ref: {__name__}:PydanticStateFlow.finish
    listen: begin
"""

UNRESOLVABLE_STATE_YAML = f"""
schema: crewai.flow/v1
name: UnresolvableStateFlow
state:
  type: pydantic
  ref: definitely_not_a_module_xyz:MissingState
methods:
  begin:
    do:
      ref: {__name__}:ChainFlow.begin
    start: true
"""

DICT_STATE_YAML = f"""
schema: crewai.flow/v1
name: DictStateFlow
state:
  type: dict
  default:
    count: 5
methods:
  begin:
    do:
      ref: {__name__}:ChainFlow.begin
    start: true
"""

UNKNOWN_STATE_YAML = f"""
schema: crewai.flow/v1
name: UnknownStateFlow
state:
  type: unknown
  ref: somewhere:Something
methods:
  begin:
    do:
      ref: {__name__}:ChainFlow.begin
    start: true
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
    snapshot = dict(flow.state.model_dump())
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


def test_definition_method_without_action_is_invalid():
    with pytest.raises(ValidationError, match="do"):
        FlowDefinition.from_dict(
            {
                "schema": "crewai.flow/v1",
                "name": "NoActions",
                "methods": {"begin": {"start": True}},
            }
        )


def test_from_definition_unresolvable_ref_raises():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "BadRefs",
            "methods": {
                "begin": {
                    "start": True,
                    "do": {"ref": "definitely_not_a_module_xyz:nope"},
                }
            },
        }
    )

    with pytest.raises(ValueError, match="unresolvable actions.*begin"):
        Flow.from_definition(definition)


def test_from_definition_malformed_ref_raises():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "MalformedRefs",
            "methods": {"begin": {"start": True, "do": {"ref": "no-colon-here"}}},
        }
    )

    with pytest.raises(ValueError, match="expected 'module:qualname'"):
        Flow.from_definition(definition)


def test_from_definition_local_scope_ref_raises():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "LocalRefs",
            "methods": {
                "begin": {
                    "start": True,
                    "do": {"ref": f"{__name__}:make.<locals>.LocalFlow.begin"},
                }
            },
        }
    )

    with pytest.raises(ValueError, match="expected 'module:qualname'"):
        Flow.from_definition(definition)


def test_flow_definition_stamps_refs():
    definition = ChainFlow.flow_definition()

    assert definition.methods["begin"].do.ref == f"{__name__}:ChainFlow.begin"
    assert definition.methods["shout"].do.ref == f"{__name__}:ChainFlow.shout"


def test_pydantic_state_from_ref_parity():
    flow, result = assert_parity(PydanticStateFlow, PYDANTIC_STATE_YAML)
    assert result == "count=1"
    assert flow.state.count == 1
    assert flow.state.label == "count=1"
    assert flow.state.id


def test_pydantic_state_default_overlay():
    flow = Flow.from_definition(FlowDefinition.from_yaml(PYDANTIC_STATE_OVERLAY_YAML))
    result = flow.kickoff()
    assert result == "count=6"
    assert flow.state.count == 6


def test_json_schema_state():
    flow = Flow.from_definition(FlowDefinition.from_yaml(JSON_SCHEMA_STATE_YAML))
    result = flow.kickoff()
    assert result == "count=1"
    assert flow.state.count == 1
    assert flow.state.label == "count=1"
    assert flow.state.id


def test_json_schema_state_validates_inputs():
    flow = Flow.from_definition(FlowDefinition.from_yaml(JSON_SCHEMA_STATE_YAML))
    with pytest.raises(ValueError, match="Invalid inputs"):
        flow.kickoff(inputs={"count": "not-a-number"})


def test_pydantic_state_falls_back_to_json_schema_when_ref_unimportable():
    flow = Flow.from_definition(
        FlowDefinition.from_yaml(PYDANTIC_REF_WITH_SCHEMA_FALLBACK_YAML)
    )
    result = flow.kickoff()
    assert result == "count=1"
    assert flow.state.count == 1


def test_pydantic_state_without_ref_or_schema_falls_back_to_dict(caplog):
    with caplog.at_level("ERROR"):
        flow = Flow.from_definition(FlowDefinition.from_yaml(UNRESOLVABLE_STATE_YAML))
    assert "falling back to dict state" in caplog.text

    result = flow.kickoff()
    assert result == "hello"
    assert flow.state["begin_ran"] is True
    assert flow.state["id"]


def test_dict_state_is_a_copy_of_default_plus_id():
    definition = FlowDefinition.from_yaml(DICT_STATE_YAML)

    flow = Flow.from_definition(definition)
    assert flow.state["count"] == 5
    assert flow.state["id"]
    flow.kickoff()
    assert flow.state["begin_ran"] is True

    second = Flow.from_definition(definition)
    assert second.state["count"] == 5
    assert "begin_ran" not in second.state
    assert second.state["id"] != flow.state["id"]
    assert definition.state.default == {"count": 5}


def test_unknown_state_type_falls_back_to_dict(caplog):
    with caplog.at_level("WARNING"):
        flow = Flow.from_definition(FlowDefinition.from_yaml(UNKNOWN_STATE_YAML))
    assert "falling back to dict state" in caplog.text

    result = flow.kickoff()
    assert result == "hello"
    assert flow.state["begin_ran"] is True


class StubInputProvider:
    def request_input(self, message, flow, metadata=None):
        return "stub"


class ConfiguredFlow(Flow):
    suppress_flow_events = True
    max_method_calls = 5
    input_provider = StubInputProvider()

    @start()
    def begin(self):
        return "configured"


SUPPRESSED_CHAIN_YAML = (
    CHAIN_YAML
    + """
config:
  suppress_flow_events: true
"""
)

CAPPED_LOOP_YAML = (
    LOOP_YAML
    + """
config:
  max_method_calls: 2
"""
)

STREAMING_CHAIN_YAML = (
    CHAIN_YAML
    + """
config:
  stream: true
"""
)

DEFERRED_CHAIN_YAML = (
    CHAIN_YAML
    + """
config:
  defer_trace_finalization: true
"""
)

INPUT_PROVIDER_CHAIN_YAML = (
    CHAIN_YAML
    + f"""
config:
  input_provider: {__name__}:StubInputProvider
"""
)


def _run_capturing_flow_lifecycle(yaml_str, event_types):
    events = []
    with crewai_event_bus.scoped_handlers():
        for event_type in event_types:

            @crewai_event_bus.on(event_type)
            def capture(source, event):
                events.append(event)

        flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
        result = flow.kickoff()
    return flow, result, events


_LIFECYCLE_EVENTS = [
    FlowCreatedEvent,
    FlowStartedEvent,
    FlowFinishedEvent,
    MethodExecutionStartedEvent,
    MethodExecutionFinishedEvent,
]


def test_config_suppress_flow_events_from_yaml():
    twin_events = []
    with crewai_event_bus.scoped_handlers():
        for event_type in _LIFECYCLE_EVENTS:

            @crewai_event_bus.on(event_type)
            def capture(source, event):
                twin_events.append(type(event).__name__)

        twin_result = ChainFlow(suppress_flow_events=True).kickoff()

    flow, result, events = _run_capturing_flow_lifecycle(
        SUPPRESSED_CHAIN_YAML, _LIFECYCLE_EVENTS
    )
    assert result == twin_result == "confirmed:True"
    assert flow.suppress_flow_events is True
    assert [type(e).__name__ for e in events] == twin_events
    assert not any(
        isinstance(e, (MethodExecutionStartedEvent, MethodExecutionFinishedEvent))
        for e in events
    )


def test_config_max_method_calls_from_yaml():
    flow = Flow.from_definition(FlowDefinition.from_yaml(CAPPED_LOOP_YAML))
    with pytest.raises(RecursionError, match="has been called 2 times"):
        flow.kickoff()


def test_config_stream_from_yaml():
    flow = Flow.from_definition(FlowDefinition.from_yaml(STREAMING_CHAIN_YAML))
    streaming = flow.kickoff()
    assert isinstance(streaming, FlowStreamingOutput)
    for _ in streaming:
        pass
    assert streaming.result == "confirmed:True"
    assert flow.stream is True


def test_config_defer_trace_finalization_from_yaml():
    _, _, baseline_events = _run_capturing_flow_lifecycle(
        CHAIN_YAML, [FlowFinishedEvent]
    )
    assert len(baseline_events) == 1

    flow, result, deferred_events = _run_capturing_flow_lifecycle(
        DEFERRED_CHAIN_YAML, [FlowFinishedEvent]
    )
    assert result == "confirmed:True"
    assert flow.defer_trace_finalization is True
    assert deferred_events == []


def test_config_checkpoint_from_yaml(tmp_path):
    yaml_str = (
        CHAIN_YAML
        + f"""
config:
  checkpoint:
    location: {tmp_path}
"""
    )
    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    assert isinstance(flow.checkpoint, CheckpointConfig)
    assert flow.checkpoint.location == str(tmp_path)


def test_config_input_provider_from_yaml():
    flow = Flow.from_definition(FlowDefinition.from_yaml(INPUT_PROVIDER_CHAIN_YAML))
    assert isinstance(flow.input_provider, StubInputProvider)


def test_round_trip_config_equivalence():
    class_flow = ConfiguredFlow()
    definition = FlowDefinition.from_yaml(ConfiguredFlow.flow_definition().to_yaml())
    definition_flow = Flow.from_definition(definition)

    assert definition.config.suppress_flow_events is True
    assert definition.config.max_method_calls == 5
    assert definition.config.input_provider == f"{__name__}:StubInputProvider"
    assert definition_flow.suppress_flow_events is class_flow.suppress_flow_events
    assert definition_flow.max_method_calls == class_flow.max_method_calls
    assert isinstance(definition_flow.input_provider, StubInputProvider)

    class_result, class_events = _run_with_events(class_flow)
    definition_result, definition_events = _run_with_events(definition_flow)
    assert definition_result == class_result == "configured"
    assert definition_events == class_events


def test_unknown_schema_rejected():
    with pytest.raises(ValidationError, match="schema"):
        FlowDefinition.from_dict(
            {
                "schema": "crewai.flow/v2",
                "name": "FutureSchema",
                "methods": {
                    "begin": {"start": True, "do": {"ref": f"{__name__}:ChainFlow.begin"}}
                },
            }
        )


def test_flow_config_definition_mirrors_flow_fields():
    for name, field in FlowConfigDefinition.model_fields.items():
        assert name in Flow.model_fields
        assert field.get_default(call_default_factory=True) == Flow.model_fields[
            name
        ].get_default(call_default_factory=True)


class DefinitionStoreBackend(FlowPersistence):
    persistence_type: str = "DefinitionStoreBackend"
    store: str = "default"

    saves: ClassVar[dict[str, list[tuple[str, dict[str, Any]]]]] = defaultdict(list)
    pending: ClassVar[dict[str, tuple[dict[str, Any], PendingFeedbackContext]]] = {}

    def init_db(self) -> None:
        pass

    def save_state(self, flow_uuid, method_name, state_data):
        data = state_data if isinstance(state_data, dict) else state_data.model_dump()
        DefinitionStoreBackend.saves[self.store].append((method_name, dict(data)))

    def load_state(self, flow_uuid):
        for _, data in reversed(DefinitionStoreBackend.saves[self.store]):
            if data.get("id") == flow_uuid:
                return data
        return None

    def save_pending_feedback(self, flow_uuid, context, state_data):
        data = state_data if isinstance(state_data, dict) else state_data.model_dump()
        DefinitionStoreBackend.pending[flow_uuid] = (dict(data), context)

    def load_pending_feedback(self, flow_uuid):
        return DefinitionStoreBackend.pending.get(flow_uuid)

    def clear_pending_feedback(self, flow_uuid):
        DefinitionStoreBackend.pending.pop(flow_uuid, None)


def _saved_methods(store):
    return [name for name, _ in DefinitionStoreBackend.saves[store]]


class PersistedFlow(Flow):
    @start()
    def first(self):
        self.state["count"] = self.state.get("count", 0) + 1
        return "one"

    @listen(first)
    def second(self):
        self.state["count"] += 1
        return "two"


def _flow_level_persist_yaml(store):
    return f"""
schema: crewai.flow/v1
name: PersistedFlow
persist:
  enabled: true
  persistence:
    persistence_type: DefinitionStoreBackend
    store: {store}
methods:
  first:
    do:
      ref: {__name__}:PersistedFlow.first
    start: true
  second:
    do:
      ref: {__name__}:PersistedFlow.second
    listen: first
"""


def _method_level_persist_yaml(store):
    return f"""
schema: crewai.flow/v1
name: PersistedFlow
methods:
  first:
    do:
      ref: {__name__}:PersistedFlow.first
    start: true
    persist:
      enabled: true
      persistence:
        persistence_type: DefinitionStoreBackend
        store: {store}
  second:
    do:
      ref: {__name__}:PersistedFlow.second
    listen: first
"""


_CLASS_LEVEL_BACKEND = DefinitionStoreBackend(store="class-decorator")


@persist(_CLASS_LEVEL_BACKEND)
class ClassPersistedFlow(Flow):
    @start()
    def first(self):
        self.state["count"] = self.state.get("count", 0) + 1
        return "one"

    @listen(first)
    def second(self):
        self.state["count"] += 1
        return "two"


_COMBINED_BACKEND = DefinitionStoreBackend(store="combined-decorator")


@persist(_COMBINED_BACKEND)
class CombinedPersistedFlow(Flow):
    @start()
    @persist(_COMBINED_BACKEND)
    def first(self):
        return "one"

    @listen(first)
    def second(self):
        return "two"


class MethodPersistedFlow(Flow):
    @start()
    @persist(DefinitionStoreBackend(store="method-decorator"))
    def first(self):
        self.state["count"] = self.state.get("count", 0) + 1
        return "one"

    @listen(first)
    def second(self):
        self.state["count"] += 1
        return "two"


def test_flow_level_persist_from_yaml_saves_once_per_method():
    yaml_str = _flow_level_persist_yaml("yaml-flow-level")
    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    result = flow.kickoff()

    assert result == "two"
    assert _saved_methods("yaml-flow-level") == ["first", "second"]
    _, final_save = DefinitionStoreBackend.saves["yaml-flow-level"][-1]
    assert final_save["count"] == 2
    assert final_save["id"] == flow.state["id"]


def test_method_level_persist_from_yaml_saves_only_that_method():
    yaml_str = _method_level_persist_yaml("yaml-method-level")
    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    flow.kickoff()

    assert _saved_methods("yaml-method-level") == ["first"]
    _, save = DefinitionStoreBackend.saves["yaml-method-level"][0]
    assert save["count"] == 1


def test_method_level_persist_disabled_wins_over_flow_level():
    yaml_str = f"""
schema: crewai.flow/v1
name: PersistedFlow
persist:
  enabled: true
  persistence:
    persistence_type: DefinitionStoreBackend
    store: yaml-opt-out
methods:
  first:
    do:
      ref: {__name__}:PersistedFlow.first
    start: true
  second:
    do:
      ref: {__name__}:PersistedFlow.second
    listen: first
    persist:
      enabled: false
"""
    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    flow.kickoff()

    assert _saved_methods("yaml-opt-out") == ["first"]


def test_persist_restore_by_id_from_yaml():
    yaml_str = _flow_level_persist_yaml("yaml-restore")

    flow1 = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    flow1.kickoff()
    assert flow1.state["count"] == 2

    flow2 = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    flow2.kickoff(inputs={"id": flow1.state["id"]})
    assert flow2.state["count"] == 4


def test_combined_class_and_method_persist_saves_once_per_method():
    before = len(DefinitionStoreBackend.saves["combined-decorator"])
    CombinedPersistedFlow().kickoff()

    assert _saved_methods("combined-decorator")[before:] == ["first", "second"]


def test_method_level_persist_decorator_saves_only_that_method():
    before = len(DefinitionStoreBackend.saves["method-decorator"])
    MethodPersistedFlow().kickoff()

    assert _saved_methods("method-decorator")[before:] == ["first"]


def test_round_trip_persist_equivalence():
    definition = FlowDefinition.from_yaml(ClassPersistedFlow.flow_definition().to_yaml())

    before = len(DefinitionStoreBackend.saves["class-decorator"])
    flow = Flow.from_definition(definition)
    flow.kickoff()

    assert _saved_methods("class-decorator")[before:] == ["first", "second"]


def test_instance_persistence_overrides_definition_backend():
    before = len(DefinitionStoreBackend.saves["method-decorator"])
    flow = MethodPersistedFlow(
        persistence=DefinitionStoreBackend(store="instance-override")
    )
    flow.kickoff()

    assert _saved_methods("instance-override") == ["first"]
    assert len(DefinitionStoreBackend.saves["method-decorator"]) == before


def test_resume_synthetic_completion_persists():
    backend = DefinitionStoreBackend(store="resume-synthetic")

    class ResumableFlow(Flow):
        @start()
        @persist(DefinitionStoreBackend(store="resume-synthetic"))
        @human_feedback(message="Review:")
        def generate(self):
            return "content"

        @listen(generate)
        def process(self, result):
            return "done"

    context = PendingFeedbackContext(
        flow_id="resume-persist-1",
        flow_class="ResumableFlow",
        method_name="generate",
        method_output="content",
        message="Review:",
    )
    backend.save_pending_feedback(
        "resume-persist-1", context, {"id": "resume-persist-1"}
    )

    flow = ResumableFlow.from_pending("resume-persist-1", backend)
    result = flow.resume("looks good")

    assert result == "done"
    assert _saved_methods("resume-synthetic") == ["generate"]
