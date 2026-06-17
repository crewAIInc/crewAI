"""Tests for the static Flow Definition contract."""

from enum import Enum
import importlib
import inspect
import logging
from pathlib import Path
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ValidationError

import crewai.flow.dsl as flow_dsl
import crewai.flow.flow_definition as flow_definition
import crewai.flow.visualization.builder as visualization_builder
from crewai.experimental import ConversationConfig, RouterConfig
from crewai.flow import Flow, and_, human_feedback, listen, or_, persist, router, start


def test_flow_public_exports_are_explicit():
    import crewai.flow.visualization as flow_visualization

    flow_package = importlib.import_module("crewai.flow")

    assert "FlowDefinition" not in flow_package.__all__
    assert "FlowDefinitionDiagnostic" not in flow_package.__all__
    assert "build_flow_definition" not in flow_package.__all__
    assert "flow_structure" not in flow_package.__all__
    assert set(flow_dsl.__all__) == {
        "HumanFeedbackResult",
        "and_",
        "human_feedback",
        "listen",
        "or_",
        "router",
        "start",
    }
    assert set(flow_definition.__all__) == {
        "FlowActionDefinition",
        "FlowCodeActionDefinition",
        "FlowConfigDefinition",
        "FlowConversationalDefinition",
        "FlowConversationalRouterDefinition",
        "FlowCrewActionDefinition",
        "FlowDefinition",
        "FlowDefinitionCondition",
        "FlowDefinitionDiagnostic",
        "FlowDictStateDefinition",
        "FlowEachActionDefinition",
        "FlowEachInnerActionDefinition",
        "FlowExpressionActionDefinition",
        "FlowHumanFeedbackDefinition",
        "FlowJsonSchemaStateDefinition",
        "FlowMethodDefinition",
        "FlowPersistenceDefinition",
        "FlowPydanticStateDefinition",
        "FlowScriptActionDefinition",
        "FlowStateDefinition",
        "FlowToolActionDefinition",
        "FlowUnknownStateDefinition",
    }
    assert "build_flow_structure" in flow_visualization.__all__
    assert "calculate_node_levels" not in flow_visualization.__all__


def test_flow_definition_json_schema_carries_reference_descriptions():
    schema = flow_definition.FlowDefinition.json_schema()
    defs = schema["$defs"]

    assert schema["properties"]["schema"]["description"]
    assert schema["properties"]["methods"]["description"]

    method_properties = defs["FlowMethodDefinition"]["properties"]
    assert method_properties["do"]["description"] == "Action executed when this method runs."
    assert "Trigger condition" in method_properties["listen"]["description"]

    script_properties = defs["FlowScriptActionDefinition"]["properties"]
    assert "trusted inline Python" in script_properties["call"]["description"]
    assert "not interpolated" in script_properties["code"]["description"]
    assert "not sandboxed" in script_properties["code"]["description"]

    state_schema = schema["properties"]["state"]["anyOf"][0]
    assert state_schema["discriminator"]["propertyName"] == "type"
    assert state_schema["discriminator"]["mapping"] == {
        "dict": "#/$defs/FlowDictStateDefinition",
        "json_schema": "#/$defs/FlowJsonSchemaStateDefinition",
        "pydantic": "#/$defs/FlowPydanticStateDefinition",
        "unknown": "#/$defs/FlowUnknownStateDefinition",
    }

    dict_state_properties = defs["FlowDictStateDefinition"]["properties"]
    assert dict_state_properties["type"]["description"]
    assert "ref" not in dict_state_properties

    json_schema_state_properties = defs["FlowJsonSchemaStateDefinition"]["properties"]
    assert json_schema_state_properties["json_schema"]["description"]
    assert "json_schema" in defs["FlowJsonSchemaStateDefinition"]["required"]

    pydantic_state_properties = defs["FlowPydanticStateDefinition"]["properties"]
    assert "Fallback JSON Schema" in pydantic_state_properties["json_schema"][
        "description"
    ]

    each_properties = defs["FlowEachActionDefinition"]["properties"]
    assert "list to iterate" in each_properties["in"]["description"]
    assert "Ordered inner actions" in each_properties["do"]["description"]


def test_flow_definition_json_schema_carries_field_examples_only():
    schema = flow_definition.FlowDefinition.json_schema()
    defs = schema["$defs"]

    for model_name in [
        "FlowDefinition",
        "FlowCodeActionDefinition",
        "FlowToolActionDefinition",
        "FlowCrewActionDefinition",
        "FlowExpressionActionDefinition",
        "FlowScriptActionDefinition",
        "FlowEachActionDefinition",
        "FlowMethodDefinition",
        "FlowDictStateDefinition",
        "FlowJsonSchemaStateDefinition",
        "FlowPydanticStateDefinition",
        "FlowUnknownStateDefinition",
        "FlowConfigDefinition",
        "FlowPersistenceDefinition",
        "FlowHumanFeedbackDefinition",
        "FlowDefinitionDiagnostic",
    ]:
        model_schema = schema if model_name == "FlowDefinition" else defs[model_name]
        assert "examples" not in model_schema

    assert schema["properties"]["name"]["examples"] == ["ResearchFlow"]
    assert schema["properties"]["schema"]["examples"] == ["crewai.flow/v1"]
    assert schema["properties"]["methods"]["examples"][0]["seed"]["do"] == {
        "call": "expression",
        "expr": "state.topic",
    }

    script_properties = defs["FlowScriptActionDefinition"]["properties"]
    assert script_properties["call"]["examples"] == ["script"]
    assert "input.strip()" in script_properties["code"]["examples"][0]
    assert script_properties["language"]["examples"] == ["python"]

    action_properties = defs["FlowCodeActionDefinition"]["properties"]
    assert action_properties["ref"]["examples"] == [
        "my_project.flows:normalize_topic"
    ]
    assert action_properties["with"]["examples"] == [{"topic": "${state.topic}"}]

    each_properties = defs["FlowEachActionDefinition"]["properties"]
    assert each_properties["in"]["examples"] == ["state.rows"]
    assert each_properties["do"]["examples"][0][0]["clean"]["call"] == "script"

    method_properties = defs["FlowMethodDefinition"]["properties"]
    assert method_properties["listen"]["examples"] == [
        "seed",
        {"or": ["approved", "revise"]},
    ]
    assert method_properties["emit"]["examples"] == [["approved", "revise"]]


def test_flow_state_definition_uses_discriminated_branches():
    definition = flow_definition.FlowDefinition.model_validate(
        {
            "name": "TypedStateFlow",
            "state": {
                "type": "json_schema",
                "json_schema": {"type": "object"},
            },
        }
    )

    assert isinstance(
        definition.state,
        flow_definition.FlowJsonSchemaStateDefinition,
    )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        flow_definition.FlowDefinition.model_validate(
            {
                "name": "InvalidStateFlow",
                "state": {
                    "type": "dict",
                    "ref": "my_project.flows:ResearchState",
                },
            }
        )


def test_condition_combinators_return_nested_runtime_tree():
    condition = and_("event_a", "event_b", or_("event_c"))

    assert condition == {
        "type": "AND",
        "conditions": [
            "event_a",
            "event_b",
            {"type": "OR", "conditions": ["event_c"]},
        ],
    }


def test_flow_definition_lowers_nested_conditions():
    class NestedFlow(Flow):
        @start()
        def begin(self):
            return "begin"

        @listen(begin)
        def validated(self):
            return "validated"

        @listen(begin)
        def processed(self):
            return "processed"

        @listen(or_(and_(validated, processed), begin))
        def finalize(self):
            return "done"

    finalize = NestedFlow.flow_definition().methods["finalize"]

    assert finalize.listen == {"or": [{"and": ["validated", "processed"]}, "begin"]}


def test_flow_definition_preserves_single_branch_nested_conditions():
    class AmbiguousFlow(Flow):
        @start()
        def event_a(self):
            return "a"

        @listen(event_a)
        def event_b(self):
            return "b"

        @listen(and_(event_a, event_b, or_("event_c")))
        def event_d(self):
            return "d"

    event_d = AmbiguousFlow.flow_definition().methods["event_d"]

    assert event_d.listen == {"and": ["event_a", "event_b", {"or": ["event_c"]}]}


def test_flow_definition_rejects_invalid_condition():
    with pytest.raises(ValueError, match="Invalid condition"):
        start(123)(lambda self: None)


def test_flow_definition_contract_is_dsl_agnostic():
    source_path = Path(inspect.getsourcefile(flow_definition) or "")
    source = source_path.read_text()

    assert "DSL" not in source
    assert "flow_wrappers" not in source
    assert "build_flow_definition" not in source
    assert "extract_flow_definition" not in source


def test_flow_definition_maps_dsl_to_static_contract():
    class ContractState(BaseModel):
        topic: str = ""

    class ContractFlow(Flow[ContractState]):
        """A flow with every core DSL role."""

        initial_state = ContractState
        stream = True
        max_method_calls = 7

        @start()
        def begin(self):
            return "started"

        @listen(begin)
        def process(self):
            return "processed"

        @router(process)
        def decide(self):
            return "approved"

        @listen(or_("approved", "revise"))
        @human_feedback(
            message="Review this output.",
            emit=["done", "revise"],
            llm="gpt-4o-mini",
            default_outcome="done",
            metadata={"team": "qa"},
            learn=True,
            learn_source="hitl",
            learn_strict=True,
        )
        def review(self):
            return "review"

        @listen(and_(begin, process))
        def audit(self):
            return "audit"

    definition = ContractFlow.flow_definition()

    assert definition.schema_ == "crewai.flow/v1"
    assert definition.name == "ContractFlow"
    assert definition.description == "A flow with every core DSL role."
    assert definition.state is not None
    assert definition.state.type == "pydantic"
    assert definition.state.ref and "ContractState" in definition.state.ref
    assert definition.config.stream is True
    assert definition.config.max_method_calls == 7
    assert definition.conversational is None

    assert definition.methods["begin"].start is True
    assert definition.methods["process"].listen == "begin"

    decide = definition.methods["decide"]
    assert decide.listen == "process"
    assert decide.router is True
    assert decide.emit is None

    review = definition.methods["review"]
    assert review.listen == {"or": ["approved", "revise"]}
    assert review.router is True
    assert review.emit is None
    assert review.human_feedback is not None
    assert review.human_feedback.emit == ["done", "revise"]
    assert review.human_feedback.default_outcome == "done"
    assert review.human_feedback.metadata == {"team": "qa"}
    assert review.human_feedback.learn is True
    assert review.human_feedback.learn_strict is True

    assert definition.methods["audit"].listen == {"and": ["begin", "process"]}
    assert definition.diagnostics == []


def test_flow_definition_excludes_conversational_builtins_for_regular_flows():
    class RegularFlow(Flow):
        @start()
        def begin(self):
            return "begin"

    methods = RegularFlow.flow_definition().methods

    assert RegularFlow.flow_definition().conversational is None
    assert set(methods) == {"begin"}
    assert "conversation_start" not in methods
    assert "route_conversation" not in methods
    assert "converse_turn" not in methods


def test_flow_definition_includes_conversational_builtins_when_enabled():
    class ChatFlow(Flow):
        conversational = True

    definition = ChatFlow.flow_definition()
    methods = definition.methods

    assert definition.conversational is not None
    assert definition.conversational.enabled is True
    assert definition.conversational.defer_trace_finalization is True
    assert definition.conversational.builtin_routes == ["converse", "end"]
    assert "conversation_start" not in methods
    assert "route_conversation" in methods
    assert "converse_turn" in methods
    assert methods["route_conversation"].start is True
    assert methods["route_conversation"].router is True


def test_flow_definition_serializes_conversational_config():
    @ConversationConfig(
        system_prompt="Be concise.",
        llm="gpt-4o-mini",
        router=RouterConfig(
            prompt="Pick a route.",
            routes=["research"],
            default_intent="converse",
            fallback_intent="end",
        ),
        default_intents=["research"],
        visible_agent_outputs=["researcher"],
        defer_trace_finalization=False,
    )
    class ChatFlow(Flow):
        conversational = True

    conversational = ChatFlow.flow_definition().conversational

    assert conversational is not None
    assert conversational.system_prompt == "Be concise."
    assert conversational.llm == "gpt-4o-mini"
    assert conversational.default_intents == ["research"]
    assert conversational.visible_agent_outputs == ["researcher"]
    assert conversational.defer_trace_finalization is False
    assert conversational.router is not None
    assert conversational.router.prompt == "Pick a route."
    assert conversational.router.routes == ["research"]
    assert conversational.router.fallback_intent == "end"


def test_flow_definition_uses_collapsed_conversational_router_start():
    class ChatFlow(Flow):
        conversational = True

        def conversation_start(self) -> str | None:
            return "custom"

    methods = ChatFlow.flow_definition().methods

    assert "conversation_start" not in methods
    assert "route_conversation" in methods
    assert methods["route_conversation"].start is True
    assert methods["route_conversation"].router is True


def test_flow_definition_serializes_human_feedback_metadata():
    marker = object()

    class MetadataFlow(Flow):
        @start()
        def begin(self):
            return "started"

        @listen(begin)
        @human_feedback(message="Review this output.", metadata={"marker": marker})
        def review(self):
            return "review"

    definition = MetadataFlow.flow_definition()
    review = definition.methods["review"]

    assert review.human_feedback is not None
    assert review.human_feedback.metadata == {"ref": "builtins:dict"}
    assert any(
        diagnostic.code == "non_serializable_value"
        and diagnostic.path == "methods.review.human_feedback.metadata"
        for diagnostic in definition.diagnostics
    )
    definition.to_json()


def test_flow_definition_fragments_cover_start_listen_and_condition_sugar():
    class FragmentFlow(Flow):
        @start()
        def begin(self):
            return "begin"

        @start("restart_event")
        def restart(self):
            return "restart"

        @listen(begin)
        def by_callable(self):
            return "callable"

        @listen("manual_event")
        def by_string(self):
            return "string"

        @listen(and_(begin, by_callable))
        def by_and(self):
            return "and"

        @listen(or_(and_("manual_event", by_string), "fallback_event"))
        def nested(self):
            return "nested"

    definition = FragmentFlow.flow_definition()

    assert definition.methods["begin"].start is True
    assert definition.methods["restart"].start == "restart_event"
    assert definition.methods["by_callable"].listen == "begin"
    assert definition.methods["by_string"].listen == "manual_event"
    assert definition.methods["by_and"].listen == {"and": ["begin", "by_callable"]}
    assert definition.methods["nested"].listen == {
        "or": [{"and": ["manual_event", "by_string"]}, "fallback_event"]
    }

    assert not hasattr(FragmentFlow.__dict__["begin"], "__is_start_method__")
    assert not hasattr(FragmentFlow.__dict__["restart"], "__trigger_methods__")
    for method_name in ("by_callable", "by_string", "by_and", "nested"):
        method = FragmentFlow.__dict__[method_name]
        assert not hasattr(method, "__trigger_methods__")
        assert not hasattr(method, "__condition_type__")
        assert not hasattr(method, "__trigger_condition__")


def test_human_feedback_emit_overrides_inner_router_emit():
    class FeedbackOverRouterFlow(Flow):
        @start()
        def begin(self):
            return "data"

        @human_feedback(
            message="Review:",
            emit=["approved", "rejected"],
            llm="gpt-4o-mini",
        )
        @router(begin, emit=["x", "y"])
        def route(self):
            return "approved"

        @listen("approved")
        def proceed(self):
            return "ok"

    route = FeedbackOverRouterFlow.flow_definition().methods["route"]
    assert route.router is True
    assert route.human_feedback is not None
    assert route.human_feedback.emit == ["approved", "rejected"]
    assert route.emit is None


def test_flow_definition_classifies_start_router_from_human_feedback_emit():
    class StartRouterFlow(Flow):
        @start()
        @human_feedback(
            message="Review:",
            emit=["continue", "stop"],
            llm="gpt-4o-mini",
        )
        def entry_point(self):
            return "data"

        @listen("continue")
        def proceed(self):
            return "proceeding"

        @listen("stop")
        def halt(self):
            return "halted"

    definition = StartRouterFlow.flow_definition()
    entry_point = definition.methods["entry_point"]

    assert entry_point.is_start is True
    assert entry_point.router is True
    assert entry_point.human_feedback is not None
    assert entry_point.human_feedback.emit == ["continue", "stop"]
    assert entry_point.emit is None


def test_flow_definition_round_trips_json_and_yaml():
    class RoundTripFlow(Flow):
        @start()
        def begin(self):
            return "started"

        @router(begin)
        def decide(self):
            return "left"

        @listen("left")
        def left(self):
            return "left"

    definition = RoundTripFlow.flow_definition()

    json_round_trip = flow_definition.FlowDefinition.from_json(definition.to_json())
    yaml_round_trip = flow_definition.FlowDefinition.from_yaml(definition.to_yaml())

    assert json_round_trip.to_dict() == definition.to_dict()
    assert yaml_round_trip.to_dict() == definition.to_dict()
    assert yaml_round_trip.methods["decide"].router is True
    assert yaml_round_trip.methods["decide"].listen == "begin"


def test_each_action_round_trips_json_and_yaml():
    definition = flow_definition.FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "EachFlow",
            "methods": {
                "process_rows": {
                    "description": "Process every loaded row.",
                    "start": True,
                    "do": {
                        "call": "each",
                        "in": "state.rows",
                        "do": [
                            {
                                "normalize": {
                                    "call": "tool",
                                    "ref": "my_tools:NormalizeRowTool",
                                    "with": {"row": "${ item }"},
                                }
                            },
                            {
                                "save": {
                                    "call": "code",
                                    "ref": "my_flow:save_row",
                                    "with": {
                                        "row": "${ item }",
                                        "normalized": "${ outputs.normalize }",
                                    },
                                }
                            },
                        ],
                    },
                }
            },
        }
    )

    json_round_trip = flow_definition.FlowDefinition.from_json(definition.to_json())
    yaml_round_trip = flow_definition.FlowDefinition.from_yaml(definition.to_yaml())

    assert json_round_trip.to_dict() == definition.to_dict()
    assert yaml_round_trip.to_dict() == definition.to_dict()
    assert yaml_round_trip.methods["process_rows"].description == (
        "Process every loaded row."
    )
    assert yaml_round_trip.methods["process_rows"].do.call == "each"


def test_flow_definition_rejects_invalid_method_names():
    with pytest.raises(ValueError, match="Flow method names must match"):
        flow_definition.FlowDefinition.from_dict(
            {
                "schema": "crewai.flow/v1",
                "name": "InvalidMethodNameFlow",
                "methods": {
                    "process-rows": {
                        "start": True,
                        "do": {
                            "call": "expression",
                            "expr": "'done'",
                        },
                    }
                },
            }
        )


def test_flow_definition_detects_persist_metadata():
    @persist(verbose=True)
    class PersistedFlow(Flow[dict]):
        initial_state = {}

        @start()
        def begin(self):
            return "started"

        @persist(verbose=False)
        @listen(begin)
        def checkpoint(self):
            return "saved"

    definition = PersistedFlow.flow_definition()

    assert definition.persist is not None
    assert definition.persist.enabled is True
    assert definition.persist.verbose is True

    assert definition.methods["begin"].persist is None

    method_persist = definition.methods["checkpoint"].persist
    assert method_persist is not None
    assert method_persist.enabled is True
    assert method_persist.verbose is False


def test_flow_definition_allows_dynamic_router_emit():
    class DynamicRouterFlow(Flow):
        @start()
        def begin(self):
            return "started"

        @router(begin)
        def decide(self):
            return self.state["dynamic_event"]

    definition = DynamicRouterFlow.flow_definition()

    assert definition.methods["decide"].emit is None
    assert definition.diagnostics == []


def test_flow_definition_infers_literal_router_emit():
    class LiteralRouterFlow(Flow):
        @start()
        def begin(self):
            return "started"

        @router(begin)
        def decide(self) -> Literal["left", "right"]:
            return "left"

        @listen("left")
        def left(self):
            return "left"

        @listen("right")
        def right(self):
            return "right"

    definition = LiteralRouterFlow.flow_definition()

    assert definition.methods["decide"].emit == ["left", "right"]


def test_flow_definition_infers_enum_router_emit():
    class Decision(str, Enum):
        APPROVE = "approve"
        REJECT = "reject"

    class EnumRouterFlow(Flow):
        @start()
        def begin(self):
            return "started"

        @router(begin)
        def decide(self) -> Decision:
            return Decision.APPROVE

        @listen("approve")
        def approve(self):
            return "approve"

        @listen("reject")
        def reject(self):
            return "reject"

    definition = EnumRouterFlow.flow_definition()

    assert definition.methods["decide"].emit == ["approve", "reject"]


def test_flow_definition_infers_literal_union_router_emit():
    class LiteralUnionRouterFlow(Flow):
        @start()
        def begin(self):
            return "started"

        @router(begin)
        def decide(self) -> Literal["left"] | Literal["right"]:
            return "left"

        @listen("left")
        def left(self):
            return "left"

        @listen("right")
        def right(self):
            return "right"

    definition = LiteralUnionRouterFlow.flow_definition()

    assert definition.methods["decide"].emit == ["left", "right"]


def test_flow_definition_infers_annotated_literal_router_emit():
    class AnnotatedRouterFlow(Flow):
        @start()
        def begin(self):
            return "started"

        @router(begin)
        def decide(self) -> Annotated[Literal["left"] | None, "route"]:
            return "left"

    definition = AnnotatedRouterFlow.flow_definition()

    assert definition.methods["decide"].emit == ["left"]


def test_flow_definition_does_not_infer_container_literal_router_emit():
    class ContainerLiteralRouterFlow(Flow):
        @start()
        def begin(self):
            return "started"

        @router(begin)
        def list_route(self) -> list[Literal["left"]]:
            return ["left"]

        @router(begin)
        def dict_route(self) -> dict[str, Literal["right"]]:
            return {"route": "right"}

    definition = ContainerLiteralRouterFlow.flow_definition()

    assert definition.methods["list_route"].emit is None
    assert definition.methods["dict_route"].emit is None


def test_flow_definition_does_not_infer_unannotated_router_body_emit():
    class UnannotatedRouterFlow(Flow):
        @start()
        def begin(self):
            return "started"

        @router(begin)
        def decide(self):
            return "left"

        @listen("left")
        def left(self):
            return "left"

    definition = UnannotatedRouterFlow.flow_definition()

    assert definition.methods["decide"].emit is None


def test_flow_definition_accepts_explicit_router_events():
    class ExplicitRouterFlow(Flow):
        @start()
        def begin(self):
            return "started"

        @router(begin, emit=["left", "right", "left"])
        def decide(self):
            return self.state["dynamic_event"]

        @listen("left")
        def left(self):
            return "left"

        @listen("right")
        def right(self):
            return "right"

    definition = ExplicitRouterFlow.flow_definition()

    assert definition.methods["decide"].emit == ["left", "right"]


def test_flow_definition_preserves_diagnostics_loaded_from_contract():
    definition = flow_definition.FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "LoadedDiagnosticsFlow",
            "methods": {
                "decision": {
                    "do": {"ref": "loaded_flows:LoadedDiagnosticsFlow.decision"},
                    "router": True,
                    "emit": ["continue"],
                }
            },
            "diagnostics": [
                {
                    "code": "serialized_warning",
                    "message": "Preserved serialized diagnostic",
                    "severity": "warning",
                    "path": "methods.decision",
                },
                {
                    "code": "router_without_trigger",
                    "message": "router: true requires either start or listen",
                    "severity": "error",
                    "path": "methods.decision",
                },
            ],
        }
    )

    codes = [diagnostic.code for diagnostic in definition.diagnostics]
    assert "serialized_warning" in codes
    assert codes.count("router_without_trigger") == 1


def test_router_start_false_without_listen_reports_missing_trigger():
    definition = flow_definition.FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "LoadedFlow",
            "methods": {
                "decision": {
                    "do": {"ref": "loaded_flows:LoadedFlow.decision"},
                    "router": True,
                    "start": False,
                    "emit": ["continue"],
                }
            },
        }
    )

    assert any(
        diagnostic.code == "router_without_trigger"
        and diagnostic.path == "methods.decision"
        for diagnostic in definition.diagnostics
    )


def test_router_human_feedback_preserves_existing_router_metadata():
    class RouterHumanFeedbackFlow(Flow):
        @start()
        def begin(self):
            return "started"

        @human_feedback(message="Review route:")
        @router(begin, emit=["approved", "rejected"])
        def decide(self):
            return "approved"

        @listen("approved")
        def approved(self):
            return "approved"

    definition = RouterHumanFeedbackFlow.flow_definition()
    method = definition.methods["decide"]

    assert method.router is True
    assert method.listen == "begin"
    assert method.emit == ["approved", "rejected"]
    assert method.human_feedback is not None


def test_dynamic_router_flow_definition_has_no_diagnostics():
    class LazyDynamicRouterFlow(Flow):
        @start()
        def begin(self):
            return "started"

        @router(begin)
        def decide(self):
            return self.state["dynamic_event"]

    definition = LazyDynamicRouterFlow.flow_definition()
    assert definition.diagnostics == []


def test_dynamic_router_string_listener_is_valid_contract():
    class DynamicRouterListenerFlow(Flow):
        @start()
        def begin(self):
            return "started"

        @router(begin)
        def decide(self):
            return self.state["dynamic_event"]

        @listen("dynamic_event")
        def handle(self):
            return "handled"

    definition = DynamicRouterListenerFlow.flow_definition()

    assert definition.diagnostics == []


def test_static_string_listener_is_allowed_by_contract():
    definition = flow_definition.FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "TypoFlow",
            "methods": {
                "begin": {
                    "do": {"ref": "loaded_flows:TypoFlow.begin"},
                    "start": True,
                },
                "handle": {
                    "do": {"ref": "loaded_flows:TypoFlow.handle"},
                    "listen": "begni",
                },
            },
        }
    )
    assert definition.diagnostics == []


def test_start_false_not_classified_as_start_method():
    definition = flow_definition.FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "ExplicitNonStartFlow",
            "methods": {
                "begin": {
                    "do": {"ref": "loaded_flows:ExplicitNonStartFlow.begin"},
                    "start": True,
                },
                "handle": {
                    "do": {"ref": "loaded_flows:ExplicitNonStartFlow.handle"},
                    "start": False,
                    "listen": "begin",
                },
            },
        }
    )

    assert definition.methods["begin"].is_start is True
    assert definition.methods["handle"].is_start is False

    class ExplicitNonStartFlow(Flow):
        @start()
        def begin(self):
            return "started"

        @listen(begin)
        def handle(self):
            return "handled"

    # Attach the loaded contract (with explicit ``start: false``) so the
    # projections read from it rather than rebuilding from the DSL.
    ExplicitNonStartFlow._flow_definition = definition

    flow = ExplicitNonStartFlow()
    viz_structure = visualization_builder.build_flow_structure(flow)
    assert "handle" not in viz_structure["start_methods"]
    assert viz_structure["nodes"]["handle"]["type"] != "start"


def test_flow_definition_cache_is_not_reused_by_subclasses():
    class ParentFlow(Flow):
        @start()
        def begin(self):
            return "begin"

    parent_definition = ParentFlow.flow_definition()

    class ChildFlow(ParentFlow):
        @listen(ParentFlow.begin)
        def child_step(self):
            return "child"

    child_definition = ChildFlow.flow_definition()

    assert parent_definition.name == "ParentFlow"
    assert child_definition.name == "ChildFlow"
    assert child_definition is not parent_definition
    assert set(child_definition.methods) == {"child_step"}


def test_flow_definition_logs_diagnostics_when_loaded_from_contract(caplog):
    caplog.set_level(logging.WARNING, logger="crewai.flow.flow_definition")

    definition = flow_definition.FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "LoadedFlow",
            "methods": {
                "decision": {
                    "do": {"ref": "loaded_flows:LoadedFlow.decision"},
                    "router": True,
                    "emit": ["continue"],
                }
            },
        }
    )

    assert any(
        diagnostic.code == "router_without_trigger"
        for diagnostic in definition.diagnostics
    )
    assert any(
        record.levelno == logging.ERROR
        and "LoadedFlow" in record.message
        and "router_without_trigger" in record.message
        for record in caplog.records
    )
