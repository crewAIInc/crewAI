"""Tests for the static Flow Definition contract."""

import ast
from enum import Enum
import importlib
import inspect
import logging
from pathlib import Path
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel

import crewai.flow.dsl as flow_dsl
import crewai.flow.flow_definition as flow_definition
import crewai.flow.visualization.builder as visualization_builder
from crewai.flow import Flow, and_, human_feedback, listen, or_, persist, router, start
from crewai.flow.dsl._conditions import is_flow_condition_dict


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
        "FlowConfigDefinition",
        "FlowDefinition",
        "FlowDefinitionCondition",
        "FlowDefinitionDiagnostic",
        "FlowHumanFeedbackDefinition",
        "FlowMethodDefinition",
        "FlowPersistenceDefinition",
        "FlowStateDefinition",
    }
    assert "build_flow_structure" in flow_visualization.__all__
    assert "calculate_node_levels" not in flow_visualization.__all__


def test_flow_condition_dict_accepts_non_string_sequences():
    condition = {
        "type": "OR",
        "conditions": (
            "approved",
            {"type": "AND", "conditions": ("validated", "processed")},
        ),
    }

    assert is_flow_condition_dict(condition)
    assert not is_flow_condition_dict({"type": "OR", "conditions": "approved"})
    assert not is_flow_condition_dict({"type": "OR", "methods": b"approved"})


def test_private_flow_helpers_do_not_have_docstrings():
    import crewai.flow.flow_wrappers as flow_wrappers
    import crewai.flow.human_feedback as human_feedback
    import crewai.flow.persistence.decorators as persistence_decorators
    import crewai.flow.visualization.types as visualization_types

    modules = [
        flow_dsl,
        flow_definition,
        flow_wrappers,
        human_feedback,
        persistence_decorators,
        visualization_builder,
        visualization_types,
    ]
    violations: list[str] = []

    for module in modules:
        source_path = Path(inspect.getsourcefile(module) or "")
        tree = ast.parse(source_path.read_text())
        stack: list[ast.AST] = []
        if getattr(module, "__all__", None) == [] and ast.get_docstring(tree):
            violations.append(f"{source_path}:1:<module>")

        class PrivateDocstringVisitor(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self._check_docstring(node)
                stack.append(node)
                self.generic_visit(node)
                stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._check_docstring(node)
                stack.append(node)
                self.generic_visit(node)
                stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._check_docstring(node)
                stack.append(node)
                self.generic_visit(node)
                stack.pop()

            def _check_docstring(
                self,
                node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
            ) -> None:
                is_dunder = node.name.startswith("__") and node.name.endswith("__")
                is_private_name = node.name.startswith("_") and not is_dunder
                is_nested_function = any(
                    isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                    for parent in stack
                )
                if (is_private_name or is_nested_function) and ast.get_docstring(node):
                    violations.append(f"{source_path}:{node.lineno}:{node.name}")

        PrivateDocstringVisitor().visit(tree)

    assert violations == []


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

    assert set(methods) == {"begin"}
    assert "conversation_start" not in methods
    assert "route_conversation" not in methods
    assert "converse_turn" not in methods


@pytest.mark.skip(
    reason="Experimental conversational inherited built-ins are out of scope for the definition-first start migration."
)
def test_flow_definition_includes_conversational_builtins_when_enabled():
    class ChatFlow(Flow):
        conversational = True

    methods = ChatFlow.flow_definition().methods

    assert "conversation_start" in methods
    assert "route_conversation" in methods
    assert "converse_turn" in methods
    assert methods["conversation_start"].start is True


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
                "begin": {"start": True},
                "handle": {"listen": "begni"},
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
                "begin": {"start": True},
                "handle": {"start": False, "listen": "begin"},
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
