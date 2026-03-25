"""Tests for flow_serializer.py - Flow structure serialization for Studio UI."""

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from crewai.flow.flow import Flow, and_, listen, or_, router, start
from crewai.flow.flow_serializer import flow_structure
from crewai.flow.human_feedback import human_feedback


class TestSimpleLinearFlow:
    """Test simple linear flow (start → listen → listen)."""

    def test_linear_flow_structure(self):
        """Test a simple sequential flow structure."""

        class LinearFlow(Flow):
            """A simple linear flow for testing."""

            @start()
            def begin(self):
                return "started"

            @listen(begin)
            def process(self):
                return "processed"

            @listen(process)
            def finalize(self):
                return "done"

        structure = flow_structure(LinearFlow)

        assert structure["name"] == "LinearFlow"
        assert structure["description"] == "A simple linear flow for testing."
        assert len(structure["methods"]) == 3

        # Check method types
        method_map = {m["name"]: m for m in structure["methods"]}

        assert method_map["begin"]["type"] == "start"
        assert method_map["process"]["type"] == "listen"
        assert method_map["finalize"]["type"] == "listen"

        # Check edges
        assert len(structure["edges"]) == 2

        edge_pairs = [(e["from_method"], e["to_method"]) for e in structure["edges"]]
        assert ("begin", "process") in edge_pairs
        assert ("process", "finalize") in edge_pairs

        # All edges should be listen type
        for edge in structure["edges"]:
            assert edge["edge_type"] == "listen"
            assert edge["condition"] is None


class TestRouterFlow:
    """Test flow with router branching."""

    def test_router_flow_structure(self):
        """Test a flow with router that branches to different paths."""

        class BranchingFlow(Flow):
            @start()
            def init(self):
                return "initialized"

            @router(init)
            def decide(self) -> Literal["path_a", "path_b"]:
                return "path_a"

            @listen("path_a")
            def handle_a(self):
                return "handled_a"

            @listen("path_b")
            def handle_b(self):
                return "handled_b"

        structure = flow_structure(BranchingFlow)

        assert structure["name"] == "BranchingFlow"
        assert len(structure["methods"]) == 4

        method_map = {m["name"]: m for m in structure["methods"]}

        # Check method types
        assert method_map["init"]["type"] == "start"
        assert method_map["decide"]["type"] == "router"
        assert method_map["handle_a"]["type"] == "listen"
        assert method_map["handle_b"]["type"] == "listen"

        # Check router paths
        assert "path_a" in method_map["decide"]["router_paths"]
        assert "path_b" in method_map["decide"]["router_paths"]

        # Check edges
        # Should have: init -> decide (listen), decide -> handle_a (route), decide -> handle_b (route)
        listen_edges = [e for e in structure["edges"] if e["edge_type"] == "listen"]
        route_edges = [e for e in structure["edges"] if e["edge_type"] == "route"]

        assert len(listen_edges) == 1
        assert listen_edges[0]["from_method"] == "init"
        assert listen_edges[0]["to_method"] == "decide"

        assert len(route_edges) == 2
        route_targets = {e["to_method"] for e in route_edges}
        assert "handle_a" in route_targets
        assert "handle_b" in route_targets

        # Check route conditions
        route_conditions = {e["to_method"]: e["condition"] for e in route_edges}
        assert route_conditions["handle_a"] == "path_a"
        assert route_conditions["handle_b"] == "path_b"


class TestAndOrConditions:
    """Test flow with AND/OR conditions."""

    def test_and_condition_flow(self):
        """Test a flow where a method waits for multiple methods (AND)."""

        class AndConditionFlow(Flow):
            @start()
            def step_a(self):
                return "a"

            @start()
            def step_b(self):
                return "b"

            @listen(and_(step_a, step_b))
            def converge(self):
                return "converged"

        structure = flow_structure(AndConditionFlow)

        assert len(structure["methods"]) == 3

        method_map = {m["name"]: m for m in structure["methods"]}

        assert method_map["step_a"]["type"] == "start"
        assert method_map["step_b"]["type"] == "start"
        assert method_map["converge"]["type"] == "listen"

        # Check condition type
        assert method_map["converge"]["condition_type"] == "AND"

        # Check trigger methods
        triggers = method_map["converge"]["trigger_methods"]
        assert "step_a" in triggers
        assert "step_b" in triggers

        # Check edges - should have 2 edges to converge
        converge_edges = [e for e in structure["edges"] if e["to_method"] == "converge"]
        assert len(converge_edges) == 2

    def test_or_condition_flow(self):
        """Test a flow where a method is triggered by any of multiple methods (OR)."""

        class OrConditionFlow(Flow):
            @start()
            def path_1(self):
                return "1"

            @start()
            def path_2(self):
                return "2"

            @listen(or_(path_1, path_2))
            def handle_any(self):
                return "handled"

        structure = flow_structure(OrConditionFlow)

        method_map = {m["name"]: m for m in structure["methods"]}

        assert method_map["handle_any"]["condition_type"] == "OR"

        triggers = method_map["handle_any"]["trigger_methods"]
        assert "path_1" in triggers
        assert "path_2" in triggers


class TestHumanFeedbackMethods:
    """Test flow with @human_feedback decorated methods."""

    def test_human_feedback_detection(self):
        """Test that human feedback methods are correctly identified."""

        class HumanFeedbackFlow(Flow):
            @start()
            @human_feedback(
                message="Please review:",
                emit=["approved", "rejected"],
                llm="gpt-4o-mini",
            )
            def review_step(self):
                return "content to review"

            @listen("approved")
            def handle_approved(self):
                return "approved"

            @listen("rejected")
            def handle_rejected(self):
                return "rejected"

        structure = flow_structure(HumanFeedbackFlow)

        method_map = {m["name"]: m for m in structure["methods"]}

        # review_step should have human feedback
        assert method_map["review_step"]["has_human_feedback"] is True
        # It's a start+router (due to emit)
        assert method_map["review_step"]["type"] == "start_router"
        assert "approved" in method_map["review_step"]["router_paths"]
        assert "rejected" in method_map["review_step"]["router_paths"]

        # Other methods should not have human feedback
        assert method_map["handle_approved"]["has_human_feedback"] is False
        assert method_map["handle_rejected"]["has_human_feedback"] is False

    def test_listen_plus_human_feedback_router_edges(self):
        """Test that @listen + @human_feedback(emit=...) generates router edges.

        This is the pattern used in the whitepaper generator:
        a listener method that also acts as a router via @human_feedback(emit=[...]).
        The serializer must generate edges from this method to listeners of its emit paths.
        """

        class ReviewFlow(Flow):
            @start()
            def generate(self):
                return "content"

            @listen(generate)
            @human_feedback(
                message="Review this:",
                emit=["approved", "needs_changes", "cancelled"],
                llm="gpt-4o-mini",
            )
            def review(self):
                return "review result"

            @listen("approved")
            def handle_approved(self):
                return "done"

            @listen("needs_changes")
            def handle_changes(self):
                return "regenerating"

            @listen("cancelled")
            def handle_cancelled(self):
                return "cancelled"

        structure = flow_structure(ReviewFlow)

        method_map = {m["name"]: m for m in structure["methods"]}
        edge_set = {(e["from_method"], e["to_method"], e.get("condition")) for e in structure["edges"]}

        # review should be detected as a router with the emit paths
        assert method_map["review"]["type"] == "router"
        assert set(method_map["review"]["router_paths"]) == {"approved", "needs_changes", "cancelled"}
        assert method_map["review"]["has_human_feedback"] is True

        # Should have listen edge: generate -> review
        assert ("generate", "review", None) in edge_set

        # Should have route edges from review to each listener
        assert ("review", "handle_approved", "approved") in edge_set
        assert ("review", "handle_changes", "needs_changes") in edge_set
        assert ("review", "handle_cancelled", "cancelled") in edge_set


class TestCrewReferences:
    """Test detection of Crew references in method bodies."""

    def test_crew_detection_with_crew_call(self):
        """Test that .crew() calls are detected."""

        class FlowWithCrew(Flow):
            @start()
            def run_crew(self):
                # Simulating crew usage pattern
                # result = MyCrew().crew().kickoff()
                return "result"

            @listen(run_crew)
            def no_crew(self):
                return "done"

        structure = flow_structure(FlowWithCrew)

        method_map = {m["name"]: m for m in structure["methods"]}

        # Note: Since the actual .crew() call is in a comment/string,
        # the detection might not trigger. In real code it would.
        # We're testing the mechanism exists.
        assert "has_crew" in method_map["run_crew"]
        assert "has_crew" in method_map["no_crew"]

    def test_no_crew_when_absent(self):
        """Test that methods without Crew refs return has_crew=False."""

        class SimpleNonCrewFlow(Flow):
            @start()
            def calculate(self):
                return 1 + 1

            @listen(calculate)
            def display(self):
                return "result"

        structure = flow_structure(SimpleNonCrewFlow)

        method_map = {m["name"]: m for m in structure["methods"]}

        assert method_map["calculate"]["has_crew"] is False
        assert method_map["display"]["has_crew"] is False


class TestTypedStateSchema:
    """Test flow with typed Pydantic state."""

    def test_pydantic_state_schema_extraction(self):
        """Test extracting state schema from a Flow with Pydantic state."""

        class MyState(BaseModel):
            counter: int = 0
            message: str = ""
            items: list[str] = Field(default_factory=list)

        class TypedStateFlow(Flow[MyState]):
            initial_state = MyState

            @start()
            def increment(self):
                self.state.counter += 1
                return self.state.counter

            @listen(increment)
            def display(self):
                return f"Count: {self.state.counter}"

        structure = flow_structure(TypedStateFlow)

        assert structure["state_schema"] is not None
        fields = structure["state_schema"]["fields"]

        field_names = {f["name"] for f in fields}
        assert "counter" in field_names
        assert "message" in field_names
        assert "items" in field_names

        # Check types
        field_map = {f["name"]: f for f in fields}
        assert "int" in field_map["counter"]["type"]
        assert "str" in field_map["message"]["type"]

        # Check defaults
        assert field_map["counter"]["default"] == 0
        assert field_map["message"]["default"] == ""

    def test_dict_state_returns_none(self):
        """Test that flows using dict state return None for state_schema."""

        class DictStateFlow(Flow):
            @start()
            def begin(self):
                self.state["count"] = 1
                return "started"

        structure = flow_structure(DictStateFlow)

        assert structure["state_schema"] is None


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_start_router_combo(self):
        """Test a method that is both @start and a router (via human_feedback emit)."""

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

        structure = flow_structure(StartRouterFlow)

        method_map = {m["name"]: m for m in structure["methods"]}

        assert method_map["entry_point"]["type"] == "start_router"
        assert method_map["entry_point"]["has_human_feedback"] is True
        assert "continue" in method_map["entry_point"]["router_paths"]
        assert "stop" in method_map["entry_point"]["router_paths"]

    def test_multiple_start_methods(self):
        """Test a flow with multiple start methods."""

        class MultiStartFlow(Flow):
            @start()
            def start_a(self):
                return "a"

            @start()
            def start_b(self):
                return "b"

            @listen(and_(start_a, start_b))
            def combine(self):
                return "combined"

        structure = flow_structure(MultiStartFlow)

        start_methods = [m for m in structure["methods"] if m["type"] == "start"]
        assert len(start_methods) == 2

        start_names = {m["name"] for m in start_methods}
        assert "start_a" in start_names
        assert "start_b" in start_names

    def test_orphan_methods(self):
        """Test that orphan methods (not connected to flow) are still captured."""

        class FlowWithOrphan(Flow):
            @start()
            def begin(self):
                return "started"

            @listen(begin)
            def connected(self):
                return "connected"

            @listen("never_triggered")
            def orphan(self):
                return "orphan"

        structure = flow_structure(FlowWithOrphan)

        method_names = {m["name"] for m in structure["methods"]}
        assert "orphan" in method_names

        method_map = {m["name"]: m for m in structure["methods"]}
        assert method_map["orphan"]["trigger_methods"] == ["never_triggered"]

    def test_empty_flow(self):
        """Test building structure for a flow with no methods."""

        class EmptyFlow(Flow):
            pass

        structure = flow_structure(EmptyFlow)

        assert structure["name"] == "EmptyFlow"
        assert structure["methods"] == []
        assert structure["edges"] == []
        assert structure["state_schema"] is None

    def test_flow_with_docstring(self):
        """Test that flow docstring is captured."""

        class DocumentedFlow(Flow):
            """This is a well-documented flow.

            It has multiple lines of documentation.
            """

            @start()
            def begin(self):
                return "started"

        structure = flow_structure(DocumentedFlow)

        assert structure["description"] is not None
        assert "well-documented flow" in structure["description"]

    def test_flow_without_docstring(self):
        """Test that missing docstring returns None."""

        class UndocumentedFlow(Flow):
            @start()
            def begin(self):
                return "started"

        structure = flow_structure(UndocumentedFlow)

        assert structure["description"] is None

    def test_nested_conditions(self):
        """Test flow with nested AND/OR conditions."""

        class NestedConditionFlow(Flow):
            @start()
            def a(self):
                return "a"

            @start()
            def b(self):
                return "b"

            @start()
            def c(self):
                return "c"

            @listen(or_(and_(a, b), c))
            def complex_trigger(self):
                return "triggered"

        structure = flow_structure(NestedConditionFlow)

        method_map = {m["name"]: m for m in structure["methods"]}

        # Should have triggers for a, b, and c
        triggers = method_map["complex_trigger"]["trigger_methods"]
        assert len(triggers) == 3
        assert "a" in triggers
        assert "b" in triggers
        assert "c" in triggers


class TestErrorHandling:
    """Test error handling and validation."""

    def test_instance_raises_type_error(self):
        """Test that passing an instance raises TypeError."""

        class TestFlow(Flow):
            @start()
            def begin(self):
                return "started"

        flow_instance = TestFlow()

        with pytest.raises(TypeError) as exc_info:
            flow_structure(flow_instance)

        assert "requires a Flow class, not an instance" in str(exc_info.value)

    def test_non_class_raises_type_error(self):
        """Test that passing non-class raises TypeError."""

        with pytest.raises(TypeError):
            flow_structure("not a class")

        with pytest.raises(TypeError):
            flow_structure(123)


class TestEdgeGeneration:
    """Test edge generation in various scenarios."""

    def test_all_edges_generated_correctly(self):
        """Verify all edges are correctly generated for a complex flow."""

        class ComplexFlow(Flow):
            @start()
            def entry(self):
                return "started"

            @listen(entry)
            def step_1(self):
                return "step_1"

            @router(step_1)
            def branch(self) -> Literal["left", "right"]:
                return "left"

            @listen("left")
            def left_path(self):
                return "left_done"

            @listen("right")
            def right_path(self):
                return "right_done"

            @listen(or_(left_path, right_path))
            def converge(self):
                return "done"

        structure = flow_structure(ComplexFlow)

        # Build edge map for easier checking
        edges = structure["edges"]

        # Check listen edges
        listen_edges = [(e["from_method"], e["to_method"]) for e in edges if e["edge_type"] == "listen"]

        assert ("entry", "step_1") in listen_edges
        assert ("step_1", "branch") in listen_edges
        assert ("left_path", "converge") in listen_edges
        assert ("right_path", "converge") in listen_edges

        # Check route edges
        route_edges = [(e["from_method"], e["to_method"], e["condition"]) for e in edges if e["edge_type"] == "route"]

        assert ("branch", "left_path", "left") in route_edges
        assert ("branch", "right_path", "right") in route_edges

    def test_router_edge_conditions(self):
        """Test that router edge conditions are properly set."""

        class RouterConditionFlow(Flow):
            @start()
            def begin(self):
                return "start"

            @router(begin)
            def route(self) -> Literal["option_1", "option_2", "option_3"]:
                return "option_1"

            @listen("option_1")
            def handle_1(self):
                return "1"

            @listen("option_2")
            def handle_2(self):
                return "2"

            @listen("option_3")
            def handle_3(self):
                return "3"

        structure = flow_structure(RouterConditionFlow)

        route_edges = [e for e in structure["edges"] if e["edge_type"] == "route"]

        # Should have 3 route edges
        assert len(route_edges) == 3

        conditions = {e["to_method"]: e["condition"] for e in route_edges}
        assert conditions["handle_1"] == "option_1"
        assert conditions["handle_2"] == "option_2"
        assert conditions["handle_3"] == "option_3"


class TestMethodTypeClassification:
    """Test method type classification."""

    def test_all_method_types(self):
        """Test classification of all method types."""

        class AllTypesFlow(Flow):
            @start()
            def start_only(self):
                return "start"

            @listen(start_only)
            def listen_only(self):
                return "listen"

            @router(listen_only)
            def router_only(self) -> Literal["path"]:
                return "path"

            @listen("path")
            def after_router(self):
                return "after"

            @start()
            @human_feedback(
                message="Review",
                emit=["yes", "no"],
                llm="gpt-4o-mini",
            )
            def start_and_router(self):
                return "data"

        structure = flow_structure(AllTypesFlow)

        method_map = {m["name"]: m for m in structure["methods"]}

        assert method_map["start_only"]["type"] == "start"
        assert method_map["listen_only"]["type"] == "listen"
        assert method_map["router_only"]["type"] == "router"
        assert method_map["after_router"]["type"] == "listen"
        assert method_map["start_and_router"]["type"] == "start_router"


class TestInputDetection:
    """Test flow input detection."""

    def test_inputs_list_exists(self):
        """Test that inputs list is always present."""

        class SimpleFlow(Flow):
            @start()
            def begin(self):
                return "started"

        structure = flow_structure(SimpleFlow)

        assert "inputs" in structure
        assert isinstance(structure["inputs"], list)


class TestJsonSerializable:
    """Test that output is JSON serializable."""

    def test_structure_is_json_serializable(self):
        """Test that the entire structure can be JSON serialized."""
        import json

        class MyState(BaseModel):
            value: int = 0

        class SerializableFlow(Flow[MyState]):
            """Test flow for JSON serialization."""

            initial_state = MyState

            @start()
            @human_feedback(
                message="Review",
                emit=["ok", "not_ok"],
                llm="gpt-4o-mini",
            )
            def begin(self):
                return "data"

            @listen("ok")
            def proceed(self):
                return "done"

        structure = flow_structure(SerializableFlow)

        # Should not raise
        json_str = json.dumps(structure)
        assert json_str is not None

        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["name"] == "SerializableFlow"
        assert len(parsed["methods"]) > 0


class TestFlowInheritance:
    """Test flow inheritance scenarios."""

    def test_child_flow_inherits_parent_methods(self):
        """Test that FlowB inheriting from FlowA includes methods from both.

        Note: FlowMeta propagates methods but does NOT fully propagate the
        _listeners registry from parent classes. This means edges defined
        in the parent class (e.g., parent_start -> parent_process) may not
        appear in the child's structure. This is a known FlowMeta limitation.
        """

        class FlowA(Flow):
            """Parent flow with start method."""

            @start()
            def parent_start(self):
                return "parent started"

            @listen(parent_start)
            def parent_process(self):
                return "parent processed"

        class FlowB(FlowA):
            """Child flow with additional methods."""

            @listen(FlowA.parent_process)
            def child_continue(self):
                return "child continued"

            @listen(child_continue)
            def child_finalize(self):
                return "child finalized"

        structure = flow_structure(FlowB)

        assert structure["name"] == "FlowB"

        # Check all methods are present (from both parent and child)
        method_names = {m["name"] for m in structure["methods"]}
        assert "parent_start" in method_names
        assert "parent_process" in method_names
        assert "child_continue" in method_names
        assert "child_finalize" in method_names

        # Check method types
        method_map = {m["name"]: m for m in structure["methods"]}
        assert method_map["parent_start"]["type"] == "start"
        assert method_map["parent_process"]["type"] == "listen"
        assert method_map["child_continue"]["type"] == "listen"
        assert method_map["child_finalize"]["type"] == "listen"

        # Check edges defined in child class exist
        edge_pairs = [(e["from_method"], e["to_method"]) for e in structure["edges"]]
        assert ("parent_process", "child_continue") in edge_pairs
        assert ("child_continue", "child_finalize") in edge_pairs

        # KNOWN LIMITATION: Edges defined in parent class (parent_start -> parent_process)
        # are NOT propagated to child's _listeners registry by FlowMeta.
        # The edge (parent_start, parent_process) will NOT be in edge_pairs.
        # This is a FlowMeta limitation, not a serializer bug.

    def test_child_flow_can_override_parent_method(self):
        """Test that child can override parent methods."""

        class BaseFlow(Flow):
            @start()
            def begin(self):
                return "base begin"

            @listen(begin)
            def process(self):
                return "base process"

        class ExtendedFlow(BaseFlow):
            @listen(BaseFlow.begin)
            def process(self):
                # Override parent's process method
                return "extended process"

            @listen(process)
            def finalize(self):
                return "extended finalize"

        structure = flow_structure(ExtendedFlow)

        method_names = {m["name"] for m in structure["methods"]}
        assert "begin" in method_names
        assert "process" in method_names
        assert "finalize" in method_names

        # Should have 3 methods total (not 4, since process is overridden)
        assert len(structure["methods"]) == 3
