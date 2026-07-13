"""Tests for flow visualization and structure building."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from crewai.flow.flow import Flow, and_, listen, or_, router, start
from crewai.flow.flow_definition import FlowDefinition
from crewai.flow.visualization import (
    build_flow_structure,
    visualize_flow_structure,
)


class SimpleFlow(Flow):
    """Simple flow for testing basic visualization."""

    @start()
    def begin(self):
        return "started"

    @listen(begin)
    def process(self):
        return "processed"


class RouterFlow(Flow):
    """Flow with router for testing router visualization."""

    @start()
    def init(self):
        return "initialized"

    @router(init)
    def decide(self):
        if hasattr(self, "state") and self.state.get("path") == "b":
            return "event_b"
        return "event_a"

    @listen("event_a")
    def handle_a(self):
        return "handled_a"

    @listen("event_b")
    def handle_b(self):
        return "handled_b"


class ComplexFlow(Flow):
    """Complex flow with AND/OR conditions for testing."""

    @start()
    def start_a(self):
        return "a"

    @start()
    def start_b(self):
        return "b"

    @listen(and_(start_a, start_b))
    def converge_and(self):
        return "and_done"

    @listen(or_(start_a, start_b))
    def converge_or(self):
        return "or_done"

    @router(converge_and)
    def router_decision(self):
        return "final_event"

    @listen("final_event")
    def finalize(self):
        return "complete"


def _attach_flow_definition(
    flow_class: type[Flow], methods: dict[str, dict[str, object]]
) -> None:
    flow_class._flow_definition = FlowDefinition.from_declaration(contents=
        {
            "schema": "crewai.flow/v1",
            "name": flow_class.__name__,
            "methods": {
                name: {
                    "do": {
                        "ref": f"{flow_class.__module__}:{flow_class.__name__}.{name}"
                    },
                    **spec,
                }
                for name, spec in methods.items()
            },
        }
    )


def test_build_flow_structure_simple():
    """Test building structure for a simple sequential flow."""
    flow = SimpleFlow()
    structure = build_flow_structure(flow)

    assert structure is not None
    assert len(structure["nodes"]) == 2
    assert len(structure["edges"]) == 1

    node_names = set(structure["nodes"].keys())
    assert "begin" in node_names
    assert "process" in node_names

    assert len(structure["start_methods"]) == 1
    assert "begin" in structure["start_methods"]

    edge = structure["edges"][0]
    assert edge["source"] == "begin"
    assert edge["target"] == "process"
    assert edge["condition_type"] == "OR"


def test_build_flow_structure_from_flow_class():
    """Test building structure from a Flow class via its FlowDefinition."""
    structure = build_flow_structure(SimpleFlow)

    assert set(structure["nodes"]) == {"begin", "process"}
    assert structure["start_methods"] == ["begin"]
    assert structure["nodes"]["begin"]["class_name"] == "SimpleFlow"


def test_build_flow_structure_from_flow_definition():
    """Test building visualization directly from a FlowDefinition."""
    definition = FlowDefinition.from_declaration(contents=
        {
            "schema": "crewai.flow/v1",
            "name": "DefinedFlow",
            "methods": {
                "begin": {
                    "do": {"ref": "defined_flows:DefinedFlow.begin"},
                    "start": True,
                },
                "decide": {
                    "do": {"ref": "defined_flows:DefinedFlow.decide"},
                    "listen": "begin",
                    "router": True,
                    "emit": ["done"],
                },
                "finish": {
                    "do": {"ref": "defined_flows:DefinedFlow.finish"},
                    "listen": "done",
                },
            },
        }
    )

    structure = build_flow_structure(definition)

    assert set(structure["nodes"]) == {"begin", "decide", "finish"}
    assert structure["start_methods"] == ["begin"]
    assert structure["router_methods"] == ["decide"]
    assert structure["nodes"]["begin"]["class_name"] == "DefinedFlow"
    assert any(
        edge["source"] == "decide"
        and edge["target"] == "finish"
        and edge["router_event"] == "done"
        for edge in structure["edges"]
    )


def test_build_flow_structure_with_router():
    """Test building structure for a flow with router."""
    flow = RouterFlow()
    structure = build_flow_structure(flow)

    assert structure is not None
    assert len(structure["nodes"]) == 4

    assert len(structure["router_methods"]) == 1
    assert "decide" in structure["router_methods"]

    router_node = structure["nodes"]["decide"]
    assert router_node["type"] == "router"
    assert "router_events" not in router_node

    router_edges = [edge for edge in structure["edges"] if edge["is_router_event"]]
    assert router_edges == []


def test_build_flow_structure_with_and_or_conditions():
    """Test building structure for a flow with AND/OR conditions."""
    flow = ComplexFlow()
    structure = build_flow_structure(flow)

    assert structure is not None

    and_edges = [
        edge
        for edge in structure["edges"]
        if edge["target"] == "converge_and" and edge["condition_type"] == "AND"
    ]
    assert len(and_edges) == 2

    or_edges = [
        edge
        for edge in structure["edges"]
        if edge["target"] == "converge_or" and edge["condition_type"] == "OR"
    ]
    assert len(or_edges) == 2


def test_visualize_flow_structure_creates_html():
    """Test that visualization generates valid HTML file."""
    flow = SimpleFlow()
    structure = build_flow_structure(flow)

    html_file = visualize_flow_structure(structure, "test_flow.html", show=False)

    assert os.path.exists(html_file)

    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    assert "<!DOCTYPE html>" in html_content
    assert "<html" in html_content
    assert "CrewAI Flow Visualization" in html_content
    assert "network-container" in html_content
    assert "drawer" in html_content
    assert "nav-controls" in html_content


def test_visualize_flow_structure_creates_assets():
    """Test that visualization creates CSS and JS files."""
    flow = SimpleFlow()
    structure = build_flow_structure(flow)

    html_file = visualize_flow_structure(structure, "test_flow.html", show=False)
    html_path = Path(html_file)

    css_file = html_path.parent / f"{html_path.stem}_style.css"
    js_file = html_path.parent / f"{html_path.stem}_script.js"

    assert css_file.exists()
    assert js_file.exists()

    css_content = css_file.read_text(encoding="utf-8")
    assert len(css_content) > 0
    assert "body" in css_content or ":root" in css_content

    js_content = js_file.read_text(encoding="utf-8")
    assert len(js_content) > 0
    assert "NetworkManager" in js_content


def test_visualize_flow_structure_json_data():
    """Test that visualization includes valid JSON data in JS file."""
    flow = RouterFlow()
    structure = build_flow_structure(flow)

    html_file = visualize_flow_structure(structure, "test_flow.html", show=False)
    html_path = Path(html_file)

    js_file = html_path.parent / f"{html_path.stem}_script.js"

    js_content = js_file.read_text(encoding="utf-8")

    assert "init" in js_content
    assert "decide" in js_content
    assert "handle_a" in js_content
    assert "handle_b" in js_content

    assert "router" in js_content.lower()
    assert "event_a" in js_content
    assert "event_b" in js_content


def test_node_metadata_omits_source_info():
    """Test that definition-only visualization omits Python source metadata."""
    flow = SimpleFlow()
    structure = build_flow_structure(flow)

    for node_metadata in structure["nodes"].values():
        assert "source_code" not in node_metadata
        assert "source_lines" not in node_metadata
        assert "source_start_line" not in node_metadata
        assert "source_file" not in node_metadata


def test_node_metadata_omits_method_signature():
    """Test that definition-only visualization omits Python method signatures."""
    flow = SimpleFlow()
    structure = build_flow_structure(flow)

    begin_node = structure["nodes"]["begin"]
    assert "method_signature" not in begin_node


def test_router_node_has_correct_metadata():
    """Test that router nodes have correct type and event metadata."""
    flow = RouterFlow()
    structure = build_flow_structure(flow)

    router_node = structure["nodes"]["decide"]
    assert router_node["type"] == "router"
    assert router_node["is_router"] is True
    assert "router_events" not in router_node


def test_listen_node_has_trigger_methods():
    """Test that listen nodes include trigger method information."""
    flow = RouterFlow()
    structure = build_flow_structure(flow)

    handle_a_node = structure["nodes"]["handle_a"]
    assert handle_a_node["trigger_methods"] is not None
    assert "event_a" in handle_a_node["trigger_methods"]


def test_and_condition_node_metadata():
    """Test that AND condition nodes have correct metadata."""
    flow = ComplexFlow()
    structure = build_flow_structure(flow)

    converge_and_node = structure["nodes"]["converge_and"]
    assert converge_and_node["condition_type"] == "AND"
    assert converge_and_node["trigger_condition"] is not None
    assert converge_and_node["trigger_condition"]["type"] == "AND"
    assert len(converge_and_node["trigger_condition"]["conditions"]) == 2


def test_visualization_handles_special_characters():
    """Test that visualization properly handles special characters in method names."""

    class SpecialCharFlow(Flow):
        @start()
        def method_with_underscore(self):
            return "test"

        @listen(method_with_underscore)
        def another_method_123(self):
            return "done"

    flow = SpecialCharFlow()
    structure = build_flow_structure(flow)

    assert len(structure["nodes"]) == 2

    json_str = json.dumps(structure)
    assert json_str is not None
    assert "method_with_underscore" in json_str
    assert "another_method_123" in json_str


def test_empty_flow_structure():
    """Test building structure for a flow with no methods."""

    class EmptyFlow(Flow):
        pass

    flow = EmptyFlow()

    structure = build_flow_structure(flow)
    assert structure is not None
    assert len(structure["nodes"]) == 0
    assert len(structure["edges"]) == 0
    assert len(structure["start_methods"]) == 0


def test_topological_path_counting():
    """Test that topological path counting is accurate."""
    flow = ComplexFlow()
    structure = build_flow_structure(flow)

    assert len(structure["nodes"]) > 0
    assert len(structure["edges"]) > 0


def test_class_metadata_comes_from_declaration():
    """Test that nodes include only definition-derived class metadata."""
    flow = SimpleFlow()
    structure = build_flow_structure(flow)

    for node_metadata in structure["nodes"].values():
        assert node_metadata["class_name"] is not None
        assert node_metadata["class_name"] == "SimpleFlow"
        assert "class_signature" not in node_metadata


def test_visualization_plot_method():
    """Test that flow.plot() method works."""
    flow = SimpleFlow()

    html_file = flow.plot("test_plot.html", show=False)

    assert os.path.exists(html_file)


def test_router_events_to_string_conditions():
    """Test that router events correctly connect to listeners with string conditions."""

    class RouterToStringFlow(Flow):
        @start()
        def init(self):
            return "initialized"

        @router(init)
        def decide(self):
            if hasattr(self, "state") and self.state.get("path") == "b":
                return "event_b"
            return "event_a"

        @listen(or_("event_a", "event_b"))
        def handle_either(self):
            return "handled"

        @listen("event_b")
        def handle_b_only(self):
            return "handled_b"

    flow = RouterToStringFlow()
    _attach_flow_definition(
        RouterToStringFlow,
        {
            "init": {"start": True},
            "decide": {"listen": "init", "router": True, "emit": ["event_a", "event_b"]},
            "handle_either": {"listen": {"or": ["event_a", "event_b"]}},
            "handle_b_only": {"listen": "event_b"},
        },
    )
    structure = build_flow_structure(flow)

    decide_node = structure["nodes"]["decide"]
    assert "event_a" in decide_node["router_events"]
    assert "event_b" in decide_node["router_events"]

    router_edges = [edge for edge in structure["edges"] if edge["is_router_event"]]

    assert len(router_edges) == 3

    edges_to_handle_either = [
        edge for edge in router_edges if edge["target"] == "handle_either"
    ]
    assert len(edges_to_handle_either) == 2

    edges_to_handle_b_only = [
        edge for edge in router_edges if edge["target"] == "handle_b_only"
    ]
    assert len(edges_to_handle_b_only) == 1


def test_router_events_not_in_and_conditions():
    """Test that router events don't create edges to AND-nested conditions."""

    class RouterAndConditionFlow(Flow):
        @start()
        def init(self):
            return "initialized"

        @router(init)
        def decide(self):
            return "event_a"

        @listen("event_a")
        def step_1(self):
            return "step_1_done"

        @listen(and_("event_a", step_1))
        def step_2_and(self):
            return "step_2_done"

        @listen(or_(and_("event_a", step_1), "event_a"))
        def step_3_or(self):
            return "step_3_done"

    flow = RouterAndConditionFlow()
    _attach_flow_definition(
        RouterAndConditionFlow,
        {
            "init": {"start": True},
            "decide": {"listen": "init", "router": True, "emit": ["event_a"]},
            "step_1": {"listen": "event_a"},
            "step_2_and": {"listen": {"and": ["event_a", "step_1"]}},
            "step_3_or": {"listen": {"or": [{"and": ["event_a", "step_1"]}, "event_a"]}},
        },
    )
    structure = build_flow_structure(flow)

    router_edges = [edge for edge in structure["edges"] if edge["is_router_event"]]

    targets = [edge["target"] for edge in router_edges]

    assert "step_1" in targets
    assert "step_3_or" in targets
    assert "step_2_and" not in targets


def test_chained_routers_no_self_loops():
    """Test that chained routers don't create self-referencing edges.

    This tests the bug where routers with string triggers (like 'auth', 'exp')
    would incorrectly create edges to themselves when another router outputs
    those strings.
    """

    class ChainedRouterFlow(Flow):
        """Flow with multiple chained routers using string outputs."""

        @start()
        def entrance(self):
            return "started"

        @router(entrance)
        def session_in_cache(self):
            return "exp"

        @router("exp")
        def check_exp(self):
            return "auth"

        @router("auth")
        def call_ai_auth(self):
            return "action"

        @listen("action")
        def forward_to_action(self):
            return "done"

        @listen("authenticate")
        def forward_to_authenticate(self):
            return "need_auth"

    flow = ChainedRouterFlow()
    _attach_flow_definition(
        ChainedRouterFlow,
        {
            "entrance": {"start": True},
            "session_in_cache": {"listen": "entrance", "router": True, "emit": ["exp"]},
            "check_exp": {"listen": "exp", "router": True, "emit": ["auth"]},
            "call_ai_auth": {"listen": "auth", "router": True, "emit": ["action"]},
            "forward_to_action": {"listen": "action"},
            "forward_to_authenticate": {"listen": "authenticate"},
        },
    )
    structure = build_flow_structure(flow)

    for edge in structure["edges"]:
        assert edge["source"] != edge["target"], (
            f"Self-loop detected: {edge['source']} -> {edge['target']}"
        )

    router_edges = [edge for edge in structure["edges"] if edge["is_router_event"]]

    # session_in_cache -> check_exp (via 'exp')
    exp_edges = [
        edge
        for edge in router_edges
        if edge["router_event"] == "exp" and edge["source"] == "session_in_cache"
    ]
    assert len(exp_edges) == 1
    assert exp_edges[0]["target"] == "check_exp"

    # check_exp -> call_ai_auth (via 'auth')
    auth_edges = [
        edge
        for edge in router_edges
        if edge["router_event"] == "auth" and edge["source"] == "check_exp"
    ]
    assert len(auth_edges) == 1
    assert auth_edges[0]["target"] == "call_ai_auth"

    # call_ai_auth -> forward_to_action (via 'action')
    action_edges = [
        edge
        for edge in router_edges
        if edge["router_event"] == "action" and edge["source"] == "call_ai_auth"
    ]
    assert len(action_edges) == 1
    assert action_edges[0]["target"] == "forward_to_action"


def test_routers_with_shared_output_strings():
    """Test that routers with shared output strings don't create incorrect edges.

    This tests a scenario where multiple routers can output the same string,
    ensuring the visualization only creates edges for the router that actually
    outputs the string, not all routers.
    """

    class SharedOutputRouterFlow(Flow):
        """Flow where multiple routers can output 'auth'."""

        @start()
        def start(self):
            return "started"

        @router(start)
        def router_a(self):
            return "auth"

        @router("auth")
        def router_b(self):
            return "done"

        @listen("done")
        def finalize(self):
            return "complete"

        @listen("skip")
        def handle_skip(self):
            return "skipped"

    flow = SharedOutputRouterFlow()
    _attach_flow_definition(
        SharedOutputRouterFlow,
        {
            "start": {"start": True},
            "router_a": {"listen": "start", "router": True, "emit": ["auth"]},
            "router_b": {"listen": "auth", "router": True, "emit": ["done"]},
            "finalize": {"listen": "done"},
            "handle_skip": {"listen": "skip"},
        },
    )
    structure = build_flow_structure(flow)

    for edge in structure["edges"]:
        assert edge["source"] != edge["target"], (
            f"Self-loop detected: {edge['source']} -> {edge['target']}"
        )

    # router_a should connect to router_b via 'auth'
    router_edges = [edge for edge in structure["edges"] if edge["is_router_event"]]
    auth_from_a = [
        edge
        for edge in router_edges
        if edge["source"] == "router_a" and edge["router_event"] == "auth"
    ]
    assert len(auth_from_a) == 1
    assert auth_from_a[0]["target"] == "router_b"

    # router_b should connect to finalize via 'done'
    done_from_b = [
        edge
        for edge in router_edges
        if edge["source"] == "router_b" and edge["router_event"] == "done"
    ]
    assert len(done_from_b) == 1
    assert done_from_b[0]["target"] == "finalize"


def test_warning_for_router_without_events(caplog):
    """Test that a warning is logged when a router has no determinable events."""
    import logging

    class RouterWithoutEventsFlow(Flow):
        """Flow with a router that returns a dynamic value."""

        @start()
        def begin(self):
            return "started"

        @router(begin)
        def dynamic_router(self):
            import random
            return random.choice(["event_a", "event_b"])

        @listen("event_a")
        def handle_a(self):
            return "a"

        @listen("event_b")
        def handle_b(self):
            return "b"

    flow = RouterWithoutEventsFlow()

    with caplog.at_level(logging.WARNING):
        build_flow_structure(flow)

    assert any(
        "Router events for 'dynamic_router' are dynamic" in record.message
        for record in caplog.records
    )

    assert any(
        "Static visualization could not match listener triggers" in record.message
        for record in caplog.records
    )
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


def test_warning_for_orphaned_listeners(caplog):
    """Test that a warning is logged when a trigger has no explicit router output."""
    import logging
    from typing import Literal

    class OrphanedListenerFlow(Flow):
        """Flow where a listener waits for a trigger that no router outputs."""

        @start()
        def begin(self):
            return "started"

        @router(begin)
        def my_router(self) -> Literal["option_a", "option_b"]:
            return "option_a"

        @listen("option_a")
        def handle_a(self):
            return "a"

        @listen("option_c")  # This trigger is never output by any router
        def handle_orphan(self):
            return "orphan"

    flow = OrphanedListenerFlow()
    _attach_flow_definition(
        OrphanedListenerFlow,
        {
            "begin": {"start": True},
            "my_router": {
                "listen": "begin",
                "router": True,
                "emit": ["option_a", "option_b"],
            },
            "handle_a": {"listen": "option_a"},
            "handle_orphan": {"listen": "option_c"},
        },
    )

    with caplog.at_level(logging.WARNING):
        build_flow_structure(flow)

    assert any(
        "Static visualization could not match listener triggers" in record.message
        and "option_c" in record.message
        for record in caplog.records
    )
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


def test_no_warning_for_explicit_contract_router_events(caplog):
    """Test no warning is logged when router events are declared in the contract."""
    import logging
    from typing import Literal

    class ProperlyTypedRouterFlow(Flow):
        """Flow with properly typed router."""

        @start()
        def begin(self):
            return "started"

        @router(begin)
        def typed_router(self) -> Literal["event_a", "event_b"]:
            return "event_a"

        @listen("event_a")
        def handle_a(self):
            return "a"

        @listen("event_b")
        def handle_b(self):
            return "b"

    flow = ProperlyTypedRouterFlow()
    _attach_flow_definition(
        ProperlyTypedRouterFlow,
        {
            "begin": {"start": True},
            "typed_router": {
                "listen": "begin",
                "router": True,
                "emit": ["event_a", "event_b"],
            },
            "handle_a": {"listen": "event_a"},
            "handle_b": {"listen": "event_b"},
        },
    )

    with caplog.at_level(logging.WARNING):
        build_flow_structure(flow)

    # No warnings should be logged
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("Router events for" in msg for msg in warning_messages)
    assert not any(
        "Static visualization could not match listener triggers" in msg
        for msg in warning_messages
    )
