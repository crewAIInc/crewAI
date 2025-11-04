"""Tests for flow visualization and structure building."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from crewai.flow.flow import Flow, and_, listen, or_, router, start
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
            return "path_b"
        return "path_a"

    @listen("path_a")
    def handle_a(self):
        return "handled_a"

    @listen("path_b")
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
        return "final_path"

    @listen("final_path")
    def finalize(self):
        return "complete"


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

    if "router_paths" in router_node:
        assert len(router_node["router_paths"]) >= 1
        assert any("path" in path for path in router_node["router_paths"])

    router_edges = [edge for edge in structure["edges"] if edge["is_router_path"]]
    assert len(router_edges) >= 1


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
    assert "path_a" in js_content
    assert "path_b" in js_content


def test_node_metadata_includes_source_info():
    """Test that nodes include source code and line number information."""
    flow = SimpleFlow()
    structure = build_flow_structure(flow)

    for node_name, node_metadata in structure["nodes"].items():
        assert node_metadata["source_code"] is not None
        assert len(node_metadata["source_code"]) > 0
        assert node_metadata["source_start_line"] is not None
        assert node_metadata["source_start_line"] > 0
        assert node_metadata["source_file"] is not None
        assert node_metadata["source_file"].endswith(".py")


def test_node_metadata_includes_method_signature():
    """Test that nodes include method signature information."""
    flow = SimpleFlow()
    structure = build_flow_structure(flow)

    begin_node = structure["nodes"]["begin"]
    assert begin_node["method_signature"] is not None
    assert "operationId" in begin_node["method_signature"]
    assert begin_node["method_signature"]["operationId"] == "begin"
    assert "parameters" in begin_node["method_signature"]
    assert "returns" in begin_node["method_signature"]


def test_router_node_has_correct_metadata():
    """Test that router nodes have correct type and paths."""
    flow = RouterFlow()
    structure = build_flow_structure(flow)

    router_node = structure["nodes"]["decide"]
    assert router_node["type"] == "router"
    assert router_node["is_router"] is True
    assert router_node["router_paths"] is not None
    assert len(router_node["router_paths"]) == 2
    assert "path_a" in router_node["router_paths"]
    assert "path_b" in router_node["router_paths"]


def test_listen_node_has_trigger_methods():
    """Test that listen nodes include trigger method information."""
    flow = RouterFlow()
    structure = build_flow_structure(flow)

    handle_a_node = structure["nodes"]["handle_a"]
    assert handle_a_node["trigger_methods"] is not None
    assert "path_a" in handle_a_node["trigger_methods"]


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


def test_class_signature_metadata():
    """Test that nodes include class signature information."""
    flow = SimpleFlow()
    structure = build_flow_structure(flow)

    for node_name, node_metadata in structure["nodes"].items():
        assert node_metadata["class_name"] is not None
        assert node_metadata["class_name"] == "SimpleFlow"
        assert node_metadata["class_signature"] is not None
        assert "SimpleFlow" in node_metadata["class_signature"]


def test_visualization_plot_method():
    """Test that flow.plot() method works."""
    flow = SimpleFlow()

    html_file = flow.plot("test_plot.html", show=False)

    assert os.path.exists(html_file)


def test_router_paths_to_string_conditions():
    """Test that router paths correctly connect to listeners with string conditions."""

    class RouterToStringFlow(Flow):
        @start()
        def init(self):
            return "initialized"

        @router(init)
        def decide(self):
            if hasattr(self, "state") and self.state.get("path") == "b":
                return "path_b"
            return "path_a"

        @listen(or_("path_a", "path_b"))
        def handle_either(self):
            return "handled"

        @listen("path_b")
        def handle_b_only(self):
            return "handled_b"

    flow = RouterToStringFlow()
    structure = build_flow_structure(flow)

    decide_node = structure["nodes"]["decide"]
    assert "path_a" in decide_node["router_paths"]
    assert "path_b" in decide_node["router_paths"]

    router_edges = [edge for edge in structure["edges"] if edge["is_router_path"]]

    assert len(router_edges) == 3

    edges_to_handle_either = [
        edge for edge in router_edges if edge["target"] == "handle_either"
    ]
    assert len(edges_to_handle_either) == 2

    edges_to_handle_b_only = [
        edge for edge in router_edges if edge["target"] == "handle_b_only"
    ]
    assert len(edges_to_handle_b_only) == 1


def test_router_paths_not_in_and_conditions():
    """Test that router paths don't create edges to AND-nested conditions."""

    class RouterAndConditionFlow(Flow):
        @start()
        def init(self):
            return "initialized"

        @router(init)
        def decide(self):
            return "path_a"

        @listen("path_a")
        def step_1(self):
            return "step_1_done"

        @listen(and_("path_a", step_1))
        def step_2_and(self):
            return "step_2_done"

        @listen(or_(and_("path_a", step_1), "path_a"))
        def step_3_or(self):
            return "step_3_done"

    flow = RouterAndConditionFlow()
    structure = build_flow_structure(flow)

    router_edges = [edge for edge in structure["edges"] if edge["is_router_path"]]

    targets = [edge["target"] for edge in router_edges]

    assert "step_1" in targets
    assert "step_3_or" in targets
    assert "step_2_and" not in targets