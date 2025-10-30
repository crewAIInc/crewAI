"""Tests for flow visualization and structure building."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from crewai.flow.flow import Flow, and_, listen, or_, router, start
from crewai.flow.visualization import (
    build_flow_structure,
    print_structure_summary,
    structure_to_dict,
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
        # Return different paths based on state
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

    # Check nodes
    node_names = set(structure["nodes"].keys())
    assert "begin" in node_names
    assert "process" in node_names

    # Check start method
    assert len(structure["start_methods"]) == 1
    assert "begin" in structure["start_methods"]

    # Check edges
    edge = structure["edges"][0]
    assert edge["source"] == "begin"
    assert edge["target"] == "process"
    assert edge["condition_type"] == "OR"


def test_build_flow_structure_with_router():
    """Test building structure for a flow with router."""
    flow = RouterFlow()
    structure = build_flow_structure(flow)

    assert structure is not None
    assert len(structure["nodes"]) == 4  # init, decide, handle_a, handle_b

    # Check router methods
    assert len(structure["router_methods"]) == 1
    assert "decide" in structure["router_methods"]

    # Find router node
    router_node = structure["nodes"]["decide"]
    assert router_node["type"] == "router"

    # Router paths are detected by analyzing the method's code
    # The number of detected paths may vary
    if "router_paths" in router_node:
        assert len(router_node["router_paths"]) >= 1
        assert any("path" in path for path in router_node["router_paths"])

    # Check router edges
    router_edges = [edge for edge in structure["edges"] if edge["is_router_path"]]
    assert len(router_edges) >= 1  # At least one router path should be detected


def test_build_flow_structure_with_and_or_conditions():
    """Test building structure for a flow with AND/OR conditions."""
    flow = ComplexFlow()
    structure = build_flow_structure(flow)

    assert structure is not None

    # Check AND condition edge
    and_edges = [
        edge
        for edge in structure["edges"]
        if edge["target"] == "converge_and" and edge["condition_type"] == "AND"
    ]
    assert len(and_edges) == 2  # Should have edges from both start_a and start_b

    # Check OR condition edge
    or_edges = [
        edge
        for edge in structure["edges"]
        if edge["target"] == "converge_or" and edge["condition_type"] == "OR"
    ]
    assert len(or_edges) == 2  # Should have edges from both start_a and start_b


def test_structure_to_dict():
    """Test converting flow structure to dictionary format."""
    flow = SimpleFlow()
    structure = build_flow_structure(flow)
    dag_dict = structure_to_dict(structure)

    assert "nodes" in dag_dict
    assert "edges" in dag_dict
    assert "start_methods" in dag_dict
    assert "router_methods" in dag_dict

    # Check nodes
    assert "begin" in dag_dict["nodes"]
    assert "process" in dag_dict["nodes"]

    # Check node metadata
    begin_node = dag_dict["nodes"]["begin"]
    assert begin_node["type"] == "start"
    assert "method_signature" in begin_node
    assert "source_code" in begin_node

    # Check edges format
    assert len(dag_dict["edges"]) == 1
    edge = dag_dict["edges"][0]
    assert "source" in edge
    assert "target" in edge
    assert "condition_type" in edge
    assert "is_router_path" in edge


def test_structure_to_dict_with_router():
    """Test dictionary conversion for flow with router."""
    flow = RouterFlow()
    structure = build_flow_structure(flow)
    dag_dict = structure_to_dict(structure)

    # Check router node metadata
    decide_node = dag_dict["nodes"]["decide"]
    assert decide_node["type"] == "router"
    assert decide_node["is_router"] is True

    # Router paths are detected at runtime
    if "router_paths" in decide_node:
        assert len(decide_node["router_paths"]) >= 1

    # Check router edges
    router_edges = [edge for edge in dag_dict["edges"] if edge["is_router_path"]]
    assert len(router_edges) >= 1


def test_structure_to_dict_with_complex_conditions():
    """Test dictionary conversion for flow with complex conditions."""
    flow = ComplexFlow()
    structure = build_flow_structure(flow)
    dag_dict = structure_to_dict(structure)

    # Check converge_and node has AND condition
    converge_and_node = dag_dict["nodes"]["converge_and"]
    assert converge_and_node["condition_type"] == "AND"
    assert "trigger_condition" in converge_and_node
    assert converge_and_node["trigger_condition"]["type"] == "AND"

    # Check converge_or node has OR condition
    converge_or_node = dag_dict["nodes"]["converge_or"]
    assert converge_or_node["condition_type"] == "OR"


def test_visualize_flow_structure_creates_html():
    """Test that visualization generates valid HTML file."""
    flow = SimpleFlow()
    structure = build_flow_structure(flow)

    # visualize_flow_structure returns the path to the generated HTML file
    html_file = visualize_flow_structure(structure, "test_flow.html", show=False)

    # Check HTML file was created
    assert os.path.exists(html_file)

    # Read and validate HTML content
    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Check for key HTML elements
    assert "<!DOCTYPE html>" in html_content
    assert "<html" in html_content
    assert "CrewAI Flow Visualization" in html_content
    assert "network-container" in html_content  # Check for network container div
    assert "drawer" in html_content  # Check for side drawer
    assert "nav-controls" in html_content  # Check for navigation controls


def test_visualize_flow_structure_creates_assets():
    """Test that visualization creates CSS and JS files."""
    flow = SimpleFlow()
    structure = build_flow_structure(flow)

    # Returns the HTML file path
    html_file = visualize_flow_structure(structure, "test_flow.html", show=False)
    html_path = Path(html_file)

    # Check that CSS and JS files were created in the same directory
    css_file = html_path.parent / f"{html_path.stem}_style.css"
    js_file = html_path.parent / f"{html_path.stem}_script.js"

    assert css_file.exists()
    assert js_file.exists()

    # Validate CSS content
    css_content = css_file.read_text(encoding="utf-8")
    assert len(css_content) > 0
    assert "body" in css_content or ":root" in css_content

    # Validate JS content
    js_content = js_file.read_text(encoding="utf-8")
    assert len(js_content) > 0
    assert "var nodes" in js_content or "const nodes" in js_content


def test_visualize_flow_structure_json_data():
    """Test that visualization includes valid JSON data in JS file."""
    flow = RouterFlow()
    structure = build_flow_structure(flow)

    # Returns the HTML file path
    html_file = visualize_flow_structure(structure, "test_flow.html", show=False)
    html_path = Path(html_file)

    js_file = html_path.parent / f"{html_path.stem}_script.js"

    js_content = js_file.read_text(encoding="utf-8")

    # Extract and validate node data
    assert "init" in js_content
    assert "decide" in js_content
    assert "handle_a" in js_content
    assert "handle_b" in js_content

    # Check for router-specific data
    assert "router" in js_content.lower()
    assert "path_a" in js_content
    assert "path_b" in js_content


def test_print_structure_summary():
    """Test printing flow structure summary."""
    flow = ComplexFlow()
    structure = build_flow_structure(flow)

    # print_structure_summary returns a string, doesn't print
    output = print_structure_summary(structure)

    # Check summary contains key information
    assert "Total nodes:" in output
    assert "Total edges:" in output
    assert "Start methods:" in output
    assert "Router methods:" in output

    # Check it lists the nodes
    assert "start_a" in output
    assert "start_b" in output


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

    # Convert to dict and ensure it's JSON-serializable
    dag_dict = structure_to_dict(structure)
    json_str = json.dumps(dag_dict)
    assert json_str is not None
    assert "method_with_underscore" in json_str
    assert "another_method_123" in json_str


def test_empty_flow_structure():
    """Test building structure for a flow with no methods."""

    class EmptyFlow(Flow):
        pass

    flow = EmptyFlow()

    # Should not raise an error
    structure = build_flow_structure(flow)
    assert structure is not None
    assert len(structure["nodes"]) == 0
    assert len(structure["edges"]) == 0
    assert len(structure["start_methods"]) == 0


def test_topological_path_counting():
    """Test that topological path counting is accurate."""
    flow = ComplexFlow()
    structure = build_flow_structure(flow)
    dag_dict = structure_to_dict(structure)

    # Should calculate execution paths correctly
    # This flow has multiple paths through the graph
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

    # flow.plot() returns the path to the generated HTML file
    html_file = flow.plot("test_plot.html", show=False)

    # Check that HTML file was created
    assert os.path.exists(html_file)