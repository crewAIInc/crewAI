"""Test that the `orientation` parameter in `flow_visualizer` is applied correctly."""

import os
import pytest

from crewai.flow.flow import Flow, listen, start
from crewai.flow.flow_visualizer import FlowPlot, plot_flow
from crewai.flow.visualization_utils import compute_positions


class DummyFlow(Flow):
    """Minimal Flow subclass for testing orientation logic."""

    @start()
    def first_action(self):
        return "First action"

    @listen(first_action)
    def second_action(self):
        return "Second action"


def test_compute_positions_horizontal():
    """Compute positions should swap x/y spacing and center nodes for 'horizontal'."""
    levels = {"n1": 0, "n2": 1, "n3": 1}
    pos = compute_positions(
        flow=None,
        node_levels=levels,
        x_spacing=100,
        y_spacing=200,
        orientation="horizontal",
    )
    # Levels 0 → y = 0 * x_spacing = 0; level 1 → y = 1 * x_spacing = 100
    assert pos["n1"][1] == 0
    assert pos["n2"][1] == 100
    assert pos["n3"][1] == 100
    # Two nodes at level 1 should be centered: x = ±100
    xs = sorted([pos["n2"][0], pos["n3"][0]])
    assert xs == [-100, 100]


def test_compute_positions_vertical():
    """Compute positions should use (y, x) and center nodes for 'vertical'."""
    levels = {"n1": 0, "n2": 1, "n3": 1}
    pos = compute_positions(
        flow=None,
        node_levels=levels,
        x_spacing=50,
        y_spacing=75,
        orientation="vertical",
    )
    # n1 @ level 0 → y=0, single node → x=0
    assert pos["n1"] == (0, 0)
    # level 1 → y=75; two nodes centered: offset = -25 → x = -25, +25
    assert pos["n2"][0] == 75 and pos["n3"][0] == 75
    xs = sorted([pos["n2"][1], pos["n3"][1]])
    assert xs == [-25, 25]


def test_compute_positions_invalid_orientation():
    """Compute positions should raise ValueError for unsupported orientation."""
    with pytest.raises(ValueError) as exc:
        compute_positions(flow=None, node_levels={}, orientation="diagonal")
    assert "Invalid `orientation` value" in str(exc.value)


def test_plot_flow_calls_with_orientation(monkeypatch, tmp_path):
    """
    FlowPlot.plot should propagate orientation to compute_positions,
    add_nodes_to_network and add_edges, then write the final HTML.
    """
    dummy = DummyFlow()
    fp = FlowPlot(dummy)
    calls = {}

    # Mock calculate_node_levels → fixed levels
    monkeypatch.setattr(
        "crewai.flow.flow_visualizer.calculate_node_levels",
        lambda f: {"a": 0, "b": 1},
    )

    # Capture orientation in compute_positions
    def fake_compute(flow, levels, orientation):
        calls["orientation"] = orientation
        return {"a": (0, 0), "b": (0, 100)}

    monkeypatch.setattr(
        "crewai.flow.flow_visualizer.compute_positions",
        fake_compute,
    )
    # Record that nodes and edges were added with correct orientation
    monkeypatch.setattr(
        "crewai.flow.flow_visualizer.add_nodes_to_network",
        lambda net, flow, pos, styles: calls.update(add_nodes=True),
    )
    monkeypatch.setattr(
        "crewai.flow.flow_visualizer.add_edges",
        lambda net, flow, pos, colors, orientation: calls.update(
            add_edges_orientation=orientation
        ),
    )

    # Fake pyvis Network and HTMLTemplateHandler
    class FakeNet:
        def __init__(self, **kwargs):
            pass

        def set_options(self, _):
            pass

        def generate_html(self):
            return "<html></html>"

    monkeypatch.setattr(
        "crewai.flow.flow_visualizer.Network",
        FakeNet,
    )
    monkeypatch.setattr(
        "crewai.flow.flow_visualizer.HTMLTemplateHandler",
        lambda tpl, logo: type(
            "H",
            (),
            {
                "extract_body_content": lambda self, h: "<body/>",
                "generate_final_html": lambda self, body, logo: "<final/>",
            },
        )(),
    )

    # Run and verify
    os.chdir(tmp_path)
    fp.plot(filename="test", orientation="vertical")
    assert calls["orientation"] == "vertical"
    assert calls["add_nodes"] is True
    assert calls["add_edges_orientation"] == "vertical"
    assert (tmp_path / "test.html").read_text(encoding="utf-8").find("<final/>") != -1


def test_plot_flow_invalid_orientation_raises(tmp_path):
    """plot_flow wrapper should raise ValueError for bad orientation."""
    dummy = DummyFlow()
    with pytest.raises(ValueError):
        plot_flow(dummy, filename="x", orientation="sideways")
