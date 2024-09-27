import shutil
import warnings
from abc import ABC, abstractmethod

from pyvis.network import Network


class FlowVisualizer(ABC):
    def __init__(self, flow):
        self.flow = flow
        self.colors = {
            "bg": "#FFFFFF",
            "start": "#FF5A50",
            "method": "#333333",
            "router_outline": "#FF5A50",
            "edge": "#333333",
            "text": "#FFFFFF",
        }
        self.node_rectangles = {}
        self.node_positions = {}

    @abstractmethod
    def visualize(self, filename):
        pass


class GraphvizVisualizer(FlowVisualizer):
    def visualize(self, filename):
        import graphviz

        dot = graphviz.Digraph(comment="Flow Graph", engine="dot")
        dot.attr(rankdir="TB", size="20,20", splines="curved")
        dot.attr(bgcolor=self.colors["bg"])

        # Add nodes
        for method_name, method in self.flow._methods.items():
            if (
                hasattr(method, "__is_start_method__")
                or method_name in self.flow._listeners
                or method_name in self.flow._routers.values()
            ):
                shape = "rectangle"
                style = "filled,rounded"
                fillcolor = (
                    self.colors["start"]
                    if hasattr(method, "__is_start_method__")
                    else self.colors["method"]
                )

                dot.node(
                    method_name,
                    method_name,
                    shape=shape,
                    style=style,
                    fillcolor=fillcolor,
                    fontcolor=self.colors["text"],
                    penwidth="2",
                )

        # Add edges and routers
        for method_name, method in self.flow._methods.items():
            if method_name in self.flow._listeners:
                condition_type, trigger_methods = self.flow._listeners[method_name]
                for trigger in trigger_methods:
                    style = "dashed" if condition_type == "AND" else "solid"
                    dot.edge(
                        trigger,
                        method_name,
                        color=self.colors["edge"],
                        style=style,
                        penwidth="2",
                    )

            if method_name in self.flow._routers.values():
                for trigger, router in self.flow._routers.items():
                    if router == method_name:
                        subgraph_name = f"cluster_{method_name}"
                        subgraph = graphviz.Digraph(name=subgraph_name)
                        subgraph.attr(
                            label="",
                            style="filled,rounded",
                            color=self.colors["router_outline"],
                            fillcolor=self.colors["method"],
                            penwidth="3",
                        )
                        label = f"{method_name}\\n\\nPossible outcomes:\\n• Success\\n• Failure"
                        subgraph.node(
                            method_name,
                            label,
                            shape="plaintext",
                            fontcolor=self.colors["text"],
                        )
                        dot.subgraph(subgraph)
                        dot.edge(
                            trigger,
                            method_name,
                            color=self.colors["edge"],
                            style="solid",
                            penwidth="2",
                            lhead=subgraph_name,
                        )

        dot.render(filename, format="png", cleanup=True, view=True)
        print(f"Graph saved as {filename}.png")


class PyvisFlowVisualizer(FlowVisualizer):
    def visualize(self, filename):
        net = Network(
            directed=True, height="750px", width="100%", bgcolor=self.colors["bg"]
        )

        # Define custom node styles
        node_styles = {
            "start": {
                "color": self.colors["start"],
                "shape": "box",
                "font": {"color": self.colors["text"]},
            },
            "method": {
                "color": self.colors["method"],
                "shape": "box",
                "font": {"color": self.colors["text"]},
            },
            # "router": {
            #     "color": self.colors["router"],
            #     "shape": "box",
            #     "font": {"color": self.colors["text"]},
            # },
        }

        # Add nodes
        for method_name, method in self.flow._methods.items():
            if (
                hasattr(method, "__is_start_method__")
                or method_name in self.flow._listeners
                or method_name in self.flow._routers.values()
            ):
                if hasattr(method, "__is_start_method__"):
                    node_style = node_styles["start"]
                elif method_name in self.flow._routers.values():
                    node_style = node_styles["router"]
                else:
                    node_style = node_styles["method"]

                net.add_node(method_name, label=method_name, **node_style)

        # Add edges
        for method_name in self.flow._listeners:
            condition_type, trigger_methods = self.flow._listeners[method_name]
            is_and_condition = condition_type == "AND"
            for trigger in trigger_methods:
                if trigger in self.flow._methods:
                    net.add_edge(
                        trigger,
                        method_name,
                        color=self.colors["edge"],
                        width=2,
                        arrows="to",
                        dashes=is_and_condition,  # Dashed lines for AND conditions
                        smooth={"type": "cubicBezier"},
                    )

        # Generate and save the graph
        net.write_html(f"{filename}.html")
        print(f"Graph saved as {filename}.html")


def is_graphviz_available():
    try:
        import graphviz

        if shutil.which("dot") is None:  # Check for Graphviz executable
            raise ImportError("Graphviz executable not found")
        return True
    except ImportError:
        return False


def visualize_flow(flow, filename="flow_graph"):
    if False:
        visualizer = GraphvizVisualizer(flow)
    else:
        warnings.warn(
            "Graphviz is not available. Falling back to NetworkX and Matplotlib for visualization. "
            "For better visualization, please install Graphviz. "
            "See our documentation for installation instructions: https://docs.crewai.com/advanced-usage/visualization/"
        )
        visualizer = PyvisFlowVisualizer(flow)

    visualizer.visualize(filename)
