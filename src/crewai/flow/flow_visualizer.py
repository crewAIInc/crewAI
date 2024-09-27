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
            "router": "#FF8C00",
            "edge": "#666666",
            "text": "#FFFFFF",
        }

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
            directed=True,
            height="750px",
            width="100%",
            bgcolor=self.colors["bg"],
            layout=None,
        )

        # Define custom node styles
        node_styles = {
            "start": {
                "color": self.colors.get("start", "#FF5A50"),
                "shape": "box",
                "font": {"color": self.colors.get("text", "#FFFFFF")},
            },
            "method": {
                "color": self.colors.get("method", "#333333"),
                "shape": "box",
                "font": {"color": self.colors.get("text", "#FFFFFF")},
            },
            "router": {
                "color": self.colors.get("router", "#FF8C00"),
                "shape": "box",
                "font": {"color": self.colors.get("text", "#FFFFFF")},
            },
        }

        # Calculate levels for nodes
        node_levels = self._calculate_node_levels()

        # Assign positions to nodes based on levels
        y_spacing = 150  # Adjust spacing between levels (positive for top-down)
        x_spacing = 150  # Adjust spacing between nodes
        level_nodes = {}

        for method_name, level in node_levels.items():
            level_nodes.setdefault(level, []).append(method_name)

        # Compute positions
        for level, nodes in level_nodes.items():
            x_offset = -(len(nodes) - 1) * x_spacing / 2  # Center nodes horizontally
            for i, method_name in enumerate(nodes):
                x = x_offset + i * x_spacing
                y = level * y_spacing  # Use level directly for y position
                method = self.flow._methods.get(method_name)
                if hasattr(method, "__is_start_method__"):
                    node_style = node_styles["start"]
                elif method_name in self.flow._routers.values():
                    node_style = node_styles["router"]
                else:
                    node_style = node_styles["method"]

                net.add_node(
                    method_name,
                    label=method_name,
                    x=x,
                    y=y,
                    fixed=True,
                    physics=False,  # Disable physics for fixed positioning
                    **node_style,
                )

        # Add edges with curved lines
        for method_name in self.flow._listeners:
            condition_type, trigger_methods = self.flow._listeners[method_name]
            is_and_condition = condition_type == "AND"
            for trigger in trigger_methods:
                if trigger in self.flow._methods:
                    net.add_edge(
                        trigger,
                        method_name,
                        color=self.colors.get("edge", "#666666"),
                        width=2,
                        arrows="to",
                        dashes=is_and_condition,
                        smooth={"type": "cubicBezier"},
                    )

        # Set options for curved edges and disable physics
        net.set_options(
            """
    var options = {
      "physics": {
        "enabled": false
      },
      "edges": {
        "smooth": {
          "enabled": true,
          "type": "cubicBezier",
          "roundness": 0.5
        }
      }
    }
    """
        )

        # Generate and save the graph
        net.write_html(f"{filename}.html")
        print(f"Graph saved as {filename}.html")

    def _calculate_node_levels(self):
        levels = {}
        queue = []
        visited = set()

        # Initialize start methods at level 0
        for method_name, method in self.flow._methods.items():
            if hasattr(method, "__is_start_method__"):
                levels[method_name] = 0
                queue.append(method_name)

        # Breadth-first traversal to assign levels
        while queue:
            current = queue.pop(0)
            current_level = levels[current]
            visited.add(current)

            # Get methods that listen to the current method
            for listener_name, (
                condition_type,
                trigger_methods,
            ) in self.flow._listeners.items():
                if current in trigger_methods:
                    if (
                        listener_name not in levels
                        or levels[listener_name] > current_level + 1
                    ):
                        levels[listener_name] = current_level + 1
                        if listener_name not in visited:
                            queue.append(listener_name)

        return levels


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
