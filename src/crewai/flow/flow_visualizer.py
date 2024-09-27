import math
import shutil
import warnings
from abc import ABC, abstractmethod

from PIL import Image, ImageDraw, ImageFont


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


class PyvisFlowVisualizer:
    def __init__(self, flow):
        self.flow = flow
        self.colors = {
            "bg": "#FFFFFF",
            "start": "#FF5A50",
            "method": "#333333",
            "router": "#FF8C00",  # Orange color for routers
            "edge": "#666666",
            "text": "#FFFFFF",
        }

    def visualize(self, filename):
        # Get decorated methods
        start_methods = [
            name
            for name, method in self.flow._methods.items()
            if hasattr(method, "__is_start_method__")
        ]
        listen_methods = list(self.flow._listeners.keys())
        router_methods = list(self.flow._routers.values())

        all_methods = start_methods + listen_methods + router_methods
        node_positions = self._calculate_positions(all_methods)

        # Create image
        img_width = 800
        img_height = len(all_methods) * 120 + 100
        img = Image.new("RGB", (img_width, img_height), color=self.colors["bg"])
        draw = ImageDraw.Draw(img)

        # Draw edges
        for method_name in listen_methods + router_methods:
            if method_name in self.flow._listeners:
                _, trigger_methods = self.flow._listeners[method_name]
                for trigger in trigger_methods:
                    if trigger in node_positions and method_name in node_positions:
                        start = node_positions[trigger]
                        end = node_positions[method_name]
                        self._draw_curved_arrow(draw, start, end, self.colors["edge"])

        # Draw nodes
        for method_name, pos in node_positions.items():
            if method_name in start_methods:
                color = self.colors["start"]
            elif method_name in router_methods:
                color = self.colors["router"]
            else:
                color = self.colors["method"]

            self._draw_node(draw, method_name, pos, color)

        # Save image
        img.save(f"{filename}.png")
        print(f"Graph saved as {filename}.png")

    def _calculate_positions(self, nodes):
        positions = {}
        start_methods = [
            node
            for node in nodes
            if hasattr(self.flow._methods[node], "__is_start_method__")
        ]
        other_methods = [node for node in nodes if node not in start_methods]

        # Position start methods at the top
        for i, node in enumerate(start_methods):
            positions[node] = (400, 100 + i * 120)

        # Position other methods below start methods
        for i, node in enumerate(other_methods):
            positions[node] = (400, 100 + (len(start_methods) + i) * 120)

        return positions

    def _draw_node(self, draw, label, position, color):
        x, y = position
        if color == self.colors["router"]:
            # Draw router node as rounded rectangle
            draw.rounded_rectangle(
                [x - 70, y - 40, x + 70, y + 40],
                radius=10,
                fill=color,
                outline=self.colors["edge"],
            )
            font = ImageFont.load_default()
            text_width = draw.textlength(label, font=font)
            draw.text(
                (x - text_width / 2, y - 20), label, fill=self.colors["text"], font=font
            )
            draw.text((x - 30, y + 5), "Success", fill=self.colors["text"], font=font)
            draw.text((x - 30, y + 25), "Failure", fill=self.colors["text"], font=font)
        else:
            # Draw regular node
            draw.rectangle(
                [x - 60, y - 30, x + 60, y + 30],
                fill=color,
                outline=self.colors["edge"],
            )
            font = ImageFont.load_default()
            text_width = draw.textlength(label, font=font)
            draw.text(
                (x - text_width / 2, y - 7), label, fill=self.colors["text"], font=font
            )

    def _draw_curved_arrow(self, draw, start, end, color):
        # Calculate control point for the curve
        control_x = (start[0] + end[0]) / 2
        control_y = (
            start[1] + end[1]
        ) / 2 - 50  # Adjust this value to change curve height

        # Draw the curved line
        points = [start, (control_x, control_y), end]
        draw.line(points, fill=color, width=2, joint="curve")

        # Draw arrow head
        self._draw_arrow_head(draw, points[-2], end, color)

    def _draw_arrow_head(self, draw, start, end, color):
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        x = end[0] - 15 * math.cos(angle)
        y = end[1] - 15 * math.sin(angle)
        draw.polygon(
            [
                (x, y),
                (
                    x - 10 * math.cos(angle - math.pi / 6),
                    y - 10 * math.sin(angle - math.pi / 6),
                ),
                (
                    x - 10 * math.cos(angle + math.pi / 6),
                    y - 10 * math.sin(angle + math.pi / 6),
                ),
            ],
            fill=color,
        )


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
