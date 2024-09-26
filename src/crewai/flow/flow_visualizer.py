import shutil
import warnings
from abc import ABC, abstractmethod

import requests
from IPython.display import Image, display


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


class MermaidFlowVisualizer(FlowVisualizer):
    def visualize(self, filename):
        mermaid_code = self.generate_mermaid_code()

        # Use Mermaid.ink API to generate the diagram
        response = requests.post(
            "https://mermaid.ink/img/",
            data=mermaid_code,
            headers={"Content-Type": "text/plain"},
        )

        if response.status_code == 200:
            image_url = response.url
            print(f"Graph available at {image_url}")

            # Optionally, download the image and save it locally
            image_data = requests.get(image_url).content
            with open(f"{filename}.png", "wb") as f:
                f.write(image_data)
            print(f"Graph saved as {filename}.png")

            # Display the image in Jupyter notebook
            display(Image(image_url))
        else:
            print(f"Failed to generate graph: {response.status_code} {response.text}")

    def generate_mermaid_code(self):
        mermaid_code = ["graph TB"]

        # Add nodes
        for method_name, method in self.flow._methods.items():
            if (
                hasattr(method, "__is_start_method__")
                or method_name in self.flow._listeners
                or method_name in self.flow._routers.values()
            ):
                shape = '((" "))' if hasattr(method, "__is_start_method__") else '[" "]'
                color = (
                    self.colors["start"]
                    if hasattr(method, "__is_start_method__")
                    else self.colors["method"]
                )
                mermaid_code.append(
                    f'    {method_name}{shape}:::{"startNode" if hasattr(method, "__is_start_method__") else "methodNode"}'
                )
                mermaid_code.append(
                    f'    style {method_name} fill:{color},color:{self.colors["text"]}'
                )

        # Add edges
        for method_name, method in self.flow._methods.items():
            if method_name in self.flow._listeners:
                condition_type, trigger_methods = self.flow._listeners[method_name]
                for trigger in trigger_methods:
                    edge_style = " -.- " if condition_type == "AND" else " --> "
                    mermaid_code.append(f"    {trigger}{edge_style}{method_name}")

        # Add styles
        mermaid_code.extend(
            [
                "    classDef startNode stroke:#333,stroke-width:4px;",
                "    classDef methodNode stroke:#333,stroke-width:2px;",
            ]
        )

        return "\n".join(mermaid_code)


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
        visualizer = MermaidFlowVisualizer(flow)

    visualizer.visualize(filename)
