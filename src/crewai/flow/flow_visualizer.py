# flow_visualizer.py

import base64
import os
import re
from abc import ABC, abstractmethod

from pyvis.network import Network

DARK_GRAY = "#333333"
CREWAI_ORANGE = "#FF5A50"
GRAY = "#666666"
WHITE = "#FFFFFF"


class FlowVisualizer(ABC):
    def __init__(self, flow):
        self.flow = flow
        self.colors = {
            "bg": WHITE,
            "start": CREWAI_ORANGE,
            "method": DARK_GRAY,
            "router": DARK_GRAY,
            "router_border": CREWAI_ORANGE,
            "edge": GRAY,
            "router_edge": CREWAI_ORANGE,
            "text": WHITE,
        }
        self.node_styles = {
            "start": {
                "color": self.colors["start"],
                "shape": "box",
                "font": {"color": self.colors["text"]},
                "margin": 15,
            },
            "method": {
                "color": self.colors["method"],
                "shape": "box",
                "font": {"color": self.colors["text"]},
                "margin": 15,
            },
            "router": {
                "color": {
                    "background": self.colors["router"],
                    "border": self.colors["router_border"],
                    "highlight": {
                        "border": self.colors["router_border"],
                        "background": self.colors["router"],
                    },
                },
                "shape": "box",
                "font": {"color": self.colors["text"]},
                "borderWidth": 3,
                "borderWidthSelected": 4,
                "shapeProperties": {"borderDashes": [5, 5]},
                "margin": 15,
            },
        }

    @abstractmethod
    def visualize(self, filename):
        pass


class PyvisFlowVisualizer(FlowVisualizer):
    def visualize(self, filename):
        net = Network(
            directed=True,
            height="750px",
            width="100%",
            bgcolor=self.colors["bg"],
            layout=None,
        )

        # Calculate levels for nodes
        node_levels = self._calculate_node_levels()
        print("node_levels", node_levels)

        # Assign positions to nodes based on levels
        y_spacing = 150
        x_spacing = 150
        level_nodes = {}

        # Store node positions for edge calculations
        node_positions = {}

        for method_name, level in node_levels.items():
            level_nodes.setdefault(level, []).append(method_name)

        # Compute positions
        for level, nodes in level_nodes.items():
            x_offset = -(len(nodes) - 1) * x_spacing / 2  # Center nodes horizontally
            for i, method_name in enumerate(nodes):
                x = x_offset + i * x_spacing
                y = level * y_spacing
                node_positions[method_name] = (x, y)

                method = self.flow._methods.get(method_name)
                if hasattr(method, "__is_start_method__"):
                    node_style = self.node_styles["start"]
                elif hasattr(method, "__is_router__"):
                    node_style = self.node_styles["router"]
                else:
                    node_style = self.node_styles["method"]

                net.add_node(
                    method_name,
                    label=method_name,
                    x=x,
                    y=y,
                    fixed=True,
                    physics=False,
                    **node_style,
                )

        ancestors = self._build_ancestor_dict()
        parent_children = self._build_parent_children_dict()

        # Add edges
        for method_name in self.flow._listeners:
            condition_type, trigger_methods = self.flow._listeners[method_name]
            is_and_condition = condition_type == "AND"

            for trigger in trigger_methods:
                if (
                    trigger in self.flow._methods
                    or trigger in self.flow._routers.values()
                ):
                    is_router_edge = any(
                        trigger in paths for paths in self.flow._router_paths.values()
                    )
                    edge_color = (
                        self.colors["router_edge"]
                        if is_router_edge
                        else self.colors["edge"]
                    )

                    # Determine if this edge forms a cycle
                    is_cycle_edge = self._is_ancestor(trigger, method_name, ancestors)

                    # Determine if parent has multiple children
                    parent_has_multiple_children = (
                        len(parent_children.get(trigger, [])) > 1
                    )

                    # Edge curvature logic
                    needs_curvature = is_cycle_edge or parent_has_multiple_children

                    if needs_curvature:
                        # Get node positions
                        source_pos = node_positions.get(trigger)
                        target_pos = node_positions.get(method_name)

                        if source_pos and target_pos:
                            dx = target_pos[0] - source_pos[0]

                            if dx <= 0:
                                # Child is left or directly below
                                smooth_type = "curvedCCW"  # Curve left and down
                            else:
                                # Child is to the right
                                smooth_type = "curvedCW"  # Curve right and down

                            index = self._get_child_index(
                                trigger, method_name, parent_children
                            )
                            edge_smooth = {
                                "type": smooth_type,
                                "roundness": 0.2 + (0.1 * index),
                            }
                        else:
                            # Fallback curvature
                            edge_smooth = {"type": "cubicBezier"}
                    else:
                        edge_smooth = False  # Draw straight line

                    edge_style = {
                        "color": edge_color,
                        "width": 2,
                        "arrows": "to",
                        "dashes": True if is_router_edge or is_and_condition else False,
                        "smooth": edge_smooth,
                    }

                    net.add_edge(trigger, method_name, **edge_style)

        # Add edges from router methods to their possible paths
        for router_method_name, paths in self.flow._router_paths.items():
            for path in paths:
                for listener_name, (
                    condition_type,
                    trigger_methods,
                ) in self.flow._listeners.items():
                    if path in trigger_methods:
                        is_cycle_edge = self._is_ancestor(
                            trigger, method_name, ancestors
                        )

                        # Determine if parent has multiple children
                        parent_has_multiple_children = (
                            len(parent_children.get(router_method_name, [])) > 1
                        )

                        # Edge curvature logic
                        needs_curvature = is_cycle_edge or parent_has_multiple_children

                        if needs_curvature:
                            # Get node positions
                            source_pos = node_positions.get(router_method_name)
                            target_pos = node_positions.get(listener_name)

                            if source_pos and target_pos:
                                dx = target_pos[0] - source_pos[0]

                                if dx <= 0:
                                    # Child is left or directly below
                                    smooth_type = "curvedCCW"  # Curve left and down
                                else:
                                    # Child is to the right
                                    smooth_type = "curvedCW"  # Curve right and down

                                index = self._get_child_index(
                                    router_method_name, listener_name, parent_children
                                )
                                edge_smooth = {
                                    "type": smooth_type,
                                    "roundness": 0.2 + (0.1 * index),
                                }
                            else:
                                # Fallback curvature
                                edge_smooth = {"type": "cubicBezier"}
                        else:
                            edge_smooth = False  # Straight line

                        edge_style = {
                            "color": self.colors["router_edge"],
                            "width": 2,
                            "arrows": "to",
                            "dashes": True,
                            "smooth": edge_smooth,
                        }
                        net.add_edge(router_method_name, listener_name, **edge_style)

        # Set options to disable physics
        net.set_options(
            """
            var options = {
                "physics": {
                    "enabled": false
                }
            }
            """
        )

        network_html = net.generate_html()

        # Extract just the body content from the generated HTML
        match = re.search("<body.*?>(.*?)</body>", network_html, re.DOTALL)
        if match:
            network_body = match.group(1)
        else:
            network_body = ""

        # Read the custom template
        current_dir = os.path.dirname(__file__)
        template_path = os.path.join(
            current_dir, "assets", "crewai_flow_visual_template.html"
        )
        with open(template_path, "r", encoding="utf-8") as f:
            html_template = f.read()

        # Generate the legend items HTML
        legend_items = [
            {"label": "Start Method", "color": self.colors["start"]},
            {"label": "Method", "color": self.colors["method"]},
            {
                "label": "Router",
                "color": self.colors["router"],
                "border": self.colors["router_border"],
                "dashed": True,
            },
            {"label": "Trigger", "color": self.colors["edge"], "dashed": False},
            {"label": "AND Trigger", "color": self.colors["edge"], "dashed": True},
            {
                "label": "Router Trigger",
                "color": self.colors["router_edge"],
                "dashed": True,
            },
        ]

        legend_items_html = ""
        for item in legend_items:
            if "border" in item:
                legend_items_html += f"""
                <div class="legend-item">
                <div class="legend-color-box" style="background-color: {item['color']}; border: 2px dashed {item['border']};"></div>
                <div>{item['label']}</div>
                </div>
                """
            elif item.get("dashed") is not None:
                style = "dashed" if item["dashed"] else "solid"
                legend_items_html += f"""
                <div class="legend-item">
                <div class="legend-{style}" style="border-bottom: 2px {style} {item['color']};"></div>
                <div>{item['label']}</div>
                </div>
                """
            else:
                legend_items_html += f"""
                <div class="legend-item">
                <div class="legend-color-box" style="background-color: {item['color']};"></div>
                <div>{item['label']}</div>
                </div>
                """

        # Read the logo file and encode it
        logo_path = os.path.join(current_dir, "assets", "crewai_logo.svg")
        with open(logo_path, "rb") as logo_file:
            logo_svg_data = logo_file.read()
            logo_svg_base64 = base64.b64encode(logo_svg_data).decode("utf-8")

        # Replace placeholders in the template
        final_html_content = html_template.replace("{{ title }}", "Flow Graph")
        final_html_content = final_html_content.replace(
            "{{ network_content }}", network_body
        )
        final_html_content = final_html_content.replace(
            "{{ logo_svg_base64 }}", logo_svg_base64
        )
        final_html_content = final_html_content.replace(
            "<!-- LEGEND_ITEMS_PLACEHOLDER -->", legend_items_html
        )

        # Save the final HTML content to the file
        with open(f"{filename}.html", "w", encoding="utf-8") as f:
            f.write(final_html_content)
        print(f"Graph saved as {filename}.html")

    def _calculate_node_levels(self):
        levels = {}
        queue = []
        visited = set()
        pending_and_listeners = {}

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
                if condition_type == "OR":
                    if current in trigger_methods:
                        if (
                            listener_name not in levels
                            or levels[listener_name] > current_level + 1
                        ):
                            levels[listener_name] = current_level + 1
                            if listener_name not in visited:
                                queue.append(listener_name)
                elif condition_type == "AND":
                    if listener_name not in pending_and_listeners:
                        pending_and_listeners[listener_name] = set()
                    if current in trigger_methods:
                        pending_and_listeners[listener_name].add(current)
                    if set(trigger_methods) == pending_and_listeners[listener_name]:
                        if (
                            listener_name not in levels
                            or levels[listener_name] > current_level + 1
                        ):
                            levels[listener_name] = current_level + 1
                            if listener_name not in visited:
                                queue.append(listener_name)

            # Handle router connections (same as before)
            if current in self.flow._routers.values():
                router_method_name = current
                paths = self.flow._router_paths.get(router_method_name, [])
                for path in paths:
                    for listener_name, (
                        condition_type,
                        trigger_methods,
                    ) in self.flow._listeners.items():
                        if path in trigger_methods:
                            if (
                                listener_name not in levels
                                or levels[listener_name] > current_level + 1
                            ):
                                levels[listener_name] = current_level + 1
                                if listener_name not in visited:
                                    queue.append(listener_name)
        return levels

    def _count_outgoing_edges(self):
        # Helper method to count the number of outgoing edges from each node
        counts = {}
        for method_name in self.flow._methods:
            counts[method_name] = 0
        for method_name in self.flow._listeners:
            _, trigger_methods = self.flow._listeners[method_name]
            for trigger in trigger_methods:
                if trigger in self.flow._methods:
                    counts[trigger] += 1
        return counts

    def _build_ancestor_dict(self):
        ancestors = {node: set() for node in self.flow._methods}
        visited = set()
        for node in self.flow._methods:
            if node not in visited:
                self._dfs_ancestors(node, ancestors, visited)
        print("Ancestor Relationships:")
        for node, node_ancestors in ancestors.items():
            print(f"{node}: {node_ancestors}")
        return ancestors

    def _dfs_ancestors(self, node, ancestors, visited):
        if node in visited:
            return
        visited.add(node)

        # Handle regular listeners
        for listener_name, (_, trigger_methods) in self.flow._listeners.items():
            if node in trigger_methods:
                ancestors[listener_name].add(node)
                ancestors[listener_name].update(ancestors[node])
                self._dfs_ancestors(listener_name, ancestors, visited)

        # Handle router methods separately
        if node in self.flow._routers.values():
            router_method_name = node
            paths = self.flow._router_paths.get(router_method_name, [])
            for path in paths:
                for listener_name, (_, trigger_methods) in self.flow._listeners.items():
                    if path in trigger_methods:
                        # Only propagate the ancestors of the router method, not the router method itself
                        ancestors[listener_name].update(ancestors[node])
                        self._dfs_ancestors(listener_name, ancestors, visited)

    def _is_ancestor(self, node, ancestor_candidate, ancestors):
        return ancestor_candidate in ancestors.get(node, set())

    def _build_parent_children_dict(self):
        parent_children = {}
        # Map listeners to their trigger methods
        for listener_name, (_, trigger_methods) in self.flow._listeners.items():
            for trigger in trigger_methods:
                if trigger not in parent_children:
                    parent_children[trigger] = []
                if listener_name not in parent_children[trigger]:
                    parent_children[trigger].append(listener_name)
        # Map router methods to their paths and to listeners
        for router_method_name, paths in self.flow._router_paths.items():
            for path in paths:
                # Map router method to listeners of each path
                for listener_name, (_, trigger_methods) in self.flow._listeners.items():
                    if path in trigger_methods:
                        if router_method_name not in parent_children:
                            parent_children[router_method_name] = []
                        if listener_name not in parent_children[router_method_name]:
                            parent_children[router_method_name].append(listener_name)
        # Debugging output
        print("Parent-Children Relationships:")
        for parent, children in parent_children.items():
            print(f"{parent}: {children}")
        return parent_children

    def _get_child_index(self, parent, child, parent_children):
        # Helper method to get the index of the child among the parent's children
        children = parent_children.get(parent, [])
        children.sort()
        return children.index(child)


def visualize_flow(flow, filename="flow_graph"):
    visualizer = PyvisFlowVisualizer(flow)
    visualizer.visualize(filename)
