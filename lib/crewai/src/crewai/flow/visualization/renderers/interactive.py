"""Interactive HTML renderer for Flow structure visualization."""

import json
from pathlib import Path
import tempfile
from typing import Any, ClassVar
import webbrowser

from jinja2 import Environment, FileSystemLoader, nodes, select_autoescape
from jinja2.ext import Extension
from jinja2.parser import Parser

from crewai.flow.visualization.builder import calculate_execution_paths
from crewai.flow.visualization.types import FlowStructure


class CSSExtension(Extension):
    """Jinja2 extension for rendering CSS link tags.

    Provides {% css 'path/to/file.css' %} tag syntax.
    """

    tags: ClassVar[set[str]] = {"css"}  # type: ignore[misc]

    def parse(self, parser: Parser) -> nodes.Node:
        """Parse {% css 'styles.css' %} tag.

        Args:
            parser: Jinja2 parser instance.

        Returns:
            Output node with rendered CSS link tag.
        """
        lineno: int = next(parser.stream).lineno
        args: list[nodes.Expr] = [parser.parse_expression()]
        return nodes.Output([self.call_method("_render_css", args)]).set_lineno(lineno)

    def _render_css(self, href: str) -> str:
        """Render CSS link tag.

        Args:
            href: Path to CSS file.

        Returns:
            HTML link tag string.
        """
        return f'<link rel="stylesheet" href="{href}">'


class JSExtension(Extension):
    """Jinja2 extension for rendering script tags.

    Provides {% js 'path/to/file.js' %} tag syntax.
    """

    tags: ClassVar[set[str]] = {"js"}  # type: ignore[misc]

    def parse(self, parser: Parser) -> nodes.Node:
        """Parse {% js 'script.js' %} tag.

        Args:
            parser: Jinja2 parser instance.

        Returns:
            Output node with rendered script tag.
        """
        lineno: int = next(parser.stream).lineno
        args: list[nodes.Expr] = [parser.parse_expression()]
        return nodes.Output([self.call_method("_render_js", args)]).set_lineno(lineno)

    def _render_js(self, src: str) -> str:
        """Render script tag.

        Args:
            src: Path to JavaScript file.

        Returns:
            HTML script tag string.
        """
        return f'<script src="{src}"></script>'


CREWAI_ORANGE = "#FF5A50"
DARK_GRAY = "#333333"
WHITE = "#FFFFFF"
GRAY = "#666666"
BG_DARK = "#0d1117"
BG_CARD = "#161b22"
BORDER_SUBTLE = "#30363d"
TEXT_PRIMARY = "#e6edf3"
TEXT_SECONDARY = "#7d8590"


def calculate_node_positions(
    dag: FlowStructure,
) -> dict[str, dict[str, int | float]]:
    """Calculate hierarchical positions (level, x, y) for each node.

    Args:
        dag: FlowStructure containing nodes and edges.

    Returns:
        Dictionary mapping node names to their position data (level, x, y).
    """
    children: dict[str, list[str]] = {name: [] for name in dag["nodes"]}
    parents: dict[str, list[str]] = {name: [] for name in dag["nodes"]}

    for edge in dag["edges"]:
        source = edge["source"]
        target = edge["target"]
        if source in children and target in children:
            children[source].append(target)
            parents[target].append(source)

    levels: dict[str, int] = {}
    queue: list[tuple[str, int]] = []

    for start_method in dag["start_methods"]:
        if start_method in dag["nodes"]:
            levels[start_method] = 0
            queue.append((start_method, 0))

    visited: set[str] = set()
    while queue:
        node, level = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)

        if node not in levels or levels[node] < level:
            levels[node] = level

        for child in children.get(node, []):
            if child not in visited:
                child_level = level + 1
                if child not in levels or levels[child] < child_level:
                    levels[child] = child_level
                queue.append((child, child_level))

    for name in dag["nodes"]:
        if name not in levels:
            levels[name] = 0

    nodes_by_level: dict[int, list[str]] = {}
    for node, level in levels.items():
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node)

    positions: dict[str, dict[str, int | float]] = {}
    level_separation = 300  # Vertical spacing between levels
    node_spacing = 400  # Horizontal spacing between nodes

    parent_count: dict[str, int] = {}
    for node, parent_list in parents.items():
        parent_count[node] = len(parent_list)

    for level, nodes_at_level in sorted(nodes_by_level.items()):
        y = level * level_separation

        if level == 0:
            num_nodes = len(nodes_at_level)
            for i, node in enumerate(nodes_at_level):
                x = (i - (num_nodes - 1) / 2) * node_spacing
                positions[node] = {"level": level, "x": x, "y": y}
        else:
            for i, node in enumerate(nodes_at_level):
                parent_list = parents.get(node, [])
                parent_positions: list[float] = [
                    positions[parent]["x"]
                    for parent in parent_list
                    if parent in positions
                ]

                if parent_positions:
                    if len(parent_positions) > 1 and len(set(parent_positions)) == 1:
                        base_x = parent_positions[0]
                        avg_x = base_x + node_spacing * 0.4
                    else:
                        avg_x = sum(parent_positions) / len(parent_positions)
                else:
                    avg_x = i * node_spacing * 0.5

                positions[node] = {"level": level, "x": avg_x, "y": y}

            nodes_at_level_sorted = sorted(
                nodes_at_level, key=lambda n: positions[n]["x"]
            )
            min_spacing = node_spacing * 0.6  # Minimum horizontal distance

            for i in range(len(nodes_at_level_sorted) - 1):
                current_node = nodes_at_level_sorted[i]
                next_node = nodes_at_level_sorted[i + 1]

                current_x = positions[current_node]["x"]
                next_x = positions[next_node]["x"]

                if next_x - current_x < min_spacing:
                    positions[next_node]["x"] = current_x + min_spacing

    return positions


def render_interactive(
    dag: FlowStructure,
    filename: str = "flow_dag.html",
    show: bool = True,
) -> str:
    """Create interactive HTML visualization of Flow structure.

    Generates three output files in a temporary directory: HTML template,
    CSS stylesheet, and JavaScript. Optionally opens the visualization in
    default browser.

    Args:
        dag: FlowStructure to visualize.
        filename: Output HTML filename (basename only, no path).
        show: Whether to open in browser.

    Returns:
        Absolute path to generated HTML file in temporary directory.
    """
    node_positions = calculate_node_positions(dag)

    nodes_list: list[dict[str, Any]] = []
    for name, metadata in dag["nodes"].items():
        node_type: str = metadata.get("type", "listen")

        color_config: dict[str, Any]
        font_color: str
        border_width: int

        if node_type == "start":
            color_config = {
                "background": "var(--node-bg-start)",
                "border": "var(--node-border-start)",
                "highlight": {
                    "background": "var(--node-bg-start)",
                    "border": "var(--node-border-start)",
                },
            }
            font_color = "var(--node-text-color)"
            border_width = 3
        elif node_type == "router":
            color_config = {
                "background": "var(--node-bg-router)",
                "border": CREWAI_ORANGE,
                "highlight": {
                    "background": "var(--node-bg-router)",
                    "border": CREWAI_ORANGE,
                },
            }
            font_color = "var(--node-text-color)"
            border_width = 3
        else:
            color_config = {
                "background": "var(--node-bg-listen)",
                "border": "var(--node-border-listen)",
                "highlight": {
                    "background": "var(--node-bg-listen)",
                    "border": "var(--node-border-listen)",
                },
            }
            font_color = "var(--node-text-color)"
            border_width = 3

        title_parts: list[str] = []

        type_badge_bg: str = (
            CREWAI_ORANGE if node_type in ["start", "router"] else DARK_GRAY
        )
        title_parts.append(f"""
            <div style="border-bottom: 1px solid rgba(102,102,102,0.15); padding-bottom: 8px; margin-bottom: 10px;">
                <div style="font-size: 13px; font-weight: 700; color: {DARK_GRAY}; margin-bottom: 6px;">{name}</div>
                <span style="display: inline-block; background: {type_badge_bg}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{node_type}</span>
            </div>
        """)

        if metadata.get("condition_type"):
            condition = metadata["condition_type"]
            if condition == "AND":
                condition_badge_bg = "rgba(255,90,80,0.12)"
                condition_color = CREWAI_ORANGE
            elif condition == "IF":
                condition_badge_bg = "rgba(255,90,80,0.18)"
                condition_color = CREWAI_ORANGE
            else:
                condition_badge_bg = "rgba(102,102,102,0.12)"
                condition_color = GRAY
            title_parts.append(f"""
                <div style="margin-bottom: 8px;">
                    <div style="font-size: 10px; text-transform: uppercase; color: {GRAY}; letter-spacing: 0.5px; margin-bottom: 3px; font-weight: 600;">Condition</div>
                    <span style="display: inline-block; background: {condition_badge_bg}; color: {condition_color}; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">{condition}</span>
                </div>
            """)

        if metadata.get("trigger_methods"):
            triggers = metadata["trigger_methods"]
            triggers_items = "".join(
                [
                    f'<li style="margin: 3px 0;"><code style="background: rgba(102,102,102,0.08); padding: 2px 6px; border-radius: 3px; font-size: 10px; color: {DARK_GRAY}; border: 1px solid rgba(102,102,102,0.12);">{t}</code></li>'
                    for t in triggers
                ]
            )
            title_parts.append(f"""
                <div style="margin-bottom: 8px;">
                    <div style="font-size: 10px; text-transform: uppercase; color: {GRAY}; letter-spacing: 0.5px; margin-bottom: 4px; font-weight: 600;">Triggers</div>
                    <ul style="list-style: none; padding: 0; margin: 0;">{triggers_items}</ul>
                </div>
            """)

        if metadata.get("router_paths"):
            paths = metadata["router_paths"]
            paths_items = "".join(
                [
                    f'<li style="margin: 3px 0;"><code style="background: rgba(255,90,80,0.08); padding: 2px 6px; border-radius: 3px; font-size: 10px; color: {CREWAI_ORANGE}; border: 1px solid rgba(255,90,80,0.2); font-weight: 600;">{p}</code></li>'
                    for p in paths
                ]
            )
            title_parts.append(f"""
                <div>
                    <div style="font-size: 10px; text-transform: uppercase; color: {GRAY}; letter-spacing: 0.5px; margin-bottom: 4px; font-weight: 600;">Router Paths</div>
                    <ul style="list-style: none; padding: 0; margin: 0;">{paths_items}</ul>
                </div>
            """)

        bg_color = color_config["background"]
        border_color = color_config["border"]

        position_data = node_positions.get(name, {"level": 0, "x": 0, "y": 0})

        node_data: dict[str, Any] = {
            "id": name,
            "label": name,
            "title": "".join(title_parts),
            "shape": "custom",
            "size": 30,
            "level": position_data["level"],
            "nodeStyle": {
                "name": name,
                "bgColor": bg_color,
                "borderColor": border_color,
                "borderWidth": border_width,
                "fontColor": font_color,
            },
            "opacity": 1.0,
            "glowSize": 0,
            "glowColor": None,
        }

        # Add x,y only for graphs with 3-4 nodes
        total_nodes = len(dag["nodes"])
        if 3 <= total_nodes <= 4:
            node_data["x"] = position_data["x"]
            node_data["y"] = position_data["y"]

        nodes_list.append(node_data)

    execution_paths: int = calculate_execution_paths(dag)

    edges_list: list[dict[str, Any]] = []
    for edge in dag["edges"]:
        edge_label: str = ""
        edge_color: str = GRAY
        edge_dashes: bool | list[int] = False

        if edge["is_router_path"]:
            edge_color = CREWAI_ORANGE
            edge_dashes = [15, 10]
            if "router_path_label" in edge:
                edge_label = edge["router_path_label"]
        elif edge["condition_type"] == "AND":
            edge_label = "AND"
            edge_color = CREWAI_ORANGE
        elif edge["condition_type"] == "OR":
            edge_label = "OR"
            edge_color = GRAY

        edge_data: dict[str, Any] = {
            "from": edge["source"],
            "to": edge["target"],
            "label": edge_label,
            "arrows": "to",
            "width": 2,
            "selectionWidth": 0,
            "color": {
                "color": edge_color,
                "highlight": edge_color,
            },
        }

        if edge_dashes is not False:
            edge_data["dashes"] = edge_dashes

        edges_list.append(edge_data)

    template_dir = Path(__file__).parent.parent / "assets"
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml", "css", "js"]),
        variable_start_string="'{{",
        variable_end_string="}}'",
        extensions=[CSSExtension, JSExtension],
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="crewai_flow_"))
    output_path = temp_dir / Path(filename).name
    css_filename = output_path.stem + "_style.css"
    css_output_path = temp_dir / css_filename
    js_filename = output_path.stem + "_script.js"
    js_output_path = temp_dir / js_filename

    css_file = template_dir / "style.css"
    css_content = css_file.read_text(encoding="utf-8")

    css_content = css_content.replace("'{{ WHITE }}'", WHITE)
    css_content = css_content.replace("'{{ DARK_GRAY }}'", DARK_GRAY)
    css_content = css_content.replace("'{{ GRAY }}'", GRAY)
    css_content = css_content.replace("'{{ CREWAI_ORANGE }}'", CREWAI_ORANGE)

    css_output_path.write_text(css_content, encoding="utf-8")

    js_file = template_dir / "interactive.js"
    js_content = js_file.read_text(encoding="utf-8")

    dag_nodes_json = json.dumps(dag["nodes"])
    dag_full_json = json.dumps(dag)

    js_content = js_content.replace("{{ WHITE }}", WHITE)
    js_content = js_content.replace("{{ DARK_GRAY }}", DARK_GRAY)
    js_content = js_content.replace("{{ GRAY }}", GRAY)
    js_content = js_content.replace("{{ CREWAI_ORANGE }}", CREWAI_ORANGE)
    js_content = js_content.replace("'{{ nodeData }}'", dag_nodes_json)
    js_content = js_content.replace("'{{ dagData }}'", dag_full_json)
    js_content = js_content.replace("'{{ nodes_list_json }}'", json.dumps(nodes_list))
    js_content = js_content.replace("'{{ edges_list_json }}'", json.dumps(edges_list))

    js_output_path.write_text(js_content, encoding="utf-8")

    template = env.get_template("interactive_flow.html.j2")

    html_content = template.render(
        CREWAI_ORANGE=CREWAI_ORANGE,
        DARK_GRAY=DARK_GRAY,
        WHITE=WHITE,
        GRAY=GRAY,
        BG_DARK=BG_DARK,
        BG_CARD=BG_CARD,
        BORDER_SUBTLE=BORDER_SUBTLE,
        TEXT_PRIMARY=TEXT_PRIMARY,
        TEXT_SECONDARY=TEXT_SECONDARY,
        nodes_list_json=json.dumps(nodes_list),
        edges_list_json=json.dumps(edges_list),
        dag_nodes_count=len(dag["nodes"]),
        dag_edges_count=len(dag["edges"]),
        execution_paths=execution_paths,
        css_path=css_filename,
        js_path=js_filename,
    )

    output_path.write_text(html_content, encoding="utf-8")

    if show:
        webbrowser.open(f"file://{output_path.absolute()}")

    return str(output_path.absolute())
