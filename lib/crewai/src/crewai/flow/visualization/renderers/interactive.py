"""Interactive HTML renderer for Flow structure visualization."""

import json
from pathlib import Path
from typing import Any
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

    tags: ClassVar[set[str]] = {"css"}  # type: ignore[assignment]

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

    tags: ClassVar[set[str]] = {"js"}  # type: ignore[assignment]

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


def render_interactive(
    dag: FlowStructure,
    filename: str = "flow_dag.html",
    show: bool = True,
) -> str:
    """Create interactive HTML visualization of Flow structure.

    Generates three output files: HTML template, CSS stylesheet, and JavaScript.
    Optionally opens the visualization in default browser.

    Args:
        dag: FlowStructure to visualize.
        filename: Output HTML filename.
        show: Whether to open in browser.

    Returns:
        Absolute path to generated HTML file.
    """
    nodes_list: list[dict[str, Any]] = []
    for name, metadata in dag["nodes"].items():
        node_type: str = metadata.get("type", "listen")

        color_config: dict[str, Any]
        font_color: str
        border_width: int

        if node_type == "start":
            color_config = {
                "background": CREWAI_ORANGE,
                "border": CREWAI_ORANGE,
                "highlight": {
                    "background": CREWAI_ORANGE,
                    "border": CREWAI_ORANGE,
                },
            }
            font_color = WHITE
            border_width = 2
        elif node_type == "router":
            color_config = {
                "background": DARK_GRAY,
                "border": CREWAI_ORANGE,
                "highlight": {
                    "background": DARK_GRAY,
                    "border": CREWAI_ORANGE,
                },
            }
            font_color = WHITE
            border_width = 3
        else:
            color_config = {
                "background": DARK_GRAY,
                "border": DARK_GRAY,
                "highlight": {
                    "background": DARK_GRAY,
                    "border": DARK_GRAY,
                },
            }
            font_color = WHITE
            border_width = 2

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

        nodes_list.append(
            {
                "id": name,
                "label": name,
                "title": "".join(title_parts),
                "shape": "custom",
                "size": 30,
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
        )

    execution_paths: int = calculate_execution_paths(dag)

    edges_list: list[dict[str, Any]] = []
    for edge in dag["edges"]:
        edge_label: str = ""
        edge_color: str = GRAY
        edge_dashes: bool | list[int] = False

        if edge["is_router_path"]:
            edge_color = CREWAI_ORANGE
            edge_dashes = [15, 10]
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

    output_path = Path(filename)
    output_dir = output_path.parent
    css_filename = output_path.stem + "_style.css"
    css_output_path = output_dir / css_filename
    js_filename = output_path.stem + "_script.js"
    js_output_path = output_dir / js_filename

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
