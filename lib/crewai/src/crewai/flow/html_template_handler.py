import base64
import re

from crewai.flow.path_utils import validate_path_exists


class HTMLTemplateHandler:
    """Handles HTML template processing and generation for flow visualization diagrams."""

    def __init__(self, template_path, logo_path):
        """
        Initialize HTMLTemplateHandler with validated template and logo paths.

        Parameters
        ----------
        template_path : str
            Path to the HTML template file.
        logo_path : str
            Path to the logo image file.

        Raises
        ------
        ValueError
            If template or logo paths are invalid or files don't exist.
        """
        try:
            self.template_path = validate_path_exists(template_path, "file")
            self.logo_path = validate_path_exists(logo_path, "file")
        except ValueError as e:
            raise ValueError(f"Invalid template or logo path: {e}") from e

    def read_template(self):
        """Read and return the HTML template file contents."""
        with open(self.template_path, "r", encoding="utf-8") as f:
            return f.read()

    def encode_logo(self):
        """Convert the logo SVG file to base64 encoded string."""
        with open(self.logo_path, "rb") as logo_file:
            logo_svg_data = logo_file.read()
            return base64.b64encode(logo_svg_data).decode("utf-8")

    def extract_body_content(self, html):
        """Extract and return content between body tags from HTML string."""
        match = re.search("<body.*?>(.*?)</body>", html, re.DOTALL)
        return match.group(1) if match else ""

    def generate_legend_items_html(self, legend_items):
        """Generate HTML markup for the legend items."""
        legend_items_html = ""
        for item in legend_items:
            if "border" in item:
                legend_items_html += f"""
                <div class="legend-item">
                <div class="legend-color-box" style="background-color: {item["color"]}; border: 2px dashed {item["border"]};"></div>
                <div>{item["label"]}</div>
                </div>
                """
            elif item.get("dashed") is not None:
                style = "dashed" if item["dashed"] else "solid"
                legend_items_html += f"""
                <div class="legend-item">
                <div class="legend-{style}" style="border-bottom: 2px {style} {item["color"]};"></div>
                <div>{item["label"]}</div>
                </div>
                """
            else:
                legend_items_html += f"""
                <div class="legend-item">
                <div class="legend-color-box" style="background-color: {item["color"]};"></div>
                <div>{item["label"]}</div>
                </div>
                """
        return legend_items_html

    def generate_final_html(self, network_body, legend_items_html, title="Flow Plot"):
        """Combine all components into final HTML document with network visualization."""
        html_template = self.read_template()
        logo_svg_base64 = self.encode_logo()

        return (
            html_template.replace("{{ title }}", title)
            .replace("{{ network_content }}", network_body)
            .replace("{{ logo_svg_base64 }}", logo_svg_base64)
            .replace("<!-- LEGEND_ITEMS_PLACEHOLDER -->", legend_items_html)
        )
