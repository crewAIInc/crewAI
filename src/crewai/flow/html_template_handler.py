import base64
import os
import re
from pathlib import Path

from crewai.flow.path_utils import safe_path_join, validate_file_path


class HTMLTemplateHandler:
    """Handles HTML template processing and generation for flow visualization diagrams."""

    def __init__(self, template_path, logo_path):
        """Initialize template handler with template and logo file paths.
        
        Args:
            template_path: Path to the HTML template file
            logo_path: Path to the logo SVG file
            
        Raises:
            ValueError: If template_path or logo_path is invalid or files don't exist
        """
        try:
            self.template_path = validate_file_path(template_path)
            self.logo_path = validate_file_path(logo_path)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid file path: {str(e)}")

    def read_template(self):
        """Read and return the HTML template file contents.
        
        Returns:
            str: The contents of the template file
            
        Raises:
            IOError: If template file cannot be read
        """
        try:
            with open(self.template_path, "r", encoding="utf-8") as f:
                return f.read()
        except IOError as e:
            raise IOError(f"Failed to read template file {self.template_path}: {str(e)}")

    def encode_logo(self):
        """Convert the logo SVG file to base64 encoded string.
        
        Returns:
            str: Base64 encoded logo data
            
        Raises:
            IOError: If logo file cannot be read
            ValueError: If logo data cannot be encoded
        """
        try:
            with open(self.logo_path, "rb") as logo_file:
                logo_svg_data = logo_file.read()
                try:
                    return base64.b64encode(logo_svg_data).decode("utf-8")
                except Exception as e:
                    raise ValueError(f"Failed to encode logo data: {str(e)}")
        except IOError as e:
            raise IOError(f"Failed to read logo file {self.logo_path}: {str(e)}")

    def extract_body_content(self, html):
        """Extract and return content between body tags from HTML string.
        
        Args:
            html: HTML string to extract body content from
            
        Returns:
            str: Content between body tags, or empty string if not found
            
        Raises:
            ValueError: If input HTML is invalid
        """
        if not html or not isinstance(html, str):
            raise ValueError("Input HTML must be a non-empty string")
            
        match = re.search("<body.*?>(.*?)</body>", html, re.DOTALL)
        return match.group(1) if match else ""

    def generate_legend_items_html(self, legend_items):
        """Generate HTML markup for the legend items.
        
        Args:
            legend_items: List of dictionaries containing legend item properties
            
        Returns:
            str: Generated HTML markup for legend items
            
        Raises:
            ValueError: If legend_items is invalid or missing required properties
        """
        if not isinstance(legend_items, list):
            raise ValueError("legend_items must be a list")
            
        legend_items_html = ""
        for item in legend_items:
            if not isinstance(item, dict):
                raise ValueError("Each legend item must be a dictionary")
            if "color" not in item:
                raise ValueError("Each legend item must have a 'color' property")
            if "label" not in item:
                raise ValueError("Each legend item must have a 'label' property")
                
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
        return legend_items_html

    def generate_final_html(self, network_body, legend_items_html, title="Flow Plot"):
        """Combine all components into final HTML document with network visualization.
        
        Args:
            network_body: HTML string containing network visualization
            legend_items_html: HTML string containing legend items markup
            title: Title for the visualization page (default: "Flow Plot")
            
        Returns:
            str: Complete HTML document with all components integrated
            
        Raises:
            ValueError: If any input parameters are invalid
            IOError: If template or logo files cannot be read
        """
        if not isinstance(network_body, str):
            raise ValueError("network_body must be a string")
        if not isinstance(legend_items_html, str):
            raise ValueError("legend_items_html must be a string")
        if not isinstance(title, str):
            raise ValueError("title must be a string")
            
        try:
            html_template = self.read_template()
            logo_svg_base64 = self.encode_logo()

            final_html_content = html_template.replace("{{ title }}", title)
            final_html_content = final_html_content.replace(
                "{{ network_content }}", network_body
            )
            final_html_content = final_html_content.replace(
                "{{ logo_svg_base64 }}", logo_svg_base64
            )
            final_html_content = final_html_content.replace(
                "<!-- LEGEND_ITEMS_PLACEHOLDER -->", legend_items_html
            )

            return final_html_content
        except Exception as e:
            raise ValueError(f"Failed to generate final HTML: {str(e)}")
