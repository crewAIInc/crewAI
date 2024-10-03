import base64
import re


class HTMLTemplateHandler:
    def __init__(self, template_path, logo_path):
        self.template_path = template_path
        self.logo_path = logo_path

    def read_template(self):
        with open(self.template_path, "r", encoding="utf-8") as f:
            return f.read()

    def encode_logo(self):
        with open(self.logo_path, "rb") as logo_file:
            logo_svg_data = logo_file.read()
            return base64.b64encode(logo_svg_data).decode("utf-8")

    def extract_body_content(self, html):
        match = re.search("<body.*?>(.*?)</body>", html, re.DOTALL)
        return match.group(1) if match else ""

    def generate_legend_items_html(self, legend_items):
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
        return legend_items_html

    def generate_final_html(self, network_body, legend_items_html, title="Flow Plot"):
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
