def get_legend_items(colors):
    return [
        {"label": "Start Method", "color": colors["start"]},
        {"label": "Method", "color": colors["method"]},
        {
            "label": "Router",
            "color": colors["router"],
            "border": colors["router_border"],
            "dashed": True,
        },
        {"label": "Trigger", "color": colors["edge"], "dashed": False},
        {"label": "AND Trigger", "color": colors["edge"], "dashed": True},
        {
            "label": "Router Trigger",
            "color": colors["router_edge"],
            "dashed": True,
        },
    ]


def generate_legend_items_html(legend_items):
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
