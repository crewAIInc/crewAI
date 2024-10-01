DARK_GRAY = "#333333"
CREWAI_ORANGE = "#FF5A50"
GRAY = "#666666"
WHITE = "#FFFFFF"

COLORS = {
    "bg": WHITE,
    "start": CREWAI_ORANGE,
    "method": DARK_GRAY,
    "router": DARK_GRAY,
    "router_border": CREWAI_ORANGE,
    "edge": GRAY,
    "router_edge": CREWAI_ORANGE,
    "text": WHITE,
}

NODE_STYLES = {
    "start": {
        "color": COLORS["start"],
        "shape": "box",
        "font": {"color": COLORS["text"]},
        "margin": {"top": 10, "bottom": 8, "left": 10, "right": 10},
    },
    "method": {
        "color": COLORS["method"],
        "shape": "box",
        "font": {"color": COLORS["text"]},
        "margin": {"top": 10, "bottom": 8, "left": 10, "right": 10},
    },
    "router": {
        "color": {
            "background": COLORS["router"],
            "border": COLORS["router_border"],
            "highlight": {
                "border": COLORS["router_border"],
                "background": COLORS["router"],
            },
        },
        "shape": "box",
        "font": {"color": COLORS["text"]},
        "borderWidth": 3,
        "borderWidthSelected": 4,
        "shapeProperties": {"borderDashes": [5, 5]},
        "margin": {"top": 10, "bottom": 8, "left": 10, "right": 10},
    },
}
