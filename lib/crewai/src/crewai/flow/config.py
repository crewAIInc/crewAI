DARK_GRAY = "#333333"
CREWAI_ORANGE = "#FF5A50"
GRAY = "#666666"
WHITE = "#FFFFFF"
BLACK = "#000000"

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
        "color": CREWAI_ORANGE,
        "shape": "box",
        "font": {"color": WHITE},
        "margin": {"top": 10, "bottom": 8, "left": 10, "right": 10},
    },
    "method": {
        "color": DARK_GRAY,
        "shape": "box",
        "font": {"color": WHITE},
        "margin": {"top": 10, "bottom": 8, "left": 10, "right": 10},
    },
    "router": {
        "color": {
            "background": DARK_GRAY,
            "border": CREWAI_ORANGE,
            "highlight": {
                "border": CREWAI_ORANGE,
                "background": DARK_GRAY,
            },
        },
        "shape": "box",
        "font": {"color": WHITE},
        "borderWidth": 3,
        "borderWidthSelected": 4,
        "shapeProperties": {"borderDashes": [5, 5]},
        "margin": {"top": 10, "bottom": 8, "left": 10, "right": 10},
    },
    "crew": {
        "color": {
            "background": WHITE,
            "border": CREWAI_ORANGE,
        },
        "shape": "box",
        "font": {"color": BLACK},
        "borderWidth": 3,
        "borderWidthSelected": 4,
        "shapeProperties": {"borderDashes": False},
        "margin": {"top": 10, "bottom": 8, "left": 10, "right": 10},
    },
}
