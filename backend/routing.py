from __future__ import annotations

from dataclasses import dataclass
from typing import List

RECT = "RectangleLabels"
POLY = "PolygonLabels"
# BrushLabels is added in the segmentation phase (see plan.md "Later — segmentation").
SUPPORTED = {RECT, POLY}


@dataclass(frozen=True)
class Control:
    from_name: str   # control tag name, e.g. "label"
    to_name: str     # object tag name, e.g. "img"
    type: str        # "RectangleLabels" | "PolygonLabels"


@dataclass(frozen=True)
class LabelRoute:
    label: str
    control: Control
    engine: str      # "yolo" | "open_vocab"


def build_routes(parsed_label_config: dict, yolo_classes: set) -> List[LabelRoute]:
    """Turn ``self.parsed_label_config`` into per-label routes.

    A label's engine is decided purely by membership in ``yolo_classes`` — anything
    the YOLO model doesn't know goes to the open-vocab engine.
    """
    routes: List[LabelRoute] = []
    for from_name, info in parsed_label_config.items():
        if info.get("type") not in SUPPORTED:
            continue
        to_name = (info.get("to_name") or [""])[0]
        control = Control(from_name=from_name, to_name=to_name, type=info["type"])
        for label in info.get("labels", []):
            engine = "yolo" if label in yolo_classes else "open_vocab"
            routes.append(LabelRoute(label=label, control=control, engine=engine))
    return routes


def labels_for(routes: List[LabelRoute], engine: str) -> List[str]:
    """Sorted, de-duplicated label names handled by ``engine``."""
    return sorted({r.label for r in routes if r.engine == engine})


def controls_for(routes: List[LabelRoute], label: str) -> List[Control]:
    """Every control tag a detected ``label`` should be emitted under.

    Engine need not be checked: a label maps to exactly one engine, so all of its
    routes already share that engine.
    """
    return [r.control for r in routes if r.label == label]
