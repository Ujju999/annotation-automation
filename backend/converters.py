from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .routing import Control, RECT


@dataclass
class Detection:
    label: str
    score: float
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 in PIXELS
    mask: Optional["object"] = None           # full-image HxW bool ndarray, optional


def _bbox_poly(bbox, W: int, H: int) -> List[List[float]]:
    x1, y1, x2, y2 = bbox
    return [
        [x1 / W * 100, y1 / H * 100],
        [x2 / W * 100, y1 / H * 100],
        [x2 / W * 100, y2 / H * 100],
        [x1 / W * 100, y2 / H * 100],
    ]


def _mask_poly(mask, W: int, H: int, tolerance: float = 1.5) -> Optional[List[List[float]]]:
    import skimage.measure  # lazy: only needed for segmentation output

    contours = skimage.measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    poly = skimage.measure.approximate_polygon(max(contours, key=len), tolerance)
    if len(poly) < 3:
        return None
    # contour points are (row, col) = (y, x)
    return [[float(x) / W * 100, float(y) / H * 100] for y, x in poly]


def detection_to_result(det: Detection, control: Control, W: int, H: int) -> dict:
    """Build one Label Studio result region for a detection under a control tag."""
    base = {
        "id": uuid.uuid4().hex[:10],
        "from_name": control.from_name,
        "to_name": control.to_name,
        "original_width": W,
        "original_height": H,
        "score": det.score,
    }
    if control.type == RECT:
        x1, y1, x2, y2 = det.bbox
        base["type"] = "rectanglelabels"
        base["value"] = {
            "x": x1 / W * 100,
            "y": y1 / H * 100,
            "width": (x2 - x1) / W * 100,
            "height": (y2 - y1) / H * 100,
            "rectanglelabels": [det.label],
        }
    else:  # PolygonLabels — prefer mask contour, fall back to bbox so it never renders empty
        points = _mask_poly(det.mask, W, H) if det.mask is not None else None
        base["type"] = "polygonlabels"
        base["value"] = {
            "points": points or _bbox_poly(det.bbox, W, H),
            "polygonlabels": [det.label],
        }
    return base
