from __future__ import annotations

import importlib.util
import os
from typing import List

from .converters import Detection
from .open_vocab import OpenVocabSession

DEFAULT_MODEL = "yolov8x-worldv2.pt"


class YoloWorldSession(OpenVocabSession):
    def __init__(self, device=None, model: str = None):
        if importlib.util.find_spec("clip") is None:
            raise ImportError(
                "OPEN_VOCAB_BACKEND=yolo_world requires CLIP (for set_classes). "
                'Install it with: pip install ".[yolo_world]"'
            )
        from ultralytics import YOLOWorld

        from .device import get_device

        self.device = device or get_device()
        self.model_name = model or os.getenv("YOLO_WORLD_MODEL", DEFAULT_MODEL)
        self.model = YOLOWorld(self.model_name)
        self.model.to(self.device)

    def predict(self, image_pil, labels: List[str], score_threshold: float,
                iou_threshold: float, max_detections: int) -> List[Detection]:
        if not labels:
            return []

        # The requested labels become the detection vocabulary for this image.
        self.model.set_classes(list(labels))
        result = self.model.predict(
            image_pil, conf=score_threshold, iou=iou_threshold,
            max_det=max_detections, verbose=False,
        )[0]

        # The set_classes vocabulary. Ultralytics (8.4.x) returns this as a dict
        # {0: "drone", ...} on the first image but a plain list ["drone", ...] on
        # every later one, so normalise to indexed access that works for both.
        names = result.names
        dets: List[Detection] = []
        for i in range(len(result.boxes)):
            cls_idx = int(result.boxes.cls[i])
            label = names.get(cls_idx) if isinstance(names, dict) else (
                names[cls_idx] if 0 <= cls_idx < len(names) else None)
            if label is None:
                continue
            x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
            dets.append(Detection(
                label=label,
                score=float(result.boxes.conf[i]),
                bbox=(x1, y1, x2, y2),
            ))
        return dets
