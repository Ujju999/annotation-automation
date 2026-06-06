"""YOLO inference for the model's known classes. Returns the shared Detection type."""
from __future__ import annotations

import logging
from typing import List

from ultralytics import YOLO

from .converters import Detection

logger = logging.getLogger(__name__)


class YoloSession:
    def __init__(self, model_path: str, device, task: str = "detect"):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.class_names = self.model.names          # {0: "cat", 1: "dog", ...}
        self.task = task
        logger.info("YOLO loaded: %s (task=%s, classes=%s)",
                    model_path, task, list(self.class_names.values()))

    def predict(self, image_pil, labels, score_threshold: float = 0.3,
                iou_threshold: float = 0.5) -> List[Detection]:
        want = {i for i, n in self.class_names.items() if n in labels}
        if not want:
            return []

        result = self.model.predict(
            image_pil, conf=score_threshold, iou=iou_threshold,
            classes=list(want), verbose=False,
        )[0]

        W, H = image_pil.size
        masks = getattr(result, "masks", None)
        dets: List[Detection] = []
        for i in range(len(result.boxes)):
            label = self.class_names[int(result.boxes.cls[i])]
            x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
            mask = None
            if masks is not None and masks.xy is not None:   # segment mode
                mask = self._mask_from_xy(masks.xy[i], W, H)
            dets.append(Detection(
                label=label,
                score=float(result.boxes.conf[i]),
                bbox=(x1, y1, x2, y2),
                mask=mask,
            ))
        return dets

    @staticmethod
    def _mask_from_xy(xy, W: int, H: int):
        import numpy as np
        import skimage.draw

        mask = np.zeros((H, W), dtype=bool)
        if xy is None or len(xy) < 3:
            return mask
        rr, cc = skimage.draw.polygon(xy[:, 1], xy[:, 0], (H, W))
        mask[rr, cc] = True
        return mask
