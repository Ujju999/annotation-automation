from __future__ import annotations

import os
from typing import List

from .converters import Detection
from .open_vocab import OpenVocabSession

DEFAULT_SAM2_MODEL_ID = "facebook/sam2-hiera-large"


class GroundedSam2Session(OpenVocabSession):
    def __init__(self, device=None, model_id: str = "IDEA-Research/grounding-dino-base",
                 sam2_model_id: str = None):
        import torch
        from transformers import Sam2Model, Sam2Processor

        from .model_gdino import GdinoSession

        # Reuse the Grounding DINO backend for box detection (shares device autodetect).
        self.gdino = GdinoSession(device=device, model_id=model_id)
        self.device = self.gdino.device

        sam2_id = sam2_model_id or os.getenv("SAM2_MODEL_ID", DEFAULT_SAM2_MODEL_ID)
        self.sam2_processor = Sam2Processor.from_pretrained(sam2_id)
        self.sam2 = Sam2Model.from_pretrained(sam2_id).to(self.device).eval()
        self._torch = torch

    def predict(self, image_pil, labels: List[str], score_threshold: float,
                iou_threshold: float, max_detections: int) -> List[Detection]:
        # 1. boxes from Grounding DINO
        dets = self.gdino.predict(
            image_pil, labels, score_threshold, iou_threshold, max_detections)
        if not dets:
            return []

        # 2. one SAM 2 mask per box (boxes as prompts). input_boxes = [batch][n_boxes][4].
        boxes = [[list(d.bbox) for d in dets]]
        inputs = self.sam2_processor(
            images=image_pil, input_boxes=boxes, return_tensors="pt"
        ).to(self.device)
        with self._torch.no_grad():
            outputs = self.sam2(**inputs)

        # masks[0]: (n_boxes, n_candidates, H, W) binary at original resolution.
        masks = self.sam2_processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"]
        )[0]
        # iou_scores: (1, n_boxes, n_candidates) — keep the best candidate per box.
        iou_scores = outputs.iou_scores[0]

        for i, det in enumerate(dets):
            best = int(iou_scores[i].argmax())
            det.mask = masks[i, best].cpu().numpy().astype(bool)
        return dets
