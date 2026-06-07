"""Grounding DINO open-vocab backend — GPU-native, text-prompted detection.

The default open-vocab engine. Unlike osam/SAM3 (CPU-only ONNX), Grounding DINO runs on
the autodetected torch device (cuda/mps/cpu) and takes ALL labels in a single forward
pass ("drone. person. car."), so it is far faster than one-call-per-label SAM3.

Boxes only (no masks). For masks, see model_grounded_sam2 (GDINO boxes + SAM 2).

transformers is imported lazily inside __init__/predict so importing this module stays
cheap; the heavy import only happens when the backend is actually constructed.
"""
from __future__ import annotations

from typing import List, Optional

from .converters import Detection
from .open_vocab import OpenVocabSession

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"


def match_label(phrase: str, requested: List[str]) -> Optional[str]:
    """Map a Grounding DINO output phrase back to one of the requested labels.

    GDINO returns text spans of the prompt, which may be merged/partial (e.g. "drone"
    for a "a drone" prompt, or "pallet" for "wooden pallet"). Routing requires the label
    to be one of the project's labels, so we resolve the phrase to a requested label or
    drop the detection. Pure function — unit-tested without loading the model.
    """
    p = phrase.strip().lower()
    if not p:
        return None
    # 1. exact match
    for label in requested:
        if label.lower() == p:
            return label
    # 2. containment either direction — prefer the longest requested label that overlaps
    best = None
    for label in requested:
        ll = label.lower()
        if ll in p or p in ll:
            if best is None or len(ll) > len(best.lower()):
                best = label
    if best is not None:
        return best
    # 3. token-overlap fallback (multi-word labels split by the model)
    ptokens = set(p.split())
    best, best_overlap = None, 0
    for label in requested:
        overlap = len(ptokens & set(label.lower().split()))
        if overlap > best_overlap:
            best, best_overlap = label, overlap
    return best


class GdinoSession(OpenVocabSession):
    def __init__(self, device=None, model_id: str = DEFAULT_MODEL_ID):
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        from .device import get_device

        self._torch = torch
        self.device = device or get_device()
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            .to(self.device)
            .eval()
        )

    def predict(self, image_pil, labels: List[str], score_threshold: float,
                iou_threshold: float, max_detections: int) -> List[Detection]:
        # iou_threshold is accepted for interface uniformity; GDINO returns
        # already-deduplicated boxes, so no extra NMS is applied here.
        if not labels:
            return []

        # GDINO prompt: lowercase labels, each terminated with a period.
        text = " ".join(f"{label.strip().lower()}." for label in labels)
        inputs = self.processor(images=image_pil, text=text, return_tensors="pt").to(
            self.device
        )
        with self._torch.no_grad():
            outputs = self.model(**inputs)

        W, H = image_pil.size
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=score_threshold,
            text_threshold=score_threshold,
            target_sizes=[(H, W)],
        )[0]

        # transformers >=4.51: "text_labels" holds the phrases ("labels" became int ids).
        phrases = result.get("text_labels")
        if phrases is None:
            phrases = result["labels"]

        dets: List[Detection] = []
        for box, score, phrase in zip(result["boxes"], result["scores"], phrases):
            label = match_label(str(phrase), labels)
            if label is None:
                continue
            x1, y1, x2, y2 = box.tolist()
            dets.append(Detection(
                label=label,
                score=float(score),
                bbox=(x1, y1, x2, y2),
            ))

        dets.sort(key=lambda d: d.score, reverse=True)
        return dets[:max_detections]
