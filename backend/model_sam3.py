from __future__ import annotations

import collections
import hashlib
import logging
import os
from typing import List

import numpy as np

from .converters import Detection
from .open_vocab import OpenVocabSession

logger = logging.getLogger(__name__)

DEFAULT_SAM3_MODEL_ID = "facebook/sam3"


def _full_mask(ann, W: int, H: int) -> np.ndarray:
    """osam returns a bbox-sized mask placed at the box origin; copy the part that
    lands inside the image onto a full-size canvas.

    osam boxes can extend past the image edges (negative origin, or beyond W/H),
    so intersect the mask with the canvas on both sides before copying — a naive
    ``full[ymin:ymax] = mask`` either broadcast-errors or wraps a negative index.
    """
    full = np.zeros((H, W), dtype=bool)
    bb, mask = ann.bounding_box, ann.mask
    if mask is None or bb is None:
        return full
    mh, mw = mask.shape[:2]
    sy, sx = max(0, -bb.ymin), max(0, -bb.xmin)   # mask rows/cols off the top/left edge
    dy, dx = max(0, bb.ymin), max(0, bb.xmin)     # canvas top-left, clamped to >= 0
    h = min(mh - sy, H - dy)
    w = min(mw - sx, W - dx)
    if h > 0 and w > 0:
        full[dy:dy + h, dx:dx + w] = mask[sy:sy + h, sx:sx + w]
    return full


class Sam3Session(OpenVocabSession):
    """Native SAM3 backend (facebook/sam3) — no osam dependency, runs on CUDA/MPS/CPU.

    SAM3 accepts text prompts directly, so no separate detector is needed.
    Pipeline: text label → SAM3 → boxes + masks in one shot.
    The vision encoder output is cached by image-pixel hash so the expensive
    encode step runs once per unique image, regardless of how many labels are queried.
    """

    def __init__(self, device=None,
                 sam3_model_id: str = None,
                 cache_size: int = 10):
        import torch
        from transformers import Sam3Model, Sam3Processor

        from .device import get_device

        self._torch = torch
        self.device = device or get_device()

        model_id = sam3_model_id or os.getenv("SAM3_MODEL_ID", DEFAULT_SAM3_MODEL_ID)
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()

        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._cap = cache_size

    def _get_vision_features(self, image_pil, pixel_values):
        """Return SAM3 vision features, computing and caching on first call per image."""
        key = hashlib.sha256(image_pil.tobytes()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        with self._torch.no_grad():
            feats = self.model.get_vision_features(pixel_values)
        # Detach tensors inside the output dataclass so the computation graph is freed.
        from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput
        feats = Sam3VisionEncoderOutput(
            last_hidden_state=feats.last_hidden_state.detach() if feats.last_hidden_state is not None else None,
            pooler_output=feats.pooler_output.detach() if feats.pooler_output is not None else None,
            fpn_hidden_states=tuple(t.detach() for t in feats.fpn_hidden_states) if feats.fpn_hidden_states else None,
            fpn_position_encoding=tuple(t.detach() for t in feats.fpn_position_encoding) if feats.fpn_position_encoding else None,
        )
        self._cache[key] = feats
        while len(self._cache) > self._cap:
            self._cache.popitem(last=False)
        return feats

    def predict(self, image_pil, labels: List[str], score_threshold: float,
                iou_threshold: float, max_detections: int) -> List[Detection]:
        W, H = image_pil.size

        # Encode image once (or hit cache) — shared across all labels.
        img_inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
        vision_feats = self._get_vision_features(image_pil, img_inputs["pixel_values"])

        dets: List[Detection] = []
        for label in labels:
            text_inputs = self.processor(text=label, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                outputs = self.model(
                    vision_embeds=vision_feats,
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs.get("attention_mask"),
                )

            results = self.processor.post_process_instance_segmentation(
                outputs, threshold=score_threshold, target_sizes=[(H, W)]
            )[0]

            for score, box, mask in zip(results["scores"], results["boxes"], results["masks"]):
                x1, y1, x2, y2 = box.tolist()
                dets.append(Detection(
                    label=label,
                    score=float(score),
                    bbox=(x1, y1, x2, y2),
                    mask=mask.cpu().numpy().astype(bool),
                ))

        dets.sort(key=lambda d: d.score, reverse=True)
        return dets[:max_detections]


class OsamSam3Session(OpenVocabSession):
    """Legacy osam/SAM3 backend — CPU-only. Requires: pip install ".[osam]"."""

    def __init__(self, model_name: str = "sam3", cache_size: int = 10):
        try:
            import osam  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                'OPEN_VOCAB_BACKEND=osam requires osam. Install it with: '
                'pip install ".[osam]"'
            ) from exc
        self.model_name = model_name
        self._cache: "collections.OrderedDict[str, object]" = collections.OrderedDict()
        self._cap = cache_size

    def _remember(self, key, emb) -> None:
        self._cache[key] = emb
        self._cache.move_to_end(key)
        while len(self._cache) > self._cap:
            self._cache.popitem(last=False)

    def predict(self, image_pil, labels: List[str], score_threshold: float = 0.1,
                iou_threshold: float = 0.5, max_detections: int = 100) -> List[Detection]:
        import osam

        image_np = np.asarray(image_pil)
        H, W = image_np.shape[:2]
        image_id = hashlib.sha256(image_np.tobytes()).hexdigest()
        emb = self._cache.get(image_id)
        dets: List[Detection] = []

        for label in labels:                        # SAM3 = one text per call
            prompt = osam.types.Prompt(
                texts=[label],
                score_threshold=score_threshold,
                iou_threshold=iou_threshold,
                max_annotations=max_detections,
            )
            if emb is not None:
                req = osam.types.GenerateRequest(
                    model=self.model_name, image_embedding=emb, prompt=prompt)
            else:
                req = osam.types.GenerateRequest(
                    model=self.model_name, image=image_np, prompt=prompt)

            try:
                resp = osam.apis.generate(request=req)
            except Exception as exc:
                logger.warning("SAM3 generate failed for label '%s': %s", label, exc)
                continue

            if emb is None:
                emb = resp.image_embedding
                self._remember(image_id, emb)

            for ann in resp.annotations:
                if ann.bounding_box is None:
                    continue
                bb = ann.bounding_box
                dets.append(Detection(
                    label=label,
                    score=float(ann.score or 0.0),
                    bbox=(max(0, min(bb.xmin, W)), max(0, min(bb.ymin, H)),
                          max(0, min(bb.xmax, W)), max(0, min(bb.ymax, H))),
                    mask=_full_mask(ann, W, H),
                ))
        return dets
