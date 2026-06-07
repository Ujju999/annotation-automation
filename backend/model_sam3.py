from __future__ import annotations

import collections
import hashlib
import logging
from typing import List

import numpy as np

from .converters import Detection
from .open_vocab import OpenVocabSession

logger = logging.getLogger(__name__)


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
    def __init__(self, model_name: str = "sam3", cache_size: int = 10):
        try:
            import osam  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                'OPEN_VOCAB_BACKEND=sam3 requires osam. Install it with: '
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
        # Cache the per-image embedding so the expensive encode runs once per image,
        # keyed on the pixel bytes (replaces the old caller-supplied image_id).
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

            if emb is None:                          # first call returns the embedding
                emb = resp.image_embedding
                self._remember(image_id, emb)

            for ann in resp.annotations:
                if ann.bounding_box is None:
                    continue
                bb = ann.bounding_box
                # osam can return coords past the image edges; clamp so the LS region
                # stays within [0, W] x [0, H].
                dets.append(Detection(
                    label=label,
                    score=float(ann.score or 0.0),
                    bbox=(max(0, min(bb.xmin, W)), max(0, min(bb.ymin, H)),
                          max(0, min(bb.xmax, W)), max(0, min(bb.ymax, H))),
                    mask=_full_mask(ann, W, H),
                ))
        return dets
