"""Legacy osam/SAM3 open-vocab backend (opt-in: pip install ".[osam]").

Kept for compatibility and its masks-capable path, but no longer the default — osam 0.4.x
runs SAM3 on ONNXRuntime CPU only (no GPU), so it is the slow path. Prefer the gdino /
yolo_world / grounded_sam2 / florence2 backends for GPU.

osam 0.4.0 reality this is built around:
  * SAM3 uses only prompt.texts[0]  -> one generate() call per label.
  * GenerateResponse returns image_embedding -> encode once, reuse across labels.
  * Annotation.mask is bbox-sized -> placed into a full-image canvas (segmentation phase).

osam is imported lazily so this module (and the factory that references it) loads even when
osam isn't installed; construction raises a clear install hint if it's missing.
"""
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
    """osam returns a bbox-sized mask; place it into a full-image canvas."""
    bb = ann.bounding_box
    full = np.zeros((H, W), dtype=bool)
    if ann.mask is not None and bb is not None:
        full[bb.ymin:bb.ymax + 1, bb.xmin:bb.xmax + 1] = ann.mask
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

            resp = osam.apis.generate(request=req)

            if emb is None:                          # first call returns the embedding
                emb = resp.image_embedding
                self._remember(image_id, emb)

            for ann in resp.annotations:
                if ann.bounding_box is None:
                    continue
                bb = ann.bounding_box
                dets.append(Detection(
                    label=label,
                    score=float(ann.score or 0.0),
                    bbox=(bb.xmin, bb.ymin, bb.xmax, bb.ymax),
                    mask=_full_mask(ann, W, H),
                ))
        return dets
