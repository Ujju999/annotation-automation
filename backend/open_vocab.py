from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from typing import List

from .converters import Detection


class OpenVocabSession(ABC):
    """Interface every open-vocab backend implements.

    Canonical, PIL-based signature. Unlike the old osam path there is no ``image_id``:
    backends that cache (osam) derive their own key from the image internally.
    """

    @abstractmethod
    def predict(self, image_pil, labels: List[str], score_threshold: float,
                iou_threshold: float, max_detections: int) -> List[Detection]:
        """Detect each text label in ``image_pil``.

        Returns Detection objects (pixel bbox + optional full-image mask). Backends
        that produce no masks leave ``Detection.mask`` as None; the converter falls
        back to a bbox polygon for PolygonLabels controls.
        """
        ...


# Backend name -> lazy loader returning the backend CLASS. The import happens only
# when the loader is called (inside create_session), never at module import time.
_BACKENDS = {
    "gdino": lambda: _load("model_gdino", "GdinoSession"),
    "yolo_world": lambda: _load("model_yolo_world", "YoloWorldSession"),
    "grounded_sam2": lambda: _load("model_grounded_sam2", "GroundedSam2Session"),
    # Legacy osam/SAM3 — kept as an opt-in backend (pip install ".[osam]").
    "sam3": lambda: _load("model_sam3", "Sam3Session"),
    "osam": lambda: _load("model_sam3", "Sam3Session"),
}

DEFAULT_BACKEND = "gdino"


def _load(module: str, cls_name: str):
    mod = importlib.import_module(f"{__package__}.{module}")
    return getattr(mod, cls_name)


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Keep only the kwargs ``cls.__init__`` actually accepts.

    Lets model.py pass one superset of constructor args (device, cache_size,
    model_name, ...) to any backend without ``unexpected keyword argument`` errors.
    """
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    allowed = {name for name in params if name != "self"}
    return {k: v for k, v in kwargs.items() if k in allowed}


def create_session(backend: str, **kwargs) -> OpenVocabSession:
    """Construct the open-vocab backend named by ``backend`` (OPEN_VOCAB_BACKEND).

    Extra kwargs not used by the chosen backend's constructor are dropped, so the
    caller can pass a uniform set (device, cache_size, model_name, ...).
    """
    key = (backend or DEFAULT_BACKEND).strip().lower()
    loader = _BACKENDS.get(key)
    if loader is None:
        raise ValueError(
            f"Unknown OPEN_VOCAB_BACKEND={backend!r}. Expected one of: "
            + ", ".join(sorted(_BACKENDS))
        )
    cls = loader()
    return cls(**_filter_kwargs(cls, kwargs))
