"""Label Studio ML backend entry point — YOLO (known) + SAM3 (unknown) hybrid.

This is the module the Label Studio ML SDK loads (referenced by _wsgi.py).
It maps to the "server.py" role described in plan.md Phase 5.

Detection MVP: known classes -> YOLO boxes, unknown classes -> SAM3 boxes (via
bounding_box). Segmentation (PolygonLabels / BrushLabels) is additive later — the
converter already routes geometry by control tag.
"""
import logging
import os

import PIL.Image

# Load .env early so env vars are available before any os.getenv() below.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# The SDK reads LABEL_STUDIO_URL + LABEL_STUDIO_API_KEY for get_local_path().
# We authenticate via email/password (LABEL_STUDIO_EMAIL / LABEL_STUDIO_PASSWORD)
# and obtain a session token to satisfy the SDK's expectation.
def _inject_api_key_from_credentials():
    """If no API key is set, log in with email+password and set the env var."""
    if os.getenv("LABEL_STUDIO_API_KEY", "").strip():
        return
    email = os.getenv("LABEL_STUDIO_EMAIL", "").strip()
    password = os.getenv("LABEL_STUDIO_PASSWORD", "").strip()
    url = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080").rstrip("/")
    if not email or not password:
        return
    try:
        import requests
        s = requests.Session()
        s.get(f"{url}/user/login", timeout=5)
        csrf = s.cookies.get("csrftoken", "")
        s.post(f"{url}/user/login", data={"email": email, "password": password},
               headers={"X-CSRFToken": csrf, "Referer": f"{url}/user/login"}, timeout=5)
        r = s.get(f"{url}/api/users", timeout=5)
        # Retrieve the user's token from the profile endpoint
        r2 = s.get(f"{url}/api/current-user/reset-token/", timeout=5)
        if r2.ok:
            token = r2.json().get("token", "")
            if token:
                os.environ["LABEL_STUDIO_API_KEY"] = token
    except Exception:
        pass  # non-fatal: image URLs will fall back to direct download

_inject_api_key_from_credentials()

from label_studio_ml.model import LabelStudioMLBase

from backend.converters import detection_to_result
from backend.device import get_device
from backend.open_vocab import create_session
from backend.model_yolo import YoloSession
from backend.routing import SUPPORTED, build_routes, controls_for, labels_for

logger = logging.getLogger(__name__)

# Load heavy models once per process, regardless of how the SDK instantiates the
# backend class (it may construct one per request).
_SESSIONS: dict = {}


def _default_model_version() -> str:
    """Backend-aware version label shown on predictions, e.g. 'yolo+gdino-v1'."""
    return f"yolo+{os.getenv('OPEN_VOCAB_BACKEND', 'gdino')}-v1"


def _get_sessions() -> dict:
    if not _SESSIONS:
        # Open-vocab backend selected by OPEN_VOCAB_BACKEND (default gdino). The factory
        # drops kwargs a given backend doesn't accept, so this one uniform set works for
        # all of them (gdino uses device/model_id; osam uses model_name/cache_size).
        _SESSIONS["open_vocab"] = create_session(
            os.getenv("OPEN_VOCAB_BACKEND", "gdino"),
            device=get_device(),
            model_id=os.getenv("GDINO_MODEL_ID", "IDEA-Research/grounding-dino-base"),
            model_name=os.getenv("OSAM_MODEL", "sam3"),
            cache_size=int(os.getenv("EMBEDDING_CACHE_SIZE", "10")),
        )
        # YOLO is OPTIONAL. With no model file, every label is "unknown" and routes to
        # the open-vocab backend (text prompts) — so you can annotate without training.
        model_path = os.getenv("YOLO_MODEL_PATH", "").strip()
        if model_path and os.path.exists(model_path):
            _SESSIONS["yolo"] = YoloSession(
                model_path=model_path,
                device=get_device(),
                task=os.getenv("YOLO_TASK", "detect"),
            )
            _SESSIONS["yolo_classes"] = set(_SESSIONS["yolo"].class_names.values())
        else:
            _SESSIONS["yolo"] = None
            _SESSIONS["yolo_classes"] = set()
            logger.warning(
                "No YOLO model at YOLO_MODEL_PATH=%r — all labels are routed to the "
                "open-vocab backend (text prompts).", model_path,
            )
    return _SESSIONS


class AnnotationBackend(LabelStudioMLBase):

    def setup(self):
        self.set("model_version", os.getenv("MODEL_VERSION", _default_model_version()))

    def predict(self, tasks, context=None, **kwargs):
        sessions = _get_sessions()
        routes = build_routes(self.parsed_label_config, sessions["yolo_classes"])
        yolo_labels = labels_for(routes, "yolo")
        ov_labels = labels_for(routes, "open_vocab")

        yolo_conf = float(os.getenv("YOLO_SCORE_THRESHOLD", "0.3"))
        ov_conf = float(os.getenv("SAM3_SCORE_THRESHOLD", "0.1"))
        iou = float(os.getenv("IOU_THRESHOLD", "0.5"))
        max_det = int(os.getenv("MAX_DETECTIONS", "100"))
        model_version = self.get("model_version") or os.getenv("MODEL_VERSION", _default_model_version())

        predictions = []
        for task in tasks:
            image_pil = self._load_image(task)
            W, H = image_pil.size

            dets = []
            if yolo_labels and sessions["yolo"] is not None:
                dets += sessions["yolo"].predict(image_pil, yolo_labels, yolo_conf, iou)
            if ov_labels:
                dets += sessions["open_vocab"].predict(
                    image_pil, ov_labels, ov_conf, iou, max_det)

            results = [detection_to_result(d, c, W, H)
                       for d in dets for c in controls_for(routes, d.label)]
            score = sum(r["score"] for r in results) / len(results) if results else 0.0
            predictions.append({
                "model_version": model_version,
                "result": results,
                "score": score,
            })
        return predictions

    # --- helpers -----------------------------------------------------------

    def _image_value_key(self) -> str:
        """Data key of the <Image> object tag (e.g. 'image' from value='$image')."""
        for info in self.parsed_label_config.values():
            if info.get("type") in SUPPORTED:
                for inp in info.get("inputs", []):
                    if inp.get("type") == "Image":
                        return inp["value"]
        return "image"

    def _load_image(self, task) -> "PIL.Image.Image":
        url = task["data"][self._image_value_key()]
        try:
            path = self.get_local_path(url, task_id=task.get("id"))  # auth-aware (LS-hosted)
        except Exception as exc:  # plain public URL not served by LS
            logger.debug("get_local_path failed (%s); downloading directly", exc)
            path = self._download(url)
        return PIL.Image.open(path).convert("RGB")

    @staticmethod
    def _download(url: str) -> str:
        import tempfile

        import requests

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".img")
        tmp.write(resp.content)
        tmp.close()
        return tmp.name
