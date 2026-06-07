from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import PIL.Image
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from backend.converters import Detection, detection_to_result
from backend.open_vocab import OpenVocabSession, create_session
from backend.routing import RECT, POLY, Control

logging.basicConfig(level=logging.WARNING, format="%(message)s")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def prompt_classes(yolo_path: Optional[str]) -> tuple[list[str], list[str]]:
    """Ask the user which classes to annotate.  Returns (yolo_classes, ov_classes)."""
    yolo_classes: list[str] = []

    if yolo_path and Path(yolo_path).exists():
        from ultralytics import YOLO
        print(f"\nLoading YOLO model: {yolo_path}")
        m = YOLO(yolo_path)
        yolo_classes = [m.names[i] for i in sorted(m.names)]
        print("\nDetected YOLO classes:")
        for i, name in enumerate(yolo_classes):
            print(f"  {i}: {name}")
        print()
        raw = input("Add extra classes for open-vocab detection\n"
                    "(comma-separated, blank to skip): ").strip()
    else:
        if yolo_path:
            print(f"\nNo YOLO model found at '{yolo_path}' — using the open-vocab backend only.")
        else:
            print("\nNo YOLO model — using the open-vocab backend for all classes.")
        raw = input("Enter class names (comma-separated, e.g. drone,person,car): ").strip()

    ov_extra = [c.strip() for c in raw.split(",") if c.strip()] if raw else []
    return yolo_classes, ov_extra


def prompt_project_name(default: str) -> str:
    """Ask for the Label Studio project name, falling back to `default` on blank."""
    raw = input(f"\nProject name [{default}]: ").strip()
    return raw or default


def prompt_shape() -> str:
    """Ask for the annotation shape.  Returns RECT or POLY."""
    raw = input("\nAnnotation shape — [b]ox or [p]olygon? [box]: ").strip().lower()
    return POLY if raw in ("p", "poly", "polygon") else RECT


def prompt_recursive() -> bool:
    """Ask whether to scan subfolders for images."""
    raw = input("Include images in subfolders? [y/N]: ").strip().lower()
    return raw in ("y", "yes")


def prompt_float(label: str, default: float) -> float:
    """Ask for a numeric value, falling back to `default` on blank/invalid input."""
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  '{raw}' is not a number — using {default}.")
        return default


def collect_images(images_dir: Path, recursive: bool) -> list[Path]:
    """Gather image files from a folder, optionally recursing into subfolders."""
    it = images_dir.rglob("*") if recursive else images_dir.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def build_label_config(yolo_classes: list[str], ov_classes: list[str], shape: str = RECT) -> str:
    all_classes = list(dict.fromkeys(yolo_classes + ov_classes))
    labels_xml = "\n    ".join(f'<Label value="{c}"/>' for c in all_classes)
    return (
        "<View>\n"
        '  <Image name="img" value="$image"/>\n'
        f'  <{shape} name="label" toName="img">\n'
        f"    {labels_xml}\n"
        f"  </{shape}>\n"
        "</View>"
    )


class LSClient:
    def __init__(self, url: str, email: str, password: str):
        self.url = url.rstrip("/")
        self._s = requests.Session()
        self._login(email, password)

    def _csrf(self) -> str:
        return self._s.cookies.get("csrftoken", "")

    def _login(self, email: str, password: str):
        # Step 1: GET the login page to harvest the CSRF cookie
        self._s.get(f"{self.url}/user/login", timeout=10)
        # Step 2: POST credentials
        r = self._s.post(f"{self.url}/user/login", data={
            "email": email,
            "password": password,
        }, headers={"X-CSRFToken": self._csrf(), "Referer": f"{self.url}/user/login"},
        timeout=10, allow_redirects=True)
        # Verify login succeeded
        me = self._s.get(f"{self.url}/api/current-user/whoami", timeout=10)
        me.raise_for_status()
        name = me.json().get("username") or me.json().get("email", "?")
        print(f"Connected to Label Studio as: {name}")

    def _post(self, path: str, **kwargs):
        """POST with CSRF header automatically applied."""
        headers = kwargs.pop("headers", {})
        headers["X-CSRFToken"] = self._csrf()
        return self._s.post(f"{self.url}{path}", headers=headers, **kwargs)

    def create_project(self, name: str, label_config: str) -> int:
        r = self._post("/api/projects", json={
            "title": name,
            "label_config": label_config,
        }, timeout=15)
        r.raise_for_status()
        project_id = r.json()["id"]
        print(f"Created project '{name}'  →  {self.url}/projects/{project_id}/")
        return project_id

    def find_project(self, name: str) -> Optional[int]:
        """Return the id of an existing project whose title exactly matches, else None."""
        r = self._s.get(f"{self.url}/api/projects",
                        params={"page_size": 1000}, timeout=15)
        r.raise_for_status()
        payload = r.json()
        projects = payload.get("results", []) if isinstance(payload, dict) else payload
        for p in projects:
            if p.get("title") == name:
                return p["id"]
        return None

    def upload_image(self, project_id: int, image_path: Path) -> Optional[int]:
        """Upload one image file and return its LS task_id."""
        with open(image_path, "rb") as fh:
            r = self._post(
                f"/api/projects/{project_id}/import",
                files={"file": (image_path.name, fh, _mime(image_path))},
                timeout=60,
            )
        r.raise_for_status()

        # LS doesn't return task IDs from the import endpoint — query tasks sorted
        # by newest and match by filename.
        r2 = self._s.get(f"{self.url}/api/tasks", params={
            "project": project_id,
            "ordering": "-created_at",
            "page_size": 10,
        }, timeout=15)
        r2.raise_for_status()
        payload = r2.json()
        tasks = payload if isinstance(payload, list) else payload.get("tasks", [])

        stem = image_path.stem
        for task in tasks:
            for val in (task.get("data") or {}).values():
                if isinstance(val, str) and stem in val:
                    return task["id"]
        # fallback: assume the newest task is ours
        return tasks[0]["id"] if tasks else None

    def push_prediction(self, task_id: int, results: list, score: float, model_version: str):
        r = self._post("/api/predictions", json={
            "task": task_id,
            "result": results,
            "score": round(score, 4),
            "model_version": model_version,
        }, timeout=15)
        r.raise_for_status()


def _mime(p: Path) -> str:
    ext = p.suffix.lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "bmp": "image/bmp", "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")


def run_predictions(
    image_path: Path,
    yolo_classes: list[str],
    ov_classes: list[str],
    yolo_session,
    ov_session: Optional[OpenVocabSession],
    yolo_conf: float,
    ov_conf: float,
    iou: float,
    max_det: int,
    shape: str = RECT,
) -> tuple[list[dict], float, int, int]:
    """Return (ls_results, avg_score, n_yolo, n_ov)."""
    image_pil = PIL.Image.open(image_path).convert("RGB")
    W, H = image_pil.size
    control = Control(from_name="label", to_name="img", type=shape)

    yolo_dets: list[Detection] = []
    ov_dets: list[Detection] = []

    if yolo_classes and yolo_session is not None:
        yolo_dets = yolo_session.predict(image_pil, yolo_classes, yolo_conf, iou)

    if ov_classes and ov_session is not None:
        ov_dets = ov_session.predict(image_pil, ov_classes, ov_conf, iou, max_det)

    all_dets = yolo_dets + ov_dets
    results = [detection_to_result(d, control, W, H) for d in all_dets]
    score = sum(r["score"] for r in results) / len(results) if results else 0.0
    return results, score, len(yolo_dets), len(ov_dets)


def main():
    parser = argparse.ArgumentParser(
        description="Upload images to Label Studio and push YOLO + open-vocab bounding-box predictions."
    )
    parser.add_argument("--images", required=True, metavar="DIR",
                        help="Folder of images to process")
    parser.add_argument("--project", default=None, metavar="NAME",
                        help='Project name (default: "Annotation YYYY-MM-DD HH:MM")')
    parser.add_argument("--yolo", default=None, metavar="PATH",
                        help="Path to YOLO .pt file (overrides YOLO_MODEL_PATH in .env)")
    parser.add_argument("--shape", choices=["box", "polygon"], default=None,
                        help="Annotation shape (default: ask interactively)")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=None,
                        help="Scan subfolders for images (default: ask interactively)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        sys.exit(f"Error: '{images_dir}' is not a directory.")

    ls_url   = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
    ls_email = os.getenv("LABEL_STUDIO_EMAIL", "").strip()
    ls_pass  = os.getenv("LABEL_STUDIO_PASSWORD", "").strip()
    if not ls_email or not ls_pass:
        sys.exit(
            "Error: LABEL_STUDIO_EMAIL and LABEL_STUDIO_PASSWORD must be set in .env.\n"
            "Example:\n"
            "  LABEL_STUDIO_EMAIL=admin@example.com\n"
            "  LABEL_STUDIO_PASSWORD=yourpassword"
        )

    yolo_path = args.yolo or os.getenv("YOLO_MODEL_PATH", "").strip() or None
    backend = os.getenv("OPEN_VOCAB_BACKEND", "gdino")

    # --- interactive prompts ---
    yolo_classes, ov_classes = prompt_classes(yolo_path)
    all_class_names = list(dict.fromkeys(yolo_classes + ov_classes))
    if not all_class_names:
        sys.exit("No classes entered. Nothing to annotate.")

    default_name = f"Annotation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    project_name = args.project or prompt_project_name(default_name)

    shape = {"box": RECT, "polygon": POLY}[args.shape] if args.shape else prompt_shape()
    if shape == POLY:
        print("  Note: polygons need a mask-capable backend (sam3 / grounded_sam2);\n"
              "  box-only backends fall back to box-shaped polygons.")

    recursive = args.recursive if args.recursive is not None else prompt_recursive()
    image_files = collect_images(images_dir, recursive)
    if not image_files:
        hint = "or its subfolders" if recursive else "top level only — try --recursive"
        sys.exit(f"No images found in '{images_dir}' ({hint}). "
                 f"Supported: {', '.join(sorted(IMAGE_EXTS))}")

    yolo_conf = float(os.getenv("YOLO_SCORE_THRESHOLD", "0.3"))
    ov_conf   = float(os.getenv("SAM3_SCORE_THRESHOLD", "0.1"))
    if yolo_classes:
        yolo_conf = prompt_float("YOLO confidence threshold", yolo_conf)
    if ov_classes:
        ov_conf = prompt_float(f"'{backend}' confidence threshold", ov_conf)

    label_config = build_label_config(yolo_classes, ov_classes, shape)

    # --- summary ---
    print(f"\nProject:  {project_name}")
    print(f"Shape:    {'polygon' if shape == POLY else 'box'}")
    print(f"Classes:  {', '.join(all_class_names)}")
    print(f"Images:   {len(image_files)} files in {images_dir}{' (recursive)' if recursive else ''}")
    if yolo_classes:
        print(f"  YOLO → {', '.join(yolo_classes)}  (conf ≥ {yolo_conf})")
    if ov_classes:
        print(f"  {backend} → {', '.join(ov_classes)}  (conf ≥ {ov_conf})")
    print()
    ans = input("Proceed? [Y/n] ").strip().lower()
    if ans == "n":
        sys.exit("Aborted.")

    # --- connect to Label Studio ---
    try:
        client = LSClient(ls_url, ls_email, ls_pass)
    except Exception as exc:
        sys.exit(f"Cannot connect to Label Studio at {ls_url}: {exc}")

    # --- create or reuse the project ---
    existing = client.find_project(project_name)
    if existing is not None:
        print(f"\nA project named '{project_name}' already exists (id {existing}).")
        choice = input("  [R]euse (add these images) / [C]reate a new one / [A]bort? [R]: ").strip().lower()
        if choice == "a":
            sys.exit("Aborted.")
        if choice.startswith("c"):
            project_id = client.create_project(project_name, label_config)
        else:
            project_id = existing
            print(f"Reusing project {existing} — keeping its labeling config; new images will be added.")
    else:
        project_id = client.create_project(project_name, label_config)

    # --- init model sessions ---
    yolo_session = None
    if yolo_classes and yolo_path and Path(yolo_path).exists():
        from backend.model_yolo import YoloSession
        from backend.device import get_device
        yolo_session = YoloSession(
            model_path=yolo_path,
            device=get_device(),
            task=os.getenv("YOLO_TASK", "detect"),
        )

    ov_session = None
    if ov_classes:
        print(f"Loading open-vocab backend '{backend}' (first image may be slow while weights load)...")
        from backend.device import get_device
        ov_session = create_session(
            backend,
            device=get_device(),
            model_id=os.getenv("GDINO_MODEL_ID", "IDEA-Research/grounding-dino-base"),
            model_name=os.getenv("OSAM_MODEL", "sam3"),
            cache_size=int(os.getenv("EMBEDDING_CACHE_SIZE", "10")),
        )

    iou        = float(os.getenv("IOU_THRESHOLD", "0.5"))
    max_det    = int(os.getenv("MAX_DETECTIONS", "100"))
    model_ver  = os.getenv("MODEL_VERSION") or f"yolo+{backend}-v1"

    # --- process images ---
    print(f"\nProcessing {len(image_files)} images...\n")
    ok = skipped = errored = 0

    for i, img_path in enumerate(image_files, 1):
        tag = f"[{i}/{len(image_files)}] {img_path.name}"
        try:
            task_id = client.upload_image(project_id, img_path)
            if task_id is None:
                print(f"  {tag}  →  upload ok but task ID not found (skipped prediction)")
                skipped += 1
                continue

            results, score, n_yolo, n_ov = run_predictions(
                img_path, yolo_classes, ov_classes,
                yolo_session, ov_session,
                yolo_conf, ov_conf, iou, max_det,
                shape=shape,
            )

            if results:
                client.push_prediction(task_id, results, score, model_ver)
                parts = []
                if n_yolo:
                    parts.append(f"YOLO:{n_yolo}")
                if n_ov:
                    parts.append(f"{backend}:{n_ov}")
                detail = f"  ({', '.join(parts)})" if parts else ""
                print(f"  {tag}  →  {len(results)} boxes  score={score:.2f}{detail}  ✓")
                ok += 1
            else:
                print(f"  {tag}  →  no detections  (task created, no pre-annotation)")
                skipped += 1

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as exc:
            print(f"  {tag}  →  ERROR: {exc}")
            errored += 1

    print(f"\n{'─'*50}")
    print(f"Done.  {ok} pre-annotated  |  {skipped} empty  |  {errored} errors")
    print(f"Review at: {ls_url}/projects/{project_id}/")


if __name__ == "__main__":
    main()
