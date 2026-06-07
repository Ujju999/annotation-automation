# annotation-automation

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

A Label Studio ML backend that automatically pre-annotates images using a hybrid of a custom-trained **YOLO** model and a **pluggable open-vocabulary backend** (default: **Grounding DINO**, GPU-native).

- **YOLO** handles classes the model was trained on — fast, high-confidence predictions.
- The **open-vocab backend** handles any class YOLO doesn't know — text-prompted, no training required. Swap it with one env var (`OPEN_VOCAB_BACKEND`): `gdino`, `grounded_sam2`, `yolo_world`, or legacy `sam3`. See [Open-vocab backends](#open-vocab-backends).
- The split is fully dynamic: no config changes needed when you add or remove classes in Label Studio.

There are two ways to use it:

| Workflow | Best for |
|---|---|
| **[A] CLI import** | Batch-import a folder of images, push predictions, open Label Studio to review |
| **[B] Live backend** | Predictions appear automatically as annotators open each task |

---

## How it works

```
Label Studio calls /predict
        │
        ▼
Parse label config → get all label names
        │
        ├─► Labels in YOLO model.names ──► YOLO inference        ──► rectangles / polygons
        │
        └─► Labels NOT in YOLO          ──► open-vocab text prompt ──► boxes / polygons
        │
        ▼
Merged predictions returned to Label Studio
(remaining labels available for manual annotation)
```

YOLO is **optional** — with no weights file, every label routes to the open-vocab backend so you can start annotating brand-new classes immediately.

---

## Requirements

- Python **3.11+**
- PyTorch (installed separately so you pick the right build — see below)
- A running [Label Studio](https://labelstud.io) instance
- *(Optional)* A trained YOLO `.pt` weights file

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/annotation-automation.git
cd annotation-automation
```

### 2. Create a virtual environment

**With [uv](https://docs.astral.sh/uv/) (recommended):**

```bash
uv venv --python 3.11
uv sync
```

**With pip:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate

# Install PyTorch first — pick your hardware:
# NVIDIA (CUDA):
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Apple Silicon / CPU:
pip install torch

pip install -e .
```

### 3. (Nothing to pull for the default backend)

The default open-vocab backend, **Grounding DINO**, downloads its weights (~700 MB) from
Hugging Face automatically on the first prediction — no manual step. To use a different
backend, see [Open-vocab backends](#open-vocab-backends) (some need an extra install).

### 4. Install Label Studio (separate venv — its deps conflict)

```bash
python3 -m venv ~/.ls-venv
source ~/.ls-venv/bin/activate
pip install label-studio
label-studio start          # http://localhost:8080
```

Create an account when prompted.

---

## Open-vocab backends

The engine that handles classes YOLO doesn't know is pluggable — set `OPEN_VOCAB_BACKEND`
in `.env`. All run on the autodetected GPU (CUDA/MPS) except legacy `sam3`.

| Backend | Output | Speed | Install | Notes |
|---|---|---|---|---|
| **`gdino`** *(default)* | boxes | fast | core | Grounding DINO. Best general-purpose; weights ~700 MB auto-download. |
| **`grounded_sam2`** | boxes **+ masks** | medium | core | Grounding DINO + SAM 2. Use for `PolygonLabels`. Highest quality, heaviest load. |
| **`yolo_world`** | boxes | fastest | `pip install ".[yolo_world]"` | YOLO-World. Needs CLIP. Scores run low — lower `SAM3_SCORE_THRESHOLD`. |
| **`sam3`** *(legacy)* | boxes + masks | slow (CPU) | `pip install ".[osam]"` | osam/SAM3. CPU-only (~25 s/first image); kept for compatibility. |

```bash
# Examples
OPEN_VOCAB_BACKEND=gdino           # default, nothing to install
OPEN_VOCAB_BACKEND=grounded_sam2   # boxes + polygon masks
OPEN_VOCAB_BACKEND=yolo_world      # after: pip install ".[yolo_world]"
OPEN_VOCAB_BACKEND=sam3            # after: pip install ".[osam]"
```

> Florence-2 is **not** currently supported — its published checkpoint is incompatible
> with the transformers version Grounding DINO and SAM 2 require. Use `grounded_sam2` for
> GPU boxes + masks.

---

## Configuration

Copy `.env.example` to `.env` and fill in your Label Studio credentials:

```bash
cp .env.example .env
```

```bash
# Minimum required settings
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_EMAIL=you@example.com
LABEL_STUDIO_PASSWORD=yourpassword

# Open-vocab backend for classes YOLO doesn't know (see "Open-vocab backends")
OPEN_VOCAB_BACKEND=gdino

# Leave empty to run open-vocab-only (no YOLO model needed)
YOLO_MODEL_PATH=
```

Full `.env` reference:

| Variable | Default | Description |
|---|---|---|
| `OPEN_VOCAB_BACKEND` | `gdino` | Open-vocab engine: `gdino`, `grounded_sam2`, `yolo_world`, `sam3`. See [Open-vocab backends](#open-vocab-backends). |
| `YOLO_MODEL_PATH` | *(empty)* | Path to your YOLO `.pt` weights. Empty = open-vocab-only. |
| `YOLO_TASK` | `detect` | `detect` (bounding boxes) or `segment` (masks → polygons) |
| `GDINO_MODEL_ID` | `IDEA-Research/grounding-dino-base` | Grounding DINO checkpoint (`gdino` / `grounded_sam2`) |
| `SAM2_MODEL_ID` | `facebook/sam2-hiera-large` | SAM 2 checkpoint (`grounded_sam2`) |
| `YOLO_WORLD_MODEL` | `yolov8x-worldv2.pt` | YOLO-World weights (`yolo_world`) |
| `OSAM_MODEL` | `sam3` | osam model when `OPEN_VOCAB_BACKEND=sam3` |
| `YOLO_SCORE_THRESHOLD` | `0.3` | Minimum YOLO confidence to include a prediction |
| `SAM3_SCORE_THRESHOLD` | `0.1` | Minimum open-vocab confidence (used by all open-vocab backends) |
| `IOU_THRESHOLD` | `0.5` | NMS IoU threshold |
| `MAX_DETECTIONS` | `100` | Maximum predictions per image |
| `EMBEDDING_CACHE_SIZE` | `10` | Image-embedding cache size (`sam3` only) |
| `MODEL_VERSION` | `yolo+<backend>-v1` | Label shown on predictions in Label Studio |
| `PORT` | `9090` | Backend server port |
| `LABEL_STUDIO_URL` | `http://localhost:8080` | Label Studio base URL |
| `LABEL_STUDIO_EMAIL` | — | Your Label Studio login email |
| `LABEL_STUDIO_PASSWORD` | — | Your Label Studio login password |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | Apple Silicon: allow unsupported MPS ops to fall back to CPU |

> The open-vocab confidence floor is still read from `SAM3_SCORE_THRESHOLD` (name kept for
> backward compatibility). YOLO-World scores run low — try `0.05` if it misses objects.

---

## Workflow A — CLI import (recommended for batches)

One command creates the project, uploads images, runs predictions, and pushes annotations. Annotators open Label Studio and start reviewing immediately.

**Step 1 — Start Label Studio:**

```bash
source ~/.ls-venv/bin/activate && label-studio start
```

**Step 2 — Run the import:**

```bash
.venv/bin/python import_images.py --images /path/to/your/images
```

The CLI prompts for classes interactively:

```
No YOLO model — using the open-vocab backend for all classes.
Enter class names (comma-separated, e.g. drone,person,car): drone

Project:  Annotation 2026-06-07 14:30
Classes:  drone
Images:   12 files in /path/to/your/images

Proceed? [Y/n]
```

After confirming:

```
Connected to Label Studio as: you@example.com
Created project 'Annotation 2026-06-07 14:30'  →  http://localhost:8080/projects/3/

Processing 12 images...
  [1/12] img_001.jpg  →  2 boxes  score=0.46 (gdino:2)  ✓
  [2/12] img_002.jpg  →  1 box   score=0.39 (gdino:1)  ✓
  ...

Done. 11 pre-annotated | 1 empty | 0 errors
Review at: http://localhost:8080/projects/3/
```

**Step 3 — Review in Label Studio:**

Open the printed URL. Each task has bounding boxes as predictions:

- **Correct** → drag corners / reposition.
- **Missing detection** → draw one manually.
- **Wrong detection** → select and delete.
- **Submit** → accepts the annotation.

Export anytime via **Settings → Export** (COCO, YOLO, JSON, …).

### CLI options

```bash
# Custom project name
.venv/bin/python import_images.py --images /path/to/images --project "Drone dataset v1"

# Use a trained YOLO model
.venv/bin/python import_images.py --images /path/to/images --yolo /path/to/best.pt
```

When `--yolo` is set the CLI shows the model's classes and lets you add extra open-vocab classes on top:

```
Detected YOLO classes:
  0: cardbox
  1: crate

Add extra classes for open-vocab detection (blank to skip): person
```

---

## Workflow B — Live backend

Use this when you want predictions to appear automatically as annotators open tasks, or when images are uploaded directly in the Label Studio UI.

**Step 1 — Start Label Studio (terminal 1):**

```bash
source ~/.ls-venv/bin/activate && label-studio start
```

**Step 2 — Start the backend (terminal 2):**

```bash
.venv/bin/python _wsgi.py
```

Verify it's up:

```bash
curl http://localhost:9090/health
# {"model_class":"AnnotationBackend","status":"UP"}
```

**Step 3 — Create a project with a labeling config:**

1. `http://localhost:8080` → **Create Project** → name it.
2. **Labeling setup → Custom template** → paste and save:

   ```xml
   <View>
     <Image name="img" value="$image"/>
     <RectangleLabels name="label" toName="img">
       <Label value="drone"/>
     </RectangleLabels>
   </View>
   ```

   Add more `<Label value="..."/>` lines for more classes.

**Step 4 — Upload images:**

Project → **Import** → drag and drop your image files.

**Step 5 — Connect the backend:**

Project **Settings → Model → Connect Model**:
- **URL:** `http://localhost:9090`
- Save. Label Studio calls `/health`; the model should show **Connected**.
- *(Optional)* enable **"Retrieve predictions when loading a task automatically"**.

**Step 6 — Generate predictions:**

- **Batch:** Data Manager → select tasks → **Actions → Retrieve Predictions**.
- **Per task:** annotators open a task (auto-retrieve must be enabled).

> The first prediction is slower while the backend's weights load/download; subsequent images are fast. (Legacy `sam3` stays slow — it's CPU-only.)

---

## Docker

### NVIDIA (CUDA)

```bash
docker build -f docker/Dockerfile.cuda -t annotation-automation:cuda .
docker run --gpus all -p 9090:9090 \
  -v /path/to/best.pt:/app/best.pt \
  -e YOLO_MODEL_PATH=/app/best.pt \
  -e LABEL_STUDIO_URL=http://host.docker.internal:8080 \
  -e LABEL_STUDIO_EMAIL=you@example.com \
  -e LABEL_STUDIO_PASSWORD=yourpassword \
  annotation-automation:cuda
```

### CPU

```bash
docker build -f docker/Dockerfile.cpu -t annotation-automation:cpu .
docker run -p 9090:9090 \
  -v /path/to/best.pt:/app/best.pt \
  -e YOLO_MODEL_PATH=/app/best.pt \
  -e LABEL_STUDIO_URL=http://host.docker.internal:8080 \
  -e LABEL_STUDIO_EMAIL=you@example.com \
  -e LABEL_STUDIO_PASSWORD=yourpassword \
  annotation-automation:cpu
```

> **Apple Silicon + Docker:** MPS is not available inside Docker containers. Run the backend natively instead.

---

## Hardware auto-detection

The backend picks the best available device automatically at startup:

| Hardware | Device used |
|---|---|
| NVIDIA GPU | `cuda` |
| Apple Silicon (M1/M2/M3/M4) | `mps` |
| Everything else | `cpu` |

The default `gdino` backend (and `grounded_sam2`, `yolo_world`) run on the same device.
The legacy `sam3` backend always runs on CPU via ONNXRuntime regardless of device.
Override YOLO's device with the `DEVICE` env var:

```bash
DEVICE=cpu .venv/bin/python _wsgi.py
```

---

## Running tests

```bash
.venv/bin/pytest tests/ -v
```

---

## Project structure

```
annotation-automation/
├── model.py              # Label Studio ML backend entry point (AnnotationBackend)
├── _wsgi.py              # Flask app wrapper — what label-studio-ml start runs
├── import_images.py      # CLI: batch import images + push predictions
├── pyproject.toml
├── .env.example
├── backend/
│   ├── routing.py             # Label → engine routing (YOLO vs open-vocab, per control tag)
│   ├── open_vocab.py          # OpenVocabSession interface + create_session() factory
│   ├── model_gdino.py         # Grounding DINO backend (default)
│   ├── model_grounded_sam2.py # Grounding DINO + SAM 2 (boxes + masks)
│   ├── model_yolo_world.py    # YOLO-World backend
│   ├── model_sam3.py          # Legacy osam/SAM3 backend (opt-in)
│   ├── model_yolo.py          # Trained-YOLO (known-class) inference → Detection objects
│   ├── converters.py          # Detection → Label Studio JSON (rectanglelabels / polygonlabels)
│   └── device.py              # CUDA / MPS / CPU autodetect
├── tests/
│   ├── test_routing.py
│   ├── test_converters.py
│   ├── test_open_vocab.py
│   └── test_gdino_match.py
├── docker/
│   ├── Dockerfile.cuda
│   └── Dockerfile.cpu
└── docs/
    ├── usage.md          # Extended workflow guide
    └── how-it-works.md   # Architecture deep-dive
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `LABEL_STUDIO_EMAIL and LABEL_STUDIO_PASSWORD must be set` | Fill in both vars in `.env`. |
| `Cannot connect to Label Studio` | Label Studio not running, or wrong URL. `curl http://localhost:8080`. |
| Login fails / 403 | Wrong email or password in `.env`. |
| Backend won't connect in Settings → Model | Backend not running. `curl http://localhost:9090/health`. Use `http://` not `https://`. |
| `No module named 'redis'` on backend start | Wrong `label-studio-ml` installed. Re-run `uv sync`. |
| `Unknown OPEN_VOCAB_BACKEND=...` | Typo, or `florence2` (unsupported). Use `gdino`, `grounded_sam2`, `yolo_world`, or `sam3`. |
| `... requires CLIP` (yolo_world) | `pip install ".[yolo_world]"`. |
| `... requires osam` (sam3) | `pip install ".[osam]"`. |
| First prediction slow | Expected — the backend downloads/loads weights once. Legacy `sam3` stays slow (CPU-only). |
| Open-vocab misses objects / poor boxes | Try a clearer label word, or lower `SAM3_SCORE_THRESHOLD` (try `0.05` for `yolo_world`). |

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first —
it covers the bug report format, PR process, and code style.

First-time contributors must sign the [Contributor License Agreement](CLA.md) by
adding one sentence to their pull request (details in both files).

---

## Citation

If you use this project in research or build on it in a publication, please cite it:

```bibtex
@software{singh2026annotationautomation,
  author  = {Singh, Ujjawalpratap},
  title   = {annotation-automation: {YOLO} + open-vocabulary hybrid {Label Studio} {ML} backend},
  year    = {2026},
  url     = {https://github.com/Ujju999/annotation-automation},
  license = {AGPL-3.0}
}
```

A [CITATION.cff](CITATION.cff) file is also included — GitHub renders it as a
**"Cite this repository"** button on the repo homepage.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see [LICENSE](LICENSE).

This license applies because [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (a dependency) is AGPL-3.0. Under AGPL-3.0:

- You can use, modify, and distribute this code freely.
- Any modified version you distribute must also be released under AGPL-3.0.
- If you run a modified version as a network service, you must make the source code available to users.

If you need to use this project in a **closed-source** product, you will need a [commercial license from Ultralytics](https://ultralytics.com/license) for the YOLO component.

---

## References

- [Label Studio ML SDK](https://github.com/HumanSignal/label-studio-ml-backend)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) — default open-vocab backend
- [SAM 2](https://github.com/facebookresearch/sam2) — masks for `grounded_sam2`
- [Ultralytics YOLO / YOLO-World](https://docs.ultralytics.com)
- [osam](https://github.com/wkentaro/osam) — legacy `sam3` backend runtime
