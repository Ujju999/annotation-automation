# annotation-automation

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

A Label Studio ML backend that automatically pre-annotates images using a hybrid of a custom-trained **YOLO** model and **SAM3** (via [osam](https://github.com/wkentaro/osam)).

- **YOLO** handles classes the model was trained on — fast, high-confidence predictions.
- **SAM3** handles any class YOLO doesn't know — open-vocabulary, text-prompted. No training required.
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
        ├─► Labels in YOLO model.names ──► YOLO inference  ──► rectangles / polygons
        │
        └─► Labels NOT in YOLO          ──► SAM3 text prompt ──► polygons
        │
        ▼
Merged predictions returned to Label Studio
(remaining labels available for manual annotation)
```

YOLO is **optional** — with no weights file, every label routes to SAM3 so you can start annotating brand-new classes immediately.

---

## Requirements

- Python **3.11+** (onnxruntime dropped 3.10 wheels)
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

### 3. Pull the SAM3 model weights

```bash
.venv/bin/osam pull sam3
```

This downloads ~300 MB once; subsequent runs reuse the cached weights.

### 4. Install Label Studio (separate venv — its deps conflict)

```bash
python3 -m venv ~/.ls-venv
source ~/.ls-venv/bin/activate
pip install label-studio
label-studio start          # http://localhost:8080
```

Create an account when prompted.

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

# Leave empty to use SAM3-only (no YOLO model needed)
YOLO_MODEL_PATH=
```

Full `.env` reference:

| Variable | Default | Description |
|---|---|---|
| `YOLO_MODEL_PATH` | *(empty)* | Path to your YOLO `.pt` weights. Empty = SAM3-only. |
| `YOLO_TASK` | `detect` | `detect` (bounding boxes) or `segment` (masks → polygons) |
| `OSAM_MODEL` | `sam3` | osam model: `sam3` (masks-capable) or `yoloworld` (boxes, faster) |
| `YOLO_SCORE_THRESHOLD` | `0.3` | Minimum YOLO confidence to include a prediction |
| `SAM3_SCORE_THRESHOLD` | `0.1` | Minimum SAM3 confidence (lower than YOLO) |
| `IOU_THRESHOLD` | `0.5` | NMS IoU threshold |
| `MAX_DETECTIONS` | `100` | Maximum predictions per image |
| `EMBEDDING_CACHE_SIZE` | `10` | Number of SAM3 image embeddings to keep in memory |
| `MODEL_VERSION` | `yolo+sam3-v1` | Label shown on predictions in Label Studio |
| `PORT` | `9090` | Backend server port |
| `LABEL_STUDIO_URL` | `http://localhost:8080` | Label Studio base URL |
| `LABEL_STUDIO_EMAIL` | — | Your Label Studio login email |
| `LABEL_STUDIO_PASSWORD` | — | Your Label Studio login password |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | Apple Silicon: allow unsupported MPS ops to fall back to CPU |

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
No YOLO model — using SAM3 open-vocab for all classes.
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
  [1/12] img_001.jpg  →  2 boxes  score=0.86 (SAM3:2)  ✓
  [2/12] img_002.jpg  →  1 box   score=0.74 (SAM3:1)  ✓
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

When `--yolo` is set the CLI shows the model's classes and lets you add extra SAM3 classes on top:

```
Detected YOLO classes:
  0: cardbox
  1: crate

Add extra classes for SAM3 open-vocab (blank to skip): person
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

> First prediction takes ~25 seconds (SAM3 on CPU). Subsequent images on the same task reuse the cached embedding.

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

SAM3 always runs on CPU via ONNXRuntime regardless of device. Override YOLO's device with the `DEVICE` env var:

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
│   ├── routing.py        # Label → engine routing (YOLO vs SAM3, per control tag)
│   ├── model_yolo.py     # YOLO inference → Detection objects
│   ├── model_sam3.py     # SAM3 inference → Detection objects (text prompt)
│   ├── converters.py     # Detection → Label Studio JSON (rectanglelabels / polygonlabels)
│   └── device.py         # CUDA / MPS / CPU autodetect
├── tests/
│   ├── test_routing.py
│   └── test_converters.py
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
| `onnxruntime … no wheel for cp310` | Python 3.10 not supported. `uv venv --python 3.11 && uv sync`. |
| First prediction very slow (~25s) | Expected — SAM3 encodes the image on CPU. Subsequent images hit the embedding cache. |
| SAM3 gives poor boxes | Try a clearer label word, or lower `SAM3_SCORE_THRESHOLD` in `.env`. |

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
  title   = {annotation-automation: {YOLO} + {SAM3} hybrid {Label Studio} {ML} backend},
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
- [osam](https://github.com/wkentaro/osam) — local runtime for SAM/SAM2/SAM3/YOLO-World
- [Ultralytics YOLO](https://docs.ultralytics.com)
