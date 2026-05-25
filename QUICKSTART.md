# Lock-In — Quickstart

Get from clone to live monitoring in under 10 minutes (assuming the pretrained model is published in releases).

## Prerequisites
- Python 3.10+
- A webcam
- macOS / Linux / Windows

## 1. Install

```bash
git clone https://github.com/adit-rah/lock-in.git
cd lock-in
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

## 2. Get a model

**Option A — pretrained from releases (fastest):**
```bash
python scripts/download_model.py
```
This fetches `distraction_classifier.pt` (~45 MB) from the v1.0.0 GitHub release into `models/`.

**Option B — train your own (45–90 min on Apple Silicon, longer on CPU):**
```bash
pip install kagglehub
# Accept the rules: https://www.kaggle.com/c/state-farm-distracted-driver-detection/rules
KAGGLE_DIR=$(python -c "import kagglehub; print(kagglehub.competition_download('state-farm-distracted-driver-detection'))")
python -m scripts.prepare_state_farm --kaggle_dir "$KAGGLE_DIR" --out_dir data/state_farm_binary
python -m src.train --data_dir data/state_farm_binary --config config.yaml
```

## 3. Run

**Dashboard (recommended):**
```bash
streamlit run src/dashboard.py
```
Click **Start** in the Live tab.

**Headless CLI:**
```bash
python -m src.app
```

Press Ctrl-C (CLI) or **Stop** (dashboard) to end a session. Logs are written to `data/focus_log.db` (SQLite) and `data/focus_log.csv`.

## Common issues

**Camera fails to open** — close other apps using the webcam. To test directly:
```bash
python -c "import cv2; c=cv2.VideoCapture(0); print('open:', c.isOpened()); c.release()"
```

**Inference slow** — confirm the right device is being picked:
```bash
python -c "from src.model import pick_device; print(pick_device())"
```
Should print `mps` on Apple Silicon, `cuda` on a CUDA Linux box, `cpu` otherwise.

**Predictions look inverted** — the trained model and `config.yaml` disagree on class order. Check that `config.classes` matches the `classes` field in `checkpoints/metrics.json`.

## Tuning

Edit `config.yaml`:
- `inference.frame_interval_seconds` — how often to capture (default 3)
- `scoring.rolling_window_size` — smoothing window length (default 10)
- `scoring.alert_threshold` — `S < threshold` → distracted (default 0.3)
- `scoring.consecutive_frames_required` — debounce length before alerting (default 3)

Or change them live with the sliders in the Streamlit Live tab — no restart needed.
