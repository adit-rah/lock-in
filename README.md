# Lock-In

Real-time focus monitor. A ResNet18 fine-tuned on the State Farm distracted-driver dataset (~22K images) classifies focused vs. distracted from your webcam; a rolling-window scorer smooths predictions; a Streamlit dashboard shows the live score and past sessions backed by SQLite.

All inference runs **on-device** in PyTorch + OpenCV. No frames leave your machine.

## What's inside

- **Model** — ResNet18 (`torchvision`) with a binary head (`focused` / `distracted`), fine-tuned on State Farm. Achieves **0.88 macro F1** on a driver-disjoint validation split (per-class F1: distracted 0.97, focused 0.80; focused recall 0.86).
- **Inference pipeline** — `cv2.VideoCapture` with `CAP_PROP_BUFFERSIZE=1` and a grab/retrieve flush so the model always sees the freshest frame. TorchScript model for low-latency forward passes (<300 ms on CPU).
- **Temporal smoothing** — Rolling window of N predictions; lock-in score `S = P(focused) - P(distracted)`. Alerts fire only on sustained drops below threshold (debounced by consecutive-frame count).
- **Streamlit dashboard** — Live tab (webcam preview, score gauge, rolling chart, settings sliders) + History tab (SQLite-backed session list + drilldowns).
- **SQLite logging** — Predictions, scores, and events persisted per-session; predictions are batched into `executemany` flushes (`config.logging.batch_size`) to keep write overhead off the inference path.
- **CLI mode** — Headless `python -m src.app` for users who don't want a browser tab.

## Quick start

```bash
git clone https://github.com/adit-rah/lock-in.git
cd lock-in
pip install -e .

# Either fetch the pretrained release checkpoint...
python scripts/download_model.py

# ...or train from scratch (see "Training" below).

# Launch the dashboard:
streamlit run src/dashboard.py

# Or run headless:
python -m src.app
```

Press Ctrl-C (CLI) or click "Stop" (dashboard) to end the session — the SQLite database is flushed automatically.

## Training

Lock-In ships with a pipeline targeting [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) (~4 GB, requires a Kaggle account).

1. Download and extract the archive.
2. Build the binary, driver-disjoint dataset:
   ```bash
   python scripts/prepare_state_farm.py \
       --kaggle_dir /path/to/state-farm-distracted-driver-detection \
       --out_dir data/state_farm_binary
   ```
   `c0` (safe driving) becomes `focused`; `c1`–`c9` become `distracted`. The split is by **subject**, not by image — required for an honest F1 since consecutive frames of the same driver are nearly identical.
3. Train:
   ```bash
   python -m src.train --data_dir data/state_farm_binary --config config.yaml
   ```
   At the end you'll get `models/distraction_classifier.pt` (TorchScript) and `checkpoints/metrics.json` (macro F1, per-class precision/recall/F1, confusion matrix).

State Farm is ~10% focused / 90% distracted after binarization. `config.training.use_class_balanced_sampler: true` (the default) wraps training in a `WeightedRandomSampler` to compensate.

## Configuration

All knobs live in `config.yaml`. Key settings:

```yaml
model:
  architecture: resnet18    # also supports resnet34, mobilenet_v3_small/large
  num_classes: 2

inference:
  frame_interval_seconds: 3
  max_inference_time_ms: 300

scoring:
  rolling_window_size: 10
  alert_threshold: 0.3       # alert when S < this
  consecutive_frames_required: 3

logging:
  batch_size: 10             # SQLite write batching
```

## How the lock-in score works

```
S = mean(P(focused)) - mean(P(distracted))
```

Averaged over the rolling window. `S > threshold` → "locked in"; `S < threshold` for `consecutive_frames_required` frames → alert.

## Repository layout

```
lock-in/
├── src/
│   ├── app.py           # CLI entry point
│   ├── dashboard.py     # Streamlit app
│   ├── train.py         # State Farm training + F1 reporting
│   ├── inference.py     # InferenceEngine (cv2 + TorchScript)
│   ├── scoring.py       # FocusScorer (rolling window)
│   ├── logging_db.py    # SQLite + CSV persistence
│   ├── signals.py       # Desktop notifications
│   ├── config.py        # Dataclass-backed YAML config
│   └── model.py         # ResNet/MobileNet backbones
├── scripts/
│   ├── prepare_state_farm.py    # Kaggle -> binary, driver-disjoint
│   ├── download_model.py        # Fetches release checkpoint
│   ├── reorganize_statefarm.py  # (legacy) 3-class mapping
│   ├── capture_samples.py       # Personal data capture
│   └── extract_yawdd_frames.py  # YawDD helper
├── tests/
├── docs/                # ARCHITECTURE.md, dataset notes
├── config.yaml
└── setup.py
```

## Performance

Reported on the driver-disjoint val split of State Farm (3846 images, 22 unique drivers held out from training):

| Metric | Value |
|---|---|
| Macro F1 | **0.884** |
| Accuracy | 94.8% |
| Distracted F1 / precision / recall | 0.970 / 0.981 / 0.960 |
| Focused F1 / precision / recall | 0.797 / 0.743 / 0.860 |
| Inference latency (TorchScript ResNet18, MPS) | ~30 ms / frame |
| Inference latency (CPU) | <300 ms / frame |
| Storage | ~1 MB of logging per hour |

`checkpoints/metrics.json` has the full per-epoch history and confusion matrices.

## Privacy

All processing is local. Webcam frames are stored only as predicted-class metadata in the SQLite DB; the raw images are never written to disk.

## License

MIT — see [LICENSE](LICENSE).
