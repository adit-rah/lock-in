# Lock-In — Architecture

This document explains the moving pieces and how data flows from webcam to alert. Implementation details only — see `README.md` for the product pitch and `QUICKSTART.md` for setup.

## Components

```
                  ┌─────────────────────────────────────────┐
                  │     CLI (src/app.py)                    │
                  │     Streamlit (src/dashboard.py)        │
                  └──────────────────┬──────────────────────┘
                                     │  orchestrates
       ┌─────────────────────────────┼─────────────────────────────┐
       ▼                             ▼                             ▼
┌───────────────┐            ┌───────────────┐           ┌──────────────────┐
│ InferenceEng. │            │ FocusScorer   │           │ FocusLogger      │
│ inference.py  │──prediction│ scoring.py    │──score────│ logging_db.py    │
└───────┬───────┘            └───────┬───────┘           └──────────────────┘
        │                            │                            │
        ▼                            ▼                            ▼
   webcam + cv2 +              rolling window               SQLite (batched
   TorchScript model           lock-in score                executemany) + CSV
        │
        ▼
   src/model.py
```

| Module | Responsibility |
|---|---|
| `app.py` | Headless CLI loop. Boots all components, drives the per-frame cadence, handles Ctrl-C. |
| `dashboard.py` | Streamlit UI. Live tab spawns a background worker; History tab reads SQLite directly. |
| `inference.py` | `cv2.VideoCapture` setup with `CAP_PROP_BUFFERSIZE=1` + grab/retrieve flush; PIL/torchvision preprocess; TorchScript forward pass. Exposes `predict_frame` (single) and `predict_batch` (stacked). |
| `scoring.py` | Rolling window over the last N predictions; computes `S = mean(P(focused)) − mean(P(distracted))`; tracks the consecutive-distracted streak and decides when to fire an alert. |
| `signals.py` | Notification delivery. Uses `win10toast` on Windows and `plyer` elsewhere with console fallback. Pluggable via the `SignalHandler` ABC. |
| `logging_db.py` | SQLite (4 tables) + optional CSV mirror. Predictions are buffered and flushed via `executemany` to keep writes off the inference path. |
| `model.py` | Backbone factory (ResNet18/34, MobileNetV3-Small/Large), TorchScript save/load, device picker (`cuda` → `mps` → `cpu`). |
| `train.py` | Image-folder dataset, augmentation, training loop with macro-F1 reporting and F1-based checkpointing, optional class-balanced sampler. |
| `config.py` | Dataclass-backed YAML config. |

## Live monitoring loop

```
1. capture
   InferenceEngine.capture_frame()
   - grab() twice to drain the OpenCV buffer
   - retrieve() once to decode the freshest frame
   - returns BGR ndarray (H, W, 3)

2. preprocess
   InferenceEngine.preprocess_frame()
   - BGR → RGB
   - Resize to (input_size, input_size); default 224
   - ImageNet normalize
   - Returns (1, 3, 224, 224) tensor on the chosen device

3. inference
   TorchScript forward pass → softmax → top-1
   - Latency target: <300 ms on CPU, <50 ms on MPS/CUDA
   - Warns if measured latency exceeds the configured target

4. scoring
   FocusScorer.add_prediction()
   - Append to a deque(maxlen=rolling_window_size)
   - Score = mean(P(focused)) − mean(P(distracted)) across the window
   - is_locked_in = S >= alert_threshold
   - Increment consecutive_distracted counter when top-1 is distracted, reset on focused
   - trigger_alert = locked-in → not-locked-in edge AND streak >= consecutive_frames_required

5. logging
   FocusLogger:
   - log_prediction()   → queued; flushed in batches via executemany
   - log_score()        → committed per-call (low frequency)
   - log_to_csv()       → append-only row
   - log_event()        → on alert / session start / session end

6. signaling
   When trigger_alert is true, SignalHandler.trigger(event_dict) fires the
   desktop notification (with cooldown).

7. wait
   Sleep for inference.frame_interval_seconds and repeat.
```

## Class scheme

Binary: `distracted` (index 0), `focused` (index 1).

Two cross-checks keep these in sync:
- **Training**: `src/train.py:build_datasets` pins label indices to `config.classes` order before constructing the `DistractionDataset`. So whatever order `config.yaml` declares, that's what the trained model emits.
- **Inference**: `FocusScorer` resolves `focused_indices` and `distracted_indices` from `config.classes` and `config.distracted_classes` at startup. The lookup `config.classes[predicted_idx]` matches the trained model's index assignment.

If you retrain with a different class set, update both `config.classes` and `config.distracted_classes`; everything else flows from there.

## Database schema (`data/focus_log.db`)

```sql
sessions(session_id, start_time, end_time,
         total_frames, focused_frames, distracted_frames, focus_ratio)

predictions(prediction_id, session_id, timestamp,
            predicted_class, predicted_class_name, confidence, probabilities)
-- probabilities is a JSON-encoded float array (length = num_classes)

scores(score_id, session_id, timestamp,
       lock_in_score, is_locked_in, consecutive_distracted,
       mean_focused_prob, mean_distracted_prob)

events(event_id, session_id, timestamp, event_type, description)
```

All non-`sessions` tables `FOREIGN KEY (session_id) REFERENCES sessions`. Predictions are inserted via `executemany`; the buffer size is `config.logging.batch_size`.

## Training pipeline (`src/train.py`)

```
build_datasets(data_dir, …)
  ├─ If data_dir/train and data_dir/val exist → use them
  └─ Otherwise → random split by config.training.validation_split

DataLoader (train: shuffle or WeightedRandomSampler; val: shuffle=False)

For each epoch:
  train_epoch  → CrossEntropyLoss + Adam; running loss/acc on tqdm
  validate     → accumulates y_true/y_pred; computes macro-F1, per-class
                 P/R/F1, confusion matrix via sklearn
  scheduler.step(macro_f1)   # ReduceLROnPlateau, mode='max'
  if macro_f1 > best         # F1-based checkpointing
      save best_model_epoch_N.pth

After training:
  metrics.json  → full history (per-epoch F1, confusion matrix, etc.)
  TorchScript   → traced on CPU and saved as models/distraction_classifier.pt
                  (CPU trace ensures the .pt is portable across devices)
```

The class-balanced sampler (`config.training.use_class_balanced_sampler`) is on by default — important on State Farm where binary labels are ~10% focused / ~90% distracted.

## Inference optimizations (resume bullet 3)

- **Buffer pinning** — `cv2.CAP_PROP_BUFFERSIZE=1` so the driver buffer never holds more than one frame.
- **Grab/retrieve flush** — `grab()` is cheap (no JPEG decode); calling it before `retrieve()` skips any stale queued frames so we always classify the freshest webcam state.
- **Batched DB writes** — `FocusLogger.log_prediction` buffers up to `config.logging.batch_size` predictions, then a single `executemany` commit. Cuts fsync overhead during long sessions.
- **TorchScript** — model is traced once at training time and reloaded via `torch.jit.load`, dropping Python interpreter overhead per forward pass.
- **Device picker** — `src/model.pick_device()` returns CUDA > MPS > CPU. MPS support means M-series Macs do inference on the GPU (~5× CPU speed for ResNet18).

## Extending

| Want to add… | Hook in at… |
|---|---|
| A new alert delivery (smart bulb, hardware buzzer, etc.) | Implement `SignalHandler` in `src/signals.py`; register in `create_signal_handler`. |
| A different backbone | Add a branch in `DistractionClassifier.__init__` and reference it in `config.yaml`'s `model.architecture`. |
| A multi-class taxonomy | Update `config.classes` + `config.distracted_classes` + `config.model.num_classes`, retrain. The scorer's mean-over-indices logic handles any number of classes. |
| Multi-modal scoring (e.g. keyboard activity) | Subclass `FocusScorer`, blend its score with another signal, drop the subclass into `app.py`/`dashboard.py`. |

## Testing

```bash
pytest tests/
```

Covers config defaults, model construction, scoring math (5-class + binary), F1 computation, signal-handler factory, FocusLogger init, and the State Farm prep adapter (mapping + driver-disjoint splits, on a synthetic tiny dataset).
