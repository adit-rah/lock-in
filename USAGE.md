# Lock-In Monitor - Detailed Usage Guide

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Training Your Model](#training-your-model)
3. [Running the Monitor](#running-the-monitor)
4. [Configuration Guide](#configuration-guide)
5. [Data Collection](#data-collection)
6. [Troubleshooting](#troubleshooting)

## Initial Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

Check if PyTorch is properly installed:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

### 3. Test Webcam

Quick test to ensure webcam is accessible:

```bash
python -c "import cv2; cam = cv2.VideoCapture(0); print('Camera opened:', cam.isOpened()); cam.release()"
```

## Training Your Model

### Option 1: Using Public Datasets

**Step 1: Download Datasets**

Recommended sources:
- **State Farm Distracted Driver**: [Kaggle Link](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
- **YawDD**: [IEEE DataPort](http://ieee-dataport.org/1096)

**Step 2: Organize Data**

Create directory structure:
```
training_data/
├── focused/
├── looking_away/
├── using_phone/
├── yawning/
└── sleepy/
```

**Step 3: Train**

```bash
python -m src.train --data_dir training_data --config config.yaml
```

Expected output:
```
Loading dataset...
Training samples: 8000, Validation samples: 2000
Classes: ['focused', 'looking_away', 'using_phone', 'yawning', 'sleepy']
Creating model: mobilenet_v3_small
...
Training completed! Best validation accuracy: 87.5%
Model saved as TorchScript to models/distraction_classifier.pt
```

### Option 2: Personal Calibration Dataset

**Step 1: Capture Personal Samples**

For each class, run:
```bash
# Focused state (60 seconds of samples)
python scripts/capture_samples.py --class_name focused --duration 60 --output data/personal

# Looking away
python scripts/capture_samples.py --class_name looking_away --duration 60 --output data/personal

# Using phone
python scripts/capture_samples.py --class_name using_phone --duration 60 --output data/personal

# Yawning
python scripts/capture_samples.py --class_name yawning --duration 30 --output data/personal

# Sleepy
python scripts/capture_samples.py --class_name sleepy --duration 30 --output data/personal
```

**Step 2: Train on Personal Data**

```bash
python -m src.train --data_dir data/personal --config config.yaml
```

### Option 3: Transfer Learning (Recommended)

Train on public data, then fine-tune on personal samples:

```bash
# 1. Train on public data
python -m src.train --data_dir training_data --config config.yaml

# 2. Capture personal samples (as above)

# 3. Fine-tune (update config to use lower learning rate)
python -m src.train --data_dir data/personal --config config_finetune.yaml
```

### Evaluating Your Model

```bash
python scripts/evaluate_model.py --model_path models/distraction_classifier.pt --test_data test_data --config config.yaml
```

## Running the Monitor

### Basic Usage

```bash
python -m src.app
```

You'll see:
```
============================================================
Starting Lock-In Focus Monitoring
============================================================
Frame interval: 3s
Rolling window: 10 frames
Alert threshold: 0.3
Classes: focused, looking_away, using_phone, yawning, sleepy

Press Ctrl+C to stop monitoring
============================================================

[14:32:15] Frame    1 | Class: focused       (92.3%) | Score: +0.843 | Status: 🔒 LOCKED IN
[14:32:18] Frame    2 | Class: focused       (88.7%) | Score: +0.752 | Status: 🔒 LOCKED IN
[14:32:21] Frame    3 | Class: looking_away  (76.5%) | Score: +0.234 | Status: ⚠️  DISTRACTED
```

### Custom Configuration

```bash
python -m src.app --config my_custom_config.yaml
```

### Viewing Session History

```bash
python -m src.app --view-sessions
```

Output:
```
============================================================
RECENT SESSIONS
============================================================

Session 5:
  Start: 2024-10-12 14:30:22
  End: 2024-10-12 15:45:11
  Frames: 1489
  Focus ratio: 87.3%

Session 4:
  Start: 2024-10-12 09:15:33
  End: 2024-10-12 11:20:45
  Frames: 2501
  Focus ratio: 92.1%
```

## Configuration Guide

### Key Parameters to Tune

**1. Frame Interval** (`inference.frame_interval_seconds`)
- **Lower (1-2s)**: More responsive, higher CPU usage
- **Higher (5-10s)**: Less intrusive, may miss brief distractions
- **Recommended**: 3s for balanced performance

**2. Rolling Window Size** (`scoring.rolling_window_size`)
- **Smaller (5-7)**: Quick response, more false positives
- **Larger (15-20)**: Smoother, may delay alerts
- **Recommended**: 10 frames (30s at 3s interval)

**3. Alert Threshold** (`scoring.alert_threshold`)
- **Higher (0.5-0.7)**: Stricter, fewer alerts
- **Lower (0.1-0.3)**: More sensitive, frequent alerts
- **Recommended**: 0.3 for balanced sensitivity

**4. Consecutive Frames** (`scoring.consecutive_frames_required`)
- Controls alert trigger delay
- **Recommended**: 3 frames (9s at 3s interval)

### Example Configurations

**Strict Mode** (for deep focus work):
```yaml
scoring:
  rolling_window_size: 8
  alert_threshold: 0.5
  consecutive_frames_required: 2

notification:
  cooldown_seconds: 30
```

**Relaxed Mode** (for casual browsing):
```yaml
scoring:
  rolling_window_size: 15
  alert_threshold: 0.2
  consecutive_frames_required: 5

notification:
  cooldown_seconds: 120
```

## Data Collection

### Collecting Diverse Training Data

**Tips for quality data:**

1. **Varied Lighting**: Capture in different lighting conditions
2. **Different Times**: Morning vs evening
3. **Multiple Sessions**: Don't capture all at once
4. **Natural Poses**: Act naturally for each class
5. **Camera Angles**: Ensure camera position matches usage

**Recommended Sample Counts per Class:**
- Minimum: 200 samples per class
- Good: 500 samples per class
- Excellent: 1000+ samples per class

### Augmentation

The training script automatically applies:
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation)

## Troubleshooting

### Camera Not Opening

**Error**: `Failed to open camera 0`

**Solutions**:
1. Check if another application is using the camera
2. Try different camera index: Change `camera_index` in config
3. On Linux, check permissions: `sudo usermod -a -G video $USER`

### Model Not Found

**Error**: `Model not found at models/distraction_classifier.pt`

**Solution**: Train the model first:
```bash
python -m src.train --data_dir your_data --config config.yaml
```

### Notifications Not Showing

**Windows**:
- Install: `pip install win10toast`
- Ensure notifications are enabled in Windows settings

**Linux**:
- Install: `pip install plyer`
- May need: `sudo apt-get install python3-gi libnotify-dev`

**macOS**:
- Install: `pip install plyer`

### High CPU Usage

**Solutions**:
1. Increase `frame_interval_seconds` (e.g., from 3 to 5)
2. Use MobileNetV3 instead of ResNet (smaller model)
3. Ensure GPU is being used if available

### Poor Detection Accuracy

**Solutions**:
1. **Calibrate with personal data**: Collect 30-60s of samples per class
2. **Check lighting**: Ensure consistent lighting during training and inference
3. **Camera position**: Keep webcam position consistent
4. **More training data**: Collect more diverse samples
5. **Fine-tune**: Use transfer learning from public dataset, then fine-tune on personal data

### Slow Inference

**Check inference time**:
The app displays warnings if inference exceeds 300ms.

**Optimizations**:
1. Reduce input size in config (e.g., 224 → 160)
2. Use MobileNetV3-Small (fastest)
3. Enable GPU: `model.use_gpu: true`

### Database Errors

**Reset database**:
```bash
rm data/focus_log.db
# Restart application to create fresh database
```

## Advanced Usage

### Running in Background

**Windows** (PowerShell):
```powershell
Start-Process python -ArgumentList "-m src.app" -WindowStyle Hidden
```

**Linux/macOS**:
```bash
nohup python -m src.app > lock-in.log 2>&1 &
```

### Integrating with Productivity Tools

Export session data:
```python
from src.logging_db import FocusLogger
from src.config import load_config

config = load_config()
logger = FocusLogger(config)
sessions = logger.get_recent_sessions(30)

# Export to your format
for session in sessions:
    print(f"{session['start_time']},{session['focus_ratio']}")
```

### Custom Notification Messages

Edit `config.yaml`:
```yaml
notification:
  distracted_message: "Hey! Get back to work! 💪"
```

### Exporting Data for Analysis

Data is stored in:
- SQLite: `data/focus_log.db` (query with any SQLite tool)
- CSV: `data/focus_log.csv` (open in Excel, pandas, etc.)

**Example analysis** (with pandas):
```python
import pandas as pd

df = pd.read_csv('data/focus_log.csv')
hourly_focus = df.groupby(df['timestamp'].str[:13])['is_locked_in'].mean()
print(hourly_focus)
```

## Performance Benchmarks

Typical performance on mid-range laptop:

| Metric | Value |
|--------|-------|
| Inference Time (CPU) | 150-250ms |
| Inference Time (GPU) | 20-50ms |
| Memory Usage | ~500MB |
| CPU Usage (idle) | ~5% |
| CPU Usage (inference) | ~30% spike |
| Storage per hour | ~1MB |

## Best Practices

1. **Consistent Setup**: Keep webcam position and lighting consistent
2. **Calibration**: Collect personal samples for best accuracy
3. **Regular Breaks**: Use as a tool, not a taskmaster
4. **Privacy**: Review logs periodically, delete if needed
5. **Tune Gradually**: Start with default settings, adjust slowly
6. **Monitor Battery**: On laptops, this will impact battery life

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review logs in `data/focus_log.csv`
3. Run with verbose output: `python -m src.app --config config.yaml -v`
4. Open an issue on GitHub with:
   - Your configuration
   - Error messages
   - System information

---

**Happy focusing! 🔒✨**

