# Lock-In Monitor - Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Install (1 minute)

```bash
# Clone repository
git clone <repository-url>
cd lock-in

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Get a Pretrained Model (30 seconds)

**Option A**: Download pretrained model (if available)
```bash
# Download from releases
wget https://github.com/your-repo/lock-in/releases/download/v1.0/distraction_classifier.pt -O models/distraction_classifier.pt
```

**Option B**: Use a quick demo model
```bash
# We'll train on minimal personal data
mkdir -p data/demo
```

## Step 3: Collect Quick Calibration Data (2 minutes)

```bash
# Focused state (30 seconds)
python scripts/capture_samples.py --class_name focused --duration 30 --output data/demo

# Looking away (30 seconds)
python scripts/capture_samples.py --class_name looking_away --duration 30 --output data/demo

# Using phone (30 seconds)
python scripts/capture_samples.py --class_name using_phone --duration 30 --output data/demo

# Yawning (20 seconds)
python scripts/capture_samples.py --class_name yawning --duration 20 --output data/demo

# Sleepy (20 seconds)  
python scripts/capture_samples.py --class_name sleepy --duration 20 --output data/demo
```

## Step 4: Train Quick Model (1 minute)

```bash
# Quick training (will be fast with small dataset)
python -m src.train --data_dir data/demo --config config.yaml
```

## Step 5: Start Monitoring! (30 seconds)

```bash
python -m src.app
```

You should see:
```
============================================================
Starting Lock-In Focus Monitoring
============================================================
Frame interval: 3s
Rolling window: 10 frames
Alert threshold: 0.3

Press Ctrl+C to stop monitoring
============================================================

[14:32:15] Frame    1 | Class: focused (92.3%) | Score: +0.843 | Status: 🔒 LOCKED IN
```

---

## What's Next?

### For Better Accuracy
1. Collect more samples (100+ per class)
2. Use public datasets (State Farm, YawDD)
3. Train for longer (see [USAGE.md](USAGE.md))

### Customize Settings
Edit `config.yaml` to adjust:
- Alert frequency
- Sensitivity
- Notification messages

### View Your Progress
```bash
python -m src.app --view-sessions
```

---

## Common Issues

**Camera not working?**
```bash
# Test camera
python -c "import cv2; cam = cv2.VideoCapture(0); print('OK' if cam.isOpened() else 'FAIL'); cam.release()"
```

**Notifications not showing?**
```bash
pip install win10toast  # Windows
pip install plyer       # Linux/Mac
```

**Model accuracy poor?**
- Collect more samples (Step 3)
- Ensure good lighting
- Keep camera position consistent

---

## Full Documentation

- [Complete Usage Guide](USAGE.md)
- [Dataset Information](docs/DATASETS.md)
- [Configuration Options](README.md#configuration)

---

**Ready to lock in? Let's go! 🔒✨**

