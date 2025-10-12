# Lock-In Focus Monitor 🔒

A desktop application that monitors your focus state in real-time using computer vision. The system captures webcam frames, classifies your attention level using a trained CNN, and alerts you when you're distracted for extended periods.

## 🎯 Features

- **Real-time Focus Detection**: Classifies attention state every few seconds using your webcam
- **Smart Scoring System**: Rolling window algorithm to avoid false positives
- **Desktop Notifications**: Get alerted when losing focus
- **Comprehensive Logging**: SQLite database + CSV backup for all sessions
- **Privacy-First**: All processing happens locally, no cloud uploads
- **Modular Architecture**: Easily extensible for new input modalities or hardware
- **Configurable**: Tune sensitivity, intervals, and thresholds via YAML config

## 📋 Classification Categories

- **focused**: Actively engaged with work
- **looking_away**: Eyes not on screen
- **using_phone**: Using mobile device
- **yawning**: Signs of fatigue
- **sleepy**: Drowsiness detected

## 🏗️ System Architecture

```
[Webcam Capture] → [Preprocessing] → [PyTorch Model Inference]
                                             ↓
[Notification] ← [Signal Handler] ← [Scoring Logic] ← [Rolling Window Buffer]
                                             ↓
                                    [SQLite + CSV Logging]
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam
- Windows/Linux/macOS

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lock-in
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training Your Model

Before using the monitor, you need to train a classification model:

1. **Prepare your dataset** in the following structure:
```
data/
├── focused/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── looking_away/
│   ├── img001.jpg
│   └── ...
├── using_phone/
│   └── ...
├── yawning/
│   └── ...
└── sleepy/
    └── ...
```

2. **Recommended public datasets**:
   - [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
   - [YawDD - Yawning Detection Dataset](http://ieee-dataport.org/1096)
   - [Columbia Gaze Dataset](http://www.cs.columbia.edu/CAVE/databases/columbia_gaze/)

3. **Train the model**:
```bash
python -m src.train --data_dir data/training --config config.yaml
```

This will:
- Train a MobileNetV3 or ResNet18 classifier
- Save checkpoints to `checkpoints/`
- Export the final model as TorchScript to `models/distraction_classifier.pt`

### Running the Monitor

Once trained, start monitoring:

```bash
python -m src.app
```

Or with a custom config:

```bash
python -m src.app --config my_config.yaml
```

### View Session History

```bash
python -m src.app --view-sessions
```

## ⚙️ Configuration

Edit `config.yaml` to customize behavior:

```yaml
inference:
  frame_interval_seconds: 3  # How often to capture frames
  camera_index: 0            # Webcam device index

scoring:
  rolling_window_size: 10           # Frames to average
  alert_threshold: 0.3              # Lock-in score threshold
  consecutive_frames_required: 3    # Consecutive distracted frames before alert

notification:
  enabled: true
  cooldown_seconds: 60  # Minimum time between notifications
```

## 📊 Understanding Lock-In Score

The **lock-in score** is computed as:

```
S = mean(P(focused)) - mean(P(distracted))
```

Where:
- `P(focused)` = probability of "focused" class
- `P(distracted)` = mean probability of distracted classes (looking_away, using_phone, yawning, sleepy)

**Score ranges**:
- `S > 0.3`: Locked in ✅
- `S < 0.3`: Distracted ⚠️

## 📁 Project Structure

```
lock-in/
├── config.yaml              # Main configuration
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── src/
│   ├── __init__.py
│   ├── __main__.py         # Module entry point
│   ├── config.py           # Configuration management
│   ├── model.py            # Model architecture
│   ├── train.py            # Training script
│   ├── inference.py        # Webcam capture & inference
│   ├── scoring.py          # Rolling window scoring logic
│   ├── signals.py          # Notification system
│   ├── logging_db.py       # Database logging
│   └── app.py              # Main application
├── models/                 # Trained models
│   └── distraction_classifier.pt
├── checkpoints/            # Training checkpoints
├── data/                   # Logs and database
│   ├── focus_log.db
│   └── focus_log.csv
```

## 🔧 Advanced Usage

### Custom Notifications

The signal handler system is extensible. To add custom alerts:

```python
from src.signals import SignalHandler

class CustomHandler(SignalHandler):
    def trigger(self, event):
        # Your custom logic here
        print(f"Custom alert: {event}")
        return True
```

### Hardware Integration

Extend `HardwareSignalHandler` for device integration:

```python
from src.signals import HardwareSignalHandler

class BluetoothShockDevice(HardwareSignalHandler):
    def __init__(self, device_config):
        super().__init__(device_config)
        # Initialize Bluetooth connection
    
    def trigger(self, event):
        # Send signal to device
        self.device.send_pulse()
        return True
```

### Fine-tuning on Personal Data

Collect personal samples and fine-tune:

```python
# Capture personal samples
python scripts/capture_samples.py --output data/personal --duration 300

# Fine-tune model
python -m src.train --data_dir data/personal --config config.yaml --resume models/distraction_classifier.pt
```

## 📈 Performance Metrics

- **Inference latency**: <300ms per frame (CPU)
- **Storage**: ~1 MB/hour of logging data
- **Model size**: <100 MB
- **Accuracy**: Depends on training data quality

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
flake8 src/
```

## 🔮 Future Enhancements

- [ ] Keyboard/mouse activity tracking
- [ ] Multi-modal fusion (webcam + activity sensors)
- [ ] Adaptive thresholding based on time of day
- [ ] Integration with productivity tools (RescueTime, Toggl)
- [ ] Mobile app for remote monitoring
- [ ] Eye tracking integration
- [ ] Hardware device support (smart lights, wearables)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## ⚠️ Privacy & Ethics

This application is designed for **personal use only**. Key considerations:

- All processing is local; no data leaves your machine
- Webcam access is limited to periodic frame capture
- Use responsibly and with consent if monitoring others
- Consider potential biases in training data

## 🙏 Acknowledgments

- Pretrained models from PyTorch/torchvision
- Public datasets: State Farm, YawDD, Columbia Gaze
- Notification libraries: plyer, win10toast

## 📞 Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

**Stay focused, stay productive! 🔒✨**
