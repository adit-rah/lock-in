# Lock-In Monitor - System Architecture

This document provides a detailed technical overview of the Lock-In focus monitoring system.

## Table of Contents
1. [Overview](#overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Module Details](#module-details)
5. [Extensibility Points](#extensibility-points)
6. [Performance Considerations](#performance-considerations)

---

## Overview

Lock-In is a modular desktop application that monitors user focus state through webcam-based computer vision. The system operates entirely offline and uses a trained CNN to classify attention level.

### Design Principles
- **Modularity**: Each component is independent and replaceable
- **Privacy-First**: All processing is local, no cloud dependencies
- **Extensibility**: Easy to add new input modalities or output signals
- **Configurability**: All parameters tunable via YAML config
- **Offline-Capable**: No internet required for operation

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Lock-In Monitor                          │
│                         (app.py)                             │
└───────┬────────────────────┬────────────────┬───────────────┘
        │                    │                │
┌───────▼──────┐    ┌────────▼──────┐  ┌─────▼────────┐
│   Inference  │    │   Scoring     │  │   Logging    │
│   Engine     │───►│   Logic       │──┤   System     │
│ (inference.py)│    │ (scoring.py)  │  │(logging_db.py)│
└──────┬───────┘    └────────┬──────┘  └──────────────┘
       │                     │
┌──────▼───────┐    ┌────────▼──────┐
│  PyTorch     │    │   Signal      │
│  Model       │    │   Handler     │
│ (model.py)   │    │ (signals.py)  │
└──────────────┘    └───────────────┘
```

### Component Responsibilities

| Component | Responsibility | Dependencies |
|-----------|---------------|--------------|
| `app.py` | Main application loop, orchestration | All components |
| `inference.py` | Webcam capture, preprocessing, model inference | OpenCV, PyTorch, model |
| `scoring.py` | Rolling window logic, lock-in score computation | NumPy |
| `signals.py` | Notification system, extensible alert handlers | plyer/win10toast |
| `logging_db.py` | SQLite database, CSV backup, session tracking | SQLite3 |
| `model.py` | Model architecture, loading, TorchScript export | PyTorch |
| `train.py` | Training pipeline, data loading, optimization | PyTorch, torchvision |
| `config.py` | Configuration management, YAML parsing | PyYAML |

---

## Data Flow

### Monitoring Loop

```
1. CAPTURE
   ├─ InferenceEngine.capture_frame()
   ├─ OpenCV VideoCapture
   └─ Returns: BGR numpy array (H×W×3)

2. PREPROCESS
   ├─ InferenceEngine.preprocess_frame()
   ├─ BGR→RGB conversion
   ├─ Resize to 224×224
   ├─ Normalize (ImageNet stats)
   └─ Returns: torch.Tensor (1×3×224×224)

3. INFERENCE
   ├─ InferenceEngine.inference()
   ├─ Model forward pass
   ├─ Softmax activation
   └─ Returns: logits, probabilities

4. SCORING
   ├─ FocusScorer.add_prediction()
   ├─ Add to rolling window
   ├─ Compute mean P(focused) and P(distracted)
   ├─ Calculate score: S = P(focused) - P(distracted)
   ├─ Check threshold and consecutive frames
   └─ Returns: score_data, trigger_alert flag

5. LOGGING
   ├─ FocusLogger.log_prediction()
   ├─ FocusLogger.log_score()
   ├─ FocusLogger.log_to_csv()
   └─ Writes to SQLite + CSV

6. SIGNALING
   ├─ If trigger_alert:
   ├─ SignalHandler.trigger()
   └─ Desktop notification

7. WAIT
   ├─ time.sleep(frame_interval_seconds)
   └─ Repeat from step 1
```

### Data Structures

**Prediction Object**:
```python
{
    'timestamp': datetime,
    'frame': np.ndarray,  # BGR image
    'logits': np.ndarray,  # Shape: (5,)
    'probabilities': np.ndarray,  # Shape: (5,)
    'predicted_class': int,
    'predicted_class_name': str,
    'confidence': float
}
```

**Score Data Object**:
```python
{
    'lock_in_score': float,
    'is_locked_in': bool,
    'window_size': int,
    'consecutive_distracted': int,
    'trigger_alert': bool,
    'mean_focused_prob': float,
    'mean_distracted_prob': float
}
```

---

## Module Details

### 1. Inference Engine (`inference.py`)

**Class**: `InferenceEngine`

**Initialization**:
```python
engine = InferenceEngine(model, config)
```

**Key Methods**:
- `capture_frame()`: Captures single frame from webcam
- `preprocess_frame(frame)`: Applies transforms for model input
- `inference(tensor)`: Runs model inference
- `predict_frame(frame=None)`: End-to-end prediction pipeline

**Performance**:
- Target latency: <300ms per frame
- GPU acceleration supported
- Automatic camera reinitialization on failure

---

### 2. Scoring Logic (`scoring.py`)

**Class**: `FocusScorer`

**Algorithm**:
```python
# Maintain rolling window of N predictions
window = deque(maxlen=N)

# For each new prediction:
window.append(prediction)

# Compute mean probabilities
focused_prob = mean([p.prob[focused_classes] for p in window])
distracted_prob = mean([p.prob[distracted_classes] for p in window])

# Lock-in score
score = focused_prob - distracted_prob

# State determination
if score < threshold:
    state = "not locked in"
    if consecutive_distracted >= M:
        trigger_alert = True
else:
    state = "locked in"
```

**Configurable Parameters**:
- `rolling_window_size`: Number of frames to average (default: 10)
- `alert_threshold`: Score threshold for lock-in (default: 0.3)
- `consecutive_frames_required`: Consecutive distracted frames before alert (default: 3)

---

### 3. Signal Handlers (`signals.py`)

**Abstract Base Class**: `SignalHandler`

```python
class SignalHandler(ABC):
    @abstractmethod
    def trigger(self, event: Dict) -> bool:
        pass
```

**Implementations**:

1. **DesktopNotificationHandler**: System notifications
   - Windows: win10toast
   - Linux/Mac: plyer
   - Fallback: console output

2. **LoggingSignalHandler**: File-based event logging

3. **CompositeSignalHandler**: Combines multiple handlers

4. **HardwareSignalHandler**: Extensible for device integration

**Event Object**:
```python
{
    'event_type': 'distracted' | 'focused',
    'timestamp': datetime,
    'score_data': {...},
    'prediction_data': {...},
    'alert_number': int
}
```

---

### 4. Logging System (`logging_db.py`)

**Database Schema**:

```sql
-- Sessions table
CREATE TABLE sessions (
    session_id INTEGER PRIMARY KEY,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_frames INTEGER,
    focused_frames INTEGER,
    distracted_frames INTEGER,
    focus_ratio REAL
);

-- Predictions table
CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY,
    session_id INTEGER,
    timestamp TIMESTAMP,
    predicted_class INTEGER,
    predicted_class_name TEXT,
    confidence REAL,
    probabilities TEXT,  -- JSON array
    FOREIGN KEY (session_id) REFERENCES sessions
);

-- Scores table
CREATE TABLE scores (
    score_id INTEGER PRIMARY KEY,
    session_id INTEGER,
    timestamp TIMESTAMP,
    lock_in_score REAL,
    is_locked_in BOOLEAN,
    consecutive_distracted INTEGER,
    mean_focused_prob REAL,
    mean_distracted_prob REAL,
    FOREIGN KEY (session_id) REFERENCES sessions
);

-- Events table
CREATE TABLE events (
    event_id INTEGER PRIMARY KEY,
    session_id INTEGER,
    timestamp TIMESTAMP,
    event_type TEXT,
    description TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions
);
```

**CSV Format**:
```csv
timestamp,predicted_class,predicted_class_name,confidence,lock_in_score,is_locked_in,consecutive_distracted,event_type
2024-10-12T14:30:15,0,focused,0.92,0.843,True,0,
2024-10-12T14:30:18,0,focused,0.88,0.752,True,0,
2024-10-12T14:30:21,1,looking_away,0.76,0.234,False,1,
```

---

### 5. Model Architecture (`model.py`)

**Supported Backbones**:
- MobileNetV3-Small (default, ~5MB)
- MobileNetV3-Large (~12MB)
- ResNet18 (~45MB)
- ResNet34 (~85MB)

**Model Structure** (MobileNetV3-Small):
```
Input: (B, 3, 224, 224)
  ↓
MobileNetV3 Backbone
  ↓
Global Average Pooling
  ↓
Linear(576, 1024)
  ↓
Hardswish()
  ↓
Dropout(0.2)
  ↓
Linear(1024, 5)
  ↓
Output: (B, 5) logits
```

**Classes**:
```python
0: 'focused'
1: 'looking_away'
2: 'using_phone'
3: 'yawning'
4: 'sleepy'
```

**Training**:
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001, weight_decay=0.0001)
- Scheduler: ReduceLROnPlateau
- Augmentation: Flip, rotation, color jitter
- Early stopping: 10 epochs patience

**Export Format**: TorchScript for optimized inference

---

### 6. Configuration System (`config.py`)

**Structure**:
```python
@dataclass
class Config:
    model: ModelConfig
    inference: InferenceConfig
    scoring: ScoringConfig
    notification: NotificationConfig
    logging: LoggingConfig
    training: TrainingConfig
    classes: List[str]
    distracted_classes: List[str]
```

**Loading**:
```python
config = load_config("config.yaml")
```

**Validation**: Automatic type checking via dataclasses

---

## Extensibility Points

### 1. Adding New Input Modalities

To add keyboard/mouse activity tracking:

```python
# Create new module: src/activity.py
class ActivityMonitor:
    def get_activity_score(self) -> float:
        # Track keyboard/mouse events
        # Return activity level [0, 1]
        pass

# Integrate in scoring.py
class MultiModalScorer(FocusScorer):
    def __init__(self, config, activity_monitor):
        super().__init__(config)
        self.activity_monitor = activity_monitor
    
    def add_prediction(self, prediction):
        vision_score = super().add_prediction(prediction)
        activity_score = self.activity_monitor.get_activity_score()
        
        # Fusion logic
        combined_score = 0.7 * vision_score + 0.3 * activity_score
        return combined_score
```

### 2. Adding Custom Signal Handlers

To integrate with smart lights:

```python
from src.signals import SignalHandler

class PhilipsHueHandler(SignalHandler):
    def __init__(self, bridge_ip, light_id):
        self.bridge_ip = bridge_ip
        self.light_id = light_id
        # Initialize Hue API connection
    
    def trigger(self, event):
        if event['event_type'] == 'distracted':
            # Flash red
            self.set_color(red)
        return True
```

Register in `config.yaml`:
```yaml
hardware:
  enabled: true
  type: "philips_hue"
  bridge_ip: "192.168.1.100"
  light_id: 1
```

### 3. Custom Model Architectures

To use a custom model:

```python
from src.model import DistractionClassifier

class MyCustomModel(DistractionClassifier):
    def __init__(self, num_classes):
        super().__init__(num_classes, architecture="mobilenet_v3_small")
        # Add custom layers
        self.attention = AttentionModule()
    
    def forward(self, x):
        features = self.backbone.features(x)
        attended = self.attention(features)
        return self.backbone.classifier(attended)
```

### 4. Database Extensions

To add custom analytics:

```python
from src.logging_db import FocusLogger

class ExtendedLogger(FocusLogger):
    def _init_database(self):
        super()._init_database()
        
        # Add custom table
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS productivity_metrics (
                metric_id INTEGER PRIMARY KEY,
                session_id INTEGER,
                timestamp TIMESTAMP,
                productivity_score REAL,
                FOREIGN KEY (session_id) REFERENCES sessions
            )
        ''')
        conn.commit()
        conn.close()
```

---

## Performance Considerations

### Latency Budget

| Component | Target | Typical | Max Acceptable |
|-----------|--------|---------|----------------|
| Frame Capture | <10ms | 5ms | 50ms |
| Preprocessing | <20ms | 10ms | 50ms |
| Inference (CPU) | <200ms | 150ms | 300ms |
| Inference (GPU) | <50ms | 30ms | 100ms |
| Scoring | <5ms | 2ms | 10ms |
| Logging | <10ms | 5ms | 50ms |
| **Total** | **<300ms** | **200ms** | **500ms** |

### Memory Usage

- Base application: ~200MB
- Model (MobileNetV3-Small): ~5MB
- Model (ResNet18): ~45MB
- Rolling window (10 frames): ~50MB
- Peak: ~500MB

### Storage

- Database: ~100KB per 1000 frames
- CSV: ~50KB per 1000 frames
- Hourly (3s interval): ~1MB

### Optimization Strategies

1. **Model Selection**:
   - Use MobileNetV3-Small for fastest inference
   - ResNet for highest accuracy

2. **Frame Interval**:
   - Lower interval = more responsive, higher CPU
   - Recommended: 2-5 seconds

3. **GPU Acceleration**:
   - 5-10x speedup for inference
   - Enable in config: `model.use_gpu: true`

4. **Batch Processing** (Future):
   - Process multiple frames in parallel
   - Requires async capture

---

## Threading Model

Current implementation is **single-threaded** for simplicity:
```
Main Thread: Capture → Infer → Score → Log → Notify → Sleep → Repeat
```

**Future multi-threaded design**:
```
Thread 1 (Capture):    Capture frames → Queue
Thread 2 (Inference):  Queue → Infer → Queue
Thread 3 (Processing): Queue → Score → Log → Notify
```

---

## Error Handling

### Failure Modes

1. **Camera Failure**: 
   - Automatic reinitialization
   - Retry 3 times before aborting

2. **Model Load Failure**:
   - Abort with clear error message
   - User instructed to train model

3. **Inference Timeout**:
   - Skip frame
   - Log warning
   - Continue monitoring

4. **Database Error**:
   - Continue operation
   - Log to CSV backup
   - Report error to console

---

## Security Considerations

1. **Data Privacy**:
   - All processing local
   - No network connections
   - Webcam frames not saved (unless debugging)

2. **File Permissions**:
   - Database: user read/write only
   - Logs: user read/write only

3. **Dependencies**:
   - Regular updates for security patches
   - Minimal dependency footprint

---

## Testing Strategy

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: End-to-end pipeline
3. **Performance Tests**: Latency benchmarks
4. **Accuracy Tests**: Model evaluation on test set

Run tests:
```bash
pytest tests/ -v
```

---

## Deployment

### Local Installation
```bash
pip install -r requirements.txt
```

### Package Installation
```bash
pip install -e .
lock-in  # Run application
```

### Standalone Executable (PyInstaller)
```bash
pyinstaller --onefile --windowed src/app.py
```

---

## Future Enhancements

1. **Multi-Modal Fusion**: Combine webcam + keyboard + mouse
2. **Adaptive Thresholds**: Learn personal baseline
3. **Attention Heatmaps**: Visualize gaze patterns
4. **Hardware Integration**: Smart lights, wearables
5. **Productivity Dashboard**: Web-based analytics
6. **Mobile App**: Remote monitoring
7. **Team Features**: Shared focus sessions

---

For implementation details, see source code in `src/` directory.

