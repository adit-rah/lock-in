"""Configuration management for Lock-In system"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    architecture: str = "mobilenet_v3_small"
    input_size: int = 224
    num_classes: int = 5
    model_path: str = "models/distraction_classifier.pt"
    use_gpu: bool = True


@dataclass
class InferenceConfig:
    frame_interval_seconds: int = 3
    camera_index: int = 0
    max_inference_time_ms: int = 300


@dataclass
class ScoringConfig:
    rolling_window_size: int = 10
    alert_threshold: float = 0.3
    consecutive_frames_required: int = 3


@dataclass
class NotificationConfig:
    enabled: bool = True
    title: str = "Lock-In Monitor"
    distracted_message: str = "You're losing focus! Time to lock back in."
    cooldown_seconds: int = 60


@dataclass
class LoggingConfig:
    database_path: str = "data/focus_log.db"
    log_predictions: bool = True
    log_scores: bool = True
    csv_backup: bool = True
    csv_path: str = "data/focus_log.csv"


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    classes: List[str] = field(default_factory=list)
    distracted_classes: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**data.get('model', {})),
            inference=InferenceConfig(**data.get('inference', {})),
            scoring=ScoringConfig(**data.get('scoring', {})),
            notification=NotificationConfig(**data.get('notification', {})),
            logging=LoggingConfig(**data.get('logging', {})),
            training=TrainingConfig(**data.get('training', {})),
            classes=data.get('classes', []),
            distracted_classes=data.get('distracted_classes', [])
        )
    
    def get_focused_classes(self) -> List[str]:
        """Get list of focused class names"""
        return [c for c in self.classes if c not in self.distracted_classes]


def load_config(config_path: str = "config.yaml") -> Config:
    """Convenience function to load configuration"""
    return Config.from_yaml(config_path)

