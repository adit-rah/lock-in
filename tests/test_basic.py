"""Basic tests for Lock-In system"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, ModelConfig, InferenceConfig, ScoringConfig


class TestConfig:
    """Test configuration system"""
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values"""
        config = ModelConfig()
        assert config.architecture == "mobilenet_v3_small"
        assert config.input_size == 224
        assert config.num_classes == 5
    
    def test_inference_config_defaults(self):
        """Test InferenceConfig default values"""
        config = InferenceConfig()
        assert config.frame_interval_seconds == 3
        assert config.camera_index == 0
        assert config.max_inference_time_ms == 300
    
    def test_scoring_config_defaults(self):
        """Test ScoringConfig default values"""
        config = ScoringConfig()
        assert config.rolling_window_size == 10
        assert config.alert_threshold == 0.3
        assert config.consecutive_frames_required == 3
    
    def test_full_config(self):
        """Test full Config object"""
        config = Config()
        assert config.model is not None
        assert config.inference is not None
        assert config.scoring is not None


class TestModel:
    """Test model architecture"""
    
    def test_create_mobilenet(self):
        """Test MobileNet model creation"""
        from src.model import create_model
        model = create_model(num_classes=5, architecture="mobilenet_v3_small")
        assert model is not None
        assert model.num_classes == 5
    
    def test_create_resnet(self):
        """Test ResNet model creation"""
        from src.model import create_model
        model = create_model(num_classes=5, architecture="resnet18")
        assert model is not None
        assert model.num_classes == 5


class TestScoring:
    """Test scoring logic"""
    
    def test_scorer_initialization(self):
        """Test FocusScorer initialization"""
        from src.scoring import FocusScorer
        config = Config()
        config.classes = ['focused', 'looking_away', 'using_phone', 'yawning', 'sleepy']
        config.distracted_classes = ['looking_away', 'using_phone', 'yawning', 'sleepy']
        
        scorer = FocusScorer(config)
        assert scorer.window_size == 10
        assert len(scorer.focused_indices) == 1
        assert len(scorer.distracted_indices) == 4
    
    def test_scorer_state(self):
        """Test scorer state"""
        from src.scoring import FocusScorer
        import numpy as np
        
        config = Config()
        config.classes = ['focused', 'looking_away', 'using_phone', 'yawning', 'sleepy']
        config.distracted_classes = ['looking_away', 'using_phone', 'yawning', 'sleepy']
        
        scorer = FocusScorer(config)
        
        # Mock prediction
        prediction = {
            'predicted_class': 0,  # focused
            'probabilities': np.array([0.9, 0.02, 0.03, 0.03, 0.02])
        }
        
        score_data = scorer.add_prediction(prediction)
        assert score_data['is_locked_in'] == True
        assert score_data['consecutive_distracted'] == 0


class TestSignals:
    """Test signal handlers"""
    
    def test_signal_handler_creation(self):
        """Test signal handler factory"""
        from src.signals import create_signal_handler
        config = Config()
        
        handler = create_signal_handler(config)
        assert handler is not None


class TestLogging:
    """Test logging system"""
    
    def test_logger_initialization(self):
        """Test FocusLogger initialization"""
        from src.logging_db import FocusLogger
        import tempfile
        
        config = Config()
        config.logging.database_path = tempfile.mktemp(suffix='.db')
        config.logging.csv_backup = False
        
        logger = FocusLogger(config)
        assert Path(config.logging.database_path).exists()
        
        # Cleanup
        Path(config.logging.database_path).unlink()


def test_imports():
    """Test that all modules can be imported"""
    import src
    import src.config
    import src.model
    import src.train
    import src.inference
    import src.scoring
    import src.signals
    import src.logging_db
    import src.app


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

