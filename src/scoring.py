"""Scoring logic for lock-in state determination"""

from collections import deque
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

from .config import Config


class FocusScorer:
    """Maintains rolling window and computes lock-in score"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.window_size = config.scoring.rolling_window_size
        self.threshold = config.scoring.alert_threshold
        self.consecutive_required = config.scoring.consecutive_frames_required
        
        # Rolling window of predictions
        self.prediction_window = deque(maxlen=self.window_size)
        
        # Track consecutive distracted frames
        self.consecutive_distracted = 0
        
        # Current state
        self.is_locked_in = True
        
        # Indices for focused vs distracted classes
        self.focused_indices = [i for i, cls in enumerate(config.classes) 
                               if cls not in config.distracted_classes]
        self.distracted_indices = [i for i, cls in enumerate(config.classes) 
                                  if cls in config.distracted_classes]
    
    def add_prediction(self, prediction: Dict) -> Dict:
        """Add a new prediction to the rolling window and update score
        
        Args:
            prediction: Dictionary from InferenceEngine.predict_frame()
        
        Returns:
            Dictionary containing:
                - lock_in_score: current score
                - is_locked_in: boolean state
                - window_size: current window size
                - consecutive_distracted: count of consecutive distracted frames
                - trigger_alert: whether to trigger alert
                - mean_focused_prob: mean probability of focused classes
                - mean_distracted_prob: mean probability of distracted classes
        """
        self.prediction_window.append(prediction)
        
        # Compute score only if we have enough predictions
        if len(self.prediction_window) < 2:
            return {
                'lock_in_score': 1.0,
                'is_locked_in': True,
                'window_size': len(self.prediction_window),
                'consecutive_distracted': 0,
                'trigger_alert': False,
                'mean_focused_prob': 1.0,
                'mean_distracted_prob': 0.0
            }
        
        # Calculate mean probabilities for focused vs distracted classes
        probabilities = np.array([p['probabilities'] for p in self.prediction_window])
        
        mean_focused_prob = probabilities[:, self.focused_indices].mean()
        mean_distracted_prob = probabilities[:, self.distracted_indices].mean()
        
        # Compute lock-in score: S = mean(P(focused)) - mean(P(distracted))
        lock_in_score = mean_focused_prob - mean_distracted_prob
        
        # Determine if current prediction is distracted
        current_is_distracted = prediction['predicted_class'] in self.distracted_indices
        
        # Update consecutive distracted counter
        if current_is_distracted:
            self.consecutive_distracted += 1
        else:
            self.consecutive_distracted = 0
        
        # Determine state
        was_locked_in = self.is_locked_in
        
        if lock_in_score < self.threshold:
            self.is_locked_in = False
        else:
            self.is_locked_in = True
        
        # Trigger alert if:
        # 1. Score below threshold
        # 2. Consecutive distracted frames >= required
        # 3. State changed from locked in to not locked in
        trigger_alert = (
            not self.is_locked_in and 
            self.consecutive_distracted >= self.consecutive_required and
            was_locked_in
        )
        
        return {
            'lock_in_score': float(lock_in_score),
            'is_locked_in': self.is_locked_in,
            'window_size': len(self.prediction_window),
            'consecutive_distracted': self.consecutive_distracted,
            'trigger_alert': trigger_alert,
            'mean_focused_prob': float(mean_focused_prob),
            'mean_distracted_prob': float(mean_distracted_prob)
        }
    
    def reset(self):
        """Reset the scorer state"""
        self.prediction_window.clear()
        self.consecutive_distracted = 0
        self.is_locked_in = True
    
    def get_current_state(self) -> Dict:
        """Get current state without adding new prediction"""
        if len(self.prediction_window) == 0:
            return {
                'lock_in_score': 1.0,
                'is_locked_in': True,
                'window_size': 0,
                'consecutive_distracted': 0
            }
        
        probabilities = np.array([p['probabilities'] for p in self.prediction_window])
        mean_focused_prob = probabilities[:, self.focused_indices].mean()
        mean_distracted_prob = probabilities[:, self.distracted_indices].mean()
        lock_in_score = mean_focused_prob - mean_distracted_prob
        
        return {
            'lock_in_score': float(lock_in_score),
            'is_locked_in': self.is_locked_in,
            'window_size': len(self.prediction_window),
            'consecutive_distracted': self.consecutive_distracted
        }
    
    def get_window_history(self) -> List[Dict]:
        """Get all predictions in current window"""
        return list(self.prediction_window)

