"""Test FocusScorer under the binary (focused/distracted) configuration."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.scoring import FocusScorer


def _binary_config():
    config = Config()
    config.classes = ['focused', 'distracted']
    config.distracted_classes = ['distracted']
    return config


def test_indices_resolve_to_binary():
    scorer = FocusScorer(_binary_config())
    assert scorer.focused_indices == [0]
    assert scorer.distracted_indices == [1]


def test_lock_in_score_equals_focused_minus_distracted_for_binary():
    scorer = FocusScorer(_binary_config())

    # Two predictions, both heavily focused
    scorer.add_prediction({'predicted_class': 0, 'probabilities': np.array([0.9, 0.1])})
    result = scorer.add_prediction({'predicted_class': 0, 'probabilities': np.array([0.8, 0.2])})

    # Mean focused prob = 0.85, mean distracted prob = 0.15
    assert result['mean_focused_prob'] == pytest.approx(0.85)
    assert result['mean_distracted_prob'] == pytest.approx(0.15)
    assert result['lock_in_score'] == pytest.approx(0.7)
    assert result['is_locked_in'] is True
    assert result['consecutive_distracted'] == 0


def test_consecutive_distracted_triggers_alert():
    """Alert fires when the consecutive-distracted streak reaches the threshold AT THE SAME TIME
    as the smoothed score drops below the alert threshold. The streak has to build up while
    the rolling score is still positive."""
    config = _binary_config()
    config.scoring.rolling_window_size = 3
    config.scoring.alert_threshold = 0.0
    config.scoring.consecutive_frames_required = 2
    scorer = FocusScorer(config)

    # Frame 1: focused (returns immediately because window has only 1 entry).
    scorer.add_prediction({'predicted_class': 0, 'probabilities': np.array([0.9, 0.1])})

    # Frame 2: mildly distracted. Score still positive — locked-in remains True.
    # Window mean focused = (0.9 + 0.4)/2 = 0.65; distracted = 0.35; S = 0.3.
    r1 = scorer.add_prediction({'predicted_class': 1, 'probabilities': np.array([0.4, 0.6])})
    assert r1['consecutive_distracted'] == 1
    assert r1['is_locked_in'] is True
    assert r1['trigger_alert'] is False

    # Frame 3: strongly distracted. Score tips below 0 and streak hits required.
    # Window mean focused = (0.9+0.4+0.1)/3 ≈ 0.467; distracted ≈ 0.533; S ≈ -0.067.
    r2 = scorer.add_prediction({'predicted_class': 1, 'probabilities': np.array([0.1, 0.9])})
    assert r2['consecutive_distracted'] == 2
    assert r2['is_locked_in'] is False
    assert r2['trigger_alert'] is True
