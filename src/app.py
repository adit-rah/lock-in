"""Main application for Lock-In focus monitoring"""

import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch

from .config import load_config, Config
from .model import load_model_torchscript
from .inference import InferenceEngine
from .scoring import FocusScorer
from .signals import create_signal_handler
from .logging_db import FocusLogger


class LockInMonitor:
    """Main application class for focus monitoring"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: Path to configuration file
        """
        print("Initializing Lock-In Monitor...")
        
        # Load configuration
        self.config = load_config(config_path)
        print(f"Configuration loaded from {config_path}")
        
        # Check if model exists
        model_path = Path(self.config.model.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train a model first using 'python -m src.train --data_dir <path>'"
            )
        
        # Load model
        print(f"Loading model from {self.config.model.model_path}...")
        self.model = load_model_torchscript(self.config.model.model_path)
        print("Model loaded successfully")
        
        # Initialize components
        print("Initializing inference engine...")
        self.inference_engine = InferenceEngine(self.model, self.config)
        
        print("Initializing scorer...")
        self.scorer = FocusScorer(self.config)
        
        print("Initializing signal handler...")
        self.signal_handler = create_signal_handler(self.config)
        
        print("Initializing logger...")
        self.logger = FocusLogger(self.config)
        
        # Session tracking
        self.session_id = None
        self.running = False
        
        # Statistics
        self.frame_count = 0
        self.alert_count = 0
        
        print("Lock-In Monitor initialized successfully!")
    
    def start(self):
        """Start monitoring"""
        print("\n" + "="*60)
        print("Starting Lock-In Focus Monitoring")
        print("="*60)
        print(f"Frame interval: {self.config.inference.frame_interval_seconds}s")
        print(f"Rolling window: {self.config.scoring.rolling_window_size} frames")
        print(f"Alert threshold: {self.config.scoring.alert_threshold}")
        print(f"Classes: {', '.join(self.config.classes)}")
        print("\nPress Ctrl+C to stop monitoring")
        print("="*60 + "\n")
        
        # Start session
        self.session_id = self.logger.start_session()
        self.running = True
        
        # Main monitoring loop
        try:
            while self.running:
                self._process_frame()
                time.sleep(self.config.inference.frame_interval_seconds)
        except KeyboardInterrupt:
            print("\n\nStopping monitoring...")
        finally:
            self.stop()
    
    def _process_frame(self):
        """Process a single frame"""
        try:
            # Capture and predict
            prediction = self.inference_engine.predict_frame()
            self.frame_count += 1
            
            # Update score
            score_data = self.scorer.add_prediction(prediction)
            
            # Log
            self.logger.log_prediction(self.session_id, prediction)
            self.logger.log_score(self.session_id, prediction['timestamp'], score_data)
            self.logger.log_to_csv(prediction, score_data)
            
            # Display status
            self._display_status(prediction, score_data)
            
            # Check for alert
            if score_data['trigger_alert']:
                self._handle_alert(prediction, score_data)
        
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    def _display_status(self, prediction: Dict, score_data: Dict):
        """Display current status"""
        timestamp = prediction['timestamp'].strftime("%H:%M:%S")
        predicted_class = prediction['predicted_class_name']
        confidence = prediction['confidence'] * 100
        score = score_data['lock_in_score']
        state = "🔒 LOCKED IN" if score_data['is_locked_in'] else "⚠️  DISTRACTED"
        
        print(f"[{timestamp}] Frame {self.frame_count:4d} | "
              f"Class: {predicted_class:15s} ({confidence:5.1f}%) | "
              f"Score: {score:+.3f} | "
              f"Status: {state}")
    
    def _handle_alert(self, prediction: Dict, score_data: Dict):
        """Handle distraction alert"""
        self.alert_count += 1
        
        # Log event
        self.logger.log_event(
            self.session_id, 
            'distracted', 
            f"Alert {self.alert_count}: Score={score_data['lock_in_score']:.3f}"
        )
        
        # Trigger signal
        event = {
            'event_type': 'distracted',
            'timestamp': prediction['timestamp'],
            'score_data': score_data,
            'prediction_data': prediction,
            'alert_number': self.alert_count
        }
        
        self.signal_handler.trigger(event)
        
        print(f"\n{'!'*60}")
        print(f"⚠️  ALERT {self.alert_count}: DISTRACTION DETECTED!")
        print(f"Lock-in score: {score_data['lock_in_score']:.3f}")
        print(f"Consecutive distracted frames: {score_data['consecutive_distracted']}")
        print(f"{'!'*60}\n")
    
    def stop(self):
        """Stop monitoring and cleanup"""
        self.running = False
        
        if self.session_id is not None:
            self.logger.end_session(self.session_id)
            
            # Display summary
            summary = self.logger.get_session_summary(self.session_id)
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Session ID: {summary['session_id']}")
            print(f"Duration: {summary['start_time']} to {summary['end_time']}")
            print(f"Total frames: {summary['total_frames']}")
            print(f"Focused frames: {summary['focused_frames']}")
            print(f"Distracted frames: {summary['distracted_frames']}")
            print(f"Focus ratio: {summary['focus_ratio']*100:.1f}%")
            print(f"Alerts triggered: {self.alert_count}")
            print("="*60)
        
        # Cleanup
        self.inference_engine.release()
        print("\nMonitoring stopped. Goodbye!")
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'inference_engine'):
            self.inference_engine.release()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Lock-In Focus Monitor - Monitor your focus in real-time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start monitoring with default config
  python -m src.app
  
  # Start with custom config
  python -m src.app --config my_config.yaml
  
  # View recent sessions
  python -m src.app --view-sessions
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--view-sessions',
        action='store_true',
        help='View recent session summaries and exit'
    )
    
    args = parser.parse_args()
    
    if args.view_sessions:
        # Just view sessions
        config = load_config(args.config)
        logger = FocusLogger(config)
        sessions = logger.get_recent_sessions(limit=10)
        
        print("\n" + "="*60)
        print("RECENT SESSIONS")
        print("="*60)
        
        for session in sessions:
            print(f"\nSession {session['session_id']}:")
            print(f"  Start: {session['start_time']}")
            print(f"  End: {session['end_time']}")
            print(f"  Frames: {session['total_frames']}")
            print(f"  Focus ratio: {session['focus_ratio']*100:.1f}%")
        
        print()
        return
    
    # Start monitoring
    monitor = LockInMonitor(config_path=args.config)
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal...")
        monitor.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start monitoring
    monitor.start()


if __name__ == "__main__":
    main()

