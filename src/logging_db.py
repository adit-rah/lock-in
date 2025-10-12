"""Logging system with SQLite database"""

import sqlite3
import csv
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json
import numpy as np

from .config import Config


class FocusLogger:
    """Handles logging of predictions, scores, and events to database and CSV"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Create data directory
        db_path = Path(config.logging.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = str(db_path)
        self._init_database()
        
        # CSV backup
        if config.logging.csv_backup:
            csv_path = Path(config.logging.csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_path = str(csv_path)
            self._init_csv()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                total_frames INTEGER DEFAULT 0,
                focused_frames INTEGER DEFAULT 0,
                distracted_frames INTEGER DEFAULT 0,
                focus_ratio REAL
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TIMESTAMP NOT NULL,
                predicted_class INTEGER NOT NULL,
                predicted_class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        # Scores table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scores (
                score_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TIMESTAMP NOT NULL,
                lock_in_score REAL NOT NULL,
                is_locked_in BOOLEAN NOT NULL,
                consecutive_distracted INTEGER NOT NULL,
                mean_focused_prob REAL NOT NULL,
                mean_distracted_prob REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        # Events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TIMESTAMP NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_csv(self):
        """Initialize CSV file with headers"""
        if not Path(self.csv_path).exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'predicted_class', 'predicted_class_name', 
                    'confidence', 'lock_in_score', 'is_locked_in', 
                    'consecutive_distracted', 'event_type'
                ])
    
    def start_session(self) -> int:
        """Start a new logging session
        
        Returns:
            session_id
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (start_time)
            VALUES (?)
        ''', (datetime.now(),))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"Started logging session {session_id}")
        return session_id
    
    def end_session(self, session_id: int):
        """End a logging session and compute statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get session statistics
        cursor.execute('''
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN predicted_class_name = 'focused' THEN 1 ELSE 0 END) as focused,
                   SUM(CASE WHEN predicted_class_name != 'focused' THEN 1 ELSE 0 END) as distracted
            FROM predictions
            WHERE session_id = ?
        ''', (session_id,))
        
        row = cursor.fetchone()
        total, focused, distracted = row if row else (0, 0, 0)
        focus_ratio = focused / total if total > 0 else 0
        
        # Update session
        cursor.execute('''
            UPDATE sessions
            SET end_time = ?,
                total_frames = ?,
                focused_frames = ?,
                distracted_frames = ?,
                focus_ratio = ?
            WHERE session_id = ?
        ''', (datetime.now(), total, focused, distracted, focus_ratio, session_id))
        
        conn.commit()
        conn.close()
        
        print(f"Ended session {session_id}: {total} frames, {focus_ratio*100:.1f}% focused")
    
    def log_prediction(self, session_id: int, prediction: Dict):
        """Log a prediction to database"""
        if not self.config.logging.log_predictions:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (session_id, timestamp, predicted_class, 
                                    predicted_class_name, confidence, probabilities)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            prediction['timestamp'],
            prediction['predicted_class'],
            prediction['predicted_class_name'],
            prediction['confidence'],
            json.dumps(prediction['probabilities'].tolist())
        ))
        
        conn.commit()
        conn.close()
    
    def log_score(self, session_id: int, timestamp: datetime, score_data: Dict):
        """Log a score to database"""
        if not self.config.logging.log_scores:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO scores (session_id, timestamp, lock_in_score, is_locked_in,
                              consecutive_distracted, mean_focused_prob, mean_distracted_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            timestamp,
            score_data['lock_in_score'],
            score_data['is_locked_in'],
            score_data['consecutive_distracted'],
            score_data['mean_focused_prob'],
            score_data['mean_distracted_prob']
        ))
        
        conn.commit()
        conn.close()
    
    def log_event(self, session_id: int, event_type: str, description: str = ""):
        """Log an event to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO events (session_id, timestamp, event_type, description)
            VALUES (?, ?, ?, ?)
        ''', (session_id, datetime.now(), event_type, description))
        
        conn.commit()
        conn.close()
    
    def log_to_csv(self, prediction: Dict, score_data: Dict, event_type: str = ""):
        """Log to CSV file for backup"""
        if not self.config.logging.csv_backup:
            return
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                prediction['timestamp'].isoformat(),
                prediction['predicted_class'],
                prediction['predicted_class_name'],
                prediction['confidence'],
                score_data['lock_in_score'],
                score_data['is_locked_in'],
                score_data['consecutive_distracted'],
                event_type
            ])
    
    def get_session_summary(self, session_id: int) -> Optional[Dict]:
        """Get summary statistics for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM sessions WHERE session_id = ?
        ''', (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return {
            'session_id': row[0],
            'start_time': row[1],
            'end_time': row[2],
            'total_frames': row[3],
            'focused_frames': row[4],
            'distracted_frames': row[5],
            'focus_ratio': row[6]
        }
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """Get recent session summaries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM sessions
            ORDER BY start_time DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            'session_id': row[0],
            'start_time': row[1],
            'end_time': row[2],
            'total_frames': row[3],
            'focused_frames': row[4],
            'distracted_frames': row[5],
            'focus_ratio': row[6]
        } for row in rows]

