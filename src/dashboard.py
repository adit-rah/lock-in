"""Streamlit dashboard for Lock-In: live focus monitoring + session history.

Run with:
    streamlit run src/dashboard.py

The Live tab spawns a background worker that reuses the existing
InferenceEngine / FocusScorer / FocusLogger pipeline. The History tab reads
sessions and per-session score timeseries directly from the SQLite database
written by both this dashboard and the CLI app.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
import streamlit as st

from src.config import Config, load_config
from src.inference import InferenceEngine
from src.logging_db import FocusLogger
from src.model import load_model_torchscript
from src.scoring import FocusScorer
from src.signals import create_signal_handler


CONFIG_PATH = "config.yaml"


# -------------------- background worker --------------------


class LiveWorker:
    """Runs the inference + scoring loop on a thread. UI thread polls the latest state."""

    def __init__(self, config: Config):
        self.config = config
        self.model = load_model_torchscript(config.model.model_path)
        self.engine = InferenceEngine(self.model, config)
        self.scorer = FocusScorer(config)
        self.signal_handler = create_signal_handler(config)
        self.logger = FocusLogger(config)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.session_id: Optional[int] = None
        self.latest_frame = None
        self.latest_prediction = None
        self.latest_score = None
        self.score_history: deque = deque(maxlen=120)
        self.frame_count = 0
        self.alert_count = 0
        self.start_time: Optional[datetime] = None
        self.inference_times: deque = deque(maxlen=30)

    def start(self):
        self._stop.clear()
        self.session_id = self.logger.start_session()
        self.start_time = datetime.now()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.session_id is not None:
            self.logger.end_session(self.session_id)
        self.engine.release()

    def update_scoring_params(self, window_size: int, threshold: float):
        with self._lock:
            self.scorer.window_size = window_size
            self.scorer.prediction_window = deque(self.scorer.prediction_window, maxlen=window_size)
            self.scorer.threshold = threshold

    def snapshot(self) -> dict:
        with self._lock:
            return {
                'session_id': self.session_id,
                'frame_count': self.frame_count,
                'alert_count': self.alert_count,
                'latest_frame': self.latest_frame,
                'latest_prediction': self.latest_prediction,
                'latest_score': self.latest_score,
                'score_history': list(self.score_history),
                'start_time': self.start_time,
                'mean_inference_ms': (
                    sum(self.inference_times) / len(self.inference_times)
                    if self.inference_times else None
                ),
            }

    def _run(self):
        interval = self.config.inference.frame_interval_seconds
        while not self._stop.is_set():
            try:
                t0 = time.time()
                prediction = self.engine.predict_frame()
                inference_ms = (time.time() - t0) * 1000

                with self._lock:
                    score_data = self.scorer.add_prediction(prediction)
                    self.latest_frame = prediction['frame']
                    self.latest_prediction = prediction
                    self.latest_score = score_data
                    self.score_history.append({
                        'timestamp': prediction['timestamp'],
                        'lock_in_score': score_data['lock_in_score'],
                        'is_locked_in': score_data['is_locked_in'],
                        'predicted_class_name': prediction['predicted_class_name'],
                    })
                    self.frame_count += 1
                    self.inference_times.append(inference_ms)

                self.logger.log_prediction(self.session_id, prediction)
                self.logger.log_score(self.session_id, prediction['timestamp'], score_data)
                self.logger.log_to_csv(prediction, score_data)

                if score_data['trigger_alert']:
                    with self._lock:
                        self.alert_count += 1
                    self.logger.log_event(
                        self.session_id, 'distracted',
                        f"Score={score_data['lock_in_score']:.3f}",
                    )
                    self.signal_handler.trigger({
                        'event_type': 'distracted',
                        'timestamp': prediction['timestamp'],
                        'score_data': score_data,
                        'prediction_data': prediction,
                        'alert_number': self.alert_count,
                    })
            except Exception as exc:  # noqa: BLE001 — keep the worker alive on transient errors
                print(f"[worker] frame error: {exc}")

            # Sleep but stay responsive to stop().
            self._stop.wait(interval)


# -------------------- helpers --------------------


@st.cache_data(show_spinner=False)
def _load_sessions(db_path: str) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT session_id, start_time, end_time, total_frames, focused_frames, "
            "distracted_frames, focus_ratio FROM sessions ORDER BY start_time DESC",
            conn,
        )
    finally:
        conn.close()
    return df


@st.cache_data(show_spinner=False)
def _load_session_scores(db_path: str, session_id: int) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT timestamp, lock_in_score, is_locked_in, consecutive_distracted "
            "FROM scores WHERE session_id = ? ORDER BY timestamp",
            conn, params=(session_id,),
        )
    finally:
        conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_data(show_spinner=False)
def _load_session_predictions(db_path: str, session_id: int) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT timestamp, predicted_class_name, confidence "
            "FROM predictions WHERE session_id = ? ORDER BY timestamp",
            conn, params=(session_id,),
        )
    finally:
        conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


# -------------------- UI --------------------


def render_live_tab(config: Config):
    st.subheader("Live monitoring")

    worker: Optional[LiveWorker] = st.session_state.get('worker')

    col1, col2, col3 = st.columns(3)
    start_disabled = worker is not None
    stop_disabled = worker is None

    if col1.button("Start", disabled=start_disabled, type="primary"):
        try:
            st.session_state['worker'] = LiveWorker(config)
            st.session_state['worker'].start()
            _load_sessions.clear()
            st.rerun()
        except FileNotFoundError as exc:
            st.error(str(exc))

    if col2.button("Stop", disabled=stop_disabled):
        worker.stop()
        st.session_state['worker'] = None
        _load_sessions.clear()
        st.rerun()

    col3.caption("Settings update live without restarting the session.")
    window_size = st.slider(
        "Rolling window size", min_value=2, max_value=60,
        value=config.scoring.rolling_window_size,
    )
    threshold = st.slider(
        "Alert threshold", min_value=-1.0, max_value=1.0,
        value=float(config.scoring.alert_threshold), step=0.05,
    )

    if worker is not None:
        worker.update_scoring_params(window_size, threshold)
        snap = worker.snapshot()

        frame_col, stats_col = st.columns([2, 1])
        with frame_col:
            if snap['latest_frame'] is not None:
                rgb = cv2.cvtColor(snap['latest_frame'], cv2.COLOR_BGR2RGB)
                st.image(rgb, caption=f"Frame {snap['frame_count']}", use_container_width=True)
            else:
                st.info("Warming up camera...")

        with stats_col:
            score = snap['latest_score']
            pred = snap['latest_prediction']
            if score is not None:
                pill = "🔒 Locked in" if score['is_locked_in'] else "⚠️ Distracted"
                st.metric("Lock-in score", f"{score['lock_in_score']:+.3f}", pill)
                st.metric("Class", pred['predicted_class_name'])
                st.metric("Confidence", f"{pred['confidence']*100:.1f}%")
            st.metric("Alerts", snap['alert_count'])
            if snap['mean_inference_ms'] is not None:
                st.metric("Avg inference", f"{snap['mean_inference_ms']:.0f} ms")

        if snap['score_history']:
            df = pd.DataFrame(snap['score_history']).set_index('timestamp')
            st.line_chart(df['lock_in_score'])

        # Auto-refresh while monitoring.
        time.sleep(1)
        st.rerun()
    else:
        st.info("Press Start to begin monitoring.")


def render_history_tab(config: Config):
    st.subheader("Session history")

    db_path = config.logging.database_path
    sessions = _load_sessions(db_path)

    if sessions.empty:
        st.info("No sessions yet. Start a live session to populate this view.")
        return

    sessions_display = sessions.copy()
    sessions_display['focus_ratio_pct'] = (sessions_display['focus_ratio'] * 100).round(1)
    st.dataframe(
        sessions_display[['session_id', 'start_time', 'end_time', 'total_frames',
                          'focused_frames', 'distracted_frames', 'focus_ratio_pct']],
        use_container_width=True, hide_index=True,
    )

    session_ids = sessions['session_id'].tolist()
    chosen = st.selectbox("Drill into a session", session_ids, index=0)
    if chosen is None:
        return

    scores = _load_session_scores(db_path, int(chosen))
    preds = _load_session_predictions(db_path, int(chosen))

    if scores.empty and preds.empty:
        st.warning("No frames were recorded for this session.")
        return

    if not scores.empty:
        st.markdown("**Lock-in score over time**")
        st.line_chart(scores.set_index('timestamp')['lock_in_score'])

    if not preds.empty:
        st.markdown("**Class distribution**")
        counts = preds['predicted_class_name'].value_counts()
        st.bar_chart(counts)


def main():
    st.set_page_config(page_title="Lock-In", page_icon="🔒", layout="wide")
    st.title("🔒 Lock-In Focus Monitor")

    try:
        config = load_config(CONFIG_PATH)
    except FileNotFoundError:
        st.error(f"Config file not found: {CONFIG_PATH}")
        return

    if not Path(config.model.model_path).exists():
        st.error(
            f"Model not found at {config.model.model_path}. "
            "Train a model first (`python -m src.train --data_dir ...`) "
            "or run `python scripts/download_model.py` to fetch the release checkpoint."
        )
        return

    tab_live, tab_history = st.tabs(["Live", "History"])
    with tab_live:
        render_live_tab(config)
    with tab_history:
        render_history_tab(config)


if __name__ == "__main__":
    main()
