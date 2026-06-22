# ============================================================================
# analysis/services/logging_service.py
# Structured Logging & Monitoring Service
# ============================================================================

from __future__ import annotations

import json
import logging
import os
import time
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Log directory setup
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(os.path.dirname(_BASE_DIR), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

PREDICTION_LOG_FILE = os.path.join(_LOG_DIR, "predictions.jsonl")
ERROR_LOG_FILE = os.path.join(_LOG_DIR, "errors.jsonl")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PredictionLogEntry:
    """Structured prediction log entry."""
    timestamp: str
    disease: str
    confidence: float
    is_healthy: bool
    is_low_confidence: bool
    inference_time_ms: float
    gradcam_time_ms: float = 0.0
    image_size_bytes: int = 0
    stage: str = ""
    disease_ratio: float = 0.0
    user_id: Optional[int] = None
    session_id: str = ""


# ============================================================================
# Performance Tracker
# ============================================================================

class PerformanceTracker:
    """Thread-safe tracker for inference and request performance metrics."""

    def __init__(self, max_samples: int = 500):
        self._lock = threading.Lock()
        self._inference_times: deque = deque(maxlen=max_samples)
        self._gradcam_times: deque = deque(maxlen=max_samples)
        self._request_times: deque = deque(maxlen=max_samples)
        self._prediction_count = 0
        self._error_count = 0
        self._start_time = time.time()

    def record_inference(self, duration_ms: float) -> None:
        with self._lock:
            self._inference_times.append(duration_ms)
            self._prediction_count += 1

    def record_gradcam(self, duration_ms: float) -> None:
        with self._lock:
            self._gradcam_times.append(duration_ms)

    def record_request(self, duration_ms: float) -> None:
        with self._lock:
            self._request_times.append(duration_ms)

    def record_error(self) -> None:
        with self._lock:
            self._error_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self._lock:
            def _avg(d: deque) -> float:
                return round(sum(d) / len(d), 2) if d else 0.0

            def _p95(d: deque) -> float:
                if not d:
                    return 0.0
                sorted_vals = sorted(d)
                idx = int(len(sorted_vals) * 0.95)
                return round(sorted_vals[min(idx, len(sorted_vals) - 1)], 2)

            uptime_seconds = time.time() - self._start_time

            return {
                "predictions_total": self._prediction_count,
                "errors_total": self._error_count,
                "uptime_seconds": round(uptime_seconds, 0),
                "inference": {
                    "avg_ms": _avg(self._inference_times),
                    "p95_ms": _p95(self._inference_times),
                    "samples": len(self._inference_times),
                },
                "gradcam": {
                    "avg_ms": _avg(self._gradcam_times),
                    "p95_ms": _p95(self._gradcam_times),
                    "samples": len(self._gradcam_times),
                },
                "requests": {
                    "avg_ms": _avg(self._request_times),
                    "p95_ms": _p95(self._request_times),
                    "samples": len(self._request_times),
                },
            }


# ============================================================================
# Prediction Logger
# ============================================================================

class PredictionLogger:
    """Appends structured prediction logs to a JSONL file."""

    def __init__(self, log_file: str = PREDICTION_LOG_FILE):
        self._log_file = log_file
        self._lock = threading.Lock()

    def log(self, entry: PredictionLogEntry) -> None:
        """Append a prediction log entry."""
        try:
            line = json.dumps(asdict(entry), ensure_ascii=False) + "\n"
            with self._lock:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:
            logger.error("Failed to write prediction log: %s", e)

    def get_recent(self, n: int = 50) -> List[Dict[str, Any]]:
        """Read the last N prediction log entries."""
        entries: List[Dict[str, Any]] = []
        try:
            if not os.path.exists(self._log_file):
                return entries
            with open(self._log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-n:]:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        except Exception as e:
            logger.error("Failed to read prediction log: %s", e)
        return entries


# ============================================================================
# Error Logger
# ============================================================================

def log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    log_file: str = ERROR_LOG_FILE,
) -> None:
    """
    Log a structured error entry to the error log.

    Args:
        error: The exception that occurred
        context: Optional dictionary of contextual information
        log_file: Path to the error log file
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "error_type": type(error).__name__,
        "message": str(error),
        "context": context or {},
    }
    try:
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        logger.error("Failed to write error log: %s", e)


# ============================================================================
# Module-level Singletons
# ============================================================================

performance_tracker = PerformanceTracker()
prediction_logger = PredictionLogger()
