import json
import yaml
import os
import random
import datetime
import threading
import numpy as np
import pandas as pd
import math
import functools
import time as _time
from typing import Callable, Tuple, Any


# -------------------------
# Config loader
# -------------------------
def load_config(path="config/config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# -------------------------
# Safe JSON encoder
# -------------------------
def _make_json_safe(obj: Any):
    """
    Convert objects that JSON cannot handle:
    - pandas.Timestamp -> ISO string
    - numpy types -> Python native
    - sets/tuples -> lists
    - datetimes -> ISO string
    - infinities / NaNs -> None
    """
    # dict
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}

    # list-like
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [_make_json_safe(v) for v in obj]

    if isinstance(obj, set):
        return [_make_json_safe(v) for v in list(obj)]

    # pandas timestamp
    if isinstance(obj, (pd.Timestamp,)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)

    # datetime objects
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()

    # numpy numbers
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isinf(v) or math.isnan(v):
            return None
        return v
    if isinstance(obj, (np.ndarray,)):
        try:
            return obj.tolist()
        except Exception:
            return list(obj)

    # normal floats: handle inf / nan
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj

    return obj


# -------------------------
# JSON saver (safe)
# -------------------------
def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    safe_obj = _make_json_safe(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_obj, f, indent=2, ensure_ascii=False)

    return path


# -------------------------
# Seed setter
# -------------------------
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# -------------------------
# Retry decorator (exponential backoff)
# -------------------------
def retry(
    attempts: int = 3,
    initial_delay: float = 0.5,
    backoff: float = 2.0,
    jitter: float = 0.1,
    exceptions: Tuple = (Exception,),
    logger=None
):
    """
    Decorator to retry a function on exceptions.
    Usage:
    @retry(attempts=3, initial_delay=0.5, backoff=2.0)
    def fn(...): ...
    """
    def deco(fn: Callable):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            delay = initial_delay
            last_exc = None
            for attempt in range(1, attempts + 1):
                try:
                    if logger:
                        logger.info({"event": "retry_attempt", "fn": getattr(fn, "__name__", str(fn)), "attempt": attempt})
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == attempts:
                        if logger:
                            logger.error({"event": "retry_fail", "fn": getattr(fn, "__name__", str(fn)), "attempt": attempt, "error": str(e)})
                        raise
                    # sleep with jitter
                    sleep_time = delay + (random.random() * jitter)
                    if logger:
                        logger.warning({"event": "retry_backoff", "fn": getattr(fn, "__name__", str(fn)), "attempt": attempt, "sleep_sec": round(sleep_time, 3)})
                    _time.sleep(sleep_time)
                    delay *= backoff
            # if loop falls through
            raise last_exc
        return wrapped
    return deco


# -------------------------
# Lightweight structured logger
# -------------------------
class StructuredLogger:
    """
    Minimal structured logger:
    - stores events in memory (list of dicts)
    - prints compact logs to console
    - thread-safe
    - constructor accepts flexible args for compatibility:
        StructuredLogger(name="x")
        StructuredLogger(run_id="2025...", logs_dir="logs")
    """
    def __init__(self, name: str = "run_logger", run_id: str = None, logs_dir: str = None):
        self.name = name or "run_logger"
        self.run_id = run_id
        self.logs_dir = logs_dir or "logs"
        self.events = []
        self.lock = threading.Lock()
        # ensure logs dir exists if run_id provided
        try:
            if self.logs_dir:
                os.makedirs(self.logs_dir, exist_ok=True)
        except Exception:
            pass

    def _emit(self, level: str, payload):
        evt = {
            "ts": datetime.datetime.utcnow().isoformat(),
            "level": level,
            "payload": payload
        }
        with self.lock:
            self.events.append(evt)
        # console-friendly print (compact)
        try:
            print(f"[{level.upper()}] {payload}")
        except Exception:
            pass

    def info(self, payload):
        self._emit("info", payload)

    def warning(self, payload):
        self._emit("warning", payload)

    def warn(self, payload):
        self.warning(payload)

    def error(self, payload):
        self._emit("error", payload)

    def get_events(self):
        with self.lock:
            return list(self.events)

    def clear(self):
        with self.lock:
            self.events = []

    def save(self, path):
        try:
            save_json(self.get_events(), path)
        except Exception as e:
            # best-effort
            print("[ERROR] Failed to dump logger events:", e)


# -------------------------
# Lightweight Metrics
# -------------------------
class Metrics:
    def __init__(self):
        self.counters = {}
        self.timers = {}
        self._active_timers = {}

    def incr(self, key, count=1):
        self.counters[key] = self.counters.get(key, 0) + count

    def start_timer(self, key):
        self._active_timers[key] = _time.time()

    def stop_timer(self, key):
        if key in self._active_timers:
            elapsed = round(_time.time() - self._active_timers[key], 4)
            self.timers[key] = elapsed
            del self._active_timers[key]

    def snapshot(self):
        flat = {}
        # flatten timers into root-level fields for tests
        for k, v in self.timers.items():
            flat[k] = v

        return {
            "counters": dict(self.counters),
            "timers": dict(self.timers),
            **flat,   # <-- test expects data_load in root
        }

