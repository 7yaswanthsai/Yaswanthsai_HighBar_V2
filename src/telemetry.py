# src/telemetry.py
from __future__ import annotations
import time
import json
from pathlib import Path
from typing import Dict, Any


class MetricsCollector:
    """
    Lightweight in-memory metrics collector.
    Use .counter(name, delta), .gauge(name, value), .timing(name, seconds)
    and call .emit(dest_dir) to write metrics.json into a run folder.
    """

    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "counters": {},
            "gauges": {},
            "timings": {},
            "meta": {},
        }

    def counter(self, name: str, delta: int = 1) -> None:
        self.metrics["counters"][name] = self.metrics["counters"].get(name, 0) + int(delta)

    def gauge(self, name: str, value: float) -> None:
        self.metrics["gauges"][name] = float(value)

    def timing(self, name: str, seconds: float) -> None:
        self.metrics["timings"][name] = float(seconds)

    def set_meta(self, key: str, value: Any) -> None:
        self.metrics["meta"][key] = value

    def emit(self, dest_dir: str | Path, filename: str = "metrics.json") -> Path:
        d = Path(dest_dir)
        d.mkdir(parents=True, exist_ok=True)
        out = d / filename
        out.write_text(json.dumps(self.metrics, indent=2))
        return out
