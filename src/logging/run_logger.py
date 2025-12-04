# src/logging/run_logger.py
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class RunLogger:
    """
    Hybrid logger:
      - JSON-lines per-agent files (agent_name.jsonl)
      - Single human-readable run_readable.log
    """

    def __init__(self, run_dir: str | Path, agent_name: Optional[str] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.agent = agent_name or "orchestrator"
        self.jsonl_path = self.run_dir / f"{self.agent}.jsonl"
        self.readable_path = self.run_dir / "run_readable.log"

    def _now(self):
        return time.time()

    def _write_jsonl(self, payload: Dict[str, Any]):
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            # Do not blow up logging on write errors
            pass

    def _write_readable(self, text: str):
        try:
            with open(self.readable_path, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except Exception:
            pass

    def info(self, payload: Dict[str, Any]):
        payload = dict(payload)
        payload.setdefault("level", "INFO")
        payload.setdefault("timestamp", self._now())
        payload.setdefault("agent", self.agent)
        self._write_jsonl(payload)
        # also write a compact human-readable line
        summary = f"[{self.agent}] INFO: {payload.get('event', '')} - { {k:v for k,v in payload.items() if k not in ['event','timestamp','agent','level']} }"
        self._write_readable(summary)

    def warning(self, payload: Dict[str, Any]):
        payload = dict(payload)
        payload.setdefault("level", "WARN")
        payload.setdefault("timestamp", self._now())
        payload.setdefault("agent", self.agent)
        self._write_jsonl(payload)
        summary = f"[{self.agent}] WARN: {payload.get('event', '')} - { {k:v for k,v in payload.items() if k not in ['event','timestamp','agent','level']} }"
        self._write_readable(summary)

    def error(self, payload: Dict[str, Any]):
        payload = dict(payload)
        payload.setdefault("level", "ERROR")
        payload.setdefault("timestamp", self._now())
        payload.setdefault("agent", self.agent)
        self._write_jsonl(payload)
        summary = f"[{self.agent}] ERROR: {payload.get('event', '')} - { {k:v for k,v in payload.items() if k not in ['event','timestamp','agent','level']} }"
        self._write_readable(summary)

    def human(self, text: str):
        """Write a purely human-readable line to the run_readable.log"""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_readable(f"{ts} | {self.agent} | {text}")
