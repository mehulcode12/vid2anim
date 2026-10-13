"""Unified progress tracker supporting console and streaming JSONL for Web UI / extensions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class ProgressReporter:
    """Emits progress events to terminal and an optional JSON Lines file."""

    def __init__(self, jsonl_path: Optional[str | Path] = None, quiet: bool = False) -> None:
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.quiet = quiet
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            # Clear or initialize the file
            with open(self.jsonl_path, "w", encoding="utf-8") as f:
                pass

    def report(
        self,
        step: str,
        progress: float,
        message: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a milestone: progress is 0.0 to 1.0."""
        progress_clamped = max(0.0, min(1.0, progress))
        pct = int(progress_clamped * 100)

        if not self.quiet:
            bar_len = 30
            filled = int(bar_len * progress_clamped)
            bar = "=" * filled + "-" * (bar_len - filled)
            print(f"[{bar}] {pct:3d}% | {step}: {message}")

        if self.jsonl_path:
            payload = {
                "step": step,
                "progress": round(progress_clamped, 3),
                "percent": pct,
                "message": message,
            }
            if extra:
                payload.update(extra)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
