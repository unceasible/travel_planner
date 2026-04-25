"""Encoding helpers for console output."""

from __future__ import annotations

import os
import sys


def configure_utf8_stdio() -> None:
    """Prefer UTF-8 for Python stdout/stderr when the stream supports it."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

