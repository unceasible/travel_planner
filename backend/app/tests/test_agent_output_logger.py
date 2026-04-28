# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
import shutil
from pathlib import Path
from queue import Queue
from uuid import uuid4

from app.services import agent_output_logger as logger


def _workspace_tmp_dir() -> Path:
    path = Path.cwd() / "runtime_data" / "test_logs" / "agent_output_logger" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def _reset_logger(monkeypatch, tmp_path=None) -> None:
    logger.shutdown_agent_output_logger()
    monkeypatch.setattr(logger, "_queue", Queue())
    monkeypatch.setattr(logger, "_thread", None)
    logger._stop_event.clear()
    monkeypatch.setattr(logger, "_should_log_full_to_console", lambda: False)
    monkeypatch.setattr(logger, "_should_log_full_to_file", lambda: True)
    if tmp_path is not None:
        monkeypatch.setattr(logger, "_log_file_path", lambda: tmp_path / "agent_outputs.log")


def test_log_event_does_not_compact_payload_on_caller_thread(monkeypatch):
    _reset_logger(monkeypatch)
    caller_thread_id = threading.get_ident()

    def compact_text(content):
        if threading.get_ident() == caller_thread_id:
            raise AssertionError("log_event compacted payload on caller thread")
        return "compact"

    monkeypatch.setattr(logger, "_compact_text", compact_text)
    monkeypatch.setattr(logger, "_ensure_started", lambda: None)

    logger.log_event("perf_test", {"payload": list(range(5000))})


def test_timed_event_persists_detailed_timing_metadata(monkeypatch):
    tmp_dir = _workspace_tmp_dir()
    try:
        _reset_logger(monkeypatch, tmp_dir)

        with logger.timed_event("unit.stage", {"case": "logger"}):
            pass

        logger.shutdown_agent_output_logger()
        text = (tmp_dir / "agent_outputs.log").read_text(encoding="utf-8")

        assert "unit.stage" in text
        assert "elapsed_ms" in text
        assert "started_at" in text
        assert "finished_at" in text
        assert "thread_name" in text
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
