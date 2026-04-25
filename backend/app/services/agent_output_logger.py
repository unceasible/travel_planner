"""Asynchronous full-output logger for heavy agent payloads."""
# -*- coding: utf-8 -*-
from __future__ import annotations

import atexit
import json
import time
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Iterator

from ..config import get_settings

_queue: Queue[str] = Queue()
_thread: Thread | None = None
_stop_event = Event()
_init_lock = Lock()


def _log_file_path() -> Path:
    base_dir = Path(__file__).resolve().parents[2] / "runtime_data" / "logs"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "agent_outputs.log"


def _should_log_full_to_console() -> bool:
    value = getattr(get_settings(), "log_verbose_agent_output", False)
    return bool(value)


def _should_log_full_to_file() -> bool:
    value = getattr(get_settings(), "log_verbose_agent_output_to_file", True)
    return bool(value)


def _writer_loop() -> None:
    log_file = _log_file_path()
    while not _stop_event.is_set() or not _queue.empty():
        try:
            item = _queue.get(timeout=0.2)
        except Empty:
            continue
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(item)
            handle.write("\n")
        _queue.task_done()


def _ensure_started() -> None:
    global _thread
    with _init_lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop_event.clear()
        _thread = Thread(target=_writer_loop, name="agent-output-logger", daemon=True)
        _thread.start()


def shutdown_agent_output_logger() -> None:
    if _thread is None:
        return
    _stop_event.set()
    _thread.join(timeout=2)


def _compact_text(content: Any) -> str:
    text = "" if content is None else str(content)
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        parsed = json.loads(stripped)
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def log_full_output(title: str, content: Any, preview_limit: int = 300) -> None:
    text = _compact_text(content)
    preview = " ".join(text.replace("\r", " ").replace("\n", " ").split())[:preview_limit]
    if _should_log_full_to_console():
        print(f"INFO {title} | bytes={len(text.encode('utf-8'))} preview={preview}", flush=True)
        divider = "-" * 120
        print(divider, flush=True)
        print(f"FULL OUTPUT BEGIN | {title}", flush=True)
        print(divider, flush=True)
        print(text, flush=True)
        print(divider, flush=True)
        print(f"FULL OUTPUT END | {title}", flush=True)
        print(divider, flush=True)
        return
    if not _should_log_full_to_file():
        return
    _ensure_started()
    divider = "-" * 120
    _queue.put(
        "\n".join(
            [
                divider,
                f"FULL OUTPUT BEGIN | {title}",
                divider,
                text,
                divider,
                f"FULL OUTPUT END | {title}",
                divider,
            ]
        )
    )


def log_event(title: str, content: Any) -> None:
    text = _compact_text(content)
    if _should_log_full_to_console():
        print(f"INFO {title} | {text}", flush=True)
    if not _should_log_full_to_file():
        return
    _ensure_started()
    divider = "-" * 120
    _queue.put(
        "\n".join(
            [
                divider,
                f"EVENT BEGIN | {title}",
                divider,
                text,
                divider,
                f"EVENT END | {title}",
                divider,
            ]
        )
    )


@contextmanager
def timed_event(stage: str, content: Any | None = None) -> Iterator[None]:
    start = time.perf_counter()
    status = "ok"
    error = ""
    try:
        yield
    except Exception as exc:
        status = "error"
        error = repr(exc)
        raise
    finally:
        payload = {
            "stage": stage,
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
            "status": status,
        }
        if content is not None:
            payload["context"] = content
        if error:
            payload["error"] = error
        log_event("timing", payload)


atexit.register(shutdown_agent_output_logger)
