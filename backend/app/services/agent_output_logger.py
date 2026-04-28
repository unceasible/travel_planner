"""Asynchronous full-output logger for heavy agent payloads."""
# -*- coding: utf-8 -*-
from __future__ import annotations

import atexit
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread, current_thread, get_ident
from typing import Any, Iterator

from ..config import get_settings


@dataclass
class _LogRecord:
    kind: str
    title: str
    content: Any
    preview_limit: int = 300
    enqueued_at: str = ""
    enqueued_perf_counter: float = 0.0
    caller_thread: str = ""
    caller_thread_id: int = 0


_queue: Queue[_LogRecord | str] = Queue()
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


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _safe_queue_size() -> int:
    try:
        return _queue.qsize()
    except Exception:
        return -1


def _format_record(item: _LogRecord | str) -> str:
    if isinstance(item, str):
        return item

    format_start = time.perf_counter()
    try:
        text = _compact_text(item.content)
    except Exception as exc:
        text = f"<log formatting failed: {repr(exc)}>"
    format_elapsed_ms = round((time.perf_counter() - format_start) * 1000, 2)
    label = "FULL OUTPUT" if item.kind == "full" else "EVENT"
    metadata = {
        "kind": item.kind,
        "title": item.title,
        "enqueued_at": item.enqueued_at,
        "written_at": _timestamp(),
        "queue_wait_ms": round((time.perf_counter() - item.enqueued_perf_counter) * 1000, 2),
        "format_elapsed_ms": format_elapsed_ms,
        "content_bytes": len(text.encode("utf-8")),
        "caller_thread": item.caller_thread,
        "caller_thread_id": item.caller_thread_id,
        "writer_thread": current_thread().name,
        "writer_thread_id": get_ident(),
        "pid": os.getpid(),
        "queue_depth_after_get": _safe_queue_size(),
    }
    divider = "-" * 120
    return "\n".join(
        [
            divider,
            f"{label} BEGIN | {item.title}",
            f"LOG METADATA | {json.dumps(metadata, ensure_ascii=False, separators=(',', ':'))}",
            divider,
            text,
            divider,
            f"{label} END | {item.title}",
            divider,
        ]
    )


def _writer_loop() -> None:
    log_file = _log_file_path()
    while not _stop_event.is_set() or not _queue.empty():
        try:
            first = _queue.get(timeout=0.2)
        except Empty:
            continue
        batch = [first]
        while len(batch) < 50:
            try:
                batch.append(_queue.get_nowait())
            except Empty:
                break
        try:
            rendered = [_format_record(item) for item in batch]
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(rendered))
                handle.write("\n")
        except Exception as exc:
            print(f"agent_output_logger write failed: {exc}", flush=True)
        finally:
            for _ in batch:
                _queue.task_done()


def _ensure_started() -> None:
    global _thread
    with _init_lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop_event.clear()
        _thread = Thread(target=_writer_loop, name="agent-output-logger", daemon=True)
        _thread.start()


def start_agent_output_logger() -> None:
    if _should_log_full_to_file():
        _ensure_started()


def flush_agent_output_logger(timeout: float = 2.0) -> bool:
    deadline = time.perf_counter() + max(0.0, timeout)
    while time.perf_counter() < deadline:
        if getattr(_queue, "unfinished_tasks", 0) == 0:
            return True
        time.sleep(0.01)
    return getattr(_queue, "unfinished_tasks", 0) == 0


def shutdown_agent_output_logger() -> None:
    global _thread
    if _thread is None:
        return
    _stop_event.set()
    _thread.join(timeout=2)
    if not _thread.is_alive():
        _thread = None


def _compact_text(content: Any) -> str:
    if isinstance(content, (dict, list, tuple, bool, int, float)):
        try:
            return json.dumps(content, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            pass
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
    if _should_log_full_to_console():
        text = _compact_text(content)
        preview = " ".join(text.replace("\r", " ").replace("\n", " ").split())[:preview_limit]
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
    _enqueue_record("full", title, content, preview_limit)


def log_event(title: str, content: Any) -> None:
    if _should_log_full_to_console():
        text = _compact_text(content)
        print(f"INFO {title} | {text}", flush=True)
    if not _should_log_full_to_file():
        return
    _enqueue_record("event", title, content)


def _enqueue_record(kind: str, title: str, content: Any, preview_limit: int = 300) -> None:
    thread = current_thread()
    record = _LogRecord(
        kind=kind,
        title=title,
        content=content,
        preview_limit=preview_limit,
        enqueued_at=_timestamp(),
        enqueued_perf_counter=time.perf_counter(),
        caller_thread=thread.name,
        caller_thread_id=get_ident(),
    )
    _queue.put(record)
    _ensure_started()


@contextmanager
def timed_event(stage: str, content: Any | None = None) -> Iterator[None]:
    start = time.perf_counter()
    started_at = _timestamp()
    thread = current_thread()
    status = "ok"
    error = ""
    try:
        yield
    except Exception as exc:
        status = "error"
        error = repr(exc)
        raise
    finally:
        finished_at = _timestamp()
        payload = {
            "stage": stage,
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "thread_name": thread.name,
            "thread_id": get_ident(),
        }
        if content is not None:
            payload["context"] = content
        if error:
            payload["error"] = error
        log_event("timing", payload)


atexit.register(shutdown_agent_output_logger)
