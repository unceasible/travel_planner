# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
import shutil
from concurrent.futures import Future
from pathlib import Path
from uuid import uuid4

from app.services.conversation_context import ConversationContextCompressor
from app.services.memory_store import MemoryStore, now_iso


def _task_record(task_id: str, plan_name: str, *, context=None, conversation=None):
    return {
        "task_id": task_id,
        "user_id": "user1",
        "nickname": "User",
        "status": "active",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "city": "北京",
        "date_range": "2026-05-01 to 2026-05-02",
        "travel_days": 2,
        "update_mode": "initial",
        "form_snapshot": {"city": "北京"},
        "conversation_log": list(conversation or []),
        "conversation_context": dict(context or {}),
        "current_plan": {"name": plan_name},
        "budget_ledger": {},
        "reflection_log": [],
    }


def _workspace_tmp_dir() -> Path:
    path = Path.cwd() / "runtime_data" / "test_task_memory_async" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_async_task_write_updates_memory_before_disk_write(monkeypatch):
    runtime_root = _workspace_tmp_dir()
    store = MemoryStore(runtime_root)
    task_id = "task_async_cache"
    store.write_task(_task_record(task_id, "old"))

    original_write = store._write_markdown_atomic
    write_started = threading.Event()
    allow_write = threading.Event()

    def delayed_write(path: Path, content: str) -> None:
        write_started.set()
        assert allow_write.wait(timeout=2)
        original_write(path, content)

    monkeypatch.setattr(store, "_write_markdown_atomic", delayed_write)

    try:
        future = store.write_task_async(_task_record(task_id, "new"))

        assert write_started.wait(timeout=1)
        assert store.read_task(task_id)["current_plan"]["name"] == "new"
        assert '"name": "old"' in store._task_path(task_id).read_text(encoding="utf-8")

        allow_write.set()
        future.result(timeout=2)
        assert '"name": "new"' in store._task_path(task_id).read_text(encoding="utf-8")
    finally:
        allow_write.set()
        if hasattr(store, "shutdown_async_writes"):
            store.shutdown_async_writes(wait=True)
        shutil.rmtree(runtime_root, ignore_errors=True)


def test_patch_task_fields_async_merges_into_latest_memory_without_overwriting_plan():
    runtime_root = _workspace_tmp_dir()
    store = MemoryStore(runtime_root)
    task_id = "task_patch_latest"
    store.write_task(_task_record(task_id, "old", conversation=[{"role": "assistant", "message": "old"}]))
    store.write_task_async(
        _task_record(task_id, "new", conversation=[{"role": "assistant", "message": "new"}])
    ).result(timeout=2)

    try:
        store.patch_task_fields_async(task_id, {"conversation_context": {"summary": {"latest": "context"}}}).result(
            timeout=2
        )

        latest = store.read_task(task_id)
        assert latest["current_plan"]["name"] == "new"
        assert latest["conversation_log"][0]["message"] == "new"
        assert latest["conversation_context"]["summary"]["latest"] == "context"
    finally:
        if hasattr(store, "shutdown_async_writes"):
            store.shutdown_async_writes(wait=True)
        shutil.rmtree(runtime_root, ignore_errors=True)


class _PatchOnlyStore:
    def __init__(self):
        self.patch_calls = []

    def patch_task_fields_async(self, task_id, fields):
        self.patch_calls.append((task_id, fields))
        future = Future()
        future.set_result(None)
        return future


def test_context_heavy_refresh_patches_context_instead_of_writing_full_task(monkeypatch):
    store = _PatchOnlyStore()
    compressor = ConversationContextCompressor(store)  # type: ignore[arg-type]
    monkeypatch.setattr(compressor, "_split_segments", lambda conversation, count: [conversation])
    monkeypatch.setattr(compressor, "_summarize_segment", lambda index, segment: {"items": ["summary"]})
    monkeypatch.setattr(compressor, "_merge_segment_summaries", lambda summaries: {"items": ["summary"]})

    compressor.refresh_heavy_summary("task_context_patch", [{"role": "user", "message": "请帮我改酒店"}])

    assert len(store.patch_calls) == 1
    task_id, fields = store.patch_calls[0]
    assert task_id == "task_context_patch"
    assert set(fields.keys()) == {"conversation_context"}
    assert fields["conversation_context"]["summary"] == {"items": ["summary"]}
