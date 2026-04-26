"""Markdown-backed task and user memory storage."""
# -*- coding: utf-8 -*-
import hashlib
import json
import re
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def now_iso() -> str:
    """Return local ISO timestamp with seconds precision."""
    return datetime.now().replace(microsecond=0).isoformat()


class MemoryStore:
    """Persist task snapshots and user profile memory as structured markdown."""

    def __init__(self, runtime_root: Path | None = None):
        backend_root = Path(__file__).resolve().parents[2]
        self.runtime_root = runtime_root or backend_root / "runtime_data"
        self.tasks_dir = self.runtime_root / "tasks"
        self.users_dir = self.runtime_root / "users"
        self.ensure_runtime_dirs()

    def ensure_runtime_dirs(self) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.users_dir.mkdir(parents=True, exist_ok=True)

    def create_task_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        return f"task_{stamp}_{suffix}"

    def build_user_id(self, nickname: str) -> str:
        clean = re.sub(r"[^a-z0-9]+", "-", nickname.strip().lower()).strip("-")
        digest = hashlib.sha1(nickname.strip().encode("utf-8")).hexdigest()[:8]
        prefix = clean[:24] if clean else "user"
        return f"{prefix}_{digest}"

    def load_or_create_user_memory(self, user_id: str, nickname: str) -> Dict[str, Any]:
        path = self._user_path(user_id)
        if path.exists():
            return self.read_user_memory(user_id)

        timestamp = now_iso()
        record = {
            "user_id": user_id,
            "nickname": nickname,
            "created_at": timestamp,
            "updated_at": timestamp,
            "profile": {
                "preferences": [],
                "dislikes": [],
                "constraints": [],
                "budget_sensitivity": "",
                "notes": [],
                "last_task_id": "",
            },
            "update_history": [],
        }
        self.write_user_record(record)
        return record

    def read_user_memory(self, user_id: str) -> Dict[str, Any]:
        path = self._user_path(user_id)
        text = path.read_text(encoding="utf-8")
        front_matter = self._extract_front_matter(text)
        return {
            "user_id": front_matter.get("user_id", user_id),
            "nickname": front_matter.get("nickname", ""),
            "created_at": front_matter.get("created_at", ""),
            "updated_at": front_matter.get("updated_at", ""),
            "profile": self.extract_json_section(text, "Profile", default={}),
            "update_history": self.extract_json_section(text, "Update History", default=[]),
        }

    def write_user_memory(self, user_id: str, profile: Dict[str, Any], update_entry: Dict[str, Any]) -> None:
        current = self.read_user_memory(user_id)
        merged_profile = self._merge_profile(current.get("profile", {}), profile)
        history = current.get("update_history", [])
        entry = deepcopy(update_entry)
        entry.setdefault("timestamp", now_iso())
        history.append(entry)

        record = {
            **current,
            "updated_at": now_iso(),
            "profile": merged_profile,
            "update_history": history,
        }
        self.write_user_record(record)

    def write_user_record(self, record: Dict[str, Any]) -> None:
        path = self._user_path(record["user_id"])
        self._write_markdown_atomic(path, self.render_user_markdown(record))

    def read_task(self, task_id: str) -> Dict[str, Any]:
        path = self._task_path(task_id)
        text = path.read_text(encoding="utf-8")
        front_matter = self._extract_front_matter(text)
        return {
            "task_id": front_matter.get("task_id", task_id),
            "user_id": front_matter.get("user_id", ""),
            "nickname": front_matter.get("nickname", ""),
            "status": front_matter.get("status", "active"),
            "created_at": front_matter.get("created_at", ""),
            "updated_at": front_matter.get("updated_at", ""),
            "city": front_matter.get("city", ""),
            "date_range": front_matter.get("date_range", ""),
            "travel_days": int(front_matter.get("travel_days", 0) or 0),
            "update_mode": front_matter.get("update_mode", "initial"),
            "form_snapshot": self.extract_json_section(text, "Form Snapshot", default={}),
            "conversation_log": self.extract_json_section(text, "Conversation Log", default=[]),
            "conversation_context": self.extract_json_section(text, "Conversation Context", default={}),
            "current_plan": self.extract_json_section(text, "Current Plan", default={}),
            "budget_ledger": self.extract_json_section(text, "Budget Ledger", default={}),
            "reflection_log": self.extract_json_section(text, "Reflection Log", default=[]),
        }

    def write_task(self, task_record: Dict[str, Any]) -> None:
        path = self._task_path(task_record["task_id"])
        task_record["updated_at"] = now_iso()
        self._write_markdown_atomic(path, self.render_task_markdown(task_record))

    def append_task_conversation(self, task_id: str, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        task = self.read_task(task_id)
        task.setdefault("conversation_log", []).extend(entries)
        self.write_task(task)
        return task

    def extract_json_section(self, markdown: str, section_name: str, default: Any) -> Any:
        pattern = rf"## {re.escape(section_name)}\s*```json\s*(.*?)\s*```"
        match = re.search(pattern, markdown, re.DOTALL)
        if not match:
            return deepcopy(default)
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return deepcopy(default)

    def render_task_markdown(self, task_record: Dict[str, Any]) -> str:
        front = [
            "---",
            f"task_id: \"{task_record.get('task_id', '')}\"",
            f"user_id: \"{task_record.get('user_id', '')}\"",
            f"nickname: \"{task_record.get('nickname', '')}\"",
            f"status: \"{task_record.get('status', 'active')}\"",
            f"created_at: \"{task_record.get('created_at', '')}\"",
            f"updated_at: \"{task_record.get('updated_at', '')}\"",
            f"city: \"{task_record.get('city', '')}\"",
            f"date_range: \"{task_record.get('date_range', '')}\"",
            f"travel_days: {task_record.get('travel_days', 0)}",
            f"update_mode: \"{task_record.get('update_mode', 'initial')}\"",
            "---",
        ]
        sections = [
            ("Form Snapshot", task_record.get("form_snapshot", {})),
            ("Conversation Log", task_record.get("conversation_log", [])),
            ("Conversation Context", task_record.get("conversation_context", {})),
            ("Current Plan", task_record.get("current_plan", {})),
            ("Budget Ledger", task_record.get("budget_ledger", {})),
            ("Reflection Log", task_record.get("reflection_log", [])),
        ]
        return "\n".join(front) + "\n\n" + "\n\n".join(
            self._json_section(title, payload) for title, payload in sections
        ) + "\n"

    def render_user_markdown(self, record: Dict[str, Any]) -> str:
        front = [
            "---",
            f"user_id: \"{record.get('user_id', '')}\"",
            f"nickname: \"{record.get('nickname', '')}\"",
            f"created_at: \"{record.get('created_at', '')}\"",
            f"updated_at: \"{record.get('updated_at', '')}\"",
            "---",
        ]
        sections = [
            ("Profile", record.get("profile", {})),
            ("Update History", record.get("update_history", [])),
        ]
        return "\n".join(front) + "\n\n" + "\n\n".join(
            self._json_section(title, payload) for title, payload in sections
        ) + "\n"

    def _task_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.md"

    def _user_path(self, user_id: str) -> Path:
        return self.users_dir / f"{user_id}.md"

    def _json_section(self, title: str, payload: Any) -> str:
        return f"## {title}\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"

    def _extract_front_matter(self, markdown: str) -> Dict[str, str]:
        match = re.match(r"---\s*(.*?)\s*---", markdown, re.DOTALL)
        if not match:
            return {}
        result: Dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().strip('"')
        return result

    def _write_markdown_atomic(self, path: Path, content: str) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)

    def _merge_profile(self, current: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(current)
        for key in ("preferences", "dislikes", "constraints", "notes"):
            values = list(merged.get(key, []))
            for item in patch.get(key, []):
                if item and item not in values:
                    values.append(item)
            merged[key] = values

        if patch.get("budget_sensitivity"):
            merged["budget_sensitivity"] = patch["budget_sensitivity"]
        if patch.get("last_task_id"):
            merged["last_task_id"] = patch["last_task_id"]

        return merged


_memory_store = None


def get_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store
