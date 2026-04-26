"""Conversation context compression helpers."""
# -*- coding: utf-8 -*-
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

from ..config import get_settings
from .agent_output_logger import log_event, timed_event
from .llm_service import get_cheap_openai_client, get_cheap_openai_model
from .memory_store import MemoryStore, now_iso


CONTEXT_CACHE_VERSION = 1


@dataclass
class PreparedConversationContext:
    history: List[Dict[str, Any]]
    raw_tokens: int
    prepared_tokens: int
    threshold_tokens: int
    mode: str
    should_run_heavy: bool


def estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    ascii_chars = 0
    non_ascii_chars = 0
    for char in text:
        if ord(char) < 128:
            ascii_chars += 1
        else:
            non_ascii_chars += 1
    return max(1, math.ceil(ascii_chars / 4) + non_ascii_chars)


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ConversationContextCompressor:
    """Build bounded LLM conversation context while preserving raw task history."""

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self.settings = get_settings()

    @property
    def threshold_tokens(self) -> int:
        threshold = self.settings.llm_context_window_tokens - 10 * self.settings.llm_max_output_tokens
        return max(4096, threshold)

    def prepare_context(self, task_record: Dict[str, Any]) -> PreparedConversationContext:
        conversation_log = list(task_record.get("conversation_log", []) or [])
        raw_tokens = estimate_tokens(conversation_log)
        cached_summary = self._valid_cached_summary(task_record.get("conversation_context", {}), conversation_log)
        threshold = self.threshold_tokens
        mode = "raw"

        if raw_tokens < threshold:
            history = conversation_log
            should_run_heavy = False
        else:
            mode = "light"
            history = self._build_light_context(task_record, conversation_log, cached_summary, threshold)
            should_run_heavy = True

        prepared_tokens = estimate_tokens(history)
        if prepared_tokens >= threshold:
            mode = "heavy_pending"
            history = self._build_light_context(task_record, conversation_log, cached_summary, int(threshold * 0.75))
            prepared_tokens = estimate_tokens(history)
            should_run_heavy = True

        log_event(
            "context_compression_prepare",
            {
                "task_id": task_record.get("task_id", ""),
                "mode": mode,
                "conversation_entries": len(conversation_log),
                "raw_tokens": raw_tokens,
                "prepared_tokens": prepared_tokens,
                "threshold_tokens": threshold,
                "has_cached_summary": bool(cached_summary),
                "should_run_heavy": should_run_heavy,
            },
        )
        return PreparedConversationContext(
            history=history,
            raw_tokens=raw_tokens,
            prepared_tokens=prepared_tokens,
            threshold_tokens=threshold,
            mode=mode,
            should_run_heavy=should_run_heavy,
        )

    def needs_heavy_refresh(self, task_record: Dict[str, Any]) -> bool:
        conversation_log = list(task_record.get("conversation_log", []) or [])
        if estimate_tokens(conversation_log) < self.threshold_tokens:
            return False
        cache = task_record.get("conversation_context", {})
        if not isinstance(cache, dict):
            return True
        if cache.get("cache_version") != CONTEXT_CACHE_VERSION:
            return True
        if int(cache.get("covered_count", 0) or 0) != len(conversation_log):
            return True
        return cache.get("covered_hash") != _hash_json(conversation_log)

    def refresh_heavy_summary(self, task_id: str, conversation_log: List[Dict[str, Any]]) -> None:
        if not conversation_log:
            return
        started_at = time.perf_counter()
        source_count = len(conversation_log)
        source_hash = _hash_json(conversation_log)
        raw_tokens = estimate_tokens(conversation_log)
        segment_count = 20 if raw_tokens > self.threshold_tokens * 2 else 10
        segments = self._split_segments(conversation_log, segment_count)
        if not segments:
            return

        try:
            log_event(
                "context_compression_heavy_start",
                {
                    "task_id": task_id,
                    "entries": source_count,
                    "raw_tokens": raw_tokens,
                    "segments": len(segments),
                },
            )
            workers = max(1, min(self.settings.context_heavy_summary_workers, len(segments)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._summarize_segment, index, segment): index
                    for index, segment in enumerate(segments)
                }
                segment_summaries: List[Dict[str, Any]] = [{} for _ in segments]
                for future in as_completed(futures):
                    index = futures[future]
                    segment_summaries[index] = future.result()

            merged_summary = self._merge_segment_summaries(segment_summaries)
            current = self.memory_store.read_task(task_id)
            current["conversation_context"] = {
                "cache_version": CONTEXT_CACHE_VERSION,
                "mode": "heavy",
                "created_at": now_iso(),
                "covered_count": source_count,
                "covered_hash": source_hash,
                "raw_tokens": raw_tokens,
                "segment_count": len(segments),
                "summary": merged_summary,
            }
            self.memory_store.write_task(current)
            log_event(
                "context_compression_heavy_complete",
                {
                    "task_id": task_id,
                    "entries": source_count,
                    "raw_tokens": raw_tokens,
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )
        except Exception as exc:
            log_event(
                "context_compression_heavy_error",
                {
                    "task_id": task_id,
                    "error": repr(exc),
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )

    def _build_light_context(
        self,
        task_record: Dict[str, Any],
        conversation_log: List[Dict[str, Any]],
        cached_summary: Dict[str, Any] | None,
        threshold: int,
    ) -> List[Dict[str, Any]]:
        summary_item = {
            "role": "system",
            "kind": "conversation_context",
            "compression": "light",
            "task": {
                "task_id": task_record.get("task_id", ""),
                "user_id": task_record.get("user_id", ""),
                "nickname": task_record.get("nickname", ""),
                "city": task_record.get("city", ""),
                "date_range": task_record.get("date_range", ""),
                "travel_days": task_record.get("travel_days", 0),
                "update_mode": task_record.get("update_mode", ""),
            },
            "summary": cached_summary,
            "raw_entries_total": len(conversation_log),
        }
        summary_tokens = estimate_tokens(summary_item)
        max_recent = min(self.settings.context_recent_raw_max_tokens, max(1000, threshold - summary_tokens - 512))
        recent_budget = max(
            min(self.settings.context_recent_raw_min_tokens, max_recent),
            min(max_recent, threshold // 2),
        )
        recent_entries = self._tail_by_token_budget(conversation_log, recent_budget)
        return [summary_item, *recent_entries]

    def _tail_by_token_budget(self, entries: List[Dict[str, Any]], token_budget: int) -> List[Dict[str, Any]]:
        if token_budget <= 0:
            return []
        selected: List[Dict[str, Any]] = []
        used = 0
        for entry in reversed(entries):
            entry_tokens = estimate_tokens(entry)
            if used + entry_tokens <= token_budget:
                selected.append(deepcopy(entry))
                used += entry_tokens
                continue
            remaining = token_budget - used
            if remaining > 200:
                selected.append(self._truncate_entry(entry, remaining))
            break
        selected.reverse()
        return selected

    def _truncate_entry(self, entry: Dict[str, Any], token_budget: int) -> Dict[str, Any]:
        item = deepcopy(entry)
        message = str(item.get("message") or item.get("content") or "")
        if message:
            keep_chars = max(200, token_budget * 2)
            suffix = message[-keep_chars:]
            if "message" in item:
                item["message"] = f"...{suffix}"
            else:
                item["content"] = f"...{suffix}"
            item["truncated_for_context"] = True
        return item

    def _valid_cached_summary(self, cache: Any, conversation_log: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        if not isinstance(cache, dict):
            return None
        if cache.get("cache_version") != CONTEXT_CACHE_VERSION:
            return None
        covered_count = int(cache.get("covered_count", 0) or 0)
        if covered_count <= 0 or covered_count > len(conversation_log):
            return None
        if cache.get("covered_hash") != _hash_json(conversation_log[:covered_count]):
            return None
        summary = cache.get("summary")
        return summary if isinstance(summary, dict) else None

    def _split_segments(self, entries: List[Dict[str, Any]], desired_segments: int) -> List[List[Dict[str, Any]]]:
        if not entries:
            return []
        desired_segments = max(1, min(desired_segments, len(entries)))
        total_tokens = estimate_tokens(entries)
        target_tokens = max(1, math.ceil(total_tokens / desired_segments))
        segments: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_tokens = 0
        for entry in entries:
            entry_tokens = estimate_tokens(entry)
            if current and current_tokens + entry_tokens > target_tokens and len(segments) < desired_segments - 1:
                segments.append(current)
                current = []
                current_tokens = 0
            current.append(entry)
            current_tokens += entry_tokens
        if current:
            segments.append(current)
        return segments

    def _summarize_segment(self, index: int, segment: List[Dict[str, Any]]) -> Dict[str, Any]:
        client = get_cheap_openai_client()
        model = get_cheap_openai_model()
        prompt = (
            "你是旅行助手的上下文压缩器。只总结对后续修改旅行计划有用的信息。"
            "保留用户明确要求、已确认决定、未解决问题、偏好和约束。"
            "不要编造。只输出JSON对象，字段为 summary, user_requirements, decisions, open_questions, preferences。"
        )
        payload = {"segment_index": index, "conversation": segment}
        with timed_event("context_compression.segment_llm", {"segment_index": index, "entries": len(segment)}):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0,
                max_tokens=self.settings.cheap_model_max_output_tokens,
            )
        text = response.choices[0].message.content or ""
        parsed = self._extract_json_object(text)
        parsed["segment_index"] = index
        parsed["entry_count"] = len(segment)
        return parsed

    def _merge_segment_summaries(self, segment_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(segment_summaries) == 1:
            return segment_summaries[0]
        client = get_cheap_openai_client()
        model = get_cheap_openai_model()
        prompt = (
            "你是旅行助手的全局上下文压缩器。合并多个分段摘要，去重并保留最新状态。"
            "只输出JSON对象，字段为 summary, user_requirements, decisions, open_questions, preferences。"
        )
        with timed_event("context_compression.merge_llm", {"segments": len(segment_summaries)}):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps({"segments": segment_summaries}, ensure_ascii=False)},
                ],
                temperature=0,
                max_tokens=self.settings.cheap_model_max_output_tokens,
            )
        return self._extract_json_object(response.choices[0].message.content or "")

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        try:
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "{" in text and "}" in text:
                text = text[text.find("{"):text.rfind("}") + 1]
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {
            "summary": text.strip()[:4000],
            "user_requirements": [],
            "decisions": [],
            "open_questions": [],
            "preferences": [],
        }
