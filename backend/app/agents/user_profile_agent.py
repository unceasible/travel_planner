"""Background user profile agent backed by CHEAP_MODEL."""
# -*- coding: utf-8 -*-
import json
import re
from copy import deepcopy
from typing import Any, Dict

from ..services.agent_output_logger import log_full_output
from ..services.llm_service import get_cheap_llm
from ..services.memory_store import MemoryStore


USER_PROFILE_PROMPT = """你是用户画像更新智能体。你只从输入中提取长期稳定偏好,不要记录一次性行程安排。

请只返回JSON对象,格式如下:
{
  "preferences": [],
  "dislikes": [],
  "constraints": [],
  "budget_sensitivity": "",
  "notes": []
}

规则:
1. 喜欢、偏好、更喜欢、以后都希望 -> preferences
2. 不吃、讨厌、避免、过敏、不能 -> dislikes 或 constraints
3. 预算控制、便宜点、贵一点也可以 -> budget_sensitivity
4. 只记录可复用到未来任务的长期画像
5. 不要输出Markdown
"""

USER_PROFILE_FEW_SHOT = """
Few-shot example:
Input context:
{
  "current_memory": {
    "profile": {
      "preferences": ["美食"],
      "dislikes": [],
      "constraints": [],
      "budget_sensitivity": "",
      "notes": []
    }
  },
  "context": {
    "latest_user_message": "我不吃海鲜，以后也希望多安排博物馆",
    "form": {
      "preferences": ["历史文化"]
    }
  }
}

Output JSON:
{
  "preferences": ["美食", "历史文化", "博物馆"],
  "dislikes": [],
  "constraints": ["不吃海鲜"],
  "budget_sensitivity": "未明确表达",
  "notes": []
}
"""


class UserProfileAgent:
    """Asynchronously update user memory without blocking trip planning."""

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def update_profile(self, forked_context: Dict[str, Any]) -> None:
        try:
            user_id = forked_context["user_id"]
            nickname = forked_context.get("nickname", "")
            current_memory = self.memory_store.load_or_create_user_memory(user_id, nickname)

            profile_patch = self._extract_with_cheap_model(current_memory, forked_context)
            profile_patch = self._merge_rule_based_fallback(profile_patch, forked_context)
            profile_patch["last_task_id"] = forked_context.get("task_id", "")

            self.memory_store.write_user_memory(
                user_id=user_id,
                profile=profile_patch,
                update_entry={
                    "source_task_id": forked_context.get("task_id", ""),
                    "source_message": forked_context.get("latest_user_message", ""),
                    "applied_changes": deepcopy(profile_patch),
                },
            )
        except Exception as exc:
            print(f"⚠️  UserProfileAgent后台更新失败,不阻塞主流程: {exc}")

    def _extract_with_cheap_model(self, current_memory: Dict[str, Any], forked_context: Dict[str, Any]) -> Dict[str, Any]:
        llm = get_cheap_llm()
        payload = {
            "current_memory": current_memory.get("profile", {}),
            "context": forked_context,
        }
        response = llm.invoke([
            {"role": "system", "content": USER_PROFILE_PROMPT + "\n\n" + USER_PROFILE_FEW_SHOT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ])
        self._print_full_output_block("agent_result:user_profile", response)
        return self._parse_profile_patch(response)

    def _parse_profile_patch(self, text: str) -> Dict[str, Any]:
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
        except Exception:
            parsed = {}

        return {
            "preferences": self._as_list(parsed.get("preferences")),
            "dislikes": self._as_list(parsed.get("dislikes")),
            "constraints": self._as_list(parsed.get("constraints")),
            "budget_sensitivity": parsed.get("budget_sensitivity", "") if isinstance(parsed, dict) else "",
            "notes": self._as_list(parsed.get("notes")),
        }

    def _merge_rule_based_fallback(self, patch: Dict[str, Any], forked_context: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(patch)
        form = forked_context.get("form", {}) or {}
        message = forked_context.get("latest_user_message", "") or form.get("free_text_input", "")

        for preference in form.get("preferences", []) or []:
            self._add_unique(merged, "preferences", preference)

        if re.search(r"(喜欢|更喜欢|偏好)", message):
            self._add_unique(merged, "preferences", message.strip())

        if re.search(r"(不吃|讨厌|避免|过敏|不能)", message):
            self._add_unique(merged, "constraints", message.strip())

        if re.search(r"(预算控制|便宜|省钱|贵一点也可以|预算)", message):
            merged["budget_sensitivity"] = message.strip()

        return merged

    def _as_list(self, value: Any) -> list:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _add_unique(self, data: Dict[str, Any], key: str, value: str) -> None:
        value = value.strip()
        if not value:
            return
        data.setdefault(key, [])
        if value not in data[key]:
            data[key].append(value)

    def _print_full_output_block(self, title: str, content: Any) -> None:
        log_full_output(title, content)
