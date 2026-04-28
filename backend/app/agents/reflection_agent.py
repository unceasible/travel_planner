# -*- coding: utf-8 -*-
"""Plan quality review agent backed by the main planning model."""

import json
import time
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..config import get_settings
from ..models.schemas import TripPlan
from ..services.agent_output_logger import log_event, log_full_output, timed_event
from ..services.llm_service import get_openai_client, get_openai_model
from ..services.memory_store import now_iso


class ReflectionReview(BaseModel):
    """Structured quality review for a generated trip plan."""

    score: int = Field(default=0, ge=0, le=10)
    status: Literal["pass", "needs_replan", "review_error"] = "pass"
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    improvement_instructions: str = ""
    summary: str = ""


class ReflectionAgent:
    """Review whether a TripPlan is good enough, without directly editing it."""

    def __init__(self, client: Optional[Any] = None, model: str = "") -> None:
        self.settings = get_settings()
        self.client = client or get_openai_client()
        self.model = model or get_openai_model()

    def review_plan(
        self,
        *,
        plan: TripPlan,
        form_snapshot: Dict[str, Any],
        user_profile_summary: str = "",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_message: str = "",
        update_mode: str = "initial",
    ) -> ReflectionReview:
        started_at = time.perf_counter()
        payload = {
            "form_snapshot": form_snapshot,
            "transport_preference_constraints": {
                "explicit_form_transport_wins_over_user_profile": True,
                "allow_self_drive_suggestions": (
                    str(form_snapshot.get("transportation") or "").strip() == "自驾"
                    or str(form_snapshot.get("intercity_transportation") or "").strip() == "自驾"
                ),
                "rule": "如果市内交通和大交通都未选择自驾，不要要求或建议自驾、开车、驾车或租车。",
            },
            "user_profile": user_profile_summary,
            "conversation_history": conversation_history or [],
            "latest_user_message": user_message,
            "update_mode": update_mode,
            "plan": plan.model_dump(),
            "review_dimensions": [
                "用户需求匹配",
                "用户偏好满足",
                "行程密度是否合理",
                "地理距离和路线合理性",
                "餐饮/酒店/景点是否混用",
                "天气影响是否被考虑",
                "预算是否合理",
                "修改请求是否被执行且未破坏无关部分",
            ],
        }
        try:
            raw = self._call_llm(payload)
            log_full_output("reflection_review_raw", raw)
            review = self._parse_review(raw)
            log_event(
                "reflection_review_summary",
                {
                    "status": review.status,
                    "score": review.score,
                    "issue_count": len(review.issues),
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                    "model": self.model,
                },
            )
            return review
        except Exception as exc:
            log_event(
                "reflection_review_error",
                {
                    "error": repr(exc),
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                    "model": self.model,
                },
            )
            return ReflectionReview(
                score=10,
                status="review_error",
                issues=[],
                improvement_instructions="",
                summary=f"ReflectionAgent review failed: {exc}",
            )

    def _call_llm(self, payload: Dict[str, Any]) -> str:
        system_prompt = (
            "你是旅行计划质量审阅智能体，只负责审阅 PlannerAgent 给出的计划，不直接重写计划。\n"
            "请从这些维度审阅：用户需求匹配、用户偏好满足、行程密度、地理距离/路线合理性、"
            "餐饮酒店景点是否混用、天气影响、预算合理性、修改请求是否被执行。\n"
            "表单中的本次交通选择优先级高于用户画像；如果市内交通和大交通都未选择自驾，不要因为用户画像或路线便利性要求补充自驾、开车、驾车或租车建议。\n"
            "给出 0-10 分。7 分以下表示计划质量不合格，需要 PlannerAgent 重新修订。\n"
            "只返回 JSON 对象，不要 Markdown。格式："
            "{"
            "\"score\": 0-10,"
            "\"status\": \"pass\" 或 \"needs_replan\","
            "\"issues\": [{\"type\": \"...\", \"severity\": \"low|medium|high|critical\", \"message\": \"...\"}],"
            "\"improvement_instructions\": \"给 PlannerAgent 的具体重修意见\","
            "\"summary\": \"一句话总结\""
            "}"
        )
        with timed_event("reflection.llm_request", {"model": self.model}):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0,
                max_tokens=self.settings.llm_max_output_tokens,
                response_format={"type": "json_object"},
            )
        return response.choices[0].message.content or ""

    def _parse_review(self, text: str) -> ReflectionReview:
        data = self._extract_json(text)
        score = self._clamp_score(data.get("score", 0))
        issues = data.get("issues") if isinstance(data.get("issues"), list) else []
        status = "needs_replan" if score < 7 else "pass"
        if data.get("status") == "review_error":
            status = "review_error"
        return ReflectionReview(
            score=score,
            status=status,
            issues=[item for item in issues if isinstance(item, dict)],
            improvement_instructions=str(data.get("improvement_instructions") or ""),
            summary=str(data.get("summary") or ""),
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        stripped = text.strip()
        if "```json" in stripped:
            start = stripped.find("```json") + 7
            end = stripped.find("```", start)
            stripped = stripped[start:end].strip()
        elif "```" in stripped:
            start = stripped.find("```") + 3
            end = stripped.find("```", start)
            stripped = stripped[start:end].strip()
        elif "{" in stripped and "}" in stripped:
            stripped = stripped[stripped.find("{"):stripped.rfind("}") + 1]
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("reflection review must be a JSON object")
        return parsed

    def _clamp_score(self, value: Any) -> int:
        try:
            score = int(round(float(value)))
        except Exception:
            score = 0
        return min(10, max(0, score))


def review_error_entry(error: Exception, retry_used: bool) -> Dict[str, Any]:
    """Build a persisted review entry when a caller-injected reviewer fails."""

    return {
        "phase": "quality_review",
        "timestamp": now_iso(),
        "score": None,
        "status": "review_error",
        "issues": [],
        "improvement_instructions": "",
        "summary": f"ReflectionAgent review failed: {error}",
        "retry_used": retry_used,
    }
