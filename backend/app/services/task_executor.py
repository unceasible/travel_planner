"""Trip task orchestration service."""
# -*- coding: utf-8 -*-
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from copy import deepcopy
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Callable, Dict, List, Tuple

from ..agents.trip_planner_agent import get_trip_planner_agent
from ..agents.user_profile_agent import UserProfileAgent
from ..models.schemas import (
    Budget,
    DayBudget,
    Meal,
    TransportSegment,
    TripChatRequest,
    TripPlan,
    TripPlanResponse,
    TripRequest,
    WeatherInfo,
)
from .agent_output_logger import log_event, timed_event
from .amap_service import get_amap_service
from .conversation_context import ConversationContextCompressor
from .intent_classifier import NON_PERSIST_INTENTS, get_intent_classifier
from .llm_service import get_cheap_openai_client, get_cheap_openai_model
from .memory_store import get_memory_store, now_iso


GLOBAL_REPLAN_PATTERNS = [
    r"重做",
    r"重新规划",
    r"全部重排",
    r"整体改",
    r"改城市",
    r"换城市",
    r"改日期",
    r"改天数",
    r"改成自驾",
    r"改成公共交通",
    r"改成步行",
    r"改住宿",
]

PATCH_DOMAIN_KEYWORDS = {
    "restaurants": [r"吃", r"餐", r"晚饭", r"午饭", r"早餐", r"美食"],
    "hotels": [r"酒店", r"住宿", r"民宿"],
    "weather": [r"下雨", r"天气", r"温度", r"风"],
    "attractions": [r"景点", r"博物馆", r"公园", r"展览", r"路线"],
}

MEAL_DEFAULTS = {"breakfast": 25, "lunch": 60, "dinner": 90, "snack": 30}
HOTEL_DEFAULTS = {"经济型酒店": 300, "舒适型酒店": 500, "豪华酒店": 900, "民宿": 450}


ProgressCallback = Callable[[str, Dict[str, Any]], None]


class TripTaskExecutor:
    def __init__(self) -> None:
        self.memory_store = get_memory_store()
        self.planner = get_trip_planner_agent()
        self.user_profile_agent = UserProfileAgent(self.memory_store)
        self.retrieval_executor = ThreadPoolExecutor(max_workers=4)
        self.transport_executor = ThreadPoolExecutor(max_workers=8)
        self.route_executor = ThreadPoolExecutor(max_workers=4)
        self.profile_executor = ThreadPoolExecutor(max_workers=1)
        self.context_executor = ThreadPoolExecutor(max_workers=1)
        self.intent_classifier = get_intent_classifier()
        self.context_compressor = ConversationContextCompressor(self.memory_store)

    def _emit_progress(
        self,
        progress: ProgressCallback | None,
        stage: str,
        message: str,
        percent: int,
        **extra: Any,
    ) -> None:
        if progress is None:
            return
        payload = {"stage": stage, "message": message, "percent": percent, **extra}
        progress(stage, payload)

    def plan_initial(self, request: TripRequest, progress: ProgressCallback | None = None) -> TripPlanResponse:
        print(
            f"INFO plan_initial start | city={request.city} travel_days={request.travel_days} accommodation={request.accommodation}",
            flush=True,
        )
        task_record, user_memory = self._bootstrap_task(request)
        self._emit_progress(
            progress,
            "task_created",
            "任务已创建，正在准备旅行规划",
            5,
            task_id=task_record["task_id"],
            user_id=task_record["user_id"],
        )
        self._emit_progress(progress, "profile_update_started", "正在并行更新用户画像", 8, task_id=task_record["task_id"])
        self._start_user_profile_update(
            self._fork_context(
                {
                    "kind": "initial",
                    "task_id": task_record["task_id"],
                    "user_id": task_record["user_id"],
                    "nickname": request.nickname,
                    "form": request.model_dump(),
                    "user_memory": user_memory.get("profile", {}),
                    "conversation": [],
                }
            )
        )

        self._emit_progress(progress, "retrieval_started", "正在检索景点、酒店、餐饮和天气", 15, task_id=task_record["task_id"])
        retrieval_results = self._fetch_initial_context_parallel(request)
        self._emit_progress(progress, "retrieval_completed", "检索完成，正在整理候选信息", 35, task_id=task_record["task_id"])
        print("=" * 80, flush=True)
        log_event("retrieval_completed", f"keys={list(retrieval_results.keys())}")
        print("=" * 80, flush=True)
        user_profile_summary = json.dumps(user_memory.get("profile", {}), ensure_ascii=False)
        print("INFO plan_initial build_plan_from_context start", flush=True)
        self._emit_progress(progress, "llm_started", "正在生成结构化旅行计划", 45, task_id=task_record["task_id"])
        with timed_event("plan_initial.build_plan_from_context", {"city": request.city, "travel_days": request.travel_days}):
            plan = self.planner.build_plan_from_context(
                request=request,
                attractions=retrieval_results["attractions"],
                weather=retrieval_results["weather"],
                hotels=retrieval_results["hotels"],
                restaurants=retrieval_results["restaurants"],
                user_profile_summary=user_profile_summary,
                extra_requirements=request.free_text_input or "",
            )
        self._emit_progress(progress, "llm_completed", "初步计划已生成，正在补充交通和预算", 75, task_id=task_record["task_id"])
        print("INFO plan_initial build_plan_from_context end", flush=True)
        print("INFO plan_initial add_transport start", flush=True)
        self._emit_progress(progress, "transport_started", "正在计算每日交通段", 80, task_id=task_record["task_id"])
        plan = self._add_transport_segments_parallel(plan, request.transportation)
        print("INFO plan_initial add_transport end", flush=True)
        print("INFO plan_initial aggregate_budget start", flush=True)
        self._emit_progress(progress, "budget_started", "正在汇总预算", 86, task_id=task_record["task_id"])
        with timed_event("plan_initial.aggregate_budget", {"days": len(plan.days)}):
            plan = self._aggregate_budget(plan, request.accommodation)
        print("INFO plan_initial aggregate_budget end", flush=True)
        print("INFO plan_initial reflect_and_fix start", flush=True)
        self._emit_progress(progress, "reflection_started", "正在做最终检查", 92, task_id=task_record["task_id"])
        log_event("plan_before_reflect_and_fix", plan.model_dump_json(indent=2))
        with timed_event("plan_initial.reflect_and_fix", {"days": len(plan.days)}):
            plan, reflection = self._reflect_and_fix(plan, request.model_dump())
        log_event("plan_after_reflect_and_fix", plan.model_dump_json(indent=2))
        print("INFO plan_initial reflect_and_fix end", flush=True)

        assistant_message = "已生成初步计划，你可以继续直接提意见让我修改。"
        task_record["current_plan"] = plan.model_dump()
        task_record["budget_ledger"] = self._build_budget_ledger(plan)
        task_record["reflection_log"] = [reflection]
        task_record["conversation_log"] = [
            {
                "role": "assistant",
                "message": assistant_message,
                "timestamp": now_iso(),
                "update_mode": "initial",
            }
        ]
        task_record["update_mode"] = "initial"
        with timed_event("plan_initial.write_task", {"task_id": task_record["task_id"]}):
            self.memory_store.write_task(task_record)
        self._emit_progress(progress, "persisted", "计划已保存", 98, task_id=task_record["task_id"])

        return TripPlanResponse(
            success=True,
            message="旅行计划生成成功",
            task_id=task_record["task_id"],
            user_id=task_record["user_id"],
            update_mode="initial",
            assistant_message=assistant_message,
            data=plan,
        )

    def chat(self, request: TripChatRequest, progress: ProgressCallback | None = None) -> TripPlanResponse:
        task_record = self.memory_store.read_task(request.task_id)
        form_snapshot = task_record.get("form_snapshot", {}) or {}
        current_plan = self._dict_to_plan(task_record.get("current_plan", {}), form_snapshot)
        prepared_context = self.context_compressor.prepare_context(task_record)
        conversation_history = prepared_context.history
        trip_context = self._build_trip_context(form_snapshot)
        self._emit_progress(progress, "intent_started", "正在识别你的修改意图", 8, task_id=request.task_id)
        with timed_event("chat.intent_classify", {"task_id": request.task_id}):
            intent_result = self.intent_classifier.classify(request.user_message, trip_context)
        self._emit_progress(
            progress,
            "intent_completed",
            "已识别修改意图",
            18,
            task_id=request.task_id,
            primary_intent=intent_result.primary_intent,
            domains=intent_result.domains,
            action=intent_result.action,
        )
        log_event(
            "decision_route",
            {
                "task_id": request.task_id,
                "primary_intent": intent_result.primary_intent,
                "intents": intent_result.intents,
                "domains": intent_result.domains,
                "action": intent_result.action,
                "source": intent_result.source,
                "matched_rule": intent_result.matched_rule,
            },
        )
        self._emit_progress(
            progress,
            "route_decided",
            "已选择处理分支",
            24,
            task_id=request.task_id,
            route=intent_result.primary_intent,
            context_mode=prepared_context.mode,
        )

        if intent_result.primary_intent in NON_PERSIST_INTENTS:
            assistant_message = self._build_non_persist_message(intent_result.primary_intent)
            return TripPlanResponse(
                success=True,
                message="聊天已处理",
                task_id=task_record["task_id"],
                user_id=task_record["user_id"],
                update_mode=intent_result.primary_intent,
                assistant_message=assistant_message,
                data=current_plan,
            )

        user_memory = self.memory_store.load_or_create_user_memory(
            task_record["user_id"],
            task_record.get("nickname", form_snapshot.get("nickname", "")),
        )
        self._emit_progress(progress, "profile_update_started", "正在并行更新用户画像", 28, task_id=request.task_id)
        self._start_user_profile_update(
            self._fork_context(
                {
                    "kind": "chat",
                    "task_id": task_record["task_id"],
                    "user_id": task_record["user_id"],
                    "nickname": task_record.get("nickname", ""),
                    "task_summary": self._summarize_task(task_record),
                    "latest_user_message": request.user_message,
                    "intent": intent_result.model_dump(),
                    "user_memory": user_memory.get("profile", {}),
                    "form": form_snapshot,
                }
            )
        )
        user_profile_summary = json.dumps(user_memory.get("profile", {}), ensure_ascii=False)

        if intent_result.primary_intent == "question":
            self._emit_progress(progress, "patch_started", "正在基于当前计划回答问题", 55, task_id=request.task_id)
            assistant_message = self._answer_question(current_plan, request.user_message)
            self._persist_chat_turn(
                task_record,
                current_plan,
                form_snapshot,
                request.user_message,
                assistant_message,
                "question",
            )
            self._start_context_compression_if_needed(task_record)
            self._emit_progress(progress, "persisted", "对话已保存", 96, task_id=request.task_id)
            return TripPlanResponse(
                success=True,
                message="已基于当前计划回答",
                task_id=task_record["task_id"],
                user_id=task_record["user_id"],
                update_mode="question",
                assistant_message=assistant_message,
                data=current_plan,
            )

        if intent_result.primary_intent == "satisfied":
            assistant_message = "好的，已保留当前计划。你之后还可以继续让我调整。"
            self._persist_chat_turn(
                task_record,
                current_plan,
                form_snapshot,
                request.user_message,
                assistant_message,
                "satisfied",
            )
            self._start_context_compression_if_needed(task_record)
            self._emit_progress(progress, "persisted", "对话已保存", 96, task_id=request.task_id)
            return TripPlanResponse(
                success=True,
                message="已保留当前计划",
                task_id=task_record["task_id"],
                user_id=task_record["user_id"],
                update_mode="satisfied",
                assistant_message=assistant_message,
                data=current_plan,
            )

        if intent_result.primary_intent == "unclear":
            assistant_message = "我还不太确定你想调整哪一部分。你可以告诉我是想改景点、酒店、餐饮、交通、日期，还是整体重排吗？"
            self._persist_chat_turn(
                task_record,
                current_plan,
                form_snapshot,
                request.user_message,
                assistant_message,
                "unclear",
            )
            self._start_context_compression_if_needed(task_record)
            self._emit_progress(progress, "persisted", "对话已保存", 96, task_id=request.task_id)
            return TripPlanResponse(
                success=True,
                message="需要进一步澄清",
                task_id=task_record["task_id"],
                user_id=task_record["user_id"],
                update_mode="unclear",
                assistant_message=assistant_message,
                data=current_plan,
            )

        update_mode = "replan" if intent_result.primary_intent == "replan" else "patch"
        if update_mode == "patch":
            intent_domains = {domain for domain in intent_result.domains if domain != "none"}
            selected_domains = self._domains_to_retrieval_domains(intent_result.domains)
            self._emit_progress(progress, "retrieval_started", "正在检索本次修改需要的候选信息", 35, task_id=request.task_id)
            patch_context = self._fetch_patch_context_parallel(request.user_message, form_snapshot, selected_domains)
            self._emit_progress(progress, "retrieval_completed", "候选信息检索完成", 48, task_id=request.task_id)
            if intent_domains == {"attractions"}:
                log_event("chat_patch_route", "mode=attractions_only")
                self._emit_progress(progress, "patch_started", "正在修改景点安排", 58, task_id=request.task_id)
                with timed_event("chat.revise_attractions_only", {"task_id": request.task_id}):
                    revised = self.planner.revise_attractions_only(
                        current_plan=current_plan,
                        form_snapshot=form_snapshot,
                        user_message=request.user_message,
                        patch_context=patch_context,
                        user_profile_summary=user_profile_summary,
                        conversation_history=conversation_history,
                    )
                self._emit_progress(progress, "patch_completed", "景点安排已修改", 75, task_id=request.task_id)
            else:
                log_event(
                    "chat_patch_route",
                    {
                        "mode": "general",
                        "intent_domains": sorted(intent_domains),
                        "retrieval_domains": sorted(selected_domains),
                    },
                )
                self._emit_progress(progress, "patch_started", "正在修改旅行计划", 58, task_id=request.task_id)
                with timed_event("chat.revise_plan", {"task_id": request.task_id, "domains": sorted(intent_domains)}):
                    revised = self.planner.revise_plan(
                        current_plan=current_plan,
                        form_snapshot=form_snapshot,
                        user_message=request.user_message,
                        patch_context=patch_context,
                        user_profile_summary=user_profile_summary,
                        conversation_history=conversation_history,
                    )
                self._emit_progress(progress, "patch_completed", "旅行计划已修改", 75, task_id=request.task_id)
            if self._validate_patch_result(revised, form_snapshot):
                plan = revised
            else:
                log_event("patch_validation_failed", "fallback_to_replan")
                update_mode = "replan"
                self._emit_progress(progress, "replan_started", "局部修改校验未通过，正在整体重规划", 58, task_id=request.task_id)
                plan, form_snapshot = self._replan_from_message(
                    form_snapshot,
                    request.user_message,
                    user_profile_summary,
                    conversation_history,
                )
                self._emit_progress(progress, "replan_completed", "整体重规划完成", 75, task_id=request.task_id)
        else:
            self._emit_progress(progress, "replan_started", "正在整体重规划", 35, task_id=request.task_id)
            plan, form_snapshot = self._replan_from_message(
                form_snapshot,
                request.user_message,
                user_profile_summary,
                conversation_history,
            )
            self._emit_progress(progress, "replan_completed", "整体重规划完成", 75, task_id=request.task_id)

        self._emit_progress(progress, "transport_started", "正在重新计算交通段", 82, task_id=request.task_id)
        plan = self._add_transport_segments_parallel(plan, form_snapshot.get("transportation", "公共交通"))
        self._emit_progress(progress, "budget_started", "正在重新汇总预算", 88, task_id=request.task_id)
        plan = self._aggregate_budget(plan, form_snapshot.get("accommodation", "舒适型酒店"))
        log_event("plan_before_reflect_and_fix", plan.model_dump_json(indent=2))
        self._emit_progress(progress, "reflection_started", "正在做最终检查", 92, task_id=request.task_id)
        plan, reflection = self._reflect_and_fix(plan, form_snapshot)
        log_event("plan_after_reflect_and_fix", plan.model_dump_json(indent=2))

        assistant_message = self._build_revision_message(update_mode)
        if "refuse" in intent_result.intents and intent_result.primary_intent != "refuse":
            assistant_message += " 另外，非旅游相关的问题我不能展开回答。"
        self._persist_chat_turn(
            task_record,
            plan,
            form_snapshot,
            request.user_message,
            assistant_message,
            update_mode,
            reflection,
        )
        self._start_context_compression_if_needed(task_record)
        self._emit_progress(progress, "persisted", "修改已保存", 96, task_id=request.task_id)

        return TripPlanResponse(
            success=True,
            message="旅行计划更新成功",
            task_id=task_record["task_id"],
            user_id=task_record["user_id"],
            update_mode=update_mode,
            assistant_message=assistant_message,
            data=plan,
        )

    def restore(self, task_id: str) -> TripPlanResponse:
        task_record = self.memory_store.read_task(task_id)
        plan = self._dict_to_plan(task_record.get("current_plan", {}), task_record.get("form_snapshot", {}))
        return TripPlanResponse(
            success=True,
            message="任务加载成功",
            task_id=task_record["task_id"],
            user_id=task_record["user_id"],
            update_mode="restore",
            assistant_message="",
            data=plan,
        )

    def _bootstrap_task(self, request: TripRequest) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        task_id = self.memory_store.create_task_id()
        user_id = self.memory_store.build_user_id(request.nickname)
        user_memory = self.memory_store.load_or_create_user_memory(user_id, request.nickname)
        record = {
            "task_id": task_id,
            "user_id": user_id,
            "nickname": request.nickname,
            "status": "active",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "city": request.city,
            "date_range": f"{request.start_date} to {request.end_date}",
            "travel_days": request.travel_days,
            "update_mode": "initial",
            "form_snapshot": request.model_dump(),
            "conversation_log": [],
            "current_plan": {},
            "budget_ledger": {},
            "reflection_log": [],
        }
        self.memory_store.write_task(record)
        return record, user_memory

    def _fetch_initial_context_parallel(self, request: TripRequest) -> Dict[str, str]:
        with timed_event("retrieval.initial.total", {"city": request.city, "travel_days": request.travel_days}):
            futures = {
                "attractions": self.retrieval_executor.submit(self.planner.search_initial_attractions, request),
                "hotels": self.retrieval_executor.submit(
                    self.planner.search_hotels,
                    request.city,
                    request.accommodation,
                    request.start_date,
                    request.end_date,
                ),
                "restaurants": self.retrieval_executor.submit(
                    self.planner.search_initial_restaurants,
                    request,
                ),
                "weather": self.retrieval_executor.submit(
                    self.planner.search_weather,
                    request.city,
                    request.start_date,
                    request.end_date,
                ),
            }
            return self._wait_future_map(futures)

    def _fetch_patch_context_parallel(
        self,
        message: str,
        form_snapshot: Dict[str, Any],
        selected_domains: set[str] | None = None,
    ) -> Dict[str, str]:
        with timed_event("retrieval.patch.total", {"domains": sorted(selected_domains or [])}):
            request = self._dict_to_request(form_snapshot)
            if selected_domains is None:
                selected_domains = self._select_patch_domains(message)
                if not selected_domains:
                    selected_domains.add("attractions")
            else:
                selected_domains = set(selected_domains)
            log_event("patch_retrieval_domains", sorted(selected_domains))

            futures: Dict[str, Any] = {}
            if "attractions" in selected_domains:
                futures["attractions"] = self.retrieval_executor.submit(self.planner.search_attractions, request, "")
            if "hotels" in selected_domains:
                futures["hotels"] = self.retrieval_executor.submit(
                    self.planner.search_hotels,
                    request.city,
                    request.accommodation,
                    request.start_date,
                    request.end_date,
                )
            if "restaurants" in selected_domains:
                futures["restaurants"] = self.retrieval_executor.submit(
                    self.planner.search_restaurants,
                    request.city,
                    request.preferences[0] if request.preferences else "",
                )
            if "weather" in selected_domains:
                futures["weather"] = self.retrieval_executor.submit(
                    self.planner.search_weather,
                    request.city,
                    request.start_date,
                    request.end_date,
                )
            return self._wait_future_map(futures)

    def _select_patch_domains(self, message: str) -> set[str]:
        return {domain for domain, patterns in PATCH_DOMAIN_KEYWORDS.items() if any(re.search(p, message) for p in patterns)}

    def _domains_to_retrieval_domains(self, domains: List[str]) -> set[str]:
        retrievable = {"attractions", "hotels", "restaurants", "weather"}
        return {domain for domain in domains if domain in retrievable}

    def _build_trip_context(self, form_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "city": form_snapshot.get("city", ""),
            "start_date": form_snapshot.get("start_date", ""),
            "end_date": form_snapshot.get("end_date", ""),
            "travel_days": form_snapshot.get("travel_days", 0),
            "transportation": form_snapshot.get("transportation", ""),
            "accommodation": form_snapshot.get("accommodation", ""),
            "preferences": form_snapshot.get("preferences", []) or [],
        }

    def _wait_future_map(self, futures: Dict[str, Any]) -> Dict[str, str]:
        results: Dict[str, str] = {}
        for key, future in futures.items():
            try:
                with timed_event("future.wait", {"key": key}):
                    results[key] = future.result(timeout=90)
            except Exception as exc:
                print(f"⚠️  并行任务{key}失败,使用空结果兜底: {exc}")
                results[key] = ""
        return results

    def _route_revision_intent(self, user_message: str) -> str:
        for pattern in GLOBAL_REPLAN_PATTERNS:
            if re.search(pattern, user_message):
                return "replan"
        if "城市" in user_message or "日期" in user_message or "天数" in user_message:
            return "replan"
        return "patch"

    def _replan_from_message(
        self,
        form_snapshot: Dict[str, Any],
        user_message: str,
        user_profile_summary: str,
        conversation_history: List[Dict[str, Any]] | None = None,
    ) -> Tuple[TripPlan, Dict[str, Any]]:
        merged_form = self._merge_message_into_form(form_snapshot, user_message)
        request = self._dict_to_request(merged_form)
        retrieval_results = self._fetch_initial_context_parallel(request)
        with timed_event("chat.replan.build_plan_from_context", {"city": request.city, "travel_days": request.travel_days}):
            plan = self.planner.build_plan_from_context(
                request=request,
                attractions=retrieval_results.get("attractions", ""),
                weather=retrieval_results.get("weather", ""),
                hotels=retrieval_results.get("hotels", ""),
                restaurants=retrieval_results.get("restaurants", ""),
                user_profile_summary=user_profile_summary,
                extra_requirements=request.free_text_input or "",
                conversation_history=conversation_history or [],
            )
        return plan, merged_form

    def _merge_message_into_form(self, form_snapshot: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        merged = deepcopy(form_snapshot)
        prev = str(merged.get("free_text_input", "")).strip()
        merged["free_text_input"] = f"{prev}\n{user_message}".strip()
        if re.search(r"改成自驾", user_message):
            merged["transportation"] = "自驾"
        elif re.search(r"改成公共交通", user_message):
            merged["transportation"] = "公共交通"
        elif re.search(r"改成步行", user_message):
            merged["transportation"] = "步行"
        elif re.search(r"改成混合", user_message):
            merged["transportation"] = "混合"

        if re.search(r"住民宿", user_message):
            merged["accommodation"] = "民宿"
        elif re.search(r"豪华酒店", user_message):
            merged["accommodation"] = "豪华酒店"
        return merged

    def _validate_patch_result(self, plan: TripPlan, form_snapshot: Dict[str, Any]) -> bool:
        expected_days = int(form_snapshot.get("travel_days", 1) or 1)
        return bool(plan.days) and len(plan.days) == expected_days and all(day.attractions for day in plan.days)

    def _add_transport_segments_parallel(self, plan: TripPlan, transportation: str) -> TripPlan:
        jobs = []
        route_cache: Dict[str, Dict[str, Any]] = {}
        route_cache_lock = Lock()
        for day in plan.days:
            for from_point, to_point in self._build_day_pairs(day):
                jobs.append((day.day_index, from_point, to_point, transportation, plan.city, route_cache, route_cache_lock))

        with timed_event("transport_segments.submit", {"jobs": len(jobs), "days": len(plan.days)}):
            futures = [self.transport_executor.submit(self._estimate_transport_segment, job) for job in jobs]
        grouped: Dict[int, List[TransportSegment]] = {}
        for future in futures:
            try:
                with timed_event("transport_segments.future_wait", {"jobs": len(jobs)}):
                    day_index, segment = future.result(timeout=30)
                grouped.setdefault(day_index, []).append(segment)
            except Exception as exc:
                print(f"⚠️  交通段估算失败,跳过该段: {exc}")

        for day in plan.days:
            day.transport_segments = grouped.get(day.day_index, [])
        self._log_transport_route_diagnostics(plan)
        return plan

    def _build_day_pairs(self, day) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        attractions = list(day.attractions or [])
        if not attractions:
            return []
        pairs = []
        hotel_point = {
            "name": day.hotel.name,
            "address": day.hotel.address,
            "location": day.hotel.location,
        } if day.hotel else None
        points = [{"name": item.name, "address": item.address, "location": item.location} for item in attractions]
        if hotel_point:
            pairs.append((hotel_point, points[0]))
        for idx in range(len(points) - 1):
            pairs.append((points[idx], points[idx + 1]))
        if hotel_point:
            pairs.append((points[-1], hotel_point))
        return pairs

    def _log_transport_route_diagnostics(self, plan: TripPlan) -> None:
        diagnostics = []
        far_warnings = []
        for day in plan.days:
            segments = list(day.transport_segments or [])
            distances = [float(segment.distance or 0) for segment in segments]
            max_distance = max(distances) if distances else 0.0
            total_distance = sum(distances)
            unresolved_transit = [
                {
                    "from": segment.from_name,
                    "to": segment.to_name,
                    "cost_source": segment.cost_source,
                }
                for segment in segments
                if segment.mode == "公共交通"
                and (
                    segment.cost_source in {"route_estimate", "rule_based"}
                    or "未获取到可解析" in str(segment.description or "")
                    or "工具 '" in str(segment.description or "")
                )
            ]
            item = {
                "day_index": day.day_index,
                "date": day.date,
                "segment_count": len(segments),
                "max_segment_distance_m": round(max_distance, 2),
                "total_route_distance_m": round(total_distance, 2),
                "has_unresolved_transit": bool(unresolved_transit),
                "unresolved_transit_count": len(unresolved_transit),
            }
            diagnostics.append(item)
            if max_distance > 30000 or unresolved_transit:
                far_warnings.append({**item, "unresolved_transit": unresolved_transit[:3]})
        log_event("transport_route_diagnostics", diagnostics)
        if far_warnings:
            log_event("transport_route_warning", far_warnings)

    def _estimate_transport_segment(self, job: Tuple[Any, ...]) -> Tuple[int, TransportSegment]:
        day_index, from_point, to_point, transportation, city, route_cache, route_cache_lock = job
        distance, has_geo = self._estimate_distance(from_point.get("location"), to_point.get("location"))
        mode = transportation
        if has_geo and distance < 800:
            mode = "步行"
        elif transportation == "混合":
            mode = "步行" if distance < 1500 else "公共交通"
        route_info = self._try_route_api(from_point, to_point, mode, city, route_cache, route_cache_lock)
        if route_info:
            distance = float(route_info.get("distance") or distance)
            duration = int(route_info.get("duration") or self._estimate_duration(distance, mode))
            route_cost = route_info.get("cost")
            estimated_cost = int(route_cost) if route_cost is not None else self._estimate_cost(distance, mode)
            cost_source = "route_fee" if route_cost is not None else "route_estimate"
            description = str(route_info.get("description") or "")
        else:
            duration = self._estimate_duration(distance, mode)
            estimated_cost = self._estimate_cost(distance, mode)
            cost_source = "route_estimate" if has_geo else "rule_based"
            description = self._fallback_transport_description(mode, distance, has_geo)
            log_event(
                "transport_distance_fallback",
                {
                    "from_name": from_point.get("name", ""),
                    "to_name": to_point.get("name", ""),
                    "mode": mode,
                    "has_geo": has_geo,
                    "distance": round(distance, 2),
                    "source": cost_source,
                    "description": description,
                },
            )
        segment = TransportSegment(
            from_name=from_point.get("name", "起点"),
            to_name=to_point.get("name", "终点"),
            mode=mode,
            distance=distance,
            duration=duration,
            estimated_cost=estimated_cost,
            cost_source=cost_source,
            description=description,
        )
        return day_index, segment

    def _estimate_distance(self, loc1: Any, loc2: Any) -> Tuple[float, bool]:
        point1 = self._read_location(loc1)
        point2 = self._read_location(loc2)
        if not point1 or not point2:
            return 3000.0, False
        lon1, lat1 = point1
        lon2, lat2 = point2
        radius = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lam = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(d_lam / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return max(radius * c, 100.0), True

    def _read_location(self, loc: Any) -> Tuple[float, float] | None:
        if not loc:
            return None
        try:
            if hasattr(loc, "longitude") and hasattr(loc, "latitude"):
                lon = float(loc.longitude)
                lat = float(loc.latitude)
            elif isinstance(loc, dict):
                lon = float(loc["longitude"])
                lat = float(loc["latitude"])
            else:
                return None
        except Exception:
            return None
        if abs(lon) < 0.000001 and abs(lat) < 0.000001:
            return None
        return lon, lat

    def _try_route_api(
        self,
        from_point: Dict[str, Any],
        to_point: Dict[str, Any],
        mode: str,
        city: str,
        route_cache: Dict[str, Dict[str, Any]],
        route_cache_lock: Lock,
    ) -> Dict[str, Any]:
        route_type = self._route_type_from_mode(mode)
        origin = self._route_address(from_point)
        destination = self._route_address(to_point)
        if not origin or not destination:
            return {}
        cache_key = f"{route_type}|{origin}|{destination}"
        with route_cache_lock:
            cached = route_cache.get(cache_key)
        if cached is not None:
            return cached

        log_event(
            "transport_route_api_request",
            {
                "from_name": from_point.get("name", ""),
                "to_name": to_point.get("name", ""),
                "route_type": route_type,
            },
        )
        future = self.route_executor.submit(
            get_amap_service().plan_route,
            origin,
            destination,
            city or None,
            city or None,
            route_type,
            self._location_dict(from_point.get("location")),
            self._location_dict(to_point.get("location")),
        )
        try:
            with timed_event("transport.route_api", {"route_type": route_type}):
                route_info = future.result(timeout=15)
        except TimeoutError:
            future.cancel()
            log_event(
                "transport_route_api_error",
                {"from": origin, "to": destination, "route_type": route_type, "error": "timeout"},
            )
            return {}
        except Exception as exc:
            log_event(
                "transport_route_api_error",
                {"from": origin, "to": destination, "route_type": route_type, "error": repr(exc)},
            )
            return {}

        if not self._valid_route_info(route_info):
            log_event(
                "transport_route_api_error",
                {"from": origin, "to": destination, "route_type": route_type, "error": "empty_or_unparsed_result"},
            )
            route_info = {}
        else:
            log_event(
                "transport_route_api_result",
                {
                    "from": origin,
                    "to": destination,
                    "route_type": route_type,
                    "distance": route_info.get("distance"),
                    "duration": route_info.get("duration"),
                    "has_cost": route_info.get("cost") is not None,
                    "description": route_info.get("description", ""),
                },
            )
        with route_cache_lock:
            route_cache[cache_key] = route_info
        return route_info

    def _route_type_from_mode(self, mode: str) -> str:
        if mode == "步行":
            return "walking"
        if mode == "自驾":
            return "driving"
        return "transit"

    def _route_address(self, point: Dict[str, Any]) -> str:
        address = str(point.get("address") or "").strip()
        name = str(point.get("name") or "").strip()
        if address and name and name not in address:
            return f"{address} {name}"
        return address or name

    def _location_dict(self, loc: Any) -> Dict[str, float] | None:
        point = self._read_location(loc)
        if not point:
            return None
        lon, lat = point
        return {"longitude": lon, "latitude": lat}

    def _valid_route_info(self, route_info: Any) -> bool:
        if not isinstance(route_info, dict):
            return False
        try:
            return float(route_info.get("distance") or 0) > 0 or int(route_info.get("duration") or 0) > 0
        except Exception:
            return False

    def _fallback_transport_description(self, mode: str, distance_m: float, has_geo: bool) -> str:
        if mode == "公共交通":
            if not has_geo:
                return "未获取到可用坐标，公交/地铁路线请以高德地图实时导航为准"
            if distance_m >= 30000:
                return "未获取到可解析的公交/地铁换乘方案；该段距离较远，建议以高德地图实时导航或城际交通为准"
            return "未获取到可解析的公交/地铁换乘方案，请以高德地图实时导航为准"
        if mode == "自驾":
            return "未获取到可解析的自驾路线，距离和时间为规则估算，请以实时导航为准"
        if mode == "步行":
            return "未获取到可解析的步行路线，距离和时间为规则估算，请以实时导航为准"
        return ""

    def _estimate_duration(self, distance_m: float, mode: str) -> int:
        speeds = {"步行": 4.0, "公共交通": 25.0, "自驾": 35.0}
        speed = speeds.get(mode, 25.0)
        seconds = int(distance_m / (speed * 1000.0 / 3600.0))
        return max(seconds, 300)

    def _estimate_cost(self, distance_m: float, mode: str) -> int:
        km = max(distance_m / 1000.0, 0.1)
        if mode == "步行":
            return 0
        if mode == "公共交通":
            return min(20, max(2, math.ceil(km * 0.8)))
        if mode == "自驾":
            return math.ceil(km * 0.9)
        return min(20, max(2, math.ceil(km * 0.8)))

    def _aggregate_budget(self, plan: TripPlan, accommodation: str) -> TripPlan:
        total_attractions = total_hotels = total_meals = total_transportation = 0
        for day in plan.days:
            for meal in day.meals:
                if meal.estimated_cost <= 0:
                    print(
                        f"INFO autofill meal_cost | date={day.date} meal_type={meal.type} meal_name={meal.name} default_cost={MEAL_DEFAULTS.get(meal.type, 30)}",
                        flush=True,
                    )
                    meal.estimated_cost = MEAL_DEFAULTS.get(meal.type, 30)
            if day.hotel and day.hotel.estimated_cost <= 0:
                print(
                    f"INFO autofill hotel_cost | date={day.date} hotel_name={day.hotel.name} accommodation={accommodation} default_cost={HOTEL_DEFAULTS.get(accommodation, 500)}",
                    flush=True,
                )
                day.hotel.estimated_cost = HOTEL_DEFAULTS.get(accommodation, 500)
                day.hotel.price_source = "default_estimate"
            elif day.hotel and day.hotel.price_source not in {"amap_cost", "tuniu_detail_price", "tuniu_lowest_price"}:
                day.hotel.price_source = day.hotel.price_source or "llm_estimate"

            attractions_cost = sum(max(item.ticket_price, 0) for item in day.attractions)
            meals_cost = sum(max(item.estimated_cost, 0) for item in day.meals)
            hotel_cost = day.hotel.estimated_cost if day.hotel else 0
            transportation_cost = sum(max(item.estimated_cost, 0) for item in day.transport_segments)
            day.day_budget = DayBudget(
                attractions=attractions_cost,
                meals=meals_cost,
                hotel=hotel_cost,
                transportation=transportation_cost,
                subtotal=attractions_cost + meals_cost + hotel_cost + transportation_cost,
            )
            total_attractions += attractions_cost
            total_hotels += hotel_cost
            total_meals += meals_cost
            total_transportation += transportation_cost

        plan.budget = Budget(
            total_attractions=total_attractions,
            total_hotels=total_hotels,
            total_meals=total_meals,
            total_transportation=total_transportation,
            total=total_attractions + total_hotels + total_meals + total_transportation,
        )
        return plan

    def _build_meal_from_existing_candidates(self, plan: TripPlan, meal_type: str) -> Meal | None:
        for day in plan.days:
            for meal in day.meals:
                if meal.name and meal.type == meal_type:
                    return Meal(
                        type=meal_type,
                        name=meal.name,
                        address=meal.address,
                        location=meal.location,
                        description=meal.description or "结合已有餐饮候选补充的用餐建议",
                        estimated_cost=meal.estimated_cost or MEAL_DEFAULTS.get(meal_type, 30),
                    )
        return None

    def _build_default_meal(self, city: str, meal_type: str) -> Meal:
        defaults = {
            "breakfast": (
                f"{city}当地早餐建议",
                "建议在当地享用一份便捷早餐，轻松开启当天行程",
            ),
            "lunch": (
                f"{city}当地午餐建议",
                "建议中午在景点或商圈附近安排一顿当地风味简餐",
            ),
            "dinner": (
                f"{city}当地晚餐建议",
                "建议晚上在当地享用特色菜，为当天行程收尾",
            ),
        }
        name, description = defaults.get(
            meal_type,
            (f"{city}用餐建议", "建议结合当日安排就近享用一餐"),
        )
        return Meal(
            type=meal_type,
            name=name,
            description=description,
            estimated_cost=MEAL_DEFAULTS.get(meal_type, 30),
        )

    def _reflect_and_fix(self, plan: TripPlan, form_snapshot: Dict[str, Any]) -> Tuple[TripPlan, Dict[str, Any]]:
        notes = []
        expected_days = int(form_snapshot.get("travel_days", len(plan.days) or 1) or 1)
        if len(plan.days) != expected_days:
            notes.append("day_count_mismatch")
            print(f"INFO autofill day_count | existing={len(plan.days)} expected={expected_days}", flush=True)
            fallback = self.planner.create_fallback_plan(form_snapshot)
            plan.days = (plan.days + fallback.days)[:expected_days]

        start_date = self._parse_date(form_snapshot.get("start_date"))
        for idx, day in enumerate(plan.days):
            day.day_index = idx
            if start_date:
                day.date = (start_date + timedelta(days=idx)).strftime("%Y-%m-%d")
            meal_types = {meal.type for meal in day.meals}
            for meal_type in ("breakfast", "lunch", "dinner"):
                if meal_type not in meal_types:
                    notes.append(f"day_{idx}_missing_{meal_type}")
                    candidate_meal = self._build_meal_from_existing_candidates(plan, meal_type)
                    if candidate_meal is not None:
                        print(
                            f"INFO autofill meal_from_candidate | day_index={idx} date={day.date} meal_type={meal_type} meal_name={candidate_meal.name}",
                            flush=True,
                        )
                        day.meals.append(candidate_meal)
                    else:
                        print(
                            f"INFO autofill meal_default | day_index={idx} date={day.date} meal_type={meal_type} default_cost={MEAL_DEFAULTS[meal_type]}",
                            flush=True,
                        )
                        day.meals.append(self._build_default_meal(form_snapshot.get("city", ""), meal_type))
            if not day.attractions:
                notes.append(f"day_{idx}_missing_attractions")
                print(f"INFO autofill attractions | day_index={idx} date={day.date}", flush=True)
                day.attractions = self.planner.create_fallback_plan(form_snapshot).days[0].attractions

        if len(plan.weather_info) < expected_days:
            notes.append("weather_info_incomplete")
            print(
                f"INFO autofill weather_info | existing={len(plan.weather_info)} expected={expected_days}",
                flush=True,
            )
            existing_dates = {info.date for info in plan.weather_info}
            for day in plan.days:
                if day.date not in existing_dates:
                    plan.weather_info.append(
                        WeatherInfo(
                            date=day.date,
                            day_weather="多云",
                            night_weather="晴",
                            day_temp=25,
                            night_temp=18,
                            wind_direction="东风",
                            wind_power="1-3级",
                        )
                    )

        self._aggregate_budget(plan, form_snapshot.get("accommodation", "舒适型酒店"))
        return plan, {"timestamp": now_iso(), "status": "fixed" if notes else "ok", "notes": notes}

    def _parse_date(self, value: Any):
        if not value:
            return None
        try:
            return datetime.strptime(str(value), "%Y-%m-%d")
        except Exception:
            return None

    def _dict_to_request(self, form_snapshot: Dict[str, Any]) -> TripRequest:
        return TripRequest(
            nickname=form_snapshot.get("nickname", "User"),
            city=form_snapshot.get("city", "北京"),
            start_date=form_snapshot.get("start_date", datetime.now().strftime("%Y-%m-%d")),
            end_date=form_snapshot.get("end_date", datetime.now().strftime("%Y-%m-%d")),
            travel_days=int(form_snapshot.get("travel_days", 1) or 1),
            transportation=form_snapshot.get("transportation", "公共交通"),
            accommodation=form_snapshot.get("accommodation", "舒适型酒店"),
            preferences=form_snapshot.get("preferences", []) or [],
            free_text_input=form_snapshot.get("free_text_input", ""),
        )

    def _dict_to_plan(self, data: Dict[str, Any], form_snapshot: Dict[str, Any]) -> TripPlan:
        if data:
            try:
                return TripPlan(**data)
            except Exception:
                pass
        return self.planner.create_fallback_plan(form_snapshot)

    def _build_budget_ledger(self, plan: TripPlan) -> Dict[str, Any]:
        return {
            "days": [day.day_budget.model_dump() for day in plan.days],
            "total_attractions": plan.budget.total_attractions if plan.budget else 0,
            "total_hotels": plan.budget.total_hotels if plan.budget else 0,
            "total_meals": plan.budget.total_meals if plan.budget else 0,
            "total_transportation": plan.budget.total_transportation if plan.budget else 0,
            "total": plan.budget.total if plan.budget else 0,
        }

    def _persist_chat_turn(
        self,
        task_record: Dict[str, Any],
        plan: TripPlan,
        form_snapshot: Dict[str, Any],
        user_message: str,
        assistant_message: str,
        update_mode: str,
        reflection: Dict[str, Any] | None = None,
    ) -> None:
        conversation_log = list(task_record.get("conversation_log", []) or [])
        conversation_log.extend(
            [
                {"role": "user", "message": user_message, "timestamp": now_iso()},
                {"role": "assistant", "message": assistant_message, "timestamp": now_iso(), "update_mode": update_mode},
            ]
        )

        task_record["current_plan"] = plan.model_dump()
        task_record["form_snapshot"] = form_snapshot
        task_record["budget_ledger"] = self._build_budget_ledger(plan)
        if reflection is not None:
            task_record["reflection_log"] = task_record.get("reflection_log", []) + [reflection]
        task_record["conversation_log"] = conversation_log
        task_record["update_mode"] = update_mode
        task_record["city"] = plan.city
        task_record["date_range"] = f"{plan.start_date} to {plan.end_date}"
        task_record["travel_days"] = len(plan.days)
        with timed_event("chat.write_task", {"task_id": task_record["task_id"], "update_mode": update_mode}):
            self.memory_store.write_task(task_record)

    def _answer_question(self, current_plan: TripPlan, user_message: str) -> str:
        fallback = "我可以继续帮你查看当前旅行计划，但这个问题在现有计划里没有足够信息。你可以再具体问某一天、某个景点、酒店、餐饮或预算。"
        try:
            client = get_cheap_openai_client()
            model = get_cheap_openai_model()
            payload = {
                "question": user_message,
                "current_plan": current_plan.model_dump(),
            }
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是旅游规划助手。只基于 current_plan 回答用户关于当前旅行计划的问题；"
                            "不要修改计划，不要新增景点/酒店/餐饮。计划里没有的信息要明确说明。"
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0,
                max_tokens=self.context_compressor.settings.cheap_model_max_output_tokens,
            )
            answer = (response.choices[0].message.content or "").strip()
            return answer or fallback
        except Exception as exc:
            log_event("question_answer_error", str(exc))
            return fallback

    def _build_non_persist_message(self, intent: str) -> str:
        if intent == "refuse":
            return "对不起，我只是一个旅游规划助手，不能回答这个问题。你可以继续告诉我想调整的目的地、日期、景点、酒店、餐饮或交通。"
        return "我在。关于这份旅行计划，你可以继续告诉我想改景点、酒店、餐饮、交通、日期，或者让我整体重新安排。"

    def _build_revision_message(self, update_mode: str) -> str:
        return "已根据你的意见完成局部调整。" if update_mode == "patch" else "你的需求涉及全局约束，已重新规划完整行程。"

    def _summarize_task(self, task_record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": task_record.get("task_id"),
            "city": task_record.get("city"),
            "travel_days": task_record.get("travel_days"),
            "update_mode": task_record.get("update_mode"),
            "conversation_size": len(task_record.get("conversation_log", [])),
        }

    def _fork_context(self, source: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(json.dumps(source, ensure_ascii=False))

    def _start_user_profile_update(self, context: Dict[str, Any]) -> None:
        self.profile_executor.submit(self.user_profile_agent.update_profile, context)

    def _start_context_compression_if_needed(self, task_record: Dict[str, Any]) -> None:
        if not self.context_compressor.needs_heavy_refresh(task_record):
            return
        task_id = task_record.get("task_id", "")
        conversation_log = deepcopy(task_record.get("conversation_log", []) or [])
        log_event(
            "context_compression_background_submit",
            {"task_id": task_id, "conversation_entries": len(conversation_log)},
        )
        self.context_executor.submit(self.context_compressor.refresh_heavy_summary, task_id, conversation_log)


_trip_task_executor: TripTaskExecutor | None = None


def get_trip_task_executor() -> TripTaskExecutor:
    global _trip_task_executor
    if _trip_task_executor is None:
        _trip_task_executor = TripTaskExecutor()
    return _trip_task_executor
