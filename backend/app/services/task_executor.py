"""Trip task orchestration service."""
# -*- coding: utf-8 -*-
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

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


class TripTaskExecutor:
    def __init__(self) -> None:
        self.memory_store = get_memory_store()
        self.planner = get_trip_planner_agent()
        self.user_profile_agent = UserProfileAgent(self.memory_store)
        self.retrieval_executor = ThreadPoolExecutor(max_workers=4)
        self.transport_executor = ThreadPoolExecutor(max_workers=8)
        self.profile_executor = ThreadPoolExecutor(max_workers=1)

    def plan_initial(self, request: TripRequest) -> TripPlanResponse:
        print(
            f"INFO plan_initial start | city={request.city} travel_days={request.travel_days} accommodation={request.accommodation}",
            flush=True,
        )
        task_record, user_memory = self._bootstrap_task(request)
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

        retrieval_results = self._fetch_initial_context_parallel(request)
        print("=" * 80, flush=True)
        print(f"INFO retrieval completed | keys={list(retrieval_results.keys())}", flush=True)
        print("=" * 80, flush=True)
        user_profile_summary = json.dumps(user_memory.get("profile", {}), ensure_ascii=False)
        print("INFO plan_initial build_plan_from_context start", flush=True)
        plan = self.planner.build_plan_from_context(
            request=request,
            attractions=retrieval_results["attractions"],
            weather=retrieval_results["weather"],
            hotels=retrieval_results["hotels"],
            restaurants=retrieval_results["restaurants"],
            user_profile_summary=user_profile_summary,
            extra_requirements=request.free_text_input or "",
        )
        print("INFO plan_initial build_plan_from_context end", flush=True)
        print("INFO plan_initial add_transport start", flush=True)
        plan = self._add_transport_segments_parallel(plan, request.transportation)
        print("INFO plan_initial add_transport end", flush=True)
        print("INFO plan_initial aggregate_budget start", flush=True)
        plan = self._aggregate_budget(plan, request.accommodation)
        print("INFO plan_initial aggregate_budget end", flush=True)
        print("INFO plan_initial reflect_and_fix start", flush=True)
        plan, reflection = self._reflect_and_fix(plan, request.model_dump())
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
        self.memory_store.write_task(task_record)

        return TripPlanResponse(
            success=True,
            message="旅行计划生成成功",
            task_id=task_record["task_id"],
            user_id=task_record["user_id"],
            update_mode="initial",
            assistant_message=assistant_message,
            data=plan,
        )

    def chat(self, request: TripChatRequest) -> TripPlanResponse:
        task_record = self.memory_store.read_task(request.task_id)
        form_snapshot = task_record.get("form_snapshot", {}) or {}
        user_memory = self.memory_store.load_or_create_user_memory(
            task_record["user_id"],
            task_record.get("nickname", form_snapshot.get("nickname", "")),
        )
        self._start_user_profile_update(
            self._fork_context(
                {
                    "kind": "chat",
                    "task_id": task_record["task_id"],
                    "user_id": task_record["user_id"],
                    "nickname": task_record.get("nickname", ""),
                    "task_summary": self._summarize_task(task_record),
                    "latest_user_message": request.user_message,
                    "user_memory": user_memory.get("profile", {}),
                    "form": form_snapshot,
                }
            )
        )

        update_mode = self._route_revision_intent(request.user_message)
        current_plan = self._dict_to_plan(task_record.get("current_plan", {}), form_snapshot)
        user_profile_summary = json.dumps(user_memory.get("profile", {}), ensure_ascii=False)

        if update_mode == "patch":
            patch_context = self._fetch_patch_context_parallel(request.user_message, form_snapshot)
            revised = self.planner.revise_plan(
                current_plan=current_plan,
                form_snapshot=form_snapshot,
                user_message=request.user_message,
                patch_context=patch_context,
                user_profile_summary=user_profile_summary,
            )
            if self._validate_patch_result(revised, form_snapshot):
                plan = revised
            else:
                update_mode = "replan"
                plan, form_snapshot = self._replan_from_message(form_snapshot, request.user_message, user_profile_summary)
        else:
            plan, form_snapshot = self._replan_from_message(form_snapshot, request.user_message, user_profile_summary)

        plan = self._add_transport_segments_parallel(plan, form_snapshot.get("transportation", "公共交通"))
        plan = self._aggregate_budget(plan, form_snapshot.get("accommodation", "舒适型酒店"))
        plan, reflection = self._reflect_and_fix(plan, form_snapshot)

        assistant_message = self._build_revision_message(update_mode)
        conversation_log = task_record.get("conversation_log", [])
        conversation_log.extend(
            [
                {"role": "user", "message": request.user_message, "timestamp": now_iso()},
                {"role": "assistant", "message": assistant_message, "timestamp": now_iso(), "update_mode": update_mode},
            ]
        )

        task_record["current_plan"] = plan.model_dump()
        task_record["form_snapshot"] = form_snapshot
        task_record["budget_ledger"] = self._build_budget_ledger(plan)
        task_record["reflection_log"] = task_record.get("reflection_log", []) + [reflection]
        task_record["conversation_log"] = conversation_log
        task_record["update_mode"] = update_mode
        task_record["city"] = plan.city
        task_record["date_range"] = f"{plan.start_date} to {plan.end_date}"
        task_record["travel_days"] = len(plan.days)
        self.memory_store.write_task(task_record)

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
        futures = {
            "attractions": self.retrieval_executor.submit(self.planner.search_attractions, request, ""),
            "hotels": self.retrieval_executor.submit(self.planner.search_hotels, request.city, request.accommodation),
            "restaurants": self.retrieval_executor.submit(
                self.planner.search_restaurants,
                request.city,
                request.preferences[0] if request.preferences else "",
            ),
            "weather": self.retrieval_executor.submit(
                self.planner.search_weather,
                request.city,
                request.start_date,
                request.end_date,
            ),
        }
        return self._wait_future_map(futures)

    def _fetch_patch_context_parallel(self, message: str, form_snapshot: Dict[str, Any]) -> Dict[str, str]:
        request = self._dict_to_request(form_snapshot)
        selected_domains = {domain for domain, patterns in PATCH_DOMAIN_KEYWORDS.items() if any(re.search(p, message) for p in patterns)}
        if not selected_domains:
            selected_domains.add("attractions")

        futures: Dict[str, Any] = {}
        if "attractions" in selected_domains:
            futures["attractions"] = self.retrieval_executor.submit(self.planner.search_attractions, request, "")
        if "hotels" in selected_domains:
            futures["hotels"] = self.retrieval_executor.submit(self.planner.search_hotels, request.city, request.accommodation)
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

    def _wait_future_map(self, futures: Dict[str, Any]) -> Dict[str, str]:
        results: Dict[str, str] = {}
        for key, future in futures.items():
            try:
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

    def _replan_from_message(self, form_snapshot: Dict[str, Any], user_message: str, user_profile_summary: str) -> Tuple[TripPlan, Dict[str, Any]]:
        merged_form = self._merge_message_into_form(form_snapshot, user_message)
        request = self._dict_to_request(merged_form)
        retrieval_results = self._fetch_initial_context_parallel(request)
        plan = self.planner.build_plan_from_context(
            request=request,
            attractions=retrieval_results.get("attractions", ""),
            weather=retrieval_results.get("weather", ""),
            hotels=retrieval_results.get("hotels", ""),
            restaurants=retrieval_results.get("restaurants", ""),
            user_profile_summary=user_profile_summary,
            extra_requirements=request.free_text_input or "",
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
        for day in plan.days:
            for from_point, to_point in self._build_day_pairs(day):
                jobs.append((day.day_index, from_point, to_point, transportation))

        futures = [self.transport_executor.submit(self._estimate_transport_segment, job) for job in jobs]
        grouped: Dict[int, List[TransportSegment]] = {}
        for future in futures:
            try:
                day_index, segment = future.result(timeout=30)
                grouped.setdefault(day_index, []).append(segment)
            except Exception as exc:
                print(f"⚠️  交通段估算失败,跳过该段: {exc}")

        for day in plan.days:
            day.transport_segments = grouped.get(day.day_index, [])
        return plan

    def _build_day_pairs(self, day) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        attractions = list(day.attractions or [])
        if not attractions:
            return []
        pairs = []
        hotel_point = {"name": day.hotel.name, "location": day.hotel.location} if day.hotel else None
        points = [{"name": item.name, "location": item.location} for item in attractions]
        if hotel_point:
            pairs.append((hotel_point, points[0]))
        for idx in range(len(points) - 1):
            pairs.append((points[idx], points[idx + 1]))
        if hotel_point:
            pairs.append((points[-1], hotel_point))
        return pairs

    def _estimate_transport_segment(self, job: Tuple[int, Dict[str, Any], Dict[str, Any], str]) -> Tuple[int, TransportSegment]:
        day_index, from_point, to_point, transportation = job
        distance, has_geo = self._estimate_distance(from_point.get("location"), to_point.get("location"))
        mode = transportation
        if transportation == "混合":
            mode = "步行" if distance < 1500 else "公共交通"
        segment = TransportSegment(
            from_name=from_point.get("name", "起点"),
            to_name=to_point.get("name", "终点"),
            mode=mode,
            distance=distance,
            duration=self._estimate_duration(distance, mode),
            estimated_cost=self._estimate_cost(distance, mode),
            cost_source="route_estimate" if has_geo else "rule_based",
        )
        return day_index, segment

    def _estimate_distance(self, loc1: Any, loc2: Any) -> Tuple[float, bool]:
        if not loc1 or not loc2:
            return 3000.0, False
        try:
            lon1 = float(getattr(loc1, "longitude", loc1["longitude"]))
            lat1 = float(getattr(loc1, "latitude", loc1["latitude"]))
            lon2 = float(getattr(loc2, "longitude", loc2["longitude"]))
            lat2 = float(getattr(loc2, "latitude", loc2["latitude"]))
        except Exception:
            return 3000.0, False
        radius = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lam = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(d_lam / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return max(radius * c, 100.0), True

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


_trip_task_executor: TripTaskExecutor | None = None


def get_trip_task_executor() -> TripTaskExecutor:
    global _trip_task_executor
    if _trip_task_executor is None:
        _trip_task_executor = TripTaskExecutor()
    return _trip_task_executor
