# -*- coding: utf-8 -*-
"""Trip planning multi-agent container."""

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from hello_agents import SimpleAgent
from pydantic import BaseModel, Field

from ..config import get_settings
from ..models.schemas import Attraction, DayPlan, Location, Meal, TripPlan, TripRequest
from ..services.agent_output_logger import log_event, log_full_output, timed_event
from ..services.amap_tool_pool import AmapWorkerPool
from ..services.llm_service import get_llm, get_openai_client, get_openai_model


ATTRACTION_AGENT_PROMPT = "你是景点搜索专家，必须优先调用地图工具检索景点。"
WEATHER_AGENT_PROMPT = "你是天气检索专家，必须优先调用地图天气工具。"
HOTEL_AGENT_PROMPT = "你是酒店搜索专家，必须优先调用地图工具检索酒店。"


STRUCTURED_PLAN_FEW_SHOT = """
Few-shot output example:
{
  "city": "北京",
  "start_date": "2026-04-24",
  "end_date": "2026-04-26",
  "days": [
    {
      "date": "2026-04-24",
      "day_index": 0,
      "description": "第一天安排城市经典景点与本地餐饮",
      "transportation": "公共交通",
      "accommodation": "经济型酒店",
      "hotel": {
        "source_candidate_id": "hotel_0_demo",
        "description": "靠近主要景点，交通便利",
        "visit_duration": 0,
        "estimated_cost": 320,
        "type": "经济型酒店"
      },
      "attractions": [
        {
          "source_candidate_id": "attraction_0_demo",
          "description": "城市代表性历史文化景点",
          "visit_duration": 120,
          "estimated_cost": 50,
          "type": "attraction"
        },
        {
          "source_candidate_id": "attraction_1_demo",
          "description": "与当天路线顺路的补充景点",
          "visit_duration": 90,
          "estimated_cost": 0,
          "type": "attraction"
        }
      ],
      "meals": [
        {
          "source_candidate_id": "restaurant_breakfast_demo",
          "description": "酒店附近便捷早餐",
          "visit_duration": 45,
          "estimated_cost": 25,
          "type": "breakfast"
        },
        {
          "source_candidate_id": "restaurant_lunch_demo",
          "description": "景点周边本地特色午餐",
          "visit_duration": 60,
          "estimated_cost": 60,
          "type": "lunch"
        },
        {
          "source_candidate_id": "restaurant_dinner_demo",
          "description": "适合作为当天收尾的特色晚餐",
          "visit_duration": 75,
          "estimated_cost": 90,
          "type": "dinner"
        }
      ]
    }
  ],
  "weather_info": [
    {
      "date": "2026-04-24",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 24,
      "night_temp": 16,
      "wind_direction": "东风",
      "wind_power": "1-3级"
    }
  ],
  "overall_suggestions": "建议穿舒适鞋并预留机动时间",
  "budget": {
    "total_attractions": 50,
    "total_hotels": 320,
    "total_meals": 175,
    "total_transportation": 30,
    "total": 575
  }
}
"""

ATTRACTIONS_PATCH_FEW_SHOT = """
{
  "days": [
    {
      "day_index": 0,
      "date": "2026-04-25",
      "description": "第一天增加更多历史文化景点，其他安排保持不变。",
      "attractions": [
        {
          "source_candidate_id": "attraction_1_demo",
          "description": "新增一个博物馆候选",
          "visit_duration": 120,
          "estimated_cost": 0,
          "type": "attraction"
        },
        {
          "source_candidate_id": "attraction_2_demo",
          "description": "新增一个古建筑候选",
          "visit_duration": 90,
          "estimated_cost": 0,
          "type": "attraction"
        }
      ]
    }
  ]
}
"""

RESTAURANT_AGENT_PROMPT = "你是餐饮搜索专家，必须优先调用地图工具检索餐馆。"

INITIAL_CANDIDATE_MIN = 8
INITIAL_CANDIDATE_MAX = 30
INITIAL_CANDIDATE_MULTIPLIER = 3.5
INITIAL_SEARCH_WORKERS = 4
MEAL_COST_DEFAULTS = {"breakfast": 25, "lunch": 60, "dinner": 90}
SAME_DAY_IDEAL_MAX_DISTANCE_M = 8000.0
SAME_DAY_HARD_MAX_DISTANCE_M = 15000.0

CONTEXT_PRIORITY_RULES = [
    "历史用户画像只作为长期偏好背景。",
    "当前表单是用户本次明确选择；当它与历史用户画像冲突时，以当前表单为准。",
    "当前自由输入或当前聊天消息是最新意图；除非它违反必填表单硬约束，否则优先级最高。",
    "候选池是酒店、景点、餐饮和天气证据的允许来源，不要编造候选池之外的地点。",
]


class CandidateRef(BaseModel):
    source_candidate_id: str = Field(..., description="必须来自候选池")
    description: str = ""
    visit_duration: int = 120
    estimated_cost: int = 0
    type: str = ""


class DayPlanDraft(BaseModel):
    date: str
    day_index: int
    description: str
    transportation: str
    accommodation: str
    hotel: Optional[CandidateRef] = None
    attractions: List[CandidateRef] = Field(default_factory=list)
    meals: List[CandidateRef] = Field(default_factory=list)


class WeatherInfoDraft(BaseModel):
    date: str
    day_weather: str = "多云"
    night_weather: str = "晴"
    day_temp: int = 25
    night_temp: int = 18
    wind_direction: str = "东风"
    wind_power: str = "1-3级"


class BudgetDraft(BaseModel):
    total_attractions: int = 0
    total_hotels: int = 0
    total_meals: int = 0
    total_transportation: int = 0
    total: int = 0


class TripPlanDraft(BaseModel):
    city: str
    start_date: str
    end_date: str
    days: List[DayPlanDraft]
    weather_info: List[WeatherInfoDraft] = Field(default_factory=list)
    overall_suggestions: str
    budget: Optional[BudgetDraft] = None


class AttractionRefPatchDraft(BaseModel):
    source_candidate_id: Optional[str] = None
    description: Optional[str] = None
    visit_duration: Optional[int] = None
    estimated_cost: Optional[int] = None
    type: Optional[str] = None


class DayAttractionsPatchDraft(BaseModel):
    day_index: Optional[int] = None
    date: Optional[str] = None
    description: Optional[str] = None
    attractions: List[AttractionRefPatchDraft] = Field(default_factory=list)


class AttractionsPatchDraft(BaseModel):
    days: List[DayAttractionsPatchDraft] = Field(default_factory=list)


SEARCH_FEW_SHOT_SUFFIX = """

Few-shot guidance:
Example attraction response:
- If the user asks for historical attractions in Tianjin, prefer returning concise JSON with a "pois" array.
- Each item should include name, address, location, and a useful category/type when available.

Example hotel response:
- If the user asks for budget hotels, return concise JSON with a "pois" array of hotel-like candidates only.

Example restaurant response:
- If the user asks for local food, return concise JSON with a "pois" array of restaurant-like candidates only.

Example weather response:
- Return concise JSON with city and forecasts fields.
"""


class MultiAgentTripPlanner:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = get_llm()
        self.openai_client = get_openai_client()
        self.openai_model = get_openai_model()
        self.amap_pool = AmapWorkerPool(
            llm=self.llm,
            prompts={
                "attractions": ATTRACTION_AGENT_PROMPT + SEARCH_FEW_SHOT_SUFFIX,
                "weather": WEATHER_AGENT_PROMPT + SEARCH_FEW_SHOT_SUFFIX,
                "hotels": HOTEL_AGENT_PROMPT + SEARCH_FEW_SHOT_SUFFIX,
                "restaurants": RESTAURANT_AGENT_PROMPT + SEARCH_FEW_SHOT_SUFFIX,
            },
        )
        self.amap_pool.start()

    def search_attractions(self, request: TripRequest, keyword_hint: str = "") -> str:
        keyword = keyword_hint or (request.preferences[0] if request.preferences else "景点")
        return self._safe_run("attractions", f"请搜索{request.city}适合{keyword}偏好的景点，返回名称、地址和坐标。")

    def search_initial_attractions(self, request: TripRequest) -> str:
        target = self._dynamic_candidate_target(request.travel_days)
        keywords = self._build_initial_attraction_keywords(request)
        return self._search_initial_candidate_pool(
            kind="attraction",
            target=target,
            keywords=keywords,
            search_one=lambda keyword: self.search_attractions(request, keyword),
            query_log_name="initial_attraction_query_plan",
            count_log_name="initial_attraction_candidate_count",
        )

    def search_weather(self, city: str, start_date: str = "", end_date: str = "") -> str:
        if start_date and end_date:
            query = f"请查询{city}在{start_date}到{end_date}这几天的天气情况，重点返回这段日期内每天的天气。"
        elif start_date:
            query = f"请查询{city}在{start_date}当天的天气情况。"
        else:
            query = f"请查询{city}最近天气情况。"
        return self._safe_run("weather", query)

    def search_hotels(self, city: str, accommodation: str) -> str:
        return self._safe_run("hotels", f"请搜索{city}的{accommodation}酒店。")

    def search_restaurants(self, city: str, preference_hint: str = "") -> str:
        keyword = preference_hint or "本地特色餐厅"
        return self._safe_run("restaurants", f"请搜索{city}的{keyword}。")

    def search_initial_restaurants(self, request: TripRequest) -> str:
        target = self._dynamic_candidate_target(request.travel_days)
        keywords = self._build_initial_restaurant_keywords(request)
        return self._search_initial_candidate_pool(
            kind="restaurant",
            target=target,
            keywords=keywords,
            search_one=lambda keyword: self.search_restaurants(request.city, keyword),
            query_log_name="initial_restaurant_query_plan",
            count_log_name="initial_restaurant_candidate_count",
        )

    def build_plan_from_context(
        self,
        request: TripRequest,
        attractions: str,
        weather: str,
        hotels: str,
        restaurants: str,
        user_profile_summary: str = "",
        extra_requirements: str = "",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> TripPlan:
        candidate_context = self.build_candidate_context(attractions, weather, hotels, restaurants, request.city)
        planning_constraints = self.build_initial_planning_constraints(request)
        self._attach_attraction_geo_clusters(candidate_context, planning_constraints)
        draft = self._parse_structured_plan(
            (
                "你是旅行规划专家。你只能从候选池中选择酒店、景点、餐饮。每个选择必须使用 source_candidate_id。"
                "必须遵守 planning_constraints：候选足够时，每天景点数达到 min_attractions_per_day，尽量接近 "
                "target_attractions_per_day，且不超过 max_attractions_per_day。每天必须包含 breakfast、lunch、dinner。"
                "景点只能使用 attraction_candidates，酒店只能使用 hotel_candidates，餐饮只能使用 restaurant_candidates；"
                "不要把景点当作餐厅。午餐和晚餐优先选择真实餐厅，早餐可以选择真实餐厅或酒店附近简餐。"
                "同一天景点必须优先按地理邻近性分组和排序，尽量减少酒店到景点、景点到景点、景点回酒店的移动距离。"
                "当景点数量与路线紧凑冲突时，优先保证路线合理，不要为了凑数量安排过远景点。"
                "如果 candidates.attraction_geo_clusters 存在，优先从同一个 cluster 中选择同一天景点。"
            ),
            {
                "priority_rules": CONTEXT_PRIORITY_RULES,
                "conversation_history": conversation_history or [],
                "user_profile": user_profile_summary,
                "form": request.model_dump(),
                "extra_requirements": extra_requirements,
                "planning_constraints": planning_constraints,
                "candidates": candidate_context,
            },
        )
        plan = self._draft_to_trip_plan(draft, request.model_dump(), candidate_context)
        return self._ensure_initial_plan_density(plan, request.model_dump(), candidate_context, planning_constraints)

    def revise_plan(
        self,
        current_plan: TripPlan,
        form_snapshot: Dict[str, Any],
        user_message: str,
        patch_context: Dict[str, Any],
        user_profile_summary: str = "",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> TripPlan:
        candidate_context = self.build_candidate_context(
            patch_context.get("attractions", ""),
            patch_context.get("weather", ""),
            patch_context.get("hotels", ""),
            patch_context.get("restaurants", ""),
            form_snapshot.get("city", ""),
        )
        draft = self._parse_structured_plan(
            "你是旅行计划修改专家。请根据用户的新要求调整当前计划，尽量保留未受影响的酒店、餐饮和行程信息。",
            {
                "priority_rules": CONTEXT_PRIORITY_RULES,
                "conversation_history": conversation_history or [],
                "user_profile": user_profile_summary,
                "current_plan": current_plan.model_dump(),
                "form_snapshot": form_snapshot,
                "user_message": user_message,
                "patch_context": patch_context,
                "candidates": candidate_context,
            },
        )
        merged_data = self._merge_draft_with_current_plan(current_plan, draft)
        return self._coerce_json_to_trip_plan(merged_data, form_snapshot, candidate_context)

    def revise_attractions_only(
        self,
        current_plan: TripPlan,
        form_snapshot: Dict[str, Any],
        user_message: str,
        patch_context: Dict[str, Any],
        user_profile_summary: str = "",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> TripPlan:
        candidate_context = self.build_candidate_context(
            patch_context.get("attractions", ""),
            patch_context.get("weather", ""),
            patch_context.get("hotels", ""),
            patch_context.get("restaurants", ""),
            form_snapshot.get("city", ""),
        )
        self._print_full_output_block("attractions_patch_input", json.dumps({
            "user_message": user_message,
            "current_plan": current_plan.model_dump(),
            "patch_context": patch_context,
            "candidate_context": candidate_context,
        }, ensure_ascii=False))
        patch = self._parse_attractions_patch(
            {
                "priority_rules": CONTEXT_PRIORITY_RULES,
                "conversation_history": conversation_history or [],
                "user_profile": user_profile_summary,
                "current_plan": current_plan.model_dump(),
                "form_snapshot": form_snapshot,
                "user_message": user_message,
                "patch_context": patch_context,
                "candidates": candidate_context,
            }
        )
        self._print_full_output_block("parsed_attractions_patch", patch.model_dump_json(indent=2))
        self._print_full_output_block("current_plan_before_attractions_patch_merge", current_plan.model_dump_json(indent=2))
        merged_data = self._merge_attractions_patch_with_current_plan(current_plan, patch)
        self._print_full_output_block("merged_plan_after_attractions_patch_merge", json.dumps(merged_data, ensure_ascii=False))
        log_event("chat_patch_preserve", "hotel=yes meals=yes")
        return self._coerce_json_to_trip_plan(merged_data, form_snapshot, candidate_context)

    def parse_plan_response(
        self,
        response: str,
        form_snapshot: Dict[str, Any],
        candidate_context: Optional[Dict[str, Any]] = None,
    ) -> TripPlan:
        candidates = candidate_context or {}
        try:
            data = self._extract_json(response)
            return self._coerce_json_to_trip_plan(data, form_snapshot, candidates)
        except Exception as exc:
            print(f"⚠️ TripPlan首次解析失败: {exc}", flush=True)
            repaired = self._repair_plan_json(response, form_snapshot, exc, candidates)
            if repaired is not None:
                return repaired
            print(
                f"🚨 回退兜底计划: city={form_snapshot.get('city', '')} reason={exc} raw={self._preview(response, 300)}",
                flush=True,
            )
            return self.create_fallback_plan(form_snapshot)

    def create_fallback_plan(self, form_snapshot: Dict[str, Any]) -> TripPlan:
        print(
            f"🚨 生成兜底计划: city={form_snapshot.get('city', '')} days={form_snapshot.get('travel_days', 1)}",
            flush=True,
        )
        city = str(form_snapshot.get("city", "未知目的地"))
        travel_days = int(form_snapshot.get("travel_days", 1) or 1)
        transportation = str(form_snapshot.get("transportation", "公共交通"))
        accommodation = str(form_snapshot.get("accommodation", "舒适型酒店"))
        start_date_raw = str(form_snapshot.get("start_date", ""))
        end_date_raw = str(form_snapshot.get("end_date", ""))

        try:
            start_date = datetime.strptime(start_date_raw, "%Y-%m-%d")
        except Exception:
            start_date = datetime.now()
        if not end_date_raw:
            end_date_raw = (start_date + timedelta(days=max(travel_days - 1, 0))).strftime("%Y-%m-%d")

        days = []
        for idx in range(travel_days):
            date_str = (start_date + timedelta(days=idx)).strftime("%Y-%m-%d")
            days.append(
                DayPlan(
                    date=date_str,
                    day_index=idx,
                    description=f"第{idx + 1}天行程",
                    transportation=transportation,
                    accommodation=accommodation,
                    attractions=[
                        Attraction(
                            name=f"{city}景点{idx + 1}",
                            address=f"{city}市中心",
                            location=Location(longitude=116.397128 + idx * 0.01, latitude=39.916527 + idx * 0.01),
                            visit_duration=120,
                            description=f"{city}推荐景点",
                            category="景点",
                            ticket_price=60,
                        )
                    ],
                    meals=[
                        Meal(type="breakfast", name=f"{city}早餐{idx + 1}", description="建议选择便利早餐，轻松开始当天行程", estimated_cost=25),
                        Meal(type="lunch", name=f"{city}午餐{idx + 1}", description="建议在景点或商圈附近安排本地风味简餐", estimated_cost=60),
                        Meal(type="dinner", name=f"{city}晚餐{idx + 1}", description="建议晚上享用当地特色菜，为当天行程收尾", estimated_cost=90),
                    ],
                )
            )

        return TripPlan(
            city=city,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date_raw,
            days=days,
            weather_info=[],
            overall_suggestions=f"这是为你生成的{city}兜底计划，可继续在聊天里提出修改意见。",
        )

    def build_candidate_context(self, attractions: str, weather: str, hotels: str, restaurants: str, city: str) -> Dict[str, Any]:
        return {
            "city": city,
            "attraction_candidates": self._extract_candidates_from_text(attractions, "attraction"),
            "hotel_candidates": self._extract_candidates_from_text(hotels, "hotel"),
            "restaurant_candidates": self._extract_candidates_from_text(restaurants, "restaurant"),
            "weather_raw": weather,
        }

    def build_initial_planning_constraints(self, request: TripRequest) -> Dict[str, Any]:
        target_candidates = self._dynamic_candidate_target(request.travel_days)
        relaxed = self._is_relaxed_trip(request)
        return {
            "target_attraction_candidates": target_candidates,
            "target_restaurant_candidates": target_candidates,
            "min_attractions_per_day": 1 if relaxed else 2,
            "target_attractions_per_day": 2 if relaxed else 3,
            "max_attractions_per_day": 3 if relaxed else 4,
            "meal_requirements": {
                "required_types": ["breakfast", "lunch", "dinner"],
                "lunch_and_dinner_prefer_restaurant_candidates": True,
                "breakfast_can_be_simple_near_hotel": True,
                "do_not_use_attraction_candidates_as_restaurants": True,
            },
            "geo_constraints": {
                "group_same_day_attractions_by_geo_proximity": True,
                "order_attractions_to_reduce_travel_distance": True,
                "ideal_max_same_day_attraction_distance_m": int(SAME_DAY_IDEAL_MAX_DISTANCE_M),
                "hard_max_same_day_attraction_distance_m": int(SAME_DAY_HARD_MAX_DISTANCE_M),
                "prefer_route_compactness_over_filling_attraction_count": True,
                "use_attraction_geo_clusters_when_available": True,
            },
            "relaxed_pace": relaxed,
        }

    def _attach_attraction_geo_clusters(
        self,
        candidate_context: Dict[str, Any],
        planning_constraints: Dict[str, Any],
    ) -> None:
        # Pure algorithmic geo clustering: no LLM, embedding, or external route API call here.
        candidates = list(candidate_context.get("attraction_candidates", []) or [])
        geo_constraints = planning_constraints.get("geo_constraints") or {}
        hard_max = float(geo_constraints.get("hard_max_same_day_attraction_distance_m", SAME_DAY_HARD_MAX_DISTANCE_M))
        clusters = self._cluster_candidate_records(candidates, hard_max)
        candidate_context["attraction_geo_clusters"] = clusters
        log_event(
            "initial_attraction_candidate_geo_clusters",
            {
                "source": "algorithm",
                "llm_used": False,
                "cluster_count": len(clusters),
                "hard_max_distance_m": hard_max,
                "clusters": [
                    {
                        "cluster_id": item.get("cluster_id"),
                        "size": item.get("size"),
                        "candidate_ids": item.get("candidate_ids", []),
                    }
                    for item in clusters
                ],
            },
        )

    def _cluster_candidate_records(self, candidates: List[Dict[str, Any]], max_distance_m: float) -> List[Dict[str, Any]]:
        # Greedy distance clustering over candidate coordinates. Keep this deterministic and model-free.
        clusters: List[Dict[str, Any]] = []
        for candidate in candidates:
            location = self._candidate_location(candidate)
            if self._is_zero_location(location):
                continue
            best_index = -1
            best_distance = float("inf")
            for idx, cluster in enumerate(clusters):
                distance = min(
                    self._distance_between_locations(location, member["location"])
                    for member in cluster["members"]
                )
                if distance < best_distance:
                    best_index = idx
                    best_distance = distance
            member = {
                "candidate_id": candidate.get("candidate_id", ""),
                "name": candidate.get("name", ""),
                "location": location,
            }
            if best_index >= 0 and best_distance <= max_distance_m:
                clusters[best_index]["members"].append(member)
            else:
                clusters.append({"members": [member]})

        output: List[Dict[str, Any]] = []
        for idx, cluster in enumerate(clusters):
            members = cluster["members"]
            lng = sum(item["location"].longitude for item in members) / len(members)
            lat = sum(item["location"].latitude for item in members) / len(members)
            output.append(
                {
                    "cluster_id": f"cluster_{idx}",
                    "size": len(members),
                    "center": {"longitude": round(lng, 6), "latitude": round(lat, 6)},
                    "candidate_ids": [item["candidate_id"] for item in members],
                    "names": [item["name"] for item in members],
                }
            )
        output.sort(key=lambda item: item["size"], reverse=True)
        return output

    def _dynamic_candidate_target(self, travel_days: Any) -> int:
        try:
            days = max(1, int(travel_days or 1))
        except Exception:
            days = 1
        return max(INITIAL_CANDIDATE_MIN, min(INITIAL_CANDIDATE_MAX, math.ceil(days * INITIAL_CANDIDATE_MULTIPLIER)))

    def _is_relaxed_trip(self, request: TripRequest) -> bool:
        text = " ".join([*(request.preferences or []), request.free_text_input or ""]).lower()
        return any(token in text for token in ("轻松", "休闲", "不赶", "别太赶", "慢节奏", "老人", "亲子", "带娃"))

    def _build_initial_attraction_keywords(self, request: TripRequest) -> List[str]:
        keywords: List[str] = []
        keywords.extend(request.preferences or [])
        keywords.extend(["必去景点", "经典景点", "博物馆 历史文化", "自然风光 公园 山水", "商圈 街区"])
        free_text = request.free_text_input or ""
        if any(token in free_text for token in ("博物馆", "历史", "文化", "古迹")):
            keywords.append("博物馆 历史文化 古迹")
        if any(token in free_text for token in ("自然", "风景", "山", "海", "湖", "公园")):
            keywords.append("自然风光 公园 山水")
        return self._unique_nonempty_texts(keywords)[:10]

    def _build_initial_restaurant_keywords(self, request: TripRequest) -> List[str]:
        keywords = ["本地特色餐厅", "特色菜", "小吃", "早餐", "午餐", "晚餐", "夜市", "商圈美食"]
        food_tokens = ("美食", "餐", "吃", "小吃", "菜", "海鲜", "火锅", "素食", "清真", "夜市", "咖啡")
        for item in [*(request.preferences or []), request.free_text_input or ""]:
            if any(token in item for token in food_tokens):
                keywords.append(item)
        return self._unique_nonempty_texts(keywords)[:10]

    def _unique_nonempty_texts(self, values: List[str]) -> List[str]:
        result: List[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            key = self._normalize_name(text)
            if text and key and key not in seen:
                seen.add(key)
                result.append(text)
        return result

    def _search_initial_candidate_pool(
        self,
        kind: str,
        target: int,
        keywords: List[str],
        search_one: Any,
        query_log_name: str,
        count_log_name: str,
    ) -> str:
        log_event(query_log_name, {"target": target, "keywords": keywords})
        if not keywords:
            return json.dumps({"pois": []}, ensure_ascii=False)

        raw_count = 0
        errors: List[Dict[str, str]] = []
        objects_by_keyword: Dict[str, List[Dict[str, Any]]] = {}
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        max_workers = min(INITIAL_SEARCH_WORKERS, len(keywords))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(search_one, keyword): keyword for keyword in keywords}
            for future in as_completed(futures):
                keyword = futures[future]
                try:
                    text = future.result()
                    objects = self._extract_candidate_objects(text)
                    raw_count += len(objects)
                    objects_by_keyword[keyword] = objects
                except Exception as exc:
                    errors.append({"keyword": keyword, "error": repr(exc)})

        for keyword in keywords:
            for item in objects_by_keyword.get(keyword, []):
                candidate = self._normalize_candidate_record(item, kind, len(merged))
                if not candidate:
                    continue
                key = self._candidate_dedupe_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(candidate)

        returned = merged[:target]
        log_event(
            count_log_name,
            {
                "target": target,
                "keyword_count": len(keywords),
                "raw_count": raw_count,
                "deduped_count": len(merged),
                "returned_count": len(returned),
                "errors": errors[:3],
            },
        )
        return json.dumps({"pois": returned}, ensure_ascii=False)

    def _candidate_dedupe_key(self, candidate: Dict[str, Any]) -> str:
        poi_id = str(candidate.get("poi_id") or "").strip()
        if poi_id:
            return f"id:{poi_id}"
        name = self._normalize_name(str(candidate.get("name") or ""))
        address = self._normalize_name(str(candidate.get("address") or ""))
        location = candidate.get("location") or {}
        try:
            lng = round(float(location.get("longitude")), 5)
            lat = round(float(location.get("latitude")), 5)
            loc_key = f"{lng},{lat}"
        except Exception:
            loc_key = ""
        return f"name:{name}|address:{address}|loc:{loc_key}"

    def _safe_run(self, domain: str, query: str) -> str:
        print(f"INFO Agent dispatch: {domain} | query={self._preview(query)}", flush=True)
        try:
            with timed_event("agent.search", {"domain": domain}):
                text = self.amap_pool.run(domain, query)
            self._print_full_output_block(f"agent_result:{domain}", text)
            return text
        except Exception as exc:
            text = f"地图工具检索失败，当前领域无法返回候选结果: {exc}"
            print(f"WARN Agent error: {domain} | {self._preview(text)}", flush=True)
            return text

    def _parse_structured_plan(self, instructions: str, payload: Dict[str, Any]) -> TripPlanDraft:
        payload_text = json.dumps(payload, ensure_ascii=False)
        schema_text = json.dumps(TripPlanDraft.model_json_schema(), ensure_ascii=False)
        base_url = getattr(self.openai_client, "base_url", None)
        base_url_text = str(base_url) if base_url else ""
        total_start = time.perf_counter()
        llm_attempts: List[Dict[str, Any]] = []
        validation_attempts: List[Dict[str, Any]] = []
        summary_status = "error"
        summary_error = ""
        print(
            f"INFO structured_parse start | model={self.openai_model} custom_base_url={bool(base_url_text and 'api.openai.com' not in base_url_text)} payload_bytes={len(payload_text.encode('utf-8'))}",
            flush=True,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"{instructions}\n\nHere is a valid example of the target JSON structure:\n{STRUCTURED_PLAN_FEW_SHOT}\n\n"
                    "Return only valid JSON. Do not use markdown code fences. "
                    "The JSON must satisfy this schema exactly: "
                    f"{schema_text}"
                ),
            },
            {"role": "user", "content": payload_text},
        ]
        try:
            llm_start = time.perf_counter()
            try:
                with timed_event("planner.structured_llm_request", {"model": self.openai_model, "payload_bytes": len(payload_text.encode("utf-8"))}):
                    response = self.openai_client.chat.completions.create(
                        model=self.openai_model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=self.settings.llm_max_output_tokens,
                        response_format={"type": "json_object"},
                    )
                llm_attempts.append(
                    {
                        "attempt": 1,
                        "elapsed_ms": round((time.perf_counter() - llm_start) * 1000, 2),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                llm_attempts.append(
                    {
                        "attempt": 1,
                        "elapsed_ms": round((time.perf_counter() - llm_start) * 1000, 2),
                        "status": "error",
                        "error": repr(exc),
                    }
                )
                raise
            content = response.choices[0].message.content or ""
            self._print_full_output_block("planner_structured_raw_content", content)
            validation_start = time.perf_counter()
            try:
                with timed_event("planner.structured_validate"):
                    data = self._extract_json(content)
                    data = self._unwrap_structured_plan_payload(data)
                    parsed = TripPlanDraft.model_validate(data)
                validation_attempts.append(
                    {
                        "attempt": 1,
                        "elapsed_ms": round((time.perf_counter() - validation_start) * 1000, 2),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                validation_attempts.append(
                    {
                        "attempt": 1,
                        "elapsed_ms": round((time.perf_counter() - validation_start) * 1000, 2),
                        "status": "error",
                        "error": repr(exc),
                    }
                )
                raise
            summary_status = "ok"
            print("INFO structured_parse success", flush=True)
            return parsed
        except Exception as exc:
            summary_error = repr(exc)
            print(
                f"ERROR structured_parse failed | model={self.openai_model} base_url={base_url_text} error={type(exc).__name__}: {exc} payload_preview={self._preview(payload_text, 500)}",
                flush=True,
            )
            raise
        finally:
            log_event(
                "planner_validation_summary",
                {
                    "stage": "structured_plan",
                    "status": summary_status,
                    "error": summary_error,
                    "total_elapsed_ms": round((time.perf_counter() - total_start) * 1000, 2),
                    "llm_retry_count": max(0, len(llm_attempts) - 1),
                    "validation_retry_count": max(0, len(validation_attempts) - 1),
                    "llm_attempts": llm_attempts,
                    "validation_attempts": validation_attempts,
                    "model": self.openai_model,
                    "payload_bytes": len(payload_text.encode("utf-8")),
                },
            )

    def _unwrap_structured_plan_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return data
        candidate_keys = ("data", "result", "output", "plan", "trip_plan")
        required_keys = {"city", "start_date", "end_date", "days", "overall_suggestions"}
        for key in candidate_keys:
            value = data.get(key)
            if isinstance(value, dict) and required_keys.issubset(value.keys()):
                print(f"INFO structured_parse unwrapped payload via key={key}", flush=True)
                return value
        city_value = data.get("city")
        if isinstance(city_value, dict) and required_keys.issubset(city_value.keys()):
            print("INFO structured_parse unwrapped payload via key=city", flush=True)
            return city_value
        return data

    def _parse_attractions_patch(self, payload: Dict[str, Any]) -> AttractionsPatchDraft:
        payload_text = json.dumps(payload, ensure_ascii=False)
        base_url = getattr(self.openai_client, "base_url", None)
        base_url_text = str(base_url) if base_url else ""
        total_start = time.perf_counter()
        llm_attempts: List[Dict[str, Any]] = []
        validation_attempts: List[Dict[str, Any]] = []
        summary_status = "error"
        summary_error = ""
        print(
            f"INFO attractions_patch start | model={self.openai_model} custom_base_url={bool(base_url_text and 'api.openai.com' not in base_url_text)} payload_bytes={len(payload_text.encode('utf-8'))}",
            flush=True,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are updating attractions only. Return only a JSON patch for days[].attractions and optional days[].description. "
                    "Do not change or return hotel, meals, weather_info, budget, city, start_date, end_date, or overall_suggestions. "
                    f"Here is a valid example:\n{ATTRACTIONS_PATCH_FEW_SHOT}\n\n"
                    "Return only valid JSON. Do not use markdown code fences."
                ),
            },
            {"role": "user", "content": payload_text},
        ]
        try:
            llm_start = time.perf_counter()
            try:
                with timed_event("planner.attractions_patch_llm_request", {"model": self.openai_model, "payload_bytes": len(payload_text.encode("utf-8"))}):
                    response = self.openai_client.chat.completions.create(
                        model=self.openai_model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=self.settings.llm_max_output_tokens,
                        response_format={"type": "json_object"},
                    )
                llm_attempts.append(
                    {
                        "attempt": 1,
                        "elapsed_ms": round((time.perf_counter() - llm_start) * 1000, 2),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                llm_attempts.append(
                    {
                        "attempt": 1,
                        "elapsed_ms": round((time.perf_counter() - llm_start) * 1000, 2),
                        "status": "error",
                        "error": repr(exc),
                    }
                )
                raise
            content = response.choices[0].message.content or ""
            self._print_full_output_block("attractions_patch_raw_content", content)
            validation_start = time.perf_counter()
            try:
                with timed_event("planner.attractions_patch_validate"):
                    data = self._extract_json(content)
                    if isinstance(data.get("days"), list):
                        patch_data = data
                    else:
                        patch_data = data.get("data") if isinstance(data.get("data"), dict) else data
                    parsed = AttractionsPatchDraft.model_validate(patch_data)
                validation_attempts.append(
                    {
                        "attempt": 1,
                        "elapsed_ms": round((time.perf_counter() - validation_start) * 1000, 2),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                validation_attempts.append(
                    {
                        "attempt": 1,
                        "elapsed_ms": round((time.perf_counter() - validation_start) * 1000, 2),
                        "status": "error",
                        "error": repr(exc),
                    }
                )
                raise
            summary_status = "ok"
            return parsed
        except Exception as exc:
            summary_error = repr(exc)
            raise
        finally:
            log_event(
                "planner_validation_summary",
                {
                    "stage": "attractions_patch",
                    "status": summary_status,
                    "error": summary_error,
                    "total_elapsed_ms": round((time.perf_counter() - total_start) * 1000, 2),
                    "llm_retry_count": max(0, len(llm_attempts) - 1),
                    "validation_retry_count": max(0, len(validation_attempts) - 1),
                    "llm_attempts": llm_attempts,
                    "validation_attempts": validation_attempts,
                    "model": self.openai_model,
                    "payload_bytes": len(payload_text.encode("utf-8")),
                },
            )

    def _merge_attractions_patch_with_current_plan(self, current_plan: TripPlan, patch: AttractionsPatchDraft) -> Dict[str, Any]:
        current_data = current_plan.model_dump()
        current_day_map: Dict[str, Dict[str, Any]] = {}
        for idx, day in enumerate(current_data.get("days", [])):
            if not isinstance(day, dict):
                continue
            key = str(day.get("date") or idx)
            current_day_map[key] = deepcopy(day)

        updated_days: List[int] = []
        for idx, patch_day in enumerate(patch.days):
            key = str(patch_day.date or patch_day.day_index or idx)
            current_day = current_day_map.get(key)
            if not current_day:
                continue
            if patch_day.description:
                current_day["description"] = patch_day.description
            if patch_day.attractions:
                current_day["attractions"] = [item.model_dump(exclude_none=True) for item in patch_day.attractions]
            current_day_map[key] = current_day
            updated_days.append(current_day.get("day_index", idx))

        merged_days = []
        for idx, day in enumerate(current_data.get("days", [])):
            key = str(day.get("date") or idx)
            merged_days.append(current_day_map.get(key, day))
        merged = deepcopy(current_data)
        merged["days"] = merged_days
        log_event("attractions_patch_merged", f"updated_days={updated_days}")
        return merged

    def _draft_to_trip_plan(self, draft: TripPlanDraft, form_snapshot: Dict[str, Any], candidate_context: Dict[str, Any]) -> TripPlan:
        data = {
            "city": draft.city,
            "start_date": draft.start_date,
            "end_date": draft.end_date,
            "overall_suggestions": draft.overall_suggestions,
            "days": [day.model_dump() for day in draft.days],
            "weather_info": [item.model_dump() for item in draft.weather_info],
            "budget": draft.budget.model_dump() if draft.budget else {},
        }
        self._print_full_output_block("planner_raw_plan", json.dumps(data, ensure_ascii=False))
        return self._coerce_json_to_trip_plan(data, form_snapshot, candidate_context)

    def _ensure_initial_plan_density(
        self,
        plan: TripPlan,
        form_snapshot: Dict[str, Any],
        candidate_context: Dict[str, Any],
        planning_constraints: Dict[str, Any],
    ) -> TripPlan:
        min_attractions = int(planning_constraints.get("min_attractions_per_day", 2) or 2)
        attraction_candidates = list(candidate_context.get("attraction_candidates", []) or [])
        restaurant_candidates = list(candidate_context.get("restaurant_candidates", []) or [])

        before_counts = [len(day.attractions or []) for day in plan.days]
        used_attractions = {self._place_key_from_attraction(item) for day in plan.days for item in (day.attractions or [])}
        unused_attractions = [
            candidate for candidate in attraction_candidates if self._candidate_dedupe_key(candidate) not in used_attractions
        ]
        attraction_autofill: List[Dict[str, Any]] = []
        far_route_warnings: List[Dict[str, Any]] = []
        hard_max_distance = float(
            (planning_constraints.get("geo_constraints") or {}).get(
                "hard_max_same_day_attraction_distance_m",
                SAME_DAY_HARD_MAX_DISTANCE_M,
            )
        )
        geo_repair = self._repair_selected_attractions_by_geo(plan, planning_constraints)
        far_route_warnings.extend(geo_repair.get("warnings", []))
        for day in plan.days:
            while len(day.attractions or []) < min_attractions and unused_attractions:
                selection = self._select_nearest_attraction_candidate(day, unused_attractions, hard_max_distance)
                if selection is None:
                    far_route_warnings.append(
                        {
                            "day_index": day.day_index,
                            "reason": "candidate_too_far_or_missing_location",
                            "current_count": len(day.attractions or []),
                            "remaining_candidates": len(unused_attractions),
                            "hard_max_distance_m": hard_max_distance,
                        }
                    )
                    break
                candidate, selected_index, distance_m, anchor = selection
                unused_attractions.pop(selected_index)
                attraction = self._candidate_to_attraction(candidate)
                day.attractions.append(attraction)
                attraction_autofill.append(
                    {
                        "day_index": day.day_index,
                        "name": attraction.name,
                        "poi_id": attraction.poi_id or "",
                        "distance_m": round(distance_m, 2) if distance_m is not None else None,
                        "anchor": anchor,
                    }
                )
            self._sort_day_attractions_by_nearest_neighbor(day)

        after_counts = [len(day.attractions or []) for day in plan.days]
        log_event(
            "initial_attraction_density",
            {
                "min_per_day": min_attractions,
                "target_per_day": planning_constraints.get("target_attractions_per_day"),
                "max_per_day": planning_constraints.get("max_attractions_per_day"),
                "candidate_count": len(attraction_candidates),
                "before": before_counts,
                "after": after_counts,
                "underfilled_days": [
                    {"day_index": day.day_index, "count": len(day.attractions or [])}
                    for day in plan.days
                    if len(day.attractions or []) < min_attractions
                ],
            },
        )
        log_event("initial_attraction_autofill", attraction_autofill)
        geo_compactness = self._build_geo_compactness_log(plan, planning_constraints)
        log_event("initial_day_geo_compactness", geo_compactness)
        for item in geo_compactness:
            if item.get("exceeds_hard_limit"):
                far_route_warnings.append(
                    {
                        "day_index": item.get("day_index"),
                        "reason": "day_route_exceeds_hard_limit",
                        "max_adjacent_distance_m": item.get("max_adjacent_distance_m"),
                        "hard_max_distance_m": hard_max_distance,
                    }
                )
        log_event("initial_attraction_geo_autofill", attraction_autofill)
        log_event("initial_far_route_warning", far_route_warnings)

        required_meals = ["breakfast", "lunch", "dinner"]
        meal_before = [
            {"day_index": day.day_index, "types": sorted({meal.type for meal in (day.meals or [])})}
            for day in plan.days
        ]
        used_restaurants = {self._place_key_from_meal(meal) for day in plan.days for meal in (day.meals or [])}
        unused_restaurants = [
            candidate for candidate in restaurant_candidates if self._candidate_dedupe_key(candidate) not in used_restaurants
        ]
        meal_autofill: List[Dict[str, Any]] = []
        for day in plan.days:
            existing_types = {meal.type for meal in (day.meals or [])}
            for meal_type in required_meals:
                if meal_type in existing_types:
                    continue
                if not unused_restaurants:
                    continue
                candidate = unused_restaurants.pop(0)
                meal = self._candidate_to_meal(candidate, meal_type)
                day.meals.append(meal)
                existing_types.add(meal_type)
                meal_autofill.append({"day_index": day.day_index, "type": meal_type, "name": meal.name})

        meal_after = [
            {"day_index": day.day_index, "types": sorted({meal.type for meal in (day.meals or [])})}
            for day in plan.days
        ]
        log_event(
            "initial_meal_coverage",
            {
                "required_types": required_meals,
                "candidate_count": len(restaurant_candidates),
                "before": meal_before,
                "after": meal_after,
                "missing_after": [
                    {
                        "day_index": day.day_index,
                        "missing": [meal_type for meal_type in required_meals if meal_type not in {meal.type for meal in (day.meals or [])}],
                    }
                    for day in plan.days
                    if any(meal_type not in {meal.type for meal in (day.meals or [])} for meal_type in required_meals)
                ],
            },
        )
        log_event("initial_meal_autofill", meal_autofill)
        return plan

    def _candidate_to_attraction(self, candidate: Dict[str, Any]) -> Attraction:
        return Attraction(
            name=str(candidate.get("name") or "景点"),
            address=str(candidate.get("address") or "待补充地址"),
            location=self._candidate_location(candidate),
            visit_duration=120,
            description="根据候选池自动补充的景点",
            category=str(candidate.get("category") or candidate.get("type") or "景点"),
            rating=self._to_float_or_none(candidate.get("rating")),
            poi_id=str(candidate.get("poi_id") or ""),
            ticket_price=self._parse_money(candidate.get("ticket_price", candidate.get("estimated_cost", 0))),
        )

    def _candidate_to_meal(self, candidate: Dict[str, Any], meal_type: str) -> Meal:
        return Meal(
            type=meal_type,
            name=str(candidate.get("name") or f"{meal_type}推荐"),
            address=str(candidate.get("address") or "") or None,
            location=self._candidate_location(candidate),
            description="根据餐饮候选池自动补充",
            estimated_cost=self._parse_money(
                candidate.get("estimated_cost") or candidate.get("price_range") or MEAL_COST_DEFAULTS.get(meal_type, 60)
            ),
        )

    def _repair_selected_attractions_by_geo(
        self,
        plan: TripPlan,
        planning_constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Pure algorithmic repair: regroup selected attractions by haversine distance; never call an LLM here.
        before = self._build_geo_compactness_log(plan, planning_constraints)
        if not any(item.get("exceeds_hard_limit") for item in before):
            for day in plan.days:
                self._sort_day_attractions_by_nearest_neighbor(day)
            return {"changed": False, "warnings": []}

        geo_constraints = planning_constraints.get("geo_constraints") or {}
        hard_max = float(geo_constraints.get("hard_max_same_day_attraction_distance_m", SAME_DAY_HARD_MAX_DISTANCE_M))
        max_per_day = int(planning_constraints.get("max_attractions_per_day", 4) or 4)
        attraction_records: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for day in plan.days:
            for attraction in day.attractions or []:
                key = self._place_key_from_attraction(attraction)
                if key in seen:
                    continue
                seen.add(key)
                attraction_records.append(
                    {
                        "attraction": attraction,
                        "first_day_index": day.day_index,
                        "location": attraction.location,
                    }
                )

        clusters = self._cluster_attraction_records(attraction_records, hard_max)
        buckets: List[List[Attraction]] = [[] for _ in plan.days]
        warnings: List[Dict[str, Any]] = []
        for cluster in clusters:
            target_index = self._choose_day_bucket_for_cluster(cluster, buckets, max_per_day)
            if target_index is None:
                target_index = min(range(len(buckets)), key=lambda idx: len(buckets[idx])) if buckets else 0
                warnings.append(
                    {
                        "reason": "geo_cluster_overflow",
                        "cluster_size": len(cluster),
                        "target_day_index": target_index,
                    }
                )
            buckets[target_index].extend(item["attraction"] for item in cluster)

        for idx, day in enumerate(plan.days):
            day.attractions = buckets[idx] if idx < len(buckets) else []
            self._sort_day_attractions_by_nearest_neighbor(day)

        after = self._build_geo_compactness_log(plan, planning_constraints)
        log_event(
            "initial_geo_repair",
            {
                "source": "algorithm",
                "llm_used": False,
                "changed": True,
                "hard_max_distance_m": hard_max,
                "before": before,
                "after": after,
                "cluster_count": len(clusters),
            },
        )
        for item in after:
            if item.get("exceeds_hard_limit"):
                warnings.append(
                    {
                        "day_index": item.get("day_index"),
                        "reason": "geo_repair_still_exceeds_hard_limit",
                        "max_adjacent_distance_m": item.get("max_adjacent_distance_m"),
                        "hard_max_distance_m": hard_max,
                    }
                )
        return {"changed": True, "warnings": warnings}

    def _cluster_attraction_records(
        self,
        records: List[Dict[str, Any]],
        max_distance_m: float,
    ) -> List[List[Dict[str, Any]]]:
        # Greedy distance clustering for already-selected attractions. This is intentionally model-free.
        clusters: List[List[Dict[str, Any]]] = []
        for record in records:
            location = record.get("location")
            if not location or self._is_zero_location(location):
                clusters.append([record])
                continue
            best_index = -1
            best_distance = float("inf")
            for idx, cluster in enumerate(clusters):
                located_members = [
                    member for member in cluster if member.get("location") and not self._is_zero_location(member["location"])
                ]
                if not located_members:
                    continue
                distance = min(self._distance_between_locations(location, member["location"]) for member in located_members)
                if distance < best_distance:
                    best_index = idx
                    best_distance = distance
            if best_index >= 0 and best_distance <= max_distance_m:
                clusters[best_index].append(record)
            else:
                clusters.append([record])
        clusters.sort(key=lambda cluster: min(item.get("first_day_index", 999) for item in cluster))
        return clusters

    def _choose_day_bucket_for_cluster(
        self,
        cluster: List[Dict[str, Any]],
        buckets: List[List[Attraction]],
        max_per_day: int,
    ) -> Optional[int]:
        preferred = min((item.get("first_day_index", 0) for item in cluster), default=0)
        ordered_indices = list(range(len(buckets)))
        ordered_indices.sort(key=lambda idx: (idx != preferred, len(buckets[idx])))
        for idx in ordered_indices:
            if len(buckets[idx]) + len(cluster) <= max_per_day:
                return idx
        for idx in ordered_indices:
            if not buckets[idx]:
                return idx
        return None

    def _candidate_location(self, candidate: Dict[str, Any]) -> Location:
        location = candidate.get("location") or {}
        try:
            return Location.model_validate(location)
        except Exception:
            return Location(longitude=0.0, latitude=0.0)

    def _select_nearest_attraction_candidate(
        self,
        day: DayPlan,
        candidates: List[Dict[str, Any]],
        hard_max_distance_m: float,
    ) -> Optional[tuple[Dict[str, Any], int, Optional[float], str]]:
        if not candidates:
            return None
        anchors: List[tuple[str, Location]] = [
            (f"attraction:{item.name}", item.location)
            for item in (day.attractions or [])
            if item.location and not self._is_zero_location(item.location)
        ]
        if not anchors and day.hotel and day.hotel.location and not self._is_zero_location(day.hotel.location):
            anchors.append((f"hotel:{day.hotel.name}", day.hotel.location))
        if not anchors:
            return candidates[0], 0, None, "candidate_order"

        best: Optional[tuple[Dict[str, Any], int, Optional[float], str]] = None
        for index, candidate in enumerate(candidates):
            candidate_location = self._candidate_location(candidate)
            if self._is_zero_location(candidate_location):
                continue
            distances = [
                (anchor_name, self._distance_between_locations(candidate_location, anchor_location))
                for anchor_name, anchor_location in anchors
            ]
            anchor_name, min_distance = min(distances, key=lambda item: item[1])
            if min_distance > hard_max_distance_m:
                continue
            if best is None or (min_distance is not None and min_distance < (best[2] or float("inf"))):
                best = (candidate, index, min_distance, anchor_name)
        return best

    def _sort_day_attractions_by_nearest_neighbor(self, day: DayPlan) -> None:
        attractions = list(day.attractions or [])
        if len(attractions) <= 2:
            return

        ordered: List[Attraction] = []
        remaining = attractions[:]
        if day.hotel and day.hotel.location and not self._is_zero_location(day.hotel.location):
            anchor = day.hotel.location
            start_index = min(
                range(len(remaining)),
                key=lambda idx: self._distance_between_locations(anchor, remaining[idx].location),
            )
        else:
            start_index = 0
        ordered.append(remaining.pop(start_index))

        while remaining:
            anchor = ordered[-1].location
            next_index = min(
                range(len(remaining)),
                key=lambda idx: self._distance_between_locations(anchor, remaining[idx].location),
            )
            ordered.append(remaining.pop(next_index))
        day.attractions = ordered

    def _build_geo_compactness_log(self, plan: TripPlan, planning_constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        geo_constraints = planning_constraints.get("geo_constraints") or {}
        ideal_max = float(geo_constraints.get("ideal_max_same_day_attraction_distance_m", SAME_DAY_IDEAL_MAX_DISTANCE_M))
        hard_max = float(geo_constraints.get("hard_max_same_day_attraction_distance_m", SAME_DAY_HARD_MAX_DISTANCE_M))
        result: List[Dict[str, Any]] = []
        for day in plan.days:
            distances = []
            attractions = list(day.attractions or [])
            for idx in range(len(attractions) - 1):
                distances.append(self._distance_between_locations(attractions[idx].location, attractions[idx + 1].location))
            max_adjacent = max(distances) if distances else 0.0
            total = sum(distances)
            result.append(
                {
                    "day_index": day.day_index,
                    "date": day.date,
                    "attraction_count": len(attractions),
                    "max_adjacent_distance_m": round(max_adjacent, 2),
                    "total_adjacent_distance_m": round(total, 2),
                    "exceeds_ideal_limit": max_adjacent > ideal_max,
                    "exceeds_hard_limit": max_adjacent > hard_max,
                    "ideal_max_distance_m": ideal_max,
                    "hard_max_distance_m": hard_max,
                }
            )
        return result

    def _distance_between_locations(self, loc1: Any, loc2: Any) -> float:
        try:
            lon1 = float(getattr(loc1, "longitude", loc1["longitude"]))
            lat1 = float(getattr(loc1, "latitude", loc1["latitude"]))
            lon2 = float(getattr(loc2, "longitude", loc2["longitude"]))
            lat2 = float(getattr(loc2, "latitude", loc2["latitude"]))
        except Exception:
            return float("inf")
        radius = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lam = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(d_lam / 2) ** 2)
        return radius * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

    def _is_zero_location(self, location: Any) -> bool:
        try:
            lon = float(getattr(location, "longitude", location["longitude"]))
            lat = float(getattr(location, "latitude", location["latitude"]))
        except Exception:
            return True
        return abs(lon) < 0.000001 and abs(lat) < 0.000001

    def _place_key_from_attraction(self, attraction: Attraction) -> str:
        candidate = {
            "poi_id": attraction.poi_id or "",
            "name": attraction.name,
            "address": attraction.address,
            "location": attraction.location.model_dump() if attraction.location else {},
        }
        return self._candidate_dedupe_key(candidate)

    def _place_key_from_meal(self, meal: Meal) -> str:
        candidate = {
            "poi_id": "",
            "name": meal.name,
            "address": meal.address or "",
            "location": meal.location.model_dump() if meal.location else {},
        }
        return self._candidate_dedupe_key(candidate)

    def _merge_draft_with_current_plan(self, current_plan: TripPlan, draft: TripPlanDraft) -> Dict[str, Any]:
        current_data = current_plan.model_dump()
        draft_data = {
            "city": draft.city,
            "start_date": draft.start_date,
            "end_date": draft.end_date,
            "overall_suggestions": draft.overall_suggestions,
            "days": [day.model_dump() for day in draft.days],
            "weather_info": [item.model_dump() for item in draft.weather_info],
            "budget": draft.budget.model_dump() if draft.budget else {},
        }

        current_day_map: Dict[str, Dict[str, Any]] = {}
        for idx, day in enumerate(current_data.get("days", [])):
            if not isinstance(day, dict):
                continue
            key = str(day.get("date") or idx)
            current_day_map[key] = day

        merged_days: List[Dict[str, Any]] = []
        for idx, day in enumerate(draft_data.get("days", [])):
            if not isinstance(day, dict):
                continue
            key = str(day.get("date") or idx)
            current_day = deepcopy(current_day_map.get(key, {}))
            merged_day = deepcopy(day)
            if not merged_day.get("hotel") and current_day.get("hotel"):
                merged_day["hotel"] = current_day["hotel"]
            if not merged_day.get("attractions") and current_day.get("attractions"):
                merged_day["attractions"] = current_day["attractions"]
            if not merged_day.get("meals") and current_day.get("meals"):
                merged_day["meals"] = current_day["meals"]
            merged_days.append(merged_day)

        if not merged_days and current_data.get("days"):
            merged_days = current_data["days"]

        merged = deepcopy(current_data)
        merged.update(draft_data)
        merged["days"] = merged_days
        if not merged.get("weather_info"):
            merged["weather_info"] = current_data.get("weather_info", [])
        if not merged.get("budget"):
            merged["budget"] = current_data.get("budget", {})
        return merged

    def _coerce_json_to_trip_plan(self, data: Dict[str, Any], form_snapshot: Dict[str, Any], candidate_context: Dict[str, Any]) -> TripPlan:
        total_start = time.perf_counter()
        normalize_elapsed = 0.0
        validate_attempts: List[Dict[str, Any]] = []
        status = "error"
        error = ""
        try:
            normalize_start = time.perf_counter()
            normalized = dict(data)
            normalized["city"] = str(normalized.get("city") or form_snapshot.get("city") or "")
            normalized["start_date"] = str(normalized.get("start_date") or form_snapshot.get("start_date") or "")
            normalized["end_date"] = str(normalized.get("end_date") or form_snapshot.get("end_date") or "")
            normalized["overall_suggestions"] = str(normalized.get("overall_suggestions") or "")
            normalized["days"] = [
                self._normalize_day(day, idx, form_snapshot, candidate_context)
                for idx, day in enumerate(normalized.get("days", []) or [])
                if isinstance(day, dict)
            ]
            normalized["weather_info"] = self._normalize_weather(normalized.get("weather_info", []), normalized["days"])
            normalized["budget"] = normalized.get("budget") or {}
            normalize_elapsed = round((time.perf_counter() - normalize_start) * 1000, 2)
            self._print_full_output_block("final_plan_after_coerce", json.dumps(normalized, ensure_ascii=False))

            validate_start = time.perf_counter()
            try:
                with timed_event("planner.trip_plan_model_validate", {"days": len(normalized.get("days", []))}):
                    plan = TripPlan.model_validate(normalized)
                validate_attempts.append(
                    {
                        "attempt": 1,
                        "elapsed_ms": round((time.perf_counter() - validate_start) * 1000, 2),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                validate_attempts.append(
                    {
                        "attempt": 1,
                        "elapsed_ms": round((time.perf_counter() - validate_start) * 1000, 2),
                        "status": "error",
                        "error": repr(exc),
                    }
                )
                raise
            status = "ok"
            return plan
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            log_event(
                "planner_validation_summary",
                {
                    "stage": "trip_plan_coerce",
                    "status": status,
                    "error": error,
                    "total_elapsed_ms": round((time.perf_counter() - total_start) * 1000, 2),
                    "normalization_elapsed_ms": normalize_elapsed,
                    "validation_retry_count": max(0, len(validate_attempts) - 1),
                    "validation_attempts": validate_attempts,
                    "city": form_snapshot.get("city", ""),
                },
            )

    def _normalize_day(self, day: Dict[str, Any], idx: int, form_snapshot: Dict[str, Any], candidate_context: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(day)
        normalized["date"] = str(normalized.get("date") or self._date_from_snapshot(form_snapshot, idx))
        normalized["day_index"] = idx
        normalized["description"] = str(normalized.get("description") or f"第{idx + 1}天行程")
        normalized["transportation"] = str(normalized.get("transportation") or form_snapshot.get("transportation") or "公共交通")
        normalized["accommodation"] = str(normalized.get("accommodation") or form_snapshot.get("accommodation") or "舒适型酒店")

        hotel_item = normalized.get("hotel") or {}
        if hotel_item:
            normalized["hotel"] = self._normalize_hotel(hotel_item, form_snapshot, candidate_context.get("hotel_candidates", []))
            normalized["accommodation"] = normalized["hotel"].get("type") or normalized["accommodation"]

        normalized["attractions"] = [
            self._normalize_attraction(item, idx, sub_idx, candidate_context.get("attraction_candidates", []))
            for sub_idx, item in enumerate(normalized.get("attractions", []) or [])
            if isinstance(item, dict)
        ]
        normalized["meals"] = [
            self._normalize_meal(str(item.get("type", "snack")), item, candidate_context.get("restaurant_candidates", []))
            for item in normalized.get("meals", []) or []
            if isinstance(item, dict)
        ]
        normalized["transport_segments"] = normalized.get("transport_segments", []) or []
        normalized["day_budget"] = normalized.get("day_budget", {}) or {}
        return normalized

    def _normalize_attraction(self, item: Dict[str, Any], day_idx: int, attr_idx: int, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        candidate = self._resolve_candidate(item, candidates, attr_idx)
        longitude = 116.397128 + day_idx * 0.01 + attr_idx * 0.002
        latitude = 39.916527 + day_idx * 0.01 + attr_idx * 0.002
        return {
            "name": str((candidate or {}).get("name") or f"景点{attr_idx + 1}"),
            "address": str((candidate or {}).get("address") or "待补充地址"),
            "location": (candidate or {}).get("location") or {"longitude": longitude, "latitude": latitude},
            "visit_duration": self._parse_duration(item.get("visit_duration", 120)),
            "description": str(item.get("description") or ""),
            "category": str((candidate or {}).get("category") or "景点"),
            "rating": self._to_float_or_none((candidate or {}).get("rating")),
            "ticket_price": self._parse_money(item.get("estimated_cost", (candidate or {}).get("ticket_price", 0))),
            "poi_id": str((candidate or {}).get("poi_id") or ""),
        }

    def _normalize_meal(self, meal_type: str, item: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        candidate = self._resolve_candidate(item, candidates, 0)
        log_event(
            "meal_candidate_resolution",
            {
                "meal_type": meal_type,
                "source_candidate_id": item.get("source_candidate_id", ""),
                "name": item.get("name", ""),
                "description": item.get("description", ""),
                "matched_candidate": (candidate or {}).get("candidate_id", ""),
            },
        )
        return {
            "type": meal_type or "snack",
            "name": str((candidate or {}).get("name") or item.get("name") or f"{meal_type}推荐"),
            "address": str((candidate or {}).get("address") or item.get("address") or "") or None,
            "location": (candidate or {}).get("location") or item.get("location"),
            "description": str(item.get("description") or ""),
            "estimated_cost": self._parse_money(item.get("estimated_cost", (candidate or {}).get("estimated_cost", 0))),
        }

    def _normalize_hotel(self, item: Dict[str, Any], form_snapshot: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        candidate = self._resolve_candidate(item, candidates, 0)
        price_range = str((candidate or {}).get("price_range") or item.get("price_range") or "")
        amap_cost = self._parse_money(price_range)
        item_cost = self._parse_money(item.get("estimated_cost", 0))
        candidate_cost = self._parse_money((candidate or {}).get("estimated_cost", 0))
        if amap_cost > 0:
            estimated_cost = amap_cost
            price_source = "amap_cost"
        elif item_cost > 0:
            estimated_cost = item_cost
            price_source = "llm_estimate"
        elif candidate_cost > 0:
            estimated_cost = candidate_cost
            price_source = str((candidate or {}).get("price_source") or "estimate")
        else:
            estimated_cost = 0
            price_source = str(item.get("price_source") or (candidate or {}).get("price_source") or "estimate")
        return {
            "name": str((candidate or {}).get("name") or item.get("name") or "推荐酒店"),
            "address": str((candidate or {}).get("address") or item.get("address") or ""),
            "location": (candidate or {}).get("location") or item.get("location"),
            "price_range": price_range,
            "rating": str((candidate or {}).get("rating") or item.get("rating") or ""),
            "distance": str((candidate or {}).get("distance") or item.get("distance") or ""),
            "type": str((candidate or {}).get("type") or item.get("type") or form_snapshot.get("accommodation") or "舒适型酒店"),
            "estimated_cost": estimated_cost,
            "price_source": price_source,
        }

    def _repair_plan_json(self, raw_response: str, form_snapshot: Dict[str, Any], error: Exception, candidate_context: Dict[str, Any], max_retries: int = 2) -> Optional[TripPlan]:
        schema = TripPlanDraft.model_json_schema()
        current_payload = self._preview(raw_response, 4000)
        total_start = time.perf_counter()
        repair_attempts: List[Dict[str, Any]] = []
        final_error = repr(error)
        for attempt in range(1, max_retries + 1):
            attempt_start = time.perf_counter()
            try:
                repaired = self._parse_structured_plan(
                    "你是 TripPlan JSON 修复专家。你必须修复成严格符合 schema 的 JSON，且只能使用 candidates 中的地点。",
                    {
                        "attempt": attempt,
                        "validation_error": str(error),
                        "form_snapshot": form_snapshot,
                        "tripplan_schema": schema,
                        "candidates": candidate_context,
                        "raw_json_or_text": current_payload,
                    },
                )
                plan = self._draft_to_trip_plan(repaired, form_snapshot, candidate_context)
                repair_attempts.append(
                    {
                        "attempt": attempt,
                        "elapsed_ms": round((time.perf_counter() - attempt_start) * 1000, 2),
                        "status": "ok",
                    }
                )
                log_event(
                    "planner_repair_summary",
                    {
                        "stage": "trip_plan_repair",
                        "status": "ok",
                        "total_elapsed_ms": round((time.perf_counter() - total_start) * 1000, 2),
                        "max_retries": max_retries,
                        "attempt_count": len(repair_attempts),
                        "retry_count": len(repair_attempts),
                        "attempts": repair_attempts,
                        "error": "",
                    },
                )
                print(f"✅ TripPlan修复成功: attempt={attempt}", flush=True)
                return plan
            except Exception as repair_error:
                repair_attempts.append(
                    {
                        "attempt": attempt,
                        "elapsed_ms": round((time.perf_counter() - attempt_start) * 1000, 2),
                        "status": "error",
                        "error": repr(repair_error),
                    }
                )
                print(f"⚠️ TripPlan修复失败: attempt={attempt} error={repair_error}", flush=True)
                error = repair_error
                final_error = repr(repair_error)
        log_event(
            "planner_repair_summary",
            {
                "stage": "trip_plan_repair",
                "status": "failed",
                "total_elapsed_ms": round((time.perf_counter() - total_start) * 1000, 2),
                "max_retries": max_retries,
                "attempt_count": len(repair_attempts),
                "retry_count": len(repair_attempts),
                "attempts": repair_attempts,
                "error": final_error,
            },
        )
        return None

    def _normalize_weather(self, weather_info: Any, days: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for idx, item in enumerate(weather_info if isinstance(weather_info, list) else []):
            if not isinstance(item, dict):
                continue
            result.append(
                {
                    "date": str(item.get("date") or (days[idx]["date"] if idx < len(days) else "")),
                    "day_weather": str(item.get("day_weather") or "多云"),
                    "night_weather": str(item.get("night_weather") or "晴"),
                    "day_temp": self._parse_money(item.get("day_temp", 25)),
                    "night_temp": self._parse_money(item.get("night_temp", 18)),
                    "wind_direction": str(item.get("wind_direction") or "东风"),
                    "wind_power": str(item.get("wind_power") or "1-3级"),
                }
            )
        return result

    def _extract_candidates_from_text(self, text: str, kind: str) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in self._extract_candidate_objects(text):
            candidate = self._normalize_candidate_record(item, kind, len(candidates))
            if not candidate:
                continue
            key = self._normalize_name(candidate["name"])
            if key and key not in seen:
                seen.add(key)
                candidates.append(candidate)
        return candidates

    def _extract_candidate_objects(self, text: str) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []
        try:
            parsed = self._extract_json(text)
            self._collect_candidate_dicts(parsed, objects)
        except Exception:
            pass
        return objects

    def _collect_candidate_dicts(self, value: Any, out: List[Dict[str, Any]]) -> None:
        if isinstance(value, dict):
            keys = {str(k).lower() for k in value.keys()}
            if {"name", "address", "location", "type", "id"} & keys:
                out.append(value)
            for sub in value.values():
                self._collect_candidate_dicts(sub, out)
        elif isinstance(value, list):
            for item in value:
                self._collect_candidate_dicts(item, out)

    def _normalize_candidate_record(self, item: Dict[str, Any], kind: str, index: int) -> Optional[Dict[str, Any]]:
        name = str(item.get("name") or item.get("名称") or "").strip()
        if not name:
            return None
        price_range = str(item.get("price_range") or "")
        price_from_amap = self._parse_money(price_range)
        estimated_cost = self._parse_money(item.get("estimated_cost", 0))
        if kind == "hotel" and price_from_amap > 0:
            estimated_cost = price_from_amap
        return {
            "candidate_id": f"{kind}_{index}_{self._normalize_name(name)[:24]}",
            "name": name,
            "address": str(item.get("address") or item.get("地址") or ""),
            "location": self._coerce_location(item.get("location") or item.get("坐标") or item.get("经纬度")),
            "category": str(item.get("category") or item.get("type") or kind),
            "type": str(item.get("type") or kind),
            "rating": self._to_float_or_none(item.get("rating")),
            "ticket_price": self._parse_money(item.get("ticket_price", 0)),
            "price_range": price_range,
            "estimated_cost": estimated_cost,
            "price_source": "amap_cost" if kind == "hotel" and price_from_amap > 0 else str(item.get("price_source") or ""),
            "poi_id": str(item.get("id") or item.get("poi_id") or ""),
        }

    def _resolve_candidate(self, item: Dict[str, Any], candidates: List[Dict[str, Any]], fallback_index: int) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        candidate_id = str(item.get("source_candidate_id") or item.get("candidate_id") or "").strip()
        if candidate_id:
            for candidate in candidates:
                if candidate["candidate_id"] == candidate_id:
                    return deepcopy(candidate)
        target_name = self._normalize_name(str(item.get("name") or ""))
        if target_name:
            for candidate in candidates:
                if self._normalize_name(candidate["name"]) == target_name:
                    return deepcopy(candidate)
        description_name = self._normalize_name(str(item.get("description") or ""))
        if description_name:
            for candidate in candidates:
                normalized_candidate_name = self._normalize_name(candidate["name"])
                if normalized_candidate_name and (
                    normalized_candidate_name in description_name or description_name in normalized_candidate_name
                ):
                    return deepcopy(candidate)
        if 0 <= fallback_index < len(candidates):
            return deepcopy(candidates[fallback_index])
        return deepcopy(candidates[0])

    def _coerce_location(self, location: Any) -> Optional[Dict[str, float]]:
        if isinstance(location, dict):
            try:
                return {"longitude": float(location.get("longitude")), "latitude": float(location.get("latitude"))}
            except Exception:
                return None
        if isinstance(location, str):
            parts = re.split(r"[,，\s]+", location.strip())
            if len(parts) >= 2:
                try:
                    return {"longitude": float(parts[0]), "latitude": float(parts[1])}
                except Exception:
                    return None
        return None

    def _normalize_name(self, value: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())

    def _parse_money(self, value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            nums = re.findall(r"\d+(?:\.\d+)?", value)
            if nums:
                return int(float(nums[0]))
        return 0

    def _parse_duration(self, value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            nums = re.findall(r"\d+", value)
            if nums:
                return int(nums[0])
        return 120

    def _to_float_or_none(self, value: Any) -> Optional[float]:
        try:
            if value in (None, ""):
                return None
            return float(value)
        except Exception:
            return None

    def _date_from_snapshot(self, form_snapshot: Dict[str, Any], idx: int) -> str:
        start_date_raw = str(form_snapshot.get("start_date", ""))
        try:
            start = datetime.strptime(start_date_raw, "%Y-%m-%d")
            return (start + timedelta(days=idx)).strftime("%Y-%m-%d")
        except Exception:
            return start_date_raw or datetime.now().strftime("%Y-%m-%d")

    def _extract_json(self, text: str) -> Dict[str, Any]:
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
        return json.loads(text)

    def close(self) -> None:
        self.amap_pool.close()

    def _preview(self, value: Any, limit: int = 50) -> str:
        text = "" if value is None else str(value)
        text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
        return text[:limit]

    def _print_full_output_block(self, title: str, content: Any) -> None:
        log_full_output(title, content)


_multi_agent_planner: Optional[MultiAgentTripPlanner] = None


def get_trip_planner_agent() -> MultiAgentTripPlanner:
    global _multi_agent_planner
    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()
    return _multi_agent_planner


def close_trip_planner_agent() -> None:
    global _multi_agent_planner
    if _multi_agent_planner is not None:
        _multi_agent_planner.close()
        _multi_agent_planner = None
