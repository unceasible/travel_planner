# -*- coding: utf-8 -*-
import json
from types import SimpleNamespace

from app.agents import trip_planner_agent as planner_module
from app.agents.trip_planner_agent import MultiAgentTripPlanner
from app.agents.reflection_agent import ReflectionReview
from app.models.schemas import Attraction, DayPlan, Hotel, Location, Meal, TripPlan
from app.services.amap_tool_pool import _AmapWorker
from app.services.task_executor import TripTaskExecutor


def _planner() -> MultiAgentTripPlanner:
    planner = MultiAgentTripPlanner.__new__(MultiAgentTripPlanner)
    planner.settings = SimpleNamespace(
        hotel_min_candidates=6,
        hotel_ideal_distance_to_main_cluster_m=8000,
        hotel_hard_distance_to_main_cluster_m=15000,
        hotel_supplement_enabled=True,
        hotel_max_supplement_queries=4,
        tuniu_hotel_limit=20,
        tuniu_hotel_supplement_enabled=True,
        tuniu_hotel_max_supplement_queries=5,
    )
    return planner


def _hotel(candidate_id: str, name: str, longitude: float, latitude: float, price: int = 300):
    return {
        "candidate_id": candidate_id,
        "name": name,
        "address": f"{name} address",
        "location": {"longitude": longitude, "latitude": latitude},
        "type": "hotel",
        "estimated_cost": price,
        "price_range": str(price),
        "price_source": "tuniu_lowest_price",
        "candidate_score": 70,
        "hotel_score": 70,
    }


def _attraction_candidate(candidate_id: str, name: str, longitude: float, latitude: float):
    return {
        "candidate_id": candidate_id,
        "name": name,
        "address": f"{name} address",
        "location": {"longitude": longitude, "latitude": latitude},
        "type": "attraction",
        "candidate_score": 80,
    }


def _far_hotel_plan() -> TripPlan:
    return TripPlan(
        city="Tianjin",
        start_date="2026-05-01",
        end_date="2026-05-01",
        days=[
            DayPlan(
                date="2026-05-01",
                day_index=0,
                description="City route",
                transportation="public",
                accommodation="budget",
                hotel=Hotel(
                    name="Far Binhai Hotel",
                    address="Far hotel address",
                    location=Location(longitude=117.385526, latitude=38.865476),
                    estimated_cost=103,
                ),
                attractions=[
                    Attraction(
                        name="Downtown Museum",
                        address="Museum address",
                        location=Location(longitude=117.203601, latitude=39.110567),
                        visit_duration=120,
                        description="Downtown stop",
                        ticket_price=0,
                    )
                ],
                meals=[
                    Meal(type="breakfast", name="Breakfast", estimated_cost=25),
                    Meal(type="lunch", name="Lunch", estimated_cost=60),
                    Meal(type="dinner", name="Dinner", estimated_cost=80),
                ],
            )
        ],
        weather_info=[],
        overall_suggestions="",
    )


def _placeholder_hotel_plan() -> TripPlan:
    plan = _far_hotel_plan()
    plan.days[0].hotel = Hotel(
        name="推荐酒店",
        address="",
        location=None,
        type="经济型酒店",
        estimated_cost=164,
        price_source="llm_estimate",
    )
    return plan


class _FakeTuniuService:
    def search_hotels(self, **kwargs):
        return json.dumps(
            {
                "pois": [
                    {
                        "id": "far1",
                        "name": "Far Binhai Hotel",
                        "address": "Far hotel address",
                        "location": {"longitude": 117.385526, "latitude": 38.865476},
                        "type": "hotel",
                        "estimated_cost": 103,
                        "price_source": "tuniu_lowest_price",
                    },
                    {
                        "id": "far2",
                        "name": "Another Far Hotel",
                        "address": "Another far address",
                        "location": {"longitude": 117.386989, "latitude": 38.867267},
                        "type": "hotel",
                        "estimated_cost": 263,
                        "price_source": "tuniu_lowest_price",
                    },
                ]
            },
            ensure_ascii=False,
        )


class _MultiRoundTuniuService:
    def __init__(self):
        self.calls = []

    def search_hotels(self, **kwargs):
        self.calls.append(kwargs)
        keyword = kwargs.get("keyword", "")
        if keyword == "市中心酒店":
            pois = [
                {
                    "id": f"near_tuniu_{index}",
                    "name": f"City Center Tuniu Hotel {index}",
                    "address": "Downtown address",
                    "location": {"longitude": 117.205 + index * 0.001, "latitude": 39.115},
                    "type": "hotel",
                    "estimated_cost": 360 + index,
                    "price_source": "tuniu_lowest_price",
                    "source": "tuniu",
                }
                for index in range(5)
            ]
        else:
            pois = [
                {
                    "id": "far_tuniu",
                    "name": "Far Binhai Tuniu Hotel",
                    "address": "Far hotel address",
                    "location": {"longitude": 117.385526, "latitude": 38.865476},
                    "type": "hotel",
                    "estimated_cost": 164,
                    "price_source": "tuniu_lowest_price",
                    "source": "tuniu",
                }
            ]
        return json.dumps({"pois": pois}, ensure_ascii=False)


class _FakeReflectionAgent:
    def __init__(self):
        self.reviews = [
            ReflectionReview(
                score=4,
                status="needs_replan",
                issues=[{"severity": "critical", "type": "hotel_location", "message": "hotel too far"}],
                improvement_instructions="Move hotel downtown.",
                summary="Bad hotel location",
            ),
            ReflectionReview(
                score=4,
                status="needs_replan",
                issues=[{"severity": "critical", "type": "hotel_location", "message": "still too far"}],
                improvement_instructions="Move hotel downtown.",
                summary="Still bad hotel location",
            ),
        ]

    def review_plan(self, **kwargs):
        return self.reviews.pop(0)


def test_geo_helpers_accept_location_models():
    planner = _planner()
    left = Location(longitude=117.2, latitude=39.1)
    right = Location(longitude=117.21, latitude=39.11)

    assert planner._is_zero_location(left) is False
    distance = planner._distance_between_locations(left, right)
    assert 0 < distance < 2000


def test_search_hotels_supplements_when_tuniu_returns_too_few(monkeypatch):
    planner = _planner()
    safe_run_calls = []

    monkeypatch.setattr(planner_module, "get_tuniu_hotel_service", lambda: _FakeTuniuService())

    def fake_safe_run(domain, query):
        safe_run_calls.append((domain, query))
        return json.dumps(
            {
                "pois": [
                    {
                        "id": "near1",
                        "name": "City Center Hotel",
                        "address": "Downtown address",
                        "location": {"longitude": 117.205, "latitude": 39.115},
                        "type": "hotel",
                        "price_range": "360",
                    }
                ]
            },
            ensure_ascii=False,
        )

    planner._safe_run = fake_safe_run

    result = json.loads(planner.search_hotels("Tianjin", "budget", "2026-05-01", "2026-05-04"))

    assert any(item["name"] == "City Center Hotel" for item in result["pois"])
    assert safe_run_calls
    assert all(query.startswith("请搜索Tianjin的") for _, query in safe_run_calls)


def test_search_hotels_uses_tuniu_supplement_before_amap(monkeypatch):
    planner = _planner()
    tuniu = _MultiRoundTuniuService()
    monkeypatch.setattr(planner_module, "get_tuniu_hotel_service", lambda: tuniu)

    def fail_if_amap_called(domain, query):
        raise AssertionError("Amap hotel supplement should not run before Tuniu supplement is exhausted")

    planner._safe_run = fail_if_amap_called

    result = json.loads(planner.search_hotels("Tianjin", "经济型酒店", "2026-05-01", "2026-05-04"))

    assert any(item["name"].startswith("City Center Tuniu Hotel") for item in result["pois"])
    assert len(tuniu.calls) >= 2
    assert tuniu.calls[0].get("keyword", "") == ""
    assert any(call.get("keyword") == "市中心酒店" for call in tuniu.calls)
    assert all(call.get("limit") == 20 for call in tuniu.calls)


def test_amap_hotel_query_parser_keeps_city_separate_from_keyword(monkeypatch):
    worker = _AmapWorker.__new__(_AmapWorker)
    captured = {}

    def fake_search_poi(city, keyword, types, relax_filter=False):
        captured.update({"city": city, "keyword": keyword})
        return json.dumps({"pois": []}, ensure_ascii=False)

    monkeypatch.setattr(worker, "_search_poi", fake_search_poi)

    worker._search_hotels("请搜索天津的经济型酒店市中心酒店。返回名称、地址和坐标。")

    assert captured["city"] == "天津"
    assert captured["keyword"] == "经济型酒店市中心酒店"


def test_far_hotels_are_rejected_after_attraction_clusters():
    planner = _planner()
    candidate_context = {
        "attraction_candidates": [
            _attraction_candidate("attraction_0", "Downtown Museum", 117.203601, 39.110567),
            _attraction_candidate("attraction_1", "Downtown Park", 117.202302, 39.124795),
        ],
        "hotel_candidates": [
            _hotel("hotel_far", "Far Binhai Hotel", 117.385526, 38.865476, 103),
            _hotel("hotel_near", "City Center Hotel", 117.205, 39.115, 360),
        ],
        "restaurant_candidates": [],
    }
    constraints = {
        "geo_constraints": {
            "hard_max_same_day_attraction_distance_m": 15000,
            "hotel_ideal_distance_to_main_cluster_m": 8000,
            "hotel_hard_distance_to_main_cluster_m": 15000,
        }
    }

    planner._attach_attraction_geo_clusters(candidate_context, constraints)

    assert [item["name"] for item in candidate_context["hotel_candidates"]] == ["City Center Hotel"]
    assert [item["name"] for item in candidate_context["rejected_hotel_candidates"]] == ["Far Binhai Hotel"]
    assert candidate_context["hotel_candidates"][0]["hotel_geo_status"] == "near_main_cluster"


def test_coerce_plan_replaces_rejected_hotel_with_nearest_eligible_candidate():
    planner = _planner()
    candidate_context = {
        "hotel_candidates": [
            _hotel("hotel_far", "Far Binhai Hotel", 117.385526, 38.865476, 103),
            _hotel("hotel_near", "City Center Hotel", 117.205, 39.115, 360),
        ],
        "rejected_hotel_candidates": [
            _hotel("hotel_far", "Far Binhai Hotel", 117.385526, 38.865476, 103)
        ],
        "attraction_candidates": [
            _attraction_candidate("attraction_0", "Downtown Museum", 117.203601, 39.110567)
        ],
        "restaurant_candidates": [],
    }
    data = {
        "city": "Tianjin",
        "start_date": "2026-05-01",
        "end_date": "2026-05-01",
        "days": [
            {
                "date": "2026-05-01",
                "description": "Downtown day",
                "transportation": "public",
                "accommodation": "budget",
                "hotel": {"source_candidate_id": "hotel_far"},
                "attractions": [{"source_candidate_id": "attraction_0", "visit_duration": 120}],
                "meals": [
                    {"type": "breakfast", "name": "Breakfast", "estimated_cost": 25},
                    {"type": "lunch", "name": "Lunch", "estimated_cost": 60},
                    {"type": "dinner", "name": "Dinner", "estimated_cost": 80},
                ],
            }
        ],
        "weather_info": [],
        "overall_suggestions": "",
    }

    plan = planner._coerce_json_to_trip_plan(data, {"city": "Tianjin"}, candidate_context)

    assert plan.days[0].hotel is not None
    assert plan.days[0].hotel.name == "City Center Hotel"


def test_coerce_plan_clears_placeholder_hotel_when_no_eligible_candidate():
    planner = _planner()
    candidate_context = {
        "hotel_candidates": [],
        "rejected_hotel_candidates": [],
        "attraction_candidates": [
            _attraction_candidate("attraction_0", "Downtown Museum", 117.203601, 39.110567)
        ],
        "restaurant_candidates": [],
    }
    data = {
        "city": "Tianjin",
        "start_date": "2026-05-01",
        "end_date": "2026-05-01",
        "days": [
            {
                "date": "2026-05-01",
                "description": "Downtown day",
                "transportation": "public",
                "accommodation": "budget",
                "hotel": {"name": "推荐酒店", "type": "经济型酒店", "estimated_cost": 164},
                "attractions": [{"source_candidate_id": "attraction_0", "visit_duration": 120}],
                "meals": [
                    {"type": "breakfast", "name": "Breakfast", "estimated_cost": 25},
                    {"type": "lunch", "name": "Lunch", "estimated_cost": 60},
                    {"type": "dinner", "name": "Dinner", "estimated_cost": 80},
                ],
            }
        ],
        "weather_info": [],
        "overall_suggestions": "",
    }

    plan = planner._coerce_json_to_trip_plan(data, {"city": "Tianjin"}, candidate_context)

    assert plan.days[0].hotel is None


def test_quality_gate_removes_far_hotel_when_second_review_still_critical():
    executor = TripTaskExecutor.__new__(TripTaskExecutor)
    executor.planner = SimpleNamespace(create_fallback_plan=lambda form: _far_hotel_plan())
    executor.reflection_agent = _FakeReflectionAgent()
    executor.settings = SimpleNamespace(hotel_hard_distance_to_main_cluster_m=15000)
    executor._add_transport_segments_parallel = lambda plan, transportation: plan

    plan, entries = executor._run_quality_gate(
        _far_hotel_plan(),
        {"city": "Tianjin", "start_date": "2026-05-01", "travel_days": 1, "accommodation": "budget"},
        user_profile_summary="{}",
        conversation_history=[],
        user_message="",
        update_mode="initial",
        retry_once=lambda feedback: _far_hotel_plan(),
    )

    assert plan.days[0].hotel is None
    assert any("unsafe_hotel_removed" in entry.get("notes", []) for entry in entries)


def test_quality_gate_removes_placeholder_hotel_without_location():
    executor = TripTaskExecutor.__new__(TripTaskExecutor)
    executor.settings = SimpleNamespace(hotel_hard_distance_to_main_cluster_m=15000)

    quality_entry = {
        "issues": [
            {
                "severity": "critical",
                "type": "placeholder_accommodation",
                "message": "placeholder hotel has no name, address or coordinates",
            }
        ]
    }

    assert executor._quality_entry_has_critical_hotel_issue(quality_entry) is True

    plan, entry = executor._remove_unsafe_hotels_by_distance(
        _placeholder_hotel_plan(),
        {"accommodation": "经济型酒店"},
    )

    assert plan.days[0].hotel is None
    assert entry is not None
    assert "unsafe_hotel_removed" in entry.get("notes", [])
