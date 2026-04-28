# -*- coding: utf-8 -*-
from __future__ import annotations

from concurrent.futures import Future

from app.models.schemas import Attraction, DayPlan, Hotel, Location, Meal, TransportSegment, TripPlan
from app.services.task_executor import TripTaskExecutor


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


def _hotel() -> Hotel:
    return Hotel(
        name="中心酒店",
        address="市中心路1号",
        location=Location(longitude=117.2, latitude=39.12),
        type="经济型酒店",
        estimated_cost=300,
        price_source="tuniu_lowest_price",
    )


def _attraction(name: str, offset: float) -> Attraction:
    return Attraction(
        name=name,
        address=f"{name}地址",
        location=Location(longitude=117.21 + offset, latitude=39.12 + offset),
        visit_duration=90,
        description="测试景点",
        ticket_price=0,
    )


def _meals():
    return [
        Meal(type="breakfast", name="早餐", estimated_cost=20),
        Meal(type="lunch", name="午餐", estimated_cost=50),
        Meal(type="dinner", name="晚餐", estimated_cost=80),
    ]


def _plan_with_only_first_day_hotel() -> TripPlan:
    return TripPlan(
        city="天津",
        start_date="2026-05-01",
        end_date="2026-05-03",
        days=[
            DayPlan(
                date="2026-05-01",
                day_index=0,
                description="第一天",
                transportation="公共交通",
                accommodation="经济型酒店",
                hotel=_hotel(),
                attractions=[_attraction("景点A", 0.0), _attraction("景点B", 0.01)],
                meals=_meals(),
            ),
            DayPlan(
                date="2026-05-02",
                day_index=1,
                description="第二天",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[_attraction("景点C", 0.02), _attraction("景点D", 0.03)],
                meals=_meals(),
            ),
            DayPlan(
                date="2026-05-03",
                day_index=2,
                description="第三天",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[_attraction("景点E", 0.04), _attraction("景点F", 0.05)],
                meals=_meals(),
            ),
        ],
        weather_info=[],
        overall_suggestions="测试计划",
    )


def test_reflect_and_fix_reuses_selected_hotel_for_all_days():
    executor = TripTaskExecutor.__new__(TripTaskExecutor)

    plan, entry = executor._reflect_and_fix(
        _plan_with_only_first_day_hotel(),
        {
            "city": "天津",
            "start_date": "2026-05-01",
            "travel_days": 3,
            "accommodation": "经济型酒店",
        },
    )

    assert [day.hotel.name if day.hotel else None for day in plan.days] == ["中心酒店", "中心酒店", "中心酒店"]
    assert "day_1_hotel_inherited" in entry["notes"]
    assert "day_2_hotel_inherited" in entry["notes"]


def test_transport_routes_start_at_hotel_but_do_not_return_on_last_day():
    executor = TripTaskExecutor.__new__(TripTaskExecutor)
    executor.transport_executor = _ImmediateExecutor()

    def fake_segment(job):
        day_index, from_point, to_point, *_ = job
        return day_index, TransportSegment(
            from_name=from_point["name"],
            to_name=to_point["name"],
            mode="公共交通",
            distance=1000,
            duration=600,
            estimated_cost=2,
        )

    executor._estimate_transport_segment = fake_segment
    plan, _ = executor._reflect_and_fix(
        _plan_with_only_first_day_hotel(),
        {"city": "天津", "start_date": "2026-05-01", "travel_days": 3, "accommodation": "经济型酒店"},
    )

    result = executor._add_transport_segments_parallel(plan, "公共交通")

    assert [(item.from_name, item.to_name) for item in result.days[0].transport_segments] == [
        ("中心酒店", "景点A"),
        ("景点A", "景点B"),
        ("景点B", "中心酒店"),
    ]
    assert [(item.from_name, item.to_name) for item in result.days[-1].transport_segments] == [
        ("中心酒店", "景点E"),
        ("景点E", "景点F"),
    ]


def test_hotel_budget_counts_nights_not_route_start_on_last_day():
    executor = TripTaskExecutor.__new__(TripTaskExecutor)
    plan, _ = executor._reflect_and_fix(
        _plan_with_only_first_day_hotel(),
        {"city": "天津", "start_date": "2026-05-01", "travel_days": 3, "accommodation": "经济型酒店"},
    )

    assert [day.day_budget.hotel for day in plan.days] == [300, 300, 0]
    assert plan.budget.total_hotels == 600
