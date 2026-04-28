# -*- coding: utf-8 -*-
from app.models.schemas import (
    Budget,
    IntercityTransportOption,
    IntercityTransportPlan,
    TripPlan,
    TripRequest,
)


def test_trip_request_intercity_defaults_are_backward_compatible():
    request = TripRequest(
        nickname="Alice",
        city="北京",
        start_date="2026-05-01",
        end_date="2026-05-03",
        travel_days=3,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
    )

    assert request.departure_city == ""
    assert request.intercity_transportation == "智能推荐"


def test_trip_plan_serializes_intercity_transport_and_budget():
    outbound = IntercityTransportOption(
        direction="outbound",
        mode="火车",
        provider="tuniu_train",
        departure_city="天津",
        arrival_city="北京",
        date="2026-05-01",
        departure_time="08:30",
        arrival_time="10:00",
        duration_minutes=90,
        estimated_cost=120,
        code="G101",
        data_source="tuniu_real_time",
    )
    inbound = outbound.model_copy(update={
        "direction": "return",
        "departure_city": "北京",
        "arrival_city": "天津",
        "date": "2026-05-03",
        "code": "G102",
    })
    plan = TripPlan(
        departure_city="天津",
        city="北京",
        start_date="2026-05-01",
        end_date="2026-05-03",
        days=[],
        weather_info=[],
        overall_suggestions="测试计划",
        budget=Budget(total_intercity_transportation=240, total=240),
        intercity_transport=IntercityTransportPlan(
            status="ok",
            preference="火车",
            outbound_candidates=[outbound],
            return_candidates=[inbound],
            selected_outbound=outbound,
            selected_return=inbound,
            schedule_constraints={
                "first_day_max_attractions": 2,
                "last_day_max_attractions": 1,
            },
        ),
    )

    dumped = plan.model_dump()
    assert dumped["departure_city"] == "天津"
    assert dumped["intercity_transport"]["selected_outbound"]["code"] == "G101"
    assert dumped["budget"]["total_intercity_transportation"] == 240
