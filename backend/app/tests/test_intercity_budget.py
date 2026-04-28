# -*- coding: utf-8 -*-
from app.models.schemas import (
    DayPlan,
    IntercityTransportOption,
    IntercityTransportPlan,
    TripPlan,
)
from app.services.task_executor import TripTaskExecutor


def _option(direction: str, cost: int) -> IntercityTransportOption:
    return IntercityTransportOption(
        direction=direction,
        mode="自驾",
        provider="amap",
        departure_city="天津" if direction == "outbound" else "北京",
        arrival_city="北京" if direction == "outbound" else "天津",
        date="2026-05-01" if direction == "outbound" else "2026-05-03",
        duration_minutes=120,
        estimated_cost=cost,
        data_source="amap_route",
    )


def test_aggregate_budget_includes_intercity_transportation():
    executor = TripTaskExecutor.__new__(TripTaskExecutor)
    plan = TripPlan(
        departure_city="天津",
        city="北京",
        start_date="2026-05-01",
        end_date="2026-05-03",
        days=[
            DayPlan(
                date="2026-05-01",
                day_index=0,
                description="首日",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[],
                meals=[],
            )
        ],
        weather_info=[],
        overall_suggestions="测试计划",
        intercity_transport=IntercityTransportPlan(
            status="ok",
            preference="自驾",
            selected_outbound=_option("outbound", 80),
            selected_return=_option("return", 90),
        ),
    )

    result = executor._aggregate_budget(plan, "经济型酒店")

    assert result.budget is not None
    assert result.budget.total_intercity_transportation == 170
    assert result.budget.total >= 170
