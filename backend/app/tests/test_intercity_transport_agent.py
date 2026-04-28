# -*- coding: utf-8 -*-
import pytest

from app.models.schemas import IntercityTransportOption
from app.services.intercity_transport_agent import (
    IntercityTransportAgent,
    TuniuIntercityTransportService,
)


class _FailingClient:
    def close(self):
        pass

    def post(self, *args, **kwargs):
        raise AssertionError("HTTP request should not be sent")


class _FakeTuniuTransportService:
    def search_flights(self, **kwargs):
        return [
            IntercityTransportOption(
                direction=kwargs["direction"],
                mode="飞机",
                provider="tuniu_flight",
                departure_city=kwargs["departure_city"],
                arrival_city=kwargs["arrival_city"],
                date=kwargs["date"],
                departure_time="09:00",
                arrival_time="11:00",
                duration_minutes=120,
                estimated_cost=900,
                code="CA1001",
                data_source="tuniu_real_time",
            )
        ]

    def search_trains(self, **kwargs):
        return [
            IntercityTransportOption(
                direction=kwargs["direction"],
                mode="火车",
                provider="tuniu_train",
                departure_city=kwargs["departure_city"],
                arrival_city=kwargs["arrival_city"],
                date=kwargs["date"],
                departure_time="08:30",
                arrival_time="10:10",
                duration_minutes=100,
                estimated_cost=350,
                code="G101",
                data_source="tuniu_real_time",
            )
        ]


class _FakeAmapService:
    def plan_route(self, origin_address, destination_address, origin_city=None, destination_city=None, route_type="driving", **kwargs):
        return {
            "distance": 120000,
            "duration": 7200,
            "cost": 80,
            "description": "高速优先，自驾约2小时",
        }


def test_tuniu_intercity_disallows_order_tools_before_http_request():
    service = TuniuIntercityTransportService()
    service.client = _FailingClient()
    try:
        with pytest.raises(ValueError, match="not allowed"):
            service._call_tool("saveOrder", {})
    finally:
        service.close()


def test_tuniu_intercity_normalization_drops_booking_fields():
    service = TuniuIntercityTransportService()
    try:
        normalized = service._normalize_flight(
            {
                "flightNo": "CA1001",
                "depCity": "天津",
                "arrCity": "北京",
                "depTime": "2026-05-01 09:00",
                "arrTime": "2026-05-01 11:00",
                "lowestPrice": 888,
                "bookParam": {"mobile": "should-not-leak"},
                "order": {"id": "secret"},
            },
            direction="outbound",
            departure_city="天津",
            arrival_city="北京",
            date="2026-05-01",
        )
        assert normalized.code == "CA1001"
        assert normalized.estimated_cost == 888
        dumped = str(normalized.model_dump())
        assert "bookParam" not in dumped
        assert "mobile" not in dumped
        assert "order" not in dumped
        assert "secret" not in dumped
    finally:
        service.close()


def test_intercity_agent_skips_when_departure_city_is_missing_or_same():
    agent = IntercityTransportAgent(
        tuniu_service=_FakeTuniuTransportService(),
        amap_service=_FakeAmapService(),
    )

    missing = agent.search(
        departure_city="",
        destination_city="北京",
        start_date="2026-05-01",
        end_date="2026-05-03",
        preference="智能推荐",
    )
    same = agent.search(
        departure_city="北京",
        destination_city="北京",
        start_date="2026-05-01",
        end_date="2026-05-03",
        preference="智能推荐",
    )

    assert missing.status == "skipped"
    assert same.status == "skipped"


def test_intercity_agent_smart_recommendation_and_schedule_constraints():
    agent = IntercityTransportAgent(
        tuniu_service=_FakeTuniuTransportService(),
        amap_service=_FakeAmapService(),
    )

    result = agent.search(
        departure_city="天津",
        destination_city="北京",
        start_date="2026-05-01",
        end_date="2026-05-03",
        preference="智能推荐",
    )

    assert result.status == "ok"
    assert result.selected_outbound is not None
    assert result.selected_outbound.mode == "火车"
    assert result.selected_return is not None
    assert result.schedule_constraints["first_day_max_attractions"] == 2
    assert result.schedule_constraints["last_day_max_attractions"] == 0


def test_intercity_agent_smart_recommendation_excludes_driving_when_not_allowed():
    class _CountingAmapService(_FakeAmapService):
        def __init__(self):
            self.calls = 0

        def plan_route(self, *args, **kwargs):
            self.calls += 1
            return super().plan_route(*args, **kwargs)

    amap = _CountingAmapService()
    agent = IntercityTransportAgent(
        tuniu_service=_FakeTuniuTransportService(),
        amap_service=amap,
    )

    result = agent.search(
        departure_city="天津",
        destination_city="北京",
        start_date="2026-05-01",
        end_date="2026-05-03",
        preference="智能推荐",
        allow_self_drive=False,
    )

    assert amap.calls == 0
    assert all(option.mode != "自驾" for option in result.outbound_candidates)
    assert all(option.mode != "自驾" for option in result.return_candidates)


def test_intercity_driving_route_failure_returns_rule_based_candidate():
    class _FailingAmapService:
        def plan_route(self, *args, **kwargs):
            return {}

    agent = IntercityTransportAgent(
        tuniu_service=_FakeTuniuTransportService(),
        amap_service=_FailingAmapService(),
    )

    option = agent._search_driving("outbound", "Beijing", "Shanghai", "2026-05-01")

    assert option.data_source == "rule_based"
    assert option.provider == "amap"
