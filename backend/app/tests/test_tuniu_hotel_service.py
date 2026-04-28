# -*- coding: utf-8 -*-
import json
from types import SimpleNamespace

from app.services.tuniu_hotel_service import TuniuHotelService


class _FailingClient:
    def close(self):
        pass

    def post(self, *args, **kwargs):
        raise AssertionError("HTTP request should not be sent")


def test_tuniu_disallows_order_tool_before_http_request():
    service = TuniuHotelService()
    service.client = _FailingClient()
    try:
        try:
            service._call_tool("tuniu_hotel_create_order", {})
        except ValueError as exc:
            assert "not allowed" in str(exc)
        else:
            raise AssertionError("Expected ValueError for disallowed tool")
    finally:
        service.close()


def test_tuniu_normalization_drops_booking_fields():
    service = TuniuHotelService()
    try:
        normalized = service._normalize_hotel(
            {
                "hotelId": 123,
                "hotelName": "测试酒店",
                "address": "测试地址",
                "longitude": 117.2,
                "latitude": 39.1,
                "lowestPrice": 299,
                "preBookParam": {"mobile": "should-not-leak"},
            },
            {
                "roomTypes": [{"ratePlans": [{"price": 288, "payment": {"url": "secret"}}]}],
                "order": {"id": "secret"},
            },
            "天津",
        )
        assert normalized["estimated_cost"] == 288
        assert normalized["price_source"] == "tuniu_detail_price"
        dumped = str(normalized)
        assert "preBookParam" not in dumped
        assert "payment" not in dumped
        assert "order" not in dumped
        assert "mobile" not in dumped
    finally:
        service.close()


def test_tuniu_keyword_uses_single_broad_term():
    service = TuniuHotelService()
    try:
        assert service._build_keyword("经济型酒店") == "经济型酒店"
        assert service._build_keyword("经济型酒店", "快捷酒店") == "快捷酒店"
        assert "快捷酒店" not in service._build_keyword("经济型酒店")
    finally:
        service.close()


def test_tuniu_search_uses_configured_limit_by_default(monkeypatch):
    service = TuniuHotelService()
    calls = []
    hotels = [
        {
            "hotelId": index,
            "hotelName": f"天津市中心酒店{index}",
            "address": "和平区测试路1号",
            "longitude": 117.2 + index * 0.001,
            "latitude": 39.1,
            "lowestPrice": 200 + index,
        }
        for index in range(30)
    ]

    def fake_call_tool(tool_name, arguments):
        calls.append((tool_name, arguments))
        return {"hotels": hotels}

    monkeypatch.setattr(service, "settings", SimpleNamespace(tuniu_api_key="key", tuniu_hotel_limit=20))
    monkeypatch.setattr(service, "_call_tool", fake_call_tool)
    monkeypatch.setattr(service, "_fetch_details", lambda hotels, check_in, check_out: {})

    try:
        result = json.loads(service.search_hotels("天津", "经济型酒店", "2026-05-09", "2026-05-11"))
    finally:
        service.close()

    assert calls[0][1]["keyword"] == "经济型酒店"
    assert calls[0][1].get("keyword") != "经济型酒店 经济型 快捷酒店"
    assert len(result["pois"]) == 20
