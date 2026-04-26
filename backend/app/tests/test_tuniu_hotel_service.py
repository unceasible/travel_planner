# -*- coding: utf-8 -*-
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
