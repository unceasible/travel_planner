# -*- coding: utf-8 -*-

from app.services.amap_service import AmapService


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeHttpClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def get(self, url, params):
        self.calls += 1
        if not self.payloads:
            raise AssertionError("unexpected extra route HTTP request")
        return _FakeResponse(self.payloads.pop(0))


def _location(lng, lat):
    return {"longitude": lng, "latitude": lat}


def _configure_fast_route_retry(monkeypatch, service, max_retries=2):
    monkeypatch.setattr(service.settings, "amap_route_min_interval_seconds", 0.0, raising=False)
    monkeypatch.setattr(service.settings, "amap_route_rate_limit_backoff_seconds", 0.0, raising=False)
    monkeypatch.setattr(service.settings, "amap_route_max_retries", max_retries, raising=False)


def test_amap_route_retries_once_after_qps_limit(monkeypatch):
    service = AmapService()
    _configure_fast_route_retry(monkeypatch, service, max_retries=1)
    service.http_client = _FakeHttpClient(
        [
            {"status": "0", "info": "CUQPS_HAS_EXCEEDED_THE_LIMIT", "infocode": "10021"},
            {"status": "1", "route": {"paths": [{"distance": "1000", "duration": "600"}]}},
        ]
    )

    result = service._plan_route_by_location(
        _location(116.397, 39.908),
        _location(116.407, 39.918),
        "Beijing",
        "driving",
    )

    assert service.http_client.calls == 2
    assert result["distance"] == 1000.0
    assert result["duration"] == 600


def test_amap_route_returns_empty_after_repeated_qps_limits(monkeypatch):
    service = AmapService()
    _configure_fast_route_retry(monkeypatch, service, max_retries=2)
    service.http_client = _FakeHttpClient(
        [
            {"status": "0", "info": "CUQPS_HAS_EXCEEDED_THE_LIMIT", "infocode": "10021"},
            {"status": "0", "info": "CUQPS_HAS_EXCEEDED_THE_LIMIT", "infocode": "10021"},
            {"status": "0", "info": "CUQPS_HAS_EXCEEDED_THE_LIMIT", "infocode": "10021"},
        ]
    )

    result = service._plan_route_by_location(
        _location(116.397, 39.908),
        _location(116.407, 39.918),
        "Beijing",
        "driving",
    )

    assert service.http_client.calls == 3
    assert result == {}


def test_plan_route_skips_mcp_fallback_when_disabled(monkeypatch):
    service = AmapService()
    monkeypatch.setattr(service.settings, "amap_route_mcp_fallback_enabled", False, raising=False)

    def _fail_mcp_tool():
        raise AssertionError("MCP fallback should not be called when disabled")

    monkeypatch.setattr(service, "_mcp_tool", _fail_mcp_tool)

    assert service.plan_route("Beijing", "Shanghai", "Beijing", "Shanghai", "driving") == {}
