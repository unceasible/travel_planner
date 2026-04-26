"""Tuniu hotel MCP client.

This client is intentionally read-only. It only allows hotel search and
detail lookup. Booking/order/payment tools are not exposed or callable.
"""
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from ..config import get_settings
from .agent_output_logger import log_event, timed_event

ALLOWED_TUNIU_TOOLS = {"tuniu_hotel_search", "tuniu_hotel_detail"}
TUNIU_DETAIL_PRICE_SOURCE = "tuniu_detail_price"
TUNIU_LOWEST_PRICE_SOURCE = "tuniu_lowest_price"


class TuniuHotelService:
    """Read-only Tuniu hotel search/detail service."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = httpx.Client(timeout=18.0)
        self.detail_executor = ThreadPoolExecutor(max_workers=4)

    def close(self) -> None:
        self.client.close()
        self.detail_executor.shutdown(wait=False, cancel_futures=True)

    def search_hotels(
        self,
        city: str,
        accommodation: str,
        start_date: str,
        end_date: str,
        keyword: str = "",
        limit: int = 5,
    ) -> str:
        if not self.settings.tuniu_api_key:
            raise RuntimeError("TUNIU_API_KEY is not configured")
        check_in, check_out = self._build_check_dates(start_date, end_date)
        search_keyword = self._build_keyword(accommodation, keyword)
        args: Dict[str, Any] = {
            "cityName": city,
            "checkIn": check_in,
            "checkOut": check_out,
        }
        if search_keyword:
            args["keyword"] = search_keyword

        log_event(
            "tuniu_hotel_search_request",
            {
                "city": city,
                "check_in": check_in,
                "check_out": check_out,
                "keyword": search_keyword,
            },
        )
        with timed_event("tuniu.hotel_search", {"city": city}):
            payload = self._call_tool("tuniu_hotel_search", args)
        hotels = self._extract_hotels(payload)[:limit]
        log_event(
            "tuniu_hotel_search_result",
            {
                "city": city,
                "count": len(hotels),
                "has_prices": sum(1 for item in hotels if self._money(self._first_value(item, ("lowestPrice", "lowest_price", "price"))) > 0),
            },
        )
        if not hotels:
            return json.dumps({"pois": []}, ensure_ascii=False)

        detail_by_id = self._fetch_details(hotels, check_in, check_out)
        normalized = [
            self._normalize_hotel(item, detail_by_id.get(self._hotel_key(item), {}), city)
            for item in hotels
        ]
        normalized = [item for item in normalized if item.get("name")]
        return json.dumps({"pois": normalized}, ensure_ascii=False)

    def get_hotel_detail(self, hotel_id: Any, check_in: str, check_out: str) -> Dict[str, Any]:
        if not hotel_id:
            return {}
        log_event(
            "tuniu_hotel_detail_request",
            {"hotel_id": str(hotel_id), "check_in": check_in, "check_out": check_out},
        )
        payload = self._call_tool(
            "tuniu_hotel_detail",
            {"hotelId": hotel_id, "checkIn": check_in, "checkOut": check_out},
        )
        price = self._extract_lowest_price(payload)
        log_event(
            "tuniu_hotel_detail_result",
            {"hotel_id": str(hotel_id), "has_price": price > 0, "price": price if price > 0 else 0},
        )
        return payload if isinstance(payload, dict) else {}

    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in ALLOWED_TUNIU_TOOLS:
            raise ValueError(f"Tuniu hotel tool is not allowed: {tool_name}")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "apiKey": self.settings.tuniu_api_key,
        }
        if self.settings.tuniu_member_key:
            headers["Authorization"] = f"Bearer {self.settings.tuniu_member_key}"
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        response = self.client.post(self.settings.tuniu_mcp_url, headers=headers, json=body)
        response.raise_for_status()
        return self._parse_mcp_response(response.text)

    def _parse_mcp_response(self, text: str) -> Dict[str, Any]:
        parsed = self._parse_json_or_sse(text)
        if not isinstance(parsed, dict):
            return {}
        if parsed.get("error"):
            raise RuntimeError(str(parsed.get("error")))
        result = parsed.get("result", parsed)
        content = result.get("content") if isinstance(result, dict) else None
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_text = item.get("text")
                if isinstance(item_text, str) and item_text.strip():
                    return self._parse_json_or_sse(item_text)
        if isinstance(result, dict):
            return result
        return {}

    def _parse_json_or_sse(self, text: str) -> Dict[str, Any]:
        stripped = str(text or "").strip()
        if not stripped:
            return {}
        if stripped.startswith("data:") or "\ndata:" in stripped:
            chunks = []
            for line in stripped.splitlines():
                if line.startswith("data:"):
                    data = line.split(":", 1)[1].strip()
                    if data and data != "[DONE]":
                        chunks.append(data)
            stripped = "\n".join(chunks).strip()
        try:
            parsed = json.loads(stripped)
        except Exception:
            match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if not match:
                return {}
            try:
                parsed = json.loads(match.group())
            except Exception:
                return {}
        if isinstance(parsed, str):
            return self._parse_json_or_sse(parsed)
        return parsed if isinstance(parsed, dict) else {}

    def _fetch_details(self, hotels: List[Dict[str, Any]], check_in: str, check_out: str) -> Dict[str, Dict[str, Any]]:
        futures = {}
        for item in hotels:
            hotel_id = self._first_value(item, ("hotelId", "hotel_id", "id"))
            if not hotel_id:
                continue
            futures[self.detail_executor.submit(self.get_hotel_detail, hotel_id, check_in, check_out)] = self._hotel_key(item)
        details: Dict[str, Dict[str, Any]] = {}
        for future in as_completed(futures):
            key = futures[future]
            try:
                details[key] = future.result()
            except Exception as exc:
                log_event("tuniu_hotel_error", {"stage": "detail", "hotel_key": key, "error": repr(exc)})
        return details

    def _extract_hotels(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        for key in ("hotels", "hotelList", "hotel_list", "list", "data"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = self._extract_hotels(value)
                if nested:
                    return nested
        for value in payload.values() if isinstance(payload, dict) else []:
            if isinstance(value, dict):
                nested = self._extract_hotels(value)
                if nested:
                    return nested
        return []

    def _normalize_hotel(self, item: Dict[str, Any], detail: Dict[str, Any], city: str) -> Dict[str, Any]:
        detail_price = self._extract_lowest_price(detail)
        search_price = self._money(self._first_value(item, ("lowestPrice", "lowest_price", "price", "minPrice")))
        price = detail_price or search_price
        price_source = TUNIU_DETAIL_PRICE_SOURCE if detail_price else TUNIU_LOWEST_PRICE_SOURCE if search_price else ""
        hotel_id = self._first_value(item, ("hotelId", "hotel_id", "id"))
        name = str(self._first_value(item, ("hotelName", "hotel_name", "name")) or "").strip()
        address = str(self._first_value(item, ("address", "addr")) or "").strip()
        location = self._extract_location(item) or self._extract_location(detail)
        if not location and address:
            try:
                from .amap_service import get_amap_service

                geocoded = get_amap_service().geocode(address, city)
                if geocoded:
                    location = {"longitude": geocoded.longitude, "latitude": geocoded.latitude}
            except Exception as exc:
                log_event("tuniu_hotel_error", {"stage": "geocode", "hotel_name": name, "error": repr(exc)})
        return {
            "id": str(hotel_id or ""),
            "poi_id": str(hotel_id or ""),
            "name": name,
            "address": address,
            "location": location or {"longitude": 0.0, "latitude": 0.0},
            "type": self._hotel_type(item, detail),
            "rating": str(self._first_value(item, ("commentScore", "score", "rating")) or ""),
            "price_range": str(int(price)) if price > 0 and float(price).is_integer() else str(price or ""),
            "estimated_cost": int(math.ceil(price)) if price > 0 else 0,
            "price_source": price_source,
            "source": "tuniu",
            "city": city,
        }

    def _extract_lowest_price(self, payload: Any) -> float:
        if isinstance(payload, dict):
            direct = self._money(self._first_value(payload, ("lowestPrice", "lowest_price", "minPrice", "price", "salePrice", "totalPrice")))
            prices = [direct] if direct > 0 else []
            for key, value in payload.items():
                lowered = str(key).lower()
                if any(token in lowered for token in ("prebook", "payment", "order", "bookparam")):
                    continue
                if isinstance(value, (dict, list)):
                    nested = self._extract_lowest_price(value)
                    if nested > 0:
                        prices.append(nested)
            return min(prices) if prices else 0.0
        if isinstance(payload, list):
            prices = [self._extract_lowest_price(item) for item in payload]
            prices = [item for item in prices if item > 0]
            return min(prices) if prices else 0.0
        return 0.0

    def _extract_location(self, item: Dict[str, Any]) -> Optional[Dict[str, float]]:
        location = item.get("location") or item.get("coordinate")
        if isinstance(location, dict):
            lng = self._number(self._first_value(location, ("longitude", "lng", "lon")))
            lat = self._number(self._first_value(location, ("latitude", "lat")))
        else:
            lng = self._number(self._first_value(item, ("longitude", "lng", "lon")))
            lat = self._number(self._first_value(item, ("latitude", "lat")))
        if lng and lat:
            return {"longitude": lng, "latitude": lat}
        return None

    def _hotel_type(self, item: Dict[str, Any], detail: Dict[str, Any]) -> str:
        parts = []
        for key in ("starName", "star_name", "brandName", "brand_name", "hotelType"):
            value = self._first_value(item, (key,)) or self._first_value(detail, (key,))
            if value and str(value) not in parts:
                parts.append(str(value))
        return ";".join(parts) if parts else "酒店"

    def _first_value(self, item: Any, names: tuple[str, ...]) -> Any:
        if not isinstance(item, dict):
            return None
        lower_map = {str(key).lower(): value for key, value in item.items()}
        for name in names:
            if name in item:
                return item[name]
            value = lower_map.get(name.lower())
            if value is not None:
                return value
        return None

    def _hotel_key(self, item: Dict[str, Any]) -> str:
        hotel_id = self._first_value(item, ("hotelId", "hotel_id", "id"))
        if hotel_id:
            return f"id:{hotel_id}"
        return f"name:{self._first_value(item, ('hotelName', 'name')) or ''}"

    def _money(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        match = re.search(r"[\d.]+", str(value))
        return float(match.group()) if match else 0.0

    def _number(self, value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _build_check_dates(self, start_date: str, end_date: str) -> tuple[str, str]:
        check_in = datetime.strptime(start_date, "%Y-%m-%d").date()
        last_day = datetime.strptime(end_date, "%Y-%m-%d").date()
        check_out = last_day + timedelta(days=1)
        if check_out <= check_in:
            check_out = check_in + timedelta(days=1)
        return check_in.isoformat(), check_out.isoformat()

    def _build_keyword(self, accommodation: str, keyword: str = "") -> str:
        parts = [str(keyword or "").strip(), str(accommodation or "").strip()]
        return " ".join(part for part in parts if part)


_tuniu_hotel_service: TuniuHotelService | None = None


def get_tuniu_hotel_service() -> TuniuHotelService:
    global _tuniu_hotel_service
    if _tuniu_hotel_service is None:
        _tuniu_hotel_service = TuniuHotelService()
    return _tuniu_hotel_service


def close_tuniu_hotel_service() -> None:
    global _tuniu_hotel_service
    if _tuniu_hotel_service is not None:
        _tuniu_hotel_service.close()
        _tuniu_hotel_service = None
