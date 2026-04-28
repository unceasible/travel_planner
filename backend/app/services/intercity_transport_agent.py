# -*- coding: utf-8 -*-
"""Read-only intercity transport retrieval agent."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from ..config import get_settings
from ..models.schemas import IntercityTransportOption, IntercityTransportPlan
from .agent_output_logger import log_event, timed_event
from .amap_service import get_amap_service


ALLOWED_TUNIU_INTERCITY_TOOLS = {
    "searchLowestPriceFlight",
    "multiCabinDetails",
    "searchLowestPriceTrain",
    "queryTrainDetail",
}


class TuniuIntercityTransportService:
    """Read-only Tuniu MCP client for flights and trains."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = httpx.Client(timeout=18.0)

    def close(self) -> None:
        self.client.close()

    def search_flights(
        self,
        *,
        direction: str,
        departure_city: str,
        arrival_city: str,
        date: str,
        limit: int = 5,
    ) -> List[IntercityTransportOption]:
        if not self.settings.tuniu_api_key:
            raise RuntimeError("TUNIU_API_KEY is not configured")
        args = {
            "departureCity": departure_city,
            "arrivalCity": arrival_city,
            "departureDate": date,
        }
        with timed_event("tuniu.flight_search", {"direction": direction}):
            payload = self._call_tool("searchLowestPriceFlight", args, self.settings.tuniu_flight_mcp_url)
        options = [
            self._normalize_flight(item, direction, departure_city, arrival_city, date)
            for item in self._extract_items(payload)
        ]
        return [item for item in options if item][:limit]

    def search_trains(
        self,
        *,
        direction: str,
        departure_city: str,
        arrival_city: str,
        date: str,
        limit: int = 5,
    ) -> List[IntercityTransportOption]:
        if not self.settings.tuniu_api_key:
            raise RuntimeError("TUNIU_API_KEY is not configured")
        args = {
            "departureCity": departure_city,
            "arrivalCity": arrival_city,
            "departureDate": date,
        }
        with timed_event("tuniu.train_search", {"direction": direction}):
            payload = self._call_tool("searchLowestPriceTrain", args, self.settings.tuniu_train_mcp_url)
        options = [
            self._normalize_train(item, direction, departure_city, arrival_city, date)
            for item in self._extract_items(payload)
        ]
        return [item for item in options if item][:limit]

    def _call_tool(self, tool_name: str, arguments: Dict[str, Any], endpoint: str = "") -> Dict[str, Any]:
        if tool_name not in ALLOWED_TUNIU_INTERCITY_TOOLS:
            raise ValueError(f"Tuniu intercity tool is not allowed: {tool_name}")
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
        response = self.client.post(endpoint or self.settings.tuniu_flight_mcp_url, headers=headers, json=body)
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
        return result if isinstance(result, dict) else {}

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

    def _extract_items(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        for key in ("flights", "flightList", "trains", "trainList", "list", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = self._extract_items(value)
                if nested:
                    return nested
        for value in payload.values():
            nested = self._extract_items(value)
            if nested:
                return nested
        return []

    def _normalize_flight(
        self,
        item: Dict[str, Any],
        direction: str,
        departure_city: str,
        arrival_city: str,
        date: str,
    ) -> IntercityTransportOption:
        departure_time = self._time_text(self._first_value(item, ("depTime", "departureTime", "takeoffTime", "startTime")))
        arrival_time = self._time_text(self._first_value(item, ("arrTime", "arrivalTime", "landTime", "endTime")))
        return IntercityTransportOption(
            direction=direction,
            mode="飞机",
            provider="tuniu_flight",
            departure_city=str(self._first_value(item, ("depCity", "departureCity", "fromCity")) or departure_city),
            arrival_city=str(self._first_value(item, ("arrCity", "arrivalCity", "toCity")) or arrival_city),
            date=date,
            departure_time=departure_time,
            arrival_time=arrival_time,
            duration_minutes=self._duration_minutes(departure_time, arrival_time, date),
            estimated_cost=self._money(self._first_value(item, ("lowestPrice", "price", "priceWithTax", "salePrice", "adultPrice"))),
            code=str(self._first_value(item, ("flightNo", "flightNumber", "code")) or ""),
            data_source="tuniu_real_time",
            description=str(self._first_value(item, ("airlineName", "airline", "cabinName")) or ""),
        )

    def _normalize_train(
        self,
        item: Dict[str, Any],
        direction: str,
        departure_city: str,
        arrival_city: str,
        date: str,
    ) -> IntercityTransportOption:
        departure_time = self._time_text(self._first_value(item, ("depTime", "departureTime", "startTime", "fromTime")))
        arrival_time = self._time_text(self._first_value(item, ("arrTime", "arrivalTime", "arriveTime", "endTime", "toTime")))
        return IntercityTransportOption(
            direction=direction,
            mode="火车",
            provider="tuniu_train",
            departure_city=str(self._first_value(item, ("depCity", "departureCity", "fromCity", "startCity")) or departure_city),
            arrival_city=str(self._first_value(item, ("arrCity", "arrivalCity", "toCity", "endCity")) or arrival_city),
            date=date,
            departure_time=departure_time,
            arrival_time=arrival_time,
            duration_minutes=self._duration_minutes(departure_time, arrival_time, date),
            estimated_cost=self._money(self._first_value(item, ("lowestPrice", "price", "minPrice", "secondSeatPrice", "hardSeatPrice"))),
            code=str(self._first_value(item, ("trainNo", "trainNum", "trainCode", "code")) or ""),
            data_source="tuniu_real_time",
            description=str(self._first_value(item, ("seatName", "trainType", "note")) or ""),
        )

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

    def _time_text(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        match = re.search(r"(\d{1,2}:\d{2})", text)
        return match.group(1) if match else text

    def _duration_minutes(self, departure_time: str, arrival_time: str, date: str) -> int:
        if not departure_time or not arrival_time:
            return 0
        try:
            depart = datetime.strptime(f"{date} {departure_time[:5]}", "%Y-%m-%d %H:%M")
            arrive = datetime.strptime(f"{date} {arrival_time[:5]}", "%Y-%m-%d %H:%M")
            if arrive < depart:
                arrive += timedelta(days=1)
            return max(0, int((arrive - depart).total_seconds() // 60))
        except Exception:
            return 0

    def _money(self, value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(round(value))
        match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
        return int(float(match.group())) if match else 0


class IntercityTransportAgent:
    """Retrieve and select round-trip intercity transport options."""

    def __init__(self, tuniu_service: Optional[Any] = None, amap_service: Optional[Any] = None) -> None:
        self.tuniu_service = tuniu_service or TuniuIntercityTransportService()
        self.amap_service = amap_service or get_amap_service()

    def search(
        self,
        *,
        departure_city: str,
        destination_city: str,
        start_date: str,
        end_date: str,
        preference: str = "智能推荐",
        allow_self_drive: bool = True,
    ) -> IntercityTransportPlan:
        departure_city = str(departure_city or "").strip()
        destination_city = str(destination_city or "").strip()
        preference = str(preference or "智能推荐").strip() or "智能推荐"
        if not departure_city or not destination_city or self._same_city(departure_city, destination_city):
            return IntercityTransportPlan(
                status="skipped",
                preference=preference,
                warnings=["出发城市为空或与目的地相同，已跳过大交通检索。"],
            )

        outbound, outbound_warnings = self._search_direction(
            direction="outbound",
            departure_city=departure_city,
            arrival_city=destination_city,
            date=start_date,
            preference=preference,
            allow_self_drive=allow_self_drive,
        )
        inbound, inbound_warnings = self._search_direction(
            direction="return",
            departure_city=destination_city,
            arrival_city=departure_city,
            date=end_date,
            preference=preference,
            allow_self_drive=allow_self_drive,
        )
        selected_outbound = self._select_best(outbound)
        selected_return = self._select_best(inbound)
        warnings = [*outbound_warnings, *inbound_warnings]
        if selected_outbound and selected_return:
            status = "ok"
        elif outbound or inbound:
            status = "partial"
            warnings.append("只检索到部分大交通候选，计划已继续生成。")
        else:
            status = "unavailable"
            warnings.append("未获取到实时票务或自驾路线，计划已继续生成。")
        return IntercityTransportPlan(
            status=status,
            preference=preference,
            outbound_candidates=outbound,
            return_candidates=inbound,
            selected_outbound=selected_outbound,
            selected_return=selected_return,
            schedule_constraints=self._build_schedule_constraints(selected_outbound, selected_return),
            warnings=warnings,
        )

    def _search_direction(
        self,
        *,
        direction: str,
        departure_city: str,
        arrival_city: str,
        date: str,
        preference: str,
        allow_self_drive: bool,
    ) -> tuple[List[IntercityTransportOption], List[str]]:
        modes = self._modes_for_preference(preference, allow_self_drive)
        candidates: List[IntercityTransportOption] = []
        warnings: List[str] = []
        with ThreadPoolExecutor(max_workers=len(modes)) as executor:
            futures = {}
            for mode in modes:
                if mode == "飞机":
                    futures[executor.submit(
                        self.tuniu_service.search_flights,
                        direction=direction,
                        departure_city=departure_city,
                        arrival_city=arrival_city,
                        date=date,
                    )] = mode
                elif mode == "火车":
                    futures[executor.submit(
                        self.tuniu_service.search_trains,
                        direction=direction,
                        departure_city=departure_city,
                        arrival_city=arrival_city,
                        date=date,
                    )] = mode
                elif mode == "自驾":
                    futures[executor.submit(
                        self._search_driving,
                        direction,
                        departure_city,
                        arrival_city,
                        date,
                    )] = mode
            for future in as_completed(futures):
                mode = futures[future]
                try:
                    result = future.result()
                    if isinstance(result, list):
                        candidates.extend(result)
                    elif result:
                        candidates.append(result)
                except Exception as exc:
                    warnings.append(f"{direction} {mode}检索失败: {exc}")
                    log_event("intercity_transport_error", {"direction": direction, "mode": mode, "error": repr(exc)})
        candidates.sort(key=lambda item: self._option_score(item), reverse=True)
        return candidates[:10], warnings

    def _search_driving(self, direction: str, departure_city: str, arrival_city: str, date: str) -> IntercityTransportOption:
        route = self.amap_service.plan_route(
            departure_city,
            arrival_city,
            departure_city,
            arrival_city,
            "driving",
        )
        distance = float((route or {}).get("distance") or 0)
        duration_seconds = int((route or {}).get("duration") or 0)
        route_cost = (route or {}).get("cost")
        cost = int(route_cost) if route_cost is not None else self._estimate_driving_cost(distance)
        return IntercityTransportOption(
            direction=direction,
            mode="自驾",
            provider="amap",
            departure_city=departure_city,
            arrival_city=arrival_city,
            date=date,
            duration_minutes=max(0, duration_seconds // 60),
            estimated_cost=cost,
            data_source="amap_route" if route else "rule_based",
            description=str((route or {}).get("description") or "自驾大交通估算"),
        )

    def _modes_for_preference(self, preference: str, allow_self_drive: bool = True) -> List[str]:
        if preference in {"飞机", "火车", "自驾"}:
            return [preference]
        modes = ["飞机", "火车"]
        if allow_self_drive:
            modes.append("自驾")
        return modes

    def _select_best(self, options: List[IntercityTransportOption]) -> Optional[IntercityTransportOption]:
        if not options:
            return None
        return sorted(options, key=lambda item: self._option_score(item), reverse=True)[0]

    def _option_score(self, option: IntercityTransportOption) -> float:
        score = 50.0
        if option.estimated_cost > 0:
            score += max(0.0, 40.0 - option.estimated_cost / 30.0)
        if option.duration_minutes > 0:
            score += max(0.0, 30.0 - option.duration_minutes / 30.0)
        if option.departure_time:
            score += 5
        if option.arrival_time:
            score += 5
        if option.code:
            score += 5
        if option.data_source in {"tuniu_real_time", "amap_route"}:
            score += 5
        return score

    def _build_schedule_constraints(
        self,
        outbound: Optional[IntercityTransportOption],
        inbound: Optional[IntercityTransportOption],
    ) -> Dict[str, Any]:
        constraints: Dict[str, Any] = {}
        if outbound and outbound.arrival_time:
            constraints["first_day_max_attractions"] = self._arrival_cap(outbound.arrival_time)
            constraints["outbound_arrival_time"] = outbound.arrival_time
        if inbound and inbound.departure_time:
            constraints["last_day_max_attractions"] = self._departure_cap(inbound.departure_time)
            constraints["return_departure_time"] = inbound.departure_time
        return constraints

    def _arrival_cap(self, time_text: str) -> int:
        minutes = self._minutes_since_midnight(time_text)
        if minutes < 12 * 60:
            return 2
        if minutes < 16 * 60:
            return 1
        return 0

    def _departure_cap(self, time_text: str) -> int:
        minutes = self._minutes_since_midnight(time_text)
        if minutes < 12 * 60:
            return 0
        if minutes < 16 * 60:
            return 1
        return 2

    def _minutes_since_midnight(self, time_text: str) -> int:
        match = re.search(r"(\d{1,2}):(\d{2})", str(time_text or ""))
        if not match:
            return 12 * 60
        return int(match.group(1)) * 60 + int(match.group(2))

    def _estimate_driving_cost(self, distance_m: float) -> int:
        if distance_m <= 0:
            return 0
        return int(round(distance_m / 1000.0 * 0.9))

    def _same_city(self, left: str, right: str) -> bool:
        return self._normalize_city(left) == self._normalize_city(right)

    def _normalize_city(self, value: str) -> str:
        return re.sub(r"(市|省|自治区|特别行政区)$", "", str(value or "").strip())


_intercity_transport_agent: Optional[IntercityTransportAgent] = None


def get_intercity_transport_agent() -> IntercityTransportAgent:
    global _intercity_transport_agent
    if _intercity_transport_agent is None:
        _intercity_transport_agent = IntercityTransportAgent()
    return _intercity_transport_agent
