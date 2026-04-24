"""Persistent Amap query worker pool backed by HTTP APIs."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from queue import LifoQueue
from threading import Lock
from typing import Any, Dict, Iterator, List, Optional

import httpx

from ..config import get_settings


SEARCH_PREFIX = "\u8bf7\u641c\u7d22"
QUERY_PREFIX = "\u8bf7\u67e5\u8be2"
SUITABLE = "\u9002\u5408"
PREFERENCE = "\u504f\u597d"
WEATHER_DAYS = "\u8fd9\u51e0\u5929"
ATTRACTIONS_SUFFIX = "\u7684\u666f\u70b9"
WEATHER_AT = "\u5728"
POSSESSIVE = "\u7684"
FULL_STOP = "\u3002"
COMMA = "\uff0c"

TYPE_ATTRACTION_1 = "\u98ce\u666f\u540d\u80dc"
TYPE_ATTRACTION_2 = "\u79d1\u6559\u6587\u5316\u670d\u52a1"
TYPE_HOTEL = "\u4f4f\u5bbf\u670d\u52a1"
TYPE_RESTAURANT = "\u9910\u996e\u670d\u52a1"
TYPE_SHOPPING = "\u8d2d\u7269\u670d\u52a1"

DEFAULT_ATTRACTION = "\u666f\u70b9"
DEFAULT_HOTEL = "\u9152\u5e97"
DEFAULT_FOOD = "\u7f8e\u98df"
FOOD_FALLBACK_KEYWORDS = [
    "\u7f8e\u98df",
    "\u7279\u8272\u83dc",
    "\u672c\u5730\u5c0f\u5403",
    "\u9910\u9986",
    "\u519c\u5bb6\u83dc",
    "\u519c\u5bb6\u9662",
]


def _split_location(value: str) -> Dict[str, float]:
    try:
        lng, lat = value.split(",", 1)
        return {"longitude": float(lng), "latitude": float(lat)}
    except Exception:
        return {"longitude": 0.0, "latitude": 0.0}


class _AmapWorker:
    def __init__(self, worker_id: int) -> None:
        settings = get_settings()
        self.worker_id = worker_id
        self.api_key = settings.amap_api_key
        self.client = httpx.Client(timeout=20.0)

    def close(self) -> None:
        self.client.close()

    def run(self, domain: str, query: str) -> str:
        if domain == "attractions":
            return self._search_attractions(query)
        if domain == "hotels":
            return self._search_hotels(query)
        if domain == "restaurants":
            return self._search_restaurants(query)
        if domain == "weather":
            return self._get_weather(query)
        raise ValueError(f"Unsupported domain: {domain}")

    def _search_attractions(self, query: str) -> str:
        city = self._extract_city_after_prefix(query, SEARCH_PREFIX)
        keyword = self._extract_between(query, SUITABLE, PREFERENCE)
        if not keyword:
            keyword = self._extract_between(query, city, ATTRACTIONS_SUFFIX)
        return self._search_poi(city=city, keyword=keyword or DEFAULT_ATTRACTION, types=[TYPE_ATTRACTION_1, TYPE_ATTRACTION_2])

    def _search_hotels(self, query: str) -> str:
        city = self._extract_between(query, SEARCH_PREFIX, POSSESSIVE) or self._extract_city_after_prefix(query, SEARCH_PREFIX)
        keyword = self._extract_between(query, POSSESSIVE, FULL_STOP) or self._extract_after(query, POSSESSIVE)
        return self._search_poi(city=city, keyword=(keyword or DEFAULT_HOTEL).strip(), types=[TYPE_HOTEL])

    def _search_restaurants(self, query: str) -> str:
        city = self._extract_between(query, SEARCH_PREFIX, POSSESSIVE) or self._extract_city_after_prefix(query, SEARCH_PREFIX)
        keyword = self._extract_between(query, POSSESSIVE, FULL_STOP) or self._extract_after(query, POSSESSIVE)
        primary_keyword = (keyword or DEFAULT_FOOD).strip()
        generic_keywords = [DEFAULT_FOOD] + FOOD_FALLBACK_KEYWORDS
        search_keywords: List[str] = []
        for item in [primary_keyword] + generic_keywords:
            item = item.strip()
            if item and item not in search_keywords:
                search_keywords.append(item)
        for search_keyword in search_keywords:
            result_text = self._search_poi(city=city, keyword=search_keyword, types=[TYPE_RESTAURANT, TYPE_SHOPPING])
            try:
                result_data = json.loads(result_text)
            except Exception:
                result_data = {}
            if result_data.get("pois"):
                if search_keyword != primary_keyword:
                    print(
                        f"INFO restaurant search fallback hit | city={city} primary_keyword={primary_keyword} fallback_keyword={search_keyword}",
                        flush=True,
                    )
                return result_text
        return json.dumps({"pois": []}, ensure_ascii=False)

    def _get_weather(self, query: str) -> str:
        city = self._extract_between(query, QUERY_PREFIX, WEATHER_AT) or self._extract_between(query, QUERY_PREFIX, WEATHER_DAYS)
        city = city.strip()
        params = {
            "key": self.api_key,
            "city": city,
            "extensions": "all",
            "output": "JSON",
        }
        print(f"INFO Amap HTTP call: worker={self.worker_id} domain=weather city={city}", flush=True)
        response = self.client.get("https://restapi.amap.com/v3/weather/weatherInfo", params=params)
        response.raise_for_status()
        data = response.json()
        return json.dumps({"city": city, "forecasts": data.get("forecasts", [])}, ensure_ascii=False)

    def _search_poi(self, city: str, keyword: str, types: List[str]) -> str:
        city = city.strip()
        keyword = keyword.strip()
        params = {
            "key": self.api_key,
            "keywords": keyword,
            "region": city,
            "city_limit": "true",
            "show_fields": "business",
            "page_size": "10",
            "page_num": "1",
        }
        print(f"INFO Amap HTTP call: worker={self.worker_id} domain=poi city={city} keyword={keyword}", flush=True)
        response = self.client.get("https://restapi.amap.com/v5/place/text", params=params)
        response.raise_for_status()
        data = response.json()
        pois = []
        for item in data.get("pois", []):
            category = str(item.get("type", ""))
            if types and not any(token in category for token in types):
                continue
            pois.append(
                {
                    "id": item.get("id", ""),
                    "name": item.get("name", ""),
                    "address": item.get("address", ""),
                    "location": _split_location(item.get("location", "0,0")),
                    "type": category,
                    "rating": item.get("business", {}).get("rating", ""),
                    "price_range": item.get("business", {}).get("cost", ""),
                }
            )
        return json.dumps({"pois": pois[:5]}, ensure_ascii=False)

    @staticmethod
    def _extract_between(text: str, start: str, end: str) -> str:
        if start not in text or end not in text:
            return ""
        return text.split(start, 1)[1].split(end, 1)[0].strip()

    @staticmethod
    def _extract_after(text: str, start: str) -> str:
        if start not in text:
            return ""
        return text.split(start, 1)[1].strip()

    @staticmethod
    def _extract_city_after_prefix(text: str, prefix: str) -> str:
        if prefix not in text:
            return ""
        remainder = text.split(prefix, 1)[1]
        for stop in (SUITABLE, POSSESSIVE, WEATHER_AT, FULL_STOP, COMMA):
            if stop in remainder:
                return remainder.split(stop, 1)[0].strip()
        return remainder.strip()


class AmapWorkerPool:
    def __init__(self, llm: Any, prompts: Dict[str, str], size: Optional[int] = None) -> None:
        configured_size = size or int(os.getenv("AMAP_MCP_POOL_SIZE", "2"))
        self.size = max(1, configured_size)
        self._lock = Lock()
        self._queue: LifoQueue[_AmapWorker] = LifoQueue(maxsize=self.size)
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            for worker_id in range(self.size):
                self._queue.put(_AmapWorker(worker_id=worker_id))
            self._started = True
            print(f"INFO Amap worker pool started with size={self.size}", flush=True)

    def close(self) -> None:
        with self._lock:
            if not self._started:
                return
            workers = []
            while not self._queue.empty():
                workers.append(self._queue.get_nowait())
            for worker in workers:
                worker.close()
            self._started = False
            print("INFO Amap worker pool closed", flush=True)

    @contextmanager
    def acquire(self) -> Iterator[_AmapWorker]:
        self.start()
        worker = self._queue.get()
        try:
            yield worker
        finally:
            self._queue.put(worker)

    def run(self, domain: str, query: str) -> str:
        with self.acquire() as worker:
            return worker.run(domain, query)
