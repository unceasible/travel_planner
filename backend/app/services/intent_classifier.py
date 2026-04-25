"""Intent classification for trip chat requests."""
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Sequence

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from .agent_output_logger import log_event, log_full_output, timed_event
from .llm_service import get_cheap_openai_client, get_cheap_openai_model


ALLOWED_INTENTS = {"modify", "replan", "question", "satisfied", "chitchat", "refuse", "unclear"}
ALLOWED_DOMAINS = {
    "attractions",
    "hotels",
    "restaurants",
    "transportation",
    "schedule",
    "budget",
    "weather",
    "city",
    "dates",
    "preferences",
    "none",
}
ALLOWED_ACTIONS = {
    "add",
    "remove",
    "replace",
    "increase",
    "decrease",
    "reorder",
    "refresh",
    "ask",
    "confirm",
    "chat",
    "refuse",
    "none",
}
ALLOWED_SOURCES = {"rule", "embedding", "llm", "fallback"}

PRIMARY_PRIORITY = {
    "replan": 600,
    "modify": 500,
    "question": 400,
    "satisfied": 300,
    "chitchat": 200,
    "unclear": 100,
    "refuse": 50,
}

NON_PERSIST_INTENTS = {"chitchat", "refuse"}
EMBEDDING_ACCEPT_THRESHOLD = 0.82
EMBEDDING_CACHE_VERSION = 1


class IntentResult(BaseModel):
    primary_intent: str
    intents: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    action: str = "none"
    confidence: float = 0.0
    source: str = "fallback"
    matched_rule: str = ""

    @field_validator("primary_intent")
    @classmethod
    def validate_primary_intent(cls, value: str) -> str:
        if value not in ALLOWED_INTENTS:
            raise ValueError(f"invalid primary_intent: {value}")
        return value

    @field_validator("intents")
    @classmethod
    def validate_intents(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("intents must not be empty")
        invalid = [item for item in value if item not in ALLOWED_INTENTS]
        if invalid:
            raise ValueError(f"invalid intents: {invalid}")
        return _unique_keep_order(value)

    @field_validator("domains")
    @classmethod
    def validate_domains(cls, value: List[str]) -> List[str]:
        if not value:
            return ["none"]
        invalid = [item for item in value if item not in ALLOWED_DOMAINS]
        if invalid:
            raise ValueError(f"invalid domains: {invalid}")
        return _unique_keep_order(value)

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: str) -> str:
        if value not in ALLOWED_ACTIONS:
            raise ValueError(f"invalid action: {value}")
        return value

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        if value not in ALLOWED_SOURCES:
            raise ValueError(f"invalid source: {value}")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        confidence = float(value)
        if confidence < 0 or confidence > 1:
            raise ValueError(f"confidence out of range: {confidence}")
        return confidence


@dataclass(frozen=True)
class IntentRule:
    name: str
    intent: str
    domains: List[str]
    action: str
    confidence: float
    patterns: List[str]


INTENT_RULES = [
    IntentRule("out_of_travel_scope", "refuse", ["none"], "refuse", 0.95, [
        r"股票|股市|基金|理财|投资建议|炒股",
        r"政治|大选|总统|选举|政党",
        r"头疼|发烧|吃什么药|诊断|处方|疾病|法律|律师|合同",
        r"攻击代码|黑客|漏洞利用|木马|病毒",
        r"\bGDP\b|国内生产总值",
    ]),
    IntentRule("replan_city", "replan", ["city"], "replace", 0.94, [
        r"(换|改)(成|去)?[一-龥]{2,8}(玩|旅游|旅行|行程)",
        r"(换|改).{0,6}(城市|目的地)",
    ]),
    IntentRule("replan_dates", "replan", ["dates"], "replace", 0.93, [
        r"(日期|时间).{0,8}(改|换|调整)",
        r"(改|换|调整).{0,8}(日期|时间)",
        r"改到.{0,12}(五一|十一|国庆|春节|\d{1,2}月\d{1,2}日|\d{4}-\d{1,2}-\d{1,2})",
    ]),
    IntentRule("increase_days", "replan", ["dates"], "increase", 0.92, [
        r"多玩.{0,5}(天|日)",
        r"增加.{0,5}(天|日)",
        r"多.{0,3}(一|两|二|三|四|五|\d+).{0,2}(天|日)",
    ]),
    IntentRule("decrease_days", "replan", ["dates"], "decrease", 0.92, [
        r"少玩.{0,5}(天|日)",
        r"减少.{0,5}(天|日)",
        r"缩短.{0,8}行程",
    ]),
    IntentRule("replan_all", "replan", ["schedule"], "refresh", 0.95, [
        r"重新规划|重新安排|从头.*规划|全部重排|整体重做|全局调整",
    ]),
    IntentRule("increase_attractions", "modify", ["attractions"], "increase", 0.95, [
        r"(多|增加|加|更多|丰富).{0,8}(景点|景区|博物馆|公园|打卡)",
        r"(景点|景区|博物馆|公园|打卡).{0,8}(多|增加|加|更多|少)",
    ]),
    IntentRule("replace_attractions", "modify", ["attractions", "schedule"], "replace", 0.9, [
        r"(第[一二三四五六七\d]+天|上午|下午|晚上).{0,12}(换|改).{0,12}(景点|博物馆|公园)",
        r"(不要|不想去|删掉).{0,8}(景点|博物馆|公园)",
    ]),
    IntentRule("replace_hotels", "modify", ["hotels"], "replace", 0.92, [
        r"(酒店|住宿|住的地方|民宿).{0,12}(换|改|重新|推荐)",
        r"(换|改|重新).{0,12}(酒店|住宿|住的地方|民宿)",
    ]),
    IntentRule("cheaper_hotels", "modify", ["hotels", "budget"], "decrease", 0.92, [
        r"(酒店|住宿|住的地方).{0,10}(便宜|省钱|经济)",
        r"(便宜|省钱|经济).{0,10}(酒店|住宿|住的地方)",
    ]),
    IntentRule("replace_restaurants", "modify", ["restaurants"], "replace", 0.9, [
        r"(餐厅|吃|美食|午餐|晚餐|早餐).{0,12}(换|改|重新|推荐)",
        r"(不吃|不要吃|避免|过敏).{0,12}(辣|海鲜|牛肉|羊肉|火锅|烧烤)",
    ]),
    IntentRule("replace_transportation", "modify", ["transportation"], "replace", 0.92, [
        r"(交通|出行|路线).{0,12}(换|改|调整)",
        r"改成.{0,4}(自驾|公共交通|公交|地铁|步行|打车|混合)",
    ]),
    IntentRule("decrease_schedule", "modify", ["schedule"], "decrease", 0.88, [
        r"(太累|太赶|轻松|慢一点|少走路|休闲)",
    ]),
    IntentRule("reorder_schedule", "modify", ["schedule"], "reorder", 0.86, [
        r"(先|后|顺序|调换|提前|推迟)",
    ]),
    IntentRule("budget_change", "modify", ["budget"], "decrease", 0.88, [
        r"(预算|花费|费用|价格).{0,12}(低|少|省|便宜|控制)",
    ]),
    IntentRule("weather_refresh", "modify", ["weather"], "refresh", 0.85, [
        r"(天气|下雨|温度|冷|热).{0,12}(查|看看|调整|影响)",
    ]),
    IntentRule("ask_schedule", "question", ["schedule"], "ask", 0.9, [
        r"(第[一二三四五六七\d]+天|上午|下午|晚上).{0,10}(去哪|安排|干什么|有什么)",
        r"(去哪|安排是什么|行程是什么)",
    ]),
    IntentRule("ask_budget", "question", ["budget"], "ask", 0.9, [
        r"(预算|花费|费用|总价|多少钱|多少元)",
    ]),
    IntentRule("ask_hotel", "question", ["hotels"], "ask", 0.88, [
        r"(酒店|住宿).{0,10}(在哪|哪里|怎么样|叫什么)",
    ]),
    IntentRule("ask_weather", "question", ["weather"], "ask", 0.88, [
        r"(天气|温度).{0,10}(怎么样|如何|多少)",
    ]),
    IntentRule("satisfied_confirm", "satisfied", ["none"], "confirm", 0.95, [
        r"^(可以|好|好的|行|就这样|不用改|不改了|挺好|不错|满意)[。！!,.，\s]*$",
        r"^(就这样|不用改|不改了|可以了|满意|没问题|挺好)[。！!,.，\s]*(不用改|不改了)?[。！!,.，\s]*$",
        r"(计划|安排).{0,8}(挺好|不错|满意|可以)",
    ]),
    IntentRule("thanks_or_greeting", "chitchat", ["none"], "chat", 0.95, [
        r"^(谢谢|感谢|辛苦了|你好|您好|哈哈|嘿嘿|你真厉害)[。！!,.，\s]*$",
    ]),
]


def _unique_keep_order(items: Sequence[str]) -> List[str]:
    result: List[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _example_texts_hash(examples: Sequence[Dict[str, Any]]) -> str:
    texts = [str(item.get("text", "")) for item in examples]
    payload = json.dumps(texts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _primary_from_intents(intents: Sequence[str]) -> str:
    filtered = [intent for intent in intents if intent != "refuse"]
    if filtered:
        return max(filtered, key=lambda item: PRIMARY_PRIORITY.get(item, 0))
    if "refuse" in intents:
        return "refuse"
    return "unclear"


def _normalize_result(data: Dict[str, Any], source: str, matched_rule: str = "") -> IntentResult:
    raw_intents = data.get("intents") or [data.get("primary_intent") or "unclear"]
    intents = _unique_keep_order([str(item) for item in raw_intents if item])
    explicit_primary = data.get("primary_intent")
    primary = str(explicit_primary or _primary_from_intents(intents))
    if explicit_primary and primary not in intents:
        raise ValueError("primary_intent must be present in intents")
    if not explicit_primary and primary not in intents:
        intents.append(primary)
    primary = _primary_from_intents(intents)

    raw_domains = data.get("domains") or ["none"]
    domains = _unique_keep_order([str(item) for item in raw_domains if item])
    if primary in {"satisfied", "chitchat", "refuse", "unclear"}:
        domains = ["none"]
    elif not domains or domains == ["none"]:
        raise ValueError(f"{primary} requires at least one concrete domain")

    action = str(data.get("action") or "none")
    if primary == "satisfied":
        action = "confirm"
    elif primary == "chitchat":
        action = "chat"
    elif primary == "refuse":
        action = "refuse"
    elif primary == "unclear":
        action = "none"

    result = IntentResult(
        primary_intent=primary,
        intents=intents,
        domains=domains,
        action=action,
        confidence=float(data.get("confidence", 0.0)),
        source=source,
        matched_rule=matched_rule or str(data.get("matched_rule") or ""),
    )
    if result.primary_intent not in result.intents:
        raise ValueError("primary_intent must be present in intents")
    return result


def fallback_unclear(source: str = "fallback") -> IntentResult:
    return IntentResult(
        primary_intent="unclear",
        intents=["unclear"],
        domains=["none"],
        action="none",
        confidence=0.0,
        source=source,
        matched_rule="",
    )


class SiliconFlowEmbeddingClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("SILICONFLOW_API_KEY", "")
        self.base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
        self.model = os.getenv("INTENT_EMBEDDING_MODEL", "netease-youdao/bce-embedding-base_v1")
        self.timeout = float(os.getenv("INTENT_EMBEDDING_TIMEOUT", "20"))

    def available(self) -> bool:
        return bool(self.api_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is not configured")
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data", [])
        rows = sorted(rows, key=lambda item: item.get("index", 0))
        return [row["embedding"] for row in rows]


class IntentClassifier:
    def __init__(self) -> None:
        self.embedding_client = SiliconFlowEmbeddingClient()
        self.examples = self._load_examples()
        self.examples_hash = _example_texts_hash(self.examples)
        self.embedding_cache_path = (
            Path(__file__).resolve().parents[2] / "runtime_data" / "cache" / "intent_example_embeddings.json"
        )
        self._embedding_cache_lock = Lock()
        self._example_embeddings: List[List[float]] | None = None

    def classify(self, user_message: str, trip_context: Dict[str, Any]) -> IntentResult:
        payload = {"user_message": user_message, "trip_context": trip_context}
        log_event("intent_input", payload)

        with timed_event("intent.rule"):
            rule_result = self._classify_by_rules(user_message)
        if rule_result is not None:
            log_event("intent_final", rule_result.model_dump())
            return rule_result

        with timed_event("intent.embedding"):
            embedding_result = self._classify_by_embedding(user_message)
        if embedding_result and embedding_result.confidence >= EMBEDDING_ACCEPT_THRESHOLD:
            log_event("intent_final", embedding_result.model_dump())
            return embedding_result

        with timed_event("intent.llm"):
            llm_result = self._classify_by_llm(payload)
        log_event("intent_final", llm_result.model_dump())
        return llm_result

    def _load_examples(self) -> List[Dict[str, Any]]:
        path = Path(__file__).with_name("intent_examples.json")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log_event("intent_examples_load_error", str(exc))
            return []

    def warmup_example_embeddings(self) -> bool:
        if not self.examples:
            log_event("intent_embedding_warmup", "skipped: missing examples")
            return False
        if not self.embedding_client.available():
            log_event("intent_embedding_warmup", "skipped: SILICONFLOW_API_KEY is not configured")
            return False
        try:
            with timed_event("intent.embedding_warmup"):
                embeddings = self._load_example_embeddings()
            log_event("intent_embedding_warmup", {"status": "ok", "count": len(embeddings)})
            return True
        except Exception as exc:
            log_event("intent_embedding_warmup", f"error: {exc}")
            return False

    def _load_example_embeddings(self) -> List[List[float]]:
        if self._example_embeddings is not None:
            log_event("intent_embedding_cache_hit", {"source": "memory", "count": len(self._example_embeddings)})
            return self._example_embeddings

        with self._embedding_cache_lock:
            if self._example_embeddings is not None:
                log_event("intent_embedding_cache_hit", {"source": "memory", "count": len(self._example_embeddings)})
                return self._example_embeddings

            with timed_event("intent.embedding_cache_read"):
                cached = self._read_embedding_cache()
            if cached is not None:
                self._example_embeddings = cached
                return cached

            log_event("intent_embedding_cache_miss", {"reason": "regenerate", "count": len(self.examples)})
            texts = [item["text"] for item in self.examples]
            with timed_event("intent.example_embedding_request", {"count": len(texts), "model": self.embedding_client.model}):
                embeddings = self.embedding_client.embed(texts)
            self._example_embeddings = embeddings
            with timed_event("intent.embedding_cache_write"):
                self._write_embedding_cache(embeddings)
            return embeddings

    def _read_embedding_cache(self) -> List[List[float]] | None:
        path = self.embedding_cache_path
        if not path.exists():
            log_event("intent_embedding_cache_miss", {"reason": "file_not_found", "path": str(path)})
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("cache_version") != EMBEDDING_CACHE_VERSION:
                log_event("intent_embedding_cache_miss", {"reason": "cache_version"})
                return None
            if payload.get("model") != self.embedding_client.model:
                log_event("intent_embedding_cache_miss", {"reason": "model"})
                return None
            if payload.get("base_url") != self.embedding_client.base_url:
                log_event("intent_embedding_cache_miss", {"reason": "base_url"})
                return None
            if payload.get("examples_hash") != self.examples_hash:
                log_event("intent_embedding_cache_miss", {"reason": "examples_hash"})
                return None

            embeddings = payload.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(self.examples):
                log_event("intent_embedding_cache_miss", {"reason": "embedding_count"})
                return None
            if not all(isinstance(row, list) for row in embeddings):
                log_event("intent_embedding_cache_miss", {"reason": "embedding_shape"})
                return None

            log_event("intent_embedding_cache_hit", {"source": "disk", "count": len(embeddings)})
            return embeddings
        except Exception as exc:
            log_event("intent_embedding_cache_error", f"read: {exc}")
            return None

    def _write_embedding_cache(self, embeddings: List[List[float]]) -> None:
        path = self.embedding_cache_path
        payload = {
            "cache_version": EMBEDDING_CACHE_VERSION,
            "model": self.embedding_client.model,
            "base_url": self.embedding_client.base_url,
            "examples_hash": self.examples_hash,
            "created_at": datetime.utcnow().replace(microsecond=0).isoformat(),
            "embeddings": embeddings,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
            temp_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            os.replace(temp_path, path)
            log_event("intent_embedding_cache_write", {"path": str(path), "count": len(embeddings)})
        except Exception as exc:
            log_event("intent_embedding_cache_error", f"write: {exc}")

    def _classify_by_rules(self, user_message: str) -> IntentResult | None:
        matches: List[IntentRule] = []
        for rule in INTENT_RULES:
            if any(re.search(pattern, user_message, re.IGNORECASE) for pattern in rule.patterns):
                matches.append(rule)

        log_event("intent_rule_matches", [
            {"name": rule.name, "intent": rule.intent, "domains": rule.domains, "action": rule.action}
            for rule in matches
        ])
        if not matches:
            return None

        intents = _unique_keep_order([rule.intent for rule in matches])
        primary = _primary_from_intents(intents)
        primary_matches = [rule for rule in matches if rule.intent == primary] or matches
        domains: List[str] = []
        for rule in matches:
            for domain in rule.domains:
                if domain != "none":
                    domains.append(domain)
        domains = _unique_keep_order(domains) or ["none"]
        action = primary_matches[0].action
        confidence = max(rule.confidence for rule in primary_matches)
        matched_rule = ",".join(rule.name for rule in matches)
        return _normalize_result(
            {
                "primary_intent": primary,
                "intents": intents,
                "domains": domains,
                "action": action,
                "confidence": confidence,
            },
            source="rule",
            matched_rule=matched_rule,
        )

    def _classify_by_embedding(self, user_message: str) -> IntentResult | None:
        if not self.examples or not self.embedding_client.available():
            log_event("intent_embedding_match", "skipped: missing examples or SILICONFLOW_API_KEY")
            return None
        try:
            example_embeddings = self._load_example_embeddings()
            with timed_event("intent.user_embedding_request", {"model": self.embedding_client.model}):
                user_embedding = self.embedding_client.embed([user_message])[0]
            with timed_event("intent.embedding_similarity", {"examples": len(self.examples)}):
                scored = [
                    (_cosine_similarity(user_embedding, embedding), example)
                    for embedding, example in zip(example_embeddings, self.examples)
                ]
            score, example = max(scored, key=lambda item: item[0])
            log_event("intent_embedding_match", {"score": score, "example": example})
            if score < EMBEDDING_ACCEPT_THRESHOLD:
                return None
            return _normalize_result({**example, "confidence": score}, source="embedding")
        except Exception as exc:
            log_event("intent_embedding_match", f"error: {exc}")
            return None

    def _classify_by_llm(self, payload: Dict[str, Any]) -> IntentResult:
        schema = {
            "primary_intent": sorted(ALLOWED_INTENTS),
            "intents": sorted(ALLOWED_INTENTS),
            "domains": sorted(ALLOWED_DOMAINS),
            "action": sorted(ALLOWED_ACTIONS),
            "confidence": "float from 0 to 1",
            "source": "llm",
            "matched_rule": "",
        }
        last_raw = ""
        last_error = ""
        total_start = time.perf_counter()
        llm_attempts: List[Dict[str, Any]] = []
        validation_attempts: List[Dict[str, Any]] = []
        for attempt in range(2):
            llm_recorded = False
            try:
                llm_start = time.perf_counter()
                with timed_event("intent.llm_request", {"attempt": attempt + 1}):
                    raw = self._call_llm(payload, schema, last_raw, last_error)
                llm_attempts.append(
                    {
                        "attempt": attempt + 1,
                        "elapsed_ms": round((time.perf_counter() - llm_start) * 1000, 2),
                        "status": "ok",
                    }
                )
                llm_recorded = True
                last_raw = raw
                log_full_output("intent_llm_raw", raw)
                validation_start = time.perf_counter()
                try:
                    with timed_event("intent.llm_validate", {"attempt": attempt + 1}):
                        parsed = json.loads(self._extract_json_text(raw))
                        missing = [
                            key
                            for key in ("primary_intent", "intents", "domains", "action", "confidence")
                            if key not in parsed
                        ]
                        if missing:
                            raise ValueError(f"missing required fields: {missing}")
                        if parsed.get("source", "llm") != "llm":
                            raise ValueError(f"invalid source for llm result: {parsed.get('source')}")
                        result = _normalize_result({**parsed, "source": "llm"}, source="llm")
                    validation_attempts.append(
                        {
                            "attempt": attempt + 1,
                            "elapsed_ms": round((time.perf_counter() - validation_start) * 1000, 2),
                            "status": "ok",
                        }
                    )
                except Exception as exc:
                    validation_attempts.append(
                        {
                            "attempt": attempt + 1,
                            "elapsed_ms": round((time.perf_counter() - validation_start) * 1000, 2),
                            "status": "error",
                            "error": repr(exc),
                        }
                    )
                    raise
                log_event(
                    "intent_llm_summary",
                    {
                        "status": "ok",
                        "total_elapsed_ms": round((time.perf_counter() - total_start) * 1000, 2),
                        "max_attempts": 2,
                        "attempt_count": attempt + 1,
                        "retry_count": attempt,
                        "llm_attempts": llm_attempts,
                        "validation_attempts": validation_attempts,
                        "error": "",
                    },
                )
                return result
            except Exception as exc:
                if not llm_recorded:
                    llm_attempts.append(
                        {
                            "attempt": attempt + 1,
                            "elapsed_ms": round((time.perf_counter() - llm_start) * 1000, 2),
                            "status": "error",
                            "error": repr(exc),
                        }
                    )
                last_error = str(exc)
                log_event("intent_validation_error", {"attempt": attempt + 1, "error": last_error, "raw": last_raw})
        log_event(
            "intent_llm_summary",
            {
                "status": "fallback",
                "total_elapsed_ms": round((time.perf_counter() - total_start) * 1000, 2),
                "max_attempts": 2,
                "attempt_count": len(llm_attempts),
                "retry_count": max(0, len(llm_attempts) - 1),
                "llm_attempts": llm_attempts,
                "validation_attempts": validation_attempts,
                "error": last_error,
            },
        )
        return fallback_unclear()

    def _call_llm(self, payload: Dict[str, Any], schema: Dict[str, Any], previous_raw: str, previous_error: str) -> str:
        client = get_cheap_openai_client()
        model = get_cheap_openai_model()
        examples = [
            {
                "user_message": "计划挺好的，但是酒店换便宜点",
                "output": {
                    "primary_intent": "modify",
                    "intents": ["satisfied", "modify"],
                    "domains": ["hotels", "budget"],
                    "action": "replace",
                    "confidence": 0.92,
                    "source": "llm",
                    "matched_rule": "",
                },
            },
            {
                "user_message": "第二天去哪？顺便多加一个景点",
                "output": {
                    "primary_intent": "modify",
                    "intents": ["question", "modify"],
                    "domains": ["schedule", "attractions"],
                    "action": "add",
                    "confidence": 0.9,
                    "source": "llm",
                    "matched_rule": "",
                },
            },
            {
                "user_message": "你怎么看股市？另外第二天换成博物馆",
                "output": {
                    "primary_intent": "modify",
                    "intents": ["refuse", "modify"],
                    "domains": ["schedule", "attractions"],
                    "action": "replace",
                    "confidence": 0.88,
                    "source": "llm",
                    "matched_rule": "",
                },
            },
            {
                "user_message": "帮我分析股票",
                "output": {
                    "primary_intent": "refuse",
                    "intents": ["refuse"],
                    "domains": ["none"],
                    "action": "refuse",
                    "confidence": 0.95,
                    "source": "llm",
                    "matched_rule": "",
                },
            },
        ]
        repair_note = ""
        if previous_raw or previous_error:
            repair_note = (
                "\nThe previous output failed validation. Return a corrected JSON object only.\n"
                f"Validation error: {previous_error}\n"
                f"Previous output: {previous_raw}\n"
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是旅游助手的意图识别器，只做分类，不生成旅行计划。\n"
                    "必须只返回一个 JSON 对象，不要 Markdown，不要解释。\n"
                    "所有字段只能从给定枚举中选择；source 必须是 llm，matched_rule 必须是空字符串。\n"
                    "如果一句话有多个意图，填入 intents，并按 replan > modify > question > satisfied > chitchat > unclear 选择 primary_intent。\n"
                    "refuse 特殊处理：如果同时有旅游相关修改，primary_intent 选择旅游相关意图；如果只有非旅游问题，primary_intent=refuse。\n"
                    f"Allowed schema/enums: {json.dumps(schema, ensure_ascii=False)}\n"
                    f"Few-shot examples: {json.dumps(examples, ensure_ascii=False)}"
                    f"{repair_note}"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    def _extract_json_text(self, text: str) -> str:
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        if "{" in text and "}" in text:
            return text[text.find("{"):text.rfind("}") + 1]
        return text


_intent_classifier: IntentClassifier | None = None


def get_intent_classifier() -> IntentClassifier:
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier
