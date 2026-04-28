# -*- coding: utf-8 -*-
from types import SimpleNamespace

from app.services import task_executor as task_executor_module


class _FakeUserProfileAgent:
    def __init__(self, memory_store):
        self.memory_store = memory_store


class _FakeContextCompressor:
    def __init__(self, memory_store):
        self.memory_store = memory_store


def test_trip_task_executor_uses_configured_route_workers(monkeypatch):
    monkeypatch.setattr(task_executor_module, "get_memory_store", lambda: object())
    monkeypatch.setattr(task_executor_module, "get_trip_planner_agent", lambda: object())
    monkeypatch.setattr(task_executor_module, "UserProfileAgent", _FakeUserProfileAgent)
    monkeypatch.setattr(task_executor_module, "ReflectionAgent", lambda: object())
    monkeypatch.setattr(task_executor_module, "get_intercity_transport_agent", lambda: object())
    monkeypatch.setattr(task_executor_module, "get_intent_classifier", lambda: object())
    monkeypatch.setattr(task_executor_module, "ConversationContextCompressor", _FakeContextCompressor)
    monkeypatch.setattr(
        task_executor_module,
        "get_settings",
        lambda: SimpleNamespace(amap_route_max_workers=2),
        raising=False,
    )

    executor = task_executor_module.TripTaskExecutor()
    try:
        assert executor.route_executor._max_workers == 2
    finally:
        executor.retrieval_executor.shutdown(wait=False, cancel_futures=True)
        executor.transport_executor.shutdown(wait=False, cancel_futures=True)
        executor.route_executor.shutdown(wait=False, cancel_futures=True)
        executor.profile_executor.shutdown(wait=False, cancel_futures=True)
        executor.context_executor.shutdown(wait=False, cancel_futures=True)
