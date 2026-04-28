# -*- coding: utf-8 -*-
from types import SimpleNamespace

from app.agents.reflection_agent import ReflectionReview
from app.models.schemas import (
    Attraction,
    DayPlan,
    Location,
    Meal,
    TripChatRequest,
    TripPlan,
)
from app.services.intent_classifier import IntentResult
from app.services.task_executor import TripTaskExecutor


def _attraction(name: str = "Museum") -> Attraction:
    return Attraction(
        name=name,
        address="Test address",
        location=Location(longitude=116.4, latitude=39.9),
        visit_duration=90,
        description="Indoor culture stop",
        ticket_price=0,
    )


def _meal(meal_type: str) -> Meal:
    return Meal(type=meal_type, name=f"{meal_type} place", estimated_cost=30)


def _plan(meals=None, attraction_name: str = "Museum") -> TripPlan:
    return TripPlan(
        city="北京",
        start_date="2026-05-01",
        end_date="2026-05-01",
        days=[
            DayPlan(
                date="2026-05-01",
                day_index=0,
                description="A focused day",
                transportation="公共交通",
                accommodation="舒适型酒店",
                attractions=[_attraction(attraction_name)],
                meals=list(meals) if meals is not None else [_meal("breakfast"), _meal("lunch"), _meal("dinner")],
            )
        ],
        weather_info=[],
        overall_suggestions="Keep the route compact.",
    )


class _FakePlanner:
    def __init__(self):
        self.revise_plan_feedback = []
        self.revise_attractions_only_calls = 0
        self.revise_plan_calls = 0

    def create_fallback_plan(self, form_snapshot):
        return _plan(attraction_name="Fallback")

    def revise_attractions_only(self, **kwargs):
        self.revise_attractions_only_calls += 1
        return _plan(attraction_name="Attractions-only plan")

    def revise_plan(self, **kwargs):
        self.revise_plan_calls += 1
        self.revise_plan_feedback.append(kwargs.get("reflection_feedback", ""))
        return _plan(attraction_name="General revised plan")


class _FakeReflectionAgent:
    def __init__(self, reviews=None, error=None):
        self.reviews = list(reviews or [])
        self.error = error
        self.calls = []

    def review_plan(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.reviews.pop(0)


class _FakeMemoryStore:
    def __init__(self, task_record):
        self.task_record = task_record
        self.written_task = None

    def read_task(self, task_id):
        return self.task_record

    def load_or_create_user_memory(self, user_id, nickname):
        return {"profile": {}}

    def write_task(self, task_record):
        self.written_task = task_record


def _executor(reflection_agent=None, planner=None):
    executor = TripTaskExecutor.__new__(TripTaskExecutor)
    executor.planner = planner or _FakePlanner()
    executor.reflection_agent = reflection_agent or _FakeReflectionAgent(
        [ReflectionReview(score=8, status="pass", issues=[], improvement_instructions="", summary="Good")]
    )
    executor._add_transport_segments_parallel = lambda plan, transportation: plan
    return executor


def test_quality_gate_does_not_retry_when_score_passes():
    planner = _FakePlanner()
    executor = _executor(
        planner=planner,
        reflection_agent=_FakeReflectionAgent(
            [ReflectionReview(score=8, status="pass", issues=[], improvement_instructions="", summary="Good")]
        ),
    )

    plan, entries = executor._run_quality_gate(
        _plan(),
        {"city": "北京", "start_date": "2026-05-01", "travel_days": 1, "accommodation": "舒适型酒店"},
        user_profile_summary="{}",
        conversation_history=[],
        user_message="",
        update_mode="initial",
        retry_once=lambda feedback: planner.revise_plan(reflection_feedback=feedback),
    )

    assert plan.days[0].attractions[0].name == "Museum"
    assert planner.revise_plan_calls == 0
    assert [entry["phase"] for entry in entries] == ["format_fix", "quality_review"]
    assert entries[-1]["score"] == 8
    assert entries[-1]["retry_used"] is False


def test_quality_gate_retries_once_when_score_is_low_and_passes_feedback_to_planner():
    planner = _FakePlanner()
    executor = _executor(
        planner=planner,
        reflection_agent=_FakeReflectionAgent(
            [
                ReflectionReview(
                    score=6,
                    status="needs_replan",
                    issues=[{"type": "route_too_far", "message": "第1天路线过远"}],
                    improvement_instructions="压缩第1天路线，减少跨区移动。",
                    summary="Needs route repair",
                ),
                ReflectionReview(score=8, status="pass", issues=[], improvement_instructions="", summary="Better"),
            ]
        ),
    )

    plan, entries = executor._run_quality_gate(
        _plan(),
        {"city": "北京", "start_date": "2026-05-01", "travel_days": 1, "accommodation": "舒适型酒店"},
        user_profile_summary="{}",
        conversation_history=[],
        user_message="",
        update_mode="initial",
        retry_once=lambda feedback: planner.revise_plan(reflection_feedback=feedback),
    )

    assert plan.days[0].attractions[0].name == "General revised plan"
    assert planner.revise_plan_calls == 1
    assert "第1天路线过远" in planner.revise_plan_feedback[0]
    quality_entries = [entry for entry in entries if entry["phase"] == "quality_review"]
    assert [entry["retry_used"] for entry in quality_entries] == [False, True]
    assert [entry["score"] for entry in quality_entries] == [6, 8]


def test_quality_gate_retries_at_most_once_when_second_review_is_still_low():
    planner = _FakePlanner()
    executor = _executor(
        planner=planner,
        reflection_agent=_FakeReflectionAgent(
            [
                ReflectionReview(score=5, status="needs_replan", issues=[], improvement_instructions="重修", summary="Bad"),
                ReflectionReview(score=6, status="needs_replan", issues=[], improvement_instructions="仍需重修", summary="Still bad"),
            ]
        ),
    )

    plan, entries = executor._run_quality_gate(
        _plan(),
        {"city": "北京", "start_date": "2026-05-01", "travel_days": 1, "accommodation": "舒适型酒店"},
        user_profile_summary="{}",
        conversation_history=[],
        user_message="",
        update_mode="initial",
        retry_once=lambda feedback: planner.revise_plan(reflection_feedback=feedback),
    )

    assert plan.days[0].attractions[0].name == "General revised plan"
    assert planner.revise_plan_calls == 1
    quality_entries = [entry for entry in entries if entry["phase"] == "quality_review"]
    assert len(quality_entries) == 2
    assert quality_entries[-1]["score"] == 6
    assert quality_entries[-1]["retry_used"] is True


def test_quality_gate_records_review_error_without_retrying():
    planner = _FakePlanner()
    executor = _executor(
        planner=planner,
        reflection_agent=_FakeReflectionAgent(error=RuntimeError("model unavailable")),
    )

    plan, entries = executor._run_quality_gate(
        _plan(),
        {"city": "北京", "start_date": "2026-05-01", "travel_days": 1, "accommodation": "舒适型酒店"},
        user_profile_summary="{}",
        conversation_history=[],
        user_message="",
        update_mode="initial",
        retry_once=lambda feedback: planner.revise_plan(reflection_feedback=feedback),
    )

    assert plan.days[0].attractions[0].name == "Museum"
    assert planner.revise_plan_calls == 0
    assert entries[-1]["status"] == "review_error"
    assert "model unavailable" in entries[-1]["summary"]


def test_quality_gate_keeps_existing_format_autofill_behavior():
    executor = _executor(
        reflection_agent=_FakeReflectionAgent(
            [ReflectionReview(score=8, status="pass", issues=[], improvement_instructions="", summary="Good")]
        )
    )

    plan, entries = executor._run_quality_gate(
        _plan(meals=[]),
        {"city": "北京", "start_date": "2026-05-01", "travel_days": 1, "accommodation": "舒适型酒店"},
        user_profile_summary="{}",
        conversation_history=[],
        user_message="",
        update_mode="initial",
        retry_once=None,
    )

    assert {meal.type for meal in plan.days[0].meals} == {"breakfast", "lunch", "dinner"}
    assert entries[0]["phase"] == "format_fix"
    assert "day_0_missing_breakfast" in entries[0]["notes"]


def test_quality_gate_removes_self_drive_suggestion_when_not_selected():
    executor = _executor(
        reflection_agent=_FakeReflectionAgent(
            [ReflectionReview(score=8, status="pass", issues=[], improvement_instructions="", summary="Good")]
        )
    )
    plan = _plan()
    plan.overall_suggestions = "建议乘坐公共交通游览。也可以自驾前往景点，开车更灵活。"

    result, entries = executor._run_quality_gate(
        plan,
        {
            "city": "北京",
            "start_date": "2026-05-01",
            "travel_days": 1,
            "accommodation": "舒适型酒店",
            "transportation": "公共交通",
            "intercity_transportation": "火车",
        },
        user_profile_summary='{"preferences":["自驾"]}',
        conversation_history=[],
        user_message="",
        update_mode="initial",
        retry_once=None,
    )

    assert "公共交通" in result.overall_suggestions
    assert "自驾" not in result.overall_suggestions
    assert "开车" not in result.overall_suggestions
    assert any(entry.get("phase") == "transport_preference_guard" for entry in entries)


def test_chat_attractions_only_low_score_retries_with_general_revision():
    planner = _FakePlanner()
    task_record = {
        "task_id": "task1",
        "user_id": "user1",
        "nickname": "Alice",
        "form_snapshot": {
            "city": "北京",
            "start_date": "2026-05-01",
            "end_date": "2026-05-01",
            "travel_days": 1,
            "transportation": "公共交通",
            "accommodation": "舒适型酒店",
            "preferences": ["博物馆"],
        },
        "current_plan": _plan().model_dump(),
        "conversation_log": [],
    }
    executor = _executor(
        planner=planner,
        reflection_agent=_FakeReflectionAgent(
            [
                ReflectionReview(
                    score=6,
                    status="needs_replan",
                    issues=[{"type": "preference_miss", "message": "没有满足博物馆偏好"}],
                    improvement_instructions="补充更符合博物馆偏好的安排。",
                    summary="Bad patch",
                ),
                ReflectionReview(score=8, status="pass", issues=[], improvement_instructions="", summary="Better"),
            ]
        ),
    )
    executor.memory_store = _FakeMemoryStore(task_record)
    executor.intent_classifier = SimpleNamespace(
        classify=lambda message, context: IntentResult(
            primary_intent="modify",
            intents=["modify"],
            domains=["attractions"],
            action="replace",
            confidence=0.95,
            source="rule",
            matched_rule="replace_attractions",
        )
    )
    executor.context_compressor = SimpleNamespace(
        prepare_context=lambda task: SimpleNamespace(history=[], mode="raw"),
        needs_heavy_refresh=lambda task: False,
    )
    executor._start_user_profile_update = lambda context: None
    executor._start_context_compression_if_needed = lambda task: None
    executor._fetch_patch_context_parallel = lambda message, form, domains: {
        "attractions": "",
        "weather": "",
        "hotels": "",
        "restaurants": "",
    }

    response = executor.chat(TripChatRequest(task_id="task1", user_message="把景点换成博物馆"))

    assert response.success is True
    assert planner.revise_attractions_only_calls == 1
    assert planner.revise_plan_calls == 1
    assert "没有满足博物馆偏好" in planner.revise_plan_feedback[0]
    assert response.data.days[0].attractions[0].name == "General revised plan"
