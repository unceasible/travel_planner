# -*- coding: utf-8 -*-
from app.services.intent_classifier import IntentClassifier
from app.services.task_executor import TripTaskExecutor


def test_intercity_followup_rules_route_to_replan_domain():
    classifier = IntentClassifier()

    for message in [
        "从上海出发",
        "改成坐火车",
        "帮我查机票",
        "看看航班",
        "查一下高铁",
        "这次自驾过去",
    ]:
        result = classifier._classify_by_rules(message)

        assert result is not None, message
        assert result.primary_intent == "replan", message
        assert "intercity_transport" in result.domains, message


def test_patch_domain_keywords_cover_intercity_transport_phrases():
    executor = TripTaskExecutor.__new__(TripTaskExecutor)

    for message in ["帮我查机票", "查一下高铁", "开车过去", "自驾往返"]:
        assert "intercity_transport" in executor._select_patch_domains(message)


def test_intercity_self_drive_does_not_overwrite_local_transportation():
    executor = TripTaskExecutor.__new__(TripTaskExecutor)
    form_snapshot = {
        "departure_city": "北京",
        "city": "天津",
        "intercity_transportation": "智能推荐",
        "transportation": "公共交通",
        "free_text_input": "",
    }

    merged = executor._merge_message_into_form(form_snapshot, "改成自驾过去")

    assert merged["intercity_transportation"] == "自驾"
    assert merged["transportation"] == "公共交通"


def test_local_transportation_change_still_updates_local_field():
    executor = TripTaskExecutor.__new__(TripTaskExecutor)
    form_snapshot = {
        "departure_city": "北京",
        "city": "天津",
        "intercity_transportation": "智能推荐",
        "transportation": "公共交通",
        "free_text_input": "",
    }

    merged = executor._merge_message_into_form(form_snapshot, "交通方式改成自驾")

    assert merged["intercity_transportation"] == "智能推荐"
    assert merged["transportation"] == "自驾"
