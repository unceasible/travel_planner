"""LLM service helpers for HelloAgents and OpenAI-compatible clients."""
# -*- coding: utf-8 -*-
import os

from hello_agents import HelloAgentsLLM
from openai import OpenAI

from ..config import get_settings

_llm_instance = None
_cheap_llm_instance = None
_openai_client_instance = None
_cheap_openai_client_instance = None


def get_llm() -> HelloAgentsLLM:
    """Return the main HelloAgents LLM instance."""
    global _llm_instance

    if _llm_instance is None:
        _llm_instance = HelloAgentsLLM()
        print("✅ LLM服务初始化成功")
        print(f"   提供商: {_llm_instance.provider}")
        print(f"   模型: {_llm_instance.model}")

    return _llm_instance


def get_cheap_llm() -> HelloAgentsLLM:
    """Return the cheap HelloAgents LLM instance for background profile updates."""
    global _cheap_llm_instance

    if _cheap_llm_instance is None:
        settings = get_settings()
        model = os.getenv("CHEAP_MODEL") or settings.cheap_model
        api_key = os.getenv("CHEAP_MODEL_API_KEY") or settings.cheap_model_api_key
        base_url = os.getenv("CHEAP_MODEL_BASE_URL") or settings.cheap_model_base_url

        if not model:
            model = os.getenv("LLM_MODEL_ID") or settings.openai_model
            api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or settings.openai_api_key
            base_url = base_url or os.getenv("LLM_BASE_URL") or settings.openai_base_url
            print("⚠️  CHEAP_MODEL未配置,UserProfileAgent将降级使用主模型配置")

        _cheap_llm_instance = HelloAgentsLLM(
            model=model,
            api_key=api_key or None,
            base_url=base_url or None,
            temperature=0.2,
            max_tokens=settings.cheap_model_max_output_tokens,
        )
        print("✅ 用户画像小模型初始化成功")
        print(f"   模型: {_cheap_llm_instance.model}")

    return _cheap_llm_instance


def get_openai_client() -> OpenAI:
    """Return the main OpenAI-compatible client for structured outputs."""
    global _openai_client_instance

    if _openai_client_instance is None:
        settings = get_settings()
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or settings.openai_api_key
        base_url = os.getenv("LLM_BASE_URL") or settings.openai_base_url
        _openai_client_instance = OpenAI(api_key=api_key, base_url=base_url)

    return _openai_client_instance


def get_cheap_openai_client() -> OpenAI:
    """Return the cheap OpenAI-compatible client for background profile work."""
    global _cheap_openai_client_instance

    if _cheap_openai_client_instance is None:
        settings = get_settings()
        api_key = os.getenv("CHEAP_MODEL_API_KEY") or settings.cheap_model_api_key
        base_url = os.getenv("CHEAP_MODEL_BASE_URL") or settings.cheap_model_base_url or settings.openai_base_url
        if not api_key:
            api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or settings.openai_api_key
        _cheap_openai_client_instance = OpenAI(api_key=api_key, base_url=base_url)

    return _cheap_openai_client_instance


def get_openai_model() -> str:
    settings = get_settings()
    return os.getenv("LLM_MODEL_ID") or settings.openai_model


def get_cheap_openai_model() -> str:
    settings = get_settings()
    return os.getenv("CHEAP_MODEL") or settings.cheap_model or get_openai_model()


def reset_llm() -> None:
    """Reset cached clients."""
    global _llm_instance, _cheap_llm_instance, _openai_client_instance, _cheap_openai_client_instance
    _llm_instance = None
    _cheap_llm_instance = None
    _openai_client_instance = None
    _cheap_openai_client_instance = None
