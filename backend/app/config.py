"""Application configuration."""
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings


load_dotenv()

helloagents_env = Path(__file__).parent.parent.parent.parent / "HelloAgents" / ".env"
if helloagents_env.exists():
    load_dotenv(helloagents_env, override=False)


class Settings(BaseSettings):
    app_name: str = "HelloAgents智能旅行助手"
    app_version: str = "1.0.0"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"

    amap_api_key: str = ""
    amap_route_max_workers: int = 1
    amap_route_min_interval_seconds: float = 0.35
    amap_route_rate_limit_backoff_seconds: float = 1.2
    amap_route_max_retries: int = 2
    amap_route_mcp_fallback_enabled: bool = False
    hotel_min_candidates: int = 6
    hotel_ideal_distance_to_main_cluster_m: int = 8000
    hotel_hard_distance_to_main_cluster_m: int = 15000
    hotel_supplement_enabled: bool = True
    hotel_max_supplement_queries: int = 4
    tuniu_hotel_limit: int = 20
    tuniu_hotel_supplement_enabled: bool = True
    tuniu_hotel_max_supplement_queries: int = 5
    tuniu_api_key: str = ""
    tuniu_member_key: str = ""
    tuniu_mcp_url: str = "https://openapi.tuniu.cn/mcp/hotel"
    tuniu_flight_mcp_url: str = "https://openapi.tuniu.cn/mcp/flight"
    tuniu_train_mcp_url: str = "https://openapi.tuniu.cn/mcp/train"
    unsplash_access_key: str = ""
    unsplash_secret_key: str = ""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4"
    llm_context_window_tokens: int = 128000
    llm_max_output_tokens: int = 8192

    cheap_model: str = ""
    cheap_model_base_url: str = ""
    cheap_model_api_key: str = ""
    cheap_model_max_output_tokens: int = 1200
    context_recent_raw_min_tokens: int = 10000
    context_recent_raw_max_tokens: int = 40000
    context_heavy_summary_workers: int = 4

    log_level: str = "INFO"
    log_verbose_agent_output: bool = False
    log_verbose_agent_output_to_file: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on", "debug", "dev"}:
                return True
            if lowered in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False
        return False

    def get_cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()


def get_settings() -> Settings:
    return settings


def validate_config() -> bool:
    errors = []
    warnings = []

    if not settings.amap_api_key:
        errors.append("AMAP_API_KEY未配置")

    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or settings.openai_api_key
    if not llm_api_key:
        warnings.append("LLM_API_KEY或OPENAI_API_KEY未配置, LLM功能可能无法使用")

    if not (os.getenv("CHEAP_MODEL") or settings.cheap_model):
        warnings.append("CHEAP_MODEL未配置, 用户画像Agent将降级到主模型配置")

    if errors:
        raise ValueError("配置错误:\n" + "\n".join(f"  - {item}" for item in errors))

    if warnings:
        print("\n⚠️  配置警告:")
        for item in warnings:
            print(f"  - {item}")

    return True


def print_config() -> None:
    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or settings.openai_api_key
    llm_base_url = os.getenv("LLM_BASE_URL") or settings.openai_base_url
    llm_model = os.getenv("LLM_MODEL_ID") or settings.openai_model
    cheap_model = os.getenv("CHEAP_MODEL") or settings.cheap_model

    print(f"应用名称: {settings.app_name}")
    print(f"版本: {settings.app_version}")
    print(f"服务器: {settings.host}:{settings.port}")
    print(f"高德地图API Key: {'已配置' if settings.amap_api_key else '未配置'}")
    print(f"途牛酒店API Key: {'已配置' if settings.tuniu_api_key else '未配置,酒店将回退高德'}")
    print(f"LLM API Key: {'已配置' if llm_api_key else '未配置'}")
    print(f"LLM Base URL: {llm_base_url}")
    print(f"LLM Model: {llm_model}")
    print(f"LLM Max Output Tokens: {settings.llm_max_output_tokens}")
    print(f"LLM Context Window Tokens: {settings.llm_context_window_tokens}")
    print(f"CHEAP_MODEL: {cheap_model if cheap_model else '未配置,将降级'}")
    print(f"日志级别: {settings.log_level}")
