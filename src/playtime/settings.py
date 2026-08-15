"""All config in one place, loaded from the environment / .env.

The load-bearing constants (model names, token budgets, loop limits) live
HERE, named, not scattered as magic numbers. Later
stages will add context budgets to this class — watch it grow.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    litellm_base_url: str = "http://127.0.0.1:4000/v1"
    litellm_api_key: str = ""

    # Two ROLES, not two hardcoded models. "agent" is the brain (reasoning,
    # tool calls); "utility" does cheap grunt work (summarize, compact —
    # arrives in stage 4). Swapping either is a .env edit, not a code change.
    agent_model: str = "qwen-chat"
    utility_model: str = "qwen-chat"

    gen_max_tokens: int = 1024  # per-response generation cap


@lru_cache
def get_settings() -> Settings:
    return Settings()
