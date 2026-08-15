"""The ONLY model access path. Everything speaks OpenAI-chat-completions to the
LiteLLM proxy; the proxy decides what actually serves the alias (local vLLM,
cloud failover). No other file may import langchain_openai or know a base URL.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from playtime.settings import get_settings


def make_llm(role: str = "agent", **kw) -> ChatOpenAI:
    """role: "agent" (the brain) or "utility" (summarize/compact, stage 4)."""
    s = get_settings()
    model = s.agent_model if role == "agent" else s.utility_model
    extra_body: dict = {}
    if not model.startswith(("openai/", "anthropic/")):
        # Qwen emits reasoning traces by default; on tool-routing turns they
        # burn latency and generation budget for nothing. vLLM honors this
        # chat-template flag, cloud providers would reject it — hence the guard.
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    return ChatOpenAI(
        model=model,
        base_url=s.litellm_base_url,
        api_key=s.litellm_api_key,
        temperature=kw.pop("temperature", 0.1),  # tool calls must be precise
        max_tokens=kw.pop("max_tokens", s.gen_max_tokens),
        timeout=kw.pop("timeout", 120),
        extra_body=extra_body or None,
        **kw,
    )
