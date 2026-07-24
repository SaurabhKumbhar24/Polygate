from typing import Any, Dict, List, Optional

import requests

from ..retry import KeyInput, Retry, execute, resolve_keys
from ..types import ChatResponse, Usage

API_URL = "https://api.openai.com/v1/chat/completions"
ENV_VAR = "OPENAI_API_KEY"


def chat(
    *,
    model: str,
    messages: List[Dict[str, str]],
    api_key: KeyInput = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    retry: Optional[Retry] = None,
    **kwargs: Any,
) -> ChatResponse:
    pool = resolve_keys(api_key, ENV_VAR, "openai")

    payload: Dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    payload.update(kwargs)

    def _attempt(key: str):
        return requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )

    response = execute("openai", pool, _attempt, retry)

    data = response.json()
    choice = data["choices"][0]
    usage_raw = data.get("usage", {})
    usage = Usage(
        prompt_tokens=usage_raw.get("prompt_tokens", 0),
        completion_tokens=usage_raw.get("completion_tokens", 0),
        total_tokens=usage_raw.get("total_tokens", 0),
    )
    return ChatResponse(
        content=choice["message"]["content"],
        role=choice["message"].get("role", "assistant"),
        model=data.get("model", model),
        provider="openai",
        usage=usage,
        raw=data,
    )
