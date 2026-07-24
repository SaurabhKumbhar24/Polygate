from typing import Any, Dict, List, Optional

import requests

from ..retry import KeyInput, Retry, execute, resolve_keys
from ..types import ChatResponse, Usage

API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ENV_VAR = "ANTHROPIC_API_KEY"
DEFAULT_MAX_TOKENS = 1024


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
    pool = resolve_keys(api_key, ENV_VAR, "anthropic")

    # Anthropic takes "system" as a top-level field, not a message role.
    system = None
    formatted_messages = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            formatted_messages.append({"role": m["role"], "content": m["content"]})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": formatted_messages,
        "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
    }
    if system:
        payload["system"] = system
    if temperature is not None:
        payload["temperature"] = temperature
    payload.update(kwargs)

    def _attempt(key: str):
        return requests.post(
            API_URL,
            headers={
                "x-api-key": key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json=payload,
            timeout=120,
        )

    response = execute("anthropic", pool, _attempt, retry)
    data = response.json()
    content = "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    )
    usage_raw = data.get("usage", {})
    usage = Usage(
        prompt_tokens=usage_raw.get("input_tokens", 0),
        completion_tokens=usage_raw.get("output_tokens", 0),
        total_tokens=usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0),
    )
    return ChatResponse(
        content=content,
        role="assistant",
        model=data.get("model", model),
        provider="anthropic",
        usage=usage,
        raw=data,
    )
