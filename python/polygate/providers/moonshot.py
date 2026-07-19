import os
from typing import Any, Dict, List, Optional

import requests

from ..exceptions import MissingAPIKeyError, ProviderAPIError
from ..types import ChatResponse, Usage

# Moonshot's API (used to access Kimi K2 and other Moonshot models) is
# OpenAI-compatible, so the request/response shape mirrors OpenAI's.
API_URL = "https://api.moonshot.ai/v1/chat/completions"
ENV_VAR = "MOONSHOT_API_KEY"


def chat(
    *,
    model: str,
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> ChatResponse:
    key = api_key or os.environ.get(ENV_VAR)
    if not key:
        raise MissingAPIKeyError(
            f"No Moonshot API key provided. Pass api_key= or set {ENV_VAR}."
        )

    payload: Dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    payload.update(kwargs)

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    if response.status_code != 200:
        raise ProviderAPIError("moonshot", response.status_code, response.text)

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
        provider="moonshot",
        usage=usage,
        raw=data,
    )
