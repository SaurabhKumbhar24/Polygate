import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..exceptions import MissingAPIKeyError, ProviderAPIError
from ..types import ChatResponse, Usage

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
ENV_VAR = "GEMINI_API_KEY"


def _to_gemini_contents(
    messages: List[Dict[str, str]]
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    contents = []
    system_instruction = None
    for m in messages:
        if m["role"] == "system":
            system_instruction = m["content"]
            continue
        # Gemini calls the assistant role "model" instead of "assistant".
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    return contents, system_instruction


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
            f"No Gemini API key provided. Pass api_key= or set {ENV_VAR}."
        )

    contents, system_instruction = _to_gemini_contents(messages)

    payload: Dict[str, Any] = {"contents": contents}
    generation_config: Dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
    if generation_config:
        payload["generationConfig"] = generation_config
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    payload.update(kwargs)

    url = f"{API_BASE}/{model}:generateContent?key={key}"
    response = requests.post(url, json=payload, timeout=120)
    if response.status_code != 200:
        raise ProviderAPIError("gemini", response.status_code, response.text)

    data = response.json()
    candidate = data["candidates"][0]
    content = "".join(part.get("text", "") for part in candidate["content"]["parts"])
    usage_raw = data.get("usageMetadata", {})
    usage = Usage(
        prompt_tokens=usage_raw.get("promptTokenCount", 0),
        completion_tokens=usage_raw.get("candidatesTokenCount", 0),
        total_tokens=usage_raw.get("totalTokenCount", 0),
    )
    return ChatResponse(
        content=content,
        role="assistant",
        model=model,
        provider="gemini",
        usage=usage,
        raw=data,
    )
