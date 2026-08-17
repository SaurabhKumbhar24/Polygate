"""Any OpenAI-compatible endpoint you host yourself.

Ollama, vLLM, LM Studio, llama.cpp's server, TGI, a corporate gateway, a
staging proxy — all of them speak the OpenAI chat-completions format, so none of
them needs its own adapter. What they need is somewhere to put the URL.

    polygate.chat(
        provider="custom",
        model="llama3.1:8b",
        base_url="http://localhost:11434/v1",
        messages=[{"role": "user", "content": "hello"}],
    )

Authentication is optional here, unlike every other provider. A local runtime
usually has none, and demanding a key would make the most common case — a laptop
running Ollama — fail for a reason that does not exist. Set ``POLYGATE_BASE_URL``
to skip passing ``base_url`` on every call, and ``CUSTOM_API_KEY`` when the
endpoint does want one.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ._openai_compatible import _resolve_url
from ._openai_compatible import chat as _compat_chat
from ..exceptions import UnsupportedProviderError
from ..retry import KeyInput, Retry
from ..types import ChatResponse

ENV_VAR = "CUSTOM_API_KEY"
BASE_URL_ENV = "POLYGATE_BASE_URL"


def chat(
    *,
    model: str,
    messages: List[Dict[str, str]],
    api_key: KeyInput = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    retry: Optional[Retry] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> ChatResponse:
    url = base_url or os.getenv(BASE_URL_ENV)
    if not url:
        # Raised rather than defaulted to localhost: guessing an endpoint sends
        # the caller's prompt somewhere they did not choose.
        raise UnsupportedProviderError(
            "provider='custom' needs an endpoint. Pass base_url='http://host:port/v1' "
            f"or set ${BASE_URL_ENV}."
        )
    return _compat_chat(
        provider="custom",
        api_url=_resolve_url(url),
        env_var=ENV_VAR,
        model=model,
        messages=messages,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        retry=retry,
        require_key=False,
        **kwargs,
    )
