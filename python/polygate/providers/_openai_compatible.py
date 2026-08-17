"""Shared adapter for providers that speak the OpenAI chat-completions wire format.

Most of the market does. OpenAI defined the shape, and Groq, Together, Fireworks,
Mistral, xAI, Perplexity, Cerebras, DeepSeek, Moonshot, Z.AI, and every local
runtime with an OpenAI-compatible mode all accept the same request and return the
same response. Only three things differ between them: the URL, the environment
variable the key lives in, and the name reported back on the response.

So this module holds the one implementation and each provider file is a few
lines of configuration. The alternative — a copy of the same forty lines per
provider — was already in place for two of them, and would have become twelve
places to fix the next time a response field moved.

Custom endpoints
----------------
``base_url`` makes the same adapter reach anything that speaks the format:
a self-hosted vLLM, an Ollama server, a corporate gateway. That is the whole
implementation of the "bring your own endpoint" case — no separate provider, no
separate code path.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import requests

from ..exceptions import MissingEndpointError
from ..retry import KeyInput, Retry, execute, resolve_keys
from ..types import ChatResponse, Usage

#: Requests can legitimately take a while on large generations; the cap exists
#: so a hung connection surfaces rather than pinning a worker forever.
TIMEOUT_SECONDS = 120


def chat(
    *,
    provider: str,
    api_url: str,
    env_var: str,
    model: str,
    messages: List[Dict[str, str]],
    api_key: KeyInput = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    retry: Optional[Retry] = None,
    require_key: bool = True,
    extra_headers: Optional[Dict[str, str]] = None,
    auth: Optional[Callable[[str], Dict[str, str]]] = None,
    **kwargs: Any,
) -> ChatResponse:
    """One request against any OpenAI-compatible chat-completions endpoint.

    ``require_key=False`` is for local runtimes that authenticate nothing —
    demanding a key there would make the common case (a laptop running Ollama)
    fail for a reason that does not exist.

    ``auth`` builds the credential headers, defaulting to ``Authorization:
    Bearer``. It is a hook because "OpenAI-compatible" describes the body, not
    the handshake: Azure wants the key in its own ``api-key`` header, and a
    Bearer token there returns a 401 indistinguishable from a wrong key.
    """
    if require_key:
        pool = resolve_keys(api_key, env_var, provider)
    else:
        # A placeholder key keeps one code path through execute(); local servers
        # ignore the header entirely.
        pool = resolve_keys(api_key or "local", env_var, provider)

    payload: Dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    payload.update(kwargs)

    build_auth = auth or bearer_auth

    def _attempt(key: str):
        headers = {"Content-Type": "application/json"}
        headers.update(build_auth(key))
        if extra_headers:
            headers.update(extra_headers)
        return requests.post(api_url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)

    response = execute(provider, pool, _attempt, retry)
    return parse(response.json(), provider=provider, fallback_model=model)


def parse(data: Dict[str, Any], *, provider: str, fallback_model: str) -> ChatResponse:
    """Normalize an OpenAI-shaped body into a ChatResponse.

    Written defensively because "OpenAI-compatible" is a claim, not a standard:
    several providers omit ``usage`` on some routes, and at least one returns an
    empty ``content`` with the text in ``reasoning_content``. A missing field
    should cost the caller a zero in a token count, never an exception on a
    response that actually arrived.
    """
    choices = data.get("choices") or [{}]
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        # Reasoning models on several providers put the answer here when the
        # visible content is empty.
        content = message.get("reasoning_content") or ""

    usage_raw = data.get("usage") or {}
    usage = Usage(
        prompt_tokens=usage_raw.get("prompt_tokens", 0) or 0,
        completion_tokens=usage_raw.get("completion_tokens", 0) or 0,
        total_tokens=usage_raw.get("total_tokens", 0) or 0,
    )
    return ChatResponse(
        content=content,
        role=message.get("role", "assistant"),
        model=data.get("model", fallback_model),
        provider=provider,
        usage=usage,
        raw=data,
    )


def bearer_auth(key: str) -> Dict[str, str]:
    """The near-universal case: ``Authorization: Bearer <key>``."""
    return {"Authorization": f"Bearer {key}"}


def api_key_header_auth(header: str = "api-key") -> Callable[[str], Dict[str, str]]:
    """Auth for hosts that carry the key in a bare header, not a Bearer token.

    Azure OpenAI is the one that matters here.
    """
    def _auth(key: str) -> Dict[str, str]:
        return {header: key}
    return _auth


def make_adapter(
    *,
    provider: str,
    api_url: Optional[str] = None,
    env_var: str,
    require_key: bool = True,
    auth: Optional[Callable[[str], Dict[str, str]]] = None,
    url_builder: Optional[Callable[[str, str], str]] = None,
    base_url_env: Optional[str] = None,
) -> Callable[..., ChatResponse]:
    """Build a provider module's ``chat`` from its distinguishing facts.

    ``auth`` and ``url_builder`` exist for hosts that speak the OpenAI body
    format but not its conventions — Azure puts the deployment in the path and
    the key in its own header.

    ``base_url_env`` is for providers with no fixed host, because the URL
    contains your own workspace, account or project — Databricks, Cloudflare,
    Vertex, Azure. Each gets its **own** variable rather than sharing one: a
    single global would mean pointing it at a local Ollama silently redirected
    every other endpoint-hungry provider too. Omitting the endpoint raises an
    error naming what to pass, which beats a request to ``None`` failing
    somewhere far less informative.
    """

    def _chat(
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
        # A caller-supplied base_url wins, so a proxy or a regional endpoint
        # needs no new provider.
        root = base_url or (os.environ.get(base_url_env) if base_url_env else None) or api_url
        if not root:
            raise MissingEndpointError(
                f"{provider} has no fixed host — its URL contains your own "
                f"account, workspace or project. Pass base_url=..."
                + (f" or set ${base_url_env}." if base_url_env else ".")
            )
        target = url_builder(root, model) if url_builder else _resolve_url(root)
        return chat(
            provider=provider,
            api_url=target,
            env_var=env_var,
            model=model,
            messages=messages,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            retry=retry,
            require_key=require_key,
            auth=auth,
            **kwargs,
        )

    _chat.__name__ = "chat"
    _chat.__doc__ = (
        f"Chat via {provider} ({api_url or 'an endpoint you supply'}).\n\n"
        f"Key resolution falls back to ${env_var}. Pass ``base_url`` to point the "
        f"same wire format at a different host."
    )
    #: True when this provider cannot be called without an explicit endpoint.
    #: Read by UIs that need to prompt for one before offering the provider.
    _chat.endpoint_required = api_url is None  # type: ignore[attr-defined]
    return _chat


def _resolve_url(base_url: str) -> str:
    """Accept either a bare host root or a full completions URL.

    People pass both, and silently posting to the wrong path produces a 404 that
    reads like a broken provider rather than a mistyped setting.
    """
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


__all__ = [
    "chat", "parse", "make_adapter", "bearer_auth",
    "api_key_header_auth", "TIMEOUT_SECONDS",
]
