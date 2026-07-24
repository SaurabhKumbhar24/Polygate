from typing import Any, Dict, List, Optional

from .providers import get_provider
from .retry import KeyInput, Retry
from .types import ChatResponse


def chat(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    api_key: KeyInput = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    retry: Optional[Retry] = None,
    **kwargs: Any,
) -> ChatResponse:
    """
    Send a chat request to any supported LLM provider through one
    consistent interface.

    Args:
        provider: "anthropic"/"claude", "openai"/"gpt", "gemini"/"google",
            or "moonshot"/"kimi".
        model: The model name as expected by that provider's API
            (e.g. "claude-sonnet-4-6", "gpt-4o", "gemini-2.0-flash",
            "kimi-k2-0711-preview").
        messages: A list of {"role": ..., "content": ...} dicts using
            "system" | "user" | "assistant" roles. Adapters translate
            this into whatever shape the underlying provider expects.
        api_key: API key for the request. Accepts a single key, a list of
            keys, or a reusable ``KeyPool``. If omitted, falls back to the
            provider's standard environment variable (e.g.
            ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY,
            MOONSHOT_API_KEY), which may itself hold a comma-separated
            list, so a pool can be configured without touching code.

            With more than one key, requests rotate round-robin. A key the
            provider rejects (401/403) is dropped for the rest of the
            process; a rate-limited key (429) is parked for its
            ``Retry-After`` and traffic moves to another key immediately,
            which recovers faster than waiting out a backoff.
        temperature: Optional sampling temperature.
        max_tokens: Optional max output tokens.
        retry: Optional :class:`Retry` policy. Omitted means no retries:
            one request, one response, errors raised straight through, which
            is the behavior callers had before this parameter existed.
            ``Retry()`` opts into exponential backoff with jitter on
            transient failures (429, 5xx, 529, connection errors) while
            still failing fast on deterministic ones (400, 404, 422).
        **kwargs: Passed straight through to the provider's request
            payload, for anything not normalized here (e.g. tools,
            top_p, stop sequences).

    Returns:
        A normalized ChatResponse with .content, .role, .model,
        .provider, .usage, and .raw (the untouched provider response).
    """
    adapter = get_provider(provider)
    return adapter.chat(
        model=model,
        messages=messages,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        retry=retry,
        **kwargs,
    )
