from typing import Any, Dict, List, Optional

from .providers import get_provider
from .types import ChatResponse


def chat(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
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
        api_key: API key for the request. If omitted, falls back to the
            provider's standard environment variable (e.g.
            ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY,
            MOONSHOT_API_KEY).
        temperature: Optional sampling temperature.
        max_tokens: Optional max output tokens.
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
        **kwargs,
    )
