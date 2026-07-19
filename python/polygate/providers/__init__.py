from ..exceptions import UnsupportedProviderError
from . import anthropic, gemini, moonshot, openai

# Multiple aliases map to the same adapter so callers can use whichever
# name feels natural (e.g. "claude" or "anthropic", "kimi" or "moonshot").
_REGISTRY = {
    "anthropic": anthropic,
    "claude": anthropic,
    "openai": openai,
    "gpt": openai,
    "gemini": gemini,
    "google": gemini,
    "moonshot": moonshot,
    "kimi": moonshot,
}

SUPPORTED_PROVIDERS = sorted(_REGISTRY.keys())


def get_provider(name: str):
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise UnsupportedProviderError(
            f"Unknown provider '{name}'. Supported: {SUPPORTED_PROVIDERS}"
        )
    return _REGISTRY[key]
