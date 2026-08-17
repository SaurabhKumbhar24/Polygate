"""OpenRouter — one key, several hundred models across every major vendor.

A router rather than a host: you address models as ``vendor/model``
(``anthropic/claude-sonnet-4.5``), and OpenRouter picks an upstream provider and
fails over between them. That makes it the cheapest way to reach a model nobody
here has an adapter for.

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

API_URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_VAR = "OPENROUTER_API_KEY"

chat = make_adapter(provider="openrouter", api_url=API_URL, env_var=ENV_VAR)
