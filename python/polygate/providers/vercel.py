"""Vercel AI Gateway — a routing layer in front of every major vendor.

Like OpenRouter, models are addressed ``vendor/model``. The environment variable
is ``AI_GATEWAY_API_KEY`` rather than a Vercel-branded name, matching what
Vercel's own tooling sets.

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

API_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
ENV_VAR = "AI_GATEWAY_API_KEY"

chat = make_adapter(provider="vercel", api_url=API_URL, env_var=ENV_VAR)
