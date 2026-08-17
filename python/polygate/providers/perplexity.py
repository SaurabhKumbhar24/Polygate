"""Perplexity's Sonar models. The host has no /v1 segment.

These models search the web as part of answering, so a response can cite
sources in ``raw`` that no other provider returns.

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

API_URL = "https://api.perplexity.ai/chat/completions"
ENV_VAR = "PERPLEXITY_API_KEY"

chat = make_adapter(provider="perplexity", api_url=API_URL, env_var=ENV_VAR)
