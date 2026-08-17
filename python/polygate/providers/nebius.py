"""Nebius AI Studio — open models from a European cloud.

The host is ``api.studio.nebius.com``, not ``api.nebius.com``; the latter is the
cloud control plane and will not serve inference.

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

API_URL = "https://api.studio.nebius.com/v1/chat/completions"
ENV_VAR = "NEBIUS_API_KEY"

chat = make_adapter(provider="nebius", api_url=API_URL, env_var=ENV_VAR)
