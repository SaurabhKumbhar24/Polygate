"""Novita AI — open models, GPU marketplace pricing.

The version segment is ``/v3``, not the ``/v1`` almost everyone else uses.

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

API_URL = "https://api.novita.ai/v3/openai/chat/completions"
ENV_VAR = "NOVITA_API_KEY"

chat = make_adapter(provider="novita", api_url=API_URL, env_var=ENV_VAR)
