"""Fireworks AI. Note the /inference prefix — it is not the usual /v1 root.

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
ENV_VAR = "FIREWORKS_API_KEY"

chat = make_adapter(provider="fireworks", api_url=API_URL, env_var=ENV_VAR)
