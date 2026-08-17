"""Groq's LPU inference — the draw is latency, so the retry policy matters
less here than elsewhere.

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

API_URL = "https://api.groq.com/openai/v1/chat/completions"
ENV_VAR = "GROQ_API_KEY"

chat = make_adapter(provider="groq", api_url=API_URL, env_var=ENV_VAR)
