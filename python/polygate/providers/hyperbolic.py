"""Hyperbolic — open models on spare GPU capacity.
An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

API_URL = "https://api.hyperbolic.xyz/v1/chat/completions"
ENV_VAR = "HYPERBOLIC_API_KEY"

chat = make_adapter(provider="hyperbolic", api_url=API_URL, env_var=ENV_VAR)
