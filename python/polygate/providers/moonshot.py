"""Moonshot's API, used to reach Kimi K2 and its siblings.

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

API_URL = "https://api.moonshot.ai/v1/chat/completions"
ENV_VAR = "MOONSHOT_API_KEY"

chat = make_adapter(provider="moonshot", api_url=API_URL, env_var=ENV_VAR)
