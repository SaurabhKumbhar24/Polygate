"""The format every provider below imitates.

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

API_URL = "https://api.openai.com/v1/chat/completions"
ENV_VAR = "OPENAI_API_KEY"

chat = make_adapter(provider="openai", api_url=API_URL, env_var=ENV_VAR)
