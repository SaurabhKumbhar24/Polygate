"""SambaNova Cloud — open models on their own RDU silicon.
An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

API_URL = "https://api.sambanova.ai/v1/chat/completions"
ENV_VAR = "SAMBANOVA_API_KEY"

chat = make_adapter(provider="sambanova", api_url=API_URL, env_var=ENV_VAR)
