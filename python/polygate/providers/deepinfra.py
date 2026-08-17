"""DeepInfra — open models billed per token.

Note the path: the OpenAI-compatible surface lives under ``/v1/openai``, beside
DeepInfra's own native API at ``/v1/inference``.

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
ENV_VAR = "DEEPINFRA_API_KEY"

chat = make_adapter(provider="deepinfra", api_url=API_URL, env_var=ENV_VAR)
