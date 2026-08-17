"""Baseten Model APIs — hosted open models, no deployment step.

Baseten also serves *your own* deployed models, each on its own URL. Those are
reachable with ``base_url``; this default is the shared Model APIs endpoint.

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

API_URL = "https://inference.baseten.co/v1/chat/completions"
ENV_VAR = "BASETEN_API_KEY"

chat = make_adapter(provider="baseten", api_url=API_URL, env_var=ENV_VAR)
