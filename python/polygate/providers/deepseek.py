"""DeepSeek's API. Reasoning models here return the answer in
``reasoning_content`` when ``content`` is empty, which the shared parser
already handles.

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

API_URL = "https://api.deepseek.com/chat/completions"
ENV_VAR = "DEEPSEEK_API_KEY"

chat = make_adapter(provider="deepseek", api_url=API_URL, env_var=ENV_VAR)
