"""Z.AI's GLM models (Zhipu AI).

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.

Two hosts, one wire format
--------------------------
``api.z.ai`` is the international platform and the default here. The mainland
platform is the same API on ``open.bigmodel.cn``, and keys are **not**
interchangeable between them. Rather than ship a second provider that differs
only by hostname, reach it with the ``base_url`` the shared adapter already
takes::

    polygate.chat(
        provider="zai",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4.6",
        messages=[...],
    )

The path is unusual — ``/api/paas/v4`` rather than the ``/v1`` almost everyone
else uses — which is exactly the sort of detail that belongs in a provider file
instead of in every caller.
"""
from ._openai_compatible import make_adapter

API_URL = "https://api.z.ai/api/paas/v4/chat/completions"
ENV_VAR = "ZAI_API_KEY"

#: The mainland endpoint, for callers that need it. Not a separate provider:
#: pass it as ``base_url``.
CHINA_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

chat = make_adapter(provider="zai", api_url=API_URL, env_var=ENV_VAR)
