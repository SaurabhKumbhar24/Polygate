"""Azure OpenAI — the same models, behind Azure's own conventions.

Three things differ from OpenAI proper, and each is a distinct way to get a 401
or 404 that looks like something else:

1. **The key is not a Bearer token.** It goes in an ``api-key`` header. Sending
   ``Authorization: Bearer`` returns a 401 indistinguishable from a wrong key.
2. **The model is a deployment name in the path**, not a field in the body. Your
   deployment can be called anything, so ``model="gpt-4o"`` only works if you
   named the deployment ``gpt-4o``.
3. **An ``api-version`` query parameter is mandatory.** Omitting it is a 404.

The host is your own resource, so ``base_url`` or ``$AZURE_OPENAI_ENDPOINT`` is
required::

    polygate.chat(
        provider="azure",
        base_url="https://my-resource.openai.azure.com",
        model="my-gpt4o-deployment",
        messages=[...],
    )

An OpenAI-compatible *body*; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from __future__ import annotations

import os

from ._openai_compatible import api_key_header_auth, make_adapter

ENV_VAR = "AZURE_OPENAI_API_KEY"
BASE_URL_ENV = "AZURE_OPENAI_ENDPOINT"
API_VERSION_ENV = "AZURE_OPENAI_API_VERSION"

#: Pinned rather than tracking latest. Azure's versions change behaviour, and a
#: default that drifts under callers would turn a library upgrade into a silent
#: change in what the model does. Override per-call or via the environment.
DEFAULT_API_VERSION = "2024-10-21"


def _url(root: str, model: str) -> str:
    """Assemble the deployment URL, unless a full one was already supplied."""
    base = root.rstrip("/")
    version = os.environ.get(API_VERSION_ENV) or DEFAULT_API_VERSION
    if "/chat/completions" in base:
        return base if "api-version=" in base else f"{base}?api-version={version}"
    return f"{base}/openai/deployments/{model}/chat/completions?api-version={version}"


chat = make_adapter(
    provider="azure",
    env_var=ENV_VAR,
    base_url_env=BASE_URL_ENV,
    auth=api_key_header_auth("api-key"),
    url_builder=_url,
)
