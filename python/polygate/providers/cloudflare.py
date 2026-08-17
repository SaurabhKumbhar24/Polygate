"""Cloudflare Workers AI — open models on Cloudflare's edge.

The URL contains your account id, so one of ``base_url``,
``$CLOUDFLARE_BASE_URL`` or ``$CLOUDFLARE_ACCOUNT_ID`` is required::

    polygate.chat(
        provider="cloudflare",
        base_url="<your-account-id>",          # or the full .../ai/v1 URL
        model="@cf/meta/llama-3.1-8b-instruct",
        messages=[...],
    )

Model ids carry an ``@cf/`` prefix naming the publisher, which is Cloudflare's
own convention rather than anything OpenAI-shaped.

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from __future__ import annotations

import os
from typing import Any

from ._openai_compatible import make_adapter
from ..types import ChatResponse

ENV_VAR = "CLOUDFLARE_API_TOKEN"
BASE_URL_ENV = "CLOUDFLARE_BASE_URL"
ACCOUNT_ENV = "CLOUDFLARE_ACCOUNT_ID"

API_ROOT = "https://api.cloudflare.com/client/v4/accounts"


def account_url(account_id: str) -> str:
    """The OpenAI-compatible root for an account id."""
    return f"{API_ROOT}/{account_id}/ai/v1"


def _url(root: str, model: str) -> str:
    """Accept a bare account id as readily as a full URL.

    The account id is the thing a user actually has to hand — it is what the
    dashboard shows — so making them assemble a five-segment URL around it adds
    a step that exists only to be got wrong.
    """
    base = root.rstrip("/")
    if "://" not in base:
        base = account_url(base)
    if not base.endswith("/chat/completions"):
        base = f"{base}/chat/completions"
    return base


_adapter = make_adapter(
    provider="cloudflare",
    env_var=ENV_VAR,
    base_url_env=BASE_URL_ENV,
    url_builder=_url,
)


def chat(**kwargs: Any) -> ChatResponse:
    """Resolve the account id at call time, then delegate.

    Read here rather than at import so that configuring the environment after
    importing polygate still works — which is the normal order in a notebook, a
    test that monkeypatches, or anything using python-dotenv.
    """
    if not kwargs.get("base_url"):
        account = os.environ.get(ACCOUNT_ENV)
        if account and not os.environ.get(BASE_URL_ENV):
            kwargs["base_url"] = account_url(account)
    return _adapter(**kwargs)


chat.endpoint_required = True  # type: ignore[attr-defined]
