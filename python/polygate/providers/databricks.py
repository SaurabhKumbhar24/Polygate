"""Databricks Model Serving — foundation models and your own, on your workspace.

There is no fixed host: the URL is your workspace, so ``base_url`` (or
``$DATABRICKS_HOST``) is required::

    polygate.chat(
        provider="databricks",
        base_url="https://dbc-1234abcd-5678.cloud.databricks.com/serving-endpoints",
        model="databricks-claude-sonnet-4-5",
        messages=[...],
    )

The ``model`` is the **serving endpoint name**, not a vendor model id — which is
why two workspaces on the same underlying model can use different names for it.

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

ENV_VAR = "DATABRICKS_TOKEN"
BASE_URL_ENV = "DATABRICKS_HOST"

#: Appended when a bare workspace host is given, so both of these work:
#: "https://ws.cloud.databricks.com" and ".../serving-endpoints".
SERVING_PATH = "/serving-endpoints"


def _url(root: str, model: str) -> str:
    """Build the completions URL from whatever shape of root was supplied.

    Databricks documents the endpoint three ways — workspace root, root plus
    ``/serving-endpoints``, and the full per-endpoint invocation URL — and all
    three are things people paste. Accepting each beats a 404 whose cause is a
    missing path segment.
    """
    base = root.rstrip("/")
    if base.endswith("/invocations"):
        return base
    if not base.endswith(SERVING_PATH):
        base = f"{base}{SERVING_PATH}"
    return f"{base}/{model}/invocations"


chat = make_adapter(
    provider="databricks",
    env_var=ENV_VAR,
    base_url_env=BASE_URL_ENV,
    url_builder=_url,
)
