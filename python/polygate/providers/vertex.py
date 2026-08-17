"""Google Vertex AI, through its OpenAI-compatible endpoint.

Vertex exposes an OpenAI-shaped surface at
``/endpoints/openapi/chat/completions``, which is what this uses — the native
Vertex API is a different shape again from the Gemini Developer API this package
already speaks in ``gemini.py``.

**The credential is an OAuth access token, not an API key.** Vertex does not
issue long-lived keys; it authenticates with a short-lived token, typically from
``gcloud auth print-access-token`` or a service-account exchange. Those expire in
about an hour, so a token pasted into an environment variable will start
returning 401 rather than continuing to work — which is a property of Vertex, not
a fault here, but it is the thing that surprises people.

Minting tokens is deliberately out of scope: doing it properly means service
account JWT signing and a refresh cycle, which would mean a crypto dependency in
a package whose entire premise is that it has almost none. Pass a fresh token::

    polygate.chat(
        provider="vertex",
        api_key=subprocess.check_output(
            ["gcloud", "auth", "print-access-token"], text=True
        ).strip(),
        base_url="https://us-central1-aiplatform.googleapis.com/v1/"
                 "projects/my-project/locations/us-central1/endpoints/openapi",
        model="google/gemini-2.0-flash",
        messages=[...],
    )

An OpenAI-compatible endpoint; see ``_openai_compatible`` for the shared
implementation and why it is shared.
"""
from ._openai_compatible import make_adapter

ENV_VAR = "GOOGLE_ACCESS_TOKEN"
BASE_URL_ENV = "VERTEX_BASE_URL"

#: The endpoint template, for callers assembling one from parts.
URL_TEMPLATE = (
    "https://{region}-aiplatform.googleapis.com/v1/"
    "projects/{project}/locations/{region}/endpoints/openapi"
)


def endpoint(project: str, region: str = "us-central1") -> str:
    """Build the base URL from a project and region.

    Offered because the template has the region in it twice, and a mismatch
    between the two produces a 404 rather than anything that names the problem.
    """
    return URL_TEMPLATE.format(project=project, region=region)


chat = make_adapter(
    provider="vertex",
    env_var=ENV_VAR,
    base_url_env=BASE_URL_ENV,
)
