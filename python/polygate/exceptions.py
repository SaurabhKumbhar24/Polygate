class PolygateError(Exception):
    """Base exception for all polygate errors."""


class UnsupportedProviderError(PolygateError):
    """Raised when an unknown provider name is requested."""


class MissingAPIKeyError(PolygateError):
    """Raised when no API key is supplied and none is found in the environment."""


class MissingEndpointError(PolygateError):
    """Raised when a provider that has no fixed host is called without one.

    Kept distinct from a missing key because the fix is different: Databricks,
    Cloudflare, Azure and Vertex serve from a URL containing your own workspace,
    account, resource or project, so there is nothing to default to.
    """


class ProviderAPIError(PolygateError):
    """Raised when a provider's API returns a non-success response."""

    def __init__(self, provider: str, status_code: int, message: str):
        self.provider = provider
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{provider}] API error {status_code}: {message}")
