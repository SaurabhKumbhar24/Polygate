export class PolygateError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PolygateError";
  }
}

export class UnsupportedProviderError extends PolygateError {
  constructor(message: string) {
    super(message);
    this.name = "UnsupportedProviderError";
  }
}

export class MissingAPIKeyError extends PolygateError {
  constructor(message: string) {
    super(message);
    this.name = "MissingAPIKeyError";
  }
}

/**
 * Thrown when a provider that has no fixed host is called without one.
 *
 * Kept distinct from a missing key because the fix is different: Databricks,
 * Cloudflare, Azure and Vertex serve from a URL containing your own workspace,
 * account, resource or project, so there is nothing to default to.
 */
export class MissingEndpointError extends PolygateError {
  constructor(message: string) {
    super(message);
    this.name = "MissingEndpointError";
  }
}

export class ProviderAPIError extends PolygateError {
  provider: string;
  statusCode: number;

  constructor(provider: string, statusCode: number, message: string) {
    super(`[${provider}] API error ${statusCode}: ${message}`);
    this.name = "ProviderAPIError";
    this.provider = provider;
    this.statusCode = statusCode;
  }
}
