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
