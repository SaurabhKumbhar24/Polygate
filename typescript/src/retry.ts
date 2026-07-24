/**
 * Optional key rotation and backoff retries.
 *
 * Both are off by default. `chat()` with no `retry` and a single `apiKey`
 * behaves exactly as it did before this module existed: one request, one
 * response, errors thrown straight through.
 *
 * The interesting failure in high-volume LLM traffic is not the network, it is
 * the rate limit, and a rate limit is per *key*, not per host. Sleeping on a
 * 429 while another key sits idle wastes the whole backoff window, so the loop
 * below rotates first and sleeps only when it runs out of keys.
 *
 * Failure classes are handled differently on purpose:
 *   - 401/403  the key is bad, not busy: evict it and try the next immediately.
 *              A dead key left in a pool otherwise burns one attempt on every
 *              later request, forever.
 *   - 429      the key is rate limited: park it for its `Retry-After` and
 *              rotate at once.
 *   - 5xx/529  the provider is struggling: back off.
 *   - 400/404/422  deterministic. Retrying re-sends a request that cannot
 *              succeed, and bills for it. Throw immediately.
 */
import { MissingAPIKeyError, ProviderAPIError } from "./errors";

/** Statuses worth trying again. 529 is Anthropic's "overloaded". */
export const DEFAULT_RETRY_ON = [408, 409, 429, 500, 502, 503, 504, 529];

const AUTH_FAILURE_CODES = [401, 403];

export interface Retry {
  /** Total attempts, not extra ones: 1 is equivalent to no retry. */
  maxAttempts?: number;
  baseDelay?: number;
  maxDelay?: number;
  jitter?: boolean;
  retryOn?: number[];
  /** Prefer the provider's `Retry-After` header over computed backoff. */
  respectRetryAfter?: boolean;
}

interface ResolvedRetry extends Required<Retry> {}

export function resolveRetry(retry?: Retry): ResolvedRetry {
  return {
    maxAttempts: retry?.maxAttempts ?? 3,
    baseDelay: retry?.baseDelay ?? 0.5,
    maxDelay: retry?.maxDelay ?? 20,
    jitter: retry?.jitter ?? true,
    retryOn: retry?.retryOn ?? DEFAULT_RETRY_ON,
    respectRetryAfter: retry?.respectRetryAfter ?? true,
  };
}

export function delayFor(
  policy: ResolvedRetry,
  attempt: number,
  retryAfter?: number
): number {
  if (policy.respectRetryAfter && retryAfter !== undefined) {
    return Math.min(retryAfter, policy.maxDelay);
  }
  const delay = Math.min(policy.baseDelay * 2 ** attempt, policy.maxDelay);
  // Full jitter: spreads a herd of clients rate limited by the same upstream
  // at the same moment.
  return policy.jitter ? Math.random() * delay : delay;
}

/** A rotating set of API keys for one provider. */
export class KeyPool {
  private keys: string[];
  private cursor = 0;
  private cooldown = new Map<string, number>();
  private dead = new Set<string>();

  constructor(keys: string[]) {
    const cleaned = keys.map((k) => (k ?? "").trim()).filter(Boolean);
    if (cleaned.length === 0) {
      throw new Error("KeyPool requires at least one non-empty key");
    }
    // Preserve order, drop duplicates. A repeated key is not extra capacity.
    this.keys = [...new Set(cleaned)];
  }

  get size(): number {
    return this.keys.length;
  }

  /** Keys neither evicted nor currently cooling down. */
  get usable(): number {
    const now = Date.now();
    return this.keys.filter(
      (k) => !this.dead.has(k) && (this.cooldown.get(k) ?? 0) <= now
    ).length;
  }

  get exhausted(): boolean {
    return this.keys.every((k) => this.dead.has(k));
  }

  /**
   * Next usable key, round-robin, or undefined if every key is evicted.
   * When all live keys are cooling down, returns whichever recovers first so a
   * caller willing to wait still makes progress.
   */
  acquire(): string | undefined {
    const now = Date.now();
    const live = this.keys.filter((k) => !this.dead.has(k));
    if (live.length === 0) return undefined;

    for (let offset = 0; offset < live.length; offset++) {
      const key = live[(this.cursor + offset) % live.length];
      if ((this.cooldown.get(key) ?? 0) <= now) {
        this.cursor = (this.cursor + offset + 1) % live.length;
        return key;
      }
    }
    return live.reduce((a, b) =>
      (this.cooldown.get(a) ?? 0) <= (this.cooldown.get(b) ?? 0) ? a : b
    );
  }

  coolDown(key: string, seconds: number): void {
    this.cooldown.set(key, Date.now() + Math.max(0, seconds) * 1000);
  }

  /** Permanently remove a key the provider rejected. */
  evict(key: string): void {
    this.dead.add(key);
  }
}

export type KeyInput = string | string[] | KeyPool | undefined;

/**
 * Turn whatever the caller passed into a pool. Falls back to the provider's
 * standard environment variable, which may hold a comma-separated list so a
 * pool can be configured without a code change.
 */
export function resolveKeys(
  apiKey: KeyInput,
  envVar: string,
  provider: string
): KeyPool {
  if (apiKey instanceof KeyPool) return apiKey;

  let keys: string[];
  if (typeof apiKey === "string") {
    keys = [apiKey];
  } else if (Array.isArray(apiKey) && apiKey.length > 0) {
    keys = apiKey;
  } else {
    keys = (process.env[envVar] ?? "")
      .split(",")
      .map((k) => k.trim())
      .filter(Boolean);
  }

  if (keys.length === 0) {
    throw new MissingAPIKeyError(
      `No ${provider} API key provided. Pass apiKey or set ${envVar}.`
    );
  }
  return new KeyPool(keys);
}

function retryAfterSeconds(response: Response): number | undefined {
  const raw =
    response.headers?.get?.("Retry-After") ??
    response.headers?.get?.("retry-after");
  if (!raw) return undefined;
  const parsed = Number(raw);
  // The HTTP-date form is legal but rare from LLM providers; computed backoff
  // is a fine substitute.
  return Number.isFinite(parsed) ? parsed : undefined;
}

const wait = (seconds: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, seconds * 1000));

/**
 * Run `attempt(key)` until it returns a 2xx, or give up.
 * Returns the successful Response; throws ProviderAPIError otherwise.
 */
export async function execute(
  provider: string,
  pool: KeyPool,
  attempt: (key: string) => Promise<Response>,
  retry?: Retry,
  sleep: (seconds: number) => Promise<void> = wait
): Promise<Response> {
  const policy = resolveRetry(retry ?? { maxAttempts: 1 });
  const attempts = Math.max(1, policy.maxAttempts);
  let lastStatus = 0;
  let lastBody = "";

  for (let i = 0; i < attempts; i++) {
    const key = pool.acquire();
    if (key === undefined) {
      throw new ProviderAPIError(
        provider,
        lastStatus || 401,
        lastBody || "every API key was rejected by the provider"
      );
    }

    let response: Response;
    try {
      response = await attempt(key);
    } catch (err) {
      lastStatus = 0;
      lastBody = String(err);
      if (i === attempts - 1) {
        throw new ProviderAPIError(provider, 0, `request failed: ${lastBody}`);
      }
      await sleep(delayFor(policy, i));
      continue;
    }

    if (response.ok) return response;

    lastStatus = response.status;
    lastBody = await safeBody(response);

    if (AUTH_FAILURE_CODES.includes(response.status)) {
      // Not transient: stop using this key entirely and try the next one right
      // away, without spending a backoff window.
      pool.evict(key);
      if (pool.exhausted) {
        throw new ProviderAPIError(provider, response.status, lastBody);
      }
      continue;
    }

    if (!policy.retryOn.includes(response.status)) {
      throw new ProviderAPIError(provider, response.status, lastBody);
    }

    const after = retryAfterSeconds(response);
    if (response.status === 429) {
      pool.coolDown(key, after ?? policy.baseDelay * 2 ** i);
      if (pool.usable > 0 && i < attempts - 1) continue; // another key is free
    }

    if (i === attempts - 1) break;
    await sleep(delayFor(policy, i, after));
  }

  throw new ProviderAPIError(provider, lastStatus, lastBody);
}

async function safeBody(response: Response): Promise<string> {
  try {
    return await response.text();
  } catch {
    return "";
  }
}
