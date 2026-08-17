/**
 * Keep many requests in flight at once.
 *
 * This is the client half of continuous batching. The technique itself lives in
 * the server — vLLM, TGI and every hosted provider run an iteration-level
 * scheduler that slots new sequences into a forward pass as older ones finish,
 * so the GPU never idles waiting for the slowest generation in a fixed batch. No
 * client library can perform that; it happens inside the model server.
 *
 * What a client controls is whether the scheduler has anything to schedule. A
 * loop that awaits each request before sending the next leaves a
 * continuous-batching server running a batch size of one — same GPU, a fraction
 * of the throughput, and you are billed per token either way. Keeping N requests
 * outstanding is the entire client-side contribution, and it earns a module
 * because doing it naively goes wrong in three specific ways:
 *
 * **Unbounded fan-out.** `Promise.all(items.map(chat))` on ten thousand prompts
 * opens ten thousand sockets and earns an immediate 429.
 *
 * **Losing the mapping.** Completion order is not submission order. Results have
 * to be placed back by index or every answer is attached to the wrong question —
 * a failure that produces plausible output and no error.
 *
 * **One failure killing the run.** `Promise.all` rejects on the first error,
 * discarding thousands of answers already paid for.
 *
 * For work nobody is waiting on, `batch.ts` is roughly half the price. This is
 * for when you want the answers now.
 */
import { chat } from "./client";
import { ProviderAPIError } from "./errors";
import type { KeyInput, Retry } from "./retry";
import { ChatResponse } from "./types";

/**
 * Enough to keep a server's batcher fed without looking like an attack. Hosted
 * providers throttle well below this on free tiers, which is what the adaptive
 * limiter is for.
 */
export const DEFAULT_CONCURRENCY = 8;

/** Never drop below this. A limiter that can reach zero deadlocks. */
export const MIN_CONCURRENCY = 1;

export interface Completion {
  /** Which input this answers. Needed because results finish out of order. */
  index: number;
  response: ChatResponse | null;
  error: unknown | null;
}

export interface Progress {
  done: number;
  total: number;
  failed: number;
  concurrency: number;
  elapsedMs: number;
}

/**
 * A concurrency ceiling that backs off when the provider pushes back.
 *
 * Additive-increase, multiplicative-decrease — the same shape as TCP congestion
 * control, for the same reason: the right level is unknown, differs per provider
 * and per key, and changes with load. Guessing once and holding it is wrong in
 * both directions, and guessing high costs a rate-limit ban rather than a slow
 * run. Halving on a 429 and recovering by one keeps the steady state just under
 * the limit instead of oscillating across it.
 */
export class AdaptiveLimiter {
  limit: number;
  readonly maximum: number;
  private active = 0;
  private waiters: (() => void)[] = [];

  constructor(start = DEFAULT_CONCURRENCY, maximum?: number) {
    this.limit = Math.max(MIN_CONCURRENCY, start);
    this.maximum = maximum ?? Math.max(MIN_CONCURRENCY, start);
  }

  async acquire(): Promise<void> {
    if (this.active < this.limit) {
      this.active += 1;
      return;
    }
    await new Promise<void>((resolve) => this.waiters.push(resolve));
    this.active += 1;
  }

  release(): void {
    this.active -= 1;
    this.wake();
  }

  onSuccess(): void {
    if (this.limit < this.maximum) {
      this.limit += 1;
      this.wake();
    }
  }

  onRateLimit(): void {
    // Takes effect immediately for anything not yet started; requests already
    // in flight are left to finish rather than cancelled.
    this.limit = Math.max(MIN_CONCURRENCY, Math.floor(this.limit / 2));
  }

  private wake(): void {
    while (this.waiters.length && this.active < this.limit) {
      const next = this.waiters.shift();
      next?.();
    }
  }
}

function isRateLimit(error: unknown): boolean {
  return error instanceof ProviderAPIError && error.statusCode === 429;
}

export interface MapOptions {
  apiKey?: KeyInput;
  concurrency?: number;
  /** Let concurrency find its own level against the provider's real limit. */
  adaptive?: boolean;
  retry?: Retry;
  /** Throw on the first failure instead of collecting them. */
  raiseOnError?: boolean;
  onProgress?: (progress: Progress) => void;
  [key: string]: unknown;
}

/**
 * Run many conversations against one model, N at a time.
 *
 * Returns a list **in input order**, one entry per conversation, each either a
 * response or an error. Errors are captured rather than thrown by default: a
 * ten-thousand-item run will contain failures, and discarding the successes
 * because of them wastes money already spent.
 */
export async function mapChat(
  provider: string,
  model: string,
  conversations: { role: string; content: string }[][],
  options: MapOptions = {}
): Promise<Completion[]> {
  const {
    concurrency = DEFAULT_CONCURRENCY,
    adaptive = true,
    raiseOnError = false,
    onProgress,
    ...rest
  } = options;

  const items = [...conversations];
  if (!items.length) return [];

  const limiter = adaptive ? new AdaptiveLimiter(concurrency, concurrency) : null;
  const results: Completion[] = new Array(items.length);
  const started = Date.now();
  let done = 0;
  let failed = 0;

  // A fixed pool of workers pulling from a shared cursor. Bounded by
  // construction, so nothing can fan out past `concurrency`.
  let cursor = 0;
  const worker = async () => {
    for (;;) {
      const index = cursor;
      cursor += 1;
      if (index >= items.length) return;

      if (limiter) await limiter.acquire();
      try {
        const response = await chat({
          provider, model, messages: items[index], ...rest,
        } as any);
        limiter?.onSuccess();
        results[index] = { index, response, error: null };
      } catch (error) {
        if (limiter && isRateLimit(error)) limiter.onRateLimit();
        results[index] = { index, response: null, error };
        failed += 1;
      } finally {
        limiter?.release();
      }
      done += 1;
      onProgress?.({
        done, total: items.length, failed,
        concurrency: limiter?.limit ?? concurrency,
        elapsedMs: Date.now() - started,
      });
    }
  };

  await Promise.all(
    Array.from({ length: Math.min(concurrency, items.length) }, worker)
  );

  if (raiseOnError) {
    const firstFailure = results.find((r) => r.error !== null);
    if (firstFailure) throw firstFailure.error;
  }
  return results;
}

/** `mapChat` for the common case of one user message each. */
export async function mapPrompts(
  provider: string,
  model: string,
  prompts: string[],
  options: MapOptions & { system?: string } = {}
): Promise<Completion[]> {
  const { system, ...rest } = options;
  const conversations = prompts.map((prompt) => {
    const turns: { role: string; content: string }[] = [];
    if (system) turns.push({ role: "system", content: system });
    turns.push({ role: "user", content: prompt });
    return turns;
  });
  return mapChat(provider, model, conversations, rest);
}

/**
 * Just the text, with failures rendered as `fallback`.
 *
 * For feeding results into something that wants strings and treats a blank as
 * "no answer".
 */
export function contents(completions: Completion[], fallback = ""): string[] {
  return completions.map((c) => c.response?.content ?? fallback);
}
