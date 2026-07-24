import assert from "node:assert";
import { test } from "node:test";

import { MissingAPIKeyError, ProviderAPIError } from "../errors";
import { KeyPool, delayFor, execute, resolveKeys, resolveRetry } from "../retry";

/**
 * Mirrors sdk/python/tests/test_retry.py. The rules under test, in order of how
 * expensive they are to get wrong:
 *
 *   - a deterministic 4xx is never retried (retrying re-sends a request that
 *     cannot succeed, and bills for it)
 *   - a rejected key is evicted, not just skipped, so it stops burning one
 *     attempt on every later request
 *   - a rate-limited key rotates *before* it sleeps
 *   - opting out is the default: no `retry` means exactly one attempt
 */

function fakeResponse(status: number, body = "", headers: Record<string, string> = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (k: string) => headers[k] ?? headers[k.toLowerCase()] ?? null },
    text: async () => body,
  } as unknown as Response;
}

/** An attempt fn returning the given statuses in order, logging keys used. */
function recorder(...statuses: number[]) {
  const keysUsed: string[] = [];
  const queue = [...statuses];
  const fn = async (key: string) => {
    keysUsed.push(key);
    return fakeResponse(queue.shift() ?? 200);
  };
  return Object.assign(fn, { keysUsed });
}

function sleepSpy() {
  const calls: number[] = [];
  const fn = async (s: number) => {
    calls.push(s);
  };
  return Object.assign(fn, { calls });
}

// ── opting out ───────────────────────────────────────────────────────────────

test("no retry policy means one attempt", async () => {
  const attempt = recorder(500, 200);
  await assert.rejects(
    () => execute("openai", new KeyPool(["k1"]), attempt, undefined, sleepSpy()),
    ProviderAPIError
  );
  assert.equal(attempt.keysUsed.length, 1, "retried despite no retry policy");
});

test("a successful response is returned", async () => {
  const resp = await execute("openai", new KeyPool(["k1"]), recorder(200), undefined, sleepSpy());
  assert.equal(resp.status, 200);
});

// ── what must not be retried ─────────────────────────────────────────────────

for (const status of [400, 404, 422]) {
  test(`deterministic ${status} is not retried`, async () => {
    const attempt = recorder(status, 200);
    await assert.rejects(
      () => execute("openai", new KeyPool(["k1", "k2"]), attempt, { maxAttempts: 5 }, sleepSpy()),
      (err: ProviderAPIError) => err.statusCode === status
    );
    assert.equal(attempt.keysUsed.length, 1, `${status} was retried`);
  });
}

// ── auth failures evict ──────────────────────────────────────────────────────

test("a rejected key is evicted and the next is tried immediately", async () => {
  const attempt = recorder(401, 200);
  const sleep = sleepSpy();
  const resp = await execute("openai", new KeyPool(["bad", "good"]), attempt, { maxAttempts: 3 }, sleep);
  assert.equal(resp.status, 200);
  assert.deepEqual(attempt.keysUsed, ["bad", "good"]);
  assert.deepEqual(sleep.calls, [], "slept on an auth failure instead of rotating");
});

test("eviction persists across calls", async () => {
  const pool = new KeyPool(["bad", "good"]);
  await execute("openai", pool, recorder(401, 200), { maxAttempts: 3 }, sleepSpy());

  const attempt = recorder(200);
  await execute("openai", pool, attempt, { maxAttempts: 3 }, sleepSpy());
  assert.deepEqual(attempt.keysUsed, ["good"], "evicted key came back");
});

test("all keys rejected throws", async () => {
  await assert.rejects(
    () => execute("openai", new KeyPool(["a", "b"]), recorder(403, 403), { maxAttempts: 5 }, sleepSpy()),
    (err: ProviderAPIError) => err.statusCode === 403
  );
});

// ── rate limits rotate before sleeping ───────────────────────────────────────

test("a rate-limited key rotates without sleeping when another is free", async () => {
  const attempt = recorder(429, 200);
  const sleep = sleepSpy();
  const resp = await execute("openai", new KeyPool(["k1", "k2"]), attempt, { maxAttempts: 3 }, sleep);
  assert.equal(resp.status, 200);
  assert.deepEqual(attempt.keysUsed, ["k1", "k2"]);
  assert.deepEqual(sleep.calls, [], "burned a backoff window with an idle key available");
});

test("a single rate-limited key falls back to sleeping", async () => {
  const sleep = sleepSpy();
  const resp = await execute("openai", new KeyPool(["only"]), recorder(429, 200), { maxAttempts: 3 }, sleep);
  assert.equal(resp.status, 200);
  assert.equal(sleep.calls.length, 1, "did not back off with no other key to try");
});

test("Retry-After is honoured", async () => {
  let n = 0;
  const attempt = async () => (++n === 1 ? fakeResponse(429, "", { "Retry-After": "7" }) : fakeResponse(200));
  const sleep = sleepSpy();
  await execute("openai", new KeyPool(["only"]), attempt, { maxAttempts: 3, jitter: false }, sleep);
  assert.deepEqual(sleep.calls, [7]);
});

test("Retry-After is capped by maxDelay", async () => {
  let n = 0;
  const attempt = async () => (++n === 1 ? fakeResponse(429, "", { "Retry-After": "9999" }) : fakeResponse(200));
  const sleep = sleepSpy();
  await execute("openai", new KeyPool(["only"]), attempt, { maxAttempts: 3, maxDelay: 5, jitter: false }, sleep);
  assert.deepEqual(sleep.calls, [5], "a hostile Retry-After stalled the caller");
});

// ── transient server errors ──────────────────────────────────────────────────

test("5xx retries then succeeds", async () => {
  const attempt = recorder(503, 200);
  const resp = await execute("openai", new KeyPool(["k1"]), attempt, { maxAttempts: 3 }, sleepSpy());
  assert.equal(resp.status, 200);
  assert.equal(attempt.keysUsed.length, 2);
});

test("529 overloaded is retryable", () => {
  assert.ok(resolveRetry().retryOn.includes(529), "Anthropic's overloaded code is not retried");
});

test("attempts are bounded", async () => {
  const attempt = recorder(500, 500, 500, 500, 500);
  await assert.rejects(
    () => execute("openai", new KeyPool(["k1"]), attempt, { maxAttempts: 3 }, sleepSpy()),
    ProviderAPIError
  );
  assert.equal(attempt.keysUsed.length, 3);
});

// ── pool mechanics ───────────────────────────────────────────────────────────

test("pool rotates round-robin", () => {
  const pool = new KeyPool(["a", "b", "c"]);
  assert.deepEqual([pool.acquire(), pool.acquire(), pool.acquire(), pool.acquire()], ["a", "b", "c", "a"]);
});

test("pool drops duplicates", () => {
  assert.equal(new KeyPool(["a", "a", "b"]).size, 2);
});

test("pool rejects empty input", () => {
  assert.throws(() => new KeyPool([]));
  assert.throws(() => new KeyPool(["", "   "]));
});

test("backoff is exponential and capped", () => {
  const policy = resolveRetry({ baseDelay: 1, maxDelay: 10, jitter: false });
  assert.deepEqual([0, 1, 2, 3, 4].map((i) => delayFor(policy, i)), [1, 2, 4, 8, 10]);
});

test("jitter stays within bounds", () => {
  const policy = resolveRetry({ baseDelay: 1, maxDelay: 10, jitter: true });
  for (let i = 0; i < 4; i++) {
    const d = delayFor(policy, i);
    assert.ok(d >= 0 && d <= Math.min(2 ** i, 10));
  }
});

// ── key resolution ───────────────────────────────────────────────────────────

test("resolves a single key, a list, and a pool", () => {
  assert.equal(resolveKeys("k1", "SOME_ENV", "openai").size, 1);
  assert.equal(resolveKeys(["k1", "k2"], "SOME_ENV", "openai").size, 2);
  const pool = new KeyPool(["k1"]);
  assert.equal(resolveKeys(pool, "SOME_ENV", "openai"), pool);
});

test("reads a comma-separated env var", () => {
  process.env.POLYGATE_TEST_KEYS = "k1, k2 ,k3";
  try {
    assert.equal(resolveKeys(undefined, "POLYGATE_TEST_KEYS", "openai").size, 3);
  } finally {
    delete process.env.POLYGATE_TEST_KEYS;
  }
});

test("throws when nothing is configured", () => {
  delete process.env.POLYGATE_TEST_KEYS;
  assert.throws(() => resolveKeys(undefined, "POLYGATE_TEST_KEYS", "openai"), MissingAPIKeyError);
});
