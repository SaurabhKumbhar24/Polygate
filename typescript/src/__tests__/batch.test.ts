/**
 * Offline batch jobs and concurrent dispatch.
 *
 * Two features that get confused with each other, tested together because the
 * distinction is the point: `batch` queues a job with the provider at half
 * price; `concurrent` keeps requests in flight so a server's continuous batcher
 * has something to schedule.
 */
import assert from "node:assert";
import { afterEach, test } from "node:test";
import * as batch from "../batch";
import {
  AdaptiveLimiter, contents, mapChat, mapPrompts,
} from "../concurrent";
import { ProviderAPIError } from "../errors";

const realFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = realFetch;
  for (const key of ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"]) {
    delete process.env[key];
  }
});

const REQS: batch.BatchRequest[] = [
  { customId: "a", model: "gpt-4o-mini", messages: [{ role: "user", content: "1" }] },
  { customId: "b", model: "gpt-4o-mini", messages: [{ role: "user", content: "2" }] },
];

/** Queue up canned responses, recording every call. */
function stubSequence(responses: { body?: unknown; text?: string; status?: number }[]) {
  const calls: { url: string; init: any }[] = [];
  let index = 0;
  globalThis.fetch = (async (url: string, init: any) => {
    calls.push({ url: String(url), init });
    const next = responses[Math.min(index, responses.length - 1)];
    index += 1;
    const text = next.text ?? JSON.stringify(next.body ?? {});
    return {
      ok: (next.status ?? 200) < 400,
      status: next.status ?? 200,
      json: async () => next.body ?? {},
      text: async () => text,
    };
  }) as unknown as typeof fetch;
  return calls;
}

// ── Guardrails that fail before a job is paid for ────────────────────────────

test("a provider without a batch tier points at the alternative", async () => {
  // A 404 from a missing /batches route reads like an outage. Naming the
  // fallback matters because there is a real one that works everywhere.
  await assert.rejects(
    () => batch.submit("gemini", REQS),
    (error: Error) =>
      error instanceof batch.BatchNotSupportedError && /mapChat/.test(error.message)
  );
});

test("duplicate customIds are refused up front", async () => {
  // Providers key results by customId, so a duplicate makes some results
  // unreachable — and it surfaces at retrieval, hours and dollars later.
  process.env.OPENAI_API_KEY = "k";
  await assert.rejects(
    () =>
      batch.submit("openai", [
        { customId: "same", model: "m", messages: [] },
        { customId: "same", model: "m", messages: [] },
      ]),
    /unique/
  );
});

test("an empty batch is refused", async () => {
  process.env.OPENAI_API_KEY = "k";
  await assert.rejects(() => batch.submit("openai", []), /at least one/);
});

test("batch aliases resolve like chat aliases", async () => {
  process.env.ANTHROPIC_API_KEY = "k";
  stubSequence([{ body: { id: "msgbatch_1", processing_status: "in_progress" } }]);
  const job = await batch.submit("claude", [
    { customId: "a", model: "claude-sonnet-4-5", messages: [] },
  ]);
  assert.equal(job.provider, "anthropic");
});

// ── OpenAI-style ─────────────────────────────────────────────────────────────

test("openai uploads jsonl then creates the job", async () => {
  process.env.OPENAI_API_KEY = "k";
  const calls = stubSequence([
    { body: { id: "file-1" } },
    {
      body: {
        id: "batch_1", status: "validating",
        request_counts: { total: 2, completed: 0, failed: 0 },
      },
    },
  ]);
  const job = await batch.submit("openai", REQS);

  assert.ok(calls[0].url.endsWith("/files"));
  assert.ok(calls[1].url.endsWith("/batches"));
  assert.equal(JSON.parse(calls[1].init.body).input_file_id, "file-1");
  assert.equal(job.status, "pending");
  assert.equal(job.total, 2);
});

test("statuses normalize and finalizing is not treated as done", async () => {
  // The results file does not exist yet at "finalizing"; stopping there returns
  // an empty set that looks like a job with no output.
  process.env.OPENAI_API_KEY = "k";
  const cases: [string, string][] = [
    ["validating", "pending"],
    ["in_progress", "running"],
    ["finalizing", "running"],
    ["completed", "completed"],
    ["cancelled", "cancelled"],
    ["canceled", "cancelled"],
  ];
  for (const [raw, expected] of cases) {
    stubSequence([{ body: { id: "b", status: raw } }]);
    const job = await batch.status("openai", "b");
    assert.equal(job.status, expected, `${raw} should normalize to ${expected}`);
  }
  stubSequence([{ body: { id: "b", status: "finalizing" } }]);
  assert.equal(batch.isDone(await batch.status("openai", "b")), false);
});

test("groq asks for the longer completion window", async () => {
  // Under load a 24h Groq job is likelier to expire than to finish, and an
  // expired batch must be resubmitted whole.
  process.env.GROQ_API_KEY = "k";
  const calls = stubSequence([
    { body: { id: "file-1" } },
    { body: { id: "batch_1", status: "validating" } },
  ]);
  await batch.submit("groq", REQS);
  assert.equal(JSON.parse(calls[1].init.body).completion_window, "7d");
});

test("failed requests are read from the error file too", async () => {
  // Successes and failures land in two different files. Reading only the output
  // file silently loses every failure in the job.
  process.env.OPENAI_API_KEY = "k";
  globalThis.fetch = (async (url: string) => {
    const target = String(url);
    if (target.includes("/batches/")) {
      return {
        ok: true, status: 200,
        json: async () => ({
          id: "batch_1", status: "completed",
          output_file_id: "file-out", error_file_id: "file-err",
        }),
        text: async () => "",
      };
    }
    if (target.includes("file-out")) {
      return {
        ok: true, status: 200, json: async () => ({}),
        text: async () =>
          JSON.stringify({
            custom_id: "a",
            response: {
              body: {
                model: "gpt-4o-mini",
                choices: [{ message: { role: "assistant", content: "hi" } }],
                usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
              },
            },
          }),
      };
    }
    return {
      ok: true, status: 200, json: async () => ({}),
      text: async () =>
        JSON.stringify({ custom_id: "b", error: { message: "context_length_exceeded" } }),
    };
  }) as unknown as typeof fetch;

  const out = await batch.results("openai", "batch_1");
  const byId = Object.fromEntries(out.map((r) => [r.customId, r]));
  assert.equal(byId.a.response?.content, "hi");
  assert.equal(byId.a.response?.usage.promptTokens, 3);
  assert.equal(byId.b.response, null);
  assert.match(byId.b.error!, /context_length_exceeded/);
});

// ── Anthropic ────────────────────────────────────────────────────────────────

test("anthropic sends requests inline with no file upload", async () => {
  process.env.ANTHROPIC_API_KEY = "k";
  const calls = stubSequence([
    { body: { id: "msgbatch_1", processing_status: "in_progress" } },
  ]);
  await batch.submit("anthropic", [
    {
      customId: "a", model: "claude-sonnet-4-5",
      messages: [
        { role: "system", content: "be brief" },
        { role: "user", content: "hi" },
      ],
    },
  ]);
  assert.equal(calls.length, 1, "no file upload step for Anthropic");
  const entry = JSON.parse(calls[0].init.body).requests[0];
  assert.equal(entry.custom_id, "a");
  // System prompts travel beside the messages, not inside them.
  assert.equal(entry.params.system, "be brief");
  assert.deepEqual(entry.params.messages, [{ role: "user", content: "hi" }]);
  // Required by the Messages API; a batch missing it is rejected only after the
  // whole job has been accepted.
  assert.ok(entry.params.max_tokens > 0);
});

test("anthropic counts unrun requests as failed, not successful", async () => {
  // Cancelled and expired requests never ran. Counting them as successes
  // overstates the job; omitting them stalls progress below 1 forever.
  process.env.ANTHROPIC_API_KEY = "k";
  stubSequence([
    {
      body: {
        id: "b", processing_status: "ended",
        request_counts: {
          processing: 0, succeeded: 5, errored: 2, canceled: 2, expired: 1,
        },
      },
    },
  ]);
  const job = await batch.status("anthropic", "b");
  assert.equal(job.total, 10);
  assert.equal(job.completed, 5);
  assert.equal(job.failed, 5);
  assert.equal(batch.progress(job), 1);
});

test("the system prompt is split out of the message list", () => {
  const { system, turns } = batch.splitSystem([
    { role: "system", content: "one" },
    { role: "system", content: "two" },
    { role: "user", content: "hi" },
  ]);
  assert.equal(system, "one\n\ntwo");
  assert.deepEqual(turns, [{ role: "user", content: "hi" }]);
});

// ── run() ────────────────────────────────────────────────────────────────────

test("a timeout says the job is still running", async () => {
  // Giving up watching is not the same as cancelling. The batch id is still
  // good, and abandoning a paid job silently would be worse than saying so.
  process.env.OPENAI_API_KEY = "k";
  stubSequence([{ body: { id: "b7", status: "in_progress" } }]);
  await assert.rejects(
    () =>
      batch.run("openai", REQS, {
        timeoutSeconds: -1, pollSeconds: 0, sleep: async () => {},
      }),
    /b7/
  );
});

// ── Concurrency: the client half of continuous batching ─────────────────────

test("results come back in input order", async () => {
  // Completion order is not submission order. Placing results by arrival
  // attaches every answer to the wrong question — silently, with no error.
  globalThis.fetch = (async (_url: string, init: any) => {
    const body = JSON.parse(init.body);
    const index = Number(body.messages[0].content);
    // Later items finish first.
    await new Promise((r) => setTimeout(r, (20 - index) % 7));
    return {
      ok: true, status: 200,
      json: async () => ({ choices: [{ message: { content: `answer-${index}` } }] }),
      text: async () => "",
    };
  }) as unknown as typeof fetch;

  const out = await mapChat(
    "openai", "gpt-4o-mini",
    Array.from({ length: 20 }, (_, i) => [{ role: "user", content: String(i) }]),
    { apiKey: "k", adaptive: false }
  );
  assert.deepEqual(out.map((c) => c.index), Array.from({ length: 20 }, (_, i) => i));
  assert.deepEqual(
    out.map((c) => c.response?.content),
    Array.from({ length: 20 }, (_, i) => `answer-${i}`)
  );
});

test("one failure does not discard the rest", async () => {
  // A single 400 in item 4,000 of 10,000 must not throw away 3,999 answers that
  // have already been paid for.
  globalThis.fetch = (async (_url: string, init: any) => {
    const body = JSON.parse(init.body);
    if (body.messages[0].content === "3") {
      return { ok: false, status: 400, json: async () => ({}), text: async () => "bad" };
    }
    return {
      ok: true, status: 200,
      json: async () => ({ choices: [{ message: { content: "ok" } }] }),
      text: async () => "",
    };
  }) as unknown as typeof fetch;

  const out = await mapChat(
    "openai", "m",
    Array.from({ length: 6 }, (_, i) => [{ role: "user", content: String(i) }]),
    { apiKey: "k", adaptive: false }
  );
  assert.equal(out.filter((c) => c.response !== null).length, 5);
  assert.ok(out[3].error instanceof ProviderAPIError);
});

test("concurrency is actually bounded", async () => {
  // Unbounded fan-out over ten thousand prompts opens ten thousand sockets and
  // earns an immediate 429.
  let live = 0;
  let peak = 0;
  globalThis.fetch = (async () => {
    live += 1;
    peak = Math.max(peak, live);
    await new Promise((r) => setTimeout(r, 1));
    live -= 1;
    return {
      ok: true, status: 200,
      json: async () => ({ choices: [{ message: { content: "ok" } }] }),
      text: async () => "",
    };
  }) as unknown as typeof fetch;

  await mapChat(
    "openai", "m",
    Array.from({ length: 60 }, () => [{ role: "user", content: "x" }]),
    { apiKey: "k", concurrency: 4, adaptive: false }
  );
  assert.ok(peak <= 4, `peak concurrency was ${peak}`);
});

test("an empty input does no work", async () => {
  assert.deepEqual(await mapChat("openai", "m", [], { apiKey: "k" }), []);
});

test("mapPrompts builds the conversations", async () => {
  let seen: any = null;
  globalThis.fetch = (async (_url: string, init: any) => {
    seen = JSON.parse(init.body);
    return {
      ok: true, status: 200,
      json: async () => ({ choices: [{ message: { content: "ok" } }] }),
      text: async () => "",
    };
  }) as unknown as typeof fetch;

  await mapPrompts("openai", "m", ["one"], {
    system: "be brief", apiKey: "k", adaptive: false,
  });
  assert.deepEqual(seen.messages[0], { role: "system", content: "be brief" });
  assert.equal(seen.messages[1].role, "user");
});

test("contents renders failures as the fallback", () => {
  const out = contents(
    [
      { index: 0, response: { content: "hi" } as any, error: null },
      { index: 1, response: null, error: new Error("boom") },
    ],
    "<failed>"
  );
  assert.deepEqual(out, ["hi", "<failed>"]);
});

// ── The adaptive limiter ─────────────────────────────────────────────────────

test("the limiter halves on a rate limit and never reaches zero", () => {
  // A ceiling of zero deadlocks the whole run.
  const limiter = new AdaptiveLimiter(8);
  limiter.onRateLimit();
  assert.equal(limiter.limit, 4);
  for (let i = 0; i < 10; i += 1) limiter.onRateLimit();
  assert.ok(limiter.limit >= 1);
});

test("an idle limiter still recovers", () => {
  // Regression guard shared with the Python SDK: an earlier design tracked
  // shrinkage as debt paid off by later releases, so a limiter with nothing in
  // flight could never grow back — it sat at half speed for the rest of the run.
  const limiter = new AdaptiveLimiter(8);
  limiter.onRateLimit();
  for (let i = 0; i < 20; i += 1) limiter.onSuccess();
  assert.equal(limiter.limit, 8, "recovers, but never above the configured ceiling");
});

test("shrinking takes effect immediately for new work", async () => {
  const limiter = new AdaptiveLimiter(4);
  for (let i = 0; i < 4; i += 1) await limiter.acquire();
  limiter.onRateLimit(); // 4 -> 2
  for (let i = 0; i < 4; i += 1) limiter.release();

  await limiter.acquire();
  await limiter.acquire();

  let granted = false;
  void limiter.acquire().then(() => {
    granted = true;
  });
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(granted, false, "a third slot must not exist at limit 2");
});
