/**
 * Offline batch inference — submit many requests, come back later, pay half.
 *
 * Several providers run a second, asynchronous tier: you hand over a whole job,
 * they process it within a window (usually 24 hours, often much sooner), and it
 * costs roughly 50% of the synchronous price. For evals, backfills and
 * classification sweeps — anything where nobody is waiting on the answer — that
 * is the difference between a run you can afford and one you cannot.
 *
 * No two providers agree on how it works:
 *
 *   OpenAI-style   upload a JSONL file, reference the file id in a job, poll the
 *                  job, then download a second file of results.
 *   Anthropic      send the requests inline in the job itself, poll, then stream
 *                  a JSONL from a URL the job hands back.
 *
 * Different verbs, field names, status vocabularies and result envelopes. This
 * normalizes all of it to `submit` / `status` / `results` / `cancel`, plus `run`
 * when you just want the answers and are happy to wait.
 *
 * **Not the same thing as continuous batching.** This is an offline job queued
 * for later. Continuous batching is a server-side scheduling technique that no
 * client can perform; see `concurrent.ts` for the client-side counterpart.
 */
import {
  PolygateError,
  ProviderAPIError,
  UnsupportedProviderError,
} from "./errors";
import { resolveKeys } from "./retry";
import type { KeyInput } from "./retry";
import { ChatResponse } from "./types";

/** Batches run for hours, so a tight poll loop buys only rate-limit errors. */
export const DEFAULT_POLL_SECONDS = 30;

/**
 * Thrown for a provider with no asynchronous batch tier.
 *
 * Its own type because the fallback is real and worth naming: everything
 * polygate reaches synchronously can be driven through `mapChat`, which costs
 * full price but works everywhere and returns in seconds rather than hours.
 */
export class BatchNotSupportedError extends UnsupportedProviderError {
  constructor(message: string) {
    super(message);
    this.name = "BatchNotSupportedError";
  }
}

/** Terminal states. Polling stops here. */
export const TERMINAL = new Set([
  "completed", "failed", "cancelled", "expired",
]);

export interface BatchRequest {
  /**
   * How a result is matched back to its input. Required rather than generated,
   * because results come back unordered and often partially — matching by
   * position would quietly mis-assign every answer in a job where one request
   * failed.
   */
  customId: string;
  model: string;
  messages: { role: string; content: string }[];
  temperature?: number;
  maxTokens?: number;
  extra?: Record<string, unknown>;
}

export interface BatchJob {
  id: string;
  provider: string;
  /** pending | running | completed | failed | cancelled | expired */
  status: string;
  total: number;
  completed: number;
  failed: number;
  /** The provider's untouched job body, for anything not normalized here. */
  raw: Record<string, any>;
}

export interface BatchResult {
  customId: string;
  /** Null when the request failed. Never both this and `error`. */
  response: ChatResponse | null;
  error: string | null;
}

export function isDone(job: BatchJob): boolean {
  return TERMINAL.has(job.status);
}

export function progress(job: BatchJob): number {
  return job.total ? (job.completed + job.failed) / job.total : 0;
}

/**
 * OpenAI's vocabulary mapped to ours. "finalizing" is deliberately *not*
 * terminal: the results file does not exist yet, and treating it as done
 * returns an empty set that looks like a job with no output.
 */
const NORMALIZE_OPENAI: Record<string, string> = {
  validating: "pending",
  in_progress: "running",
  finalizing: "running",
  completed: "completed",
  failed: "failed",
  expired: "expired",
  cancelling: "running",
  cancelled: "cancelled",
  canceled: "cancelled",
};

/** Anthropic reports only two states; per-request counts distinguish the rest. */
const NORMALIZE_ANTHROPIC: Record<string, string> = {
  in_progress: "running",
  canceling: "running",
  ended: "completed",
};

interface Backend {
  provider: string;
  envVar: string;
  completionWindow?: string;
  submit(key: string, requests: BatchRequest[]): Promise<BatchJob>;
  status(key: string, batchId: string): Promise<BatchJob>;
  results(key: string, batchId: string): Promise<BatchResult[]>;
  cancel(key: string, batchId: string): Promise<BatchJob>;
}

async function check(response: Response, provider: string): Promise<any> {
  if (!response.ok) {
    throw new ProviderAPIError(
      provider, response.status, (await response.text()).slice(0, 500)
    );
  }
  return response.json();
}

// ── OpenAI-style: upload a file, then reference it in a job ──────────────────

class OpenAIStyle implements Backend {
  constructor(
    public provider: string,
    private base: string,
    public envVar: string,
    public completionWindow = "24h"
  ) {
    this.base = base.replace(/\/+$/, "");
  }

  private headers(key: string) {
    return { Authorization: `Bearer ${key}` };
  }

  async submit(key: string, requests: BatchRequest[]): Promise<BatchJob> {
    const endpoint = "/v1/chat/completions";
    const lines = requests.map((item) => {
      const body: Record<string, unknown> = {
        model: item.model,
        messages: item.messages,
      };
      if (item.temperature !== undefined) body.temperature = item.temperature;
      if (item.maxTokens !== undefined) body.max_tokens = item.maxTokens;
      Object.assign(body, item.extra ?? {});
      return JSON.stringify({
        custom_id: item.customId, method: "POST", url: endpoint, body,
      });
    });

    const form = new FormData();
    form.append("purpose", "batch");
    form.append(
      "file",
      new Blob([lines.join("\n")], { type: "application/jsonl" }),
      "batch.jsonl"
    );

    const uploaded = await check(
      await fetch(`${this.base}/files`, {
        method: "POST", headers: this.headers(key), body: form,
      }),
      this.provider
    );
    const created = await check(
      await fetch(`${this.base}/batches`, {
        method: "POST",
        headers: { ...this.headers(key), "Content-Type": "application/json" },
        body: JSON.stringify({
          input_file_id: uploaded.id,
          endpoint,
          completion_window: this.completionWindow,
        }),
      }),
      this.provider
    );
    return this.toJob(created);
  }

  async status(key: string, batchId: string): Promise<BatchJob> {
    return this.toJob(
      await check(
        await fetch(`${this.base}/batches/${batchId}`, { headers: this.headers(key) }),
        this.provider
      )
    );
  }

  async cancel(key: string, batchId: string): Promise<BatchJob> {
    return this.toJob(
      await check(
        await fetch(`${this.base}/batches/${batchId}/cancel`, {
          method: "POST", headers: this.headers(key),
        }),
        this.provider
      )
    );
  }

  async results(key: string, batchId: string): Promise<BatchResult[]> {
    const job = await this.status(key, batchId);
    const out: BatchResult[] = [];
    // Successes and failures land in two separate files; reading only the
    // output file silently loses every failed request.
    for (const [fileId, isError] of [
      [job.raw.output_file_id, false],
      [job.raw.error_file_id, true],
    ] as [string | undefined, boolean][]) {
      if (!fileId) continue;
      const response = await fetch(`${this.base}/files/${fileId}/content`, {
        headers: this.headers(key),
      });
      if (!response.ok) {
        throw new ProviderAPIError(
          this.provider, response.status, (await response.text()).slice(0, 500)
        );
      }
      const text = await response.text();
      for (const line of text.split("\n")) {
        if (line.trim()) out.push(this.toResult(JSON.parse(line), isError));
      }
    }
    return out;
  }

  private toJob(data: any): BatchJob {
    const counts = data.request_counts ?? {};
    return {
      id: data.id ?? "",
      provider: this.provider,
      status: NORMALIZE_OPENAI[data.status] ?? "running",
      total: counts.total ?? 0,
      completed: counts.completed ?? 0,
      failed: counts.failed ?? 0,
      raw: data,
    };
  }

  private toResult(row: any, isError: boolean): BatchResult {
    const customId = row.custom_id ?? "";
    if (isError || row.error) {
      return {
        customId,
        response: null,
        error: JSON.stringify(row.error ?? row).slice(0, 500),
      };
    }
    const body = row.response?.body ?? {};
    const message = body.choices?.[0]?.message ?? {};
    const usage = body.usage ?? {};
    return {
      customId,
      error: null,
      response: {
        content: message.content || message.reasoning_content || "",
        role: message.role ?? "assistant",
        model: body.model ?? "",
        provider: this.provider,
        usage: {
          promptTokens: usage.prompt_tokens ?? 0,
          completionTokens: usage.completion_tokens ?? 0,
          totalTokens: usage.total_tokens ?? 0,
        },
        raw: body,
      },
    };
  }
}

// ── Anthropic: requests inline, results streamed from a URL ──────────────────

class AnthropicStyle implements Backend {
  provider = "anthropic";

  constructor(
    private base = "https://api.anthropic.com/v1",
    public envVar = "ANTHROPIC_API_KEY"
  ) {}

  private headers(key: string) {
    return {
      "x-api-key": key,
      "anthropic-version": "2023-06-01",
      "Content-Type": "application/json",
    };
  }

  async submit(key: string, requests: BatchRequest[]): Promise<BatchJob> {
    const payload = requests.map((item) => {
      const { system, turns } = splitSystem(item.messages);
      const params: Record<string, unknown> = {
        model: item.model,
        messages: turns,
        // Required by the Messages API, and a batch missing it is rejected only
        // after the whole job has been accepted.
        max_tokens: item.maxTokens ?? 1024,
      };
      if (system) params.system = system;
      if (item.temperature !== undefined) params.temperature = item.temperature;
      Object.assign(params, item.extra ?? {});
      return { custom_id: item.customId, params };
    });

    return this.toJob(
      await check(
        await fetch(`${this.base}/messages/batches`, {
          method: "POST",
          headers: this.headers(key),
          body: JSON.stringify({ requests: payload }),
        }),
        this.provider
      )
    );
  }

  async status(key: string, batchId: string): Promise<BatchJob> {
    return this.toJob(
      await check(
        await fetch(`${this.base}/messages/batches/${batchId}`, {
          headers: this.headers(key),
        }),
        this.provider
      )
    );
  }

  async cancel(key: string, batchId: string): Promise<BatchJob> {
    return this.toJob(
      await check(
        await fetch(`${this.base}/messages/batches/${batchId}/cancel`, {
          method: "POST", headers: this.headers(key),
        }),
        this.provider
      )
    );
  }

  async results(key: string, batchId: string): Promise<BatchResult[]> {
    const job = await this.status(key, batchId);
    const url = job.raw.results_url;
    if (!url) {
      // Ended without results means the job failed or expired, which is not the
      // same as "no requests succeeded" and must not look like an empty success.
      if (TERMINAL.has(job.status) && job.status !== "completed") {
        throw new PolygateError(
          `Anthropic batch ${batchId} ended as ${job.status} with no results.`
        );
      }
      return [];
    }
    const response = await fetch(url, { headers: this.headers(key) });
    if (!response.ok) {
      throw new ProviderAPIError(
        this.provider, response.status, (await response.text()).slice(0, 500)
      );
    }
    const text = await response.text();
    return text
      .split("\n")
      .filter((line) => line.trim())
      .map((line) => this.toResult(JSON.parse(line)));
  }

  private toJob(data: any): BatchJob {
    const counts = data.request_counts ?? {};
    const succeeded = counts.succeeded ?? 0;
    const errored = counts.errored ?? 0;
    const canceled = counts.canceled ?? 0;
    const expired = counts.expired ?? 0;
    const processing = counts.processing ?? 0;
    return {
      id: data.id ?? "",
      provider: this.provider,
      status: NORMALIZE_ANTHROPIC[data.processing_status] ?? "running",
      total: succeeded + errored + canceled + expired + processing,
      completed: succeeded,
      // Cancelled and expired requests never ran. Counting them as successes
      // overstates the job; omitting them stalls progress below 1 forever.
      failed: errored + canceled + expired,
      raw: data,
    };
  }

  private toResult(row: any): BatchResult {
    const customId = row.custom_id ?? "";
    const result = row.result ?? {};
    if (result.type !== "succeeded") {
      return {
        customId,
        response: null,
        error: JSON.stringify(result.error ?? { type: result.type }).slice(0, 500),
      };
    }
    const message = result.message ?? {};
    const usage = message.usage ?? {};
    const blocks: any[] = message.content ?? [];
    return {
      customId,
      error: null,
      response: {
        content: blocks.map((b) => b?.text ?? "").join(""),
        role: message.role ?? "assistant",
        model: message.model ?? "",
        provider: this.provider,
        usage: {
          promptTokens: usage.input_tokens ?? 0,
          completionTokens: usage.output_tokens ?? 0,
          totalTokens: (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0),
        },
        raw: message,
      },
    };
  }
}

/** Anthropic takes the system prompt beside the messages, not inside them. */
export function splitSystem(messages: { role: string; content: string }[]) {
  const system = messages
    .filter((m) => m.role === "system")
    .map((m) => m.content)
    .filter(Boolean)
    .join("\n\n");
  const turns = messages
    .filter((m) => m.role !== "system")
    .map((m) => ({
      role: m.role === "assistant" ? "assistant" : "user",
      content: m.content ?? "",
    }));
  return { system, turns };
}

/**
 * Only providers with a documented batch tier. Deliberately not every
 * OpenAI-compatible host: most implement the chat route and not the batch route,
 * and a request to a missing /batches returns a 404 that reads like an outage
 * rather than an unsupported feature.
 */
const BACKENDS: Record<string, Backend> = {
  openai: new OpenAIStyle("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
  anthropic: new AnthropicStyle(),
  groq: new OpenAIStyle(
    "groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
    // Groq recommends the longer window: under load a 24h job is likelier to
    // expire than to finish, and an expired batch must be resubmitted whole.
    "7d"
  ),
};

export const BATCH_PROVIDERS = Object.keys(BACKENDS).sort();

function backendFor(provider: string): Backend {
  let key = (provider ?? "").toLowerCase().trim();
  // Aliases resolve through the main registry so "claude" and "gpt" work here
  // exactly as they do for chat.
  const canonical = ALIASES[key];
  if (canonical) key = canonical;
  const backend = BACKENDS[key];
  if (!backend) {
    throw new BatchNotSupportedError(
      `'${provider}' has no batch API. Supported: ${BATCH_PROVIDERS.join(", ")}. ` +
        `For everything else use mapChat(), which runs requests concurrently ` +
        `at normal prices.`
    );
  }
  return backend;
}

/** Only the aliases that can reach a batch-capable provider. */
const ALIASES: Record<string, string> = {
  gpt: "openai",
  claude: "anthropic",
};

function keyFor(backend: Backend, apiKey?: KeyInput): string {
  // Batch jobs are long-lived, so one key is used rather than a pool. Rotating
  // mid-job would submit under one credential and poll under another, and
  // providers scope a batch id to the key that created it.
  return resolveKeys(apiKey, backend.envVar, backend.provider).acquire()!;
}

// ── Public API ───────────────────────────────────────────────────────────────

export async function submit(
  provider: string,
  requests: BatchRequest[],
  options: { apiKey?: KeyInput } = {}
): Promise<BatchJob> {
  const backend = backendFor(provider);
  if (!requests.length) throw new Error("A batch needs at least one request.");
  const ids = new Set(requests.map((r) => r.customId));
  if (ids.size !== requests.length) {
    // Providers key results by customId, so duplicates make some results
    // unreachable — and it fails silently at retrieval, hours later.
    throw new Error("customId must be unique within a batch.");
  }
  return backend.submit(keyFor(backend, options.apiKey), requests);
}

export async function status(
  provider: string,
  batchId: string,
  options: { apiKey?: KeyInput } = {}
): Promise<BatchJob> {
  const backend = backendFor(provider);
  return backend.status(keyFor(backend, options.apiKey), batchId);
}

export async function results(
  provider: string,
  batchId: string,
  options: { apiKey?: KeyInput } = {}
): Promise<BatchResult[]> {
  const backend = backendFor(provider);
  return backend.results(keyFor(backend, options.apiKey), batchId);
}

export async function cancel(
  provider: string,
  batchId: string,
  options: { apiKey?: KeyInput } = {}
): Promise<BatchJob> {
  const backend = backendFor(provider);
  return backend.cancel(keyFor(backend, options.apiKey), batchId);
}

export interface RunOptions {
  apiKey?: KeyInput;
  pollSeconds?: number;
  timeoutSeconds?: number;
  onProgress?: (job: BatchJob) => void;
  /** Injectable for tests; defaults to a real timer. */
  sleep?: (seconds: number) => Promise<void>;
}

/**
 * Submit, wait, and return the results.
 *
 * Convenient and occasionally the wrong tool: this blocks for as long as the
 * provider takes, which can be hours. Prefer `submit` plus your own polling for
 * anything that must survive a restart — the batch id is durable, this call's
 * stack frame is not.
 *
 * `timeoutSeconds` gives up waiting; it does **not** cancel the job, which keeps
 * running and can be collected later with `results`. Abandoning a paid job
 * silently would be worse than reporting that we stopped watching.
 */
export async function run(
  provider: string,
  requests: BatchRequest[],
  options: RunOptions = {}
): Promise<BatchResult[]> {
  const {
    pollSeconds = DEFAULT_POLL_SECONDS,
    timeoutSeconds,
    onProgress,
    sleep = (s: number) => new Promise((r) => setTimeout(r, s * 1000)),
  } = options;

  let job = await submit(provider, requests, options);
  const started = Date.now();
  while (!isDone(job)) {
    if (timeoutSeconds !== undefined && (Date.now() - started) / 1000 > timeoutSeconds) {
      throw new Error(
        `Batch ${job.id} still ${job.status} after ${timeoutSeconds}s. It is ` +
          `still running — collect it later with results('${provider}', '${job.id}').`
      );
    }
    await sleep(pollSeconds);
    job = await status(provider, job.id, options);
    onProgress?.(job);
  }
  if (job.status !== "completed") {
    throw new PolygateError(`Batch ${job.id} ended as ${job.status}.`);
  }
  return results(provider, job.id, options);
}
