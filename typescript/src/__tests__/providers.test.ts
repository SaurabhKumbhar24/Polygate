/**
 * The provider registry and the shared OpenAI-compatible adapter.
 *
 * One adapter backs thirteen providers, which is only safe if the things that do
 * differ — URL, environment variable, reported name — stay per-provider. That
 * is what these pin.
 */
import assert from "node:assert";
import { afterEach, test } from "node:test";
import { chat } from "../client";
import { UnsupportedProviderError } from "../errors";
import {
  CANONICAL_PROVIDERS,
  SUPPORTED_PROVIDERS,
  getProvider,
} from "../providers";
import { parse, resolveUrl } from "../providers/openaiCompatible";
import * as zai from "../providers/zai";

/** provider → [module host fragment, env var] */
const COMPATIBLE: [string, string, string][] = [
  ["openai", "api.openai.com", "OPENAI_API_KEY"],
  ["groq", "api.groq.com", "GROQ_API_KEY"],
  ["together", "api.together.xyz", "TOGETHER_API_KEY"],
  ["fireworks", "api.fireworks.ai", "FIREWORKS_API_KEY"],
  ["mistral", "api.mistral.ai", "MISTRAL_API_KEY"],
  ["xai", "api.x.ai", "XAI_API_KEY"],
  ["perplexity", "api.perplexity.ai", "PERPLEXITY_API_KEY"],
  ["cerebras", "api.cerebras.ai", "CEREBRAS_API_KEY"],
  ["deepseek", "api.deepseek.com", "DEEPSEEK_API_KEY"],
  ["moonshot", "api.moonshot.ai", "MOONSHOT_API_KEY"],
  ["zai", "api.z.ai", "ZAI_API_KEY"],
];

const realFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = realFetch;
});

function stubFetch(captured: { url?: string; headers?: any }) {
  globalThis.fetch = (async (url: string, init: any) => {
    captured.url = String(url);
    captured.headers = init?.headers;
    return {
      ok: true,
      status: 200,
      json: async () => ({
        model: "m",
        choices: [{ message: { role: "assistant", content: "hi" } }],
        usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 },
      }),
    };
  }) as any;
}

// ── Registry ─────────────────────────────────────────────────────────────────

test("popular names resolve alongside canonical ones", () => {
  // People reach for "grok" long before "xai".
  const aliases: [string, string][] = [
    ["grok", "xai"],
    ["sonar", "perplexity"],
    ["ollama", "custom"],
    ["kimi", "moonshot"],
    ["claude", "anthropic"],
    ["glm", "zai"],
    ["z.ai", "zai"],
    ["zhipu", "zai"],
    ["bigmodel", "zai"],
  ];
  for (const [alias, _canonical] of aliases) {
    assert.ok(SUPPORTED_PROVIDERS.includes(alias), `${alias} missing`);
    assert.doesNotThrow(() => getProvider(alias));
  }
});

test("provider names are case and space insensitive", () => {
  assert.doesNotThrow(() => getProvider("  OpenAI "));
});

test("an unknown provider lists the known ones", () => {
  assert.throws(() => getProvider("nope"), UnsupportedProviderError);
});

test("canonical list holds no aliases and no duplicates", () => {
  assert.ok(CANONICAL_PROVIDERS.length < SUPPORTED_PROVIDERS.length);
  assert.equal(new Set(CANONICAL_PROVIDERS).size, CANONICAL_PROVIDERS.length);
  for (const name of CANONICAL_PROVIDERS) {
    assert.ok(SUPPORTED_PROVIDERS.includes(name), `${name} not registered`);
  }
});

// ── Each provider keeps its own endpoint and key ─────────────────────────────

for (const [provider, host, envVar] of COMPATIBLE) {
  test(`${provider} posts to its own host with its own key`, async () => {
    // A shared implementation collapsing onto one URL — or one env var — would
    // send every provider to the wrong place with the wrong credential.
    const saved: Record<string, string | undefined> = {};
    for (const [, , other] of COMPATIBLE) {
      saved[other] = process.env[other];
      delete process.env[other];
    }
    process.env[envVar] = "sentinel-key";

    const captured: { url?: string; headers?: any } = {};
    stubFetch(captured);

    const result = await chat({
      provider,
      model: "m",
      messages: [{ role: "user", content: "x" }],
    });

    assert.ok(captured.url?.includes(host), `${provider} hit ${captured.url}`);
    assert.equal(captured.headers.Authorization, "Bearer sentinel-key");
    assert.equal(result.provider, provider);

    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });
}

test("baseUrl overrides the default host", async () => {
  process.env.OPENAI_API_KEY = "k";
  const captured: { url?: string } = {};
  stubFetch(captured);

  await chat({
    provider: "openai",
    model: "m",
    messages: [{ role: "user", content: "x" }],
    baseUrl: "https://proxy.internal/v1",
  });
  assert.equal(captured.url, "https://proxy.internal/v1/chat/completions");
});

// ── Custom / local endpoints ─────────────────────────────────────────────────

test("a custom endpoint needs no api key", async () => {
  // A laptop running Ollama authenticates nothing.
  delete process.env.CUSTOM_API_KEY;
  const captured: { url?: string } = {};
  stubFetch(captured);

  const result = await chat({
    provider: "ollama",
    model: "llama3.1:8b",
    messages: [{ role: "user", content: "x" }],
    baseUrl: "http://localhost:11434/v1",
  });
  assert.equal(result.provider, "custom");
  assert.equal(captured.url, "http://localhost:11434/v1/chat/completions");
});

test("custom without an endpoint says so rather than guessing localhost", async () => {
  delete process.env.POLYGATE_BASE_URL;
  await assert.rejects(
    () => chat({ provider: "custom", model: "m", messages: [] }),
    UnsupportedProviderError
  );
});

// ── URL normalisation ────────────────────────────────────────────────────────

test("a bare root gains the completions path", () => {
  assert.equal(resolveUrl("http://x/v1"), "http://x/v1/chat/completions");
});

test("a full url is left alone", () => {
  const full = "http://x/v1/chat/completions";
  assert.equal(resolveUrl(full), full);
});

test("a trailing slash does not double up", () => {
  assert.equal(resolveUrl("http://x/v1/"), "http://x/v1/chat/completions");
});

// ── Response parsing ─────────────────────────────────────────────────────────

test("missing usage is zeros, not an error", () => {
  // "OpenAI-compatible" is a claim, not a standard.
  const result = parse(
    { choices: [{ message: { content: "hi" } }] },
    "groq",
    "m"
  );
  assert.equal(result.usage.totalTokens, 0);
  assert.equal(result.content, "hi");
});

test("reasoning_content is used when content is empty", () => {
  const result = parse(
    { choices: [{ message: { content: "", reasoning_content: "answer" } }] },
    "deepseek",
    "m"
  );
  assert.equal(result.content, "answer");
});

test("an empty choices list does not crash", () => {
  const result = parse({ choices: [] }, "groq", "fallback");
  assert.equal(result.content, "");
  assert.equal(result.model, "fallback");
});

test("raw carries the untouched body", () => {
  // Perplexity returns citations no other provider does.
  const body = { choices: [{ message: { content: "x" } }], citations: ["https://a"] };
  const result = parse(body, "perplexity", "m") as any;
  assert.deepEqual(result.raw.citations, ["https://a"]);
});

// ── Z.AI ─────────────────────────────────────────────────────────────────────

test("zai keeps its unusual /api/paas/v4 path", () => {
  // Z.AI serves the OpenAI format from /api/paas/v4, not /v1. Getting this
  // wrong yields a 404 that reads like a dead provider rather than a bad path.
  assert.equal(zai.API_URL, "https://api.z.ai/api/paas/v4/chat/completions");
});

test("the brand spelling with a dot resolves", () => {
  // "z.ai" is how the name is written everywhere it appears; a user typing it
  // the way the vendor does must not be told the provider is unknown.
  assert.doesNotThrow(() => getProvider("z.ai"));
  assert.doesNotThrow(() => getProvider(" Z.AI "));
});

test("the mainland host is reachable without a second provider", async () => {
  // Zhipu's mainland platform is the same API on a different host, and keys do
  // not cross between them. baseUrl covers it, so there is no near-duplicate
  // provider to keep in step.
  process.env.ZAI_API_KEY = "k";
  let seen = "";
  globalThis.fetch = (async (url: string) => {
    seen = String(url);
    return {
      ok: true,
      status: 200,
      json: async () => ({ choices: [{ message: { content: "hi" } }] }),
    };
  }) as unknown as typeof fetch;

  await chat({
    provider: "zai",
    baseUrl: zai.CHINA_BASE_URL,
    model: "glm-4.6",
    messages: [{ role: "user", content: "hi" }],
  });
  assert.equal(seen, "https://open.bigmodel.cn/api/paas/v4/chat/completions");
});
