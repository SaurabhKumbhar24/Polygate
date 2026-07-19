import assert from "node:assert";
import { test } from "node:test";
import { chat } from "../client";
import { MissingAPIKeyError, UnsupportedProviderError } from "../errors";

test("throws on unsupported provider", async () => {
  await assert.rejects(
    () => chat({ provider: "not-real", model: "x", messages: [] }),
    UnsupportedProviderError
  );
});

test("throws when no api key is available", async () => {
  const original = process.env.OPENAI_API_KEY;
  delete process.env.OPENAI_API_KEY;

  await assert.rejects(
    () =>
      chat({
        provider: "openai",
        model: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
      }),
    MissingAPIKeyError
  );

  if (original !== undefined) process.env.OPENAI_API_KEY = original;
});

test("openai chat returns a normalized response", async () => {
  const originalFetch = global.fetch;
  global.fetch = (async () =>
    ({
      ok: true,
      json: async () => ({
        model: "gpt-4o",
        choices: [{ message: { role: "assistant", content: "Hello!" } }],
        usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
      }),
    }) as unknown as Response) as typeof fetch;

  const result = await chat({
    provider: "openai",
    model: "gpt-4o",
    messages: [{ role: "user", content: "hi" }],
    apiKey: "test-key",
  });

  assert.strictEqual(result.content, "Hello!");
  assert.strictEqual(result.provider, "openai");
  assert.strictEqual(result.usage.totalTokens, 7);

  global.fetch = originalFetch;
});

test("anthropic chat returns a normalized response", async () => {
  const originalFetch = global.fetch;
  global.fetch = (async () =>
    ({
      ok: true,
      json: async () => ({
        model: "claude-sonnet-4-6",
        content: [{ type: "text", text: "Hi there!" }],
        usage: { input_tokens: 4, output_tokens: 3 },
      }),
    }) as unknown as Response) as typeof fetch;

  const result = await chat({
    provider: "claude",
    model: "claude-sonnet-4-6",
    messages: [{ role: "user", content: "hi" }],
    apiKey: "test-key",
  });

  assert.strictEqual(result.content, "Hi there!");
  assert.strictEqual(result.provider, "anthropic");
  assert.strictEqual(result.usage.totalTokens, 7);

  global.fetch = originalFetch;
});

test("gemini chat returns a normalized response", async () => {
  const originalFetch = global.fetch;
  global.fetch = (async () =>
    ({
      ok: true,
      json: async () => ({
        candidates: [{ content: { parts: [{ text: "Bonjour!" }] } }],
        usageMetadata: {
          promptTokenCount: 3,
          candidatesTokenCount: 2,
          totalTokenCount: 5,
        },
      }),
    }) as unknown as Response) as typeof fetch;

  const result = await chat({
    provider: "gemini",
    model: "gemini-2.0-flash",
    messages: [{ role: "user", content: "hi" }],
    apiKey: "test-key",
  });

  assert.strictEqual(result.content, "Bonjour!");
  assert.strictEqual(result.provider, "gemini");

  global.fetch = originalFetch;
});

test("moonshot chat returns a normalized response", async () => {
  const originalFetch = global.fetch;
  global.fetch = (async () =>
    ({
      ok: true,
      json: async () => ({
        model: "kimi-k2-0711-preview",
        choices: [{ message: { role: "assistant", content: "Ni hao!" } }],
        usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 },
      }),
    }) as unknown as Response) as typeof fetch;

  const result = await chat({
    provider: "kimi",
    model: "kimi-k2-0711-preview",
    messages: [{ role: "user", content: "hi" }],
    apiKey: "test-key",
  });

  assert.strictEqual(result.content, "Ni hao!");
  assert.strictEqual(result.provider, "moonshot");

  global.fetch = originalFetch;
});
