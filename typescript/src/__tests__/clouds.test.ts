/**
 * Clouds and gateways in the TypeScript SDK.
 *
 * The signing is the part worth the most attention: it is hand-rolled to avoid
 * a heavyweight AWS SDK, which is only defensible if it is provably the same
 * algorithm. The Python suite diffs its signer against botocore; this file pins
 * the TS signer to the same expected output, so the two SDKs cannot drift apart
 * or away from AWS.
 */
import assert from "node:assert";
import { afterEach, test } from "node:test";
import { chat } from "../client";
import { MissingEndpointError } from "../errors";
import * as azure from "../providers/azure";
import * as bedrock from "../providers/bedrock";
import * as cloudflare from "../providers/cloudflare";
import * as databricks from "../providers/databricks";
import * as vertex from "../providers/vertex";
import { encodePath, signRequest, splitCredentials } from "../providers/sigv4";

const realFetch = globalThis.fetch;
const FIXED = new Date("2026-08-16T12:00:00.000Z");

afterEach(() => {
  globalThis.fetch = realFetch;
  for (const key of [
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_REGION",
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION",
    "DATABRICKS_TOKEN", "DATABRICKS_HOST", "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_BASE_URL", "GOOGLE_ACCESS_TOKEN",
    "VERTEX_BASE_URL",
  ]) {
    delete process.env[key];
  }
});

function stub(payload: unknown, capture?: (url: string, init: any) => void) {
  globalThis.fetch = (async (url: string, init: any) => {
    capture?.(String(url), init);
    return { ok: true, status: 200, json: async () => payload };
  }) as unknown as typeof fetch;
}

// ── SigV4 ────────────────────────────────────────────────────────────────────

test("credentials unpack with and without a session token", () => {
  assert.deepEqual(splitCredentials("AKIA:secret"), {
    accessKeyId: "AKIA", secretAccessKey: "secret", sessionToken: "",
  });
  assert.equal(splitCredentials("AKIA:secret:tok").sessionToken, "tok");
});

test("a malformed credential says what the format is", () => {
  // A bare access key would otherwise sign with an empty secret and come back
  // as a 403 that reads like a permissions problem.
  assert.throws(
    () => splitCredentials("just-a-key"),
    /ACCESS_KEY_ID:SECRET_ACCESS_KEY/
  );
});

test("the canonical path is double encoded", () => {
  // SigV4 encodes path segments twice for every service except S3, so a model
  // id ending "v2:0" is sent as "v2%3A0" and signed as "v2%253A0".
  const path = encodePath(
    "model", "anthropic.claude-3-5-sonnet-20241022-v2:0", "converse"
  );
  assert.equal(
    path,
    "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse"
  );
});

test("the TS signer matches the Python signer, which matches botocore", () => {
  // These two strings were produced by the Python implementation, which the
  // Python suite diffs against botocore. Pinning them here chains the guarantee
  // across all three: if the TS signer drifts from Python, or Python from AWS,
  // one of the two suites fails. Without this the SDKs could silently disagree,
  // and the symptom would be a 403 in one language only.
  //
  // The body is written out literally rather than built with JSON.stringify,
  // because the two languages serialize the same object differently —
  // json.dumps inserts a space after every colon and comma, JSON.stringify does
  // not. Each SDK signs the bytes it actually sends, so that difference is
  // harmless in production but fatal to a cross-language comparison.
  const SECRET = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
  const body = '{"messages":[{"role":"user","content":[{"text":"hi"}]}]}';
  const call = (model: string, token?: string) =>
    signRequest({
      key: `AKIDEXAMPLE:${SECRET}` + (token ? `:${token}` : ""),
      region: "us-east-1",
      service: "bedrock",
      host: "bedrock-runtime.us-east-1.amazonaws.com",
      path: encodePath("model", model, "converse"),
      body,
      now: FIXED,
    }).Authorization;

  assert.equal(
    call("anthropic.claude-3-5-sonnet-20241022-v2:0"),
    "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260816/us-east-1/bedrock/aws4_request, " +
      "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date, " +
      "Signature=729ee740226f12c7f319edf372ae222c672bdc330462f5f7257f66eeb5369e3f"
  );
  assert.equal(
    call("us.anthropic.claude-sonnet-4-20250514-v1:0", "tok-123"),
    "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260816/us-east-1/bedrock/aws4_request, " +
      "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token, " +
      "Signature=911bcd8cfa1c731bb46a8bc4e50ca386c63f321d0215aeba856499d1fddbcc4a"
  );
});

test("the signature covers the body", () => {
  const base = {
    key: "AKIA:secret", region: "us-east-1", service: "bedrock",
    host: "h", path: "/p", now: FIXED,
  };
  assert.notEqual(
    signRequest({ ...base, body: JSON.stringify({ a: 1 }) }).Authorization,
    signRequest({ ...base, body: JSON.stringify({ a: 2 }) }).Authorization
  );
});

test("the signature is scoped to region and service", () => {
  const base = {
    key: "AKIA:secret", host: "h", path: "/p", body: "{}", now: FIXED,
  };
  const east = signRequest({ ...base, region: "us-east-1", service: "bedrock" });
  const west = signRequest({ ...base, region: "us-west-2", service: "bedrock" });
  assert.notEqual(east.Authorization, west.Authorization);
  assert.ok(east.Authorization.includes("us-east-1/bedrock/aws4_request"));
});

test("a session token is signed and sent", () => {
  const headers = signRequest({
    key: "AKIA:secret:tok123", region: "us-east-1", service: "bedrock",
    host: "h", path: "/p", body: "{}", now: FIXED,
  });
  assert.equal(headers["X-Amz-Security-Token"], "tok123");
  assert.ok(headers.Authorization.includes("x-amz-security-token"));
});

// ── Bedrock ──────────────────────────────────────────────────────────────────

test("system messages move to their own field", () => {
  const { system, turns } = bedrock.toConverse([
    { role: "system", content: "be brief" },
    { role: "user", content: "hello" },
  ]);
  assert.deepEqual(system, [{ text: "be brief" }]);
  assert.deepEqual(turns, [{ role: "user", content: [{ text: "hello" }] }]);
});

test("converse usage is normalized", () => {
  // Bedrock reports inputTokens/outputTokens; everything else reports
  // promptTokens/completionTokens. Callers must not have to know which.
  const result = bedrock.parse(
    {
      output: { message: { role: "assistant", content: [{ text: "hi" }] } },
      usage: { inputTokens: 10, outputTokens: 4, totalTokens: 14 },
    },
    "anthropic.claude-3-5-sonnet-20241022-v2:0"
  );
  assert.equal(result.content, "hi");
  assert.equal(result.usage.promptTokens, 10);
  assert.equal(result.usage.completionTokens, 4);
});

test("a tool use block does not break the parser", () => {
  // Tool blocks carry no text key at all.
  const result = bedrock.parse(
    { output: { message: { content: [{ toolUse: { name: "s" } }, { text: "prose" }] } } },
    "m"
  );
  assert.equal(result.content, "prose");
});

test("bedrock posts to the regional converse url", async () => {
  process.env.AWS_ACCESS_KEY_ID = "AKIA";
  process.env.AWS_SECRET_ACCESS_KEY = "secret";
  process.env.AWS_REGION = "eu-west-1";
  let seen = "";
  stub({ output: { message: { content: [{ text: "hi" }] } } }, (url) => {
    seen = url;
  });

  await chat({
    provider: "bedrock",
    model: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages: [{ role: "user", content: "hi" }],
  });
  assert.ok(seen.startsWith("https://bedrock-runtime.eu-west-1.amazonaws.com/model/"));
  assert.ok(seen.endsWith("/converse"));
  assert.ok(seen.includes("%3A0"), "the colon in the model id must be encoded");
});

test("the standard AWS variables work unpacked", async () => {
  // Nobody has AWS_ACCESS_KEY_ID holding "id:secret" — they have the two real
  // variables the AWS CLI writes.
  process.env.AWS_ACCESS_KEY_ID = "AKIA";
  process.env.AWS_SECRET_ACCESS_KEY = "secret";
  let headers: any = {};
  stub({ output: { message: { content: [{ text: "x" }] } } }, (_url, init) => {
    headers = init.headers;
  });
  await chat({
    provider: "bedrock", model: "m",
    messages: [{ role: "user", content: "hi" }],
  });
  assert.ok(headers.Authorization.startsWith("AWS4-HMAC-SHA256"));
});

// ── Azure ────────────────────────────────────────────────────────────────────

test("azure puts the key in its own header, not a Bearer token", async () => {
  process.env.AZURE_OPENAI_API_KEY = "k";
  let headers: any = {};
  stub({ choices: [{ message: { content: "hi" } }] }, (_url, init) => {
    headers = init.headers;
  });

  await chat({
    provider: "azure", model: "my-deployment",
    messages: [{ role: "user", content: "hi" }],
    baseUrl: "https://res.openai.azure.com",
  } as any);
  assert.equal(headers["api-key"], "k");
  assert.equal(headers.Authorization, undefined);
});

test("azure puts the deployment in the path with an api-version", async () => {
  process.env.AZURE_OPENAI_API_KEY = "k";
  let seen = "";
  stub({ choices: [{ message: { content: "hi" } }] }, (url) => {
    seen = url;
  });
  await chat({
    provider: "azure", model: "my-deployment",
    messages: [{ role: "user", content: "hi" }],
    baseUrl: "https://res.openai.azure.com",
  } as any);
  assert.equal(
    seen,
    "https://res.openai.azure.com/openai/deployments/my-deployment" +
      `/chat/completions?api-version=${azure.DEFAULT_API_VERSION}`
  );
});

test("azure without an endpoint names what to pass", async () => {
  process.env.AZURE_OPENAI_API_KEY = "k";
  await assert.rejects(
    () =>
      chat({
        provider: "azure", model: "d",
        messages: [{ role: "user", content: "hi" }],
      }),
    (error: Error) =>
      error instanceof MissingEndpointError &&
      /AZURE_OPENAI_ENDPOINT/.test(error.message)
  );
});

// ── Databricks / Cloudflare / Vertex ─────────────────────────────────────────

test("databricks accepts every documented root shape", () => {
  // All three are things people paste out of the docs; a missing path segment
  // would 404 without saying which part was wrong.
  const expected =
    "https://ws.cloud.databricks.com/serving-endpoints/my-endpoint/invocations";
  for (const root of [
    "https://ws.cloud.databricks.com",
    "https://ws.cloud.databricks.com/",
    "https://ws.cloud.databricks.com/serving-endpoints",
  ]) {
    assert.equal(databricks.buildUrl(root, "my-endpoint"), expected);
  }
});

test("a bare cloudflare account id is expanded into the full url", () => {
  assert.equal(
    cloudflare.buildUrl("abc123", "@cf/meta/llama-3.1-8b-instruct"),
    "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1/chat/completions"
  );
});

test("the cloudflare account id is read at call time, not import time", async () => {
  // Configuring the environment after importing polygate is the normal order
  // in a test, a script using dotenv, or anything that reads config first.
  process.env.CLOUDFLARE_API_TOKEN = "t";
  process.env.CLOUDFLARE_ACCOUNT_ID = "set-after-import";
  let seen = "";
  stub({ choices: [{ message: { content: "hi" } }] }, (url) => {
    seen = url;
  });
  await chat({
    provider: "cloudflare",
    model: "@cf/meta/llama-3.1-8b-instruct",
    messages: [{ role: "user", content: "hi" }],
  });
  assert.ok(seen.includes("accounts/set-after-import/ai/v1"));
});

test("the vertex endpoint helper fills the region twice", () => {
  // The template names the region in two places and a mismatch 404s without
  // naming the problem.
  const url = vertex.endpoint("my-project", "europe-west4");
  assert.ok(url.startsWith("https://europe-west4-aiplatform.googleapis.com/"));
  assert.ok(url.includes("/locations/europe-west4/"));
  assert.ok(url.includes("/projects/my-project/"));
});

test("vertex without a project says so", async () => {
  process.env.GOOGLE_ACCESS_TOKEN = "ya29.token";
  await assert.rejects(
    () =>
      chat({
        provider: "vertex", model: "google/gemini-2.0-flash",
        messages: [{ role: "user", content: "hi" }],
      }),
    (error: Error) =>
      error instanceof MissingEndpointError &&
      /VERTEX_BASE_URL/.test(error.message)
  );
});
