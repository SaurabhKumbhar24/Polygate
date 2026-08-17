/**
 * Azure OpenAI — the same models, behind Azure's own conventions.
 *
 * Three things differ from OpenAI proper, and each is a distinct way to get a
 * 401 or 404 that looks like something else:
 *
 * 1. **The key is not a Bearer token.** It goes in an `api-key` header. Sending
 *    `Authorization: Bearer` returns a 401 indistinguishable from a wrong key.
 * 2. **The model is a deployment name in the path**, not a body field. Your
 *    deployment can be called anything, so `model: "gpt-4o"` only works if you
 *    named the deployment `gpt-4o`.
 * 3. **An `api-version` query parameter is mandatory.** Omitting it is a 404.
 *
 * The host is your own resource, so `baseUrl` or `$AZURE_OPENAI_ENDPOINT` is
 * required.
 *
 * An OpenAI-compatible *body*; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { apiKeyHeaderAuth, makeAdapter } from "./openaiCompatible";

export const ENV_VAR = "AZURE_OPENAI_API_KEY";
export const BASE_URL_ENV = "AZURE_OPENAI_ENDPOINT";
export const API_VERSION_ENV = "AZURE_OPENAI_API_VERSION";

/**
 * Pinned rather than tracking latest. Azure's versions change behaviour, and a
 * default that drifts under callers would turn a library upgrade into a silent
 * change in what the model does.
 */
export const DEFAULT_API_VERSION = "2024-10-21";

export function buildUrl(root: string, model: string): string {
  const base = root.replace(/\/+$/, "");
  const version = process.env[API_VERSION_ENV] || DEFAULT_API_VERSION;
  if (base.includes("/chat/completions")) {
    return base.includes("api-version=") ? base : `${base}?api-version=${version}`;
  }
  return `${base}/openai/deployments/${model}/chat/completions?api-version=${version}`;
}

export const chat = makeAdapter({
  provider: "azure",
  envVar: ENV_VAR,
  baseUrlEnv: BASE_URL_ENV,
  auth: apiKeyHeaderAuth("api-key"),
  urlBuilder: buildUrl,
});
