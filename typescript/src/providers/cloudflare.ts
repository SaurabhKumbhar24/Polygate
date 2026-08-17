/**
 * Cloudflare Workers AI — open models on Cloudflare's edge.
 *
 * The URL contains your account id, so one of `baseUrl`,
 * `$CLOUDFLARE_BASE_URL` or `$CLOUDFLARE_ACCOUNT_ID` is required. Model ids
 * carry an `@cf/` prefix naming the publisher, which is Cloudflare's own
 * convention rather than anything OpenAI-shaped.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { ChatResponse, ProviderChatOptions } from "../types";
import { makeAdapter } from "./openaiCompatible";

export const ENV_VAR = "CLOUDFLARE_API_TOKEN";
export const BASE_URL_ENV = "CLOUDFLARE_BASE_URL";
export const ACCOUNT_ENV = "CLOUDFLARE_ACCOUNT_ID";

export const API_ROOT = "https://api.cloudflare.com/client/v4/accounts";

/** The OpenAI-compatible root for an account id. */
export function accountUrl(accountId: string): string {
  return `${API_ROOT}/${accountId}/ai/v1`;
}

/**
 * Accept a bare account id as readily as a full URL.
 *
 * The account id is the thing a user actually has to hand — it is what the
 * dashboard shows — so making them assemble a five-segment URL around it adds a
 * step that exists only to be got wrong.
 */
export function buildUrl(root: string, _model: string): string {
  let base = root.replace(/\/+$/, "");
  if (!base.includes("://")) base = accountUrl(base);
  return base.endsWith("/chat/completions") ? base : `${base}/chat/completions`;
}

const adapter = makeAdapter({
  provider: "cloudflare",
  envVar: ENV_VAR,
  baseUrlEnv: BASE_URL_ENV,
  urlBuilder: buildUrl,
});

/**
 * Resolve the account id at call time, then delegate.
 *
 * Read here rather than at module load so configuring the environment after
 * importing polygate still works — the normal order in a test, a script using
 * dotenv, or anything that reads config before it reads code.
 */
export async function chat(
  options: ProviderChatOptions
): Promise<ChatResponse> {
  if (!options.baseUrl) {
    const account = process.env[ACCOUNT_ENV];
    if (account && !process.env[BASE_URL_ENV]) {
      options = { ...options, baseUrl: accountUrl(account) };
    }
  }
  return adapter(options);
}

(chat as any).endpointRequired = true;
