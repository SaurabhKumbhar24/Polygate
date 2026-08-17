/**
 * Vercel AI Gateway — a routing layer in front of every major vendor.
 *
 * Like OpenRouter, models are addressed `vendor/model`. The environment variable
 * is `AI_GATEWAY_API_KEY` rather than a Vercel-branded name, matching what
 * Vercel's own tooling sets.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://ai-gateway.vercel.sh/v1/chat/completions";
export const ENV_VAR = "AI_GATEWAY_API_KEY";

export const chat = makeAdapter({
  provider: "vercel",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
