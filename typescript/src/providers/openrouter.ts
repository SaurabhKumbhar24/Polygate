/**
 * OpenRouter — one key, several hundred models across every major vendor.
 *
 * A router rather than a host: models are addressed `vendor/model`
 * (`anthropic/claude-sonnet-4.5`), and OpenRouter picks an upstream provider and
 * fails over between them. That makes it the cheapest way to reach a model
 * nobody here has an adapter for.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://openrouter.ai/api/v1/chat/completions";
export const ENV_VAR = "OPENROUTER_API_KEY";

export const chat = makeAdapter({
  provider: "openrouter",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
