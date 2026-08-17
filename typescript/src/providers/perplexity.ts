/**
 * Perplexity's Sonar models. The host has no /v1 segment, and responses can
 * carry citations in `raw` that no other provider returns.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://api.perplexity.ai/chat/completions";
export const ENV_VAR = "PERPLEXITY_API_KEY";

export const chat = makeAdapter({
  provider: "perplexity",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
