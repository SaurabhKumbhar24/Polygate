/**
 * Novita AI — open models, GPU marketplace pricing.
 *
 * The version segment is `/v3`, not the `/v1` almost everyone else uses.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://api.novita.ai/v3/openai/chat/completions";
export const ENV_VAR = "NOVITA_API_KEY";

export const chat = makeAdapter({
  provider: "novita",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
