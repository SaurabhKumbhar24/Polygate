/**
 * Cerebras inference.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://api.cerebras.ai/v1/chat/completions";
export const ENV_VAR = "CEREBRAS_API_KEY";

export const chat = makeAdapter({
  provider: "cerebras",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
